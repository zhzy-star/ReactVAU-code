"""Isolated DDP Stage-1 entry point, reusing the production model and dataset."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parent
sys.path.insert(0, str(ROOT / "vad"))

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from peft import PeftModel
from transformers import HfArgumentParser, Trainer, TrainerCallback, set_seed

import train_paligemma_vad_384 as stage1

logger = logging.getLogger(__name__)


@dataclass
class DDPTrainingArguments(stage1.VADTrainingArguments):
    output_dir: str = field(default=str(TEST_DIR / "artifacts" / "training"))
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = field(default=True)
    ddp_find_unused_parameters: bool = field(default=False)
    smoke_audit: bool = field(default=False)


class MainProcessHistory(stage1.MetricsHistoryCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            return super().on_log(args, state, control, logs=logs, **kwargs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero:
            return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)


class IndexedDataset(stage1.VADDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["_sample_index"] = torch.tensor(idx)
        return sample


class DDPVADTrainer(stage1.VADTrainer):
    """Keep prefix-only evaluation; normalize loss over each accumulation window."""

    def _load_best_model(self):
        if not isinstance(self.model, PeftModel):
            return super()._load_best_model()
        # PEFT 0.14 infers the unindexed "cuda" device. safetensors resolves
        # that to GPU 0 even when another LOCAL_RANK is current. Read on CPU
        # and copy into the existing parameters on each rank's own GPU.
        logger.info("Loading best adapter from %s through CPU (score: %s)",
                    self.state.best_model_checkpoint, self.state.best_metric)
        self.model.load_adapter(self.state.best_model_checkpoint,
                                self.model.active_adapters[0], torch_device="cpu")

    def get_batch_samples(self, epoch_iterator, num_batches):
        samples, _ = super().get_batch_samples(epoch_iterator, num_batches)
        # Transformers 4.47.1 uses Accelerator(accumulation_steps=1). The custom
        # production compute_loss ignores num_items_in_batch, so explicitly
        # normalize over the window, including a partial final window.
        denominator = torch.zeros((), dtype=torch.float64, device=self.args.device)
        for batch in samples:
            if self.label_format == "text":
                denominator += batch["token_weights"][..., 1:].sum().to(denominator)
            else:
                denominator += (batch["labels"][..., 1:] != -100).sum().to(denominator)
        if dist.is_initialized():
            dist.all_reduce(denominator)
            denominator /= dist.get_world_size()
        if denominator.item() <= 0:
            raise ValueError("Accumulation window contains no supervised tokens")
        return samples, denominator.item()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_format == "text":
            numerator = inputs["token_weights"][..., 1:].sum()
        else:
            numerator = (inputs["labels"][..., 1:] != -100).sum()
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        loss, outputs = result if return_outputs else (result, None)
        if model.training and num_items_in_batch is not None:
            loss = loss * (numerator.to(loss) / num_items_in_batch)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.args.world_size > 1 and not isinstance(model, DistributedDataParallel):
            raise RuntimeError("Expected a DistributedDataParallel training model")
        if self.args.smoke_audit:
            if not hasattr(self, "smoke_train_indices"):
                self.smoke_train_indices = []
            self.smoke_train_indices.extend(inputs.pop("_sample_index").tolist())
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        if "loss" in logs:
            totals = torch.tensor([
                sum(self._accumulated_yes_probs), sum(self._accumulated_no_probs),
                len(self._accumulated_yes_probs),
            ], dtype=torch.float64, device=self.args.device)
            if dist.is_initialized():
                dist.all_reduce(totals)
            yes, no, count = totals.tolist()
            logs.update(avg_yes_prob=yes / max(count, 1),
                        avg_no_prob=no / max(count, 1), n_samples_in_step=int(count))
            self._accumulated_yes_probs.clear()
            self._accumulated_no_probs.clear()
            self._accumulation_count = 0
        # Bypass the production override, whose signature predates 4.47.1.
        Trainer.log(self, logs, start_time)
        if self.is_world_process_zero():
            logger.info("STEP %s LOG: %s", self.state.global_step, logs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(
            model, self._prepare_inputs(inputs), prediction_loss_only, ignore_keys)


class SmokeAudit(TrainerCallback):
    """Verify real optimizer updates and exact cross-rank gradient agreement."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        params = dict(model.named_parameters())
        self.names = [next(n for n in params if "lora_B" in n),
                      next(n for n, p in params.items()
                           if "multi_modal_projector" in n and p.requires_grad)]
        self.initial = {n: params[n].detach().cpu().clone() for n in self.names}
        self.gradient_checks = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        params = dict(model.named_parameters())
        trainable = [p for p in params.values() if p.requires_grad]
        if any(p.grad is None for p in trainable):
            raise AssertionError("A trainable parameter did not receive a gradient")
        if not torch.stack([torch.isfinite(p.grad).all() for p in trainable]).all():
            raise AssertionError("Non-finite gradients")
        for name in self.names:
            gradient = params[name].grad
            reference = gradient.clone()
            dist.broadcast(reference, src=0)
            if not torch.equal(gradient, reference):
                raise AssertionError(f"DDP gradients differ: {name}")
        self.gradient_checks += 1

    def on_train_end(self, args, state, control, model=None, **kwargs):
        params = dict(model.named_parameters())
        self.updates = {n: (params[n].detach().cpu() - self.initial[n]).abs().max().item()
                        for n in self.names}
        if not all(value > 0 for value in self.updates.values()):
            raise AssertionError(f"Expected LoRA and projector updates: {self.updates}")


def dataset_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def train():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    parser = HfArgumentParser((stage1.ModelArguments, stage1.DataArguments, DDPTrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(TEST_DIR):
        raise ValueError("This isolated entry point requires output_dir inside test/")
    timestamp = [datetime.now().strftime("%Y%m%d_%H%M%S_%f") if args.process_index == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(timestamp, src=0)
    run_dir = output / f"training_{timestamp[0]}"
    args.output_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Each process owns its log; shared metadata and model files have one writer.
    log_name = "training.log" if args.process_index == 0 else f"training.rank{args.process_index}.log"
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s rank={args.process_index} %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / log_name), logging.StreamHandler()],
                        force=True)
    set_seed(args.seed)
    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    logger.info("world_size=%s rank=%s device=%s global_batch=%s", args.world_size,
                args.process_index, args.device,
                args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps)

    model, processor = stage1.setup_model_and_processor(model_args)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.get_base_model().language_model.config.use_cache = False
    model.get_base_model().language_model.model.config.use_cache = False
    if not args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    all_data = json.loads(Path(data_args.data_path).read_text())
    train_data, val_data, split_info = stage1.split_data_by_video(all_data, data_args.val_split, args.seed)
    train_videos = {x["image"].split("/")[1] for x in train_data}
    val_videos = {x["image"].split("/")[1] for x in val_data}
    if train_videos & val_videos:
        raise AssertionError("Video leakage between training and validation")
    dataset_class = IndexedDataset if args.smoke_audit else stage1.VADDataset
    datasets = [dataset_class(
        data, processor, data_args.image_root, data_args.max_length, split,
        label_format=data_args.label_format,
        decision_token_weight=args.decision_token_weight,
        event_token_weight=args.event_token_weight,
    ) for data, split in ((train_data, "train"), (val_data, "val"))]
    if any(len(ds) != len(data) for ds, data in zip(datasets, (train_data, val_data))):
        raise ValueError("Missing images: fix dataset paths before distributed training")
    if not all(len(ds) for ds in datasets):
        raise ValueError("Training and validation datasets must both be nonempty")
    metadata = {"run_dir": str(run_dir), "train_digest": dataset_digest(train_data),
                "val_digest": dataset_digest(val_data), "train_samples": len(datasets[0]),
                "val_samples": len(datasets[1])}
    gathered = [None] * args.world_size
    if dist.is_initialized():
        dist.all_gather_object(gathered, metadata)
    else:
        gathered = [metadata]
    if any(value != metadata for value in gathered):
        raise AssertionError("Ranks disagree on run directory or dataset split")
    history = MainProcessHistory(str(run_dir))
    audit = SmokeAudit() if args.smoke_audit else None
    trainer = DDPVADTrainer(
        model=model, args=args, train_dataset=datasets[0], eval_dataset=datasets[1],
        data_collator=stage1.VADDataCollator(), compute_metrics=stage1.compute_metrics,
        callbacks=[history] + ([audit] if audit else []),
        yes_token_id=processor.tokenizer.encode("Yes", add_special_tokens=False)[0],
        no_token_id=processor.tokenizer.encode("No", add_special_tokens=False)[0],
        label_format=data_args.label_format, decision_token_weight=args.decision_token_weight,
        event_token_weight=args.event_token_weight,
    )
    if trainer.is_world_process_zero():
        (run_dir / "training_config.json").write_text(json.dumps({
            "model_args": vars(model_args), "data_args": vars(data_args),
            "training_args": args.to_dict(), "split_info": split_info,
            "global_batch_size": args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps,
        }, indent=2))
    result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(run_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(run_dir)
        trainer.save_state()
        (run_dir / "metrics_summary.json").write_text(json.dumps({
            "train_metrics": result.metrics, "final_eval": metrics,
            "train_history": history.train_history, "eval_history": history.eval_history,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }, indent=2))
    if audit:
        metadata.update(rank=args.process_index, train_indices=trainer.smoke_train_indices,
                        device=str(args.device), global_step=trainer.state.global_step,
                        gradient_checks=audit.gradient_checks, parameter_updates=audit.updates,
                        peak_memory_mib=torch.cuda.max_memory_allocated() / 1024 ** 2)
        (run_dir / f"audit.rank{args.process_index}.json").write_text(json.dumps(metadata, indent=2))
    trainer.accelerator.wait_for_everyone()
    logger.info("Training complete: %s", run_dir)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
