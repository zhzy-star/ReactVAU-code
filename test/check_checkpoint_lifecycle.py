"""Regress unsaved best steps, checkpoint rotation, and final-step selection."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import tempfile

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import GemmaConfig, GemmaForCausalLM, Trainer, TrainingArguments, set_seed

from train_paligemma_vad_384_ddp import DDPVADTrainer, EvaluateFinalStep


class CheckpointTrainer(DDPVADTrainer):
    # Exercise the production checkpoint hooks in the real Trainer loop, using
    # a tiny local PEFT language model and deterministic validation metrics.
    compute_loss = Trainer.compute_loss
    training_step = Trainer.training_step
    get_batch_samples = Trainer.get_batch_samples
    prediction_step = Trainer.prediction_step
    log = Trainer.log


def run_checks():
    torch.set_num_threads(1)
    checks = []
    runtime = Path(__file__).resolve().parent / "artifacts" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    # The last step (5) is outside both eval_steps=2 and save_steps=4.
    # In the first case, best step 2 must survive subsequent checkpoint rotation.
    # In the second, final step 5 must be evaluated and selected as the best.
    for name, metrics, expected_step in (
        ("unscheduled_best_survives_rotation", [0.9, 0.8, 0.7], 2),
        ("final_step_can_win", [0.7, 0.8, 0.9], 5),
    ):
        with tempfile.TemporaryDirectory(prefix="checkpoint-regression-", dir=runtime) as tmp:
            set_seed(42)
            model = get_peft_model(GemmaForCausalLM(GemmaConfig(
                vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, max_position_embeddings=32,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
            )), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                          task_type="CAUSAL_LM"))
            sample = {"input_ids": torch.tensor([1, 3, 4, 5]),
                      "attention_mask": torch.ones(4, dtype=torch.long),
                      "labels": torch.tensor([-100, 3, 4, 5])}
            values = iter(metrics)
            trainer = CheckpointTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmp, use_cpu=True, max_steps=5,
                    per_device_train_batch_size=1, per_device_eval_batch_size=1,
                    eval_strategy="steps", eval_steps=2, save_strategy="steps", save_steps=4,
                    save_total_limit=2, load_best_model_at_end=True,
                    metric_for_best_model="eval_auc", greater_is_better=True,
                    report_to="none", disable_tqdm=True, logging_strategy="no",
                    optim="adamw_torch", learning_rate=0.01,
                ),
                train_dataset=[sample] * 5, eval_dataset=[sample],
                compute_metrics=lambda _: {"auc": next(values)},
                callbacks=[EvaluateFinalStep()],
            )
            result = trainer.train()
            best = Path(trainer.state.best_model_checkpoint)
            assert best.name == f"checkpoint-{expected_step}", best
            assert trainer.state.best_metric == max(metrics)
            eval_steps = [x["step"] for x in trainer.state.log_history if "eval_auc" in x]
            assert eval_steps == [2, 4, 5], eval_steps
            checkpoints = sorted(Path(tmp).glob("checkpoint-*"))
            assert len(checkpoints) <= 2 and best in checkpoints, checkpoints
            saved = load_file(str(best / "adapter_model.safetensors"))
            loaded = get_peft_model_state_dict(trainer.model)
            assert saved.keys() == loaded.keys()
            assert all(torch.equal(t, loaded[k].cpu()) for k, t in saved.items())
            trainer.save_model(tmp)
            trainer.save_state()
            exported = load_file(str(Path(tmp) / "adapter_model.safetensors"))
            assert all(torch.equal(t, exported[k]) for k, t in saved.items())
            state = json.loads((Path(tmp) / "trainer_state.json").read_text())
            assert state["best_model_checkpoint"] == str(best)
            assert result.global_step == 5
            # A broken local adapter must fail explicitly, without a Hub lookup
            # or silently selecting a different checkpoint.
            broken = Path(tmp) / "incomplete"
            broken.mkdir()
            trainer.state.best_model_checkpoint = str(broken)
            for config_present in (False, True):
                if config_present:
                    (broken / "adapter_config.json").write_text("{}")
                try:
                    trainer._load_best_model()
                except FileNotFoundError as error:
                    assert str(broken) in str(error)
                else:
                    raise AssertionError("Incomplete best adapter was accepted")
            checks.append({"case": name, "best_step": expected_step,
                           "evaluation_steps": eval_steps, "retained_checkpoints": len(checkpoints),
                           "best_reloaded_and_exported": True})
    return checks


if __name__ == "__main__":
    for check in run_checks():
        print("PASS", check)
