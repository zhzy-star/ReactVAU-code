"""Run the real 3B model on four GPUs and verify isolated Stage-1 outputs."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for name, suffix in {
    "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "HF_HOME": "huggingface",
    "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TRITON_CACHE_DIR": "triton",
}.items():
    target = HERE / "artifacts" / "runtime" / suffix
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)
os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

import json
import math
import random
import subprocess
from collections import defaultdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version


def dependency_report():
    report = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" not in line:
            continue
        name, expected = line.strip().split("==", 1)
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        report[name] = {"required": expected, "installed": installed,
                        "matches": installed is not None and installed.split("+")[0] == expected}
    for name in ("torch", "torchvision", "transformers", "accelerate", "peft", "safetensors"):
        if not report[name]["matches"]:
            raise RuntimeError(f"Core dependency differs from requirements.txt: {name}: {report[name]}")
    return report


def prepare_data(destination, image_root):
    source = Path(os.environ.get("DATA_PATH", image_root / "combined_train_binary.json"))
    videos = defaultdict(list)
    for item in json.loads(source.read_text()):
        videos[item["image"].split("/")[1]].append(item)
    # Six videos of each class; production split assigns five of each to val.
    # Keep 32 grids from each of the two training videos and one per val video:
    # 64 training samples = exactly two optimizer steps at global batch 32.
    groups = [[], []]
    for video, samples in sorted(videos.items()):
        label = int(any(x["suffix"].startswith("Yes") for x in samples))
        if len(samples) >= 32 and len(groups[label]) < 6:
            # Retain at least one positive from every anomaly video after
            # subsampling; otherwise the production split reclassifies it.
            valid = [x for x in samples
                     if int(x["suffix"].startswith("Yes")) == label
                     and (image_root / x["image"]).is_file()]
            if len(valid) >= 32:
                groups[label].append((video, valid))
    if any(len(group) != 6 for group in groups):
        raise RuntimeError("Need six videos per class with at least 32 valid grids")
    # Mirror production's seeded shuffle: anomaly first, then normal.
    rng = random.Random(42)
    anomaly, normal = groups[1][:], groups[0][:]
    rng.shuffle(anomaly)
    rng.shuffle(normal)
    train_videos = {anomaly[-1][0], normal[-1][0]}
    samples = []
    for group in (groups[0], groups[1]):
        for video, items in group:
            samples.extend(items[:32] if video in train_videos else items[:1])
    destination.write_text(json.dumps(samples, indent=2))
    return {"source": str(source), "samples": len(samples), "train_samples": 64,
            "val_samples": 10, "video_count": 12}


def verify_outputs(run_dir, world_size):
    audits = [json.loads((run_dir / f"audit.rank{i}.json").read_text())
              for i in range(world_size)]
    for rank, audit in enumerate(audits):
        assert audit["rank"] == rank and audit["device"] == f"cuda:{rank}", audit
        assert audit["global_step"] == 2 and audit["gradient_checks"] == 2, audit
        assert audit["train_samples"] == 64 and audit["val_samples"] == 10, audit
        assert len(audit["train_indices"]) == 64 // world_size, audit
        assert all(value > 0 for value in audit["parameter_updates"].values()), audit
    shards = [set(audit["train_indices"]) for audit in audits]
    for rank, shard in enumerate(shards):
        assert len(shard) == 64 // world_size, "Repeated training samples within a rank"
        assert all(not shard & previous for previous in shards[:rank]), "Training samples overlap across ranks"
    assert set().union(*shards) == set(range(64)), "Missing training samples"
    assert len({audit["train_digest"] for audit in audits}) == 1
    assert len({audit["val_digest"] for audit in audits}) == 1
    for filename in ("adapter_model.safetensors", "adapter_config.json", "preprocessor_config.json",
                     "training_config.json", "metrics_summary.json", "trainer_state.json",
                     "train_history.csv", "eval_history.csv", "training.log",
                     *(f"training.rank{rank}.log" for rank in range(1, world_size))):
        assert (run_dir / filename).is_file(), filename
    assert (run_dir / "checkpoint-1" / "adapter_model.safetensors").is_file()
    assert (run_dir / "checkpoint-2" / "adapter_model.safetensors").is_file()
    summary = json.loads((run_dir / "metrics_summary.json").read_text())
    assert len(summary["train_history"]) == 2, summary
    assert all(math.isfinite(entry["loss"]) for entry in summary["train_history"]), summary
    assert summary["best_model_checkpoint"], summary
    final_eval = summary["final_eval"]
    assert 0 <= final_eval["eval_auc"] <= 1
    assert 0 <= final_eval["eval_accuracy"] <= 1
    config = json.loads((run_dir / "training_config.json").read_text())
    assert config["global_batch_size"] == 32
    assert config["training_args"]["per_device_train_batch_size"] == 2
    assert config["training_args"]["gradient_accumulation_steps"] == 32 // (world_size * 2)
    return {"ranks": audits, "training": summary}


def verify_reload(run_dir, image_root):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoProcessor

    sys.path.insert(0, str(HERE))
    import train_paligemma_vad_384_ddp as ddp

    config = json.loads((run_dir / "training_config.json").read_text())
    model_args = ddp.stage1.ModelArguments(**config["model_args"])
    torch.cuda.set_device(0)
    # Load the inference model independently of the training setup/LoRA init.
    base = ddp.stage1.PaliGemmaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation, low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda:0")
    ddp.stage1.load_streamforest_vision_weights(base, model_args.streamforest_vision_weights_path)
    model = PeftModel.from_pretrained(base, str(run_dir), is_trainable=False)
    ddp.stage1.patch_vision_feature_layer(model, model_args.vision_feature_layer)
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    base.language_model.config.use_cache = False
    base.language_model.model.config.use_cache = False
    model.eval()
    processor = AutoProcessor.from_pretrained(str(run_dir), local_files_only=True)
    assert processor.image_seq_length == 729
    assert processor.image_processor.size == {"height": 384, "width": 384}
    saved = load_file(str(run_dir / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    assert saved.keys() == loaded.keys(), "Adapter keys changed on reload"
    for name, value in saved.items():
        assert torch.equal(value, loaded[name].detach().cpu()), f"Adapter reload mismatch: {name}"
    model_saved_tensors = len(saved)
    del saved, loaded

    data = json.loads(Path(config["data_args"]["data_path"]).read_text())
    _, val, _ = ddp.stage1.split_data_by_video(data, seed=42)
    yes, no = (processor.tokenizer.encode(token, add_special_tokens=False)[0] for token in ("Yes", "No"))
    predictions, labels = [], []
    dataset = ddp.stage1.VADDataset(val, processor, str(image_root), split_name="reload")
    for sample in dataset:
        suffix_start = (sample["labels"] != -100).nonzero(as_tuple=True)[0][0].item()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=sample["input_ids"][:suffix_start].unsqueeze(0).cuda(),
                            attention_mask=sample["attention_mask"][:suffix_start].unsqueeze(0).cuda(),
                            pixel_values=sample["pixel_values"].unsqueeze(0).to("cuda", torch.bfloat16))
        logits = outputs.logits[0, -1, [yes, no]].float()
        assert torch.isfinite(logits).all(), "Reloaded model produced invalid logits"
        predictions.append([torch.softmax(logits, dim=0)[0].item(), int(logits[0] > logits[1])])
        labels.append(sample["is_anomaly"].item())
    import numpy as np
    metrics = ddp.stage1.compute_metrics((np.asarray(predictions), np.asarray(labels)))
    expected = json.loads((run_dir / "metrics_summary.json").read_text())["final_eval"]
    for name, value in metrics.items():
        assert math.isclose(value, expected[f"eval_{name}"], abs_tol=1e-8), (metrics, expected)
    return {"saved_tensors_verified": model_saved_tensors, "prefix_only_inference": True,
            "validation_samples": len(labels), "metrics_match_ddp": True,
            "metrics": metrics, "processor_image_tokens": 729}


def main():
    world_size = int(os.environ.get("NPROC_PER_NODE", "4"))
    if world_size <= 0 or 32 % (world_size * 2):
        raise ValueError("GPU count * batch 2 must divide global batch 32")
    grad_accum = 32 // (world_size * 2)
    session = HERE / "artifacts" / "smoke" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True)
    report = {"status": "running", "dependencies": dependency_report(),
              "launch_config": {"world_size": world_size,
                                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                                "per_device_train_batch_size": 2,
                                "gradient_accumulation_steps": grad_accum,
                                "global_batch_size": 32}}
    from check_loss_normalization import run_checks
    report["loss_normalization"] = run_checks()
    image_root = Path(os.environ.get("IMAGE_ROOT", os.environ.get(
        "VAD_TRAIN_ROOT", ROOT / "vad" / "vad_data" / "paligemma_train")))
    data_path = session / "smoke_data.json"
    report["data"] = prepare_data(data_path, image_root)
    env = dict(os.environ, DATA_PATH=str(data_path), IMAGE_ROOT=str(image_root),
               OUTPUT_DIR=str(session / "outputs"), NPROC_PER_NODE=str(world_size), BATCH_SIZE="2",
               GRAD_ACCUM=str(grad_accum), MAX_STEPS="2", LOGGING_STEPS="1", SAVE_STEPS="1",
               EVAL_STEPS="1", SMOKE_AUDIT="True", DATALOADER_NUM_WORKERS="2",
               WARMUP_RATIO="0")
    report_path = session / "smoke_report.json"
    try:
        print(f"Running {world_size}-GPU smoke test; log: {session / 'launcher.log'}", flush=True)
        with (session / "launcher.log").open("w") as log:
            subprocess.run(["bash", str(HERE / "train_paligemma_vad_384_ddp.sh")],
                           env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           timeout=1200, check=True)
        runs = list((session / "outputs").glob("training_*"))
        assert len(runs) == 1, f"Expected one shared run directory: {runs}"
        run_dir = runs[0]
        report["run_dir"] = str(run_dir)
        report.update(verify_outputs(run_dir, world_size))
        print("DDP training, evaluation, gradients, and checkpoints passed. Verifying reload...", flush=True)
        report["reload"] = verify_reload(run_dir, image_root)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Smoke report: {report_path}", flush=True)
    print(f"PASS: real {world_size}-GPU Stage-1 training and adapter reload", flush=True)


if __name__ == "__main__":
    main()
