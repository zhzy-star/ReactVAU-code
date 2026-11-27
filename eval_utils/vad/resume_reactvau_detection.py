"""Resume a full ReactVAU VAD run without changing the original evaluator.

Usage (inside the ReactVAU conda environment)::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_utils/vad/resume_reactvau_detection.py \
        --resume-dir eval_results/vad/reactvau-... --dry-run
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_utils/vad/resume_reactvau_detection.py \
        --resume-dir eval_results/vad/reactvau-...

The GPU count and settings are recovered from detection.log. Completed rank
shards are read only. Missing ranks keep their original video assignments.
New predictions are saved atomically after each video under _resume/videos/;
all shards are retained after merging. Resume logs have separate filenames.
This entry point supports full runs, not the original random test subset.
Unlogged model options default to the original shell launcher's settings.
"""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ARRAY_NAMES = ("scores", "labels", "smoothed", "online_smoothed")
TIMING_LISTS = ("memory_update_times", "rt_anomaly_encode_times",
                "sf_inference_times", "anomaly_pool_update_times", "sf_trigger_total_times")


def recover_config(directory):
    """Read settings from the source log, rather than current environment defaults."""
    text = (directory / "detection.log").read_text()
    if "Test mode:" in text:
        raise ValueError("Only full evaluations can be resumed by this entry point")

    def match(pattern, cast=str):
        found = re.search(pattern, text, re.MULTILINE)
        if found is None:
            raise ValueError(f"Source log is missing a required setting: {pattern}")
        return cast(found.group(1).strip())

    config = {}
    for name, label in (
        ("paligemma_model_path", "PaliGemma model"),
        ("paligemma_lora_path", "PaliGemma LoRA"),
        ("streamforest_model_path", "StreamForest model"),
        ("streamforest_model_base", "StreamForest base"),
        ("streamforest_quantization", "StreamForest quantization"),
        ("dataset", "Dataset"), ("trigger_mode", "Trigger mode"),
        ("sf_scoring_method", "SF scoring method"),
        ("sf_prompt_style", "SF prompt style"),
    ):
        value = match(re.escape(label) + r":\s*([^\n]+)")
        config[name] = None if value == "None" else value
    for name, pattern, cast in (
        ("world_size", r"Rank: 0/(\d+),", int),
        ("num_selected_videos", r"Total videos to process: (\d+)", int),
        ("anomaly_threshold", r"Anomaly threshold: ([\d.eE+-]+)", float),
        ("fusion_alpha", r"Score fusion: \w+ \(alpha=([\d.eE+-]+)\)", float),
        ("target_fps", r"Target FPS: (\d+),", int),
        ("query_interval", r"Query interval: (\d+)", int),
        ("batch_size", r"PaliGemma batch size: (\d+)", int),
        ("paligemma_image_size", r"PaliGemma image size: (\d+)x", int),
        ("pool_threshold", r"Pool threshold: ([\d.eE+-]+)", float),
        ("online_smooth_alpha", r"Online smoothing: alpha=([\d.eE+-]+),", float),
        ("online_smooth_beta", r"Online smoothing: alpha=[^,]+, beta=([\d.eE+-]+)", float),
    ):
        # The current shell launcher always uses weighted score fusion.
        if name == "fusion_alpha" and "Score fusion: weighted" not in text:
            config[name] = None
        else:
            config[name] = match(pattern, cast)
    config["score_fusion"] = match(r"Score fusion: (\w+)")
    config["deterministic"] = match(r"Deterministic execution: (True|False)") == "True"
    config["enable_memory_enhancement"] = match(r"Memory enhancement \(Pool\+APS\): (\w+)") == "ENABLED"
    config["enable_rt_anomaly"] = match(r"RT-Anomaly dense encoding: (\w+)") == "ENABLED"
    config["anno_path"] = match(r"Loading annotation from ([^\n]+)")
    config["paligemma_streamforest_weights"] = None
    config["paligemma_vision_feature_layer"] = -2
    if config["paligemma_image_size"] == 384:
        config["paligemma_streamforest_weights"] = match(r"PaliGemma StreamForest weights: ([^\n]+)")
        config["paligemma_vision_feature_layer"] = match(r"PaliGemma vision feature layer: (-?\d+)", int)
    config["paligemma_prompt"] = match(r"PaliGemma prompt: '(.*)'$")
    config["sf_prompt_preview"] = match(r"SF prompt preview: ([\s\S]*?)\.\.\.")
    return config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, arrays, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            payload = dict(arrays)
            if metadata is not None:
                payload["metadata"] = np.frombuffer(
                    json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_arrays(arrays):
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Missing prediction arrays")
    if any(a.ndim != 1 for a in arrays.values()) or len({len(a) for a in arrays.values()}) != 1:
        raise ValueError("Mismatched frame array shapes")
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite predictions")
    if not np.isin(arrays["labels"], [0, 1]).all():
        raise ValueError("Nonbinary labels")


def load_arrays(path, with_metadata=False):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ARRAY_NAMES}
        metadata = json.loads(data["metadata"].tobytes()) if with_metadata else None
    validate_arrays(arrays)
    return arrays, metadata


def checkpoint_path(directory, key):
    # Hashing also handles XD-Violence names containing slashes or punctuation.
    return directory / "_resume" / "videos" / (hashlib.sha256(key.encode()).hexdigest() + ".npz")


def validate_video(key, video, annotation):
    if video["video_name"] != annotation["video_name"] or video["events_frames"] != annotation["intervals_raw"]:
        raise ValueError(f"Annotation differs from saved result: {key}")
    n = video["num_queries"]
    if n <= 0 or n != len(video["query_scores"]) or n != len(video["paligemma_scores"]):
        raise ValueError(f"Invalid query counts: {key}")
    if not np.isfinite(video["query_scores"]).all() or not np.isfinite(video["paligemma_scores"]).all():
        raise ValueError(f"Nonfinite query scores: {key}")
    if video["n_frames"] <= 0 or not np.isfinite(video["fps"]) or video["fps"] <= 0:
        raise ValueError(f"Invalid frame metadata: {key}")


def load_rank(directory, rank, assigned, annotations):
    npz = directory / "_shards" / f"rank-{rank}.npz"
    meta = npz.with_suffix(".json")
    if not npz.exists() and not meta.exists():
        return None
    if not npz.exists() or not meta.exists():
        # An interrupted save_rank can leave its NPZ without the JSON commit.
        # Only our own complete per-video checkpoints authorize rebuilding it.
        if not meta.exists() and all(
                load_video(directory, key, annotations[key]) is not None for key in assigned):
            return None
        raise ValueError(f"Incomplete shard pair for rank {rank}; inspect {npz.parent}")
    metadata = json.loads(meta.read_text())
    stats = metadata["rank_stats"]
    if stats["rank"] != rank or stats["assigned_video_keys"] != assigned:
        raise ValueError(f"Source assignment differs for rank {rank}")
    videos = metadata["video_results"]
    if metadata["failed_videos"] or set(videos) != set(assigned):
        raise ValueError(f"Rank {rank} has failures or incomplete coverage; cannot reuse as a completed shard")
    if stats["num_videos"] != len(assigned) or stats["num_failed_videos"]:
        raise ValueError(f"Inconsistent rank statistics: rank {rank}")
    arrays, _ = load_arrays(npz)
    from detect_utils import make_gt_labels_from_anno
    offset = 0
    for key, video in videos.items():
        validate_video(key, video, annotations[key])
        labels = make_gt_labels_from_anno(annotations[key], video["n_frames"], video["fps"])
        if not np.array_equal(arrays["labels"][offset:offset + len(labels)], labels):
            raise ValueError(f"Saved labels differ from annotation: {key}")
        offset += len(labels)
    if offset != len(arrays["labels"]):
        raise ValueError(f"Saved frame count differs from video metadata: rank {rank}")
    if metadata["total_queries"] != sum(v["num_queries"] for v in videos.values()):
        raise ValueError(f"Invalid query total: rank {rank}")
    return metadata


def load_video(directory, key, annotation):
    path = checkpoint_path(directory, key)
    if not path.exists():
        return None
    arrays, metadata = load_arrays(path, with_metadata=True)
    if metadata["video_key"] != key:
        raise ValueError(f"Wrong checkpoint key: {path}")
    validate_video(key, metadata["video"], annotation)
    from detect_utils import make_gt_labels_from_anno
    video = metadata["video"]
    labels = make_gt_labels_from_anno(annotation, video["n_frames"], video["fps"])
    if not np.array_equal(labels, arrays["labels"]):
        raise ValueError(f"Saved frame labels differ: {key}")
    return arrays, metadata


def make_plan(directory, config, annotations):
    keys = sorted(annotations)
    if len(keys) != config["num_selected_videos"]:
        raise ValueError("Annotation video count changed since original run")
    plan = []
    for rank in range(config["world_size"]):
        assigned = keys[rank::config["world_size"]]
        shard = load_rank(directory, rank, assigned, annotations)
        saved = len(assigned) if shard is not None else sum(
            load_video(directory, key, annotations[key]) is not None for key in assigned)
        plan.append({"rank": rank, "assigned": len(assigned), "saved": saved,
                     "pending": len(assigned) - saved, "completed_shard": shard is not None})
    return plan


def build_detector(base, config, local_rank):
    device = f"cuda:{local_rank}"
    sf = base.StreamForestReasoner(
        model_path=config["streamforest_model_path"], model_base=config["streamforest_model_base"],
        conv_template=config["streamforest_conv_template"], device=device,
        time_msg_style=config["streamforest_time_msg"],
        load_4bit=config["streamforest_quantization"] == "4bit",
        load_8bit=config["streamforest_quantization"] == "8bit")
    base.torch.cuda.empty_cache()
    pg = base.PaliGemmaDetector(
        model_path=config["paligemma_model_path"], lora_path=config["paligemma_lora_path"],
        device=device, attn_implementation=config["paligemma_attn"],
        image_size=config["paligemma_image_size"],
        streamforest_weights_path=config["paligemma_streamforest_weights"],
        vision_feature_layer=config["paligemma_vision_feature_layer"])
    base.torch.cuda.empty_cache()
    return base.CombinedAnomalyDetector(
        paligemma_detector=pg, streamforest_reasoner=sf,
        **{name: config[name] for name in (
            "anomaly_threshold", "trigger_mode", "paligemma_prompt", "sf_prompt_template",
            "sf_prompt_style", "score_fusion", "fusion_alpha", "sf_scoring_method",
            "enable_memory_enhancement", "enable_rt_anomaly", "pool_threshold",
            "online_smooth_alpha", "online_smooth_beta")})


def process_video(base, detector, key, annotation, config):
    subdir = "ucf-crime" if config["dataset"] == "ucf-crime" else "xd-violence"
    path = Path(config["video_dir"]) / "videos" / subdir / "videos/test" / (annotation["video_name"] + ".mp4")
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = base.cv2.VideoCapture(str(path))
    fps = cap.get(base.cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    started = time.monotonic()
    result = detector.detect_video(
        video_path=str(path), target_fps=config["target_fps"],
        query_interval=config["query_interval"], batch_size=config["batch_size"],
        verbose=config["verbose"])
    arrays = {name: np.asarray(result[source]) for name, source in (
        ("scores", "frame_scores"), ("smoothed", "smoothed_frame_scores"),
        ("online_smoothed", "online_smoothed_frame_scores"))}
    arrays["labels"] = np.asarray(base.make_gt_labels_from_anno(annotation, result["total_frames"], fps), dtype=np.uint8)
    validate_arrays(arrays)
    video = {name: result[name] for name in ("num_queries", "trigger_count", "query_scores", "paligemma_scores")}
    video.update(video_name=annotation["video_name"], n_frames=result["total_frames"], fps=fps,
                 events_frames=annotation["intervals_raw"], trigger_details=result["trigger_details"][:10])
    validate_video(key, video, annotation)
    metadata = {"video_key": key, "video": video, "num_segments": result["num_segments"],
                "timing": result["timing"], "elapsed_time_sec": time.monotonic() - started}
    return arrays, metadata


def save_rank(directory, rank, local_rank, assigned, annotations, peak_memory):
    parts = {name: [] for name in ARRAY_NAMES}
    metadata = {"video_results": {}, "failed_videos": {}, "total_queries": 0,
                "total_segments": 0, "total_triggers": 0,
                "timing": {"pg_infer_total": 0.0, **{name: [] for name in TIMING_LISTS}}}
    elapsed = 0.0
    for key in assigned:
        arrays, record = load_video(directory, key, annotations[key])
        for name in ARRAY_NAMES:
            parts[name].append(arrays[name])
        video = record["video"]
        metadata["video_results"][key] = video
        metadata["total_queries"] += video["num_queries"]
        metadata["total_segments"] += record["num_segments"]
        metadata["total_triggers"] += video["trigger_count"]
        metadata["timing"]["pg_infer_total"] += record["timing"].get("pg_infer_total", 0.0)
        for name in TIMING_LISTS:
            metadata["timing"][name].extend(record["timing"].get(name, []))
        elapsed += record["elapsed_time_sec"]
    metadata["rank_stats"] = {"rank": rank, "local_rank": local_rank,
                              "assigned_video_keys": assigned, "num_videos": len(assigned),
                              "num_failed_videos": 0, "elapsed_time_sec": elapsed,
                              "peak_gpu_memory_allocated_mb": peak_memory}
    arrays = {name: np.concatenate(values) if values else np.array([]) for name, values in parts.items()}
    # JSON is the commit marker. A crash between these writes is repaired on retry.
    path = directory / "_shards" / f"rank-{rank}.npz"
    atomic_npz(path, arrays)
    atomic_json(path.with_suffix(".json"), metadata)


def finalize(base, directory, config, annotations):
    arrays, videos, totals, timing, rank_stats, failures = base.merge_rank_results(
        directory / "_shards", config["world_size"])
    if failures or set(videos) != set(annotations):
        raise ValueError("Refusing to report full metrics without complete video coverage")
    validate_arrays(arrays)
    summary = dict(config)
    summary.pop("sf_prompt_preview", None)
    summary.pop("verbose", None)
    summary.update(model="combined_paligemma_streamforest", rank_stats=rank_stats,
                   num_videos=len(videos), num_failed_videos=0, failed_videos={},
                   total_frames=len(arrays["scores"]), **totals)
    summary["trigger_rate_pct"] = round(totals["total_triggers"] / max(1, totals["total_queries"]) * 100, 2)
    summary["elapsed_time_sec"] = max(s["elapsed_time_sec"] for s in rank_stats)
    summary["timing"] = {"pg_total_sec": round(timing["pg_infer_total"], 3)}
    pg_ms = timing["pg_infer_total"] / max(1, totals["total_queries"]) * 1000
    summary["timing"]["pg_avg_per_query_ms"] = round(pg_ms, 1)
    for field, source in (
        ("memory_avg_per_query_ms", "memory_update_times"),
        ("sf_trigger_avg_ms", "sf_trigger_total_times"),
        ("sf_inference_avg_ms", "sf_inference_times"),
        ("rt_anomaly_encode_avg_ms", "rt_anomaly_encode_times"),
        ("anomaly_pool_update_avg_ms", "anomaly_pool_update_times"),
    ):
        summary["timing"][field] = round(float(np.mean(timing[source])) * 1000, 1) if timing[source] else 0.0
    summary["timing"]["normal_query_avg_ms"] = round(pg_ms + (
        float(np.mean(timing["memory_update_times"])) * 1000 if timing["memory_update_times"] else 0), 1)
    for field, source in (("metrics_raw", "scores"), ("metrics_smoothed", "smoothed"),
                          ("metrics_online_smoothed", "online_smoothed")):
        metrics = base.compute_metrics(arrays[source], arrays["labels"])
        summary[field] = {k: v for k, v in metrics.items() if k not in ("fpr", "tpr")}
    pg_parts = []
    for video in videos.values():
        interval = max(1, int(video["fps"] / config["target_fps"]))
        sampled = (video["n_frames"] + interval - 1) // interval
        segments = np.repeat(video["paligemma_scores"], config["query_interval"])[:sampled]
        pg_parts.append(np.repeat(segments, interval)[:video["n_frames"]])
    pg_scores = np.concatenate(pg_parts)
    if len(pg_scores) != len(arrays["labels"]):
        raise ValueError("PaliGemma-only frame predictions are not aligned")
    summary["metrics_pg_only"] = {k: v for k, v in base.compute_metrics(pg_scores, arrays["labels"]).items()
                                  if k not in ("fpr", "tpr")}
    summary["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    summary["resumed_from"] = str(directory)
    summary["elapsed_time_note"] = "Max rank processing time, including reused work; excludes loading and downtime."
    # summary.json is written last so it signals that both final files exist.
    atomic_json(directory / "video_results.json", videos)
    atomic_json(directory / "summary.json", summary)
    logging.info("Saved complete results: %s", directory)
    for name in ("metrics_raw", "metrics_smoothed", "metrics_online_smoothed", "metrics_pg_only"):
        logging.info("%s: %s", name, summary[name])
    return summary


def configure_environment():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("BNB_CUDA_VERSION", "121")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def run_worker(directory, config, annotations, plan):
    configure_environment()
    import eval_reactvau_detection as base

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != config["world_size"]:
        raise ValueError(f"Resume requires original WORLD_SIZE={config['world_size']}; got {world_size}")
    if plan[rank]["pending"]:
        if not base.torch.cuda.is_available() or local_rank >= base.torch.cuda.device_count():
            raise ValueError(f"CUDA device {local_rank} is unavailable")
        base.torch.cuda.set_device(local_rank)
    if config["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        base.torch.use_deterministic_algorithms(True)
    # Verify prompts against the source code before combining old and new predictions.
    if base.get_grid_prompt(add_special_tokens=False) != config["paligemma_prompt"]:
        raise ValueError("PaliGemma prompt has changed since the source run")
    config["sf_prompt_template"] = base.get_sf_prompt(config["trigger_mode"], config["sf_scoring_method"], config["sf_prompt_style"])
    if config["sf_prompt_template"][:200].strip() != config["sf_prompt_preview"]:
        raise ValueError("StreamForest prompt preview has changed since the source run")

    with contextlib.ExitStack() as stack:
        if rank == 0:
            lock = stack.enter_context((directory / ".resume.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if world_size > 1:
            base.dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=24))
            stack.callback(lambda: base.dist.destroy_process_group() if base.dist.is_initialized() else None)
        logging.basicConfig(level=logging.INFO, force=True,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(directory / f"resume_rank{rank}.log")])
        manifest = {k: v for k, v in config.items() if k not in ("verbose", "sf_prompt_template")}
        manifest_path = directory / "_resume" / "config.json"
        if rank == 0:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Resume configuration changed; refusing to mix checkpoints")
            atomic_json(manifest_path, manifest)
        if world_size > 1:
            base.dist.barrier()
        assigned = sorted(annotations)[rank::world_size]
        logging.info("Rank %s: %s", rank, plan[rank])
        if not plan[rank]["completed_shard"]:
            detector = build_detector(base, config, local_rank) if plan[rank]["pending"] else None
            failures = {}
            for key in assigned:
                if load_video(directory, key, annotations[key]) is not None:
                    continue
                try:
                    arrays, metadata = process_video(base, detector, key, annotations[key], config)
                    atomic_npz(checkpoint_path(directory, key), arrays, metadata)
                    logging.info("Saved %s (%s queries)", key, metadata["video"]["num_queries"])
                except Exception as error:
                    failures[key] = str(error)
                    logging.exception("Error processing %s", key)
            atomic_json(directory / "_resume" / f"failures-rank-{rank}.json", failures)
            if failures:
                raise RuntimeError(f"Rank {rank}: {len(failures)} videos failed; saved videos will be reused on retry")
            peak = base.torch.cuda.max_memory_allocated() / 1024 ** 2 if detector is not None else 0.0
            save_rank(directory, rank, local_rank, assigned, annotations, peak)
            logging.info("Rank %s shard saved", rank)
        if world_size > 1:
            base.dist.barrier()
        if rank == 0:
            make_plan(directory, config, annotations)  # Validate all new shards before merging.
            finalize(base, directory, config, annotations)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate existing data and show pending counts; writes nothing")
    parser.add_argument("--video-dir", type=Path, help="Dataset root; defaults to annotation directory's parent")
    parser.add_argument("--paligemma-attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"],
                        help="Unlogged source option; original shell launcher uses sdpa")
    parser.add_argument("--streamforest-conv-template", default="qwen_2")
    parser.add_argument("--streamforest-time-msg", default="short_online_v2",
                        choices=["short_online", "short_online_v2", "simple", "none"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    directory = args.resume_dir.resolve()
    config = recover_config(directory)
    config.update(video_dir=str((args.video_dir or Path(config["anno_path"]).parent.parent).resolve()),
                  paligemma_attn=args.paligemma_attn,
                  streamforest_conv_template=args.streamforest_conv_template,
                  streamforest_time_msg=args.streamforest_time_msg, verbose=args.verbose)
    from detect_utils import load_anno_txt
    annotations = load_anno_txt(config["anno_path"], config["dataset"], video_dir=config["video_dir"])
    plan = make_plan(directory, config, annotations)
    if "RANK" not in os.environ:
        print(json.dumps({"resume_dir": str(directory), "world_size": config["world_size"],
                          "selected": len(annotations), "saved": sum(row["saved"] for row in plan),
                          "pending": sum(row["pending"] for row in plan), "ranks": plan}, indent=2), flush=True)
    if args.dry_run:
        return
    manifest = directory / "_resume" / "config.json"
    if (directory / "summary.json").exists():
        raise ValueError("summary.json already exists; refusing to overwrite final results")
    if (directory / "video_results.json").exists() and not (
            manifest.exists() and all(row["pending"] == 0 for row in plan)):
        raise ValueError("video_results.json already exists outside a completed resume; refusing to overwrite it")
    if "RANK" not in os.environ and config["world_size"] > 1:
        configure_environment()
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc_per_node={config['world_size']}", str(Path(__file__).resolve()), *sys.argv[1:]]
        subprocess.run(command, check=True)
    else:
        run_worker(directory, config, annotations, plan)


if __name__ == "__main__":
    main()
