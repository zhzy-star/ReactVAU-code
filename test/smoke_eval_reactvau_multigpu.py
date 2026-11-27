"""Check real-model four-GPU ReactVAU inference, parity, and empty ranks.

Set STREAMFOREST_MODEL_PATH and PALIGEMMA_LORA_PATH to trained checkpoints.
Generated clips and results stay under test/artifacts/reactvau_eval/.
"""

import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval_utils/vad"))
from eval_reactvau_detection import merge_rank_results
from detect_utils import compute_metrics


def check_merge(directory):
    shards = directory / "metric fixtures"
    shards.mkdir(parents=True)
    for rank in range(4):
        scores = [[0.6, 0.5], [0.4, 0.3], [], []][rank]
        np.savez(shards / f"rank-{rank}.npz", scores=scores,
                 labels=[1, 0] if scores else [], smoothed=scores,
                 online_smoothed=scores)
        metadata = {
            "video_results": {f"video{rank}": {}} if scores else {},
            "failed_videos": {}, "total_queries": len(scores),
            "total_segments": len(scores), "total_triggers": len(scores),
            "timing": {"pg_infer_total": float(rank),
                       "sf_trigger_total_times": [0.1] * len(scores)},
            "rank_stats": {"rank": rank},
        }
        (shards / f"rank-{rank}.json").write_text(json.dumps(metadata))
    arrays, videos, totals, timing, ranks, failures = merge_rank_results(shards, 4)
    for name in ("scores", "smoothed", "online_smoothed"):
        assert compute_metrics(arrays[name], arrays["labels"])["roc_auc"] == 0.75
    assert len(videos) == 2 and len(ranks) == 4 and not failures
    assert all(value == 4 for value in totals.values())
    assert timing["pg_infer_total"] == 6.0 and len(timing["sf_trigger_total_times"]) == 4
    path = shards / "rank-1.json"
    original = path.read_text()
    metadata = json.loads(original)
    metadata["video_results"] = {"video0": {}}
    path.write_text(json.dumps(metadata))
    try:
        merge_rank_results(shards, 4)
    except ValueError as error:
        assert "multiple ranks" in str(error)
    else:
        raise AssertionError("Duplicate videos were accepted")
    path.write_text(original)
    np.savez(shards / "rank-3.npz", scores=[], labels=[], smoothed=[], online_smoothed=[0.1])
    try:
        merge_rank_results(shards, 4)
    except ValueError as error:
        assert "Mismatched" in str(error)
    else:
        raise AssertionError("Unaligned frame arrays were accepted")


def make_clips(dataset_root):
    videos = dataset_root / "videos/ucf-crime/videos/test"
    videos.mkdir(parents=True)
    annotations = dataset_root / "raw_annotations"
    annotations.mkdir()
    sizes, rows = {}, []
    for index, count in enumerate([32, 48, 64, 40, 56]):
        name = f"clip{index}"
        sizes[name] = count
        writer = cv2.VideoWriter(str(videos / f"{name}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (128, 128))
        assert writer.isOpened()
        for frame_index in range(count):
            frame = np.full((128, 128, 3), 30 + index * 35, dtype=np.uint8)
            x = frame_index % 96
            cv2.rectangle(frame, (x, 40), (x + 24, 80), (40, 180, 220), -1)
            writer.write(frame)
        writer.release()
        rows.append(f"{name}.mp4 Abuse 9 24 -1 -1" if index % 2 == 0
                    else f"{name}.mp4 Normal -1 -1")
    (annotations / "ucf_database_test_anno.txt").write_text("\n".join(rows) + "\n")
    return sizes


def run_case(directory, name, gpu_count, sizes, samples=None):
    output = directory / name
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", NPROC_PER_NODE=str(gpu_count),
               HIVAU_ROOT=str(directory / "synthetic dataset"),
               VIDEO_DIR=str(directory / "synthetic dataset"), DATASET="ucf-crime",
               OUTPUT_PATH=str(output), PALIGEMMA_BATCH_SIZE="1",
               TEST_MODE="true" if samples is not None else "false",
               TEST_SAMPLES=str(samples or len(sizes)), ANOMALY_THRESHOLD="0",
               DETERMINISTIC="true",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    log = directory / f"{name}.log"
    print(f"Running {name}; log: {log}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(["bash", "scripts/eval/run_eval_reactvau_detect.sh"],
                                cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {log}\n{log.read_text()[-10000:]}")
    summaries = list(output.glob("*/summary.json"))
    assert len(summaries) == 1, summaries
    run_dir = summaries[0].parent
    summary = json.loads(summaries[0].read_text())
    videos = json.loads((run_dir / "video_results.json").read_text())
    expected = samples if samples is not None else len(sizes)
    assert summary["world_size"] == gpu_count
    assert summary["deterministic"] is True
    assert summary["num_videos"] == summary["num_selected_videos"] == len(videos) == expected
    assert summary["num_failed_videos"] == 0 and not summary["failed_videos"]
    expected_frames = sum(sizes[key] for key in videos)
    assert summary["total_frames"] == expected_frames
    assert summary["total_segments"] == expected_frames // 2
    assert summary["total_queries"] == expected_frames // 8
    assert summary["total_triggers"] == summary["total_queries"] > 0
    assert not (run_dir / "_shards").exists()
    assigned = []
    for rank, stats in enumerate(summary["rank_stats"]):
        assert stats["rank"] == stats["local_rank"] == rank
        keys = stats["assigned_video_keys"]
        assigned.extend(keys)
        assert stats["num_videos"] == len(keys)
        if keys:
            assert 20000 < stats["peak_gpu_memory_allocated_mb"] < 24564
        else:
            assert stats["peak_gpu_memory_allocated_mb"] == 0
        text = (run_dir / ("detection.log" if rank == 0 else f"detection_rank{rank}.log")).read_text()
        assert f"Rank: {rank}/{gpu_count}, CUDA device: {rank}" in text
        assert "Traceback" not in text and "Error processing" not in text
    assert len(assigned) == len(set(assigned)) == expected
    assert set(assigned) == videos.keys()
    print(f"PASS {name}: {expected} videos, {summary['total_queries']} queries, "
          f"{summary['total_triggers']} Slow triggers", flush=True)
    return summary, videos, str(run_dir)


def main():
    for variable in ("PALIGEMMA_LORA_PATH", "STREAMFOREST_MODEL_PATH"):
        path = Path(os.environ[variable])
        assert (path / "adapter_config.json").is_file(), path
    directory = ROOT / "test/artifacts/reactvau_eval" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print(f"Smoke-test artifacts: {directory}", flush=True)
    check_merge(directory)
    sizes = make_clips(directory / "synthetic dataset")
    multi, multi_videos, multi_path = run_case(directory, "four GPUs", 4, sizes)
    single, single_videos, single_path = run_case(directory, "one GPU", 1, sizes)
    assert multi_videos.keys() == single_videos.keys()
    max_delta = 0.0
    for key in multi_videos:
        for field in ("query_scores", "paligemma_scores"):
            a, b = np.asarray(multi_videos[key][field]), np.asarray(single_videos[key][field])
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
            max_delta = max(max_delta, float(np.max(np.abs(a - b))))
    for group in ("metrics_raw", "metrics_smoothed", "metrics_online_smoothed", "metrics_pg_only"):
        for metric in ("roc_auc", "pr_auc", "ap"):
            np.testing.assert_allclose(multi[group][metric], single[group][metric], rtol=0, atol=1e-12)
    empty, _, empty_path = run_case(directory, "empty shards", 4, sizes, samples=2)
    assert [stats["num_videos"] for stats in empty["rank_stats"]] == [1, 1, 0, 0]
    report = {"status": "passed", "global_metrics_not_rank_averages": "passed",
              "duplicate_and_mismatched_shards_rejected": "passed",
              "four_gpu_vs_single_gpu_max_score_delta": max_delta,
              "deterministic_execution": True,
              "all_four_metric_groups_match": "passed", "empty_ranks": "passed",
              "runs": {"four_gpu": multi_path, "single_gpu": single_path, "empty_ranks": empty_path},
              "note": "Synthetic-video real-model smoke test with forced Slow triggers; not a benchmark."}
    (directory / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: four-GPU inference, single-GPU parity, global metrics, and empty ranks", flush=True)


if __name__ == "__main__":
    main()
