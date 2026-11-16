"""Verify real four-GPU evaluation against one GPU using short synthetic clips.

Run in the ReactVAU environment with LORA_PATH pointing to a trained adapter.
All generated data, logs, and evaluation outputs stay under test/artifacts/.
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
sys.path.insert(0, str(ROOT / "eval_utils/vad"))
sys.path.insert(0, str(ROOT))
from eval_paligemma_detection import merge_rank_results
from detect_utils import compute_metrics


def check_global_auc(directory):
    """Perfect local rankings can still produce an imperfect global AUC."""
    shards = directory / "metric fixtures"
    shards.mkdir(parents=True)
    for rank in range(4):
        scores = [[0.6, 0.5], [0.4, 0.3], [], []][rank]
        labels = [1, 0] if scores else []
        np.savez(shards / f"rank-{rank}.npz", scores=scores,
                 labels=labels, smoothed=scores)
        metadata = {"video_results": {f"video{rank}": {}} if scores else {},
                    "total_queries": len(scores), "total_segments": len(scores)}
        (shards / f"rank-{rank}.json").write_text(json.dumps(metadata))
    scores, labels, _, videos, queries, segments = merge_rank_results(shards, 4)
    assert compute_metrics(scores, labels)["roc_auc"] == 0.75
    assert len(videos) == 2 and queries == segments == 4


def make_clips(dataset_root):
    videos = dataset_root / "videos/ucf-crime/videos/test"
    videos.mkdir(parents=True)
    annotations = dataset_root / "raw_annotations"
    annotations.mkdir()
    rows = []
    for index in range(4):
        name = f"clip{index}"
        writer = cv2.VideoWriter(str(videos / f"{name}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (128, 128))
        assert writer.isOpened()
        for frame_index in range(64):
            frame = np.full((128, 128, 3), 30 + index * 40, dtype=np.uint8)
            x = frame_index % 96
            cv2.rectangle(frame, (x, 40), (x + 24, 80), (40, 180, 220), -1)
            writer.write(frame)
        writer.release()
        rows.append(f"{name}.mp4 Abuse 17 40 -1 -1" if index % 2 == 0
                    else f"{name}.mp4 Normal -1 -1")
    (annotations / "ucf_database_test_anno.txt").write_text("\n".join(rows) + "\n")


def run_case(directory, name, gpu_count, samples=None):
    output = directory / name
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", NPROC_PER_NODE=str(gpu_count),
               HIVAU_ROOT=str(directory / "synthetic dataset"), DATASET="ucf-crime",
               OUTPUT_PATH=str(output), BATCH_SIZE="2", IMAGE_SIZE="384",
               TEST_MODE="true" if samples is not None else "false",
               TEST_SAMPLES=str(samples or 4), HF_HUB_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    log = directory / f"{name}.log"
    with log.open("w") as stream:
        result = subprocess.run(["bash", "scripts/eval/run_eval_paligemma_detect.sh"],
                                cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=300)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {log}\n{log.read_text()[-8000:]}")
    summaries = list(output.glob("*/summary.json"))
    assert len(summaries) == 1, summaries
    run_dir = summaries[0].parent
    summary = json.loads(summaries[0].read_text())
    videos = json.loads((run_dir / "video_results.json").read_text())
    expected = samples if samples is not None else 4
    assert summary["world_size"] == gpu_count
    assert summary["num_videos"] == len(videos) == expected
    assert summary["total_frames"] == expected * 64
    assert summary["total_queries"] == expected * 8
    assert summary["total_segments"] == expected * 32
    assert not (run_dir / "_shards").exists()
    for rank in range(gpu_count):
        text = (run_dir / ("detection.log" if rank == 0 else f"detection_rank{rank}.log")).read_text()
        assert f"Rank: {rank}/{gpu_count}, CUDA device: {rank}" in text
    return summary, videos


if __name__ == "__main__":
    directory = ROOT / "test/artifacts/eval" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    check_global_auc(directory)
    make_clips(directory / "synthetic dataset")
    print(f"Smoke-test artifacts: {directory}", flush=True)
    multi, multi_videos = run_case(directory, "four GPUs", 4)
    single, single_videos = run_case(directory, "one GPU", 1)
    assert multi_videos.keys() == single_videos.keys()
    for key in multi_videos:
        np.testing.assert_allclose(multi_videos[key]["query_scores"],
                                   single_videos[key]["query_scores"], rtol=0, atol=1e-7)
    for group in ("metrics_raw", "metrics_smoothed"):
        for metric in ("roc_auc", "pr_auc", "ap"):
            np.testing.assert_allclose(multi[group][metric], single[group][metric],
                                       rtol=0, atol=1e-12)
    run_case(directory, "empty shards", 4, samples=2)
    print("PASS: global AUC, four-GPU inference, single-GPU parity, and empty shards", flush=True)
