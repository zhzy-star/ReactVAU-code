"""CPU regression tests for resume coverage, interruption recovery, and merging.

Run inside the ReactVAU environment:
    python test/test_resume_reactvau_detection.py

Inference is replaced by deterministic fixtures; no model weights are loaded.
The distributed test exercises four real Gloo workers on CPU.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval_utils/vad"))
sys.path.insert(0, str(ROOT))
import resume_reactvau_detection as resume

resume.configure_environment()
import eval_reactvau_detection as base


def write_log(directory, count=6, world_size=4):
    anno = directory / "raw_annotations" / "ucf_database_test_anno.txt"
    anno.parent.mkdir()
    anno.write_text("".join(f"clip{i}.mp4 Abuse 1 2 -1 -1\n" for i in range(count)))
    pg_prompt = base.get_grid_prompt(add_special_tokens=False)
    sf_preview = base.get_sf_prompt("score", "binary", "skeptical")[:200]
    (directory / "detection.log").write_text(f"""Rank: 0/{world_size}, CUDA device: 0
Deterministic execution: True
PaliGemma model: /fixture/pg
PaliGemma LoRA: None
StreamForest model: /fixture/sf
StreamForest base: None
StreamForest quantization: none
Dataset: ucf-crime
Trigger mode: score
Score fusion: weighted (alpha=0.4)
SF scoring method: binary
Anomaly threshold: 0.3486
Target FPS: 4, Query interval: 4
PaliGemma batch size: 1
PaliGemma image size: 384x384
PaliGemma StreamForest weights: /fixture/vision.safetensors
PaliGemma vision feature layer: -2
PaliGemma prompt: '{pg_prompt}'
SF prompt preview: {sf_preview}...
Loading annotation from {anno}
Total videos to process: {count}
Memory enhancement (Pool+APS): ENABLED
RT-Anomaly dense encoding: ENABLED
Pool threshold: 0.6 (training=0.6)
SF prompt style: skeptical
Online smoothing: alpha=0.6, beta=0.95
""")


def config_for(directory):
    config = resume.recover_config(directory)
    config.update(video_dir=str(directory), paligemma_attn="sdpa",
                  streamforest_conv_template="qwen_2", streamforest_time_msg="short_online_v2", verbose=False)
    annotations = base.load_anno_txt(config["anno_path"], config["dataset"])
    return config, annotations


def fixture_video(_base, _detector, key, annotation, _config):
    index = int(key.removeprefix("clip"))
    scores = np.array([0.1 + index * 0.1, 0.9 - index * 0.1] * 2)
    labels = base.make_gt_labels_from_anno(annotation, 4)
    arrays = dict(scores=scores, labels=labels, smoothed=scores * 0.9, online_smoothed=scores * 0.8)
    metadata = {"video_key": key, "video": dict(
        video_name=key, num_queries=1, trigger_count=1, query_scores=[float(scores[0])],
        paligemma_scores=[float(scores[1])], n_frames=4, fps=4.0,
        events_frames=annotation["intervals_raw"], trigger_details=[]),
        "num_segments": 4, "timing": {"pg_infer_total": 0.1, **{k: [0.01] for k in resume.TIMING_LISTS}},
        "elapsed_time_sec": 0.2}
    return arrays, metadata


def checkpoint(directory, key, annotation, config):
    arrays, metadata = fixture_video(base, None, key, annotation, config)
    resume.atomic_npz(resume.checkpoint_path(directory, key), arrays, metadata)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(directory):
    config, annotations = config_for(directory)
    rank = int(os.environ["RANK"])
    plan = resume.make_plan(directory, config, annotations)

    def build(_base, _config, local_rank):
        (directory / f"loaded-{local_rank}").write_text("loaded")
        return object()

    def process(*args):
        key = args[2]
        if resume.checkpoint_path(directory, key).exists():
            raise AssertionError(f"Repeated inference: {key}")
        (directory / f"inferred-{key}").write_text(str(rank))
        return fixture_video(*args)

    with mock.patch.object(base.torch.cuda, "is_available", return_value=True), \
         mock.patch.object(base.torch.cuda, "device_count", return_value=4), \
         mock.patch.object(base.torch.cuda, "set_device"), \
         mock.patch.object(base.torch.cuda, "max_memory_allocated", return_value=0), \
         mock.patch.object(resume, "build_detector", side_effect=build), \
         mock.patch.object(resume, "process_video", side_effect=process):
        resume.run_worker(directory, config, annotations, plan)


class ResumeTests(unittest.TestCase):
    def test_distributed_resume_and_global_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_log(directory)
            config, annotations = config_for(directory)
            keys = sorted(annotations)
            # Reuse two completed source ranks, with no per-video files for them.
            for rank in (1, 2):
                assigned = keys[rank::4]
                for key in assigned:
                    checkpoint(directory, key, annotations[key], config)
                resume.save_rank(directory, rank, rank, assigned, annotations, 0)
                for key in assigned:
                    resume.checkpoint_path(directory, key).unlink()
            checkpoint(directory, "clip0", annotations["clip0"], config)
            before = {p: digest(p) for p in directory.rglob("*") if p.is_file()}
            plan = resume.make_plan(directory, config, annotations)
            self.assertEqual([p["pending"] for p in plan], [1, 0, 0, 1])
            for p, value in before.items():
                self.assertEqual(digest(p), value)
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                       str(Path(__file__).resolve()), "--worker", str(directory)]
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout)
            for p, value in before.items():
                self.assertEqual(digest(p), value, f"Source file changed: {p}")
            self.assertEqual({p.name for p in directory.glob("loaded-*")}, {"loaded-0", "loaded-3"})
            self.assertEqual({p.name for p in directory.glob("inferred-*")}, {"inferred-clip4", "inferred-clip3"})
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["num_videos"], 6)
            self.assertEqual(summary["num_failed_videos"], 0)
            self.assertEqual(summary["total_frames"], 24)
            self.assertEqual(summary["total_queries"], 6)
            self.assertEqual(summary["total_triggers"], 6)
            self.assertEqual(summary["total_segments"], 24)
            self.assertEqual(len(list((directory / "_shards").glob("*.npz"))), 4)
            fixtures = [fixture_video(base, None, k, annotations[k], config)[0] for k in keys]
            labels = np.concatenate([a["labels"] for a in fixtures])
            for field, source in (("metrics_raw", "scores"), ("metrics_smoothed", "smoothed"),
                                  ("metrics_online_smoothed", "online_smoothed")):
                expected = base.compute_metrics(np.concatenate([a[source] for a in fixtures]), labels)
                for metric in ("roc_auc", "pr_auc", "ap"):
                    self.assertAlmostEqual(summary[field][metric], expected[metric], places=12)
            pg_scores = np.concatenate([np.repeat(fixture_video(base, None, k, annotations[k], config)[1]
                                                 ["video"]["paligemma_scores"], 4) for k in keys])
            expected = base.compute_metrics(pg_scores, labels)
            for metric in ("roc_auc", "pr_auc", "ap"):
                self.assertAlmostEqual(summary["metrics_pg_only"][metric], expected[metric], places=12)

    def test_interrupted_shard_commit_and_corrupt_data(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_log(directory, count=1, world_size=1)
            config, annotations = config_for(directory)
            key = "clip0"
            checkpoint(directory, key, annotations[key], config)
            resume.save_rank(directory, 0, 0, [key], annotations, 0)
            shard = directory / "_shards/rank-0.json"
            shard.unlink()  # Simulate an interruption before JSON commit.
            plan = resume.make_plan(directory, config, annotations)
            self.assertEqual(plan[0]["pending"], 0)
            self.assertFalse(plan[0]["completed_shard"])
            resume.save_rank(directory, 0, 0, [key], annotations, 0)
            self.assertTrue(resume.make_plan(directory, config, annotations)[0]["completed_shard"])
            metadata = json.loads(shard.read_text())
            metadata["rank_stats"]["assigned_video_keys"] = ["other"]
            resume.atomic_json(shard, metadata)
            with self.assertRaisesRegex(ValueError, "assignment differs"):
                resume.make_plan(directory, config, annotations)
            shard.unlink()
            arrays, metadata = resume.load_video(directory, key, annotations[key])
            arrays["scores"][0] = float("nan")
            resume.atomic_npz(resume.checkpoint_path(directory, key), arrays, metadata)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                resume.make_plan(directory, config, annotations)

    def test_failed_video_retry_reuses_successes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_log(directory, count=2, world_size=1)
            config, annotations = config_for(directory)
            plan = resume.make_plan(directory, config, annotations)

            def interrupted(*args):
                if args[2] == "clip1":
                    raise RuntimeError("fixture interruption")
                return fixture_video(*args)

            with mock.patch.object(base.torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(base.torch.cuda, "device_count", return_value=1), \
                 mock.patch.object(base.torch.cuda, "set_device"), \
                 mock.patch.object(base.torch.cuda, "max_memory_allocated", return_value=0), \
                 mock.patch.object(resume, "build_detector", return_value=object()), \
                 mock.patch.object(resume, "process_video", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "1 videos failed"):
                    resume.run_worker(directory, config, annotations, plan)
                self.assertFalse((directory / "summary.json").exists())
                saved = resume.checkpoint_path(directory, "clip0")
                old_hash = digest(saved)
                plan = resume.make_plan(directory, config, annotations)
                self.assertEqual(plan[0]["pending"], 1)
                with mock.patch.object(resume, "process_video", side_effect=fixture_video) as infer:
                    resume.run_worker(directory, config, annotations, plan)
                    self.assertEqual(infer.call_count, 1)
                    self.assertEqual(infer.call_args.args[2], "clip1")
                self.assertEqual(digest(saved), old_hash)
                self.assertTrue((directory / "summary.json").exists())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    else:
        unittest.main()
