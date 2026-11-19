"""Smoke-test paired-GPU PG precomputation on real HIVAU clips at batch size 32.

Uses the supplied Stage-1 adapter. All fixtures, logs and score files are isolated
under test/artifacts/precompute; the production annotation/output are never changed.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--lora-path", required=True)
    args = parser.parse_args()
    annotation = json.loads(Path(args.anno_path).read_text())
    videos = sorted({row["video"] for row in annotation if "video" in row})[:96]
    assert len(videos) == 96, "Need 96 real clips to exercise full batches in both workers"
    assert all((Path(args.data_root) / video).is_file() for video in videos)
    directory = ROOT / "test/artifacts/precompute" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    subset = directory / "annotation.json"
    subset.write_text(json.dumps([{"video": video} for video in videos]))
    print(f"Smoke-test artifacts: {directory}", flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")

    def run(name, gpu_count, group_size, resume=False):
        output = directory / f"{name}.json"
        log = directory / f"{name}{'_resume' if resume else ''}.log"
        command = [sys.executable, "scripts/precompute/precompute_pg_scores.py",
                   "--anno-path", str(subset), "--data-root", args.data_root,
                   "--paligemma-lora-path", args.lora_path,
                   "--paligemma-image-size", "384", "--paligemma-vision-feature-layer", "-2",
                   "--paligemma-attn", "eager", "--batch-size", "32",
                   "--num-gpus", str(gpu_count), "--gpus-per-worker", str(group_size),
                   "--output-path", str(output)]
        if resume:
            command.append("--resume")
        print(f"Running {name}, resume={resume}, batch_size=32", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
        log_text = log.read_text()
        assert result.returncode == 0, f"{name} failed: {log}\n{log_text[-10000:]}"
        scores = json.loads(output.read_text())
        assert set(scores) == set(videos), "Missing or duplicate video assignments"
        assert not output.with_name(output.stem + "_failed.json").exists()
        assert not output.with_name(output.name + ".tmp").exists()
        for video, entry in scores.items():
            assert entry["num_queries"] == len(entry["pg_scores"]) > 0, video
            assert all(isinstance(score, (float, int)) and 0 <= score <= 1
                       for score in entry["pg_scores"]), video
        if resume:
            assert "No videos remain; skipping GPU initialization" in log_text
        else:
            for rank in range(gpu_count // group_size):
                devices = list(range(rank * group_size, (rank + 1) * group_size))
                assert f"GPUs {devices}: first inference batch=32, logits=(32, 1," in log_text
                if group_size == 2:
                    assert f"GPU pair {devices}:" in log_text
            print("\n".join(line for line in log_text.splitlines()
                            if "peak allocated=" in line), flush=True)
        return scores

    paired = run("paired", 4, 2)
    before_resume = (directory / "paired.json").read_bytes()
    assert run("paired", 4, 2, resume=True) == paired
    assert (directory / "paired.json").read_bytes() == before_resume
    single = run("single", 1, 1)
    max_error = 0
    for video, entry in paired.items():
        reference = single[video]
        assert {k: v for k, v in entry.items() if k != "pg_scores"} == {
            k: v for k, v in reference.items() if k != "pg_scores"}
        for actual, expected in zip(entry["pg_scores"], reference["pg_scores"]):
            max_error = max(max_error, abs(actual - expected))
    assert max_error <= 1e-4 + 1e-12, f"Paired/single scoring mismatch: {max_error}"
    summary = dict(passed=True, videos=len(paired),
                   queries=sum(entry["num_queries"] for entry in paired.values()),
                   batch_size=32, gpu_groups=[[0, 1], [2, 3]],
                   max_score_difference=max_error, resume_preserves_output=True)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
