# Stage-1 isolated DDP training

All new code, caches, logs, smoke data, checkpoints, and adapters live in `test/`.
The original training entry point and `requirements.txt` are unchanged. Model
weights and grid images are read from the existing project paths. The Python
entry point reuses the production dataset/model code instead of copying it.

Run from the repository root:

```bash
# Full training: four GPUs, batch 2 per GPU, accumulation 4, global batch 32.
bash test/train_paligemma_vad_384_ddp.sh

# Real-model smoke test: four GPUs, two optimizer steps, eval every step/save every two,
# global batch 32, no warmup, then saved adapter/processor reload and inference.
bash test/smoke_ddp.sh

# Optional two-GPU mode, retaining global batch 32 (accumulation 8).
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash test/train_paligemma_vad_384_ddp.sh

# Optional single-GPU mode, retaining global batch 32.
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash test/train_paligemma_vad_384_ddp.sh

# CPU gradient-equivalence checks (also run by the smoke test).
conda run -n ReactVAU python -B test/check_loss_normalization.py

# CPU checkpoint regression: different eval/save intervals, retention, final-step selection.
conda run -n ReactVAU python -B test/check_checkpoint_lifecycle.py
```

The launcher reads the existing `scripts/config/paths.sh` and optional local
overrides. `MODEL_PATH`, `STREAMFOREST_WEIGHTS`, `DATA_PATH`, and `IMAGE_ROOT`
remain configurable. Outputs default to `test/artifacts/training/`; overriding
`OUTPUT_DIR` requires a path inside `test/`. Each launch creates one shared,
unique run directory. Each rank has its own log; only rank 0 writes shared
metadata, history, and final adapter files. Exceptions propagate to torchrun.

The implementation targets the pinned environment in `requirements.txt`:
PyTorch 2.5.1, Transformers 4.47.1, Accelerate 0.34.2, PEFT 0.14.0. No dependency
installation or upgrade is performed. Flash Attention is not required.
The launchers use the CUDA runtime bundled with PyTorch (cu124 in the checked
environment), rather than prepending the original script's system CUDA 12.1
libraries or forcing bitsandbytes to load CUDA 12.1.

Important compatibility changes:

- Bind each process to `LOCAL_RANK`; let Trainer create DDP and shard data.
- Retain video-level train/validation splitting and verify rank agreement.
- Retain non-reentrant gradient checkpointing and disable generation KV cache.
- Normalize the custom loss by supervised tokens across each accumulation
  window and all ranks. Transformers 4.47.1 does not automatically normalize
  this custom per-microbatch loss for gradient accumulation.
- Accept the Transformers 4.47.1 `log(logs, start_time)` interface; reduce
  monitoring probability sums/counts across ranks.
- Let Trainer gather validation predictions before computing global AUC.
- Save every new best checkpoint even outside the periodic save interval.
  Default eval/save intervals remain 250/500 steps; checkpoint rotation protects
  the saved best model. Evaluate the final optimizer step before best-model reload,
  even when it falls outside the evaluation interval.
- Check local adapter files before loading; incomplete checkpoints raise a local
  file error without a Hub lookup or automatic substitution of another model.
- Read the best adapter checkpoint through CPU before copying into each rank's
  model. PEFT 0.14's inferred `cuda` device is resolved as GPU 0 by safetensors,
  which otherwise makes all ranks stage adapter tensors on the same GPU.

Smoke outputs are under `test/artifacts/smoke/<timestamp>/`:

- `launcher.log`: combined torchrun output.
- `smoke_data.json`: references to 74 real grid images, 64 train / 10 validation,
  with no video overlap. Original data and images are unchanged.
- `smoke_report.json`: dependency versions, rank/device and sample-shard checks,
  finite gradient checks, exact cross-rank gradient agreement, LoRA/projector
  updates, global metrics, and saved tensor/processor reload verification.
- `outputs/training_*/`: rank logs, TensorBoard events, periodic/best checkpoints, selected
  adapter, processor, config, histories, and summaries.

The CPU regression forces the best score onto a step outside periodic saving,
then exercises rotation, reload, and final export. A separate case forces the
last step to win outside both periodic intervals. The real-model smoke test also
uses different eval/save intervals and checks that the final exported tensors
exactly match the selected best checkpoint.

The smoke test validates mechanics, not anomaly-detection quality or speedup.
Reload inference uses the same BF16 autocast as Trainer evaluation, and compares
the AUC/accuracy on all ten validation grids against the four-GPU evaluation.
Use its final adapter directory for reload checks only; it is not a trained
benchmark model. Full training outputs include the same adapter/processor files
expected by downstream code; point `PALIGEMMA_LORA_PATH` at the generated run
directory when intentionally using a completed full run.

## ReactVAU four-GPU evaluation smoke test

Set `PALIGEMMA_LORA_PATH` to the trained Fast adapter and `STREAMFOREST_MODEL_PATH`
to the evaluation-compatible Slow adapter/projector directory, then run in the
ReactVAU environment:

```bash
python test/smoke_eval_reactvau_multigpu.py
```

This runs the real models on five short generated videos with four GPUs and one
GPU, then tests two videos across four ranks. It forces Slow triggers only for
the smoke runs and enables deterministic CUDA operations to avoid existing
token-merge reduction variability. It verifies device assignments and disjoint video coverage, compares
query scores and all four metric groups, and checks empty ranks and temporary
shard cleanup. A separate fixture checks that global AUC is computed from merged
predictions and rejects duplicate videos and mismatched frame arrays. Artifacts,
rank logs, and `smoke_report.json` are under `test/artifacts/reactvau_eval/`.
These checks validate execution and aggregation, not model quality or speedup.
