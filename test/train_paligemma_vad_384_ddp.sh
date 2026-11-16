#!/usr/bin/env bash
# Isolated Stage-1 DDP launcher. All generated files stay below test/.
set -e -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REACTVAU_CONFIG:-${SCRIPT_DIR}/../scripts/config/paths.sh}"
reactvau_activate_conda
cd "${REACTVAU_ROOT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REACTVAU_ROOT}:${REACTVAU_ROOT}/vad:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export NVIDIA_TF32_OVERRIDE=1
# Let PyTorch/bitsandbytes select the CUDA libraries bundled with their wheels.
# Forcing CUDA_HOME=12.1 ahead of this environment's cu124 libraries breaks nvJitLink.
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-}"
export BITSANDBYTES_NOWELCOME=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Keep package caches, compiler output, and temporary files isolated too.
export TMPDIR="${SCRIPT_DIR}/artifacts/runtime/tmp"
export XDG_CACHE_HOME="${SCRIPT_DIR}/artifacts/runtime/cache"
export HF_HOME="${SCRIPT_DIR}/artifacts/runtime/huggingface"
export TORCH_HOME="${SCRIPT_DIR}/artifacts/runtime/torch"
export TORCH_EXTENSIONS_DIR="${SCRIPT_DIR}/artifacts/runtime/torch_extensions"
export TRITON_CACHE_DIR="${SCRIPT_DIR}/artifacts/runtime/triton"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" \
    "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ || ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NPROC_PER_NODE and BATCH_SIZE must be positive integers' >&2
    exit 1
fi
if [[ -z "${GRAD_ACCUM:-}" ]]; then
    if (( 32 % (NPROC_PER_NODE * BATCH_SIZE) != 0 )); then
        echo 'Set GRAD_ACCUM explicitly: global batch 32 is not divisible by GPU count * batch' >&2
        exit 1
    fi
    GRAD_ACCUM=$((32 / (NPROC_PER_NODE * BATCH_SIZE)))
fi
MODEL_PATH="${MODEL_PATH:-${PALIGEMMA_MODEL_PATH}}"
STREAMFOREST_WEIGHTS="${STREAMFOREST_WEIGHTS:-${PALIGEMMA_SF_WEIGHTS}}"
DATA_PATH="${DATA_PATH:-${VAD_TRAIN_ROOT}/combined_train_binary.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${VAD_TRAIN_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/artifacts/training}"
reactvau_require_dir "${MODEL_PATH}" 'PaliGemma model'
reactvau_require_file "${STREAMFOREST_WEIGHTS}" 'StreamForest vision weights'
reactvau_require_file "${DATA_PATH}" 'VAD training JSON'
reactvau_require_dir "${IMAGE_ROOT}" 'VAD image root'
python -c 'import sys, torch; n=int(sys.argv[1]); assert torch.cuda.device_count() >= n, "Not enough visible GPUs"' "${NPROC_PER_NODE}"
echo "DDP GPUs=${CUDA_VISIBLE_DEVICES}; global batch=${NPROC_PER_NODE} x ${BATCH_SIZE} x ${GRAD_ACCUM} = $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))"

exec python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" --log-dir="${SCRIPT_DIR}/artifacts/torchrun" \
    "${SCRIPT_DIR}/train_paligemma_vad_384_ddp.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --streamforest_vision_weights_path "${STREAMFOREST_WEIGHTS}" \
    --data_path "${DATA_PATH}" --image_root "${IMAGE_ROOT}" --output_dir "${OUTPUT_DIR}" \
    --vision_feature_layer "${VISION_FEATURE_LAYER:--2}" \
    --use_lora True --lora_r "${LORA_R:-16}" --lora_alpha "${LORA_ALPHA:-32}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" --freeze_vision_encoder True --train_projector True \
    --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
    --max_length "${MAX_LENGTH:-96}" --val_split "${VAL_SPLIT:-0.05}" \
    --num_train_epochs "${NUM_EPOCHS:-1}" --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" --gradient_checkpointing True \
    --ddp_find_unused_parameters False --ddp_broadcast_buffers False \
    --learning_rate "${LEARNING_RATE:-1e-5}" --warmup_ratio "${WARMUP_RATIO:-0.1}" \
    --weight_decay 0.01 --lr_scheduler_type cosine --bf16 True --tf32 True \
    --optim adamw_torch_fused --logging_steps "${LOGGING_STEPS:-50}" \
    --save_steps "${SAVE_STEPS:-500}" --eval_steps "${EVAL_STEPS:-250}" \
    --save_total_limit 3 --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --dataloader_pin_memory True --dataloader_prefetch_factor 2 \
    --report_to "${REPORT_TO:-tensorboard}" --load_best_model_at_end True \
    --metric_for_best_model eval_auc --greater_is_better True --seed 42 \
    --smoke_audit "${SMOKE_AUDIT:-False}" "$@"
