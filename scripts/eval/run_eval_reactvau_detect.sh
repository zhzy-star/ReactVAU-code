#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# ReactVAU Video Anomaly Detection Evaluation (four GPUs by default)
# Combined PaliGemma + StreamForest Pipeline
# Each GPU processes disjoint videos; rank 0 computes global frame metrics.

set -e

# ==========================================
# 1. Environment Setup
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${REACTVAU_CONFIG:-${SCRIPT_DIR}/../config/paths.sh}"
# shellcheck source=/dev/null
source "${CONFIG_PATH}"

reactvau_activate_conda

# Environment variables
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=1
export BNB_CUDA_VERSION=121
export BITSANDBYTES_NOWELCOME=1

# Enable real-time output (unbuffered)
export PYTHONUNBUFFERED=1

# Redirect stderr to stdout for unified logging
exec 2>&1

# Set CUDA device
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DETERMINISTIC="${DETERMINISTIC:-false}"
if [ "$DETERMINISTIC" = true ]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    DETERMINISTIC_FLAGS=(--deterministic)
else
    DETERMINISTIC_FLAGS=()
fi

# ==========================================
# 2. Path Configuration
# ==========================================
PROJECT_ROOT="${REACTVAU_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# PaliGemma2-3B (Detection Module)
PALIGEMMA_MODEL_PATH="${PALIGEMMA_MODEL_PATH:-${CKPT_ROOT}/paligemma2-3b-mix-448}"
PALIGEMMA_LORA_PATH="${PALIGEMMA_LORA_PATH:-${CKPT_ROOT}/paligemma2-3b-vad-lora-384-combined-binary/training_384}"

# PaliGemma Image Size Configuration
PALIGEMMA_IMAGE_SIZE="${PALIGEMMA_IMAGE_SIZE:-384}" # 384 for StreamForest

# StreamForest vision encoder weights
PALIGEMMA_SF_WEIGHTS="${PALIGEMMA_SF_WEIGHTS:-${CKPT_ROOT}/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors}"
PALIGEMMA_VISION_FEATURE_LAYER="${PALIGEMMA_VISION_FEATURE_LAYER:--2}"  # -2 for StreamForest (recommended), -1 for original last layer

# StreamForest-Qwen2-7B (Reasoning Module)
# Option A: no LoRA
# STREAMFOREST_MODEL_PATH="${STREAMFOREST_MODEL_BASE}"
# STREAMFOREST_MODEL_BASE=""  # Empty = full checkpoint, not LoRA

# Option B: with LoRA fine-tuning
STREAMFOREST_MODEL_BASE="${STREAMFOREST_MODEL_BASE:-${CKPT_ROOT}/StreamForest-Qwen2-7B_Siglip}"
STREAMFOREST_MODEL_PATH="${STREAMFOREST_MODEL_PATH:-${REACTVAU_CHECKPOINT_PATH}}"

# Dataset configuration
VIDEO_DIR="${VIDEO_DIR:-${HIVAU_ROOT}}"
DATASET="${DATASET:-ucf-crime}"  # Options: ucf-crime, xd-violence
if [ "$DATASET" = "ucf-crime" ]; then
    ANNO_PATH="${VIDEO_DIR}/raw_annotations/ucf_database_test_anno.txt"
else
    ANNO_PATH="${VIDEO_DIR}/raw_annotations/xd_database_test_anno.txt"
fi

# Output path
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_results/vad}"

# ==========================================
# 3. Pipeline Configuration
# ==========================================
TARGET_FPS=4            # 4 FPS sampling
QUERY_INTERVAL=4        # 4 frames per query = 1 query per second

# PaliGemma settings
PALIGEMMA_BATCH_SIZE="${PALIGEMMA_BATCH_SIZE:-1}"  # Per GPU, with both models loaded
PALIGEMMA_ATTN="sdpa"

if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ||
      ! "$PALIGEMMA_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NPROC_PER_NODE and PALIGEMMA_BATCH_SIZE must be positive integers"
    exit 1
fi
python - "$NPROC_PER_NODE" <<'PY'
import sys
import torch

requested = int(sys.argv[1])
available = torch.cuda.device_count()
if not torch.cuda.is_available() or available < requested:
    raise SystemExit(f"ERROR: Requested {requested} GPUs, but only {available} CUDA GPUs are visible")
PY

# StreamForest settings  
STREAMFOREST_CONV_TEMPLATE="qwen_2"
STREAMFOREST_TIME_MSG="short_online_v2"
STREAMFOREST_QUANTIZATION="none"

# ==========================================
# 4. Trigger Configuration
# ==========================================
TRIGGER_MODE="score"

# Anomaly threshold for triggering StreamForest
# When PaliGemma score >= threshold, StreamForest is activated
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-0.3486}"     # UCF-Crime training recall90
# ANOMALY_THRESHOLD=0.4073     # XD-Violence training recall90

# Score fusion configuration
SCORE_FUSION="weighted"
FUSION_ALPHA=0.40

# SF scoring method: "binary" (logit-based Yes/No)
SF_SCORING_METHOD="binary"

# SF memory enhancement (Anomaly Pool + APS) and RT-Anomaly
SF_ENHANCE_MEMORY="${SF_ENHANCE_MEMORY:-true}"
RT_ANOMALY="${RT_ANOMALY:-true}"
POOL_THRESHOLD=0.6

# SF prompt style
SF_PROMPT_STYLE="skeptical"

# Online smoothing parameters
ONLINE_SMOOTH_ALPHA=0.60
ONLINE_SMOOTH_BETA=0.95

# ==========================================
# 5. Test Mode Configuration
# ==========================================
TEST_MODE="${TEST_MODE:-false}"
TEST_SAMPLES="${TEST_SAMPLES:-5}"  # Override with TEST_SAMPLES=1 for a quick 24GB validation

# ==========================================
# 6. Run Evaluation
# ==========================================
cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p "$OUTPUT_PATH"

reactvau_require_dir "$PALIGEMMA_MODEL_PATH" "PaliGemma model directory"
reactvau_require_dir "$STREAMFOREST_MODEL_PATH" "ReactVAU/StreamForest checkpoint directory"
if [ -n "$STREAMFOREST_MODEL_BASE" ]; then
    reactvau_require_dir "$STREAMFOREST_MODEL_BASE" "StreamForest base model directory"
fi
reactvau_require_dir "$VIDEO_DIR" "HIVAU root"
reactvau_require_file "$ANNO_PATH" "VAD annotation"
if [ "$PALIGEMMA_IMAGE_SIZE" -eq 384 ]; then
    reactvau_require_file "$PALIGEMMA_SF_WEIGHTS" "PaliGemma StreamForest vision weights"
fi

echo "=========================================="
echo "ReactVAU VAD Evaluation (${NPROC_PER_NODE} GPUs)"
echo "=========================================="
echo "Start Time: $(date)"
echo ""
echo "PaliGemma Model: $PALIGEMMA_MODEL_PATH"
if [ -n "$PALIGEMMA_LORA_PATH" ] && [ -d "$PALIGEMMA_LORA_PATH" ]; then
    echo "PaliGemma LoRA: $PALIGEMMA_LORA_PATH"
    PG_LORA_FLAG="--paligemma-lora-path $PALIGEMMA_LORA_PATH"
else
    echo "PaliGemma LoRA: None (using base model)"
    PG_LORA_FLAG=""
fi
echo "PaliGemma Image Size: ${PALIGEMMA_IMAGE_SIZE}x${PALIGEMMA_IMAGE_SIZE}"
if [ "$PALIGEMMA_IMAGE_SIZE" -eq 384 ]; then
    echo "PaliGemma StreamForest Weights: $PALIGEMMA_SF_WEIGHTS"
    echo "PaliGemma Vision Feature Layer: $PALIGEMMA_VISION_FEATURE_LAYER"
    PG_VISION_FLAGS="--paligemma-image-size 384 --paligemma-streamforest-weights $PALIGEMMA_SF_WEIGHTS --paligemma-vision-feature-layer $PALIGEMMA_VISION_FEATURE_LAYER"
else
    PG_VISION_FLAGS="--paligemma-image-size 448"
fi
echo ""
echo "StreamForest Model: $STREAMFOREST_MODEL_PATH"
if [ -n "$STREAMFOREST_MODEL_BASE" ]; then
    echo "StreamForest Base: $STREAMFOREST_MODEL_BASE"
    SF_BASE_FLAG="--streamforest-model-base $STREAMFOREST_MODEL_BASE"
else
    echo "StreamForest Base: None (full checkpoint)"
    SF_BASE_FLAG=""
fi
echo "StreamForest Quantization: $STREAMFOREST_QUANTIZATION"
if [ "$SF_ENHANCE_MEMORY" = true ]; then
    echo "StreamForest Memory Enhancement (Pool+APS): ENABLED"
    MEMORY_FLAG="--enable-memory-enhancement"
else
    echo "StreamForest Memory Enhancement (Pool+APS): DISABLED"
    MEMORY_FLAG=""
fi
if [ "$RT_ANOMALY" = true ]; then
    echo "RT-Anomaly Dense Encoding: ENABLED"
    RT_ANOMALY_FLAG="--enable-rt-anomaly"
else
    echo "RT-Anomaly Dense Encoding: DISABLED"
    RT_ANOMALY_FLAG=""
fi
echo "StreamForest Prompt Style: $SF_PROMPT_STYLE"
echo ""
echo "Dataset: $DATASET"
echo "Video Dir: $VIDEO_DIR"
echo "Annotation: $ANNO_PATH"
echo ""
echo "Pipeline Config:"
echo "  Target FPS: $TARGET_FPS"
echo "  Query Interval: $QUERY_INTERVAL"
echo "  Trigger Mode: $TRIGGER_MODE"
echo "  Anomaly Threshold: $ANOMALY_THRESHOLD"
echo "  Pool Threshold: $POOL_THRESHOLD"
echo "  Score Fusion: $SCORE_FUSION"
echo "  SF Scoring Method: $SF_SCORING_METHOD"
echo "  GPUs: $NPROC_PER_NODE (visible: $CUDA_VISIBLE_DEVICES)"
echo "  Deterministic execution: $DETERMINISTIC"
echo "  PaliGemma Batch Size per GPU: $PALIGEMMA_BATCH_SIZE"
echo "  Online Smoothing: alpha=$ONLINE_SMOOTH_ALPHA, beta=$ONLINE_SMOOTH_BETA"
echo ""
echo "Output: $OUTPUT_PATH"
echo "=========================================="

# Build test mode flags
if [ "$TEST_MODE" = true ]; then
    echo "🧪 TEST MODE: Processing $TEST_SAMPLES random samples"
    TEST_FLAGS="--test-mode --test-samples $TEST_SAMPLES"
else
    echo "🚀 FULL EVALUATION MODE"
    TEST_FLAGS=""
fi
echo ""

# Run evaluation
LAUNCHER=(python)
if [ "$NPROC_PER_NODE" -gt 1 ]; then
    LAUNCHER+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE")
fi
"${LAUNCHER[@]}" "${PROJECT_ROOT}/eval_utils/vad/eval_reactvau_detection.py" \
    --paligemma-model-path "$PALIGEMMA_MODEL_PATH" \
    $PG_LORA_FLAG \
    $PG_VISION_FLAGS \
    --paligemma-attn "$PALIGEMMA_ATTN" \
    --streamforest-model-path "$STREAMFOREST_MODEL_PATH" \
    $SF_BASE_FLAG \
    --streamforest-conv-template "$STREAMFOREST_CONV_TEMPLATE" \
    --streamforest-time-msg "$STREAMFOREST_TIME_MSG" \
    --streamforest-quantization "$STREAMFOREST_QUANTIZATION" \
    --anomaly-threshold "$ANOMALY_THRESHOLD" \
    --pool-threshold "$POOL_THRESHOLD" \
    --trigger-mode "$TRIGGER_MODE" \
    --sf-scoring-method "$SF_SCORING_METHOD" \
    --sf-prompt-style "$SF_PROMPT_STYLE" \
    --score-fusion "$SCORE_FUSION" \
    --fusion-alpha "$FUSION_ALPHA" \
    --target-fps "$TARGET_FPS" \
    --query-interval "$QUERY_INTERVAL" \
    --batch-size "$PALIGEMMA_BATCH_SIZE" \
    --online-smooth-alpha "$ONLINE_SMOOTH_ALPHA" \
    --online-smooth-beta "$ONLINE_SMOOTH_BETA" \
    --dataset "$DATASET" \
    --video-dir "$VIDEO_DIR" \
    --anno-path "$ANNO_PATH" \
    --output-path "$OUTPUT_PATH" \
    $MEMORY_FLAG \
    $RT_ANOMALY_FLAG \
    --save-video-scores \
    "${DETERMINISTIC_FLAGS[@]}" \
    $TEST_FLAGS

echo ""
echo "=========================================="
echo "✅ Evaluation complete!"
echo "End Time: $(date)"
echo "Results saved in: $OUTPUT_PATH"
echo "=========================================="
