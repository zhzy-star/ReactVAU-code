#!/usr/bin/env bash
set -e -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${REACTVAU_CONFIG:-${SCRIPT_DIR}/../scripts/config/paths.sh}"
reactvau_activate_conda
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-}"
export BITSANDBYTES_NOWELCOME=1
exec python -B "${SCRIPT_DIR}/smoke_test_ddp.py"
