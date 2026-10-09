#!/usr/bin/env bash
set -euo pipefail
BASELINE="smat"
REPO_URL="https://github.com/spcl/smat"
# Repo root: override with SNACK_ROOT, otherwise derived from this script's location.
ROOT_DIR="${SNACK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
BASELINE_DIR="$ROOT_DIR/experiments/spmm_exps/$BASELINE"
REPO_DIR="$BASELINE_DIR/third_party/$BASELINE"
LOG_DIR="$BASELINE_DIR/logs"
SETUP_LOG="$LOG_DIR/setup.log"
mkdir -p "$LOG_DIR" "$BASELINE_DIR/third_party" "$BASELINE_DIR/results" "$BASELINE_DIR/checkpoints"
if [ ! -d "$REPO_DIR/.git" ]; then git clone --recursive "$REPO_URL" "$REPO_DIR" 2>&1 | tee -a "$SETUP_LOG"; else git -C "$REPO_DIR" pull --ff-only 2>&1 | tee -a "$SETUP_LOG" || true; fi
if command -v nvcc >/dev/null 2>&1; then
  (cd "$REPO_DIR/src" && make -j"$(nproc)") 2>&1 | tee -a "$SETUP_LOG" || true
else
  echo "[$BASELINE] nvcc not found; skipping CUDA build" | tee -a "$SETUP_LOG"
fi
