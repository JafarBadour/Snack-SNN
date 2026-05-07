#!/usr/bin/env bash
set -euo pipefail
BASELINE="spinfer"
REPO_URL="https://github.com/MachineLearningSystem/25Eurosys-SpInfer"
ROOT_DIR="/home/jafar/Arts/Parallel-Dynamic-Sparse-Training"
BASELINE_DIR="$ROOT_DIR/experiments/spmm_exps/$BASELINE"
REPO_DIR="$BASELINE_DIR/third_party/$BASELINE"
LOG_DIR="$BASELINE_DIR/logs"
SETUP_LOG="$LOG_DIR/setup.log"
mkdir -p "$LOG_DIR" "$BASELINE_DIR/third_party" "$BASELINE_DIR/results" "$BASELINE_DIR/checkpoints"
if [ ! -d "$REPO_DIR/.git" ]; then git clone --recursive "$REPO_URL" "$REPO_DIR" 2>&1 | tee -a "$SETUP_LOG"; else git -C "$REPO_DIR" pull --ff-only 2>&1 | tee -a "$SETUP_LOG" || true; fi
if [ -f "$REPO_DIR/spinfer.yml" ]; then python -m pip install -r "$REPO_DIR/requirements.txt" 2>&1 | tee -a "$SETUP_LOG" || true; fi
if command -v nvcc >/dev/null 2>&1; then
  mkdir -p "$REPO_DIR/build"
  (cd "$REPO_DIR/build" && make -j"$(nproc)") 2>&1 | tee -a "$SETUP_LOG" || true
  (cd "$REPO_DIR/kernel_benchmark" && make -j"$(nproc)") 2>&1 | tee -a "$SETUP_LOG" || true
else
  echo "[$BASELINE] nvcc not found; skipping CUDA build" | tee -a "$SETUP_LOG"
fi
