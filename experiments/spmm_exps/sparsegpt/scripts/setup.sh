#!/usr/bin/env bash
set -euo pipefail

BASELINE="sparsegpt"
REPO_URL="https://github.com/IST-DASLab/sparsegpt"
# Repo root: override with SNACK_ROOT, otherwise derived from this script's location.
ROOT_DIR="${SNACK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
BASELINE_DIR="$ROOT_DIR/experiments/spmm_exps/$BASELINE"
REPO_DIR="$BASELINE_DIR/third_party/$BASELINE"
LOG_DIR="$BASELINE_DIR/logs"
SETUP_LOG="$LOG_DIR/setup.log"

mkdir -p "$LOG_DIR" "$BASELINE_DIR/third_party" "$BASELINE_DIR/results" "$BASELINE_DIR/checkpoints"

echo "[$BASELINE] setup started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SETUP_LOG"
echo "[$BASELINE] repo: $REPO_URL" | tee -a "$SETUP_LOG"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[$BASELINE] cloning into $REPO_DIR" | tee -a "$SETUP_LOG"
  git clone --recursive "$REPO_URL" "$REPO_DIR" 2>&1 | tee -a "$SETUP_LOG"
else
  echo "[$BASELINE] updating existing checkout at $REPO_DIR" | tee -a "$SETUP_LOG"
  git -C "$REPO_DIR" fetch --all --tags --prune 2>&1 | tee -a "$SETUP_LOG"
  git -C "$REPO_DIR" pull --ff-only 2>&1 | tee -a "$SETUP_LOG"
  git -C "$REPO_DIR" submodule update --init --recursive 2>&1 | tee -a "$SETUP_LOG" || true
fi

if [ -f "$REPO_DIR/requirements.txt" ]; then
  echo "[$BASELINE] installing requirements.txt" | tee -a "$SETUP_LOG"
  python -m pip install -r "$REPO_DIR/requirements.txt" 2>&1 | tee -a "$SETUP_LOG" || true
fi

if [ -f "$REPO_DIR/pyproject.toml" ] || [ -f "$REPO_DIR/setup.py" ]; then
  echo "[$BASELINE] attempting editable install" | tee -a "$SETUP_LOG"
  python -m pip install -e "$REPO_DIR" --no-build-isolation 2>&1 | tee -a "$SETUP_LOG" || true
fi



echo "[$BASELINE] setup completed" | tee -a "$SETUP_LOG"
