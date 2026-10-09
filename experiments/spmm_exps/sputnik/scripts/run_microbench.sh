#!/usr/bin/env bash
set -euo pipefail

BASELINE="sputnik"
METHOD="test_sputnik"
# Repo root: override with SNACK_ROOT, otherwise derived from this script's location.
ROOT_DIR="${SNACK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
BASELINE_DIR="$ROOT_DIR/experiments/spmm_exps/$BASELINE"
LOG_DIR="$BASELINE_DIR/logs"
RESULTS_DIR="$BASELINE_DIR/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "[$BASELINE] torch.cuda is unavailable; skipping benchmark in this environment" | tee -a "$LOG_DIR/bench.log"
  exit 0
fi

echo "[$BASELINE] benchmark started ($(date -u +%Y-%m-%dT%H:%M:%SZ)) using method=$METHOD" | tee -a "$LOG_DIR/bench.log"
python "$ROOT_DIR/experiments/spmm_exps/tools/run_sparse_matrix_multi_method.py"   --repo-root "$ROOT_DIR"   --baseline "$BASELINE"   --method "$METHOD"   --results-dir "$RESULTS_DIR"   --logs-dir "$LOG_DIR"
echo "[$BASELINE] benchmark completed" | tee -a "$LOG_DIR/bench.log"
