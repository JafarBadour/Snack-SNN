#!/usr/bin/env bash
set -euo pipefail

BASELINE="venom"
# Repo root: override with SNACK_ROOT, otherwise derived from this script's location.
ROOT_DIR="${SNACK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
BASELINE_DIR="$ROOT_DIR/experiments/spmm_exps/$BASELINE"
LOG_DIR="$BASELINE_DIR/logs"
RESULTS_DIR="$BASELINE_DIR/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "[$BASELINE] benchmark started ($(date -u +%Y-%m-%dT%H:%M:%SZ)) via benchmark/sparse_matrix_multi runner" | tee -a "$LOG_DIR/bench.log"
python "$ROOT_DIR/benchmark/sparse_matrix_multi/run_external_baseline.py"   --baseline "$BASELINE"   --repo-root "$ROOT_DIR"   --logs-dir "$LOG_DIR"   --results-dir "$RESULTS_DIR"
echo "[$BASELINE] benchmark completed" | tee -a "$LOG_DIR/bench.log"
