#!/usr/bin/env bash
# ViT Erdős–Rényi sparsity sweep for NeurIPS rebuttal.
#
# Default: ViT-Tiny / CIFAR-10 / Dense + Dense+Mask + SNACK-Sputnik
# SNACK-COO omitted (too slow for ViT batch sizes); enable with INCLUDE_COO=1.
#
# Usage:
#   bash experiments/vit_er_snack_benchmark/run_sparsity_sweep.sh
#
# Smoke test (few steps/epoch):
#   EPOCHS=1 MAX_TRAIN_STEPS_PER_EPOCH=5 bash experiments/vit_er_snack_benchmark/run_sparsity_sweep.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT_DIR}/benchmark/sparse_matrix_multi/sputnik_torch_ext:${PYTHONPATH}"
export LD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-vit_tiny}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
SPARSE_ATTN="${SPARSE_ATTN:-0}"
INCLUDE_COO="${INCLUDE_COO:-0}"
MAX_TRAIN_STEPS_PER_EPOCH="${MAX_TRAIN_STEPS_PER_EPOCH:-0}"
SPARSITIES_CSV="${SPARSITIES_CSV:-0.25,0.50,0.75,0.85,0.90,0.95}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-experiments/vit_er_snack_benchmark/results/${MODEL}_cifar10_sweep}"

IFS=',' read -r -a SPARSITIES <<< "${SPARSITIES_CSV}"

VARIANTS=(dense dense_mask snack_sputnik)
if [[ "${INCLUDE_COO}" == "1" ]]; then
  VARIANTS+=(snack_coo)
fi

EXTRA_ARGS=()
if [[ "${SPARSE_ATTN}" == "1" ]]; then
  EXTRA_ARGS+=(--sparse-attn)
fi
if [[ "${MAX_TRAIN_STEPS_PER_EPOCH}" != "0" ]]; then
  EXTRA_ARGS+=(--max-train-steps-per-epoch "${MAX_TRAIN_STEPS_PER_EPOCH}")
fi

echo "============================================================"
echo "[$(date -Iseconds)] ViT ER sweep"
echo "  model=${MODEL} epochs=${EPOCHS} batch=${BATCH_SIZE}"
echo "  sparsities=${SPARSITIES_CSV}"
echo "  variants=${VARIANTS[*]}"
echo "  sparse_attn=${SPARSE_ATTN} include_coo=${INCLUDE_COO}"
echo "============================================================"

python experiments/vit_er_snack_benchmark/run_vit_er_snack_benchmark.py \
  --model "${MODEL}" \
  --sparsities "${SPARSITIES[@]}" \
  --variants "${VARIANTS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --output-dir "${BASE_OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
