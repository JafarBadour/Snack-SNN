#!/usr/bin/env bash
set -euo pipefail

# Sweep target sparsity values for GPT-2 DST benchmark.
# Uses the same core settings as your current long-run command.
#
# Usage:
#   bash experiments/gpt2_dst_lm_benchmark/run_sparsity_sweep.sh
#
# Optional overrides:
#   MODEL_NAME=gpt2 DATASET_CONFIG=wikitext-103-raw-v1 bash ...
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Default to local GPT-2 assets for offline servers.
MODEL_NAME="${MODEL_NAME:-$HOME/Arts/gpt2_local_fresh}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-103-raw-v1}"
DATASET_TRAIN_SPLIT="${DATASET_TRAIN_SPLIT:-train}"
DATASET_EVAL_SPLIT="${DATASET_EVAL_SPLIT:-validation}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"

SPARSITY_RAMP_STEPS="${SPARSITY_RAMP_STEPS:-2000}"
DST_INTERVAL="${DST_INTERVAL:-100}"
DST_ZETA="${DST_ZETA:-0.01}"
LR="${LR:-3e-5}"
# Keep these explicit so you can tune for throughput vs VRAM.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
PERPLEXITY_EVAL_INTERVAL="${PERPLEXITY_EVAL_INTERVAL:-100}"
PERPLEXITY_EVAL_BATCHES="${PERPLEXITY_EVAL_BATCHES:-20}"
TARGET_PERPLEXITY="${TARGET_PERPLEXITY:-30}"
SNACK_BACKEND="${SNACK_BACKEND:-sputnik}"
INFERENCE_WARMUP="${INFERENCE_WARMUP:-50}"
INFERENCE_STEPS="${INFERENCE_STEPS:-500}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-experiments/gpt2_dst_lm_benchmark/results/gpt2_wt103_sparsity_sweep}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
VARIANTS="${VARIANTS:-snack dense_mask}"

SPARSITIES_CSV="${SPARSITIES_CSV:-0.25,0.5,0.75,0.9,0.95,0.99}"
IFS=',' read -r -a SPARSITIES <<< "${SPARSITIES_CSV}"
INITIAL_SPARSITY_BASE="${INITIAL_SPARSITY_BASE:-0.5}"
read -r -a VARIANT_LIST <<< "${VARIANTS}"

# Handle a common copy layout: gpt2_local_fresh/gpt2_local_fresh.
if [[ -d "${MODEL_NAME}/gpt2_local_fresh" && -f "${MODEL_NAME}/gpt2_local_fresh/config.json" ]]; then
  MODEL_NAME="${MODEL_NAME}/gpt2_local_fresh"
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "ERROR: MODEL_NAME directory not found: ${MODEL_NAME}" >&2
  echo "Set MODEL_NAME to your local GPT-2 folder (example: \$HOME/Arts/gpt2_local_fresh)." >&2
  exit 1
fi
if [[ ! -f "${MODEL_NAME}/config.json" || ! -f "${MODEL_NAME}/tokenizer.json" ]]; then
  echo "ERROR: MODEL_NAME does not look like a full local HF model/tokenizer folder: ${MODEL_NAME}" >&2
  echo "Expected files like config.json and tokenizer.json in that directory." >&2
  exit 1
fi


# printf "Sleeping for 3 hours...\n"
# sleep 3h; 
for SPARSITY in "${SPARSITIES[@]}"; do
  # Keep initial sparsity <= target sparsity to satisfy benchmark checks.
  # For small targets (e.g. 0, 0.25), start directly at that target.
  if awk "BEGIN {exit !($SPARSITY < $INITIAL_SPARSITY_BASE)}"; then
    INITIAL_SPARSITY="$SPARSITY"
  else
    INITIAL_SPARSITY="$INITIAL_SPARSITY_BASE"
  fi

  RUN_TAG="s$(echo "$SPARSITY" | tr '.' 'p')"
  RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_TAG}"

  echo "============================================================"
  echo "[$(date -Iseconds)] Running sparsity=${SPARSITY} initial_sparsity=${INITIAL_SPARSITY}"
  echo "Variants: ${VARIANTS}"
  echo "SNACK backend: ${SNACK_BACKEND}"
  echo "Output dir base: ${RUN_OUTPUT_DIR}"
  echo "============================================================"

  PYTHONUNBUFFERED=1 python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py \
    --model-name "${MODEL_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-config "${DATASET_CONFIG}" \
    --dataset-train-split "${DATASET_TRAIN_SPLIT}" \
    --dataset-eval-split "${DATASET_EVAL_SPLIT}" \
    --text-column "${TEXT_COLUMN}" \
    --initial-sparsity "${INITIAL_SPARSITY}" \
    --sparsity "${SPARSITY}" \
    --sparsity-ramp-steps "${SPARSITY_RAMP_STEPS}" \
    --dst-interval "${DST_INTERVAL}" \
    --dst-zeta "${DST_ZETA}" \
    --lr "${LR}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --block-size "${BLOCK_SIZE}" \
    --max-train-steps "${MAX_TRAIN_STEPS}" \
    --perplexity-eval-interval "${PERPLEXITY_EVAL_INTERVAL}" \
    --perplexity-eval-batches "${PERPLEXITY_EVAL_BATCHES}" \
    --target-perplexity "${TARGET_PERPLEXITY}" \
    --log-interval "${LOG_INTERVAL}" \
    --snack-backend "${SNACK_BACKEND}" \
    --variants "${VARIANT_LIST[@]}" \
    --inference-warmup "${INFERENCE_WARMUP}" \
    --inference-steps "${INFERENCE_STEPS}" \
    --output-dir "${RUN_OUTPUT_DIR}"

  echo "[$(date -Iseconds)] Completed sparsity=${SPARSITY}"
done

echo "Sparsity sweep completed."
