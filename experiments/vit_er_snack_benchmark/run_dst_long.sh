#!/usr/bin/env bash
# Longer ViT training with SET / RiGL for SNACK vs Dense.
#
# Sparse nets need more iterations — default is 100 epochs.
# SNACK-COO only at η>=0.90 (slow on ViT batches).
#
# Usage:
#   bash experiments/vit_er_snack_benchmark/run_dst_long.sh
#
# Smoke:
#   EPOCHS=2 MAX_TRAIN_STEPS_PER_EPOCH=3 bash experiments/vit_er_snack_benchmark/run_dst_long.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${ROOT}/benchmark/sparse_matrix_multi/sputnik_torch_ext:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
DST_ZETA="${DST_ZETA:-0.05}"
DST_INTERVAL="${DST_INTERVAL:-1}"
DST_END_FRAC="${DST_END_FRAC:-0.8}"
MAX_TRAIN_STEPS_PER_EPOCH="${MAX_TRAIN_STEPS_PER_EPOCH:-0}"
OUT="${OUT:-experiments/vit_er_snack_benchmark/results/vit_tiny_cifar10_dst100}"

EXTRA=()
if [[ "${MAX_TRAIN_STEPS_PER_EPOCH}" != "0" ]]; then
  EXTRA+=(--max-train-steps-per-epoch "${MAX_TRAIN_STEPS_PER_EPOCH}")
fi

mkdir -p "$OUT"

echo "[$(date -Iseconds)] Phase A: Dense + Dense+Mask/SNACK-Sputnik with SET+RiGL at η=0.90,0.95"
python experiments/vit_er_snack_benchmark/run_vit_er_snack_benchmark.py \
  --model vit_tiny \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --num-workers 2 \
  --sparsities 0.90 0.95 \
  --variants dense dense_mask snack_sputnik \
  --dst-algos set rigl \
  --dst-interval-epochs "${DST_INTERVAL}" \
  --dst-zeta "${DST_ZETA}" \
  --dst-end-frac "${DST_END_FRAC}" \
  --disable-output-hash \
  --output-dir "${OUT}/phase_sputnik" \
  "${EXTRA[@]}" \
  2>&1 | tee "${OUT}/phase_sputnik.log"

echo "[$(date -Iseconds)] Phase B: SNACK-COO SET+RiGL at η=0.95 only"
python experiments/vit_er_snack_benchmark/run_vit_er_snack_benchmark.py \
  --model vit_tiny \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  --num-workers 2 \
  --sparsities 0.95 \
  --variants snack_coo \
  --dst-algos set rigl \
  --dst-interval-epochs "${DST_INTERVAL}" \
  --dst-zeta "${DST_ZETA}" \
  --dst-end-frac "${DST_END_FRAC}" \
  --disable-output-hash \
  --output-dir "${OUT}/phase_coo" \
  "${EXTRA[@]}" \
  2>&1 | tee "${OUT}/phase_coo.log"

python - <<PY
import csv, json
from pathlib import Path
out = Path("${OUT}")
rows = []
for p in [out/"phase_sputnik"/"metrics.csv", out/"phase_coo"/"metrics.csv"]:
    if p.is_file():
        with p.open() as f:
            rows.extend(csv.DictReader(f))
if not rows:
    raise SystemExit("no metrics to merge")
fields = list(rows[0].keys())
with (out/"metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
with (out/"results.json").open("w") as f:
    json.dump(rows, f, indent=2)
print(f"{'variant':<16} {'dst':<6} {'η':>6} {'best_acc':>10} {'best_loss':>10}")
for r in rows:
    print(f"{r['variant']:<16} {r['dst_algo']:<6} {float(r['sparsity']):6.2f} "
          f"{float(r['best_eval_acc']):10.4f} {float(r['best_eval_loss']):10.4f}")
print("DONE", out/"metrics.csv")
PY
