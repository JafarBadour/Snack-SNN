# ViT Erdős–Rényi SNACK Benchmark

Rebuttal experiment for the “add a vision transformer” ask: train a small ViT
from scratch on CIFAR-10 while sparsifying MLP (and optionally attention)
`nn.Linear` layers with a shared Erdős–Rényi topology.

## Variants

| CLI name | Meaning |
|---|---|
| `dense` | Full dense baseline |
| `dense_mask` | Dense matmul × ER binary mask |
| `snack_sputnik` | SNACK + Sputnik SpMM |
| `snack_coo` | SNACK + custom COO kernel (optional; often too slow) |

Default sparsity grid: `0.25 0.50 0.75 0.85 0.90 0.95`.

## Install / env

```bash
export PYTHONPATH="$PWD:$PWD/benchmark/sparse_matrix_multi/sputnik_torch_ext:$PYTHONPATH"
export LD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH"

# once, if using snack_sputnik
bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh
```

## Smoke test

```bash
python experiments/vit_er_snack_benchmark/run_vit_er_snack_benchmark.py \
  --synthetic \
  --variants dense dense_mask snack_sputnik \
  --sparsities 0.90 \
  --epochs 1 \
  --max-train-steps-per-epoch 5 \
  --batch-size 32 \
  --disable-output-hash \
  --output-dir experiments/vit_er_snack_benchmark/results/smoke
```

## Longer training with SET / RiGL (recommended)

Sparse models usually need more iterations. This recipe trains **100 epochs** with
dynamic sparse training:

- **SET**: magnitude prune + random regrow (`ZetaPrunerGrower`)
- **RiGL**: magnitude prune + grow by gradient criterion

```bash
bash experiments/vit_er_snack_benchmark/run_dst_long.sh
```

Defaults: ViT-Tiny, CIFAR-10, η ∈ {0.90, 0.95}, Dense + Dense+Mask + SNACK-Sputnik
(SET & RiGL), then SNACK-COO at η=0.95 only.

## Full static-ER sweep (no DST)

```bash
bash experiments/vit_er_snack_benchmark/run_sparsity_sweep.sh
```

Useful overrides:

```bash
# also sparsify attention qkv/proj
SPARSE_ATTN=1 bash experiments/vit_er_snack_benchmark/run_sparsity_sweep.sh

# include SNACK-COO (slow)
INCLUDE_COO=1 SPARSITIES_CSV=0.90,0.95 bash experiments/vit_er_snack_benchmark/run_sparsity_sweep.sh

# larger model
MODEL=vit_small BATCH_SIZE=64 bash experiments/vit_er_snack_benchmark/run_sparsity_sweep.sh
```

## Outputs

Under `--output-dir` (hashed `run_<hash>/` by default):

- `cfg.json` — full args
- `metrics.csv` — one row per (variant, sparsity)
- `summary_by_sparsity.csv` — wide table for pasting into the rebuttal
- `results.json` — same metrics as JSON

Reported metrics: train/eval loss & accuracy, best eval accuracy, peak CUDA memory, wall time.

## Design notes

- Model: ViT-Tiny (192-d, 12 blocks, 3 heads) by default; `vit_small` available.
- Dataset: CIFAR-10, 32×32, patch size 4 → 64 tokens + CLS.
- Topology: `UniformInitializer` (ER / random without replacement). Dense+Mask and SNACK share the same per-layer ER support and nonzero init for a given seed.
- Patch embed (`Conv2d`) and classifier head stay dense.
- SNACK-COO is off by default: ViT training uses large effective GEMM batches (`B × tokens`), which is outside the small-batch regime where COO wins.
