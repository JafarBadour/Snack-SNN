# GAMLP + SNACK Benchmark

This experiment benchmarks GAMLP inference variants at batch size 1:

- `dense`
- `dense_mask` (simulated sparsity on linear weights)
- `snack` (SNACK sparse kernel path)

GAMLP is added as a git submodule at `third_party/GAMLP`.

## 1) Setup submodule and dependencies

```bash
git submodule update --init --recursive
pip install ogb torch-geometric dgl pynvml numpy
```

### Install SNACK in the same environment

SNACK in this repo relies on the CUDA extension under `sparse/mult`.

```bash
# From repository root:
pip install -e sparse/mult --no-build-isolation

# Ensure local packages (DST, sparse) are importable:
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### Known-good compatibility note

If you run into DGL import issues, prefer a clean Python 3.10 environment and install:

```bash
pip install torch==2.1.2
pip install dgl==1.1.3 ogb==1.3.6 torch-geometric torchdata==0.7.1 pynvml numpy
```

## 2) Train GAMLP (inside submodule)

Example for `ogbn-products`:

```bash
python third_party/GAMLP/main.py \
  --dataset ogbn-products \
  --method R_GAMLP \
  --stages 300 \
  --train-num-epochs 0 \
  --hidden 1024 \
  --n-layers-1 4 \
  --n-layers-2 4 \
  --num-hops 5 \
  --batch-size 50000 \
  --pre-process --residual --bns
```

After training, find a stage checkpoint from `third_party/GAMLP/output/<dataset>/..._0.pkl`.

## 3) Run SNACK benchmark (batch size 1)

```bash
python experiments/gamlp_snack_benchmark/benchmark_gamlp_snack.py \
  --checkpoint-path third_party/GAMLP/output/ogbn-products/<your_stage0_checkpoint>.pkl \
  --dataset ogbn-products \
  --method R_GAMLP \
  --variants dense dense_mask snack \
  --hidden 1024 \
  --num-hops 5 \
  --n-layers-1 4 \
  --n-layers-2 4 \
  --batch-size 1 \
  --num-samples 1000 \
  --warmup 100 \
  --sparsities 0.70 0.80 0.90 0.95 0.99 \
  --root /data4/zwt/
```

Run without SNACK (dense vs masked only):

```bash
python experiments/gamlp_snack_benchmark/benchmark_gamlp_snack.py \
  --checkpoint-path third_party/GAMLP/output/ogbn-products/<your_stage0_checkpoint>.pkl \
  --dataset ogbn-products \
  --method R_GAMLP \
  --variants dense dense_mask \
  --batch-size 1 \
  --num-samples 1000
```

For RLU checkpoints, use `--use-rlu` and method `R_GAMLP_RLU` or `JK_GAMLP_RLU`.

## 4) Train with DST prune/grow (SNACK)

This repo now includes a profiled trainer for all three training variants:

- `dense` (original GAMLP layers)
- `dense_mask` (dense trainable weights with fixed sparsity mask)
- `snack` (SNACK sparse layers, optional DST prune/grow)

Every variant writes per-epoch speed/energy/memory metrics to `--metrics-csv`.

SNACK + DST example:

```bash
python experiments/gamlp_snack_benchmark/train_gamlp_snack_dst.py \
  --dataset ogbn-products \
  --method R_GAMLP_RLU \
  --use-rlu \
  --root third_party/GAMLP/data \
  --hidden 512 \
  --num-hops 5 \
  --n-layers-1 2 \
  --n-layers-2 2 \
  --n-layers-3 2 \
  --batch-size 4096 \
  --epochs 40 \
  --sparsity 0.90 \
  --dst-zeta 0.05 \
  --dst-every 5 \
  --dst-until-epoch 25 \
  --output-checkpoint experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_checkpoint.pt \
  --metrics-csv experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_training_metrics.csv
```

Dense (original) with metrics:

```bash
python experiments/gamlp_snack_benchmark/train_gamlp_snack_dst.py \
  --variant dense \
  --dataset ogbn-products \
  --method R_GAMLP_RLU \
  --use-rlu \
  --root third_party/GAMLP/data \
  --output-checkpoint experiments/gamlp_snack_benchmark/results/gamlp_dense_checkpoint.pt \
  --metrics-csv experiments/gamlp_snack_benchmark/results/gamlp_dense_training_metrics.csv
```

Dense + mask sparsity with metrics:

```bash
python experiments/gamlp_snack_benchmark/train_gamlp_snack_dst.py \
  --variant dense_mask \
  --dataset ogbn-products \
  --method R_GAMLP_RLU \
  --use-rlu \
  --root third_party/GAMLP/data \
  --sparsity 0.90 \
  --dst-zeta 0.05 \
  --dst-every 5 \
  --dst-until-epoch 25 \
  --output-checkpoint experiments/gamlp_snack_benchmark/results/gamlp_dense_mask_checkpoint.pt \
  --metrics-csv experiments/gamlp_snack_benchmark/results/gamlp_dense_mask_training_metrics.csv
```

Optional: warm-start from a dense GAMLP checkpoint before SNACK conversion:

```bash
python experiments/gamlp_snack_benchmark/train_gamlp_snack_dst.py \
  --dataset ogbn-products \
  --method R_GAMLP_RLU \
  --use-rlu \
  --init-checkpoint third_party/GAMLP/output/ogbn-products/<dense_stage0>.pkl
```

Then benchmark the resulting DST checkpoint:

```bash
python experiments/gamlp_snack_benchmark/benchmark_gamlp_snack.py \
  --checkpoint-path experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_checkpoint.pt \
  --dataset ogbn-products \
  --method R_GAMLP_RLU \
  --use-rlu \
  --variants dense dense_mask snack \
  --batch-size 1 \
  --num-samples 1000 \
  --sparsities 0.70 0.80 0.90 0.95
```

Artifact-based sparse inference comparison (recommended for your request):

```bash
python experiments/gamlp_snack_benchmark/benchmark_gamlp_snack.py \
  --checkpoint-path experiments/gamlp_snack_benchmark/results/gamlp_dense_mask_checkpoint.pt \
  --dataset ogbn-products \
  --method R_GAMLP_RLU \
  --use-rlu \
  --root third_party/GAMLP/data \
  --variants dense_mask snack \
  --batch-size 1 \
  --num-samples 1000 \
  --warmup 100 \
  --skip-first-measured 1 \
  --output-csv experiments/gamlp_snack_benchmark/results/gamlp_artifact_mask_to_snack_inference.csv
```

This path loads the trained `dense_mask` artifact and converts it to SNACK using the same learned mask topology
(no re-pruning), so the sparse pattern is matched across variants.
`--warmup` and `--skip-first-measured` make timing explicitly steady-state (no cold-start in reported latency/energy).

`--metrics-csv` logs per-epoch training efficiency:

- speed: `step_latency_ms_*`, `samples_per_sec`, `epoch_time_s`
- energy: `step_energy_mj_mean`, `epoch_energy_mj_total`, `epoch_energy_mj_per_sample`
- memory: `peak_allocated_mb`, `peak_reserved_mb`
- `dst_applied` marks epochs where prune/grow ran (`snack` and `dense_mask`; not `dense`)

## Output

CSV:

- `experiments/gamlp_snack_benchmark/results/gamlp_snack_benchmark.csv`

Columns:

- sparsity
- type (`dense`, `dense_mask`, `snack`)
- baseline_type (agreement reference variant)
- latency (`mean/p50/p95`)
- energy per sample (mJ)
- max allocated memory (MB)
- accuracy
- agreement vs baseline

## Notes

- This benchmark swaps `nn.Linear`, GAMLP `Dense`, and `GraphConvolution` weight matrices for mask/SNACK variants.
- Accuracy should remain very close across variants since surviving weights are matched by magnitude pruning.
- `ogbn-papers100M` in original GAMLP has hardcoded feature paths; `ogbn-products` is the easiest reproducible path.
- The DST trainer updates SNACK topology with `ZetaPrunerGrower.prune()` + `.regrow(UniformInitializer)` on schedule (`--dst-every`, `--dst-until-epoch`).
