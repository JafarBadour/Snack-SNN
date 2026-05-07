# GPT-2 DST Experiment Code Index

This index maps the training/inference experiment code used in this repo so we can iterate quickly on new experiment changes.

## 1) Main Entrypoints

- `experiments/gpt2_dst_lm_benchmark/run_sparsity_sweep.sh`
  - Sweep driver for WT103-style runs.
  - Loops sparsity targets and invokes:
    - `experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py`

- `experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py`
  - Pretrained GPT-2 benchmark (`Dense`, `Dense+Mask`, `SNACK`).
  - Supports SNACK backend switch (`sparse_tensor` vs `sputnik`).
  - Contains training loop + inference benchmark + CSV outputs.

- `experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_from_scratch_benchmark.py`
  - From-scratch GPT-2 benchmark (same variant idea, random init model).

## 2) Training / Inference Hotspots

### In `run_gpt2_dst_snack_benchmark.py`

- `parse_args()`
  - Controls batch sizes, sparsity schedule, DST interval, dataset config, backend, eval cadence.
- `build_lm_dataloaders(...)`
  - Dataset loading/tokenization and train/eval DataLoaders.
- `replace_gpt2_mlp_layers(...)`
  - Swaps GPT-2 MLP layers for Dense+Mask or SNACK layers.
- `train_variant(...)`
  - Core training loop.
  - Handles optimizer step, DST updates, periodic perplexity evaluation.
- `evaluate_perplexity(...)`
  - Evaluation perplexity routine.
- `benchmark_inference_token_latency(...)`
  - Token-level inference latency benchmark (warmup + measured loop).
- `main()`
  - Orchestrates variant runs and writes output CSV files.

### In `run_gpt2_dst_snack_from_scratch_benchmark.py`

- Same function pattern as above (`parse_args`, `build_lm_dataloaders`, `train_variant`, `evaluate_perplexity`, `benchmark_inference_token_latency`, `main`).

## 3) Layer / DST Integration Points

- `DenseMaskConv1D` and `SnackConv1D` classes inside benchmark scripts.
- Imported sparse/DST components:
  - `DST.layers.Snack`
  - `DST.layers.SputnikSnackFunc` (pretrained benchmark path)
  - `DST.pruner_grower.ZetaPrunerGrower`
  - `DST.initializers.uniform_initializer.UniformInitializer`

## 4) Most Important CLI Knobs

- Batch and data:
  - `--train-batch-size`
  - `--eval-batch-size`
  - `--dataset-name`, `--dataset-config`
  - `--dataset-train-split`, `--dataset-eval-split`, `--text-column`

- Sparsity and DST:
  - `--sparsity`
  - `--initial-sparsity`
  - `--sparsity-ramp-steps`
  - `--dst-interval`
  - `--dst-zeta`

- Eval and stopping:
  - `--max-train-steps`
  - `--max-eval-batches`
  - `--perplexity-eval-interval`
  - `--perplexity-eval-batches`
  - `--target-perplexity`

- Backend / variants:
  - `--snack-backend` (`sparse_tensor` or `sputnik`)
  - `--variants` (`dense`, `dense_mask`, `snack`)

## 5) Output Artifacts Per Run

Written in each run output directory:

- `cfg.json`
- `training_step_metrics.csv`
- `inference_metrics.csv`
- `table_a_training_efficiency.csv`
- `table_b_inference_batch1.csv`

## 6) Paper-Table Aggregation Path

- `benchmark/generate_paper_tables.py`
  - Aggregates run CSVs into paper-ready CSV and LaTeX tables.
  - Writes outputs under:
    - `benchmark/generated_tables/`

## 7) Typical Edit Targets (for upcoming experiments)

- Add/modify measured metrics:
  - `train_variant(...)`, `benchmark_inference_token_latency(...)`
- Change evaluation cadence:
  - `train_variant(...)` + argparse defaults
- Add new variant/backend:
  - layer swap path in `replace_gpt2_mlp_layers(...)`
  - variant loop in `main()`
- Change table generation logic:
  - `benchmark/generate_paper_tables.py`

