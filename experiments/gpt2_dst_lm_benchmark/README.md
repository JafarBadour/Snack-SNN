# GPT-2 DST SNACK Benchmark

This experiment benchmarks GPT-2 MLP layers under three variants:

- `Dense` (original)
- `Dense+Mask` (dense compute with sparsity mask + DST rewiring)
- `SNACK` (sparse kernel via SNACK layer + DST rewiring)

It fine-tunes on WikiText-2 and records:

- Training: loss, CUDA time/step, power, energy, memory
- Inference (batch size 1 token forward): latency, energy, memory, perplexity

## Install

```bash
pip install transformers datasets torch pynvml
```

## Run

```bash
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py \
  --model-name gpt2-large \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --sparsity 0.90 \
  --initial-sparsity 0.50 \
  --sparsity-ramp-steps 500 \
  --dst-interval 100 \
  --dst-zeta 0.05 \
  --lr 5e-5 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --block-size 128 \
  --max-train-steps 500 \
  --max-eval-batches 100 \
  --perplexity-eval-interval 50 \
  --perplexity-eval-batches 10 \
  --target-perplexity 30 \
  --snack-backend sparse_tensor \
  --variants dense dense_mask snack \
  --inference-warmup 50 \
  --inference-steps 500 \
  --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_large
```

`--initial-sparsity` lets you start denser and gradually increase toward `--sparsity` during DST updates.
`--perplexity-eval-interval` prints evaluation perplexity every N training steps using `--perplexity-eval-batches`.
`--log-interval` prints heartbeat progress logs (elapsed + ETA) every N steps.
`--snack-backend` selects SNACK forward backend (`sparse_tensor` default, `sputnik` optional).
Use `--target-perplexity` to stop training early once periodic eval perplexity reaches your goal.
Each run is written to a hashed subdirectory under `--output-dir` (e.g. `run_<hash>`), and a `cfg.json` with all args is saved there.
Use `--disable-output-hash` if you want to write directly into `--output-dir` without the hash suffix.

For a larger corpus, switch to:

```bash
--dataset-name wikitext --dataset-config wikitext-103-raw-v1
```

### Long-run recipe (WT103, gradual 50->90, target PPL, Sputnik)

```bash
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py \
  --model-name gpt2 \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --dataset-train-split train \
  --dataset-eval-split validation \
  --text-column text \
  --initial-sparsity 0.50 \
  --sparsity 0.5\
  --sparsity-ramp-steps 2000 \
  --dst-interval 100 \
  --dst-zeta 0.01 \
  --lr 3e-5 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --block-size 64 \
  --max-train-steps 20000 \
  --perplexity-eval-interval 100 \
  --perplexity-eval-batches 20 \
  --target-perplexity 30 \
  --snack-backend sputnik \
  --inference-warmup 50 \
  --inference-steps 500 \
  --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_wt103_50to90_target30
```

To run only SNACK and keep existing Dense/Dense+Mask outputs in the same result directory:

```bash
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py \
  --variants snack \
  --append-results \
  --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_wt103_50to90_target30 \
  # ... keep your other training flags the same ...
```

To use `--snack-backend sputnik`, build the local extension first:

```bash
bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh
export PYTHONPATH="$PWD/benchmark/sparse_matrix_multi/sputnik_torch_ext:$PYTHONPATH"
```

## Model-size sweep (crossover curve)

```bash
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py --model-name gpt2 --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_small
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py --model-name gpt2-medium --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_medium
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py --model-name gpt2-large --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_large
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py --model-name gpt2-xl --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_xl
```

## Run from scratch (no pretrained GPT-2 weights)

This variant initializes GPT-2 randomly (from config) and then benchmarks `Dense`, `Dense+Mask`, and `SNACK` while tracking the same metrics (loss, CUDA time, power, energy, memory, latency, perplexity).

By default it now uses a larger language-model corpus (`wikitext-103-raw-v1`) with configurable dataset name/config/splits.

```bash
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_from_scratch_benchmark.py \
  --model-size gpt2 \
  --tokenizer-name gpt2 \
  --sparsity 0.90 \
  --dst-interval 100 \
  --dst-zeta 0.05 \
  --lr 5e-5 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --block-size 128 \
  --max-train-steps 500 \
  --max-eval-batches 100 \
  --inference-warmup 50 \
  --inference-steps 500 \
  --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_from_scratch
```

You can also use `--model-size custom` with `--custom-n-layer`, `--custom-n-head`, and `--custom-n-embd`.
From-scratch runs also write to hashed subdirectories and save `cfg.json` by default (disable with `--disable-output-hash`).

Key dataset arguments:

- `--dataset-name` (default: `wikitext`)
- `--dataset-config` (default: `wikitext-103-raw-v1`)
- `--dataset-train-split` (default: `train`)
- `--dataset-eval-split` (default: `validation`)
- `--text-column` (default: `text`)
- `--max-train-samples` and `--max-eval-samples` (0 means full split)

## Outputs

The script writes:

- `training_step_metrics.csv`
- `inference_metrics.csv`
- `table_a_training_efficiency.csv`
- `table_b_inference_batch1.csv`

under the selected `--output-dir`.
`training_step_metrics.csv` now includes `eval_perplexity` (filled on configured perplexity eval intervals).

## Notes

- GPT-2 MLP swap targets are `model.transformer.h[i].mlp.c_fc` and `model.transformer.h[i].mlp.c_proj`.
- GPT-2 uses `Conv1D` with weights shaped `[in_features, out_features]`; the experiment preserves this layout when converting to masked or SNACK layers.
- SNACK indices are cast to `uint16` in the experiment layer wrapper (without modifying SNACK core files).

