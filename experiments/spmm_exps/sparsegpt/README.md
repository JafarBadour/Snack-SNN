# sparsegpt

Source: https://github.com/IST-DASLab/sparsegpt

## Status

- Priority: 8
- Type: mask
- Current status: planned

## Setup

```bash
python experiments/spmm_exps/run_baseline.py --baseline sparsegpt --phase setup --dry-run
python experiments/spmm_exps/run_baseline.py --baseline sparsegpt --phase setup
```

Edit `baseline.json` and `scripts/setup.sh` to pin commit SHA, dependencies, and CUDA compatibility notes.

## Benchmark

```bash
python experiments/spmm_exps/run_baseline.py --baseline sparsegpt --phase bench --dry-run
python experiments/spmm_exps/run_baseline.py --baseline sparsegpt --phase bench
```

Write parsed metrics to `results/` so we can aggregate all baselines into the paper tables/figures.
