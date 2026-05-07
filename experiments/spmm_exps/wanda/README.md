# wanda

Source: https://github.com/locuslab/wanda

## Status

- Priority: 7
- Type: mask
- Current status: planned

## Setup

```bash
python experiments/spmm_exps/run_baseline.py --baseline wanda --phase setup --dry-run
python experiments/spmm_exps/run_baseline.py --baseline wanda --phase setup
```

Edit `baseline.json` and `scripts/setup.sh` to pin commit SHA, dependencies, and CUDA compatibility notes.

## Benchmark

```bash
python experiments/spmm_exps/run_baseline.py --baseline wanda --phase bench --dry-run
python experiments/spmm_exps/run_baseline.py --baseline wanda --phase bench
```

Write parsed metrics to `results/` so we can aggregate all baselines into the paper tables/figures.
