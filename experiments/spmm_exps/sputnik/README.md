# sputnik

Source: https://github.com/google-research/sputnik

## Status

- Priority: 1
- Type: kernel
- Current status: planned

## Setup

```bash
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase setup --dry-run
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase setup
```

Edit `baseline.json` and `scripts/setup.sh` to pin commit SHA, dependencies, and CUDA compatibility notes.

## Benchmark

```bash
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase bench --dry-run
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase bench
```

Write parsed metrics to `results/` so we can aggregate all baselines into the paper tables/figures.
