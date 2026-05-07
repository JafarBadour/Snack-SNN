# rigl

Source: https://github.com/google-research/rigl

## Status

- Priority: 9
- Type: mask
- Current status: planned

## Setup

```bash
python experiments/spmm_exps/run_baseline.py --baseline rigl --phase setup --dry-run
python experiments/spmm_exps/run_baseline.py --baseline rigl --phase setup
```

Edit `baseline.json` and `scripts/setup.sh` to pin commit SHA, dependencies, and CUDA compatibility notes.

## Benchmark

```bash
python experiments/spmm_exps/run_baseline.py --baseline rigl --phase bench --dry-run
python experiments/spmm_exps/run_baseline.py --baseline rigl --phase bench
```

Write parsed metrics to `results/` so we can aggregate all baselines into the paper tables/figures.
