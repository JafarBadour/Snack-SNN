# venom

Source: https://github.com/UDC-GAC/venom

## Status

- Priority: 6
- Type: kernel_nm
- Current status: planned

## Setup

```bash
python experiments/spmm_exps/run_baseline.py --baseline venom --phase setup --dry-run
python experiments/spmm_exps/run_baseline.py --baseline venom --phase setup
```

Edit `baseline.json` and `scripts/setup.sh` to pin commit SHA, dependencies, and CUDA compatibility notes.

## Benchmark

```bash
python experiments/spmm_exps/run_baseline.py --baseline venom --phase bench --dry-run
python experiments/spmm_exps/run_baseline.py --baseline venom --phase bench
```

Write parsed metrics to `results/` so we can aggregate all baselines into the paper tables/figures.
