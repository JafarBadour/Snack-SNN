# SpMM Experiment Hub

This directory standardizes baseline setup and benchmarking for sparse deep learning kernels and mask generators.

## Baselines

| Baseline | Type | Priority | Folder |
|---|---|---:|---|
| Sputnik | kernel | 1 | `sputnik` |
| Flash-LLM | kernel | 2 | `flash-llm` |
| FlashSparse | kernel | 3 | `flashsparse` |
| SpInfer | kernel | 4 | `spinfer` |
| SMaT | kernel | 5 | `smat` |
| VENOM | N:M kernel | 6 | `venom` |
| Wanda | mask generator | 7 | `wanda` |
| SparseGPT | mask generator | 8 | `sparsegpt` |
| RigL | mask generator | 9 | `rigl` |
| SparTA | kernel | 10 | `sparta` |

## Standard layout per baseline

- `baseline.json`: source URL, setup commands, benchmark commands, and tracking metadata.
- `scripts/setup.sh`: deterministic environment/setup entrypoint.
- `scripts/run_microbench.sh`: baseline-specific microbenchmark entrypoint.
- `logs/`: setup and benchmark logs.
- `results/`: parsed benchmark artifacts (CSV/JSON).
- `checkpoints/`: optional downloaded/prepared model artifacts.
- `third_party/`: optional checkout location for upstream source.

## Generic runner

Run from repo root:

```bash
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase setup
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase bench
```

Use `--dry-run` first to verify commands. The runner saves execution metadata to each baseline's `logs/` directory.

## Suite runner

Run all baselines in priority order:

```bash
python experiments/spmm_exps/run_suite.py --phase setup --dry-run
python experiments/spmm_exps/run_suite.py --phase setup --keep-going
```

Run only a selected subset:

```bash
python experiments/spmm_exps/run_suite.py --phase setup --baselines sputnik flash-llm flashsparse
```

Each baseline `scripts/setup.sh` now performs clone-or-update into `third_party/<baseline>/` and logs actions to `logs/setup.log`.

## Benchmark wiring status

- `sputnik`: fully wired to `benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py test_sputnik`
- `flashsparse`: fully wired to `benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py test_flashsparse`
- `flash-llm`, `spinfer`, `smat`, `venom`, `sparta`, `wanda`, `sparsegpt`, `rigl`: benchmark launch is now centralized in `benchmark/sparse_matrix_multi/run_external_baseline.py`

The `run_microbench.sh` scripts in each baseline folder call that centralized launcher, so benchmark execution code is visible under `benchmark/`.

## Typical workflow

```bash
# 1) Build everything (clone + deps + baseline-specific build steps)
python experiments/spmm_exps/run_suite.py --phase setup --keep-going

# 2) Run benchmarks for all baselines
python experiments/spmm_exps/run_suite.py --phase bench --keep-going

# 3) For harness-integrated baselines only
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase bench
python experiments/spmm_exps/run_baseline.py --baseline flashsparse --phase bench
```

Model-dependent benchmark examples:

```bash
WANDA_MODEL=meta-llama/Llama-2-7b-hf \
  python experiments/spmm_exps/run_baseline.py --baseline wanda --phase bench

SPARSEGPT_MODEL=facebook/opt-125m \
  python experiments/spmm_exps/run_baseline.py --baseline sparsegpt --phase bench

VENOM_BENCH_SCRIPT=benchmark/run_spmm_spatha.sh \
  python experiments/spmm_exps/run_baseline.py --baseline venom --phase bench
```
