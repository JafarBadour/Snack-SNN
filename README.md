# SNACK: Sparse Network Active CUDA Kernel

### Beyond Masked Sparsity: SNACK Enables Truly Sparse Neural Networks on GPU

**Accepted at NeurIPS 2026** &nbsp;·&nbsp; Paris

[![Paper](https://img.shields.io/badge/arXiv-2610.04093-b31b1b.svg)](https://arxiv.org/abs/2610.04093)
[![Venue](https://img.shields.io/badge/NeurIPS-2026-68217a.svg)](https://neurips.cc/)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776ab.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.4%20%7C%2012.6-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

[Jafar Badour](https://jafarbadour.com), Maurice van Keulen, Elena Mocanu
*University of Twente, The Netherlands*

---

Deep neural networks keep growing, and Dynamic Sparse Training (DST) promises to cut the
cost — but most implementations are **binary masks over dense tensors**, which recover
almost none of the theoretical compute, memory, or energy savings. The dense matmul still
runs, and the dense weight matrix is still stored.

**SNACK is a truly sparse GPU layer**: it stores and computes only non-zero connections.
It exposes a PyTorch API for restructuring connections and backpropagating gradients
entirely in the sparse paradigm, and ships **SNACK-COO**, a custom COO-format SpMM CUDA
kernel with a batch-to-SM mapping tuned for the small-batch, high-sparsity regime typical
of large-model training and single-stream inference.

### Headline results

| | Result |
|---|---|
| **Kernel** | Up to **7× faster** than masked-dense; competitive with cuSPARSE, Sputnik and FlashSparse at 95% sparsity |
| **Single layer @ 90% sparsity** | **8×** / **3.7×** faster training and **4×** / **2×** faster inference vs. Dense+Mask / dense; **72% less memory** than dense |
| **GPT-2 small DST** | Same perplexity as Dense+Mask with up to **40% less training memory**; inference at **2.2× less memory** (1256 → 579 MB) and **22% less energy/token** |
| **GAMLP, `ogbn-products`, batch 1, η=0.99** | Latency 1.01 → **0.21 ms** (4.8×), memory 310 → **111 MB**, energy 51 → **10 mJ** |
| **Correctness** | Gradients match Dense+Mask to 1e-4; RigL(SNACK) matches RigL on ImageNet |

Drop-in for `nn.Linear`. Backend-agnostic — swap SNACK-COO, Sputnik, cuSPARSE or
FlashSparse per regime while the gradient and initializer stay sparse.

### Citation

```bibtex
@inproceedings{badour2026snack,
  title     = {Beyond Masked Sparsity: {SNACK} Enables Truly Sparse Neural Networks on {GPU}},
  author    = {Badour, Jafar and van Keulen, Maurice and Mocanu, Elena},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2610.04093},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

---

# Structure


```
├── benchmark
│   ├── gpu
│   │   └── profiler.py
│   ├── memory_leak
│   │   └── without_replacement.py
│   ├── plots
│   │   └── matrix_matrix.py
│   ├── pruner_grower
│   │   └── test_zeta_pruner_grower.py
│   └── sparse_matrix_multi
│       ├── __test_methods.py
│       ├── test_outer_join.py
│       ├── test.py
│       └── test_speed_sparse_tensor_vs_dense_tensor.py
├── DST
│   ├── initializers
│   │   ├── erdos.py
│   │   ├── fixed_degree.py
│   │   ├── grand.py
│   │   ├── __init__.py
│   │   └── uniform_initializer.py
│   ├── layers
│   │   ├── Dense.py
│   │   ├── __init__.py
│   │   └── Snack.py
│   └── pruner_grower
│       ├── __init__.py
│       └── zeta.py
├── sparse
│   └── mult
│       ├── csr
│       │   └── warp_csr.py
│       ├── setup.py
│       └── tensor
│           ├── __cls__.py
│           ├── exceptions.py
│           ├── __init__.py
│           └── utils.py
└── train
    └── DST
        ├── mnist.py
        └── train.py
```


There are mainly three experiment tracks in the repository
```
train/DST/train.py # results explained in figure 4 
train/DST/mnist.py # results not reported in the paper but it basically shows how to use SNACK in a real world training problem

benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py # this reports Figure 3


```
to plot the results please find the necessary notebooks at
```
├── DST
│   └── plots and stats.ipynb # figure 4, [1-6] in Appendix
├── notebooks
│   ├── stats.ipynb  # Figure 3
│   └── testy.ipynb
└── train
    └── DST
        └── minst.ipynb # not reported in paper
```

# Reproducing the benchmarks

All benchmarks expect the repo root to be on `PYTHONPATH` and the `sparse/mult`
CUDA extension to be installed (see [Installation](#installation)). After
activating your environment from the repo root:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH"
```

## 1) End-to-end SpMM + MLP sweep (Figure 3, Table `tab:sparse_performance`, `tab:performance_comparison_snack`)

The SpMM benchmark sweeps every backend (Sputnik, FlashSparse, cuSPARSE
COO/CSR, SparseTorch, SparseCuPy, JAX-BSR, ge-spmm/dgsparse, Sputnik-CSR-DL,
and our SNACK / SparseUT kernel) over a grid of `(batch_size, dense_level,
sparsity_level)`. The single-shot driver below activates the venv, installs
the runtime/benchmark requirements, builds `sparse/mult`, builds
`sputnik_torch_ext`, builds FlashSparse, and then launches every method,
continuing past method-level failures:

```bash
bash benchmark/sparse_matrix_multi/run_everything_spmm_mlp.sh
```

Useful overrides:

```bash
# Pick a subset of methods (any value from method_dict in
# test_speed_sparse_tensor_vs_dense_tensor.py):
METHODS="test_sputnik test_dense test_sparse_ut" \
  bash benchmark/sparse_matrix_multi/run_everything_spmm_mlp.sh

# Tune the MLP backend sweep (check_snack_backends.py):
SPARSITY=0.95 WARMUP=10 ITERS=30 \
  bash benchmark/sparse_matrix_multi/run_everything_spmm_mlp.sh
```

To run a single SpMM backend by hand (skipping the dependency / build steps
above), use the underlying entry point directly:

```bash
python -m benchmark.sparse_matrix_multi.test_speed_sparse_tensor_vs_dense_tensor test_sputnik
```

Each backend writes a CSV to
`benchmark/apr-13-log_mult_incl_cupy-<method>.csv`. The notebook
`notebooks/stats.ipynb` consumes those CSVs and produces Figure 3
(`SpMM vs DenseMM-3panel-b1-b2-b4.pdf`) plus the appendix variants. The MLP
backend correctness/speed sweep (`check_snack_backends.py`) writes its
own CSVs/JSON next to itself and feeds the SNACK-vs-Dense+Mask tables.

### Slurm submission

```bash
# Defaults: 1× Lovelace GPU, 16 CPUs, 6 GB RAM, 2 h walltime.
sbatch benchmark/sparse_matrix_multi/sbatch_gpu_benchmarks.sh

# Or just the tst5 SpMM correctness check:
sbatch benchmark/sparse_matrix_multi/sbatch_tst5_spmm.sh
```

## 2) External SpMM baselines (Sputnik / FlashSparse / Flash-LLM / SpInfer / SMaT / VENOM / SparTA / Wanda / SparseGPT / RigL)

These baselines live under `experiments/spmm_exps/` and are orchestrated by a
generic runner with a uniform `--phase setup | bench` interface:

```bash
# 1) Build everything (clones into experiments/spmm_exps/<baseline>/third_party
#    and runs the per-baseline build script):
python experiments/spmm_exps/run_suite.py --phase setup --keep-going

# 2) Run all baselines in priority order:
python experiments/spmm_exps/run_suite.py --phase bench --keep-going

# Single baseline (use --dry-run first to inspect the commands):
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase setup
python experiments/spmm_exps/run_baseline.py --baseline sputnik --phase bench
```

Mask-generator baselines need a model checkpoint:

```bash
WANDA_MODEL=meta-llama/Llama-2-7b-hf \
  python experiments/spmm_exps/run_baseline.py --baseline wanda --phase bench

SPARSEGPT_MODEL=facebook/opt-125m \
  python experiments/spmm_exps/run_baseline.py --baseline sparsegpt --phase bench
```

Sputnik and FlashSparse are wired directly into the in-tree harness above;
the rest go through `benchmark/sparse_matrix_multi/run_external_baseline.py`.
See `experiments/spmm_exps/README.md` for the full table.

## 3) GPT-2 + DST + SNACK (Tables `tab:gpt2_training_compact`, `tab:gpt2_inference_compact`)

Benchmarks GPT-2 MLP layers under `Dense`, `Dense+Mask` (DST rewiring), and
`SNACK` (sparse kernel + DST). Records per-step training loss, CUDA time,
power, energy, memory, and inference latency / energy / perplexity at batch
size 1.

```bash
pip install transformers datasets pynvml

python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py \
  --model-name gpt2-large \
  --dataset-name wikitext --dataset-config wikitext-2-raw-v1 \
  --sparsity 0.90 --initial-sparsity 0.50 --sparsity-ramp-steps 500 \
  --dst-interval 100 --dst-zeta 0.05 \
  --lr 5e-5 --train-batch-size 1 --eval-batch-size 1 --block-size 128 \
  --max-train-steps 500 --max-eval-batches 100 \
  --perplexity-eval-interval 50 --perplexity-eval-batches 10 \
  --target-perplexity 30 \
  --snack-backend sparse_tensor \
  --variants dense dense_mask snack \
  --inference-warmup 50 --inference-steps 500 \
  --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_large
```

To use the Sputnik backend (`--snack-backend sputnik`) you must build the
extension first:

```bash
bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh
export PYTHONPATH="$PWD/benchmark/sparse_matrix_multi/sputnik_torch_ext:$PYTHONPATH"
```

Model-size sweep used for the crossover curve in the appendix:

```bash
for size in gpt2 gpt2-medium gpt2-large gpt2-xl; do
  python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_benchmark.py \
    --model-name "$size" \
    --output-dir "experiments/gpt2_dst_lm_benchmark/results/${size//-/_}"
done
```

A from-scratch (random-init) variant lives next to it:

```bash
python experiments/gpt2_dst_lm_benchmark/run_gpt2_dst_snack_from_scratch_benchmark.py \
  --model-size gpt2 --tokenizer-name gpt2 \
  --sparsity 0.90 --dst-interval 100 --dst-zeta 0.05 \
  --output-dir experiments/gpt2_dst_lm_benchmark/results/gpt2_from_scratch
```

Each run writes `training_step_metrics.csv`, `inference_metrics.csv`,
`table_a_training_efficiency.csv`, and `table_b_inference_batch1.csv`
under `--output-dir/run_<hash>/` (use `--disable-output-hash` to write
directly into `--output-dir`). See `experiments/gpt2_dst_lm_benchmark/README.md`
for the long-run WT103 recipe and append-mode usage.

## 4) GAMLP + SNACK (Table `tab:gamlp_training_compact`)

GAMLP is a git submodule; pull it before running:

```bash
git submodule update --init --recursive
pip install ogb torch-geometric dgl pynvml numpy
```

Train GAMLP (e.g. on `ogbn-products`):

```bash
python third_party/GAMLP/main.py \
  --dataset ogbn-products --method R_GAMLP \
  --stages 300 --train-num-epochs 0 \
  --hidden 1024 --n-layers-1 4 --n-layers-2 4 --num-hops 5 \
  --batch-size 50000 --pre-process --residual --bns
```

Benchmark inference at batch size 1 across `dense / dense_mask / snack`:

```bash
python experiments/gamlp_snack_benchmark/benchmark_gamlp_snack.py \
  --checkpoint-path third_party/GAMLP/output/ogbn-products/<your_stage0>.pkl \
  --dataset ogbn-products --method R_GAMLP \
  --variants dense dense_mask snack \
  --hidden 1024 --num-hops 5 --n-layers-1 4 --n-layers-2 4 \
  --batch-size 1 --num-samples 1000 --warmup 100 \
  --sparsities 0.70 0.80 0.90 0.95 0.99
```

Train all three variants (with optional DST prune/grow on `snack` /
`dense_mask`) and log per-epoch speed/energy/memory metrics:

```bash
python experiments/gamlp_snack_benchmark/train_gamlp_snack_dst.py \
  --dataset ogbn-products --method R_GAMLP_RLU --use-rlu \
  --root third_party/GAMLP/data \
  --hidden 512 --num-hops 5 --n-layers-1 2 --n-layers-2 2 --n-layers-3 2 \
  --batch-size 4096 --epochs 40 \
  --sparsity 0.90 --dst-zeta 0.05 --dst-every 5 --dst-until-epoch 25 \
  --output-checkpoint experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_checkpoint.pt \
  --metrics-csv      experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_training_metrics.csv
```

Replace `--variant snack` with `dense` or `dense_mask` for the corresponding
baselines. Full options (RLU checkpoints, artifact-based mask→snack
inference, etc.) are documented in `experiments/gamlp_snack_benchmark/README.md`.

## 5) DST training curves (Figure 4, Appendix Figs 1–6)

```bash
python -m train.DST.train
```

Then open `DST/plots and stats.ipynb` to regenerate the figures from the
logged outputs.

## 6) Generating the paper-ready tables

Once the SpMM CSVs and the GPT-2 / GAMLP result directories are populated,
build all CSV/LaTeX tables in one step:

```bash
python benchmark/generate_paper_tables.py \
  --gpt2-results-dir experiments/gpt2_dst_lm_benchmark/results \
  --spmm-glob "benchmark/apr-13-log_mult_incl_cupy-test_*.csv" \
  --output-dir benchmark/generated_tables
```

This writes one `.csv` and one `.tex` per table plus a combined
`all_tables.tex` that mirrors the tables included in the paper.


# Installation

## Prerequisites

- CUDA 12.1 or 12.6
- Python 3.12+
- nvcc compiler

## Setup

1. Create a conda environment:
```bash
conda create -p ./venv python=3.12.7
conda activate ./venv
```

2. Install CUDA toolkit (if not already installed):
```bash
sudo apt install nvidia-cuda-toolkit
# Or download from: https://developer.nvidia.com/cuda-downloads
```

3. Install PyTorch with CUDA support:
```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. Install core dependencies:
```bash
pip install -r requirements.txt
```

5. (Optional) Install benchmark dependencies for comparison tests:
```bash
pip install -r requirements-benchmark.txt
```

Note: Some benchmark dependencies (jax, cupy, spconv) may require additional setup. See `requirements-benchmark.txt` for details.


```
# On Windows, set TORCH_INCLUDE to your PyTorch installation path
# Example (modify for your username):
# set TORCH_INCLUDE=C:\Users\USERNAME\path\to\venv\Lib\site-packages\torch\include\torch
```


To compile the CUDA extensions (`sparse_tensor_multiply_kernel.cu`, `conv2d_sparse_tensor_multiply_kernel.cu`) with **CUDA 12.4**, load your toolkit and install from `sparse/mult` (see `sparse/mult/setup.py` for details):

```bash
source /etc/profile.d/modules.sh   # if `module` is not defined
module load nvidia/cuda-12.4
bash sparse/mult/install_cuda124.sh
```

Or manually (after `pip install -r requirements.txt` so **torch** is present; `setup.py` imports torch, so use **`--no-build-isolation`**):

```bash
module load nvidia/cuda-12.4   # optional
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
# On a login node with no GPU, set an arch (e.g. Ada/L40: 8.9) or use setup.py default:
# export TORCH_CUDA_ARCH_LIST=8.9
pip install -e sparse/mult --no-build-isolation
```

Set the PyTorch library path (Linux) — adjust for your Python installation:

```bash
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

If the dataset lives on a remote server, forward a notebook port over SSH:

```bash
ssh -L 8888:localhost:8888 -p 2222 username@HOST
```


## Acknowledgements

This work was supported by the Modular Integrated Sustainable Datacenter (MISD) project,
funded by the Dutch Ministry of Economic Affairs and Climate under the European IPCEI-CIS
programme. Jafar Badour is fully funded by the MISD project; Elena Mocanu is partially
funded by the MISD project.
