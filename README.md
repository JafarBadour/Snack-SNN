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


There are mainly three experiemnt tracks in the repository
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

# Set PyTorch library path (Linux)
# Modify the path according to your Python installation
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH