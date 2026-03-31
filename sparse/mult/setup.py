"""
CUDA extensions (including conv2d_sparse_tensor_multiply_kernel.cu).

PyTorch's CUDAExtension uses nvcc from CUDA_HOME / PATH. To build with CUDA 12.4:

  source /etc/profile.d/modules.sh   # if needed
  module load nvidia/cuda-12.4
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  pip install -e .

Or run: bash sparse/mult/install_cuda124.sh

On a machine **without a GPU** (e.g. login node), set TORCH_CUDA_ARCH_LIST, e.g. 8.9 for Ada (L40),
or the build can fail with IndexError in torch cpp_extension (empty arch list).
"""
import os

import torch

# PyTorch 2.x: if no device is visible and this is unset, _get_cuda_arch_flags() can crash.
if not os.environ.get("TORCH_CUDA_ARCH_LIST") and not torch.cuda.is_available():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sparse_tensor_multiply",
    ext_modules=[
        CUDAExtension(
            name="sparse_tensor_multiply",
            sources=[
                "sparse_tensor_multiply.cpp",
                "sparse_tensor_multiply_kernel.cu",
                "conv2d_sparse_tensor_multiply.cpp",
                "conv2d_sparse_tensor_multiply_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
