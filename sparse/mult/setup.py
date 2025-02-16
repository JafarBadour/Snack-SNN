from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sparse_tensor_multiply",
    ext_modules=[
        CUDAExtension(
            name="sparse_tensor_multiply",
            sources=["sparse_tensor_multiply.cpp", "sparse_tensor_multiply_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
