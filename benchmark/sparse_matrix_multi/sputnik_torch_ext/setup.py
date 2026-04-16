from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parents[3]
SPUTNIK_INSTALL = ROOT / "third_party" / "sputnik" / "install"
INCLUDE_DIR = SPUTNIK_INSTALL / "include"
LIB_DIR = SPUTNIK_INSTALL / "lib"
TORCH_LIB_DIR = Path(torch.__file__).resolve().parent / "lib"

if not (INCLUDE_DIR / "sputnik" / "sputnik.h").exists():
    raise RuntimeError(
        f"Missing Sputnik headers at {INCLUDE_DIR}. "
        "Build/install Sputnik first: bash benchmark/sparse_matrix_multi/install_sputnik.sh"
    )
if not (LIB_DIR / "libsputnik.so").exists():
    raise RuntimeError(
        f"Missing libsputnik.so at {LIB_DIR}. "
        "Build/install Sputnik first: bash benchmark/sparse_matrix_multi/install_sputnik.sh"
    )

setup(
    name="sputnik_torch_ext",
    ext_modules=[
        CUDAExtension(
            name="sputnik_torch_ext",
            sources=["sputnik_torch_ext.cpp"],
            include_dirs=[str(INCLUDE_DIR)],
            library_dirs=[str(LIB_DIR)],
            libraries=["sputnik"],
            extra_compile_args=["-O3"],
            extra_link_args=[
                f"-Wl,-rpath,{LIB_DIR}",
                f"-Wl,-rpath,{TORCH_LIB_DIR}",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
