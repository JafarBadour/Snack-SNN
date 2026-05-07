import typing
import sys

from tqdm import tqdm
from sparse.mult.tensor.utils import create_random_sparse_matrix
import torch
import pandas as pd
from benchmark.sparse_matrix_multi.__test_methods import (
    test_jax,
    test_dense,
    test_sparse_ut,
    test_sparse_torch,
    test_sparse_cupy,
    test_sparse_cupy_bsr,
    test_jax_bsr,
    test_jax_csr,
    test_sparse_cupy_csr,
    test_sparse_torch_csr,
    test_flashsparse,
    test_sputnik,
    test_cusparse_csr_library,
    test_cusparse_coo_library,
    test_sputnik_csr_dl_optimized,
    test_ge_spmm_dgsparse_csr_gnn_optimized,
)

TESTING_DEVICE = "cuda"  # "cuda"
sparsity_levels = (
    list(range(0, 50, 24))
    + list(range(50, 80, 10))
    + list(range(80, 96, 5))
    + [96, 97, 98, 99, 99.5, 99.9]
)

# dense level is layer A with layer B that are after one another in the model architecture
dense_levels = {
    "500x500": {"Reps": 2},
    "100x100": {"Reps": 2},
    "250x100000": {"Reps": 2},
    "500x500": {"Reps": 2},
    "500x500": {"Reps": 2},
     "1000x1000": {"Reps": 2},
    "5000x5000": {"Reps": 2},
    "7500x7500": {"Reps": 2},
    "8500x8500": {"Reps": 2},
    "10000x10000" : {'Reps' : 2},
    "12500x12500": {"Reps": 2},
    "15000x15000": {"Reps": 2},
    "17500x17500": {"Reps": 2},
}

csv_name = "apr-13-log_mult_incl_cupy"
batches_cnt = [1, 2, 4, 8, 16]

import warnings

warnings.filterwarnings("ignore")



method_dict = dict(
    test_sparse_cupy=test_sparse_cupy, #
    test_jax=test_jax,
    test_sparse_ut=test_sparse_ut,
    test_sparse_torch=test_sparse_torch,
    test_dense=test_dense,
    # test_sparse_cupy_bsr=test_sparse_cupy_bsr, # not implemented error
    test_jax_bsr=test_jax_bsr,
    # test_jax_csr=test_jax_csr, # not implemented error
    test_sparse_cupy_csr=test_sparse_cupy_csr,
    test_sparse_torch_csr=test_sparse_torch_csr,
    test_flashsparse=test_flashsparse,
    test_sputnik=test_sputnik,
    test_cusparse_csr_library=test_cusparse_csr_library,
    test_cusparse_coo_library=test_cusparse_coo_library,
    test_sputnik_csr_dl_optimized=test_sputnik_csr_dl_optimized,
    test_ge_spmm_dgsparse_csr_gnn_optimized=test_ge_spmm_dgsparse_csr_gnn_optimized,
)


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def benchmark(method_name : str):
    log = []
    for BATCH_SIZE in tqdm(batches_cnt, desc="Batches processing"):
        for dense_level in tqdm(dense_levels.keys(), desc="Processing Dense Levels", leave=False):
            layera, layerb = list(map(int, dense_level.split("x")))

            for sparsity_level in tqdm(sparsity_levels, desc="Sparsity Lvls", leave=False):
                caller : typing.Callable = method_dict[method_name]
                try:
                    ones = torch.rand((BATCH_SIZE, layera)).to(TESTING_DEVICE)
                    sparse_matrix = create_random_sparse_matrix(layera, layerb, sparsity_level)
                    sparse_matrix = sparse_matrix.to(TESTING_DEVICE)
                    kwargs = dict(
                        log=log,
                        sparse_matrix=sparse_matrix,
                        layera=layera,
                        layerb=layerb,
                        ones=ones,
                        sparsity_level=sparsity_level,
                        dense_level=dense_level,
                        reps=dense_levels[dense_level]["Reps"],
                        batsh_sz=BATCH_SIZE,
                    )
                    caller(**kwargs)
                except Exception as e:
                    print(e)
                    if is_oom_error(e) and TESTING_DEVICE.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue

                df = pd.DataFrame(log)

                df.to_csv(f"./benchmark/{csv_name}-{method_name}.csv", index=False)

    df = pd.DataFrame(log)
    df.to_csv(f"./benchmark/{csv_name}-{method_name}.csv", index=False)

if __name__ == "__main__":
    benchmark(sys.argv[1])