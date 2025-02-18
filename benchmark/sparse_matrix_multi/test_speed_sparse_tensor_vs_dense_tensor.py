from tqdm import tqdm
from sparse.mult.tensor.utils import create_random_sparse_matrix
import torch
import pandas as pd
from benchmark.sparse_matrix_multi.__test_methods import (
    test_jax,
    test_dense,
    test_sparse_ut,
    test_sparse_torch,
)

TESTING_DEVICE = "cuda"  # "cuda"
sparsity_levels = (
    # list(range(0, 50, 24)) +
    list(range(50, 80, 10))
    + list(range(80, 96, 5))
    + [96, 97, 98, 99, 99.5, 99.9]
)

# dense level is layer A with layer B that are after one another in the model architecture
dense_levels = {
    "500x500": {"Reps": 5},
    "1000x1000": {"Reps": 5},
    "5000x5000": {"Reps": 5},
    "5200x5200": {"Reps": 5},
    # "40000x10000" : {'Reps' : 10},
    # "10000x10000" : {'Reps' : 10},
}

csv_name = "log_mult_5.2kx5.2k"
batches_cnt = [1 << i for i in range(9)]

import warnings

warnings.filterwarnings("ignore")


log = []

for BATCH_SIZE in tqdm(batches_cnt, desc="Batches processing"):
    for dense_level in tqdm(dense_levels.keys(), desc="Processing Dense Levels", leave=False):
        layera, layerb = list(map(int, dense_level.split("x")))

        for sparsity_level in tqdm(sparsity_levels, desc="Sparsity Lvls", leave=False):

            ones = torch.rand((BATCH_SIZE, layera)).to(TESTING_DEVICE)

            sparse_matrix = create_random_sparse_matrix(layera, layerb, sparsity_level)
            test_jax(
                log,
                sparse_matrix,
                layera,
                layerb,
                ones,
                sparsity_level=sparsity_level,
                dense_level=dense_level,
                reps=dense_levels[dense_level]["Reps"],
                batsh_sz=BATCH_SIZE,
            )

            # import ipdb;ipdb.set_trace()
            sparse_matrix = sparse_matrix.to(TESTING_DEVICE)
            test_sparse_ut(
                log,
                sparse_matrix,
                layera,
                layerb,
                ones,
                sparsity_level=sparsity_level,
                dense_level=dense_level,
                reps=dense_levels[dense_level]["Reps"],
                batsh_sz=BATCH_SIZE,
            )

            test_dense(
                log,
                sparse_matrix,
                layera,
                layerb,
                ones,
                sparsity_level=sparsity_level,
                dense_level=dense_level,
                reps=dense_levels[dense_level]["Reps"],
                batsh_sz=BATCH_SIZE,
            )

            test_sparse_torch(
                log,
                sparse_matrix,
                layera,
                layerb,
                ones,
                sparsity_level=sparsity_level,
                dense_level=dense_level,
                reps=dense_levels[dense_level]["Reps"],
                batsh_sz=BATCH_SIZE,
            )

            df = pd.DataFrame(log)

            df.to_csv(f"./benchmark/{csv_name}.csv", index=False)

df = pd.DataFrame(log)
df.to_csv(f"./benchmark/{csv_name}.csv", index=False)
