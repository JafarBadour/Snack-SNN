import random
import math

import torch

from sparse.mult.tensor import SparseTensor


def create_random_sparse_matrix(layer_a_sz: int, layer_b_sz: int, sparsity_level: int, seed=42) -> SparseTensor:
    random.seed(seed)

    cutoff = int(((100 - sparsity_level) / 100) * (layer_b_sz * layer_a_sz))
    sqrt = int(math.sqrt(cutoff))
    indices_a = list(range(layer_a_sz))
    random.shuffle(indices_a)
    indices_b = list(range(layer_b_sz))
    random.shuffle(indices_b)
    indices_a = torch.tensor(indices_a[:sqrt])
    indices_b = torch.tensor(indices_b[:sqrt])

    indices = torch.cartesian_prod(indices_a, indices_b)
    values = [random.random() for _ in range(len(indices))]
    indices = torch.tensor(indices, dtype=torch.int32)
    values = torch.tensor(values, dtype=torch.float32)
    return SparseTensor(indices=indices, values=values, matrix_shape=(layer_a_sz, layer_b_sz))
