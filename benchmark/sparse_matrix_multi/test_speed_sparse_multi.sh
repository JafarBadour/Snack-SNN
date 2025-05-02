#!/bin/bash

for idx in test_sparse_cupy \
           test_sparse_cupy_csr \
           test_sparse_torch_csr \
           test_jax_bsr test_jax \
           test_sparse_ut \
           test_sparse_torch \
           test_dense
do
    python benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py "$idx"
done
