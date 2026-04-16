#!/usr/bin/env bash
set -euo pipefail

# Run all available benchmark methods from test_speed_sparse_tensor_vs_dense_tensor.py.
# Optional methods:
# - test_flashsparse: requires `bash benchmark/sparse_matrix_multi/install_flashsparse.sh`
# - test_sputnik:     requires `bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh`
METHODS=(
  test_sparse_cupy_csr
  test_sparse_cupy
  test_sparse_torch_csr
  test_jax_bsr
  test_jax
  test_sparse_ut
  test_sparse_torch
  test_dense
  test_flashsparse
  test_sputnik
)

for idx in "${METHODS[@]}"; do
  python benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py "$idx"
done

# # 
# import pandas as pd
# categories = [ "test_sparse_cupy_csr", "test_sparse_ut", "test_dense"]

# df_total = pd.concat([pd.read_csv(f"benchmark/log_mult_incl_cupy-{idx}.csv") for idx in categories],axis=0)
# stats = df_total.groupby(["sparsity_level", "batch_size", "isSparse"])['cuda_elapsed_time'].mean().reset_index()
# stats[stats.sparsity_level == 95]
# stats[stats.sparsity_level == 98]
# stats[stats.sparsity_level == 99]