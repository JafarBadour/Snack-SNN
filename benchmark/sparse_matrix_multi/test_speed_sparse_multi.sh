#!/usr/bin/env bash
set -euo pipefail

# Run benchmark methods from test_speed_sparse_tensor_vs_dense_tensor.py.
# Usage examples:
#   bash benchmark/sparse_matrix_multi/test_speed_sparse_multi.sh
#   METHODS="test_dense test_cusparse_csr_library" bash benchmark/sparse_matrix_multi/test_speed_sparse_multi.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_METHODS=(
  test_sputnik
  test_flashsparse
  test_sparse_ut
  test_dense
  test_jax_bsr
  test_jax
  test_sparse_torch
  test_cusparse_coo_library
  test_cusparse_csr_library
  test_sputnik_csr_dl_optimized
)

# Optional methods (require extra installs/backends):
#   test_ge_spmm_dgsparse_csr_gnn_optimized
#   test_sparse_cupy
#   test_sparse_cupy_csr
#   test_sparse_torch_csr
#   test_jax
#   test_jax_bsr

if [[ -n "${METHODS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_METHODS=(${METHODS})
else
  RUN_METHODS=("${DEFAULT_METHODS[@]}")
fi

echo "Running methods: ${RUN_METHODS[*]}"
for method in "${RUN_METHODS[@]}"; do
  echo ""
  echo "========== ${method} =========="
  python -m benchmark.sparse_matrix_multi.test_speed_sparse_tensor_vs_dense_tensor "${method}"
done