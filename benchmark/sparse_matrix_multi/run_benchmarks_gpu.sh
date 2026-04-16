#!/usr/bin/env bash
# Run sparse matrix benchmark methods on a GPU (interactive node or inside sbatch).
# Usage (from repo root, after activating venv):
#   bash benchmark/sparse_matrix_multi/run_benchmarks_gpu.sh
# Methods (space-separated); default is SparseUT + Sputnik only:
#   METHODS="test_sparse_ut test_sputnik" bash benchmark/sparse_matrix_multi/run_benchmarks_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load nvidia/cuda-12.4 2>/dev/null || true
fi

if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH:-}}"

# Default: your SparseTensor op vs Sputnik (override with METHODS="...")
METHODS="${METHODS:-test_sputnik}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for method in ${METHODS}; do
  echo ""
  echo "========== ${method} =========="
  python -m benchmark.sparse_matrix_multi.test_speed_sparse_tensor_vs_dense_tensor "${method}"
done

echo "Done. CSVs under ${REPO_ROOT}/benchmark/"

