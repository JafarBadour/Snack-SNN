#!/usr/bin/env bash
# Run benchmark/sparse_matrix_multi/test.py (default: tst5 — SpMM correctness vs dense).
#
# From the head node (submit a GPU job — same pattern as sbatch_gpu_benchmarks.sh):
#   cd ~/Parallel-Dynamic-Sparse-Training
#   sbatch benchmark/sparse_matrix_multi/sbatch_tst5_spmm.sh
#
# Already on a GPU node / interactive allocation:
#   bash benchmark/sparse_matrix_multi/run_tst5_spmm.sh
#
# Optional one-shot:
#   srun --gres=gpu:1 ... bash benchmark/sparse_matrix_multi/run_tst5_spmm.sh
#
# If Sputnik is built elsewhere: export SPUTNIK_PYTHON_SO=/path/to/libsputnik_python.so before sbatch/srun.

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

echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

exec python "${REPO_ROOT}/benchmark/sparse_matrix_multi/test.py"
