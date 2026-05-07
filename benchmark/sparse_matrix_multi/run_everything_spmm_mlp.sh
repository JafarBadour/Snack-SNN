#!/usr/bin/env bash
# Run full SpMM + MLP benchmark suite in foreground.
# - Performs environment preflight
# - Installs/updates benchmark dependencies
# - Builds required CUDA/PyTorch extensions when missing
# - Continues across method-level failures
#
# Usage:
#   bash benchmark/sparse_matrix_multi/run_everything_spmm_mlp.sh
#
# Optional overrides:
#   SPARSITY=0.95 WARMUP=10 ITERS=30 bash benchmark/sparse_matrix_multi/run_everything_spmm_mlp.sh
#   METHODS="test_sputnik test_dense" bash benchmark/sparse_matrix_multi/run_everything_spmm_mlp.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
warn() { echo "[$(timestamp)] WARNING: $*" >&2; }
err() { echo "[$(timestamp)] ERROR: $*" >&2; }

# Keep "set -e" globally, but run individual jobs in a tolerant wrapper.
run_tolerant() {
  local name="$1"
  shift
  log "START ${name}"
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    warn "FAILED ${name} (exit=${rc})"
  else
    log "DONE ${name}"
  fi
  return $rc
}

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
  log "Activated venv at ${REPO_ROOT}/venv"
else
  warn "No venv found at ${REPO_ROOT}/venv/bin/activate; using current Python environment."
fi

if ! command -v python >/dev/null 2>&1; then
  err "python not found in PATH."
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH:-}}"

log "REPO_ROOT=${REPO_ROOT}"
log "PYTHON=$(command -v python)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || warn "nvidia-smi not available."

log "Installing python dependencies"
run_tolerant "pip requirements.txt" pip install -r "${REPO_ROOT}/requirements.txt"
if [[ -f "${REPO_ROOT}/requirements-benchmark.txt" ]]; then
  run_tolerant "pip requirements-benchmark.txt" pip install -r "${REPO_ROOT}/requirements-benchmark.txt"
fi

log "Ensuring sparse CUDA extension is installed"
run_tolerant "install sparse/mult extension" pip install -e "${REPO_ROOT}/sparse/mult" --no-build-isolation

if ! python -c "import sputnik_torch_ext" >/dev/null 2>&1; then
  log "sputnik_torch_ext missing; building it"
  run_tolerant "install_sputnik_torch" bash "${REPO_ROOT}/benchmark/sparse_matrix_multi/install_sputnik_torch.sh"
else
  log "sputnik_torch_ext already available"
fi

# FlashSparse is required by test_flashsparse.
if ! python -c "import FS_SpMM, FS_Block_gpu" >/dev/null 2>&1; then
  log "FlashSparse modules missing; installing"
  run_tolerant "install_flashsparse" bash "${REPO_ROOT}/benchmark/sparse_matrix_multi/install_flashsparse.sh"
else
  log "FlashSparse modules already available"
fi

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
  test_ge_spmm_dgsparse_csr_gnn_optimized
)

if [[ -n "${METHODS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_METHODS=(${METHODS})
else
  RUN_METHODS=("${DEFAULT_METHODS[@]}")
fi

SPARSITY="${SPARSITY:-0.9}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
KERNEL_SIZES="${KERNEL_SIZES:-768x3072,3072x768,1024x4096,4096x1024,12500x12500,15000x15000,17500x17500}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32}"
SPUTNIK_LARGE_BATCHES="${SPUTNIK_LARGE_BATCHES:-1024,2048,3072,4096,12288}"

log "Running SpMM sweep methods: ${RUN_METHODS[*]}"
FAILED_METHODS=()
for method in "${RUN_METHODS[@]}"; do
  if ! run_tolerant "spmm-${method}" python -m benchmark.sparse_matrix_multi.test_speed_sparse_tensor_vs_dense_tensor "${method}"; then
    FAILED_METHODS+=("${method}")
  fi
done

log "Running MLP backend sweep"
if ! run_tolerant "mlp-check-snack-backends" \
  python benchmark/sparse_matrix_multi/check_snack_backends.py \
    --device cuda \
    --sparsity "${SPARSITY}" \
    --kernel-sizes "${KERNEL_SIZES}" \
    --batch-sizes "${BATCH_SIZES}" \
    --sputnik-large-batches "${SPUTNIK_LARGE_BATCHES}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}"; then
  warn "MLP backend sweep failed."
fi

echo
log "All requested runs completed."
log "SpMM CSV outputs: ${REPO_ROOT}/benchmark/apr-13-log_mult_incl_cupy-<method>.csv"
if [[ ${#FAILED_METHODS[@]} -gt 0 ]]; then
  warn "Methods with non-zero exit: ${FAILED_METHODS[*]}"
else
  log "All SpMM methods exited successfully."
fi

