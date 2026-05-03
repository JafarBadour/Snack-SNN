#!/usr/bin/env bash
# Clone ParCIS FlashSparse (PPoPP 2025) and pip-install CUDA extensions FS_SpMM + FS_Block_gpu.
# Requires: git, CUDA toolkit, same Python as your venv, PyTorch with CUDA.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${ROOT}/third_party/FlashSparse"
PKG="${DEST}/FlashSparse"

mkdir -p "${ROOT}/third_party"
if [[ ! -d "${DEST}/.git" ]]; then
  git clone --recursive https://github.com/ParCIS/FlashSparse.git "${DEST}"
else
  git -C "${DEST}" pull --ff-only || true
fi

if [[ ! -f "${PKG}/setup.py" ]]; then
  echo "Expected ${PKG}/setup.py (inner FlashSparse package). Repo layout may have changed." >&2
  exit 1
fi

cd "${PKG}"
# FlashSparse setup.py imports torch at configure time; pip's isolated build env has no torch.
python -c "import torch" >/dev/null 2>&1 || {
  echo "Activate your project venv first (needs torch installed)." >&2
  exit 1
}
pip install -e . --no-build-isolation

echo "Done. Re-run: python benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py test_flashsparse"
