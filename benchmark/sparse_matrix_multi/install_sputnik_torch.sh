#!/usr/bin/env bash
# Build Sputnik (C++/CUDA) and compile a local PyTorch wrapper extension in-place.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WRAP_DIR="${ROOT}/benchmark/sparse_matrix_multi/sputnik_torch_ext"

bash "${ROOT}/benchmark/sparse_matrix_multi/install_sputnik.sh"

python -c "import torch" >/dev/null 2>&1 || {
  echo "PyTorch not found in current Python. Activate your project venv first." >&2
  exit 1
}

cd "${WRAP_DIR}"
python setup.py build_ext --inplace

echo "Done. Check import with:"
echo "  PYTHONPATH=\"${WRAP_DIR}:\$PYTHONPATH\" python -c \"import sputnik_torch_ext; print('sputnik_torch_ext ok')\""
