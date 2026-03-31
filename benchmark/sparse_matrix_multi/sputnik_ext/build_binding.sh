#!/usr/bin/env bash
# Build libsputnik_python.so next to this script. Requires a built Sputnik tree.
# Usage:
#   SPUTNIK_ROOT=~/sputnik SPUTNIK_BUILD=~/sputnik/build/sputnik ./build_binding.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPUTNIK_ROOT="${SPUTNIK_ROOT:-${HOME}/sputnik}"
SPUTNIK_BUILD="${SPUTNIK_BUILD:-${SPUTNIK_ROOT}/build/sputnik}"

if [[ ! -f "${SPUTNIK_BUILD}/libsputnik.so" ]]; then
  echo "Missing ${SPUTNIK_BUILD}/libsputnik.so — build Sputnik first (cmake + make)." >&2
  exit 1
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load nvidia/cuda-12.4 2>/dev/null || true
fi

CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
if [[ ! -d "${CUDA_HOME}/include" ]]; then
  echo "CUDA include dir not found (set CUDA_HOME or load a cuda module)." >&2
  exit 1
fi

g++ -shared -fPIC -O2 -std=c++11 \
  -I"${SPUTNIK_ROOT}" \
  -I"${CUDA_HOME}/include" \
  "${HERE}/sputnik_spmm_binding.cpp" \
  -o "${HERE}/libsputnik_python.so" \
  -L"${SPUTNIK_BUILD}" -Wl,-rpath,"${SPUTNIK_BUILD}" \
  -lsputnik \
  -L"${CUDA_HOME}/lib64" -lcudart

echo "Wrote ${HERE}/libsputnik_python.so"
