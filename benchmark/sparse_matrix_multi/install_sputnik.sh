#!/usr/bin/env bash
# Build and locally install Sputnik (C++/CUDA shared library) under third_party/sputnik/install.
# Note: upstream Sputnik does NOT provide a Python/Torch extension in this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SPUTNIK_DIR="${ROOT}/third_party/sputnik"
BUILD_DIR="${SPUTNIK_DIR}/build"
INSTALL_DIR="${SPUTNIK_DIR}/install"

if [[ ! -d "${SPUTNIK_DIR}" ]]; then
  echo "Sputnik submodule not found at ${SPUTNIK_DIR}." >&2
  echo "Run: git submodule update --init --recursive third_party/sputnik" >&2
  exit 1
fi

# You can override this when needed, e.g. CUDA_ARCHS="89".
CUDA_ARCHS="${CUDA_ARCHS:-80;86;89;90}"

echo "Configuring Sputnik (CUDA_ARCHS=${CUDA_ARCHS})..."
cmake -S "${SPUTNIK_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHS="${CUDA_ARCHS}"

echo "Building Sputnik..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "Installing Sputnik to ${INSTALL_DIR}..."
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}"

echo "Done."
echo "Library: ${INSTALL_DIR}/lib/libsputnik.so"
