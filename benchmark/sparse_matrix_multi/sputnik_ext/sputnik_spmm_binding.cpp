// Thin C wrapper so Python can call Sputnik CudaSpmm via ctypes.
#include <cuda_runtime.h>
#include <cstdint>

#include "sputnik/spmm/cuda_spmm.h"

extern "C" {

int sputnik_spmm_float(
    int m, int k, int n, int nonzeros,
    const int* row_indices,
    const float* values,
    const int* row_offsets,
    const int* column_indices,
    const float* dense_matrix,
    float* output_matrix,
    void* stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t err = sputnik::CudaSpmm(
      m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
      dense_matrix, output_matrix, s);
  return static_cast<int>(err);
}

}  // extern "C"
