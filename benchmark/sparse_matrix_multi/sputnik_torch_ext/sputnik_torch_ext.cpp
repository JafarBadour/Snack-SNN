#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "sputnik/spmm/cuda_spmm.h"

namespace {

void CheckCudaTensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void CheckInt32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
}

void CheckFloat32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

torch::Tensor SputnikSpmm(torch::Tensor row_indices,
                          torch::Tensor values,
                          torch::Tensor row_offsets,
                          torch::Tensor column_indices,
                          torch::Tensor dense_matrix) {
  CheckCudaTensor(row_indices, "row_indices");
  CheckCudaTensor(values, "values");
  CheckCudaTensor(row_offsets, "row_offsets");
  CheckCudaTensor(column_indices, "column_indices");
  CheckCudaTensor(dense_matrix, "dense_matrix");

  CheckInt32(row_indices, "row_indices");
  CheckFloat32(values, "values");
  CheckInt32(row_offsets, "row_offsets");
  CheckInt32(column_indices, "column_indices");
  CheckFloat32(dense_matrix, "dense_matrix");

  TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");
  TORCH_CHECK(values.dim() == 1, "values must be 1D");
  TORCH_CHECK(row_offsets.dim() == 1, "row_offsets must be 1D");
  TORCH_CHECK(column_indices.dim() == 1, "column_indices must be 1D");
  TORCH_CHECK(dense_matrix.dim() == 2, "dense_matrix must be 2D");

  const int m = static_cast<int>(row_indices.size(0));
  const int k = static_cast<int>(dense_matrix.size(0));
  const int n = static_cast<int>(dense_matrix.size(1));
  const int nonzeros = static_cast<int>(values.size(0));

  TORCH_CHECK(row_offsets.size(0) == m + 1, "row_offsets must have m + 1 entries");
  TORCH_CHECK(column_indices.size(0) == nonzeros, "column_indices and values size mismatch");

  auto out = torch::empty({m, n}, dense_matrix.options());
  const cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  cudaError_t err = sputnik::CudaSpmm(
      m,
      k,
      n,
      nonzeros,
      row_indices.data_ptr<int>(),
      values.data_ptr<float>(),
      row_offsets.data_ptr<int>(),
      column_indices.data_ptr<int>(),
      dense_matrix.data_ptr<float>(),
      out.data_ptr<float>(),
      stream);

  TORCH_CHECK(err == cudaSuccess, "sputnik::CudaSpmm failed: ", cudaGetErrorString(err));
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm", &SputnikSpmm, "Sputnik SpMM (CUDA)");
}
