#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor conv2d_sparse_tensor_multiply_cuda(
    torch::Tensor input_feature_map, 
    torch::Tensor sparse_filters_matrix_values,
    torch::Tensor sparse_filters_matrix_indices_f,
    torch::Tensor sparse_filters_matrix_indices_r,
    torch::Tensor sparse_filters_matrix_indices_kw,
    torch::Tensor sparse_filters_matrix_indices_kh,
    int Kw,
    int Kh,
    int64_t filters_len,
    int stride,
    int padding);

// Wrapper function with input validation
torch::Tensor conv2d_sparse_tensor_multiply(
    torch::Tensor input_feature_map, 
    torch::Tensor sparse_filters_matrix_values,
    torch::Tensor sparse_filters_matrix_indices_f,
    torch::Tensor sparse_filters_matrix_indices_r,
    torch::Tensor sparse_filters_matrix_indices_kw,
    torch::Tensor sparse_filters_matrix_indices_kh,
    int Kw,
    int Kh,
    int64_t filters_len,
    int stride,
    int padding) {
    
    TORCH_CHECK(input_feature_map.device().is_cuda(), "input_feature_map must be a CUDA tensor");
    TORCH_CHECK(sparse_filters_matrix_values.device().is_cuda(), "sparse_filters_matrix_values must be a CUDA tensor");
    TORCH_CHECK(sparse_filters_matrix_indices_f.device().is_cuda(), "sparse_filters_matrix_indices_f must be a CUDA tensor");
    TORCH_CHECK(sparse_filters_matrix_indices_r.device().is_cuda(), "sparse_filters_matrix_indices_r must be a CUDA tensor");
    TORCH_CHECK(sparse_filters_matrix_indices_kw.device().is_cuda(), "sparse_filters_matrix_indices_kw must be a CUDA tensor");
    TORCH_CHECK(sparse_filters_matrix_indices_kh.device().is_cuda(), "sparse_filters_matrix_indices_kh must be a CUDA tensor");
    TORCH_CHECK(input_feature_map.dim() == 4, "input_feature_map must be 4D (N, C, H, W)");
    
    return conv2d_sparse_tensor_multiply_cuda(
        input_feature_map, 
        sparse_filters_matrix_values,
        sparse_filters_matrix_indices_f,
        sparse_filters_matrix_indices_r,
        sparse_filters_matrix_indices_kw,
        sparse_filters_matrix_indices_kh,
        Kw, Kh, filters_len, stride, padding);
}
