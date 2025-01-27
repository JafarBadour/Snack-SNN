#include <torch/extension.h>
#include <vector>


torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices,
    int64_t sparseCols // or next layer how many neurons
    );


torch::Tensor sparse_multiply(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices,
    int64_t sparseCols // or next layer how many neurons
    ) {
    // regular asserts
    TORCH_CHECK(activations.device().is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(sparse_matrix_values.device().is_cuda(), "sparse_matrix must be a CUDA tensor");
    TORCH_CHECK(sparse_matrix_indices.device().is_cuda(), "sparse_matrix must be a CUDA tensor");


    return sparse_multiply_cuda(activations, sparse_matrix_values, sparse_matrix_indices, sparseCols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_multiply", &sparse_multiply, "Sparse Tensor Multiplication (CUDA)");
}
