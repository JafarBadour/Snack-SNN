#include <torch/extension.h>
#include <vector>


torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix, // 1d tensor (x, y, v)
    int64_t sparseCols
    );


torch::Tensor sparse_multiply(
    torch::Tensor activations, torch::Tensor sparse_matrix,
    int64_t sparseCols // or next layer how many neurons
    ) {
    // regular asserts
    TORCH_CHECK(activations.device().is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(sparse_matrix.device().is_cuda(), "sparse_matrix must be a CUDA tensor");


    return sparse_multiply_cuda(activations, sparse_matrix, sparseCols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_multiply", &sparse_multiply, "Sparse Tensor Multiplication (CUDA)");
}
