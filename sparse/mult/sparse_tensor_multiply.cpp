#include <torch/extension.h>
#include <vector>


torch::Tensor sparse_multiply_cuda(
    torch::Tensor indices1, torch::Tensor values1, 
    torch::Tensor indices2, torch::Tensor values2, 
    int64_t size);

// i think here we can add N by M tensor, or N by Z where Z is a tensor by itself but it should be flattened
torch::Tensor sparse_multiply(
    torch::Tensor indices1, torch::Tensor values1, 
    torch::Tensor indices2, torch::Tensor values2, 
    int64_t size) {
    // regular asserts
    TORCH_CHECK(indices1.device().is_cuda(), "Indices1 must be a CUDA tensor");
    TORCH_CHECK(indices2.device().is_cuda(), "Indices2 must be a CUDA tensor");
    TORCH_CHECK(values1.device().is_cuda(), "Values1 must be a CUDA tensor");
    TORCH_CHECK(values2.device().is_cuda(), "Values2 must be a CUDA tensor");

    return sparse_multiply_cuda(indices1, values1, indices2, values2, size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_multiply", &sparse_multiply, "Sparse Tensor Multiplication (CUDA)");
}
