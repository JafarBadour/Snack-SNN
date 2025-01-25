#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;

__global__ void sparse_multiply_kernel(
    const float* activations, int nnz1,
    const float* sparse_matrix_values, int nnz2,
    const int* sparse_matrix_indices, int nnz3,
    float* output_values, int sparseCols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // printf("%d - %d  \n", tid, nnz2);
    if (tid >= nnz2) return;
    // sparse_matrix =
    /*

      | 0 0 0 4 |
    w=| 0 1 0 0 |, => sparse_matrix = |0 3 4 1 1 1 2 0 1 2 1 1 2 2 1 2 3 1|
      | 1 1 1 1 |
      meaning sparse_matrix = { X Y VALUE ...}

    */
    tid = tid * 2;
    int x = sparse_matrix_indices[tid];
    int y = sparse_matrix_indices[tid + 1];
    atomicAdd(&output_values[x], activations[y] * sparse_matrix_values[tid]);
    
}

// Sparse tensor multiplication interface
torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices,
    int64_t sparseCols) {
    auto output_values = torch::zeros({sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int nnz1 = activations.size(0);
    int nnz2 = sparse_matrix_values.size(0);
    int nnz3 = sparse_matrix_values.size(0);

    const int threads = 512; // this was 256
    const int blocks = (nnz2 + threads - 1) / threads;

    sparse_multiply_kernel<<<blocks, threads>>>(
        activations.data_ptr<float>() , nnz1,
        sparse_matrix_values.data_ptr<float>(), nnz2,
        sparse_matrix_indices.data_ptr<int>(), nnz3,
        output_values.data_ptr<float>(), sparseCols);

    return output_values;
}
