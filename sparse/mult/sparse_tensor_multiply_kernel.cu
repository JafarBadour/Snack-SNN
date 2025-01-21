#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;

__global__ void sparse_multiply_kernel(
    const float* activations, int nnz1,
    const float* sparse_matrix, int nnz2,
    float* output_values, int sparseCols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // printf("%d - %d  \n", tid, nnz2);
    if (tid >= nnz1) return;
    // sparse_matrix =
    /*

      | 0 0 0 4 |
    w=| 0 1 0 0 |, => sparse_matrix = |0 3 4 1 1 1 2 0 1 2 1 1 2 2 1 2 3 1|
      | 1 1 1 1 |
      meaning sparse_matrix = { X Y VALUE ...}

    */
    atomicAdd(&output_values[sparse_matrix[tid+1]], activations[sparse_matrix[tid] * sparse_matrix[tid+2]);
    
}

// Sparse tensor multiplication interface
torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix,
    int64_t sparseCols) {
    auto output_values = torch::zeros({sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int nnz1 = activations.size(0);
    int nnz2 = sparse_matrix.size(0) / 3;

    const int threads = 256; // this was 256
    const int blocks = (nnz2 + threads - 1) / threads;

    sparse_multiply_kernel<<<blocks, threads>>>(
        activations.data_ptr<float>() , nnz1,
        sparse_matrix.data_ptr<float>(), nnz2,
        output_values.data_ptr<float>(), sparseCols);

    return output_values;
}
