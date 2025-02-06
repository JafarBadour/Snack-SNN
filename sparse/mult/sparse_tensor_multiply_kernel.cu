#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;

__global__ void sparse_multiply_kernel(
    const float* activations, int nnz1, int nnz1_2,
    const float* sparse_matrix_values, int nnz2,
    const unsigned short* sparse_matrix_indices_a,
     const unsigned short* sparse_matrix_indices_b,
     int nnz3,
    float* output_values, int sparseCols) {

    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    int batch_n = blockIdx.x;



    if (tid >= nnz2) return;
    //printf("%d,%d -- %d, %d, %d, %d  -- %d %d \n",
     //tid, batch_n, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,  blockDim.x, blockDim.y);

    // sparse_matrix =
    /*

      | 0 0 0 4 |
    w=| 0 1 0 0 |, => sparse_matrix = |0 3 4 1 1 1 2 0 1 2 1 1 2 2 1 2 3 1|
      | 1 1 1 1 |

      {1 1 1} = {1 2 1 5}
      meaning sparse_matrix = { X Y VALUE ...}

    */


    unsigned short out_offset = batch_n * sparseCols;
     unsigned short act_offset = batch_n * nnz1_2;

    unsigned short x = sparse_matrix_indices_a[tid];
    unsigned short y = sparse_matrix_indices_b[tid];
    float sp_val = sparse_matrix_values[tid];

    atomicAdd(&output_values[out_offset + y], activations[act_offset + x] * sp_val);

    
}

// Sparse tensor multiplication interface
torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices_a,
    torch::Tensor sparse_matrix_indices_b, int64_t sparseCols) {

    
    int nnz1 = activations.size(0);
    int nnz1_2 = activations.size(1);
    auto output_values = torch::zeros({nnz1, sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    int nnz2 = sparse_matrix_indices_a.size(0);
    int nnz3 = sparse_matrix_values.size(0);

    const int threads = 1024; // this was 256
    const int blocks = (nnz2 + threads - 1) / threads;
    dim3 threadsPerBlock(threads);    // 16 threads in each dimension; 16 is batch for ex
    dim3 blocksPerGrid(nnz1, blocks);      //

    sparse_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        activations.data_ptr<float>() , nnz1, nnz1_2,
        sparse_matrix_values.data_ptr<float>(), nnz2,
        sparse_matrix_indices_a.data_ptr<unsigned short>(),
        sparse_matrix_indices_b.data_ptr<unsigned short>(), nnz3,
        output_values.data_ptr<float>(), sparseCols);

    return output_values;
}
