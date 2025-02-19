#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;

__global__ void sparse_multiply_kernel(
    const float* activations, int nnz1, int nnz1_2,
    const float* sparse_matrix_values, int nnz2,
    const unsigned short* sparse_matrix_indices_a,
     const unsigned short* sparse_matrix_indices_b, int nnz3,
    float* output_values, int sparseCols) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;



    if (tid >= nnz2) return;
    // sparse_matrix =
    /*

      | 0 0 0 4 |
    w=| 0 1 0 0 |, => sparse_matrix = |0 3 4 1 1 1 2 0 1 2 1 1 2 2 1 2 3 1|
      | 1 1 1 1 |

      {1 1 1} = {1 2 1 5}
      meaning sparse_matrix = { X Y VALUE ...}

    */

    unsigned short x = sparse_matrix_indices_a[tid];
    unsigned short y = sparse_matrix_indices_b[tid];
    unsigned short batch_ind = 0;
    unsigned short out_shift = 0;

    for(int i=0;i<nnz1;i++){

        atomicAdd(&output_values[out_shift + y], activations[batch_ind + x] * sparse_matrix_values[tid]);
        batch_ind = batch_ind + nnz1_2;
        out_shift = out_shift + sparseCols;
    }

}
torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices_a,
    torch::Tensor sparse_matrix_indices_b, int64_t sparseCols) {

    int nnz1 = activations.size(0);
    int nnz1_2 = activations.size(1);
    auto output_values = torch::zeros({nnz1, sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    int nnz2 = sparse_matrix_indices_a.size(0);
    int nnz3 = sparse_matrix_values.size(0);
    printf("===> %d", nnz1);
    const int threads = 512; // this was 256
    const int blocks = (nnz2 + threads - 1) / threads;
    // std::cout<< activations << ' ' << sparse_matrix_indices << ' ' << sparse_matrix_values << std::endl;
    sparse_multiply_kernel<<<blocks, threads>>>(
        activations.data_ptr<float>() , nnz1, nnz1_2,
        sparse_matrix_values.data_ptr<float>(), nnz2,
        sparse_matrix_indices_a.data_ptr<unsigned short>(),
        sparse_matrix_indices_b.data_ptr<unsigned short>(), nnz3,
        output_values.data_ptr<float>(), sparseCols);

    return output_values;
}