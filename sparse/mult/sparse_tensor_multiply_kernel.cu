#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;
const int MAX_THREADS = 1024;
__global__ void sparse_multiply_kernel(
    const float* activations, int nnz1, int nnz1_2,
    const float* sparse_matrix_values, int nnz2,
    const unsigned short* sparse_matrix_indices_a,
     const unsigned short* sparse_matrix_indices_b, int nnz3,
    float* output_values, int sparseCols) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_idx = blockIdx.y;


    if (tid >= nnz2) return;
    // sparse_matrix =
    /*

      | 0 0 0 4 |
    w=| 0 1 0 0 |, => sparse_matrix = |0 3 4 1 1 1 2 0 1 2 1 1 2 2 1 2 3 1|
      | 1 1 1 1 |

      {1 1 1} = {1 2 1 5}
      meaning sparse_matrix = { X Y VALUE ...}

    */
    __shared__ float shared_output[MAX_THREADS];

    unsigned short x = sparse_matrix_indices_a[tid];
    unsigned short y = sparse_matrix_indices_b[tid];
    unsigned short batch_ind = 0;
    unsigned short out_shift = 0;


    shared_output[threadIdx.x] = activations[batch_idx * nnz1_2 + x] * sparse_matrix_values[tid];

    __syncthreads();

    atomicAdd(&output_values[sparseCols * batch_idx + y], shared_output[threadIdx.x]);



}
torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices_a,
    torch::Tensor sparse_matrix_indices_b, int64_t sparseCols) {

    int nnz1 = activations.size(0);
    int nnz1_2 = activations.size(1);
    auto output_values = torch::zeros({nnz1, sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    int nnz2 = sparse_matrix_indices_a.size(0);
    int nnz3 = sparse_matrix_values.size(0);

    const int threads = MAX_THREADS; // this was 256
    //const int blocks = (nnz2 + threads - 1) / threads;
    dim3 blocks((nnz2 + threads - 1) / threads, nnz1);
    // std::cout<< activations << ' ' << sparse_matrix_indices << ' ' << sparse_matrix_values << std::endl;
    sparse_multiply_kernel<<<blocks, threads>>>(
        activations.data_ptr<float>() , nnz1, nnz1_2,
        sparse_matrix_values.data_ptr<float>(), nnz2,
        sparse_matrix_indices_a.data_ptr<unsigned short>(),
        sparse_matrix_indices_b.data_ptr<unsigned short>(), nnz3,
        output_values.data_ptr<float>(), sparseCols);
    return output_values;
}


__global__ void sparse_warp_csr_multiply_kernel(
    const float* activations, 
    int activation_len,
    int activation_batch_sz,
    const float* sparse_matrix_values, 
    const unsigned int* sparse_matrix_indptr,
    const unsigned int* sparse_matrix_indices,
    const int sparse_matrix_nnz, // nnz
    float* output_values, 
    int sparseCols, 
    int sparseRows) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_idx = blockIdx.y;

    
    
}

torch::Tensor sparse_warp_csr_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indptr,
    torch::Tensor sparse_matrix_indices, int64_t sparseCols, int64_t sparseRows) {

    int sparse_matrix_nnz = sparse_matrix_indices.size(0);
    int activation_len = activations.size(-1);
    int activation_batch_sz = activations.numel() / activation_len; // effectively batch size

    TORCH_CHECK(activation_len % activation_batch_sz == 0, "activations tensor is corrupted");
    
    auto output_values = torch::zeros({sparseRows, sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const int threads = MAX_THREADS; // this was 256
    dim3 blocks((sparse_matrix_nnz + threads - 1) / threads, activation_batch_sz);
    
    return output_values;
}



__global__ void sparse_outer_product_multiply_kernel(
    const float* left, int left_len,
    const unsigned short* indices_left,
    const float* right, int right_len,
     const unsigned short* indices_right,
    float* output_values, int max_len) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int batch_idx = blockIdx.y;


    if (tid >= max_len) return;

    //__shared__ float shared_output[MAX_THREADS];

    float x = left[indices_left[tid] + left_len * batch_idx];
    float y = right[indices_right[tid]+ right_len * batch_idx];


    atomicAdd(&output_values[tid], x * y);
}
torch::Tensor sparse_outer_product_multiply_cuda(
    torch::Tensor left, torch::Tensor indices_left, torch::Tensor right,
    torch::Tensor indices_right) {

    int max_len = std::max(indices_left.size(0), indices_right.size(0));
    auto output_values = torch::zeros({max_len}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    int batch_sz = left.size(0);

    const int threads = MAX_THREADS; // this was 256
    dim3 blocks((max_len + threads - 1) / threads, batch_sz);

    // std::cout<< activations << ' ' << sparse_matrix_indices << ' ' << sparse_matrix_values << std::endl;
    sparse_outer_product_multiply_kernel<<<blocks, threads>>>(
        left.data_ptr<float>(), left.size(1),
        indices_left.data_ptr<unsigned short>(),
        right.data_ptr<float>(), right.size(1),
        indices_right.data_ptr<unsigned short>(),
        output_values.data_ptr<float>(),
        max_len
        );
    output_values = output_values / (batch_sz);
    return output_values;
}

