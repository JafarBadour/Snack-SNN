#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
using namespace std;

__global__ void sparse_multiply_kernel(
    const int64_t* indices1, const float* values1, int nnz1,
    const int64_t* indices2, const float* values2, int nnz2,
    float* output_values, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // printf("%d - %d  \n", tid, nnz2);
    if (tid >= nnz1) return;
    
    int st, en, mid;
    st = 0;
    en = nnz1;
    while(st < en){
        mid = (st + en) >> 1;
        if(indices1[tid] < indices2[mid])
            st = mid + 1;
        else if (indices1[tid] > indices2[mid])
        {
            en = mid - 1;
        }   
        else{
            atomicAdd(&output_values[indices1[tid]], values1[tid] * values2[mid]);
            return;
        }    
    }
    
}

// Sparse tensor multiplication interface
torch::Tensor sparse_multiply_cuda(
    torch::Tensor indices1, torch::Tensor values1,
    torch::Tensor indices2, torch::Tensor values2,
    int64_t size) {
    auto output_values = torch::zeros({size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    int nnz1 = indices1.size(0);
    int nnz2 = indices2.size(0);

    const int threads = 256; // this was 256
    const int blocks = (nnz1 + threads - 1) / threads;

    sparse_multiply_kernel<<<blocks, threads>>>(
        indices1.data_ptr<int64_t>(), values1.data_ptr<float>(), nnz1,
        indices2.data_ptr<int64_t>(), values2.data_ptr<float>(), nnz2,
        output_values.data_ptr<float>(), size);

    return output_values;
}
