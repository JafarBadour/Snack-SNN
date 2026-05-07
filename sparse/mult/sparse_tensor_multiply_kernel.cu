#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
using namespace std;
const int MAX_THREADS = 1024;

__global__ void sparse_multiply_kernel(
    const float* activations, int nnz1, int nnz1_2,
    const float* sparse_matrix_values, int nnz2,
    const unsigned short* sparse_matrix_indices_a,
    const unsigned short* sparse_matrix_indices_b, int nnz3,
    float* output_values, int sparseCols);

namespace {

std::unordered_map<uint64_t, int> g_threads_cache;

inline int nearest_bucket(int v, int min_bucket, int max_bucket) {
    int b = min_bucket;
    while (b < v && b < max_bucket) b <<= 1;
    return std::min(std::max(b, min_bucket), max_bucket);
}

inline bool autotune_enabled() {
    const char* env = std::getenv("SNACK_AUTOTUNE");
    if (env == nullptr) return true;
    return std::string(env) != "0";
}

inline bool autotune_verbose() {
    const char* env = std::getenv("SNACK_AUTOTUNE_VERBOSE");
    return env != nullptr && std::string(env) == "1";
}

inline int parse_threads_override() {
    const char* env = std::getenv("SNACK_THREADS");
    if (env == nullptr) return -1;
    const int v = std::atoi(env);
    if (v == 128 || v == 256 || v == 512 || v == 1024) return v;
    return -1;
}

inline int default_threads_for_device() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Hopper/Ada/Ampere usually likes 256-512 here.
    if (prop.major >= 9) return 512;
    if (prop.major >= 8) return 512;
    return 256;
}

inline uint64_t make_cache_key(int device, int nnz_bucket, int batch_bucket) {
    return (static_cast<uint64_t>(device & 0xFFFF) << 48) |
           (static_cast<uint64_t>(nnz_bucket & 0xFFFFFF) << 24) |
           static_cast<uint64_t>(batch_bucket & 0xFFFFFF);
}

float benchmark_threads_candidate(
    int threads,
    const float* activations, int nnz1, int nnz1_2,
    const float* sparse_matrix_values, int nnz2,
    const unsigned short* sparse_matrix_indices_a,
    const unsigned short* sparse_matrix_indices_b, int nnz3,
    int sparseCols,
    torch::Tensor output_scratch
) {
    dim3 blocks((nnz2 + threads - 1) / threads, nnz1);

    // Warmup launch.
    sparse_multiply_kernel<<<blocks, threads>>>(
        activations, nnz1, nnz1_2,
        sparse_matrix_values, nnz2,
        sparse_matrix_indices_a, sparse_matrix_indices_b, nnz3,
        output_scratch.data_ptr<float>(), sparseCols
    );
    cudaDeviceSynchronize();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        sparse_multiply_kernel<<<blocks, threads>>>(
            activations, nnz1, nnz1_2,
            sparse_matrix_values, nnz2,
            sparse_matrix_indices_a, sparse_matrix_indices_b, nnz3,
            output_scratch.data_ptr<float>(), sparseCols
        );
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms / 5.0f;
}

int resolve_threads_for_sparse_multiply(
    int nnz1,
    int nnz2,
    int sparseCols,
    const float* activations,
    int nnz1_2,
    const float* sparse_matrix_values,
    const unsigned short* sparse_matrix_indices_a,
    const unsigned short* sparse_matrix_indices_b,
    int nnz3,
    torch::Tensor output_template
) {
    const int override_threads = parse_threads_override();
    if (override_threads > 0) return override_threads;

    int device = 0;
    cudaGetDevice(&device);
    const int nnz_bucket = nearest_bucket(nnz2, 1024, 1 << 20);
    const int batch_bucket = nearest_bucket(nnz1, 1, 256);
    const uint64_t key = make_cache_key(device, nnz_bucket, batch_bucket);

    auto it = g_threads_cache.find(key);
    if (it != g_threads_cache.end()) return it->second;

    int selected = default_threads_for_device();
    if (autotune_enabled()) {
        const int candidates[] = {128, 256, 512, 1024};
        float best_ms = std::numeric_limits<float>::max();
        int best_threads = selected;
        for (int t : candidates) {
            if (t > MAX_THREADS) continue;
            torch::Tensor scratch = torch::zeros_like(output_template);
            float ms = benchmark_threads_candidate(
                t,
                activations, nnz1, nnz1_2,
                sparse_matrix_values, nnz2,
                sparse_matrix_indices_a, sparse_matrix_indices_b, nnz3,
                sparseCols,
                scratch
            );
            if (ms < best_ms) {
                best_ms = ms;
                best_threads = t;
            }
        }
        selected = best_threads;
        if (autotune_verbose()) {
            std::cout << "[SNACK autotune] device=" << device
                      << " nnz_bucket=" << nnz_bucket
                      << " batch_bucket=" << batch_bucket
                      << " selected_threads=" << selected
                      << std::endl;
        }
    }
    g_threads_cache[key] = selected;
    return selected;
}

}  // namespace

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

    const int threads = resolve_threads_for_sparse_multiply(
        nnz1,
        nnz2,
        static_cast<int>(sparseCols),
        activations.data_ptr<float>(),
        nnz1_2,
        sparse_matrix_values.data_ptr<float>(),
        sparse_matrix_indices_a.data_ptr<unsigned short>(),
        sparse_matrix_indices_b.data_ptr<unsigned short>(),
        nnz3,
        output_values
    );
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

    // Computes the per-(indices_left[k], indices_right[k]) sum over the batch
    // axis of left and right:
    //   output[k] = sum_b left[b, indices_left[k]] * right[b, indices_right[k]]
    //
    // This is exactly d L / d W[indices_a[k], indices_b[k]] for a linear layer
    // Y = X @ W when called as sparse_outer_join(grad_Y, indices_b, X, indices_a).
    // The chain rule already absorbs any mean-reduction in dL/dY, so we must
    // NOT divide by batch_sz here. Doing so would shrink the weight gradient by
    // an extra 1/B factor and silently undertrain the SNACK weights relative
    // to a Dense+Mask reference.

    int max_len = std::max(indices_left.size(0), indices_right.size(0));
    auto output_values = torch::zeros({max_len}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    int batch_sz = left.size(0);

    const int threads = MAX_THREADS; // this was 256
    dim3 blocks((max_len + threads - 1) / threads, batch_sz);

    sparse_outer_product_multiply_kernel<<<blocks, threads>>>(
        left.data_ptr<float>(), left.size(1),
        indices_left.data_ptr<unsigned short>(),
        right.data_ptr<float>(), right.size(1),
        indices_right.data_ptr<unsigned short>(),
        output_values.data_ptr<float>(),
        max_len
        );
    return output_values;
}

