

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>



using namespace std;
const int MAX_THREADS = 1024;
__global__ void conv2d_sparse_tensor_multiply_kernel(
    const float* input_feature_map, 
    const float* sparse_filters_matrix,
    int W, 
    int batch_sz,
    int H, 
    int in_channels, 
    int out_channels,
    int stride,
    int padding,
    int dilation,
    bool bias,

    unsigned short* sparse_filters_matrix_indices_f,
    // [0, 123, 254] -> [0:123] is Cin * Kw * Kh after sparsifying
    // then we use indices_r, indices_kw, indices_kh to get the values 
    // sparse_filters_matrix_values[indices_f + i]
    // r = indices_r[indices_f + i]
    // kw = indices_kw[indices_f + i]
    // kh = indices_kh[indices_f + i]

    unsigned short* sparse_filters_matrix_indices_r, // maybe all of these should be ints? #TODO

    unsigned short* sparse_filters_matrix_indices_kw,

    unsigned short* sparse_filters_matrix_indices_kh,

    float* sparse_filters_matrix_values,
    int filters_len,

    float* output_values  // output tensor

) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // blocks are [batch_sz* W, out_channels * H]

    int batch_idx = blockIdx.y;


    if (tid >= sparse_filters_matrix_values[filters_len-1]) 
        return;
    
    int batch_idx = blockIdx.x / batch_sz;
    int pixel_idx = blockIdx.x % batch_sz;
    int pixel_idy = blockIdx.y % out_channels;
    int out_c_idx = blockIdx.y / out_channels;


    
    int filter_idx = sparse_filters_matrix_indices_f[out_c_idx];
    int filter_end = sparse_filters_matrix_indices_f[out_c_idx+1];
    float out_pixel_value = 0;
    for(int i=filter_idx;i<filter_end;i++) {
        int r = sparse_filters_matrix_indices_r[i];
        int kw = sparse_filters_matrix_indices_kw[i];
        int kh = sparse_filters_matrix_indices_kh[i];
        
        // Compute input position with stride and padding
        // Output position (pixel_idx, pixel_idy) maps to input starting at (pixel_idx * stride - padding, pixel_idy * stride - padding)
        int input_h = pixel_idx * stride - padding + kh;
        int input_w = pixel_idy * stride - padding + kw;
        
        // Bounds check for padding (zero-padding: skip if out of bounds)
        if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
            int input_idx = r * H * W + input_h * W + input_w;
            out_pixel_value += sparse_filters_matrix_values[i] * input_feature_map[input_idx];
        }
        // else: padded region, treated as zero (no contribution)
    }
    output_values[out_c_idx * W * H + pixel_idx] = out_pixel_value;

}
torch::Tensor conv2d_sparse_tensor_multiply_cuda(
    torch::Tensor input_feature_map, 
    torch::Tensor sparse_filters_matrix_values,
    torch::Tensor sparse_filters_matrix_indices_f,
    torch::Tensor sparse_filters_matrix_indices_r,
    torch::Tensor sparse_filters_matrix_indices_kw,
    torch::Tensor sparse_filters_matrix_indices_kh,
    int Kw,
    int Kh,
    int64_t filters_len,
    int stride,
    int padding) {

    int N = input_feature_map.size(0);           // batch size
    int in_channels = input_feature_map.size(1); // Cin
    int H_in = input_feature_map.size(2);        // input height
    int W_in = input_feature_map.size(3);        // input width
    int Cout = sparse_filters_matrix_indices_f.size(0);  // output channels

    // Compute output dimensions: (H_in + 2*padding - Kh) / stride + 1
    int H_out = (H_in + 2 * padding - Kh) / stride + 1;
    int W_out = (W_in + 2 * padding - Kw) / stride + 1;

    // Output shape: (N, Cout, H_out, W_out)
    auto output_values = torch::zeros({N, Cout, H_out, W_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Launch kernel
    const int threads = MAX_THREADS;
    // Grid: (H_out * W_out, N * Cout)
    dim3 blocks((H_out * W_out + threads - 1) / threads, N * Cout);
    
    conv2d_sparse_tensor_multiply_kernel<<<blocks, threads>>>(
        input_feature_map.data_ptr<float>(),
        sparse_filters_matrix_values.data_ptr<float>(),
        W_in,
        N,
        H_in,
        in_channels,
        Cout,
        stride,
        padding,
        1,  // dilation (default 1)
        false,  // bias
        sparse_filters_matrix_indices_f.data_ptr<unsigned short>(),
        sparse_filters_matrix_indices_r.data_ptr<unsigned short>(),
        sparse_filters_matrix_indices_kw.data_ptr<unsigned short>(),
        sparse_filters_matrix_indices_kh.data_ptr<unsigned short>(),
        sparse_filters_matrix_values.data_ptr<float>(),
        filters_len,
        output_values.data_ptr<float>()
    );

    return output_values;
}
