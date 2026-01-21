

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <unordered_map>



using namespace std;
const int MAX_THREADS = 1024;
__global__ void conv2d_sparse_tensor_multiply_kernel(
    const float* input_feature_map, 
    const float* sparse_filters_matrix,
    int W_in, 
    int H_in,
    int W_out,
    int H_out,
    int batch_sz,
    int in_channels, 
    int out_channels,
    int stride,
    int padding,
    int dilation,
    bool bias,

    int* sparse_filters_matrix_indices_f,
    // [0, 123, 254] -> [0:123] is Cin * Kw * Kh after sparsifying
    // then we use indices_r, indices_kw, indices_kh to get the values 
    // sparse_filters_matrix_values[indices_f + i]
    // r = indices_r[indices_f + i]
    // kw = indices_kw[indices_f + i]
    // kh = indices_kh[indices_f + i]

    int* sparse_filters_matrix_indices_r, // maybe all of these should be ints? #TODO

    int* sparse_filters_matrix_indices_kw,

    int* sparse_filters_matrix_indices_kh,

    float* sparse_filters_matrix_values,
    int filters_len,

    float* output_values  // output tensor

) {
    // Simple 1D grid: each thread computes one output element
    // Total outputs = N * Cout * H_out * W_out
    int total_outputs = batch_sz * out_channels * H_out * W_out;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_outputs)
        return;
    
    // Decode output position: (batch, out_channel, h, w)
    int batch_idx = tid / (out_channels * H_out * W_out);
    int remainder = tid % (out_channels * H_out * W_out);
    int out_c_idx = remainder / (H_out * W_out);
    remainder = remainder % (H_out * W_out);
    int pixel_h = remainder / W_out;
    int pixel_w = remainder % W_out;
    
    // Pre-compute loop-invariant values
    const int out_channels_H_out_W_out = out_channels * H_out * W_out;
    const int H_out_W_out = H_out * W_out;
    const int batch_input_offset = batch_idx * (in_channels * H_in * W_in);
    const int H_in_W_in = H_in * W_in;
    const int pixel_h_stride_minus_padding = pixel_h * stride - padding;
    const int pixel_w_stride_minus_padding = pixel_w * stride - padding;
    const int output_idx = batch_idx * out_channels_H_out_W_out + out_c_idx * H_out_W_out + pixel_h * W_out + pixel_w;
    
    // Each thread computes one output pixel (batch, out_channel, h, w)
    // Loop over sparse connections for this output channel
    int filter_idx = sparse_filters_matrix_indices_f[out_c_idx];
    int filter_end = sparse_filters_matrix_indices_f[out_c_idx + 1];
    float out_pixel_value = 0.0f;
    
    for (int i = filter_idx; i < filter_end; i++) {
        int r = sparse_filters_matrix_indices_r[i];  // input channel
        int kw = sparse_filters_matrix_indices_kw[i];
        int kh = sparse_filters_matrix_indices_kh[i];
        
        // Compute input position with stride and padding (using pre-computed values)
        int input_h = pixel_h_stride_minus_padding + kh;
        int input_w = pixel_w_stride_minus_padding + kw;
        
        // Bounds check for padding
        if (input_h >= 0 && input_h < H_in && input_w >= 0 && input_w < W_in) {
            int input_idx = batch_input_offset + r * H_in_W_in + input_h * W_in + input_w;
            out_pixel_value += sparse_filters_matrix_values[i] * input_feature_map[input_idx];
        }
    }
    
    // Write output
    output_values[output_idx] = out_pixel_value;
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
    int Cout = sparse_filters_matrix_indices_f.size(0) - 1;  // output channels (indices_f has Cout+1 elements with end marker)

    // Compute output dimensions: (H_in + 2*padding - Kh) / stride + 1
    int H_out = (H_in + 2 * padding - Kh) / stride + 1;
    int W_out = (W_in + 2 * padding - Kw) / stride + 1;

    // Output shape: (N, Cout, H_out, W_out)
    auto output_values = torch::zeros({N, Cout, H_out, W_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Launch kernel
    // Simple 1D grid: each thread computes one output element
    int total_outputs = N * Cout * H_out * W_out;
    const int threads = 256;  // Use 256 threads per block for better occupancy
    int num_blocks = (total_outputs + threads - 1) / threads;
    
    conv2d_sparse_tensor_multiply_kernel<<<num_blocks, threads>>>(
        input_feature_map.data_ptr<float>(),
        sparse_filters_matrix_values.data_ptr<float>(),
        W_in,
        H_in,
        W_out,
        H_out,
        N,
        in_channels,
        Cout,
        stride,
        padding,
        1,  // dilation (default 1)
        false,  // bias
        sparse_filters_matrix_indices_f.data_ptr<int>(),
        sparse_filters_matrix_indices_r.data_ptr<int>(),
        sparse_filters_matrix_indices_kw.data_ptr<int>(),
        sparse_filters_matrix_indices_kh.data_ptr<int>(),
        sparse_filters_matrix_values.data_ptr<float>(),
        filters_len,
        output_values.data_ptr<float>()
    );

    return output_values;
}
