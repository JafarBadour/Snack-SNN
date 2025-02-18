#include <torch/extension.h>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <unordered_set>
#include <random>
#include <bitset>

typedef long long ll;
torch::Tensor sparse_multiply_cuda(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices_a,
    torch::Tensor sparse_matrix_indices_b,
    int64_t sparseCols // or next layer how many neurons
    );


torch::Tensor sparse_multiply(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices_a,
    torch::Tensor sparse_matrix_indices_b, int64_t sparseCols // or next layer how many neurons
    ) {
    // regular asserts
    TORCH_CHECK(activations.device().is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(sparse_matrix_values.device().is_cuda(), "sparse_matrix must be a CUDA tensor");
    TORCH_CHECK(sparse_matrix_indices_a.device().is_cuda(), "sparse_matrix must be a CUDA tensor");
    TORCH_CHECK(sparse_matrix_indices_b.device().is_cuda(), "sparse_matrix must be a CUDA tensor");


    return sparse_multiply_cuda(activations, sparse_matrix_values, sparse_matrix_indices_a, sparse_matrix_indices_b, sparseCols);
}



torch::Tensor sparse_multiply_cpu(
    torch::Tensor activations, torch::Tensor sparse_matrix_values, torch::Tensor sparse_matrix_indices_a,
    torch::Tensor sparse_matrix_indices_b,
    int64_t sparseCols // or next layer how many neurons
    ) {

    TORCH_CHECK(activations.device().is_cpu(), "activations must be a CPU tensor");
    TORCH_CHECK(sparse_matrix_values.device().is_cpu(), "sparse_matrix must be a CPU tensor");
    TORCH_CHECK(sparse_matrix_indices_a.device().is_cpu(), "sparse_matrix_indices_a must be a CPU tensor");
    TORCH_CHECK(sparse_matrix_indices_b.device().is_cpu(), "sparse_matrix_indices_b must be a CPU tensor");

    int nnz1 = activations.size(0);
    int nnz1_2 = activations.size(1);
    auto output_values = torch::zeros({nnz1, sparseCols}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    int nnz2 = sparse_matrix_indices_a.size(0);
    int nnz3 = sparse_matrix_values.size(0);
    float* act = activations.data_ptr<float>();
    float* sp_v = sparse_matrix_values.data_ptr<float>();
    unsigned short * sp_i_a = sparse_matrix_indices_a.data_ptr<unsigned short>();
    unsigned short * sp_i_b = sparse_matrix_indices_b.data_ptr<unsigned short>();
    float * out = output_values.data_ptr<float>();
    //printf("sup?==========\n");
    for(int i=0;i<nnz2;i++){ // sp indices count/ count of values
        for(int j=0;j<nnz1;j++){ // batches count
            int x,y;
            x = sp_i_a[i];
            y = sp_i_b[i];
            out[j * sparseCols + y] = sp_v[i] * act[j*nnz1 + x];
        }
    }
    return output_values;
    ///
    }




torch::Tensor random_init_without_replacement(
        const int input_size, const int output_size,const int sub_edges_sz,  torch::Tensor built_edges
    ) {
    if(1ll * input_size * output_size  / 2 < sub_edges_sz)
        throw std::runtime_error("Uniform distribution accepts sparsity > 0.5 only");
    std::vector<std::vector<bool>> hashy(input_size, std::vector<bool>(output_size));
    std::random_device rd;
    std::random_device rd2;
    std::mt19937 gen(rd());  // Mersenne Twister engine, initialized with the random seed
    std::mt19937 gen2(rd2());  // Mersenne Twister engine, initialized with the random seed
    std::uniform_int_distribution<> dis_a(0, input_size-1);
    std::uniform_int_distribution<> dis_b(0, output_size-1);

    torch::Tensor res = torch::zeros({sub_edges_sz, 2}, torch::kInt);

    int* ptr = res.data_ptr<int>();  // Get pointer to Tensor's internal memory

    int * built_edges_ptr = built_edges.data_ptr<int>();
    int maxy = 1 + (input_size > output_size) ? input_size : output_size;
    for(int i=0;i<built_edges.size(0); i++){
            hashy[built_edges_ptr[i]][built_edges_ptr[i+1]] = 1;
    }

    for(int i=0;i< 2*sub_edges_sz;i+=2){
        int depth = 0;
        int hash_x, hash_y;
        hash_x = -1;
        hash_y = -1;
        do
        {
            if(depth > 100){
                throw std::runtime_error("Random gen error: depth 25 trying to sample unreplaced number");
            }
            depth = depth + 1;
            hash_x= dis_a(gen);
            hash_y = dis_b(gen2);
            ptr[i] = hash_x;
            ptr[i+1] = hash_y;
            if(hashy[hash_x][hash_y])
                continue;
            hashy[hash_x][hash_y] = 1;
            break;
        }while(true);

    }

    return res;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_multiply", &sparse_multiply, "Sparse Tensor Multiplication (CUDA)");
    m.def("sparse_multiply_cpu", &sparse_multiply_cpu, "Sparse Tensor Multiplication (CPU)");

    m.def("random_init_without_replacement", &random_init_without_replacement, "random_init_without_replacement");
}


