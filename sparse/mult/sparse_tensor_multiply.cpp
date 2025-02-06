#include <torch/extension.h>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <unordered_set>
#include <random>

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
        int input_size, int output_size, int sub_edges_sz,  torch::Tensor built_edges
    ) {
    int * ptr = new int [2 * sub_edges_sz];
    std::unordered_set<long long> hashy;
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister engine, initialized with the random seed
    std::mt19937 gen2(rd());  // Mersenne Twister engine, initialized with the random seed
    std::uniform_int_distribution<> dis_a(0, input_size-1);
    std::uniform_int_distribution<> dis_b(0, output_size-1);

    int * built_edges_ptr = built_edges.data_ptr<int>();
    int maxy = 1 + (input_size > output_size) ? input_size : output_size;
    for(int i=0;i<built_edges.size(0); i++){
            long long hash_t = 1ll * built_edges_ptr[i] * maxy + built_edges_ptr[i+1];
            hashy.insert(hash_t);
    }
    for(int i=0;i< 2*sub_edges_sz;i+=2){
        int depth = 0;
        int64_t hash_p = -1;
        do
        {
            hashy.insert(hash_p);
            //assert(0 && "depth exceeded generating random numbers issue");
            if(depth > 25000){
                throw std::runtime_error("Random gen error: depth 25k trying to sample unreplaced number");
            }
            depth = depth + 1;
            ptr[i] = dis_a(gen);
            ptr[i+1] = dis_b(gen2);
            hash_p = 1ll* ptr[i] * maxy + ptr[i+1];

        }while(hashy.find(hash_p)!= hashy.end() || 0 > ptr[i] || ptr[i]>=input_size || ptr[i+1] < 0 || ptr[i+1] >= output_size);
        hashy.insert(hash_p);
    }

    auto res =  at::from_blob(ptr, {sub_edges_sz, 2}, at::kInt);;
   // delete ptr;
    return res;
}




torch::Tensor random_init_without_replacement(
        int input_size, int output_size, int sub_edges_sz,  torch::Tensor built_edges
    ) {
    int * ptr = new int [2 * sub_edges_sz];
    std::unordered_set<long long> hashy;
    std::random_device rd;
    std::mt19937 gen(rd());  // Mersenne Twister engine, initialized with the random seed
    std::mt19937 gen2(rd());  // Mersenne Twister engine, initialized with the random seed
    std::uniform_int_distribution<> dis_a(0, input_size-1);
    std::uniform_int_distribution<> dis_b(0, output_size-1);

    int * built_edges_ptr = built_edges.data_ptr<int>();
    int maxy = 1 + (input_size > output_size) ? input_size : output_size;
    for(int i=0;i<built_edges.size(0); i++){
            long long hash_t = 1ll * built_edges_ptr[i] * maxy + built_edges_ptr[i+1];
            hashy.insert(hash_t);
    }
    for(int i=0;i< 2*sub_edges_sz;i+=2){
        int depth = 0;
        int64_t hash_p = -1;
        do
        {
            hashy.insert(hash_p);
            //assert(0 && "depth exceeded generating random numbers issue");
            if(depth > 25000){
                throw std::runtime_error("Random gen error: depth 25k trying to sample unreplaced number");
            }
            depth = depth + 1;
            ptr[i] = dis_a(gen);
            ptr[i+1] = dis_b(gen2);
            hash_p = 1ll* ptr[i] * maxy + ptr[i+1];

        }while(hashy.find(hash_p)!= hashy.end() || 0 > ptr[i] || ptr[i]>=input_size || ptr[i+1] < 0 || ptr[i+1] >= output_size);
        hashy.insert(hash_p);
    }

    auto res =  at::from_blob(ptr, {sub_edges_sz, 2}, at::kInt);;
   // delete ptr;
    return res;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_multiply", &sparse_multiply, "Sparse Tensor Multiplication (CUDA)");
    m.def("sparse_multiply_cpu", &sparse_multiply_cpu, "Sparse Tensor Multiplication (CPU)");

    m.def("random_init_without_replacement", &random_init_without_replacement, "random_init_without_replacement");
}


