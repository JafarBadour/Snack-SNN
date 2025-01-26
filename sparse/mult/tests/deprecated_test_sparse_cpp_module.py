import torch
import os, sys
from time import time as tic
from time import sleep as sleep
import random
random.seed(0)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
def ram_info():
    import psutil

    # Get system memory usage
    memory_info = psutil.virtual_memory()

    # Total RAM in bytes
    total_ram = memory_info.total

    # RAM used in bytes
    used_ram = memory_info.used

    # RAM free in bytes
    free_ram = memory_info.available

    # Print RAM usage in human-readable format (e.g., MB)
    print(f"Used RAM: {used_ram / 1024 ** 2:.2f} MB / {total_ram / 1024 ** 2:.2f} MB")




from sparse_tensor_multiply import sparse_multiply

activation_size = 10000
next_layer_size = 10000
REPs = 500
SPARSITY_LEVEL = float(sys.argv[2])
def return_inps():


    activations = torch.Tensor([random.random() for i in range(activation_size)])
    W_matrix = []
    for i in range(activation_size):
        for j in range(next_layer_size):

            r = random.randint(0, 1000)
            if r <= SPARSITY_LEVEL*10: # sparsity level
                continue
            W_matrix.append([i, j , random.random()])
    W_matrix = torch.Tensor(W_matrix)
    #W_matrix = torch.Tensor([[i, i , 1] for i in range(min(activation_size, next_layer_size))])

    # W_matrix = W_matrix.reshape(-1)
    return W_matrix, activations

W_matrix, activations = return_inps()
print(W_matrix.shape)
activations = activations.to('cuda')

W_matrix = W_matrix.to('cuda')






if sys.argv[1] == 'a':

    ## doing it the old way
    W_matrix_full = [[0.0 for _ in range(activation_size)] for __ in range(next_layer_size)]
    W_matrix=W_matrix.reshape(-1)
    for i in range(0, len(W_matrix), 3):
        x = int(W_matrix[i])
        y = int(W_matrix[i + 1])
        v = W_matrix[i + 2]

        W_matrix_full[x][y] = v
    W_matrix_full = torch.Tensor(W_matrix_full).to('cuda')
    start_event.record()
    t1 = tic()
    torch.cuda.synchronize()
    multy = activations

    for _ in range(REPs):
        multy = torch.matmul(multy, W_matrix_full)
        multy = multy/multy.max()
    print(multy.sum())
    end_event.record()
    torch.cuda.synchronize()
    print(ram_info())
    print("Dense", tic() - t1)
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Execution time: {elapsed_time_ms:.6f} ms")
else:

    W_matrix_values = W_matrix[:, 2]
    W_matrix_indices = W_matrix[:, :2].to(torch.int32)
    start_event.record()

    t1 = tic()

    torch.cuda.synchronize()
    res = activations
    for _ in range(REPs):
        # print(res.shape, W_matrix_values.shape, W_matrix_indices.shape)
        # print(res.dtype, W_matrix_values.dtype, W_matrix_indices.dtype)
        res = sparse_multiply(res, W_matrix_values, W_matrix_indices, next_layer_size)
        res = res / res.max()

    print(res.sum())
    end_event.record()
    torch.cuda.synchronize()

    print(ram_info())
    print("Sparse", tic() - t1)
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Execution time: {elapsed_time_ms:.6f} ms")


#assert(all(res == multy))