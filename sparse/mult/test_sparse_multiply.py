import torch
import os, sys


sys.path.append("C:/Users/BadourJ/Arts/Parallel-Dynamic-Sparse-Training/sparse/mult")
from sparse_tensor_multiply import sparse_multiply

def convert_sparse_to_dense(sparse_tensor : torch.Tensor, indices):
    max_sz = max(indices1) + 1
    res = torch.zeros(max_sz, device='cuda')
    res[indices] = sparse_tensor
    return res
get_diagonal = lambda siz: torch.tensor([[i,i] for i in range(siz)], dtype=torch.int64, device='cuda')
SZ = 30000     
indices1 = get_diagonal(SZ)
values1 = torch.tensor(list(range(SZ)), device='cuda')

indices2 = get_diagonal(SZ)
values2 = torch.tensor(list(range(SZ)), device='cuda')

import time


max_sz = max(indices1) + 1
values_cuda_3, values_cuda_4 = convert_sparse_to_dense(values1, indices1), convert_sparse_to_dense(values2, indices2)
values_3, values_4 = values_cuda_3.cpu(), values_cuda_4.cpu()
t2 = time.time()
cpu_res = values_3 * values_4
torch.cuda.synchronize()
t4 = time.time()

cpu_time = t4 - t2
print(cpu_res.shape)

print("=============  cpu multiplication: ", cpu_time)


max_sz = max(indices1) + 1

values_3, values_4 = values_cuda_3, values_cuda_4


t2 = time.time()
cpu_res = values_3 * values_4
torch.cuda.synchronize()
t4 = time.time()
print(cpu_res)  # cput out
cuda_not_sparse_time = t4 - t2
print(cpu_res.shape)

print("============= cuda multiplication", cuda_not_sparse_time)

t1 = time.time()
output = sparse_multiply(indices1, values1, indices2, values2, max_sz)
torch.cuda.synchronize()
t2 = time.time()
cuda_sparse_time = t2 - t1
print(output.shape)

print("============= time sparse cuda multiplication", cuda_sparse_time)

print("cuda sparse time / cuda not sparse time ", cuda_sparse_time/ cuda_not_sparse_time)


