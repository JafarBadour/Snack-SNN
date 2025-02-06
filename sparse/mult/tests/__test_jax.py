import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import time
from tqdm import tqdm
from sparse.mult.tensor import SparseTensor
import torch
def test_jax(log : list, sparse_matrix :SparseTensor, layera : int, layerb: int,
             ones : torch.Tensor, sparsity_level, dense_level, reps, batsh_sz):

    gpu = jax.devices("gpu")[0] if jax.local_devices() else None
    sparse_matrix = jsparse.BCOO.fromdense(sparse_matrix.dense())
    sparse_matrix_gpu = jax.device_put(sparse_matrix, gpu)
    ones_numpy = ones.cpu().numpy()  # Move to CPU and convert to NumPy
    ones_jax = jax.device_put(jnp.array(ones_numpy), gpu)
    vector_gpu = jax.device_put(ones_jax, gpu)

    @jax.jit
    def sparse_mv(sparse_mat, vec):
        return sparse_mat @ vec

    batched_sparse_mv = jax.vmap(sparse_mv, in_axes=(None, 0))

    for rep in tqdm(list(range(reps)), desc='Jax', leave=False):
        gpu_result = vector_gpu
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        t1 = time.time()
        start_event.record()
        for _ in range(100):
            # print("shapes ##")
            # print(sparse_matrix_gpu.shape)
            # print(vector_gpu.shape)
            gpu_result = batched_sparse_mv(sparse_matrix_gpu, vector_gpu).block_until_ready()
            gpu_result = gpu_result / gpu_result.max()
            gpu_result = vector_gpu
        # check correctness
        end_event.record()
        t2 = time.time()

        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        log.append({"isSparse": "JaxSparse", "dense_level": dense_level,
                    "sparsity_level": sparsity_level, "time": t2 - t1, "cuda_elapsed_time": elapsed_time_ms,
                    'rep': rep, 'batch_size' : batsh_sz})

