import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import time
from tqdm import tqdm
from sparse.mult.tensor import SparseTensor
import torch
import scipy.sparse as sp


def test_jax(
    log: list,
    sparse_matrix: SparseTensor,
    layera: int,
    layerb: int,
    ones: torch.Tensor,
    sparsity_level,
    dense_level,
    reps,
    batsh_sz,
    csr=False,
    bsr=False,
):

    gpu = jax.devices("gpu")[0] if jax.local_devices() else None


    # Convert to CSR format using scipy

    if csr:
        sparse_matrix = jsparse.csr_fromdense(sparse_matrix.dense().cpu())
    elif bsr:
        sparse_matrix = jsparse.bcsr_fromdense(sparse_matrix.dense().cpu())
    else:
        sparse_matrix = jsparse.coo_fromdense(sparse_matrix.dense().cpu())

    sparse_matrix_gpu = jax.device_put(sparse_matrix, gpu)
    ones_numpy = ones.cpu().numpy()  # Move to CPU and convert to NumPy
    ones_jax = jax.device_put(jnp.array(ones_numpy), gpu)
    vector_gpu = jax.device_put(ones_jax, gpu)

    @jax.jit
    def sparse_mv(sparse_mat, vec):
        return sparse_mat @ vec

    batched_sparse_mv = jax.vmap(sparse_mv, in_axes=(None, 0))

    for rep in tqdm(list(range(reps)), desc="Jax", leave=False):
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
        log.append(
            {
                "isSparse": "JaxSparse",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )


def test_sparse_ut(
    log: list,
    sparse_matrix: SparseTensor,
    layera: int,
    layerb: int,
    ones: torch.Tensor,
    sparsity_level,
    dense_level,
    reps,
    batsh_sz,
):
    # import ipdb;ipdb.set_trace()
    temp = sparse_matrix @ ones
    for rep in tqdm(list(range(reps)), desc="Reps sparse", leave=False):
        res = ones
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            res = sparse_matrix @ res
            res = res / res.max()
            res = ones
        s = res.sum()

        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        log.append(
            {
                "isSparse": "SparseUT",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )


def test_dense(
    log: list,
    sparse_matrix: SparseTensor,
    layera: int,
    layerb: int,
    ones: torch.Tensor,
    sparsity_level,
    dense_level,
    reps,
    batsh_sz,
):
    dense_matrix = sparse_matrix.dense()

    for rep in tqdm(list(range(reps)), desc="Reps dense", leave=False):
        res = ones
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            res = res @ dense_matrix
            res = res / res.max()
            res = ones
        s = res.sum()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        t2 = time.time()

        log.append(
            {
                "isSparse": "Dense",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )

    del dense_matrix


def test_sparse_torch(
    log: list,
    sparse_matrix: SparseTensor,
    layera: int,
    layerb: int,
    ones: torch.Tensor,
    sparsity_level,
    dense_level,
    reps,
    batsh_sz,
    csr=False,
):
    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )

    sparse_tensor = torch.sparse_coo_tensor(indices, sparse_matrix.values, sparse_matrix.matrix_shape, device="cuda")
    if csr:
        sparse_tensor=sparse_tensor.to_sparse_csr()
    del sparse_matrix
    torch.cuda.empty_cache()

    temp = ones @ sparse_tensor
    for rep in tqdm(list(range(reps)), desc="Reps sparse torch", leave=False):
        res = ones
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            res = res @ sparse_tensor
            res = res / res.max()
            res = ones
        s = res.sum()

        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        log.append(
            {
                "isSparse": "SparseTorch",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )






def test_sparse_cupy(
    log: list,
    sparse_matrix: SparseTensor,
    layera: int,
    layerb: int,
    ones: torch.Tensor,
    sparsity_level,
    dense_level,
    reps,
    batsh_sz,
    csr=False,
):

    import cupy as cp
    from cupy import sparse
    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )

    sparse_tensor = sparse.coo_matrix((
        cp.asarray(sparse_matrix.values.cpu().numpy()),
        cp.asarray(indices.to(dtype=torch.int32).cpu().numpy())),
        shape=sparse_matrix.matrix_shape
    )

    del sparse_matrix
    if csr:
        sparse_tensor = sparse_tensor.tocsr()
    torch.cuda.empty_cache()
    ones = cp.asarray(ones.cpu().numpy())
    temp = ones @ sparse_tensor
    for rep in tqdm(list(range(reps)), desc="Reps sparse cupy", leave=False):
        res = ones
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            res = res @ sparse_tensor
            res = res / res.max()
            res = ones
        s = res.sum()

        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        log.append(
            {
                "isSparse": "CuPy Sparse CSR",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )




def test_sparse_cupy_bsr(
    log: list,
    sparse_matrix: SparseTensor,
    layera: int,
    layerb: int,
    ones: torch.Tensor,
    sparsity_level,
    dense_level,
    reps,
    batsh_sz,
):

    import cupy as cp
    from cupy import sparse
    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )

    sparse_tensor = sparse.coo_matrix((
        cp.asarray(sparse_matrix.values.cpu().numpy()),
        cp.asarray(indices.to(dtype=torch.int32).cpu().numpy())),
        shape=sparse_matrix.matrix_shape
    )
    sparse_tensor = sparse_tensor.tobsr(blocksize=(4, 4))
    del sparse_matrix

    torch.cuda.empty_cache()
    ones = cp.asarray(ones.cpu().numpy())
    temp = ones @ sparse_tensor
    for rep in tqdm(list(range(reps)), desc="Reps sparse cupy bsr", leave=False):
        res = ones
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            res = res @ sparse_tensor
            res = res / res.max()
            res = ones
        s = res.sum()

        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        log.append(
            {
                "isSparse": "CuPy Sparse CSR",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )

def test_sparse_cupy_csr(**kwargs):
    kwargs["csr"]=True
    test_sparse_cupy(**kwargs)
    kwargs["csr"] = False

def test_sparse_torch_csr(**kwargs):
    kwargs["csr"] = True
    test_sparse_torch(**kwargs)
    kwargs["csr"] = False


def test_jax_csr(**kwargs):
    kwargs["csr"] = True
    test_jax(**kwargs)
    kwargs["csr"] = False

def test_jax_bsr(**kwargs):
    kwargs["bsr"] = True
    test_jax(**kwargs)
    kwargs["bsr"] = False




