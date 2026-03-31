import ctypes
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
import time
from tqdm import tqdm
from sparse.mult.tensor import SparseTensor
import torch
import scipy.sparse as sp

_SPUTNIK_LIB = None


def _load_sputnik_python_lib():
    """Load libsputnik_python.so (build via sputnik_ext/build_binding.sh)."""
    global _SPUTNIK_LIB
    if _SPUTNIK_LIB is not None:
        return _SPUTNIK_LIB
    base = Path(__file__).resolve().parent / "sputnik_ext" / "libsputnik_python.so"
    path = os.environ.get("SPUTNIK_PYTHON_SO", str(base))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Sputnik Python binding not found at {path}. Build with:\n"
            f"  bash {base.parent / 'build_binding.sh'}"
        )
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    lib.sputnik_spmm_float.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.sputnik_spmm_float.restype = ctypes.c_int
    _SPUTNIK_LIB = lib
    return lib


def spmm_dense_reference(sparse_matrix: SparseTensor, ones: torch.Tensor) -> torch.Tensor:
    """Dense reference for benchmark SpMM: (B, la) @ (la, lb)."""
    return ones @ sparse_matrix.dense()


def sputnik_spmm_result(
    sparse_matrix: SparseTensor,
    ones: torch.Tensor,
    layera: int,
    layerb: int,
) -> torch.Tensor:
    """
    One Sputnik SpMM matching the benchmark: returns (B, lb) float32 on ``ones.device``.
    Does not consume ``sparse_matrix`` (benchmark path may ``del`` it for memory).
    """
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse

    lib = _load_sputnik_python_lib()

    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )
    coo = cpsparse.coo_matrix(
        (
            cp.asarray(sparse_matrix.values.detach().cpu().numpy()),
            cp.asarray(indices.to(dtype=torch.int32).cpu().numpy()),
        ),
        shape=sparse_matrix.matrix_shape,
    )
    csr = coo.tocsr()
    csr_t = csr.transpose().tocsr()

    m = layerb
    k = layera
    batch = int(ones.shape[0])
    n = batch
    nnz = int(csr_t.nnz)

    indptr = cp.asnumpy(csr_t.indptr)
    lengths = np.diff(indptr.astype(np.int64))
    order = np.argsort(-lengths).astype(np.int32)
    row_indices = cp.asarray(order)

    row_offsets = csr_t.indptr.astype(cp.int32)
    column_indices = csr_t.indices.astype(cp.int32)
    values = csr_t.data.astype(cp.float32)

    dense_b = cp.asarray(ones.detach().cpu().numpy(), dtype=cp.float32).T.copy(order="C")
    out = cp.zeros((m, n), dtype=cp.float32)

    stream = torch.cuda.current_stream().cuda_stream
    stream_p = ctypes.c_void_p(stream)

    err = lib.sputnik_spmm_float(
        m,
        k,
        n,
        nnz,
        row_indices.data.ptr,
        values.data.ptr,
        row_offsets.data.ptr,
        column_indices.data.ptr,
        dense_b.data.ptr,
        out.data.ptr,
        stream_p,
    )
    if err != 0:
        raise RuntimeError(f"sputnik::CudaSpmm returned cudaError_t {err}")
    torch.cuda.synchronize()
    # Kernel layout (lb, batch); reference is (batch, lb)
    return torch.from_numpy(cp.asnumpy(out).T.copy()).to(device=ones.device, dtype=torch.float32)


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
    import cupyx.scipy.sparse as cpsparse
    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )

    sparse_tensor = cpsparse.coo_matrix((
        cp.asarray(sparse_matrix.values.cpu().numpy()),
        cp.asarray(indices.to(dtype=torch.int32).cpu().numpy())),
        shape=sparse_matrix.matrix_shape
    )

    del sparse_matrix
    if csr:
        sparse_tensor = sparse_tensor.tocsr()
        print(sparse_tensor.shape, sparse_tensor.nnz)
        
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
    import cupyx.scipy.sparse as cpsparse
    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )

    sparse_tensor = cpsparse.coo_matrix((
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


def test_sputnik(
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
    """
    Sputnik sparse×dense kernel. Benchmark path computes (batch, la) @ (la, lb) sparse
    as S^T @ X^T with S sparse (lb×la), X^T dense (la×batch), then transpose.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse

    lib = _load_sputnik_python_lib()

    indices = torch.concat(
        (
            sparse_matrix.indices_a.reshape(1, -1),
            sparse_matrix.indices_b.reshape(1, -1),
        ),
        axis=0,
    )
    coo = cpsparse.coo_matrix(
        (
            cp.asarray(sparse_matrix.values.detach().cpu().numpy()),
            cp.asarray(indices.to(dtype=torch.int32).cpu().numpy()),
        ),
        shape=sparse_matrix.matrix_shape,
    )
    csr = coo.tocsr()
    del sparse_matrix
    torch.cuda.empty_cache()

    # Sputnik API: sparse (m×k) × dense (k×n). We need (B,la)@(la,lb) = ((lb,la)_sparse @ (la,B)_dense)^T.
    csr_t = csr.transpose().tocsr()
    csr = None

    m = layerb
    k = layera
    batch = int(ones.shape[0])
    n = batch
    nnz = int(csr_t.nnz)

    indptr = cp.asnumpy(csr_t.indptr)
    lengths = np.diff(indptr.astype(np.int64))
    order = np.argsort(-lengths).astype(np.int32)
    row_indices = cp.asarray(order)

    row_offsets = csr_t.indptr.astype(cp.int32)
    column_indices = csr_t.indices.astype(cp.int32)
    values = csr_t.data.astype(cp.float32)

    dense_b = cp.asarray(ones.detach().cpu().numpy(), dtype=cp.float32).T.copy(order="C")
    out = cp.zeros((m, n), dtype=cp.float32)

    stream = torch.cuda.current_stream().cuda_stream
    stream_p = ctypes.c_void_p(stream)

    def launch():
        err = lib.sputnik_spmm_float(
            m,
            k,
            n,
            nnz,
            row_indices.data.ptr,
            values.data.ptr,
            row_offsets.data.ptr,
            column_indices.data.ptr,
            dense_b.data.ptr,
            out.data.ptr,
            stream_p,
        )
        if err != 0:
            raise RuntimeError(f"sputnik::CudaSpmm returned cudaError_t {err}")

    launch()
    torch.cuda.synchronize()

    for rep in tqdm(list(range(reps)), desc="Reps sputnik", leave=False):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            launch()
        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        log.append(
            {
                "isSparse": "Sputnik",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )


