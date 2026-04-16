from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import time
from tqdm import tqdm
from sparse.mult.tensor import SparseTensor
import torch
import scipy.sparse as sp

_flashsparse_modules = None


def _prepend_sys_path(p: str) -> None:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _flashsparse_package_dir(base: Path) -> Path | None:
    if (base / "setup.py").exists():
        return base
    inner = base / "FlashSparse"
    if (inner / "setup.py").exists():
        return inner
    return None


def _flashsparse_add_search_paths() -> None:
    """Put dirs that may contain FS_SpMM*.so / FS_Block_gpu*.so on sys.path."""
    repo_root = Path(__file__).resolve().parents[2]

    for entry in os.environ.get("FLASHSPARSE_PYTHONPATH", "").split(os.pathsep):
        _prepend_sys_path(entry.strip())

    roots: list[Path] = []
    raw = os.environ.get("FLASHSPARSE_ROOT", "").strip()
    if raw:
        roots.append(Path(raw).expanduser().resolve())

    for rel in (
        "third_party/FlashSparse/FlashSparse",
        "third_party/FlashSparse",
        "FlashSparse/FlashSparse",
        "FlashSparse",
        "vendor/FlashSparse/FlashSparse",
    ):
        roots.append((repo_root / rel).resolve())

    seen: set[str] = set()
    for base in roots:
        if not base.is_dir():
            continue
        pkg = _flashsparse_package_dir(base)
        if pkg is None:
            continue
        key = str(pkg.resolve())
        if key in seen:
            continue
        seen.add(key)
        _prepend_sys_path(key)
        for so in pkg.glob("FS_SpMM*.so"):
            _prepend_sys_path(str(so.parent))
        for so in pkg.glob("FS_Block_gpu*.so"):
            _prepend_sys_path(str(so.parent))
        bdir = pkg / "build"
        if bdir.is_dir():
            for so in bdir.rglob("FS_SpMM*.so"):
                _prepend_sys_path(str(so.parent))
            for so in bdir.rglob("FS_Block_gpu*.so"):
                _prepend_sys_path(str(so.parent))


def _get_flashsparse():
    """Lazy import for ParCIS FlashSparse (FS_SpMM + FS_Block_gpu). See https://github.com/ParCIS/FlashSparse"""
    global _flashsparse_modules
    if _flashsparse_modules is not None:
        return _flashsparse_modules
    _flashsparse_add_search_paths()
    import FS_SpMM  # type: ignore  # noqa: PLC0415
    import FS_Block_gpu  # type: ignore  # noqa: PLC0415

    _flashsparse_modules = (FS_SpMM, FS_Block_gpu)
    return _flashsparse_modules


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


def test_flashsparse(
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
    ParCIS FlashSparse TF32 SpMM (Swap-and-Transpose / TC path), PPoPP 2025.
    Same numeric task as test_dense / test_sparse_torch: Y = ones @ S with
    S (layera, layera) in our square configs — implemented as S.T @ ones.T.

    Build: clone https://github.com/ParCIS/FlashSparse, then from repo
    ``FlashSparse/`` run ``bash compile.sh`` (or ``pip install -e .`` in that folder).
    Optional: set FLASHSPARSE_PYTHONPATH if the .so modules are not discoverable.
    """
    try:
        FS_SpMM, FS_Block_gpu = _get_flashsparse()
    except ImportError as e:
        repo = Path(__file__).resolve().parents[2]
        raise ImportError(
            "FlashSparse extensions FS_SpMM / FS_Block_gpu not importable.\n"
            "  One-shot:  bash benchmark/sparse_matrix_multi/install_flashsparse.sh\n"
            "  Manual:    git clone --recursive https://github.com/ParCIS/FlashSparse.git "
            f"{repo / 'third_party' / 'FlashSparse'}\n"
            "             cd that repo’s inner FlashSparse/ (folder with setup.py) && pip install -e .\n"
            "  Or set     FLASHSPARSE_ROOT=/path/to/clone   or   FLASHSPARSE_PYTHONPATH=/dir/with/the.so\n"
            "  (needs CUDA toolkit + GPU matching their build; see upstream README)."
        ) from e

    import numpy as np
    from scipy.sparse import csr_matrix

    # (B, layera) @ S(layera, layerb)  ==  (S.T @ ones.T).T  with S.T (layerb, layera)
    row = sparse_matrix.indices_b.int().cpu().numpy()
    col = sparse_matrix.indices_a.int().cpu().numpy()
    data = sparse_matrix.values.float().cpu().numpy()
    m, k = layerb, layera
    csr = csr_matrix((data, (row, col)), shape=(m, k))

    num_nodes_ori = m
    num_nodes = m if m % 8 == 0 else m + (8 - (m % 8))
    if num_nodes > m:
        old_indptr = csr.indptr.astype(np.int32, copy=False)
        new_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
        new_indptr[: m + 1] = old_indptr
        nnz = int(old_indptr[-1])
        new_indptr[m + 1 :] = nnz
        csr = csr_matrix((csr.data, csr.indices, new_indptr), shape=(num_nodes, k))

    row_ptr = torch.from_numpy(np.ascontiguousarray(csr.indptr.astype(np.int32)))
    col_idx = torch.from_numpy(np.ascontiguousarray(csr.indices.astype(np.int32)))
    vals = torch.from_numpy(np.ascontiguousarray(csr.data.astype(np.float32)))

    num_edges = int(csr.nnz)
    window, wide = 8, 4

    row_ptr, col_idx, tc_values, _pre_ms = FS_Block_gpu.preprocess_gpu_fs(
        row_ptr, col_idx, num_nodes, num_edges, window, wide
    )

    rhs = ones.detach().float().cpu().transpose(0, 1).contiguous()
    assert rhs.shape == (layera, batsh_sz), (rhs.shape, (layera, batsh_sz))

    def _to_cpu_int32(t):
        if t.dtype != torch.int32:
            t = t.to(torch.int32)
        if t.is_cuda:
            t = t.cpu()
        return t.contiguous()

    def _to_cpu_float(t):
        t = t.float()
        if t.is_cuda:
            t = t.cpu()
        return t.contiguous()

    row_ptr = _to_cpu_int32(row_ptr)
    col_idx = _to_cpu_int32(col_idx)
    tc_values = _to_cpu_float(tc_values)

    _ = FS_SpMM.forward_tf32(
        row_ptr,
        col_idx,
        tc_values,
        rhs,
        num_nodes,
        rhs.shape[1],
        num_nodes_ori,
        1,
    )

    for rep in tqdm(list(range(reps)), desc="Reps FlashSparse", leave=False):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            out, _inner_ms = FS_SpMM.forward_tf32(
                row_ptr,
                col_idx,
                tc_values,
                rhs,
                num_nodes,
                rhs.shape[1],
                num_nodes_ori,
                1,
            )
            _ = out / (out.abs().max().clamp(min=1e-8))
        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        log.append(
            {
                "isSparse": "FlashSparse",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )


