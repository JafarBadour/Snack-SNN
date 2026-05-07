from __future__ import annotations

import ctypes
import os
import sys
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

_flashsparse_modules = None
_sputnik_module = None
_SPUTNIK_LIB = None
_gespmm_module = None


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


def _get_sputnik_torch_ext():
    """Lazy import for the local PyTorch wrapper around upstream Sputnik."""
    global _sputnik_module
    if _sputnik_module is not None:
        return _sputnik_module
    repo_root = Path(__file__).resolve().parents[2]
    ext_dir = repo_root / "benchmark" / "sparse_matrix_multi" / "sputnik_torch_ext"
    _prepend_sys_path(str(ext_dir))
    import sputnik_torch_ext  # type: ignore  # noqa: PLC0415

    _sputnik_module = sputnik_torch_ext
    return _sputnik_module


def _get_gespmm_module():
    """
    Lazy import for GE-SpMM style CUDA extension (module names seen in forks:
    GESpMM_kernel / GESPMM). Requires the extension to be prebuilt.
    """
    global _gespmm_module
    if _gespmm_module is not None:
        return _gespmm_module

    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "third_party/FlashSparse/Baseline/GESpMM",
        "experiments/spmm_exps/flashsparse/third_party/flashsparse/Baseline/GESpMM",
    ):
        _prepend_sys_path(str((repo_root / rel).resolve()))

    for name in ("GESpMM_kernel", "GESPMM"):
        try:
            mod = __import__(name)
            _gespmm_module = mod
            return mod
        except ImportError:
            continue

    raise ImportError(
        "GE-SpMM extension not importable (GESpMM_kernel / GESPMM).\n"
        "Build it first (example):\n"
        "  cd third_party/FlashSparse/Baseline/GESpMM && python setup.py build_ext --inplace"
    )


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
    label="SparseTorch",
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
                "isSparse": label,
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

    import cupyx.scipy.sparse as cpsparse
    import cupy as cp
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

    import cupyx.scipy.sparse as cpsparse
    import cupy as cp
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
    label="Sputnik",
):
    """
    Sputnik SpMM via local torch extension.
    We benchmark the same numeric task as others: ones @ S.
    Sputnik computes A @ B with CSR(A), so we execute S.T @ ones.T and transpose.
    """
    try:
        sputnik_torch_ext = _get_sputnik_torch_ext()
    except ImportError as e:
        raise ImportError(
            "sputnik_torch_ext is not importable.\n"
            "  Build Sputnik + wrapper: bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh\n"
            "  Then export PYTHONPATH to include benchmark/sparse_matrix_multi/sputnik_torch_ext."
        ) from e

    import numpy as np
    from scipy.sparse import csr_matrix

    row = sparse_matrix.indices_b.int().cpu().numpy()
    col = sparse_matrix.indices_a.int().cpu().numpy()
    data = sparse_matrix.values.float().cpu().numpy()
    m, k = layerb, layera
    csr = csr_matrix((data, (row, col)), shape=(m, k))

    row_offsets = torch.from_numpy(np.ascontiguousarray(csr.indptr.astype(np.int32))).cuda()
    column_indices = torch.from_numpy(np.ascontiguousarray(csr.indices.astype(np.int32))).cuda()
    values = torch.from_numpy(np.ascontiguousarray(csr.data.astype(np.float32))).cuda()
    row_indices = torch.arange(m, dtype=torch.int32, device="cuda")

    rhs = ones.detach().float().transpose(0, 1).contiguous()
    assert rhs.shape == (layera, batsh_sz), (rhs.shape, (layera, batsh_sz))

    _ = sputnik_torch_ext.spmm(row_indices, values, row_offsets, column_indices, rhs)

    for rep in tqdm(list(range(reps)), desc="Reps Sputnik", leave=False):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            out = sputnik_torch_ext.spmm(
                row_indices, values, row_offsets, column_indices, rhs
            )
            _ = out / (out.abs().max().clamp(min=1e-8))
        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        log.append(
            {
                "isSparse": label,
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )


def test_cusparse_csr_library(**kwargs):
    kwargs["csr"] = True
    kwargs["label"] = "cuSPARSE CSRCSR Library"
    test_sparse_torch(**kwargs)
    kwargs["csr"] = False


def test_cusparse_coo_library(**kwargs):
    kwargs["csr"] = False
    kwargs["label"] = "cuSPARSE COOCOO Library"
    test_sparse_torch(**kwargs)


def test_sputnik_csr_dl_optimized(**kwargs):
    kwargs["label"] = "Sputnik CSR DL-optimized"
    test_sputnik(**kwargs)


def test_ge_spmm_dgsparse_csr_gnn_optimized(
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
    GE-SpMM / dgSPARSE CSR GNN-style baseline path.
    Requires a built GE-SpMM extension module (GESpMM_kernel or GESPMM).
    """
    gespmm_mod = _get_gespmm_module()

    row = sparse_matrix.indices_b.int().cpu().numpy()
    col = sparse_matrix.indices_a.int().cpu().numpy()
    data = sparse_matrix.values.float().cpu().numpy()
    m, k = layerb, layera
    csr = sp.csr_matrix((data, (row, col)), shape=(m, k))

    row_offsets = torch.from_numpy(np.ascontiguousarray(csr.indptr.astype(np.int32))).cuda()
    column_indices = torch.from_numpy(np.ascontiguousarray(csr.indices.astype(np.int32))).cuda()
    values = torch.from_numpy(np.ascontiguousarray(csr.data.astype(np.float32))).cuda()
    rhs = ones.detach().float().transpose(0, 1).contiguous()
    nnz = int(values.numel())

    # warmup
    _ = gespmm_mod.forward(row_offsets, column_indices, values, rhs, m, batsh_sz, nnz, 1, 1)

    for rep in tqdm(list(range(reps)), desc="Reps GE-SpMM/dgSPARSE", leave=False):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        t1 = time.time()
        for _ in range(100):
            out = gespmm_mod.forward(
                row_offsets, column_indices, values, rhs, m, batsh_sz, nnz, 1, 1
            )
            # forward may return tuple/list; normalize first tensor if needed
            if isinstance(out, (tuple, list)) and len(out) > 0:
                out0 = out[0]
                _ = out0 / (out0.abs().max().clamp(min=1e-8))
        t2 = time.time()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        log.append(
            {
                "isSparse": "GE-SpMM / dgSPARSE CSR GNN-optimized",
                "dense_level": dense_level,
                "sparsity_level": sparsity_level,
                "time": t2 - t1,
                "cuda_elapsed_time": elapsed_time_ms,
                "rep": rep,
                "batch_size": batsh_sz,
            }
        )


