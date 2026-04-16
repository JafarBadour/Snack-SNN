from sparse.mult.tensor import SparseTensor, create_random_sparse_matrix
import torch
import math
from pathlib import Path
import sys


def _try_get_sputnik_module():
    """
    Try importing local Sputnik torch extension built in-place.
    Returns module or None if unavailable.
    """
    ext_dir = Path(__file__).resolve().parent / "sputnik_torch_ext"
    if ext_dir.is_dir() and str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    try:
        import sputnik_torch_ext  # type: ignore
    except ImportError:
        return None
    return sputnik_torch_ext


def _sputnik_spmm_from_sparse_tensor(sp: SparseTensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    Compute (S^T @ rhs) using Sputnik where S is SparseTensor with shape (layera, layerb).
    rhs is expected to be shape (layera, batch) to match existing benchmark conventions.
    """
    sputnik_mod = _try_get_sputnik_module()
    if sputnik_mod is None:
        raise ImportError(
            "sputnik_torch_ext is not importable. "
            "Run: bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh"
        )

    import numpy as np
    from scipy.sparse import csr_matrix

    layera, layerb = sp.matrix_shape
    row = sp.indices_b.int().cpu().numpy()
    col = sp.indices_a.int().cpu().numpy()
    data = sp.values.float().cpu().numpy()
    csr = csr_matrix((data, (row, col)), shape=(layerb, layera))

    row_offsets = torch.from_numpy(np.ascontiguousarray(csr.indptr.astype(np.int32))).cuda()
    column_indices = torch.from_numpy(np.ascontiguousarray(csr.indices.astype(np.int32))).cuda()
    values = torch.from_numpy(np.ascontiguousarray(csr.data.astype(np.float32))).cuda()
    row_indices = torch.arange(layerb, dtype=torch.int32, device="cuda")
    rhs = rhs.float().contiguous()
    return sputnik_mod.spmm(row_indices, values, row_offsets, column_indices, rhs)

def tst1():

    def get_sparse_tensor(sp):

        import torch

        indices = torch.concat((sp.indices_a.reshape(1, -1), sp.indices_b.reshape(1, -1)), axis=0)

        return torch.sparse_coo_tensor(indices, sp.values, sp.matrix_shape, device="cuda")

    sp = create_random_sparse_matrix(5000, 5000, 50)

    sp = sp.cuda()

    ones = torch.randn((32, sp.matrix_shape[0])).cuda()

    r1 = sp @ ones

    # st = get_sparse_tensor(sp)

    r2 = ones @ sp.dense()

    print(torch.abs(r2 - r1).max())


def tst2():
    torch.set_printoptions(sci_mode=False)
    sp = SparseTensor(
        indices=torch.tensor([[0, 1], [1, 1], [1, 0]]),
        values=torch.tensor([0.1, 1, 1]),
        matrix_shape=(2, 2),
    ).cuda()
    activations = (
        torch.tensor(
            [
                [
                    1,
                    1,
                ],
                [10, 10],
            ]
        )
        .float()
        .cuda()
    )

    print((sp @ activations))

    print(activations @ sp.dense())

    print(activations)
    print(sp.dense())
    print(((sp @ activations) - (activations @ sp.dense())).max())


def tst3():
    torch.set_printoptions(sci_mode=False)
    sp = create_random_sparse_matrix(3, 2, 0).cuda()
    activations = torch.rand((2, 5)).cuda()
    activations = torch.tensor([[1, 2, 3], [10, 20, 30]]).float().cuda()
    print(sp)
    r1 = sp @ activations
    r2 = activations @ sp.dense()
    print(r1)
    print(r2)
    print((r1 - r2).abs().max())


def tst4():
    sp = create_random_sparse_matrix(1000, 1000, 0).cuda()
    activations = torch.rand((1, 1000)).cuda().to(torch.float32) * math.sqrt(2)
    sp.values = sp.values.to(torch.float32) * math.sqrt(2)
    d1 = sp.dense().to(torch.float32)
    d2 = sp.dense().to(torch.float64)
    
    

    r1 =  activations @ d1
    r2 = activations.to(torch.float64) @ d2 .to(torch.float64)
    r3 = sp @ activations
    
    print("Dense float vs dense double", (r1 - r2).abs().max())
    print("Dense float vs sparse", (r1 - r3).abs().max())
    print("Dense double vs sparse", (r2 - r3).abs().max())


def tst5_sputnik_correctness():
    """
    Validate Sputnik wrapper numerics against dense and SparseTensor paths.
    """
    sputnik_mod = _try_get_sputnik_module()
    if sputnik_mod is None:
        print(
            "Sputnik wrapper not found. Build first:\n"
            "  bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh"
        )
        return

    torch.set_printoptions(sci_mode=False)
    layera, layerb, batch = 1000, 1000, 8
    sparsity = 95

    sp = create_random_sparse_matrix(layera, layerb, sparsity).cuda()
    activations = torch.rand((batch, layera), device="cuda", dtype=torch.float32) * math.sqrt(2)
    sp.values = sp.values.to(torch.float32) * math.sqrt(2)

    dense_ref = activations @ sp.dense()
    sparse_ref = sp @ activations
    rhs = activations.transpose(0, 1).contiguous()  # (layera, batch)
    sputnik_out_t = _sputnik_spmm_from_sparse_tensor(sp, rhs)  # (layerb, batch)
    sputnik_out = sputnik_out_t.transpose(0, 1).contiguous()  # (batch, layerb)

    print("Dense vs SparseTensor", (dense_ref - sparse_ref).abs().max())
    print("Dense vs Sputnik", (dense_ref - sputnik_out).abs().max())
    print("SparseTensor vs Sputnik", (sparse_ref - sputnik_out).abs().max())
    print(
        "allclose(Dense, Sputnik):",
        torch.allclose(dense_ref, sputnik_out, rtol=1e-4, atol=1e-4),
    )

if __name__ == "__main__":
    tst4()
    tst5_sputnik_correctness()
