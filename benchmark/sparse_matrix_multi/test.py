"""
Ad-hoc SpMM checks (max abs diff). Same style for every ``tst*``:
print tensors or a single max-abs line vs dense ``ones @ W``.

Structured unittest (same math): ``test_spmm_correctness.py``.
"""
import math
import os
import sys
from pathlib import Path

import torch

from sparse.mult.tensor import SparseTensor, create_random_sparse_matrix


def _ensure_repo_root_on_path() -> None:
    """So ``python benchmark/sparse_matrix_multi/test.py`` can import ``benchmark.*``."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


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


def tst5():
    """
    Benchmark backends vs dense reference ``ones @ sp.dense()`` (max abs diff), like tst1/tst3.
    Torch + CuPy + optional Sputnik (needs built ``sputnik_ext/libsputnik_python.so``).
    """
    torch.manual_seed(0)
    layera, layerb = 256, 128
    batch = 8
    sp = create_random_sparse_matrix(layera, layerb, 90.0).cuda()
    sp.values = sp.values.float()
    ones = torch.randn(batch, layera, device="cuda", dtype=torch.float32)
    ref = ones @ sp.dense()

    print("SparseUT vs dense", (sp @ ones - ref).abs().max().item())

    indices = torch.concat(
        (sp.indices_a.reshape(1, -1), sp.indices_b.reshape(1, -1)),
        axis=0,
    )
    for label, use_csr in [("Torch sparse COO", False), ("Torch sparse CSR", True)]:
        st = torch.sparse_coo_tensor(
            indices, sp.values, sp.matrix_shape, device="cuda", dtype=sp.values.dtype
        )
        if use_csr:
            st = st.to_sparse_csr()
        print(f"{label} vs dense", (ones @ st - ref).abs().max().item())

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse
    except ImportError as e:
        print("CuPy (skip)", e)
    else:
        for label, use_csr in [("CuPy COO", False), ("CuPy CSR", True)]:
            coo = cpsparse.coo_matrix(
                (
                    cp.asarray(sp.values.cpu().numpy()),
                    cp.asarray(indices.to(dtype=torch.int32).cpu().numpy()),
                ),
                shape=sp.matrix_shape,
            )
            mat = coo.tocsr() if use_csr else coo
            y = cp.asarray(ones.cpu().numpy()) @ mat
            got = torch.from_numpy(cp.asnumpy(y)).to(device=ones.device, dtype=ones.dtype)
            print(f"{label} vs dense", (got - ref).abs().max().item())

    so = os.environ.get(
        "SPUTNIK_PYTHON_SO",
        str(Path(__file__).resolve().parent / "sputnik_ext" / "libsputnik_python.so"),
    )
    if not os.path.isfile(so):
        print("Sputnik (skip): no", so)
        return

    _ensure_repo_root_on_path()
    from benchmark.sparse_matrix_multi.__test_methods import sputnik_spmm_result

    got = sputnik_spmm_result(sp, ones, layera, layerb)
    print("Sputnik vs dense", (got - ref).abs().max().item())


if __name__ == "__main__":
    tst5()  # set to tst1 … tst4 for the older one-off checks
