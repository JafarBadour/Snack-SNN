"""Correctness test for sparse_outer_product_multiply.

This kernel is the SNACK weight-gradient path:
    grad_values[k] = d L / d W[indices_a[k], indices_b[k]]
                   = sum_b grad_output[b, indices_b[k]] * input[b, indices_a[k]]

The reference is therefore a SUM over the batch dimension, not a mean. Earlier
revisions of this test compared against ``.mean(dim=0)``, which masked a
1/batch_size scaling bug in the kernel wrapper that under-trained SNACK
weights by a factor of B in any non-trivial-batch backward pass.
"""

from __future__ import annotations

import sys

import torch
from sparse_tensor_multiply import sparse_outer_product_multiply


def _reference_sum(left: torch.Tensor, idx_l: torch.Tensor, right: torch.Tensor, idx_r: torch.Tensor) -> torch.Tensor:
    return (left[:, idx_l] * right[:, idx_r]).sum(dim=0)


def _run_one(batch: int, left_dim: int, right_dim: int, nnz: int, seed: int = 0) -> None:
    torch.manual_seed(seed)
    left = torch.randn(batch, left_dim, device="cuda", dtype=torch.float32)
    right = torch.randn(batch, right_dim, device="cuda", dtype=torch.float32)

    idx_l_cpu = torch.randint(0, left_dim, (nnz,), dtype=torch.int64)
    idx_r_cpu = torch.randint(0, right_dim, (nnz,), dtype=torch.int64)

    expected = _reference_sum(left, idx_l_cpu.cuda(), right, idx_r_cpu.cuda())

    idx_l = idx_l_cpu.to(dtype=torch.uint16).cuda()
    idx_r = idx_r_cpu.to(dtype=torch.uint16).cuda()
    actual = sparse_outer_product_multiply(left, idx_l, right, idx_r)

    rtol, atol = 1e-4, 1e-4
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        max_abs = (actual - expected).abs().max().item()
        ratio = (actual / expected.clamp_min(1e-12)).mean().item()
        raise AssertionError(
            f"sparse_outer_product_multiply mismatch (batch={batch}, "
            f"left_dim={left_dim}, right_dim={right_dim}, nnz={nnz})\n"
            f"  max_abs={max_abs:.3e}\n"
            f"  mean(actual/expected)={ratio:.4f} "
            f"(should be ~1.0; observing 1/B would indicate the /batch_sz bug is back)"
        )
    print(
        f"[OK] batch={batch:>4d} left={left_dim:>4d} right={right_dim:>4d} nnz={nnz:>5d} "
        f"max_abs={(actual - expected).abs().max().item():.2e}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0

    # Sweep across batch sizes to make sure the wrapper does NOT divide by B.
    cases = [
        # (batch, left_dim, right_dim, nnz)
        (1, 8, 8, 16),
        (2, 8, 8, 16),
        (8, 64, 64, 256),
        (32, 128, 128, 1024),
        (128, 768, 3072, 8192),
    ]
    for case in cases:
        _run_one(*case)
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
