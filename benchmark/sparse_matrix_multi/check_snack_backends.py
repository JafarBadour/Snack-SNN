#!/usr/bin/env python3
"""Validate SNACK backend correctness and benchmark speed.

Compares:
  - sparse_tensor backend (SparseFunc)
  - sputnik backend (SputnikSnackFunc)

Checks:
  - Forward output correctness
  - Backward gradient correctness for input, values, bias
  - Forward+backward latency comparison
  - Sweep over kernel sizes and batch sizes
  - Sputnik-focused large batch sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

from DST.layers.Snack import SparseFunc, SputnikSnackFunc


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check SNACK backend correctness and speed.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=128, help="Rows in dense RHS for correctness check.")
    p.add_argument("--in-features", type=int, default=768)
    p.add_argument("--out-features", type=int, default=3072)
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--dense-rtol", type=float, default=1e-3)
    p.add_argument("--dense-atol", type=float, default=5e-3)
    p.add_argument(
        "--kernel-sizes",
        type=str,
        default="768x3072,3072x768,1024x4096,4096x1024,12500x12500,15000x15000,17500x17500",
        help="Comma-separated list like 768x3072,3072x768",
    )
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes for backend speed sweep.",
    )
    p.add_argument(
        "--sputnik-large-batches",
        type=str,
        default="1024,2048,3072,4096,12288",
        help="Comma-separated large batch sizes for Sputnik-only sweep.",
    )
    p.add_argument("--skip-correctness", action="store_true")
    return p.parse_args()


def parse_int_list(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def parse_kernel_sizes(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if "x" not in tok:
            raise ValueError(f"Kernel size '{tok}' must be in <in>x<out> format.")
        a, b = tok.split("x", 1)
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("Expected at least one kernel size.")
    return out


def load_sputnik_module():
    ext_dir = Path(__file__).resolve().parent / "sputnik_torch_ext"
    if ext_dir.is_dir() and str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    try:
        import sputnik_torch_ext  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sputnik_torch_ext is not importable.\n"
            "Build it with:\n"
            "  bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh\n"
            "And ensure PYTHONPATH includes benchmark/sparse_matrix_multi/sputnik_torch_ext."
        ) from exc
    return sputnik_torch_ext


def make_random_sparse_pattern(
    in_features: int, out_features: int, sparsity: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = in_features * out_features
    keep = max(1, int(round((1.0 - sparsity) * total)))
    flat_idx = torch.randperm(total, device=device)[:keep]
    # Match SNACK runtime: sparse.mult backward expects uint16 index tensors.
    indices_a = (flat_idx // out_features).to(dtype=torch.uint16)  # in-index
    indices_b = (flat_idx % out_features).to(dtype=torch.uint16)  # out-index
    values = torch.randn(keep, device=device, dtype=torch.float32)
    return indices_a, indices_b, values


def build_sputnik_cache(
    indices_a: torch.Tensor, indices_b: torch.Tensor, in_features: int, out_features: int
) -> Dict[str, torch.Tensor]:
    row = indices_b.to(dtype=torch.int64)  # out index
    col = indices_a.to(dtype=torch.int64)  # in index
    order = torch.argsort(row * in_features + col)
    row_sorted = row.index_select(0, order)
    col_sorted = col.index_select(0, order)
    counts = torch.bincount(row_sorted, minlength=out_features)
    row_offsets = torch.empty(out_features + 1, device=row.device, dtype=torch.int64)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(counts, dim=0)
    return {
        "row_indices": torch.arange(out_features, device=row.device, dtype=torch.int32),
        "row_offsets": row_offsets.to(dtype=torch.int32),
        "column_indices": col_sorted.to(dtype=torch.int32),
        "value_order": order.to(dtype=torch.int64),
    }


def run_sparse_tensor(
    x2d: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    return SparseFunc.apply(x2d, indices_a, indices_b, values, bias, in_features, out_features)


def run_sputnik(
    x2d: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
    cache: Dict[str, torch.Tensor],
    sputnik_mod,
) -> torch.Tensor:
    return SputnikSnackFunc.apply(
        x2d,
        indices_a,
        indices_b,
        values,
        bias,
        cache["row_indices"],
        cache["row_offsets"],
        cache["column_indices"],
        cache["value_order"],
        in_features,
        out_features,
        sputnik_mod,
    )


def dense_reference(
    x2d: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    dense_w = torch.zeros((in_features, out_features), device=x2d.device, dtype=x2d.dtype)
    dense_w[indices_a.to(dtype=torch.long), indices_b.to(dtype=torch.long)] = values
    return x2d @ dense_w + bias


def dense_reference_grads(
    x2d: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Closed-form gradients for loss = mean(out^2), avoids scatter autograd corner-cases.
    dense_w = torch.zeros((in_features, out_features), device=x2d.device, dtype=x2d.dtype)
    idx_a = indices_a.to(dtype=torch.long)
    idx_b = indices_b.to(dtype=torch.long)
    dense_w[idx_a, idx_b] = values
    out = x2d @ dense_w + bias

    grad_out = (2.0 / float(out.numel())) * out
    grad_input = grad_out @ dense_w.transpose(0, 1)
    grad_bias = grad_out.sum(dim=0)
    grad_w = x2d.transpose(0, 1) @ grad_out
    grad_values = grad_w[idx_a, idx_b]
    return out, grad_input, grad_values, grad_bias


def clone_inputs(
    x: torch.Tensor, values: torch.Tensor, bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_c = x.detach().clone().requires_grad_(True)
    values_c = values.detach().clone().requires_grad_(True)
    bias_c = bias.detach().clone().requires_grad_(True)
    return x_c, values_c, bias_c


def one_pass(
    backend: str,
    x: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
    cache: Dict[str, torch.Tensor],
    sputnik_mod,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_c, values_c, bias_c = clone_inputs(x, values, bias)
    if backend == "sparse_tensor":
        out = run_sparse_tensor(x_c, indices_a, indices_b, values_c, bias_c, in_features, out_features)
    elif backend == "sputnik":
        out = run_sputnik(
            x_c, indices_a, indices_b, values_c, bias_c, in_features, out_features, cache, sputnik_mod
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    loss = out.square().mean()
    loss.backward()
    assert x_c.grad is not None and values_c.grad is not None and bias_c.grad is not None
    return out.detach(), x_c.grad.detach(), values_c.grad.detach(), bias_c.grad.detach()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def allclose(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> bool:
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def benchmark_backend(
    backend: str,
    x: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
    cache: Dict[str, torch.Tensor],
    sputnik_mod,
    warmup: int,
    iters: int,
) -> float:
    if x.device.type != "cuda":
        # CPU fallback timing
        t0 = time.perf_counter()
        for _ in range(warmup + iters):
            x_c, values_c, bias_c = clone_inputs(x, values, bias)
            if backend == "sparse_tensor":
                out = run_sparse_tensor(x_c, indices_a, indices_b, values_c, bias_c, in_features, out_features)
            else:
                out = run_sputnik(
                    x_c, indices_a, indices_b, values_c, bias_c, in_features, out_features, cache, sputnik_mod
                )
            loss = out.square().mean()
            loss.backward()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / float(max(1, iters))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []

    for i in range(warmup + iters):
        x_c, values_c, bias_c = clone_inputs(x, values, bias)
        start.record()
        if backend == "sparse_tensor":
            out = run_sparse_tensor(x_c, indices_a, indices_b, values_c, bias_c, in_features, out_features)
        else:
            out = run_sputnik(
                x_c, indices_a, indices_b, values_c, bias_c, in_features, out_features, cache, sputnik_mod
            )
        loss = out.square().mean()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times_ms.append(float(start.elapsed_time(end)))
    return sum(times_ms) / float(max(1, len(times_ms)))


def benchmark_sputnik_forward_only(
    x: torch.Tensor,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    out_features: int,
    cache: Dict[str, torch.Tensor],
    sputnik_mod,
    warmup: int,
    iters: int,
) -> float:
    if x.device.type != "cuda":
        t0 = time.perf_counter()
        for _ in range(warmup + iters):
            _ = run_sputnik(
                x, indices_a, indices_b, values, bias, in_features, out_features, cache, sputnik_mod
            )
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / float(max(1, iters))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms = []
    for i in range(warmup + iters):
        start.record()
        _ = run_sputnik(x, indices_a, indices_b, values, bias, in_features, out_features, cache, sputnik_mod)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times_ms.append(float(start.elapsed_time(end)))
    return sum(times_ms) / float(max(1, len(times_ms)))


def main() -> None:
    args = parse_args()
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)
    batch_sizes = parse_int_list(args.batch_sizes)
    sputnik_large_batches = parse_int_list(args.sputnik_large_batches)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = args.device
    sputnik_mod = load_sputnik_module()
    correctness_pass = True
    correctness_ran = False
    if not args.skip_correctness:
        try:
            indices_a, indices_b, values = make_random_sparse_pattern(
                args.in_features, args.out_features, args.sparsity, device
            )
            bias = torch.randn(args.out_features, device=device, dtype=torch.float32)
            x = torch.randn(args.batch, args.in_features, device=device, dtype=torch.float32)
            cache = build_sputnik_cache(indices_a, indices_b, args.in_features, args.out_features)

            out_sparse, gx_sparse, gv_sparse, gb_sparse = one_pass(
                "sparse_tensor",
                x,
                indices_a,
                indices_b,
                values,
                bias,
                args.in_features,
                args.out_features,
                cache,
                sputnik_mod,
            )
            out_sputnik, gx_sputnik, gv_sputnik, gb_sputnik = one_pass(
                "sputnik",
                x,
                indices_a,
                indices_b,
                values,
                bias,
                args.in_features,
                args.out_features,
                cache,
                sputnik_mod,
            )

            out_dense, gx_dense, gv_dense, gb_dense = dense_reference_grads(
                x, indices_a, indices_b, values, bias, args.in_features, args.out_features
            )
            correctness_ran = True

            backend_checks = {
                "forward_sparse_vs_sputnik": allclose(out_sparse, out_sputnik, args.rtol, args.atol),
                "grad_input_sparse_vs_sputnik": allclose(gx_sparse, gx_sputnik, args.rtol, args.atol),
                "grad_values_sparse_vs_sputnik": allclose(gv_sparse, gv_sputnik, args.rtol, args.atol),
                "grad_bias_sparse_vs_sputnik": allclose(gb_sparse, gb_sputnik, args.rtol, args.atol),
            }
            dense_checks = {
                "forward_sparse_vs_dense": allclose(out_sparse, out_dense, args.dense_rtol, args.dense_atol),
                "grad_input_sparse_vs_dense": allclose(gx_sparse, gx_dense, args.dense_rtol, args.dense_atol),
                "grad_values_sparse_vs_dense": allclose(gv_sparse, gv_dense, args.dense_rtol, args.dense_atol),
                "grad_bias_sparse_vs_dense": allclose(gb_sparse, gb_dense, args.dense_rtol, args.dense_atol),
            }

            print("=== Correctness ===")
            print("-- Backend parity (sparse_tensor vs sputnik) --")
            for name, ok in backend_checks.items():
                print(f"{name}: {'PASS' if ok else 'FAIL'}")
            print("-- Dense reference parity (sparse_tensor vs dense) --")
            for name, ok in dense_checks.items():
                print(f"{name}: {'PASS' if ok else 'FAIL'}")
            print("max_abs_forward_sparse_vs_sputnik:", f"{max_abs(out_sparse, out_sputnik):.6e}")
            print("max_abs_grad_values_sparse_vs_sputnik:", f"{max_abs(gv_sparse, gv_sputnik):.6e}")
            print("max_abs_grad_input_sparse_vs_sputnik:", f"{max_abs(gx_sparse, gx_sputnik):.6e}")
            print("max_abs_grad_bias_sparse_vs_sputnik:", f"{max_abs(gb_sparse, gb_sputnik):.6e}")
            print("max_abs_grad_values_sparse_vs_dense:", f"{max_abs(gv_sparse, gv_dense):.6e}")
            # Diagnostic: best-fit scalar alpha s.t. gv_sparse ~= alpha * gv_dense.
            # This is the projection coefficient <gv_sparse, gv_dense> / <gv_dense, gv_dense>;
            # it's exactly 1.0 when the two gradients agree and exactly 1/B when the
            # historic /batch_sz bug in sparse_outer_product_multiply is back.
            # Element-wise mean(gv_sparse / gv_dense) is *not* robust here because
            # weight gradients contain many near-zero entries (high-sparsity layer)
            # and the ratio of two near-zeros explodes.
            gv_dense_inner = float((gv_dense.float() * gv_dense.float()).sum().item())
            if gv_dense_inner > 0.0:
                gv_alpha = float(
                    (gv_sparse.float() * gv_dense.float()).sum().item() / gv_dense_inner
                )
            else:
                gv_alpha = float("nan")
            print(
                f"best-fit alpha (gv_sparse ~= alpha * gv_dense): {gv_alpha:.6f} "
                "(expect ~1.0; ~1/B would mean the /batch_sz bug is back)"
            )

            backend_pass = all(backend_checks.values())
            dense_pass = all(dense_checks.values())
            # Both gates are now strict. Backend parity guards the kernel choice;
            # dense parity guards the math. The latter used to be advisory and that
            # is exactly how the /batch_sz weight-gradient bug went unnoticed.
            correctness_pass = backend_pass and dense_pass
            print("\nBACKEND_PARITY:", "PASS" if backend_pass else "FAIL")
            print("DENSE_REFERENCE_RESULT:", "PASS" if dense_pass else "FAIL")
            print("CORRECTNESS_RESULT:", "PASS" if correctness_pass else "FAIL")
        except RuntimeError as exc:
            if is_oom_error(exc):
                print(f"Correctness skipped due OOM: {str(exc).splitlines()[0]}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

    print("\n=== Speed Sweep: sparse_tensor vs sputnik (forward+backward) ===")
    print("kernel,batch,sparse_tensor_ms,sputnik_ms,sputnik_over_sparse_x")
    for in_features, out_features in kernel_sizes:
        try:
            indices_a, indices_b, values = make_random_sparse_pattern(
                in_features, out_features, args.sparsity, device
            )
            bias = torch.randn(out_features, device=device, dtype=torch.float32)
            cache = build_sputnik_cache(indices_a, indices_b, in_features, out_features)
        except RuntimeError as exc:
            if is_oom_error(exc):
                print(f"{in_features}x{out_features},*,OOM/ERR,OOM/ERR,kernel-setup-oom")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            raise
        for batch in batch_sizes:
            try:
                x = torch.randn(batch, in_features, device=device, dtype=torch.float32)
            except RuntimeError as exc:
                if is_oom_error(exc):
                    print(f"{in_features}x{out_features},{batch},OOM/ERR,OOM/ERR,input-alloc-oom")
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                raise
            sparse_ms: float | str
            sputnik_ms: float | str
            sparse_ms = "OOM/ERR"
            sputnik_ms = "OOM/ERR"
            sparse_err = ""
            sputnik_err = ""
            try:
                sparse_ms = benchmark_backend(
                    "sparse_tensor",
                    x,
                    indices_a,
                    indices_b,
                    values,
                    bias,
                    in_features,
                    out_features,
                    cache,
                    sputnik_mod,
                    args.warmup,
                    args.iters,
                )
            except RuntimeError as exc:
                sparse_err = str(exc).splitlines()[0]
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            try:
                sputnik_ms = benchmark_backend(
                    "sputnik",
                    x,
                    indices_a,
                    indices_b,
                    values,
                    bias,
                    in_features,
                    out_features,
                    cache,
                    sputnik_mod,
                    args.warmup,
                    args.iters,
                )
            except RuntimeError as exc:
                sputnik_err = str(exc).splitlines()[0]
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if isinstance(sparse_ms, float) and isinstance(sputnik_ms, float):
                sputnik_over_sparse = sparse_ms / max(1e-12, sputnik_ms)
                print(
                    f"{in_features}x{out_features},{batch},{sparse_ms:.3f},{sputnik_ms:.3f},{sputnik_over_sparse:.3f}"
                )
            else:
                reason = sparse_err or sputnik_err or "oom-or-runtime-error"
                print(f"{in_features}x{out_features},{batch},{sparse_ms},{sputnik_ms},{reason}")

    # Sputnik-specific large batch scaling (forward only).
    focus_in, focus_out = kernel_sizes[0]
    indices_a, indices_b, values = make_random_sparse_pattern(focus_in, focus_out, args.sparsity, device)
    bias = torch.randn(focus_out, device=device, dtype=torch.float32)
    cache = build_sputnik_cache(indices_a, indices_b, focus_in, focus_out)
    print(f"\n=== Sputnik Large-Batch Sweep (forward-only), kernel={focus_in}x{focus_out} ===")
    print("batch,sputnik_ms,tokens_per_s")
    for batch in sputnik_large_batches:
        x = torch.randn(batch, focus_in, device=device, dtype=torch.float32)
        try:
            ms = benchmark_sputnik_forward_only(
                x,
                indices_a,
                indices_b,
                values,
                bias,
                focus_in,
                focus_out,
                cache,
                sputnik_mod,
                args.warmup,
                args.iters,
            )
            tok_per_s = (batch * 1000.0) / max(1e-12, ms)
            print(f"{batch},{ms:.3f},{tok_per_s:.2f}")
        except RuntimeError as exc:
            print(f"{batch},OOM/ERR,{str(exc).splitlines()[0]}")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if correctness_ran and not correctness_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

