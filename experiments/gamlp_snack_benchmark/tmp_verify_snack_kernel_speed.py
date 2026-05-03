#!/usr/bin/env python3
"""Temporary microbenchmark: dense_mask matmul vs SNACK on artifact weights."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

import torch

from DST.initializers.uniform_initializer import UniformInitializer
from DST.layers import Snack


@dataclass
class BenchResult:
    layer_key: str
    shape: str
    density: float
    batch_size: int
    dense_mask_ms_mean: float
    dense_mask_energy_mj_mean: float
    dense_mask_memory_mb: float
    snack_ms_mean: float
    snack_energy_mj_mean: float
    snack_memory_mb: float
    speedup_snack_vs_dense_mask: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporary SNACK speed verification.")
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to dense_mask/snack artifact checkpoint (.pt or .pkl).",
    )
    p.add_argument(
        "--synthetic-shapes",
        type=str,
        nargs="+",
        default=None,
        help='Synthetic matrix shapes like "10000x10000". If set, checkpoint is not required.',
    )
    p.add_argument(
        "--synthetic-densities",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help="Densities for synthetic matrices (fraction of non-zero entries).",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: 1).",
    )
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--repeat", type=int, default=2000)
    p.add_argument(
        "--max-layers",
        type=int,
        default=12,
        help="Benchmark at most this many sparse matrices from checkpoint.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/gamlp_snack_benchmark/results/tmp_verify_snack_kernel_speed.csv"),
    )
    return p.parse_args()


class PowerReader:
    def __init__(self) -> None:
        self.backend = None
        self.nvml = None
        self.handle = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.backend = "pynvml"
        except Exception:
            self.backend = "nvidia-smi"

    def read_watts(self) -> float:
        if self.backend == "pynvml" and self.nvml is not None and self.handle is not None:
            return float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            )
            return float(result.stdout.strip().splitlines()[0])
        except Exception:
            return 0.0

    def close(self) -> None:
        if self.backend == "pynvml" and self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def collect_sparse_matrices(state_dict: Dict[str, torch.Tensor], max_layers: int) -> List[Tuple[str, torch.Tensor]]:
    mats: List[Tuple[str, torch.Tensor]] = []
    for key, w in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if not torch.is_tensor(w) or w.ndim != 2:
            continue
        mask_key = key.replace(".weight", ".mask")
        m = state_dict.get(mask_key)
        if not torch.is_tensor(m) or m.shape != w.shape:
            continue
        mats.append((key, (w * m).float().contiguous()))
    mats.sort(key=lambda x: x[0])
    return mats[:max_layers]


def parse_shape_token(token: str) -> Tuple[int, int]:
    tok = token.lower().replace(" ", "")
    if "x" not in tok:
        raise ValueError(f'Invalid shape "{token}". Expected format like 10000x10000.')
    a, b = tok.split("x", 1)
    return int(a), int(b)


def make_synthetic_sparse_matrix(in_dim: int, out_dim: int, density: float) -> torch.Tensor:
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")
    total = in_dim * out_dim
    nnz = max(1, int(round(total * density)))
    flat = torch.zeros(total, dtype=torch.float32)
    idx = torch.randperm(total)[:nnz]
    flat[idx] = torch.randn(nnz, dtype=torch.float32)
    return flat.view(in_dim, out_dim).contiguous()


def bench_dense_mask(
    x: torch.Tensor, w_sparse: torch.Tensor, warmup: int, repeat: int, power: PowerReader
) -> tuple[float, float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = x @ w_sparse
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        pw0 = power.read_watts()
        t0.record()
        for _ in range(repeat):
            _ = x @ w_sparse
        t1.record()
        torch.cuda.synchronize()
        pw1 = power.read_watts()
        ms = float(t0.elapsed_time(t1) / repeat)  # steady-state inference latency only
        energy_mj = ((pw0 + pw1) * 0.5) * ms
        mem_mb = float(torch.cuda.max_memory_allocated() / 1e6)
        return ms, energy_mj, mem_mb


def make_snack_from_sparse_weight(w_sparse: torch.Tensor) -> Snack:
    sn = Snack(
        input_size=w_sparse.shape[0],
        output_size=w_sparse.shape[1],
        sparsity=0.0,
        dense_weight=w_sparse,
        initializer=UniformInitializer,
        bias=False,
        device="cuda",
    ).cuda()
    return sn


def bench_snack(x: torch.Tensor, sn: Snack, warmup: int, repeat: int, power: PowerReader) -> tuple[float, float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = sn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        pw0 = power.read_watts()
        t0.record()
        for _ in range(repeat):
            _ = sn(x)
        t1.record()
        torch.cuda.synchronize()
        pw1 = power.read_watts()
        ms = float(t0.elapsed_time(t1) / repeat)  # steady-state inference latency only
        energy_mj = ((pw0 + pw1) * 0.5) * ms
        mem_mb = float(torch.cuda.max_memory_allocated() / 1e6)
        return ms, energy_mj, mem_mb


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    matrices: List[Tuple[str, torch.Tensor]] = []
    if args.synthetic_shapes:
        for shape_token in args.synthetic_shapes:
            in_dim, out_dim = parse_shape_token(shape_token)
            for density in args.synthetic_densities:
                key = f"synthetic.{in_dim}x{out_dim}.density_{density:.6f}"
                matrices.append((key, make_synthetic_sparse_matrix(in_dim, out_dim, density)))
        print(f"Built {len(matrices)} synthetic sparse matrices.")
    else:
        if args.checkpoint_path is None:
            raise ValueError("Provide --checkpoint-path or use --synthetic-shapes.")
        state_dict = load_state_dict(args.checkpoint_path)
        matrices = collect_sparse_matrices(state_dict, max_layers=args.max_layers)
        if not matrices:
            raise RuntimeError("No sparse matrices (.weight + .mask) found in checkpoint.")
        print(f"Found {len(matrices)} sparse matrices from artifact checkpoint.")
    rows: List[BenchResult] = []
    print("Timing mode: steady-state inference only (kernel/module initialization excluded).")
    power = PowerReader()

    try:
        for key, w_sparse_cpu in matrices:
            w_sparse = w_sparse_cpu.cuda()
            snack_layer = make_snack_from_sparse_weight(w_sparse)  # init done once, outside timed path
            in_dim, out_dim = int(w_sparse.shape[0]), int(w_sparse.shape[1])
            density = float((w_sparse != 0).float().mean().item())
            for bs in args.batch_sizes:
                x = torch.randn(bs, in_dim, device="cuda", dtype=torch.float32)
                dense_ms, dense_energy, dense_mem = bench_dense_mask(
                    x, w_sparse, warmup=args.warmup, repeat=args.repeat, power=power
                )
                snack_ms, snack_energy, snack_mem = bench_snack(
                    x, snack_layer, warmup=args.warmup, repeat=args.repeat, power=power
                )
                speedup = dense_ms / max(snack_ms, 1e-12)
                row = BenchResult(
                    layer_key=key,
                    shape=f"{in_dim}x{out_dim}",
                    density=density,
                    batch_size=bs,
                    dense_mask_ms_mean=dense_ms,
                    dense_mask_energy_mj_mean=dense_energy,
                    dense_mask_memory_mb=dense_mem,
                    snack_ms_mean=snack_ms,
                    snack_energy_mj_mean=snack_energy,
                    snack_memory_mb=snack_mem,
                    speedup_snack_vs_dense_mask=speedup,
                )
                rows.append(row)
                print(
                    f"{key:<55} bs={bs:<3} shape={in_dim}x{out_dim} dens={density:.4f} "
                    f"dense={dense_ms:.4f}ms/{dense_energy:.3f}mJ/{dense_mem:.1f}MB "
                    f"snack={snack_ms:.4f}ms/{snack_energy:.3f}mJ/{snack_mem:.1f}MB "
                    f"speedup={speedup:.3f}x"
                )
    finally:
        power.close()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_key",
                "shape",
                "density",
                "batch_size",
                "dense_mask_ms_mean",
                "dense_mask_energy_mj_mean",
                "dense_mask_memory_mb",
                "snack_ms_mean",
                "snack_energy_mj_mean",
                "snack_memory_mb",
                "speedup_snack_vs_dense_mask",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
