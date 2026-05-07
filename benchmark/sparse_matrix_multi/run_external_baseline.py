#!/usr/bin/env python3
"""
Centralized external baseline launcher for spmm_exps baselines that do not yet
run through test_speed_sparse_tensor_vs_dense_tensor.py method adapters.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BaselineSpec:
    command: str
    cwd: str
    artifacts: list[str]
    needs_nvcc: bool = False
    required_env: list[str] | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def has_nvcc() -> bool:
    return shutil.which("nvcc") is not None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_matching_artifacts(cwd: Path, patterns: list[str], out_dir: Path) -> list[str]:
    copied: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in cwd.glob(pattern):
            if src.is_dir():
                continue
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def build_specs(repo_root: Path, baseline: str) -> BaselineSpec:
    base = repo_root / "experiments" / "spmm_exps" / baseline / "third_party" / baseline
    specs: dict[str, BaselineSpec] = {
        "flash-llm": BaselineSpec(
            command='bash -lc \'cd kernel_benchmark && { [ -f test_env ] && source test_env || true; make -j"$(nproc)" || true; bash benchmark.sh || true; }\'',
            cwd=str(base),
            artifacts=["kernel_benchmark/*.csv", "kernel_benchmark/*.png"],
            needs_nvcc=True,
        ),
        "spinfer": BaselineSpec(
            command='bash -lc \'cd kernel_benchmark && { [ -f test_env ] && source test_env || true; make -j"$(nproc)" || true; bash benchmark.sh || true; }\'',
            cwd=str(base),
            artifacts=["kernel_benchmark/*.csv", "kernel_benchmark/*.png"],
            needs_nvcc=True,
        ),
        "smat": BaselineSpec(
            command='bash -lc \'cd src && make -j"$(nproc)" || true; cd src && bash run_smat.sh || true\'',
            cwd=str(base),
            artifacts=["src/log/*.log", "src/ncu/*.ncu-rep"],
            needs_nvcc=True,
        ),
        "venom": BaselineSpec(
            command='bash -lc \'mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DCUDA_ARCHS="${CUDA_ARCHS:-86}" -DBASELINE=OFF -DIDEAL_KERNEL=OFF -DOUT_32B=OFF || true; make -j"$(nproc)" || true; cd ..; bash "${VENOM_BENCH_SCRIPT:-benchmark/run_baseline_a.sh}" || true\'',
            cwd=str(base),
            artifacts=["result/*.csv", "result/*.pdf", "benchmark/*.csv"],
            needs_nvcc=True,
        ),
        "sparta": BaselineSpec(
            command='bash -lc \'python test/bench/matmul/matmul.py || true\'',
            cwd=str(base),
            artifacts=["test/bench/matmul/latency.csv", "test/bench/matmul/latency.png"],
            needs_nvcc=False,
        ),
        "wanda": BaselineSpec(
            command='bash -lc \'python main.py --model "$WANDA_MODEL" --prune_method wanda --sparsity_ratio "${WANDA_SPARSITY:-0.5}" --sparsity_type unstructured --save "$SPMM_RESULTS_DIR/wanda_out" || true\'',
            cwd=str(base),
            artifacts=[],
            required_env=["WANDA_MODEL"],
        ),
        "sparsegpt": BaselineSpec(
            command='bash -lc \'python opt.py "$SPARSEGPT_MODEL" c4 --sparsity "${SPARSEGPT_SPARSITY:-0.5}" --save "$SPMM_RESULTS_DIR/sparsegpt_out" || true\'',
            cwd=str(base),
            artifacts=[],
            required_env=["SPARSEGPT_MODEL"],
        ),
        "rigl": BaselineSpec(
            command='bash -lc \'if [ -n "${RIGL_BENCH_CMD:-}" ]; then bash -lc "$RIGL_BENCH_CMD"; else python -m rigl.sparse_optimizers_test || true; python -m rigl.sparse_utils_test || true; fi\'',
            cwd=str(base),
            artifacts=[],
            needs_nvcc=False,
        ),
    }
    if baseline not in specs:
        raise KeyError(f"Unsupported external baseline: {baseline}")
    return specs[baseline]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external baseline benchmark from benchmark/sparse_matrix_multi.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = utc_stamp()
    meta_path = logs_dir / f"{stamp}_{args.baseline}_external_meta.json"
    out_log = logs_dir / f"{stamp}_{args.baseline}_external.out.log"
    err_log = logs_dir / f"{stamp}_{args.baseline}_external.err.log"

    spec = build_specs(repo_root, args.baseline)
    cwd = Path(spec.cwd)

    if spec.needs_nvcc and not has_nvcc():
        payload = {
            "timestamp_utc": stamp,
            "baseline": args.baseline,
            "status": "skipped_no_nvcc",
            "cwd": str(cwd),
        }
        write_json(meta_path, payload)
        print(f"[{args.baseline}] nvcc not found; skipping")
        return 0

    missing_env = [k for k in (spec.required_env or []) if not os.environ.get(k)]
    if missing_env:
        payload = {
            "timestamp_utc": stamp,
            "baseline": args.baseline,
            "status": "skipped_missing_env",
            "missing_env": missing_env,
            "cwd": str(cwd),
        }
        write_json(meta_path, payload)
        print(f"[{args.baseline}] missing env vars: {', '.join(missing_env)}; skipping")
        return 0

    env = os.environ.copy()
    env["SPMM_RESULTS_DIR"] = str(results_dir)

    proc = subprocess.run(
        spec.command,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
        env=env,
    )
    out_log.write_text(proc.stdout, encoding="utf-8")
    err_log.write_text(proc.stderr, encoding="utf-8")

    copied = copy_matching_artifacts(cwd, spec.artifacts, results_dir)
    payload = {
        "timestamp_utc": stamp,
        "baseline": args.baseline,
        "status": "completed" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "cwd": str(cwd),
        "command": spec.command,
        "stdout_log": str(out_log),
        "stderr_log": str(err_log),
        "copied_artifacts": copied,
    }
    write_json(meta_path, payload)

    if proc.returncode != 0:
        print(f"[{args.baseline}] external benchmark failed; see {err_log}")
        return proc.returncode
    print(f"[{args.baseline}] external benchmark completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
