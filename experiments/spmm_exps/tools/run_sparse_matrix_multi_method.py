#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def newest_csv(csv_dir: Path, method: str) -> Path | None:
    pattern = f"apr-13-log_mult_incl_cupy-{method}.csv"
    matches = list(csv_dir.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark.sparse_matrix_multi method and archive CSV into baseline results."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--logs-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_dir = Path(args.results_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = utc_stamp()
    stdout_log = logs_dir / f"{stamp}_{args.method}.out.log"
    stderr_log = logs_dir / f"{stamp}_{args.method}.err.log"
    meta_log = logs_dir / f"{stamp}_{args.method}.json"

    cmd = [
        "python",
        "-m",
        "benchmark.sparse_matrix_multi.test_speed_sparse_tensor_vs_dense_tensor",
        args.method,
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")

    src_csv = newest_csv(repo_root / "benchmark", args.method)
    archived_csv = None
    if src_csv is not None and src_csv.exists():
        archived_csv = results_dir / f"{stamp}_{args.method}.csv"
        shutil.copy2(src_csv, archived_csv)

    payload = {
        "timestamp_utc": stamp,
        "baseline": args.baseline,
        "method": args.method,
        "return_code": proc.returncode,
        "command": " ".join(cmd),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "source_csv": str(src_csv) if src_csv else None,
        "archived_csv": str(archived_csv) if archived_csv else None,
    }
    meta_log.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if proc.returncode != 0:
        print(f"benchmark command failed; see {stderr_log}")
        return proc.returncode
    if archived_csv is None:
        print("benchmark completed but no CSV was found to archive")
        return 3
    print(f"archived benchmark CSV -> {archived_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
