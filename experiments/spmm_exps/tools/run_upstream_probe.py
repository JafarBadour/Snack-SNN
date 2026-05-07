#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def discover_command(repo_dir: Path) -> list[str] | None:
    shell_candidates = [
        "scripts/benchmark.sh",
        "scripts/run_benchmark.sh",
        "benchmark.sh",
        "run_benchmark.sh",
    ]
    for rel in shell_candidates:
        cand = repo_dir / rel
        if cand.exists():
            return ["bash", str(cand)]

    py_candidates = [
        "benchmark.py",
        "bench.py",
        "scripts/benchmark.py",
        "examples/benchmark.py",
    ]
    for rel in py_candidates:
        cand = repo_dir / rel
        if cand.exists():
            return ["python", str(cand)]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Best-effort upstream benchmark probe for baselines without local adapter."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = utc_stamp()
    out_log = logs_dir / f"{stamp}_probe.out.log"
    err_log = logs_dir / f"{stamp}_probe.err.log"
    json_log = results_dir / f"{stamp}_probe.json"

    cmd = discover_command(repo_dir)
    if cmd is None:
        payload = {
            "timestamp_utc": stamp,
            "baseline": args.baseline,
            "status": "adapter_required",
            "message": "No known benchmark entrypoint discovered automatically.",
            "repo_dir": str(repo_dir),
        }
        json_log.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[{args.baseline}] no benchmark entrypoint discovered (adapter required)")
        return 0

    proc = subprocess.run(cmd, cwd=repo_dir, text=True, capture_output=True)
    out_log.write_text(proc.stdout, encoding="utf-8")
    err_log.write_text(proc.stderr, encoding="utf-8")
    payload = {
        "timestamp_utc": stamp,
        "baseline": args.baseline,
        "status": "ran_probe",
        "command": " ".join(cmd),
        "return_code": proc.returncode,
        "stdout_log": str(out_log),
        "stderr_log": str(err_log),
        "repo_dir": str(repo_dir),
    }
    json_log.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        print(f"[{args.baseline}] probe command failed; see {err_log}")
        return proc.returncode
    print(f"[{args.baseline}] probe command completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
