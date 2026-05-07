#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(cmd: str, cwd: Path, env: dict[str, str], dry_run: bool) -> tuple[int, str, str]:
    if dry_run:
        return 0, "", ""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        shell=True,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def load_config(baseline_dir: Path) -> dict[str, Any]:
    cfg_path = baseline_dir / "baseline.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run setup/bench phases for one SpMM baseline.")
    parser.add_argument("--baseline", required=True, help="Baseline folder name under experiments/spmm_exps.")
    parser.add_argument("--phase", choices=["setup", "bench", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    baseline_dir = ROOT / args.baseline
    if not baseline_dir.exists():
        print(f"Missing baseline folder: {baseline_dir}", file=sys.stderr)
        return 2

    cfg = load_config(baseline_dir)
    phases: list[str] = ["setup", "bench"] if args.phase == "all" else [args.phase]

    env = os.environ.copy()
    env["SPMM_BASELINE"] = args.baseline
    env["SPMM_BASELINE_DIR"] = str(baseline_dir)

    ts = utc_now()
    for phase in phases:
        commands = cfg.get("setup_commands" if phase == "setup" else "benchmark_commands", [])
        if not commands:
            print(f"[{args.baseline}] {phase}: no commands configured; skipping")
            continue

        for idx, cmd in enumerate(commands, start=1):
            print(f"[{args.baseline}] {phase} #{idx}: {cmd}")
            rc, out, err = run_command(cmd, baseline_dir, env, args.dry_run)
            payload = {
                "timestamp_utc": ts,
                "baseline": args.baseline,
                "phase": phase,
                "index": idx,
                "command": cmd,
                "return_code": rc,
                "dry_run": args.dry_run,
                "stdout": out,
                "stderr": err,
            }
            log_path = baseline_dir / "logs" / f"{ts}_{phase}_{idx:02d}.json"
            save_log(log_path, payload)
            if rc != 0:
                print(f"Command failed ({rc}). See {log_path}", file=sys.stderr)
                return rc

    print(f"[{args.baseline}] completed phase={args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
