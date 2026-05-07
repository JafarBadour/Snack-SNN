#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def get_baselines() -> list[str]:
    baselines = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir():
            continue
        if (path / "baseline.json").exists():
            baselines.append(path.name)
    return baselines


def load_priority(baseline: str) -> int:
    cfg = ROOT / baseline / "baseline.json"
    if not cfg.exists():
        return 10_000
    with cfg.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("priority", 10_000))


def run_one(baseline: str, phase: str, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "run_baseline.py"),
        "--baseline",
        baseline,
        "--phase",
        phase,
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"=== {baseline} ({phase}) ===", flush=True)
    proc = subprocess.run(cmd, text=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run setup/bench across SpMM baselines.")
    parser.add_argument("--phase", choices=["setup", "bench", "all"], default="setup")
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        help="Optional explicit baseline list. Defaults to all discovered baselines.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failures instead of stopping immediately.",
    )
    args = parser.parse_args()

    selected = args.baselines or get_baselines()
    selected = sorted(selected, key=load_priority)
    if not selected:
        print("No baselines found under experiments/spmm_exps.", file=sys.stderr)
        return 2

    failures: list[tuple[str, int]] = []
    for baseline in selected:
        rc = run_one(baseline, args.phase, args.dry_run)
        if rc != 0:
            failures.append((baseline, rc))
            if not args.keep_going:
                break

    if failures:
        print("\nFailures:")
        for name, rc in failures:
            print(f"- {name}: exit_code={rc}")
        return 1

    print("\nAll selected baselines completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
