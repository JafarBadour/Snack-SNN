#!/usr/bin/env python3
"""
Generate paper-ready benchmark tables (CSV + LaTeX) from experiment outputs.

This script is designed for:
1) GPT-2 DST benchmark outputs under experiments/gpt2_dst_lm_benchmark/results
2) SpMM benchmark logs under benchmark/apr-13-log_mult_incl_cupy-test_*.csv

Outputs:
- One CSV per table
- One .tex per table (caption + label included)
- One combined all_tables.tex to copy/paste from
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Callable


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Generate CSV + LaTeX benchmark tables for paper inclusion."
    )
    p.add_argument(
        "--gpt2-results-dir",
        type=Path,
        default=repo_root / "experiments" / "gpt2_dst_lm_benchmark" / "results",
        help="Directory containing GPT-2 benchmark run outputs.",
    )
    p.add_argument(
        "--spmm-glob",
        type=str,
        default=(repo_root / "benchmark").as_posix() + "/apr-13-log_mult_incl_cupy-test_*.csv",
        help="Glob for SpMM benchmark CSV files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "benchmark" / "generated_tables",
        help="Directory to write generated CSV/TEX tables.",
    )
    p.add_argument("--target-dense-level", type=str, default="5000x5000")
    p.add_argument("--target-sparsity", type=float, default=95.0)
    p.add_argument("--training-like-batch", type=int, default=16)
    p.add_argument("--inference-like-batch", type=int, default=1)
    p.add_argument(
        "--gamlp-inference-csv",
        type=Path,
        default=repo_root / "experiments" / "gamlp_snack_benchmark" / "results" / "gamlp_inference_5k.csv",
        help="Graph/GAMLP inference CSV source.",
    )
    p.add_argument(
        "--gamlp-snack-training-csv",
        type=Path,
        default=repo_root / "experiments" / "gamlp_snack_benchmark" / "results" / "gamlp_snack_dst_training_metrics.csv",
        help="Graph/GAMLP SNACK training CSV source.",
    )
    p.add_argument(
        "--gamlp-dense-mask-training-csv",
        type=Path,
        default=repo_root / "experiments" / "gamlp_snack_benchmark" / "results" / "gamlp_dense_mask_training_metrics.csv",
        help="Graph/GAMLP dense+mask training CSV source.",
    )
    args = p.parse_args()
    args.repo_root = repo_root
    return args


def safe_float(value: str) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if not v or v.lower() in {"nan", "none"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def last_valid(values: list[float]) -> float | None:
    for v in reversed(values):
        if v is not None:
            return v
    return None


def fmt(v: float | None, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{digits}f}"


def tex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def find_run_dirs(results_dir: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for name in ("training_step_metrics.csv", "inference_metrics.csv"):
        for p in results_dir.rglob(name):
            run_dirs.add(p.parent)
    return sorted(run_dirs)


def read_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "cfg.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def summarize_training(run_dir: Path) -> list[dict]:
    path = run_dir / "training_step_metrics.csv"
    if not path.exists():
        return []
    by_model: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("model") or "").strip() or "Unknown"
            by_model.setdefault(
                model,
                {
                    "loss": [],
                    "cuda_time_ms": [],
                    "energy_mj": [],
                    "memory_mb": [],
                    "eval_perplexity": [],
                },
            )
            by_model[model]["loss"].append(safe_float(row.get("loss", "")))
            by_model[model]["cuda_time_ms"].append(safe_float(row.get("cuda_time_ms", "")))
            by_model[model]["energy_mj"].append(safe_float(row.get("energy_mj", "")))
            by_model[model]["memory_mb"].append(safe_float(row.get("memory_mb", "")))
            by_model[model]["eval_perplexity"].append(safe_float(row.get("eval_perplexity", "")))

    rows = []
    for model, m in by_model.items():
        mem_vals = [x for x in m["memory_mb"] if x is not None]
        rows.append(
            {
                "Model": model,
                "Loss": last_valid(m["loss"]),
                "Training_time_step_ms": mean([x for x in m["cuda_time_ms"] if x is not None]),
                "Energy_step_mj": mean([x for x in m["energy_mj"] if x is not None]),
                # Use median memory to reduce sensitivity to allocator spikes/outliers.
                "Memory_mb": statistics.median(mem_vals) if mem_vals else None,
                "Eval_perplexity_last": last_valid(m["eval_perplexity"]),
            }
        )
    return rows


def summarize_inference(run_dir: Path) -> list[dict]:
    path = run_dir / "inference_metrics.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "Model": (row.get("model") or "").strip() or "Unknown",
                    "Sparsity": row.get("sparsity", ""),
                    "Latency_token_ms": safe_float(row.get("latency_ms_mean", "")),
                    "Latency_p50_ms": safe_float(row.get("latency_ms_p50", "")),
                    "Latency_p95_ms": safe_float(row.get("latency_ms_p95", "")),
                    "Energy_token_mj": safe_float(row.get("energy_mj_mean", "")),
                    "Memory_mb": safe_float(row.get("memory_mb", "")),
                    "Perplexity": safe_float(row.get("perplexity", "")),
                }
            )
    return rows


def model_backend_label(model: str, cfg: dict) -> str:
    if model.strip().upper() == "SNACK":
        return cfg.get("snack_backend", "unknown")
    return "-"


def relative_experiment_name(results_dir: Path, run_dir: Path) -> str:
    rel = run_dir.relative_to(results_dir).as_posix()
    return rel


def compact_experiment_label_from_cfg(run_dir: Path, cfg: dict) -> str:
    """
    Prefer concise sparsity-based experiment labels from cfg.json.
    Falls back to a short folder-based label when cfg is missing.
    """
    s = cfg.get("sparsity", None)
    i = cfg.get("initial_sparsity", None)
    if s is not None:
        try:
            sf = float(s)
            if i is not None:
                inf = float(i)
                if abs(inf - sf) > 1e-12:
                    return f"{inf:g}->{sf:g}"
            return f"{sf:g}"
        except Exception:
            return str(s)

    # Fallback: keep short identifier only, not full nested path
    if run_dir.name.startswith("run_"):
        return run_dir.parent.name
    return run_dir.name


def normalize_target_sparsity_label(value: str | float | int | None) -> str:
    """
    Convert sparsity labels to target-only compact form.
    Examples:
      "50->75%" -> "0.75"
      "90%"     -> "0.9"
      "0.95"    -> "0.95"
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if "->" in s:
        s = s.split("->")[-1].strip()
    s = s.replace("%", "").strip()
    try:
        v = float(s)
        if v > 1.0:
            v = v / 100.0
        return f"{v:g}"
    except Exception:
        return str(value).strip()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})


def _best_by_group(
    rows: list[dict],
    group_key: str,
    metric_key: str,
    lower_is_better: bool = True,
) -> dict[tuple[str, str], bool]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for i, r in enumerate(rows):
        val = r.get(metric_key)
        if val is None:
            continue
        grouped.setdefault(str(r.get(group_key, "")), []).append((i, float(val)))
    marks: dict[tuple[str, str], bool] = {}
    for g, idx_vals in grouped.items():
        best = min(v for _, v in idx_vals) if lower_is_better else max(v for _, v in idx_vals)
        for i, v in idx_vals:
            if abs(v - best) <= 1e-12:
                marks[(g, str(i))] = True
    return marks


def make_latex_table(
    rows: list[dict],
    columns: list[str],
    caption: str,
    label: str,
    numeric_cols: set[str],
    bold_selector: Callable[[str, int], bool] | None = None,
) -> str:
    # Simple all-left/all-right layout: text left, numeric right.
    col_spec = "".join("r" if c in numeric_cols else "l" for c in columns)
    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\footnotesize")
    out.append(rf"\caption{{{caption}}}")
    out.append(rf"\label{{{label}}}")
    out.append(rf"\begin{{tabular}}{{{col_spec}}}")
    out.append(r"\hline")
    out.append(" & ".join(tex_escape(c) for c in columns) + r" \\")
    out.append(r"\hline")
    for i, r in enumerate(rows):
        if bool(r.get("__section_break_before__", False)):
            out.append(r"\hline\hline")
        cells: list[str] = []
        for c in columns:
            v = r.get(c, "")
            if c in numeric_cols:
                rendered = fmt(v, 2) if isinstance(v, (float, int)) or v is None else tex_escape(str(v))
            else:
                rendered = tex_escape(str(v))
            if bold_selector is not None and bold_selector(c, i):
                rendered = rf"\textbf{{{rendered}}}"
            cells.append(rendered)
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table*}")
    return "\n".join(out) + "\n"


def collect_gpt2_tables(args: argparse.Namespace) -> dict[str, tuple[list[dict], list[str], set[str], str, str]]:
    results_dir: Path = args.gpt2_results_dir
    train_rows: list[dict] = []
    infer_rows: list[dict] = []
    for run_dir in find_run_dirs(results_dir):
        cfg = read_cfg(run_dir)
        exp_name = compact_experiment_label_from_cfg(run_dir, cfg)
        sparsity = cfg.get("sparsity", "")
        initial_sparsity = cfg.get("initial_sparsity", "")
        for r in summarize_training(run_dir):
            row = {
                "Experiment": exp_name,
                "SNACK_backend": model_backend_label(str(r["Model"]), cfg),
                "Model": r["Model"],
                "Sparsity_target": sparsity,
                "Initial_sparsity": initial_sparsity,
                "Loss": r["Loss"],
                "Training_time_step_ms": r["Training_time_step_ms"],
                "Energy_step_mj": r["Energy_step_mj"],
                "Memory_mb": r["Memory_mb"],
                "Eval_perplexity_last": r["Eval_perplexity_last"],
            }
            train_rows.append(row)
        for r in summarize_inference(run_dir):
            row = {
                "Experiment": exp_name,
                "SNACK_backend": model_backend_label(str(r["Model"]), cfg),
                "Model": r["Model"],
                "Sparsity": r["Sparsity"],
                "Latency_token_ms": r["Latency_token_ms"],
                "Latency_p50_ms": r["Latency_p50_ms"],
                "Latency_p95_ms": r["Latency_p95_ms"],
                "Energy_token_mj": r["Energy_token_mj"],
                "Memory_mb": r["Memory_mb"],
                "Perplexity": r["Perplexity"],
            }
            infer_rows.append(row)

    def canonical_method(model: str, backend: str) -> str:
        m = (model or "").strip().lower()
        if m == "dense":
            return "dense"
        if m == "dense+mask":
            return "Dense+Mask"
        if m == "snack":
            b = (backend or "").strip().lower()
            if b == "sputnik":
                return "SNACK(Sputnik)"
            return "SNACK-COO"
        return model

    def aggregate(
        rows: list[dict],
        method_key: str,
        sparsity_key: str,
        metrics: list[str],
    ) -> list[dict]:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            key = (str(r.get(method_key, "")), str(r.get(sparsity_key, "")))
            grouped.setdefault(key, []).append(r)
        out: list[dict] = []
        for (method, sparsity), group in sorted(grouped.items()):
            row = {"Method": method, "Sparsity": sparsity}
            for metric in metrics:
                vals = [
                    float(g[metric])
                    for g in group
                    if g.get(metric) is not None
                ]
                row[metric] = statistics.fmean(vals) if vals else None
            out.append(row)
        return out

    gpt2_train_unified = []
    for r in train_rows:
        gpt2_train_unified.append(
            {
                "Method": canonical_method(str(r.get("Model", "")), str(r.get("SNACK_backend", ""))),
                "Sparsity": normalize_target_sparsity_label(r.get("Sparsity_target", "")),
                "Perplexity": r.get("Eval_perplexity_last"),
                "Memory": r.get("Memory_mb"),
                "Energy_spent": r.get("Energy_step_mj"),
                # training-step latency proxy
                "Latency_token_ms": r.get("Training_time_step_ms"),
            }
        )
    gpt2_infer_unified = []
    for r in infer_rows:
        gpt2_infer_unified.append(
            {
                "Method": canonical_method(str(r.get("Model", "")), str(r.get("SNACK_backend", ""))),
                "Sparsity": normalize_target_sparsity_label(r.get("Sparsity", "")),
                "Perplexity": r.get("Perplexity"),
                "Memory": r.get("Memory_mb"),
                "Energy_spent": r.get("Energy_token_mj"),
                "Latency_token_ms": r.get("Latency_token_ms"),
            }
        )

    gpt2_train_table = aggregate(
        gpt2_train_unified,
        method_key="Method",
        sparsity_key="Sparsity",
        metrics=["Perplexity", "Memory", "Energy_spent", "Latency_token_ms"],
    )
    # Requested: do not compare SNACK-COO in the training compact table.
    gpt2_train_table = [
        r for r in gpt2_train_table
        if str(r.get("Method", "")) != "SNACK-COO"
    ]
    gpt2_infer_table = aggregate(
        gpt2_infer_unified,
        method_key="Method",
        sparsity_key="Sparsity",
        metrics=["Perplexity", "Memory", "Energy_spent", "Latency_token_ms"],
    )
    # Requested: do not compare SNACK-COO in the inference compact table.
    gpt2_infer_table = [
        r for r in gpt2_infer_table
        if str(r.get("Method", "")) != "SNACK-COO"
    ]

    train_cols = ["Sparsity", "Method", "Perplexity", "Memory", "Energy_spent", "Latency_token_ms"]
    infer_cols = ["Sparsity", "Method", "Perplexity", "Memory", "Energy_spent", "Latency_token_ms"]
    num = {"Perplexity", "Memory", "Energy_spent", "Latency_token_ms"}

    def sparsity_sort_key(s: str) -> float:
        try:
            return float(str(s))
        except Exception:
            return float("inf")

    gpt2_train_table.sort(key=lambda r: (sparsity_sort_key(str(r.get("Sparsity", ""))), str(r.get("Method", ""))))
    gpt2_infer_table.sort(key=lambda r: (sparsity_sort_key(str(r.get("Sparsity", ""))), str(r.get("Method", ""))))
    return {
        "gpt2_training_compact": (
            gpt2_train_table,
            train_cols,
            num,
            "GPT-2 training table. Metrics are aggregated per method/sparsity: perplexity, memory, energy spent, and step latency.",
            "tab:gpt2_training_compact",
        ),
        "gpt2_inference_compact": (
            gpt2_infer_table,
            infer_cols,
            num,
            "GPT-2 inference table. Metrics are aggregated per method/sparsity: perplexity, memory, energy spent, and token latency.",
            "tab:gpt2_inference_compact",
        ),
    }


def read_spmm_rows(spmm_csv: Path) -> list[dict]:
    rows = []
    with spmm_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "Method": (row.get("isSparse") or "").strip() or spmm_csv.stem,
                    "dense_level": (row.get("dense_level") or "").strip(),
                    "sparsity_level": safe_float(row.get("sparsity_level", "")),
                    "cuda_elapsed_time": safe_float(row.get("cuda_elapsed_time", "")),
                    "batch_size": int(float(row.get("batch_size", "0"))) if row.get("batch_size") else None,
                }
            )
    return rows


def normalize_spmm_method_name(raw: str) -> str:
    """
    Align backend names with paper table naming style.
    """
    key = (raw or "").strip()
    mapping = {
        "Dense": "DenseMM(baseline)",
        "SparseUT": "SNACK-COO",
        "SparseTorch": "SpMM_COO(Torch)",
        "CuPy Sparse CSR": "SpMM_CSR(Cupy)",
        "cuSPARSE COOCOO Library": "SpMM_COO(cuSPARSE)",
        "cuSPARSE CSRCSR Library": "SpMM_CSR(cuSPARSE)",
        "Sputnik": "SpMM_CSR(Sputnik)",
        "Sputnik CSR DL-optimized": "SpMM_CSR(Sputnik-DLopt)",
        "FlashSparse": "SpMM_COO(FlashSparse)",
        "RigL": "DenseMM+Mask",
    }
    return mapping.get(key, key)


def summarize_spmm(rows: list[dict], dense_level: str, sparsity: float, batch_size: int) -> list[dict]:
    filtered = [
        r for r in rows
        if r["dense_level"] == dense_level
        and r["sparsity_level"] is not None
        and abs(float(r["sparsity_level"]) - float(sparsity)) <= 1e-9
        and r["batch_size"] == batch_size
        and r["cuda_elapsed_time"] is not None
    ]
    by_method: dict[str, list[float]] = {}
    for r in filtered:
        by_method.setdefault(r["Method"], []).append(float(r["cuda_elapsed_time"]))

    dense_mean = None
    if "Dense" in by_method and by_method["Dense"]:
        dense_mean = statistics.fmean(by_method["Dense"])

    out = []
    for method, vals in sorted(by_method.items()):
        m = statistics.fmean(vals)
        speedup = (dense_mean / m) if (dense_mean is not None and m > 0) else None
        out.append(
            {
                "Method": normalize_spmm_method_name(method),
                "Dense_level": dense_level,
                "Sparsity_percent": sparsity,
                "Batch_size": batch_size,
                "CUDA_time_us_mean": m,
                "Speedup_vs_Dense": speedup,
            }
        )
    return out


def collect_spmm_tables(args: argparse.Namespace) -> dict[str, tuple[list[dict], list[str], set[str], str, str]]:
    def read_gamlp_training(path: Path, method_label: str) -> list[dict]:
        out: list[dict] = []
        if not path.exists():
            return out
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(
                    {
                        "Method": method_label,
                        "Sparsity": normalize_target_sparsity_label(row.get("sparsity")),
                        "Loss": safe_float(row.get("loss_mean", "")),
                        "Train_acc": safe_float(row.get("train_acc", "")),
                        "Val_acc": safe_float(row.get("val_acc", "")),
                        "Memory": safe_float(row.get("peak_allocated_mb", "")),
                        "Energy_spent": safe_float(row.get("step_energy_mj_mean", "")),
                        "Step_latency_ms": safe_float(row.get("step_latency_ms_mean", "")),
                    }
                )
        return out

    def read_gamlp_inference(path: Path) -> list[dict]:
        out: list[dict] = []
        if not path.exists():
            return out
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = (row.get("type") or "").strip().lower()
                if t == "dense":
                    method = "dense"
                elif t == "dense_mask":
                    method = "Dense+Mask"
                elif t == "snack":
                    method = "SNACK-COO"
                else:
                    continue
                out.append(
                    {
                        "Method": method,
                        "Sparsity": normalize_target_sparsity_label(row.get("sparsity")),
                        "Accuracy": safe_float(row.get("accuracy", "")),
                        "Agreement_vs_baseline": safe_float(row.get("agreement_vs_baseline", "")),
                        "Memory": safe_float(row.get("memory_mb", "")),
                        "Energy_spent": safe_float(row.get("energy_mj_mean", "")),
                        "Inference_latency_ms": safe_float(row.get("latency_ms_mean", "")),
                    }
                )
        return out

    def aggregate(rows: list[dict], metrics: list[str]) -> list[dict]:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            key = (str(r.get("Method", "")), str(r.get("Sparsity", "")))
            grouped.setdefault(key, []).append(r)
        out: list[dict] = []
        for (method, sparsity), group in sorted(grouped.items()):
            row = {"Method": method, "Sparsity": sparsity}
            for metric in metrics:
                vals = [g[metric] for g in group if g.get(metric) is not None]
                if not vals:
                    row[metric] = None
                elif metric in {"Memory"}:
                    row[metric] = statistics.median(vals)
                else:
                    row[metric] = statistics.fmean(vals)
            out.append(row)
        return out

    train_raw = []
    train_raw.extend(read_gamlp_training(args.gamlp_snack_training_csv, "SNACK(Sputnik)"))
    train_raw.extend(read_gamlp_training(args.gamlp_dense_mask_training_csv, "Dense+Mask"))
    infer_raw = read_gamlp_inference(args.gamlp_inference_csv)

    train_metrics = ["Loss", "Train_acc", "Val_acc", "Memory", "Energy_spent", "Step_latency_ms"]
    infer_metrics = ["Accuracy", "Agreement_vs_baseline", "Memory", "Energy_spent", "Inference_latency_ms"]

    train_unified = aggregate(train_raw, train_metrics)
    # Requested: remove SNACK rows from graph training table (too slow/irrelevant for this view).
    train_unified = [
        r for r in train_unified
        if str(r.get("Method", "")) not in {"SNACK(Sputnik)", "SNACK-COO"}
    ]
    infer_unified = aggregate(infer_raw, infer_metrics)

    # Keep paper-facing method sets explicit for each graph table.
    expected_methods_training = [
        "dense",
        "Dense+Mask",
    ]
    expected_methods_inference = [
        "Dense+Mask",
        "SNACK-COO",
    ]

    def fill_expected(rows: list[dict], expected_methods: list[str]) -> list[dict]:
        sparsities = sorted({str(r.get("Sparsity", "")) for r in rows if str(r.get("Sparsity", ""))})
        if not sparsities:
            sparsities = [f"{args.target_sparsity:g}"]
        existing = {(str(r.get("Method", "")), str(r.get("Sparsity", ""))) for r in rows}
        out = list(rows)
        for s in sparsities:
            for m in expected_methods:
                if (m, s) not in existing:
                    out.append(
                        {
                            "Method": m,
                            "Sparsity": s,
                        "Loss": None,
                        "Train_acc": None,
                        "Val_acc": None,
                        "Accuracy": None,
                        "Agreement_vs_baseline": None,
                            "Memory": None,
                            "Energy_spent": None,
                        "Step_latency_ms": None,
                        "Inference_latency_ms": None,
                        }
                    )
        out.sort(key=lambda r: (float(r["Sparsity"]) if str(r["Sparsity"]).replace(".", "", 1).isdigit() else 999.0, expected_methods.index(r["Method"]) if r["Method"] in expected_methods else 999))
        return out

    train_unified = fill_expected(train_unified, expected_methods_training)

    # Keep a single dense baseline row, appended at the end of the same table.
    dense_rows = [r for r in infer_unified if str(r.get("Method", "")) == "dense"]
    target_sparsity_label = normalize_target_sparsity_label(str(args.target_sparsity))
    dense_ref = next((r for r in dense_rows if str(r.get("Sparsity", "")) == target_sparsity_label), None)
    if dense_ref is None and dense_rows:
        dense_rows_sorted = sorted(
            dense_rows,
            key=lambda r: float(str(r.get("Sparsity", "nan")))
            if str(r.get("Sparsity", "")).replace(".", "", 1).isdigit()
            else 999.0,
        )
        dense_ref = dense_rows_sorted[0]

    dense_baseline_row = None
    if dense_ref is not None:
        dense_baseline_row = {
            "Method": "dense",
            "Sparsity": "baseline",
            "Memory": dense_ref.get("Memory"),
            "Energy_spent": dense_ref.get("Energy_spent"),
            "Inference_latency_ms": dense_ref.get("Inference_latency_ms"),
            "__section_break_before__": True,
        }

    infer_unified = [r for r in infer_unified if str(r.get("Method", "")) != "dense"]
    infer_unified = fill_expected(infer_unified, expected_methods_inference)
    if dense_baseline_row is not None:
        infer_unified.append(dense_baseline_row)

    train_cols = ["Method", "Sparsity", "Loss", "Train_acc", "Val_acc", "Memory", "Energy_spent", "Step_latency_ms"]
    train_num = {"Loss", "Train_acc", "Val_acc", "Memory", "Energy_spent", "Step_latency_ms"}
    infer_cols = ["Method", "Sparsity", "Memory", "Energy_spent", "Inference_latency_ms"]
    infer_num = {"Memory", "Energy_spent", "Inference_latency_ms"}
    return {
        "spmm_training_compact": (
            train_unified,
            train_cols,
            train_num,
            "Graph/GAMLP training table from gamlp_snack_dst_training_metrics.csv and gamlp_dense_mask_training_metrics.csv.",
            "tab:spmm_training_compact",
        ),
        "spmm_inference_compact": (
            infer_unified,
            infer_cols,
            infer_num,
            "Graph/GAMLP inference table from gamlp_inference_5k.csv.",
            "tab:spmm_inference_compact",
        ),
    }


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {}
    tables.update(collect_gpt2_tables(args))
    tables.update(collect_spmm_tables(args))

    combined_parts = [
        "% Auto-generated by benchmark/generate_paper_tables.py",
        "% Contains table environments ready to paste into paper.",
        "",
    ]

    for name, (rows, columns, numeric_cols, caption, label) in tables.items():
        csv_path = out_dir / f"{name}.csv"
        tex_path = out_dir / f"{name}.tex"
        write_csv(csv_path, rows, columns)

        # Highlight best latency and memory where available.
        speed_marks = {}
        memory_marks = {}
        if name.startswith("gpt2_"):
            speed_marks = _best_by_group(rows, "Sparsity", "Latency_token_ms", lower_is_better=True)
            memory_marks = _best_by_group(rows, "Sparsity", "Memory", lower_is_better=True)
        elif name == "spmm_training_compact":
            speed_marks = _best_by_group(rows, "Sparsity", "Step_latency_ms", lower_is_better=True)
            memory_marks = _best_by_group(rows, "Sparsity", "Memory", lower_is_better=True)
        elif name == "spmm_inference_compact":
            speed_marks = _best_by_group(rows, "Sparsity", "Inference_latency_ms", lower_is_better=True)
            memory_marks = _best_by_group(rows, "Sparsity", "Memory", lower_is_better=True)

        def bold_selector(col: str, idx: int) -> bool:
            if name.startswith("gpt2_"):
                return (
                    (col == "Latency_token_ms" and speed_marks.get((str(rows[idx].get("Sparsity", "")), str(idx)), False))
                    or (col == "Memory" and memory_marks.get((str(rows[idx].get("Sparsity", "")), str(idx)), False))
                )
            if name == "spmm_training_compact":
                return (
                    (col == "Step_latency_ms" and speed_marks.get((str(rows[idx].get("Sparsity", "")), str(idx)), False))
                    or (col == "Memory" and memory_marks.get((str(rows[idx].get("Sparsity", "")), str(idx)), False))
                )
            if name == "spmm_inference_compact":
                if str(rows[idx].get("Method", "")) == "dense":
                    return False
                return (
                    (col == "Inference_latency_ms" and speed_marks.get((str(rows[idx].get("Sparsity", "")), str(idx)), False))
                    or (col == "Memory" and memory_marks.get((str(rows[idx].get("Sparsity", "")), str(idx)), False))
                )
            return False

        tex = make_latex_table(
            rows=rows,
            columns=columns,
            caption=caption,
            label=label,
            numeric_cols=numeric_cols,
            bold_selector=bold_selector,
        )
        tex_path.write_text(tex)
        combined_parts.append(f"% ---- {name} ----")
        combined_parts.append(tex)

    combined_path = out_dir / "all_tables.tex"
    combined_path.write_text("\n".join(combined_parts))

    print(f"Wrote tables to: {out_dir}")
    for p in sorted(out_dir.glob("*.csv")):
        print(f"  CSV: {p.name}")
    for p in sorted(out_dir.glob("*.tex")):
        print(f"  TEX: {p.name}")


if __name__ == "__main__":
    main()

