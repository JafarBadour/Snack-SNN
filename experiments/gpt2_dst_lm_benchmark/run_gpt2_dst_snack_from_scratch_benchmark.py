#!/usr/bin/env python3
"""DST benchmark for GPT-2 from scratch: Dense vs Dense+Mask vs SNACK."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.pytorch_utils import Conv1D

from DST.initializers.uniform_initializer import UniformInitializer
from DST.layers import Snack
from DST.pruner_grower import ZetaPrunerGrower


@dataclass
class StepMetric:
    model: str
    step: int
    loss: float
    cuda_time_ms: float
    power_w: float
    energy_mj: float
    memory_mb: float


@dataclass
class InferenceMetric:
    model: str
    sparsity: str
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    energy_mj_mean: float
    memory_mb: float
    perplexity: float


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 from-scratch DST benchmark with SNACK.")
    p.add_argument("--model-size", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "custom"])
    p.add_argument("--tokenizer-name", type=str, default="gpt2")
    p.add_argument("--dataset-name", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--dataset-train-split", type=str, default="train")
    p.add_argument("--dataset-eval-split", type=str, default="validation")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--custom-n-layer", type=int, default=12)
    p.add_argument("--custom-n-head", type=int, default=12)
    p.add_argument("--custom-n-embd", type=int, default=768)
    p.add_argument("--sparsity", type=float, default=0.9)
    p.add_argument("--dst-interval", type=int, default=100)
    p.add_argument("--dst-zeta", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-train-steps", type=int, default=500)
    p.add_argument("--max-eval-batches", type=int, default=100)
    p.add_argument("--inference-steps", type=int, default=500)
    p.add_argument("--inference-warmup", type=int, default=50)
    p.add_argument("--prompt", type=str, default="The history of artificial intelligence")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/gpt2_dst_lm_benchmark/results/gpt2_from_scratch"),
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    p.add_argument("--disable-output-hash", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat = weight.detach().abs().reshape(-1)
    keep = max(1, int(round(flat.numel() * (1.0 - sparsity))))
    idx = torch.topk(flat, k=keep, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=weight.dtype)
    mask[idx] = 1.0
    return mask.reshape_as(weight)


class DenseMaskConv1D(nn.Module):
    def __init__(self, conv: Conv1D, sparsity: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=True)  # [in, out]
        self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=True)
        self.register_buffer("mask", make_mask(self.weight.detach(), sparsity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight * self.mask) + self.bias

    def apply_mask_(self) -> None:
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def dst_step(self, zeta: float) -> None:
        with torch.no_grad():
            w = self.weight
            m = self.mask
            flat_w = w.view(-1)
            flat_m = m.view(-1)
            active = flat_m > 0
            active_idx = torch.where(active)[0]
            active_count = int(active_idx.numel())
            if active_count <= 1:
                return
            prune_count = max(1, int(round(zeta * active_count)))
            prune_count = min(prune_count, active_count - 1)
            scores = flat_w[active_idx].abs()
            prune_rel = torch.topk(scores, k=prune_count, largest=False, sorted=False).indices
            prune_idx = active_idx[prune_rel]
            inactive_idx = torch.where(~active)[0]
            grow_count = min(prune_count, int(inactive_idx.numel()))
            grow_idx = inactive_idx[torch.randperm(inactive_idx.numel(), device=inactive_idx.device)[:grow_count]]
            flat_m[prune_idx] = 0
            flat_w[prune_idx] = 0
            if grow_count > 0:
                bound = math.sqrt(6.0 / float(w.shape[0] + w.shape[1]))
                flat_m[grow_idx] = 1
                flat_w[grow_idx] = torch.empty(grow_count, device=w.device).uniform_(-bound, bound)


class SnackConv1D(nn.Module):
    def __init__(self, conv: Conv1D, sparsity: float, device: str) -> None:
        super().__init__()
        w = conv.weight.detach().clone()
        m = make_mask(w, sparsity)
        w_sparse = (w * m).contiguous()
        self.snack = Snack(
            input_size=w_sparse.shape[0],
            output_size=w_sparse.shape[1],
            sparsity=0.0,
            dense_weight=w_sparse,
            initializer=UniformInitializer,
            bias=True,
            device=device,
        ).to(device)
        self.snack.indices_a.data = self.snack.indices_a.data.to(dtype=torch.uint16)
        self.snack.indices_b.data = self.snack.indices_b.data.to(dtype=torch.uint16)
        with torch.no_grad():
            self.snack.bias.copy_(conv.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        y2d = self.snack(x2d)
        return y2d.view(*x.shape[:-1], y2d.size(-1))

    def dst_step(self, zeta: float, device: str) -> None:
        pg = ZetaPrunerGrower(self.snack, zeta=zeta)
        pg.prune()
        pg.regrow(init=UniformInitializer, device=device)
        self.snack.indices_a.data = self.snack.indices_a.data.to(dtype=torch.uint16)
        self.snack.indices_b.data = self.snack.indices_b.data.to(dtype=torch.uint16)


def replace_gpt2_mlp_layers(model: GPT2LMHeadModel, variant: str, sparsity: float, device: str) -> None:
    for block in model.transformer.h:
        if variant == "dense_mask":
            block.mlp.c_fc = DenseMaskConv1D(block.mlp.c_fc, sparsity=sparsity)
            block.mlp.c_proj = DenseMaskConv1D(block.mlp.c_proj, sparsity=sparsity)
        elif variant == "snack":
            block.mlp.c_fc = SnackConv1D(block.mlp.c_fc, sparsity=sparsity, device=device)
            block.mlp.c_proj = SnackConv1D(block.mlp.c_proj, sparsity=sparsity, device=device)


def iter_mask_layers(model: GPT2LMHeadModel) -> Iterable[DenseMaskConv1D]:
    for block in model.transformer.h:
        if isinstance(block.mlp.c_fc, DenseMaskConv1D):
            yield block.mlp.c_fc
        if isinstance(block.mlp.c_proj, DenseMaskConv1D):
            yield block.mlp.c_proj


def iter_snack_layers(model: GPT2LMHeadModel) -> Iterable[SnackConv1D]:
    for block in model.transformer.h:
        if isinstance(block.mlp.c_fc, SnackConv1D):
            yield block.mlp.c_fc
        if isinstance(block.mlp.c_proj, SnackConv1D):
            yield block.mlp.c_proj


def build_lm_dataloaders(
    tokenizer_name: str,
    dataset_name: str,
    dataset_config: str,
    dataset_train_split: str,
    dataset_eval_split: str,
    text_column: str,
    max_train_samples: int,
    max_eval_samples: int,
    block_size: int,
    train_batch_size: int,
    eval_batch_size: int,
) -> Tuple[GPT2TokenizerFast, DataLoader, DataLoader]:
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dataset_config.strip():
        raw = load_dataset(dataset_name, dataset_config)
    else:
        raw = load_dataset(dataset_name)
    if dataset_train_split not in raw:
        raise KeyError(f"Train split '{dataset_train_split}' not found in dataset. Available: {list(raw.keys())}")
    if dataset_eval_split not in raw:
        raise KeyError(f"Eval split '{dataset_eval_split}' not found in dataset. Available: {list(raw.keys())}")
    train_raw = raw[dataset_train_split]
    eval_raw = raw[dataset_eval_split]

    if max_train_samples > 0:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))
    if max_eval_samples > 0:
        eval_raw = eval_raw.select(range(min(max_eval_samples, len(eval_raw))))

    if text_column not in train_raw.column_names:
        raise KeyError(f"Text column '{text_column}' missing in train split columns: {train_raw.column_names}")
    if text_column not in eval_raw.column_names:
        raise KeyError(f"Text column '{text_column}' missing in eval split columns: {eval_raw.column_names}")

    def tokenize_fn(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(examples[text_column], return_attention_mask=False)

    tokenized_train = train_raw.map(tokenize_fn, batched=True, remove_columns=train_raw.column_names)
    tokenized_eval = eval_raw.map(tokenize_fn, batched=True, remove_columns=eval_raw.column_names)

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_len = (len(concatenated) // block_size) * block_size
        if total_len == 0:
            return {"input_ids": []}
        chunks = [concatenated[i : i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks}

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_eval = tokenized_eval.map(group_texts, batched=True)
    lm_train.set_format(type="torch", columns=["input_ids"])
    lm_eval.set_format(type="torch", columns=["input_ids"])

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
        return {"input_ids": input_ids}

    train_loader = DataLoader(lm_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    eval_loader = DataLoader(lm_eval, batch_size=eval_batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    return tokenizer, train_loader, eval_loader


def evaluate_perplexity(model: GPT2LMHeadModel, eval_loader: DataLoader, device: str, max_eval_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_eval_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=batch["input_ids"])
            losses.append(float(out.loss.item()))
    if not losses:
        return float("nan")
    mean_loss = sum(losses) / len(losses)
    return float(math.exp(min(mean_loss, 20.0)))


def train_variant(
    model: GPT2LMHeadModel,
    variant: str,
    train_loader: DataLoader,
    device: str,
    lr: float,
    dst_interval: int,
    dst_zeta: float,
    max_train_steps: int,
    power: PowerReader,
) -> List[StepMetric]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics: List[StepMetric] = []
    name_map = {"dense": "Dense", "dense_mask": "Dense+Mask", "snack": "SNACK"}

    for step, batch in enumerate(train_loader):
        if step >= max_train_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        pw0 = power.read_watts()
        e0.record()
        out = model(**batch, labels=batch["input_ids"])
        loss = out.loss
        loss.backward()
        optimizer.step()
        e1.record()
        torch.cuda.synchronize()
        pw1 = power.read_watts()

        if variant == "dense_mask":
            for layer in iter_mask_layers(model):
                layer.apply_mask_()

        if dst_interval > 0 and step > 0 and (step % dst_interval == 0):
            if variant == "snack":
                for layer in iter_snack_layers(model):
                    layer.dst_step(zeta=dst_zeta, device=device)
            elif variant == "dense_mask":
                for layer in iter_mask_layers(model):
                    layer.dst_step(zeta=dst_zeta)

        cuda_ms = float(e0.elapsed_time(e1))
        metrics.append(
            StepMetric(
                model=name_map[variant],
                step=step,
                loss=float(loss.item()),
                cuda_time_ms=cuda_ms,
                power_w=(pw0 + pw1) * 0.5,
                energy_mj=((pw0 + pw1) * 0.5) * cuda_ms,
                memory_mb=float(torch.cuda.memory_allocated() / 1e6),
            )
        )
        if (step + 1) % 50 == 0:
            print(f"[{name_map[variant]}] step={step+1} loss={loss.item():.4f} time={cuda_ms:.2f}ms")
    return metrics


def benchmark_inference_token_latency(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    device: str,
    warmup: int,
    steps: int,
    power: PowerReader,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    lat, en = [], []
    with torch.no_grad():
        for _ in range(steps):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            pw = power.read_watts()
            e0.record()
            _ = model(**batch)
            e1.record()
            torch.cuda.synchronize()
            ms = float(e0.elapsed_time(e1))
            lat.append(ms)
            en.append(pw * ms)

    lat_t = torch.tensor(lat, dtype=torch.float32)
    en_t = torch.tensor(en, dtype=torch.float32)
    return (
        float(lat_t.mean().item()),
        float(torch.quantile(lat_t, 0.5).item()),
        float(torch.quantile(lat_t, 0.95).item()),
        float(en_t.mean().item()),
        float(torch.cuda.max_memory_allocated() / 1e6),
    )


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _serialize_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = v
    return out


def resolve_run_output_dir(args: argparse.Namespace) -> Tuple[Path, str]:
    serial = _serialize_args(args)
    hash_payload = dict(serial)
    hash_payload.pop("output_dir", None)
    hash_payload.pop("disable_output_hash", None)
    digest = hashlib.sha1(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:10]
    if args.disable_output_hash:
        return args.output_dir, digest
    return args.output_dir / f"run_{digest}", digest


def build_gpt2_config(args: argparse.Namespace, vocab_size: int, bos_token_id: int, eos_token_id: int) -> GPT2Config:
    if args.model_size == "custom":
        return GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,
            n_ctx=1024,
            n_embd=args.custom_n_embd,
            n_layer=args.custom_n_layer,
            n_head=args.custom_n_head,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    cfg = GPT2Config.from_pretrained(args.model_size)
    cfg.vocab_size = vocab_size
    cfg.bos_token_id = bos_token_id
    cfg.eos_token_id = eos_token_id
    return cfg


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    set_seed(args.seed)
    run_output_dir, run_hash = resolve_run_output_dir(args)
    args.output_dir = run_output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _serialize_args(args)
    cfg["run_hash"] = run_hash
    with (args.output_dir / "cfg.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    print(f"Run hash: {run_hash}")

    tokenizer, train_loader, eval_loader = build_lm_dataloaders(
        tokenizer_name=args.tokenizer_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_train_split=args.dataset_train_split,
        dataset_eval_split=args.dataset_eval_split,
        text_column=args.text_column,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        block_size=args.block_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    print(
        "Dataset:",
        f"{args.dataset_name}/{args.dataset_config or '-'}",
        f"train_split={args.dataset_train_split}",
        f"eval_split={args.dataset_eval_split}",
        f"max_train_samples={args.max_train_samples if args.max_train_samples > 0 else 'all'}",
        f"max_eval_samples={args.max_eval_samples if args.max_eval_samples > 0 else 'all'}",
    )
    config = build_gpt2_config(
        args=args,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    base_model = GPT2LMHeadModel(config)

    variants = ["dense", "dense_mask", "snack"]
    name_map = {"dense": "Dense", "dense_mask": "Dense+Mask", "snack": "SNACK"}
    sparsity_map = {"dense": "-", "dense_mask": f"{int(args.sparsity*100)}%", "snack": f"{int(args.sparsity*100)}%"}
    all_step_metrics: List[StepMetric] = []
    inference_metrics: List[InferenceMetric] = []
    power = PowerReader()

    try:
        for variant in variants:
            model = copy.deepcopy(base_model).to(args.device)
            if variant != "dense":
                replace_gpt2_mlp_layers(model, variant=variant, sparsity=args.sparsity, device=args.device)

            step_metrics = train_variant(
                model=model,
                variant=variant,
                train_loader=train_loader,
                device=args.device,
                lr=args.lr,
                dst_interval=args.dst_interval,
                dst_zeta=args.dst_zeta,
                max_train_steps=args.max_train_steps,
                power=power,
            )
            all_step_metrics.extend(step_metrics)

            ppl = evaluate_perplexity(model, eval_loader, args.device, args.max_eval_batches)
            lat_mean, lat_p50, lat_p95, en_mean, mem_mb = benchmark_inference_token_latency(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                device=args.device,
                warmup=args.inference_warmup,
                steps=args.inference_steps,
                power=power,
            )
            inference_metrics.append(
                InferenceMetric(
                    model=name_map[variant],
                    sparsity=sparsity_map[variant],
                    latency_ms_mean=lat_mean,
                    latency_ms_p50=lat_p50,
                    latency_ms_p95=lat_p95,
                    energy_mj_mean=en_mean,
                    memory_mb=mem_mb,
                    perplexity=ppl,
                )
            )
            print(
                f"[{name_map[variant]}] inference latency={lat_mean:.2f}ms "
                f"energy={en_mean:.3f}mJ perplexity={ppl:.2f}"
            )
            del model
            torch.cuda.empty_cache()
    finally:
        power.close()

    step_rows = [
        {
            "model": m.model,
            "step": m.step,
            "loss": m.loss,
            "cuda_time_ms": m.cuda_time_ms,
            "power_w": m.power_w,
            "energy_mj": m.energy_mj,
            "memory_mb": m.memory_mb,
        }
        for m in all_step_metrics
    ]
    inf_rows = [
        {
            "model": m.model,
            "sparsity": m.sparsity,
            "latency_ms_mean": m.latency_ms_mean,
            "latency_ms_p50": m.latency_ms_p50,
            "latency_ms_p95": m.latency_ms_p95,
            "energy_mj_mean": m.energy_mj_mean,
            "memory_mb": m.memory_mb,
            "perplexity": m.perplexity,
        }
        for m in inference_metrics
    ]
    write_csv(args.output_dir / "training_step_metrics.csv", step_rows)
    write_csv(args.output_dir / "inference_metrics.csv", inf_rows)

    table_a = []
    for model_name in ["Dense", "Dense+Mask", "SNACK"]:
        rows = [r for r in step_rows if r["model"] == model_name]
        if not rows:
            continue
        table_a.append(
            {
                "Model": model_name,
                "Sparsity": "-" if model_name == "Dense" else f"{int(args.sparsity*100)}%",
                "Loss": rows[-1]["loss"],
                "Training_time_step_ms": sum(r["cuda_time_ms"] for r in rows) / len(rows),
                "Energy_step_mj": sum(r["energy_mj"] for r in rows) / len(rows),
                "Memory_mb": sum(r["memory_mb"] for r in rows) / len(rows),
            }
        )
    write_csv(args.output_dir / "table_a_training_efficiency.csv", table_a)

    table_b = [
        {
            "Model": r["model"],
            "Sparsity": r["sparsity"],
            "Latency_token_ms": r["latency_ms_mean"],
            "Energy_token_mj": r["energy_mj_mean"],
            "Perplexity": r["perplexity"],
        }
        for r in inf_rows
    ]
    write_csv(args.output_dir / "table_b_inference_batch1.csv", table_b)
    print(f"Saved outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
