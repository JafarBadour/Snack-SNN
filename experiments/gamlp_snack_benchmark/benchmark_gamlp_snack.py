#!/usr/bin/env python3
"""Benchmark Dense vs Dense+Mask vs SNACK on trained GAMLP checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
GAMLP_ROOT = REPO_ROOT / "third_party" / "GAMLP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAMLP_ROOT) not in sys.path:
    sys.path.insert(0, str(GAMLP_ROOT))

from DST.initializers.uniform_initializer import UniformInitializer  # noqa: E402
from DST.layers import Snack  # noqa: E402
import layer as gamlp_layer  # noqa: E402
from load_dataset import prepare_data  # noqa: E402
from utils import gen_model, gen_model_rlu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAMLP + SNACK batch-size-1 inference benchmark.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=False,
        help="Path to trained GAMLP checkpoint (.pkl/.pt). Required unless synthetic mode is enabled.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-mag"],
        help="OGB dataset used by the trained checkpoint.",
    )
    parser.add_argument("--method", type=str, default="R_GAMLP")
    parser.add_argument("--use-rlu", action="store_true", default=False)
    parser.add_argument("--root", type=str, default="third_party/GAMLP/data")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=5)
    parser.add_argument("--label-num-hops", type=int, default=9)
    parser.add_argument("--n-layers-1", type=int, default=2)
    parser.add_argument("--n-layers-2", type=int, default=2)
    parser.add_argument("--n-layers-3", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input-drop", type=float, default=0.0)
    parser.add_argument("--att-drop", type=float, default=0.5)
    parser.add_argument("--label-drop", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pre-process", dest="pre_process", action="store_true")
    parser.add_argument("--no-pre-process", dest="pre_process", action="store_false")
    parser.set_defaults(pre_process=True)
    parser.add_argument("--residual", dest="residual", action="store_true")
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.set_defaults(residual=True)
    parser.add_argument("--pre-dropout", action="store_true", default=False)
    parser.add_argument("--bns", dest="bns", action="store_true")
    parser.add_argument("--no-bns", dest="bns", action="store_false")
    parser.set_defaults(bns=True)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.70, 0.80, 0.90, 0.95, 0.99],
    )
    parser.add_argument(
        "--matrix-shapes",
        type=str,
        nargs="+",
        default=None,
        help='Run matrix-kernel inference mode, e.g. "5000x5000".',
    )
    parser.add_argument(
        "--matrix-densities",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10],
        help="Densities for matrix-kernel mode.",
    )
    # Backward-compatible aliases (hidden from help output).
    parser.add_argument("--synthetic-shapes", dest="synthetic_shapes_compat", type=str, nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--synthetic-densities",
        dest="synthetic_densities_compat",
        type=float,
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--skip-first-measured",
        type=int,
        default=1,
        help="Skip first N measured iterations after warmup (steady-state timing).",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["dense", "dense_mask", "snack"],
        choices=["dense", "dense_mask", "snack"],
        help="Which variants to benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-artifact-mask",
        action="store_true",
        default=True,
        help="If checkpoint variant is dense_mask/snack, preserve its learned mask topology when converting variants.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/gamlp_snack_benchmark/results/gamlp_snack_benchmark.csv"),
    )
    return parser.parse_args()


@dataclass
class Metrics:
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    energy_mj_mean: float
    memory_mb: float
    accuracy: float
    agreement_vs_baseline: float


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


def parse_shape_token(token: str) -> Tuple[int, int]:
    tok = token.lower().replace(" ", "")
    if "x" not in tok:
        raise ValueError(f'Invalid shape "{token}". Expected like 5000x5000.')
    a, b = tok.split("x", 1)
    return int(a), int(b)


def make_synthetic_sparse_weight(in_dim: int, out_dim: int, density: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")
    total = in_dim * out_dim
    nnz = max(1, int(round(total * density)))
    weight = torch.randn(in_dim, out_dim, device=device, dtype=torch.float32)
    flat_mask = torch.zeros(total, device=device, dtype=torch.float32)
    idx = torch.randperm(total, device=device)[:nnz]
    flat_mask[idx] = 1.0
    mask = flat_mask.view(in_dim, out_dim)
    return weight, mask


def run_matrix_variant(
    variant: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    warmup: int,
    repeat: int,
    skip_first_measured: int,
    power: PowerReader,
    device: str,
) -> Metrics:
    snack_layer = None
    if variant == "snack":
        w_sparse = (weight * mask).contiguous()
        snack_layer = Snack(
            input_size=w_sparse.shape[0],
            output_size=w_sparse.shape[1],
            sparsity=0.0,
            dense_weight=w_sparse,
            initializer=UniformInitializer,
            bias=False,
            device=device,
        ).to(device)
        snack_layer.indices_a.data = snack_layer.indices_a.data.to(dtype=torch.uint16)
        snack_layer.indices_b.data = snack_layer.indices_b.data.to(dtype=torch.uint16)

    def forward_once(inp: torch.Tensor) -> torch.Tensor:
        if variant == "dense":
            return inp @ weight
        if variant == "dense_mask":
            return inp @ (weight * mask)
        if variant == "snack":
            return snack_layer(inp)  # type: ignore[misc]
        raise ValueError(f"Unknown variant: {variant}")

    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_once(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        lat, en = [], []
        for _ in range(repeat):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            pw = power.read_watts()
            e0.record()
            _ = forward_once(x)
            e1.record()
            torch.cuda.synchronize()
            ms = float(e0.elapsed_time(e1))
            lat.append(ms)
            en.append(pw * ms)

    skip_n = min(max(0, skip_first_measured), len(lat) - 1) if len(lat) > 1 else 0
    lat_t = torch.tensor(lat[skip_n:], dtype=torch.float32)
    en_t = torch.tensor(en[skip_n:], dtype=torch.float32)
    return Metrics(
        latency_ms_mean=float(lat_t.mean().item()),
        latency_ms_p50=float(torch.quantile(lat_t, 0.50).item()),
        latency_ms_p95=float(torch.quantile(lat_t, 0.95).item()),
        energy_mj_mean=float(en_t.mean().item()),
        memory_mb=float(torch.cuda.max_memory_allocated() / 1e6),
        accuracy=0.0,
        agreement_vs_baseline=0.0,
    )


def run_snack_variant_from_sparse(
    x: torch.Tensor,
    w_sparse: torch.Tensor,
    warmup: int,
    repeat: int,
    skip_first_measured: int,
    power: PowerReader,
    device: str,
) -> Metrics:
    snack_layer = Snack(
        input_size=w_sparse.shape[0],
        output_size=w_sparse.shape[1],
        sparsity=0.0,
        dense_weight=w_sparse,
        initializer=UniformInitializer,
        bias=False,
        device=device,
    ).to(device)
    snack_layer.indices_a.data = snack_layer.indices_a.data.to(dtype=torch.uint16)
    snack_layer.indices_b.data = snack_layer.indices_b.data.to(dtype=torch.uint16)
    # We only need SNACK runtime state from here onward.
    del w_sparse
    torch.cuda.empty_cache()

    with torch.no_grad():
        for _ in range(warmup):
            _ = snack_layer(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        lat, en = [], []
        for _ in range(repeat):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            pw = power.read_watts()
            e0.record()
            _ = snack_layer(x)
            e1.record()
            torch.cuda.synchronize()
            ms = float(e0.elapsed_time(e1))
            lat.append(ms)
            en.append(pw * ms)

    skip_n = min(max(0, skip_first_measured), len(lat) - 1) if len(lat) > 1 else 0
    lat_t = torch.tensor(lat[skip_n:], dtype=torch.float32)
    en_t = torch.tensor(en[skip_n:], dtype=torch.float32)
    return Metrics(
        latency_ms_mean=float(lat_t.mean().item()),
        latency_ms_p50=float(torch.quantile(lat_t, 0.50).item()),
        latency_ms_p95=float(torch.quantile(lat_t, 0.95).item()),
        energy_mj_mean=float(en_t.mean().item()),
        memory_mb=float(torch.cuda.max_memory_allocated() / 1e6),
        accuracy=0.0,
        agreement_vs_baseline=0.0,
    )


def load_checkpoint_with_metadata(path: Path, device: str) -> tuple[Dict[str, torch.Tensor], str, float]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    ckpt_variant = "dense"
    ckpt_sparsity = 0.9
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        ckpt_variant = ckpt.get("variant", "dense")
        ckpt_sparsity = float(ckpt.get("sparsity", ckpt_sparsity))
        uint16_keys = ckpt.get("state_dict_uint16_keys", [])
        if uint16_keys:
            state_dict = restore_uint16_indices_in_state_dict(state_dict, uint16_keys)
    else:
        state_dict = ckpt
    return state_dict, ckpt_variant, ckpt_sparsity


def infer_model_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int | bool]:
    cfg: Dict[str, int | bool] = {}
    if "label_fc.layers.0.weight" in state_dict:
        cfg["hidden"] = int(state_dict["label_fc.layers.0.weight"].shape[0])
    elif "res_fc.weight" in state_dict:
        cfg["hidden"] = int(state_dict["res_fc.weight"].shape[0])

    process_idxs = []
    process_layer_idxs = []
    lr_output_layer_idxs = []
    label_fc_layer_idxs = []
    has_bns = False
    has_residual = False
    has_process = False
    for key in state_dict.keys():
        m = re.match(r"process\.(\d+)\.", key)
        if m:
            has_process = True
            process_idxs.append(int(m.group(1)))
        m = re.match(r"process\.\d+\.layers\.(\d+)\.weight", key)
        if m:
            process_layer_idxs.append(int(m.group(1)))
        m = re.match(r"lr_output\.layers\.(\d+)\.weight", key)
        if m:
            lr_output_layer_idxs.append(int(m.group(1)))
        m = re.match(r"label_fc\.layers\.(\d+)\.weight", key)
        if m:
            label_fc_layer_idxs.append(int(m.group(1)))
        if ".bns." in key:
            has_bns = True
        if key.startswith("res_fc."):
            has_residual = True

    if process_idxs:
        cfg["num_hops"] = max(process_idxs)
    if process_layer_idxs:
        cfg["n_layers_1"] = max(process_layer_idxs) + 1
    if lr_output_layer_idxs:
        cfg["n_layers_2"] = max(lr_output_layer_idxs) + 1
    if label_fc_layer_idxs:
        cfg["n_layers_3"] = max(label_fc_layer_idxs) + 1
    cfg["pre_process"] = has_process
    cfg["residual"] = has_residual
    cfg["bns"] = has_bns
    return cfg


def ensure_snack_uint16_indices(layer: Snack) -> None:
    layer.indices_a.data = layer.indices_a.data.to(dtype=torch.uint16)
    layer.indices_b.data = layer.indices_b.data.to(dtype=torch.uint16)


def restore_uint16_indices_in_state_dict(state_dict: Dict[str, torch.Tensor], uint16_keys: List[str]) -> Dict[str, torch.Tensor]:
    restored = dict(state_dict)
    for key in uint16_keys:
        if key in restored and torch.is_tensor(restored[key]):
            restored[key] = restored[key].to(dtype=torch.uint16)
    return restored


def make_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat = weight.detach().abs().reshape(-1)
    keep = max(1, int(round(flat.numel() * (1.0 - sparsity))))
    idx = torch.topk(flat, k=keep, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=weight.dtype)
    mask[idx] = 1.0
    return mask.reshape_as(weight)


class MaskedLinear(nn.Module):
    def __init__(self, linear: nn.Module, sparsity: float, use_existing_mask: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None
        if use_existing_mask and hasattr(linear, "mask"):
            self.register_buffer("mask", linear.mask.detach().clone())
        else:
            self.register_buffer("mask", make_mask(self.weight, sparsity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class SnackLinear(nn.Module):
    def __init__(self, linear: nn.Module, sparsity: float, device: str, use_existing_mask: bool = False) -> None:
        super().__init__()
        w = linear.weight.detach().clone()
        if use_existing_mask and hasattr(linear, "mask"):
            m = linear.mask.detach().clone().to(dtype=w.dtype)
        else:
            m = make_mask(w, sparsity)
        w_sparse = (w * m).t().contiguous()  # [in, out]
        self.snack = Snack(
            input_size=w_sparse.size(0),
            output_size=w_sparse.size(1),
            sparsity=0.0,
            dense_weight=w_sparse,
            initializer=UniformInitializer,
            bias=linear.bias is not None,
            device=device,
        ).to(device)
        ensure_snack_uint16_indices(self.snack)
        if linear.bias is not None:
            with torch.no_grad():
                self.snack.bias.copy_(linear.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.snack(x)


class MaskedDense(gamlp_layer.Dense):
    def __init__(self, base: gamlp_layer.Dense, sparsity: float, use_existing_mask: bool = False) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if use_existing_mask and hasattr(base, "mask"):
            self.register_buffer("mask", base.mask.detach().clone())
        else:
            self.register_buffer("mask", make_mask(self.weight, sparsity))
        self.bias = copy.deepcopy(base.bias) if isinstance(base.bias, nn.Module) else (lambda x: x)

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.mm(input, self.weight * self.mask)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


class SnackDense(gamlp_layer.Dense):
    def __init__(self, base: gamlp_layer.Dense, sparsity: float, device: str, use_existing_mask: bool = False) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        w = base.weight.detach().clone()
        if use_existing_mask and hasattr(base, "mask"):
            m = base.mask.detach().clone().to(dtype=w.dtype)
        else:
            m = make_mask(w, sparsity)
        w_sparse = (w * m).contiguous()
        self.snack = Snack(
            input_size=w_sparse.size(0),
            output_size=w_sparse.size(1),
            sparsity=0.0,
            dense_weight=w_sparse,
            initializer=UniformInitializer,
            bias=False,
            device=device,
        ).to(device)
        ensure_snack_uint16_indices(self.snack)
        self.bias = copy.deepcopy(base.bias) if isinstance(base.bias, nn.Module) else (lambda x: x)

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.snack(input)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


class MaskedGraphConvolution(gamlp_layer.GraphConvolution):
    def __init__(self, base: gamlp_layer.GraphConvolution, sparsity: float, use_existing_mask: bool = False) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.alpha = base.alpha
        self.bns = base.bns
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if use_existing_mask and hasattr(base, "mask"):
            self.register_buffer("mask", base.mask.detach().clone())
        else:
            self.register_buffer("mask", make_mask(self.weight, sparsity))
        self.bias = copy.deepcopy(base.bias)

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        support = (1 - self.alpha) * input + self.alpha * h0
        output = torch.mm(support, self.weight * self.mask)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


class SnackGraphConvolution(gamlp_layer.GraphConvolution):
    def __init__(
        self, base: gamlp_layer.GraphConvolution, sparsity: float, device: str, use_existing_mask: bool = False
    ) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.alpha = base.alpha
        self.bns = base.bns
        w = base.weight.detach().clone()
        if use_existing_mask and hasattr(base, "mask"):
            m = base.mask.detach().clone().to(dtype=w.dtype)
        else:
            m = make_mask(w, sparsity)
        w_sparse = (w * m).contiguous()
        self.snack = Snack(
            input_size=w_sparse.size(0),
            output_size=w_sparse.size(1),
            sparsity=0.0,
            dense_weight=w_sparse,
            initializer=UniformInitializer,
            bias=False,
            device=device,
        ).to(device)
        ensure_snack_uint16_indices(self.snack)
        self.bias = copy.deepcopy(base.bias)

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        support = (1 - self.alpha) * input + self.alpha * h0
        output = self.snack(support)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


def replace_layers(module: nn.Module, variant: str, sparsity: float, device: str, use_existing_mask: bool = False) -> None:
    for name, child in list(module.named_children()):
        replacement = None
        if isinstance(child, (nn.Linear, MaskedLinear)):
            if variant == "dense_mask":
                replacement = MaskedLinear(child, sparsity, use_existing_mask=use_existing_mask)
            elif variant == "snack":
                replacement = SnackLinear(child, sparsity, device=device, use_existing_mask=use_existing_mask)
        elif isinstance(child, gamlp_layer.Dense):
            if variant == "dense_mask":
                replacement = MaskedDense(child, sparsity, use_existing_mask=use_existing_mask)
            elif variant == "snack":
                replacement = SnackDense(child, sparsity, device=device, use_existing_mask=use_existing_mask)
        elif isinstance(child, gamlp_layer.GraphConvolution):
            if variant == "dense_mask":
                replacement = MaskedGraphConvolution(child, sparsity, use_existing_mask=use_existing_mask)
            elif variant == "snack":
                replacement = SnackGraphConvolution(child, sparsity, device=device, use_existing_mask=use_existing_mask)

        if replacement is not None:
            setattr(module, name, replacement)
        else:
            replace_layers(
                child,
                variant=variant,
                sparsity=sparsity,
                device=device,
                use_existing_mask=use_existing_mask,
            )


def build_gamlp_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hidden=cli.hidden,
        num_hops=cli.num_hops,
        label_num_hops=cli.label_num_hops,
        seed=cli.seed,
        lr=0.001,
        dataset=cli.dataset,
        dropout=cli.dropout,
        gpu=0,
        weight_decay=0.0,
        eval_every=1,
        batch_size=50000,
        n_layers_1=cli.n_layers_1,
        n_layers_2=cli.n_layers_2,
        n_layers_3=cli.n_layers_3,
        num_runs=1,
        patience=100,
        alpha=cli.alpha,
        temp=1.0,
        threshold=0.0,
        input_drop=cli.input_drop,
        att_drop=cli.att_drop,
        label_drop=cli.label_drop,
        gama=0.5,
        pre_process=cli.pre_process,
        residual=cli.residual,
        act=cli.act,
        method=cli.method,
        use_emb=None,
        root=cli.root,
        emb_path=str(GAMLP_ROOT / "data"),
        use_relation_subsets=str(GAMLP_ROOT / "data" / "mag"),
        use_rlu=cli.use_rlu,
        train_num_epochs=[0, 0],
        stages=[1],
        pre_dropout=cli.pre_dropout,
        bns=cli.bns,
    )


def run_model(
    model: nn.Module,
    feats: List[torch.Tensor],
    labels: torch.Tensor,
    label_emb: torch.Tensor | None,
    eval_indices: torch.Tensor,
    batch_size: int,
    warmup: int,
    skip_first_measured: int,
    power: PowerReader,
) -> Tuple[Metrics, torch.Tensor]:
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    n = int(eval_indices.numel())
    with torch.no_grad():
        for i in range(warmup):
            idx = eval_indices[(i * batch_size) % n : min(((i * batch_size) % n) + batch_size, n)]
            feat_list = [x[idx] for x in feats]
            if label_emb is None:
                _ = model(feat_list)
            else:
                _ = model(feat_list, label_emb[idx])
    torch.cuda.synchronize()

    lat, en = [], []
    pred_all = []
    with torch.no_grad():
        steps = max(1, n // batch_size)
        for i in range(steps):
            start = i * batch_size
            end = min(start + batch_size, n)
            idx = eval_indices[start:end]
            feat_list = [x[idx] for x in feats]
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            pw = power.read_watts()
            e0.record()
            if label_emb is None:
                logits = model(feat_list)
            else:
                logits = model(feat_list, label_emb[idx])
            e1.record()
            torch.cuda.synchronize()
            ms = float(e0.elapsed_time(e1))
            lat.append(ms)
            en.append(pw * ms)
            pred_all.append(logits.argmax(dim=1).detach().cpu())

    pred = torch.cat(pred_all, dim=0)
    gt = labels[eval_indices[: pred.size(0)]].detach().cpu()
    acc = float((pred == gt).float().mean().item())
    skip_n = min(max(0, skip_first_measured), len(lat) - 1) if len(lat) > 1 else 0
    lat_eval = lat[skip_n:]
    en_eval = en[skip_n:]
    lat_t = torch.tensor(lat_eval, dtype=torch.float32)
    en_t = torch.tensor(en_eval, dtype=torch.float32)
    metrics = Metrics(
        latency_ms_mean=float(lat_t.mean().item()),
        latency_ms_p50=float(torch.quantile(lat_t, 0.50).item()),
        latency_ms_p95=float(torch.quantile(lat_t, 0.95).item()),
        energy_mj_mean=float(en_t.mean().item()),
        memory_mb=float(torch.cuda.max_memory_allocated() / 1e6),
        accuracy=acc,
        agreement_vs_baseline=0.0,
    )
    return metrics, pred


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    requested_variants = list(dict.fromkeys(args.variants))

    matrix_shapes = args.matrix_shapes or args.synthetic_shapes_compat
    matrix_densities = args.matrix_densities if args.synthetic_shapes_compat is None else (
        args.synthetic_densities_compat or args.matrix_densities
    )

    if matrix_shapes:
        power = PowerReader()
        rows: List[Dict[str, float | str]] = []
        try:
            repeat = max(10, int(args.num_samples))
            for shape_token in matrix_shapes:
                in_dim, out_dim = parse_shape_token(shape_token)
                for density in matrix_densities:
                    weight, mask = make_synthetic_sparse_weight(in_dim, out_dim, density, args.device)
                    metrics_by_name: Dict[str, Metrics] = {}
                    eval_order = [n for n in requested_variants if n != "snack"] + (
                        ["snack"] if "snack" in requested_variants else []
                    )
                    for name in eval_order:
                        x = torch.randn(args.batch_size, in_dim, device=args.device, dtype=torch.float32)
                        if name == "snack":
                            w_sparse = (weight * mask).contiguous()
                            del weight
                            del mask
                            torch.cuda.empty_cache()
                            metrics_by_name[name] = run_snack_variant_from_sparse(
                                x=x,
                                w_sparse=w_sparse,
                                warmup=args.warmup,
                                repeat=repeat,
                                skip_first_measured=args.skip_first_measured,
                                power=power,
                                device=args.device,
                            )
                        else:
                            metrics_by_name[name] = run_matrix_variant(
                                variant=name,
                                x=x,
                                weight=weight,
                                mask=mask,
                                warmup=args.warmup,
                                repeat=repeat,
                                skip_first_measured=args.skip_first_measured,
                                power=power,
                                device=args.device,
                            )

                    baseline_name = "dense_mask" if "dense_mask" in metrics_by_name else requested_variants[0]
                    for name in requested_variants:
                        m = metrics_by_name[name]
                        row = {
                            "dataset": f"matrix_{in_dim}x{out_dim}",
                            "method": "matrix-kernel",
                            "sparsity": 1.0 - density,
                            "type": name,
                            "baseline_type": baseline_name,
                            "batch_size": args.batch_size,
                            "num_samples": repeat,
                            "latency_ms_mean": m.latency_ms_mean,
                            "latency_ms_p50": m.latency_ms_p50,
                            "latency_ms_p95": m.latency_ms_p95,
                            "energy_mj_mean": m.energy_mj_mean,
                            "memory_mb": m.memory_mb,
                            "accuracy": 0.0,
                            "agreement_vs_baseline": 1.0 if name == baseline_name else 0.0,
                        }
                        rows.append(row)
                        print(
                            f"shape={in_dim}x{out_dim} dens={density:.3f} {name:<10} "
                            f"lat={m.latency_ms_mean:.3f}ms energy={m.energy_mj_mean:.3f}mJ mem={m.memory_mb:.1f}MB"
                        )
        finally:
            power.close()

        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "method",
                    "sparsity",
                    "type",
                    "baseline_type",
                    "batch_size",
                    "num_samples",
                    "latency_ms_mean",
                    "latency_ms_p50",
                    "latency_ms_p95",
                    "energy_mj_mean",
                    "memory_mb",
                    "accuracy",
                    "agreement_vs_baseline",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results: {args.output_csv}")
        return

    if args.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required unless matrix mode is used (--matrix-shapes).")

    state_dict, ckpt_variant, ckpt_sparsity = load_checkpoint_with_metadata(args.checkpoint_path, args.device)
    inferred = infer_model_config_from_state_dict(state_dict)
    for key, value in inferred.items():
        setattr(args, key, value)
    print(
        "Checkpoint-aligned config: "
        f"hidden={args.hidden}, num_hops={args.num_hops}, "
        f"n_layers=({args.n_layers_1},{args.n_layers_2},{args.n_layers_3}), "
        f"pre_process={args.pre_process}, residual={args.residual}, bns={args.bns}"
    )

    gamlp_args = build_gamlp_args(args)
    feats, labels, in_size, num_classes, train_nid, val_nid, test_nid, _evaluator, label_emb = prepare_data(
        args.device, gamlp_args, teacher_probs=None
    )
    train_len, val_len, test_len = len(train_nid), len(val_nid), len(test_nid)
    test_start = train_len + val_len
    test_idx = torch.arange(test_start, test_start + test_len, device=args.device)
    test_idx = test_idx[: min(args.num_samples, test_idx.numel())]

    model = gen_model_rlu(gamlp_args, in_size, num_classes) if args.use_rlu else gen_model(gamlp_args, in_size, num_classes)
    if model is None:
        raise RuntimeError(
            f"Unsupported method/use_rlu combo: method={args.method}, use_rlu={args.use_rlu}. "
            "Use method in {R_GAMLP, JK_GAMLP} for non-RLU, "
            "or {R_GAMLP_RLU, JK_GAMLP_RLU} with --use-rlu."
        )
    model = model.to(args.device)

    if ckpt_variant in {"dense_mask", "snack"}:
        replace_layers(
            model,
            variant=ckpt_variant,
            sparsity=ckpt_sparsity,
            device=args.device,
            use_existing_mask=False,
        )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint variant={ckpt_variant} sparsity={ckpt_sparsity:.4f}")

    if ckpt_variant in {"dense_mask", "snack"} and "dense" in requested_variants:
        raise ValueError(
            "Dense variant is not supported directly from sparse artifact checkpoints. "
            "Use --variants dense_mask snack for artifact-based sparse comparison."
        )

    feats = [x.to(args.device) for x in feats]
    labels = labels.to(args.device).long()
    if label_emb is not None:
        label_emb = label_emb.to(args.device)

    power = PowerReader()
    rows: List[Dict[str, float | str]] = []
    try:
        eval_sparsities = [ckpt_sparsity] if ckpt_variant in {"dense_mask", "snack"} else args.sparsities
        for sparsity in eval_sparsities:
            variants: Dict[str, nn.Module] = {}
            for name in requested_variants:
                variants[name] = copy.deepcopy(model).eval()
                if name != ckpt_variant and name != "dense":
                    replace_layers(
                        variants[name],
                        variant=name,
                        sparsity=sparsity,
                        device=args.device,
                        use_existing_mask=(args.use_artifact_mask and ckpt_variant in {"dense_mask", "snack"}),
                    )

            metrics_by_name: Dict[str, Metrics] = {}
            preds_by_name: Dict[str, torch.Tensor] = {}
            for name in requested_variants:
                m, pred = run_model(
                    model=variants[name],
                    feats=feats,
                    labels=labels,
                    label_emb=label_emb,
                    eval_indices=test_idx,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    skip_first_measured=args.skip_first_measured,
                    power=power,
                )
                metrics_by_name[name] = m
                preds_by_name[name] = pred

            baseline_name = "dense" if "dense" in preds_by_name else requested_variants[0]
            baseline_pred = preds_by_name[baseline_name]
            for name in requested_variants:
                m = metrics_by_name[name]
                m.agreement_vs_baseline = float((preds_by_name[name] == baseline_pred).float().mean().item())
                row = {
                    "dataset": args.dataset,
                    "method": args.method,
                    "sparsity": sparsity,
                    "type": name,
                    "baseline_type": baseline_name,
                    "batch_size": args.batch_size,
                    "num_samples": int(test_idx.numel()),
                    "latency_ms_mean": m.latency_ms_mean,
                    "latency_ms_p50": m.latency_ms_p50,
                    "latency_ms_p95": m.latency_ms_p95,
                    "energy_mj_mean": m.energy_mj_mean,
                    "memory_mb": m.memory_mb,
                    "accuracy": m.accuracy,
                    "agreement_vs_baseline": m.agreement_vs_baseline,
                }
                rows.append(row)
                print(
                    f"s={sparsity:.2f} {name:<10} lat={m.latency_ms_mean:.3f}ms "
                    f"energy={m.energy_mj_mean:.3f}mJ mem={m.memory_mb:.1f}MB "
                    f"acc={m.accuracy:.4f} agree({baseline_name})={m.agreement_vs_baseline:.4f}"
                )
            del variants
            torch.cuda.empty_cache()
    finally:
        power.close()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "sparsity",
                "type",
                "baseline_type",
                "batch_size",
                "num_samples",
                "latency_ms_mean",
                "latency_ms_p50",
                "latency_ms_p95",
                "energy_mj_mean",
                "memory_mb",
                "accuracy",
                "agreement_vs_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results: {args.output_csv}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total elapsed: {time.time() - t0:.2f}s")
