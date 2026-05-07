#!/usr/bin/env python3
"""Train GAMLP variants (dense, dense+mask, snack) with profiling."""

from __future__ import annotations

import argparse
import csv
import copy
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
GAMLP_ROOT = REPO_ROOT / "third_party" / "GAMLP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GAMLP_ROOT) not in sys.path:
    sys.path.insert(0, str(GAMLP_ROOT))

from DST.initializers.uniform_initializer import UniformInitializer  # noqa: E402
from DST.layers import Snack  # noqa: E402
from DST.pruner_grower import ZetaPrunerGrower  # noqa: E402
import layer as gamlp_layer  # noqa: E402
from load_dataset import prepare_data  # noqa: E402
from utils import gen_model, gen_model_rlu  # noqa: E402


@dataclass
class TrainEpochMetrics:
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    energy_mj_mean: float
    energy_mj_total: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    samples_per_sec: float
    epoch_time_s: float


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
    parser = argparse.ArgumentParser(description="Train GAMLP variants with speed/energy/memory profiling.")
    parser.add_argument("--dataset", type=str, default="ogbn-products", choices=["ogbn-products", "ogbn-papers100M", "ogbn-mag"])
    parser.add_argument("--method", type=str, default="R_GAMLP_RLU")
    parser.add_argument("--use-rlu", action="store_true", default=True)
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
    parser.add_argument("--pre-process", action="store_true", default=True)
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--pre-dropout", action="store_true", default=False)
    parser.add_argument("--bns", dest="bns", action="store_true")
    parser.add_argument("--no-bns", dest="bns", action="store_false")
    parser.set_defaults(bns=True)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--progress-update-every", type=int, default=200, help="Update tqdm postfix every N steps.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        type=str,
        default="snack",
        choices=["dense", "dense_mask", "snack"],
        help="Training variant: dense baseline, dense with static mask, or SNACK.",
    )
    parser.add_argument("--sparsity", type=float, default=0.9, help="Initial static sparsity before DST starts.")
    parser.add_argument("--dst-zeta", type=float, default=0.05, help="Prune/grow ratio per DST step.")
    parser.add_argument("--dst-every", type=int, default=5, help="Run DST every N epochs.")
    parser.add_argument("--dst-until-epoch", type=int, default=25, help="Last epoch index (inclusive) to apply DST.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda"])
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional dense GAMLP checkpoint to initialize from.")
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_checkpoint.pt"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/gamlp_snack_benchmark/results/gamlp_snack_dst_training_metrics.csv"),
        help="Per-epoch speed/energy/memory logs.",
    )
    return parser.parse_args()


def make_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat = weight.detach().abs().reshape(-1)
    keep = max(1, int(round(flat.numel() * (1.0 - sparsity))))
    idx = torch.topk(flat, k=keep, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=weight.dtype)
    mask[idx] = 1.0
    return mask.reshape_as(weight)


def ensure_snack_uint16_indices(layer: Snack) -> None:
    layer.indices_a.data = layer.indices_a.data.to(dtype=torch.uint16)
    layer.indices_b.data = layer.indices_b.data.to(dtype=torch.uint16)


def make_state_dict_serializable(state_dict: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], List[str]]:
    converted_uint16_keys: List[str] = []
    serializable: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.dtype == torch.uint16:
            serializable[key] = value.to(dtype=torch.int32)
            converted_uint16_keys.append(key)
        else:
            serializable[key] = value
    return serializable, converted_uint16_keys


def restore_uint16_indices_in_state_dict(state_dict: Dict[str, torch.Tensor], uint16_keys: List[str]) -> Dict[str, torch.Tensor]:
    restored = dict(state_dict)
    for key in uint16_keys:
        if key in restored and torch.is_tensor(restored[key]):
            restored[key] = restored[key].to(dtype=torch.uint16)
    return restored


def collect_masked_layers(module: nn.Module) -> List[nn.Module]:
    masked: List[nn.Module] = []
    for m in module.modules():
        if isinstance(m, (MaskedLinear, MaskedDense, MaskedGraphConvolution)):
            masked.append(m)
    return masked


def apply_dense_mask_dst(masked_layers: List[nn.Module], zeta: float) -> None:
    for layer in masked_layers:
        with torch.no_grad():
            weight = layer.weight
            mask = layer.mask
            flat_w = weight.view(-1)
            flat_m = mask.view(-1)
            active = flat_m > 0
            active_idx = torch.where(active)[0]
            active_count = int(active_idx.numel())
            if active_count <= 1:
                continue

            prune_count = max(1, int(round(zeta * active_count)))
            prune_count = min(prune_count, active_count - 1)
            active_scores = flat_w[active_idx].abs()
            prune_rel = torch.topk(active_scores, k=prune_count, largest=False, sorted=False).indices
            prune_idx = active_idx[prune_rel]

            inactive_idx = torch.where(~active)[0]
            grow_count = min(prune_count, int(inactive_idx.numel()))
            if grow_count > 0:
                perm = torch.randperm(inactive_idx.numel(), device=inactive_idx.device)[:grow_count]
                grow_idx = inactive_idx[perm]
            else:
                grow_idx = torch.empty(0, dtype=inactive_idx.dtype, device=inactive_idx.device)

            flat_m[prune_idx] = 0
            flat_w[prune_idx] = 0
            if grow_count > 0:
                fan_in, fan_out = weight.shape[0], weight.shape[1]
                bound = (6.0 / float(fan_in + fan_out)) ** 0.5
                flat_m[grow_idx] = 1
                flat_w[grow_idx] = torch.empty(grow_count, device=weight.device).uniform_(-bound, bound)


class SnackLinear(nn.Module):
    def __init__(self, linear: nn.Linear, sparsity: float, device: str) -> None:
        super().__init__()
        w = linear.weight.detach().clone()
        w_sparse = (w * make_mask(w, sparsity)).t().contiguous()  # [in, out]
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


class SnackDense(gamlp_layer.Dense):
    def __init__(self, base: gamlp_layer.Dense, sparsity: float, device: str) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        w = base.weight.detach().clone()
        w_sparse = (w * make_mask(w, sparsity)).contiguous()  # [in, out]
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
        out = self.snack(input)
        out = self.bias(out)
        if self.in_features == self.out_features:
            out = out + input
        return out


class SnackGraphConvolution(gamlp_layer.GraphConvolution):
    def __init__(self, base: gamlp_layer.GraphConvolution, sparsity: float, device: str) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.alpha = base.alpha
        self.bns = base.bns
        w = base.weight.detach().clone()
        w_sparse = (w * make_mask(w, sparsity)).contiguous()  # [in, out]
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
        out = self.snack(support)
        out = self.bias(out)
        if self.in_features == self.out_features:
            out = out + input
        return out


class MaskedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, sparsity: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=True)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=True)
        else:
            self.bias = None
        self.register_buffer("mask", make_mask(self.weight.detach(), sparsity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedDense(gamlp_layer.Dense):
    def __init__(self, base: gamlp_layer.Dense, sparsity: float) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=True)
        self.register_buffer("mask", make_mask(self.weight.detach(), sparsity))
        self.bias = copy.deepcopy(base.bias) if isinstance(base.bias, nn.Module) else (lambda x: x)

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.mm(input, self.weight * self.mask)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output


class MaskedGraphConvolution(gamlp_layer.GraphConvolution):
    def __init__(self, base: gamlp_layer.GraphConvolution, sparsity: float) -> None:
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.alpha = base.alpha
        self.bns = base.bns
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=True)
        self.register_buffer("mask", make_mask(self.weight.detach(), sparsity))
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


def replace_layers(module: nn.Module, variant: str, sparsity: float, device: str) -> None:
    for name, child in list(module.named_children()):
        replacement = None
        if variant == "snack":
            if isinstance(child, nn.Linear):
                replacement = SnackLinear(child, sparsity=sparsity, device=device)
            elif isinstance(child, gamlp_layer.Dense):
                replacement = SnackDense(child, sparsity=sparsity, device=device)
            elif isinstance(child, gamlp_layer.GraphConvolution):
                replacement = SnackGraphConvolution(child, sparsity=sparsity, device=device)
        elif variant == "dense_mask":
            if isinstance(child, nn.Linear):
                replacement = MaskedLinear(child, sparsity=sparsity)
            elif isinstance(child, gamlp_layer.Dense):
                replacement = MaskedDense(child, sparsity=sparsity)
            elif isinstance(child, gamlp_layer.GraphConvolution):
                replacement = MaskedGraphConvolution(child, sparsity=sparsity)
        if replacement is not None:
            setattr(module, name, replacement)
        else:
            replace_layers(child, variant=variant, sparsity=sparsity, device=device)


def collect_snack_layers(module: nn.Module) -> List[Snack]:
    snacks: List[Snack] = []
    for m in module.modules():
        if isinstance(m, Snack):
            snacks.append(m)
        elif hasattr(m, "snack") and isinstance(m.snack, Snack):
            snacks.append(m.snack)
    # De-duplicate by object id.
    uniq = {}
    for s in snacks:
        uniq[id(s)] = s
    return list(uniq.values())


def build_gamlp_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        hidden=cli.hidden,
        num_hops=cli.num_hops,
        label_num_hops=cli.label_num_hops,
        seed=cli.seed,
        lr=cli.lr,
        dataset=cli.dataset,
        dropout=cli.dropout,
        gpu=0,
        weight_decay=cli.weight_decay,
        eval_every=1,
        batch_size=cli.batch_size,
        n_layers_1=cli.n_layers_1,
        n_layers_2=cli.n_layers_2,
        n_layers_3=cli.n_layers_3,
        num_runs=1,
        patience=100,
        alpha=cli.alpha,
        temp=1,
        threshold=0.8,
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
        emb_path="/data4/zwt/NARS-main",
        use_relation_subsets="/data4/zwt/NARS-main/sample_relation_subsets/examples/mag",
        use_rlu=cli.use_rlu,
        train_num_epochs=[0],
        stages=[cli.epochs],
        pre_dropout=cli.pre_dropout,
        bns=cli.bns,
    )


def evaluate(
    model: nn.Module,
    feats: List[torch.Tensor],
    labels: torch.Tensor,
    indices: torch.Tensor,
    label_emb: torch.Tensor | None,
    batch_size: int,
) -> float:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for start in range(0, indices.numel(), batch_size):
            idx = indices[start : start + batch_size]
            batch_feats = [x[idx] for x in feats]
            if label_emb is None:
                logits = model(batch_feats)
            else:
                logits = model(batch_feats, label_emb[idx])
            preds.append(logits.argmax(dim=1).cpu())
            ys.append(labels[idx].cpu())
    yhat = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return float((yhat == y).float().mean().item())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.batch_size == 1 and args.bns:
        print("batch_size=1 is incompatible with BatchNorm in training; disabling BN (equivalent to --no-bns).")
        args.bns = False
    torch.manual_seed(args.seed)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    gamlp_args = build_gamlp_args(args)
    feats, labels, in_size, num_classes, train_nid, val_nid, test_nid, _evaluator, label_emb = prepare_data(
        args.device, gamlp_args, teacher_probs=None
    )
    train_nid = train_nid.to(args.device)
    val_nid = val_nid.to(args.device)
    test_nid = test_nid.to(args.device)

    model = gen_model_rlu(gamlp_args, in_size, num_classes) if args.use_rlu else gen_model(gamlp_args, in_size, num_classes)
    if model is None:
        raise RuntimeError(
            f"Unsupported method/use_rlu combo: method={args.method}, use_rlu={args.use_rlu}. "
            "Use method in {R_GAMLP, JK_GAMLP} for non-RLU, or {R_GAMLP_RLU, JK_GAMLP_RLU} with --use-rlu."
        )
    model = model.to(args.device)

    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            uint16_keys = ckpt.get("state_dict_uint16_keys", [])
            if uint16_keys:
                state_dict = restore_uint16_indices_in_state_dict(state_dict, uint16_keys)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(ckpt)

    snack_layers: List[Snack] = []
    masked_layers: List[nn.Module] = []
    if args.variant in {"snack", "dense_mask"}:
        replace_layers(model, variant=args.variant, sparsity=args.sparsity, device=args.device)
        if args.variant == "snack":
            snack_layers = collect_snack_layers(model)
            if not snack_layers:
                raise RuntimeError("No SNACK layers found after replacement.")
            print(f"Converted to SNACK layers: {len(snack_layers)}")
        else:
            masked_layers = collect_masked_layers(model)
            print(f"Converted to dense_mask layers: {len(masked_layers)}")
    else:
        print("Using original dense GAMLP layers.")

    feats = [x.to(args.device) for x in feats]
    labels = labels.to(args.device).long()
    if label_emb is not None:
        label_emb = label_emb.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    power = PowerReader()

    best_val = -1.0
    best_epoch = -1
    metrics_rows: List[Dict[str, float | int | str]] = []

    try:
        model.train()
        for epoch in range(args.epochs):
            loss_vals: List[float] = []
            step_lat_ms: List[float] = []
            step_energy_mj: List[float] = []
            seen_samples = 0
            torch.cuda.reset_peak_memory_stats()
            epoch_wall_t0 = time.perf_counter()
            num_batches = (train_nid.numel() + args.batch_size - 1) // args.batch_size
            pbar = tqdm(
                range(0, train_nid.numel(), args.batch_size),
                total=num_batches,
                desc=f"epoch {epoch:03d}",
                leave=False,
                dynamic_ncols=True,
            )
            for step_idx, start in enumerate(pbar, start=1):
                idx = train_nid[start : start + args.batch_size]
                batch_feats = [x[idx] for x in feats]
                y = labels[idx]
                seen_samples += int(y.numel())
                optimizer.zero_grad(set_to_none=True)

                evt0 = torch.cuda.Event(enable_timing=True)
                evt1 = torch.cuda.Event(enable_timing=True)
                p0 = power.read_watts()
                evt0.record()
                if label_emb is None:
                    logits = model(batch_feats)
                else:
                    logits = model(batch_feats, label_emb[idx])
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                evt1.record()
                torch.cuda.synchronize()
                p1 = power.read_watts()

                elapsed_ms = float(evt0.elapsed_time(evt1))
                loss_vals.append(float(loss.item()))
                step_lat_ms.append(elapsed_ms)
                step_energy_mj.append(((p0 + p1) * 0.5) * elapsed_ms)
                if step_idx % max(1, args.progress_update_every) == 0:
                    elapsed_s = max(1e-6, time.perf_counter() - epoch_wall_t0)
                    pbar.set_postfix(
                        loss=f"{loss_vals[-1]:.4f}",
                        ms=f"{elapsed_ms:.2f}",
                        sps=f"{seen_samples / elapsed_s:.0f}",
                    )
            pbar.close()

            # DST prune/grow schedule for sparse variants.
            dst_applied = False
            if (epoch + 1) % args.dst_every == 0 and epoch <= args.dst_until_epoch:
                if args.variant == "snack":
                    for sn in snack_layers:
                        pg = ZetaPrunerGrower(sn, zeta=args.dst_zeta)
                        pg.prune()
                        pg.regrow(init=UniformInitializer, device=args.device)
                    torch.cuda.empty_cache()
                    dst_applied = True
                    print(f"[epoch {epoch}] DST prune/grow applied to SNACK (zeta={args.dst_zeta})")
                elif args.variant == "dense_mask":
                    apply_dense_mask_dst(masked_layers, zeta=args.dst_zeta)
                    dst_applied = True
                    print(f"[epoch {epoch}] DST prune/grow applied to dense_mask (zeta={args.dst_zeta})")

            epoch_time_s = time.perf_counter() - epoch_wall_t0
            lat_t = torch.tensor(step_lat_ms, dtype=torch.float32)
            en_t = torch.tensor(step_energy_mj, dtype=torch.float32)
            train_metrics = TrainEpochMetrics(
                latency_ms_mean=float(lat_t.mean().item()),
                latency_ms_p50=float(torch.quantile(lat_t, 0.50).item()),
                latency_ms_p95=float(torch.quantile(lat_t, 0.95).item()),
                energy_mj_mean=float(en_t.mean().item()),
                energy_mj_total=float(en_t.sum().item()),
                peak_allocated_mb=float(torch.cuda.max_memory_allocated() / 1e6),
                peak_reserved_mb=float(torch.cuda.max_memory_reserved() / 1e6),
                samples_per_sec=float(seen_samples / max(epoch_time_s, 1e-6)),
                epoch_time_s=float(epoch_time_s),
            )

            train_acc = evaluate(model, feats, labels, train_nid, label_emb, batch_size=max(256, args.batch_size))
            val_acc = evaluate(model, feats, labels, val_nid, label_emb, batch_size=max(256, args.batch_size))
            print(
                f"epoch={epoch:03d} loss={sum(loss_vals)/max(1,len(loss_vals)):.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"step_ms(mean/p95)={train_metrics.latency_ms_mean:.3f}/{train_metrics.latency_ms_p95:.3f} "
                f"energy_mJ(mean)={train_metrics.energy_mj_mean:.3f} "
                f"mem_mb(alloc/resv_peak)={train_metrics.peak_allocated_mb:.1f}/{train_metrics.peak_reserved_mb:.1f} "
                f"throughput(samples/s)={train_metrics.samples_per_sec:.1f}"
            )

            metrics_rows.append(
                {
                    "epoch": epoch,
                    "dataset": args.dataset,
                    "method": args.method,
                    "variant": args.variant,
                    "sparsity": args.sparsity,
                    "dst_zeta": args.dst_zeta,
                    "dst_applied": int(dst_applied),
                    "loss_mean": float(sum(loss_vals) / max(1, len(loss_vals))),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "step_latency_ms_mean": train_metrics.latency_ms_mean,
                    "step_latency_ms_p50": train_metrics.latency_ms_p50,
                    "step_latency_ms_p95": train_metrics.latency_ms_p95,
                    "step_energy_mj_mean": train_metrics.energy_mj_mean,
                    "epoch_energy_mj_total": train_metrics.energy_mj_total,
                    "epoch_energy_mj_per_sample": float(train_metrics.energy_mj_total / max(1, seen_samples)),
                    "peak_allocated_mb": train_metrics.peak_allocated_mb,
                    "peak_reserved_mb": train_metrics.peak_reserved_mb,
                    "samples_per_sec": train_metrics.samples_per_sec,
                    "epoch_time_s": train_metrics.epoch_time_s,
                    "train_samples": seen_samples,
                }
            )
            with args.metrics_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metrics_rows)

            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                save_state_dict, converted_uint16_keys = make_state_dict_serializable(model.state_dict())
                torch.save(
                    {
                        "state_dict": save_state_dict,
                        "state_dict_uint16_keys": converted_uint16_keys,
                        "method": args.method,
                        "dataset": args.dataset,
                        "variant": args.variant,
                        "sparsity": args.sparsity,
                        "dst_zeta": args.dst_zeta,
                        "epoch": epoch,
                    },
                    args.output_checkpoint,
                )
    finally:
        power.close()

    test_acc = evaluate(model, feats, labels, test_nid, label_emb, batch_size=max(256, args.batch_size))
    print(f"best_val={best_val:.4f} best_epoch={best_epoch} final_test={test_acc:.4f}")
    print(f"Saved {args.variant} checkpoint: {args.output_checkpoint}")
    print(f"Saved training metrics CSV: {args.metrics_csv}")


if __name__ == "__main__":
    main()
