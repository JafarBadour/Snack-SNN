#!/usr/bin/env python3
"""ViT Erdős–Rényi sparsity benchmark: Dense vs Dense+Mask vs SNACK.

Trains a small Vision Transformer from scratch on CIFAR-10 and sparsifies
selected nn.Linear layers with a shared Erdős–Rényi topology. Reports train/eval
loss and accuracy across sparsity levels and backends.

Variants
--------
- dense          : full dense baseline (sparsity ignored)
- dense_mask     : dense matmul with ER binary mask
- snack_sputnik  : SNACK layer, Sputnik SpMM forward
- snack_coo      : SNACK layer, custom COO kernel (slow at low η / large batch)

Example
-------
python experiments/vit_er_snack_benchmark/run_vit_er_snack_benchmark.py \\
  --sparsities 0.25 0.50 0.75 0.85 0.90 0.95 \\
  --variants dense dense_mask snack_sputnik \\
  --epochs 50 --batch-size 128
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from DST.initializers.uniform_initializer import UniformInitializer
from DST.layers import Snack, SputnikSnackFunc
from DST.pruner_grower import ZetaPrunerGrower


# ---------------------------------------------------------------------------
# Tiny ViT (DeiT-Ti / ViT-Tiny style) for CIFAR
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, N, D]
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


MODEL_PRESETS = {
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0),
}


# ---------------------------------------------------------------------------
# Sparse Linear wrappers (nn.Linear weight is [out, in]; SNACK uses [in, out])
# ---------------------------------------------------------------------------


def load_sputnik_module():
    ext_dir = Path(__file__).resolve().parents[2] / "benchmark" / "sparse_matrix_multi" / "sputnik_torch_ext"
    if ext_dir.is_dir() and str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    try:
        import sputnik_torch_ext  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "SNACK backend 'sputnik' requested but sputnik_torch_ext is not importable.\n"
            "Build it with:\n"
            "  bash benchmark/sparse_matrix_multi/install_sputnik_torch.sh\n"
            "Then ensure PYTHONPATH includes benchmark/sparse_matrix_multi/sputnik_torch_ext."
        ) from exc
    return sputnik_torch_ext


def erdos_renyi_indices(in_features: int, out_features: int, sparsity: float, device: str) -> torch.Tensor:
    """Return [nnz, 2] int32 indices (in_idx, out_idx) for an Erdős–Rényi support.

    Uses UniformInitializer when sparsity > 0.5 (CUDA kernel constraint).
    Otherwise falls back to exact-count randperm ER sampling.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    # random_init_without_replacement accepts sparsity > 0.5 only.
    if sparsity <= 0.5:
        nnz = max(1, int(round((1.0 - sparsity) * in_features * out_features)))
        flat = torch.randperm(in_features * out_features, device=device)[:nnz]
        rows = (flat // out_features).to(torch.int32)
        cols = (flat % out_features).to(torch.int32)
        return torch.stack([rows, cols], dim=1)
    return UniformInitializer.initialize(
        in_features, out_features, device=device, sparsity=sparsity
    ).to(dtype=torch.int32)


def indices_to_mask(indices: torch.Tensor, in_features: int, out_features: int, device: str) -> torch.Tensor:
    """Build [out, in] mask for nn.Linear from ER indices [nnz, 2]=(in, out)."""
    mask = torch.zeros(out_features, in_features, device=device)
    mask[indices[:, 1].long(), indices[:, 0].long()] = 1.0
    return mask


class DenseMaskLinear(nn.Module):
    """Dense matmul with an Erdős–Rényi binary mask (Dense+Mask), optional DST."""

    def __init__(self, linear: nn.Linear, mask: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
        self.register_buffer("mask", mask.detach().to(dtype=self.weight.dtype))
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def apply_mask_(self) -> None:
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def dst_set_step(self, zeta: float) -> None:
        """SET: magnitude prune + random regrow."""
        with torch.no_grad():
            flat_w = self.weight.view(-1)
            flat_m = self.mask.view(-1)
            active_idx = torch.where(flat_m > 0)[0]
            inactive_idx = torch.where(flat_m <= 0)[0]
            if active_idx.numel() <= 1 or inactive_idx.numel() == 0:
                return
            n_drop = max(1, int(round(zeta * int(active_idx.numel()))))
            n_drop = min(n_drop, int(active_idx.numel()) - 1, int(inactive_idx.numel()))
            prune_rel = torch.topk(flat_w[active_idx].abs(), k=n_drop, largest=False).indices
            prune_idx = active_idx[prune_rel]
            grow_idx = inactive_idx[torch.randperm(inactive_idx.numel(), device=inactive_idx.device)[:n_drop]]
            flat_m[prune_idx] = 0
            flat_w[prune_idx] = 0
            bound = math.sqrt(6.0 / float(self.weight.shape[0] + self.weight.shape[1]))
            flat_m[grow_idx] = 1
            flat_w[grow_idx] = torch.empty(n_drop, device=flat_w.device).uniform_(-bound, bound)

    def dst_rigl_step(self, zeta: float) -> None:
        """RiGL: magnitude prune + grow by |grad| on inactive weights."""
        with torch.no_grad():
            flat_w = self.weight.view(-1)
            flat_m = self.mask.view(-1)
            active_idx = torch.where(flat_m > 0)[0]
            inactive_idx = torch.where(flat_m <= 0)[0]
            if active_idx.numel() <= 1 or inactive_idx.numel() == 0:
                return
            n_drop = max(1, int(round(zeta * int(active_idx.numel()))))
            n_drop = min(n_drop, int(active_idx.numel()) - 1, int(inactive_idx.numel()))
            prune_rel = torch.topk(flat_w[active_idx].abs(), k=n_drop, largest=False).indices
            prune_idx = active_idx[prune_rel]
            flat_m[prune_idx] = 0
            flat_w[prune_idx] = 0
            # Recompute inactive after prune.
            inactive_idx = torch.where(flat_m <= 0)[0]
            if self.weight.grad is None:
                grow_idx = inactive_idx[torch.randperm(inactive_idx.numel(), device=inactive_idx.device)[:n_drop]]
            else:
                flat_g = self.weight.grad.detach().view(-1).abs()
                grow_rel = torch.topk(flat_g[inactive_idx], k=n_drop, largest=True).indices
                grow_idx = inactive_idx[grow_rel]
            bound = math.sqrt(6.0 / float(self.weight.shape[0] + self.weight.shape[1]))
            flat_m[grow_idx] = 1
            flat_w[grow_idx] = torch.empty(n_drop, device=flat_w.device).uniform_(-bound, bound)


class SnackLinear(nn.Module):
    """Wrap SNACK as a drop-in nn.Linear replacement."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        indices: torch.Tensor,
        values: torch.Tensor,
        bias: Optional[torch.Tensor],
        device: str,
        backend: str,
    ):
        super().__init__()
        if backend not in {"sparse_tensor", "sputnik"}:
            raise ValueError(f"Unknown SNACK backend: {backend}")
        # Dense [in, out] carrying only ER nonzeros (zeros elsewhere).
        dense_w = torch.zeros(in_features, out_features, device=device, dtype=values.dtype)
        dense_w[indices[:, 0].long(), indices[:, 1].long()] = values.to(device)
        self.snack = Snack(
            input_size=in_features,
            output_size=out_features,
            sparsity=0.0,
            dense_weight=dense_w,
            initializer=UniformInitializer,
            bias=bias is not None,
            device=device,
        )
        self.snack.indices_a.data = self.snack.indices_a.data.to(dtype=torch.uint16)
        self.snack.indices_b.data = self.snack.indices_b.data.to(dtype=torch.uint16)
        if bias is not None:
            with torch.no_grad():
                self.snack.bias.copy_(bias.detach().to(device=device, dtype=values.dtype))
        self.input_size = in_features
        self.output_size = out_features
        self.backend = backend
        self.sputnik_mod = None
        self.sputnik_row_indices: Optional[torch.Tensor] = None
        self.sputnik_row_offsets: Optional[torch.Tensor] = None
        self.sputnik_column_indices: Optional[torch.Tensor] = None
        self.sputnik_value_order: Optional[torch.Tensor] = None
        # Buffers for RiGL growth criterion (updated each forward/backward).
        self._last_x: Optional[torch.Tensor] = None
        self._last_gy: Optional[torch.Tensor] = None
        if self.backend == "sputnik":
            self.sputnik_mod = load_sputnik_module()
            self._refresh_sputnik_cache()

    def _refresh_sputnik_cache(self) -> None:
        with torch.no_grad():
            row = self.snack.indices_b.detach().to(dtype=torch.int64)  # out
            col = self.snack.indices_a.detach().to(dtype=torch.int64)  # in
            order = torch.argsort(row * self.input_size + col)
            row_sorted = row.index_select(0, order)
            col_sorted = col.index_select(0, order)
            counts = torch.bincount(row_sorted, minlength=self.output_size)
            row_offsets = torch.empty(self.output_size + 1, device=row.device, dtype=torch.int64)
            row_offsets[0] = 0
            row_offsets[1:] = torch.cumsum(counts, dim=0)
            self.sputnik_row_indices = torch.arange(self.output_size, dtype=torch.int32, device=row.device)
            self.sputnik_row_offsets = row_offsets.to(dtype=torch.int32)
            self.sputnik_column_indices = col_sorted.to(dtype=torch.int32)
            self.sputnik_value_order = order.to(dtype=torch.int64)

    def _cast_indices_uint16(self) -> None:
        self.snack.indices_a.data = self.snack.indices_a.data.to(dtype=torch.uint16)
        self.snack.indices_b.data = self.snack.indices_b.data.to(dtype=torch.uint16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        # Keep a small activation snapshot for RiGL (cap rows for memory).
        with torch.no_grad():
            if x2d.size(0) > 2048:
                idx = torch.randperm(x2d.size(0), device=x2d.device)[:2048]
                self._last_x = x2d.index_select(0, idx).detach()
            else:
                self._last_x = x2d.detach()

        if self.backend == "sputnik":
            assert self.sputnik_mod is not None
            y2d = SputnikSnackFunc.apply(
                x2d,
                self.snack.indices_a,
                self.snack.indices_b,
                self.snack.values,
                self.snack.bias,
                self.sputnik_row_indices,
                self.sputnik_row_offsets,
                self.sputnik_column_indices,
                self.sputnik_value_order,
                self.input_size,
                self.output_size,
                self.sputnik_mod,
            )
        else:
            y2d = self.snack(x2d)

        if y2d.requires_grad:
            def _save_gy(grad: torch.Tensor) -> None:
                with torch.no_grad():
                    if self._last_x is not None and grad.size(0) != self._last_x.size(0):
                        # Align to the stored activation subsample if needed.
                        self._last_gy = grad[: self._last_x.size(0)].detach()
                    else:
                        self._last_gy = grad.detach()

            y2d.register_hook(_save_gy)
        return y2d.view(*x.shape[:-1], y2d.size(-1))

    def dst_set_step(self, zeta: float, device: str) -> None:
        """SET: magnitude prune + random Erdős–Rényi regrow (ZetaPrunerGrower)."""
        pg = ZetaPrunerGrower(self.snack, zeta=zeta)
        pg.prune()
        pg.regrow(init=UniformInitializer, device=device)
        self._cast_indices_uint16()
        if self.backend == "sputnik":
            self._refresh_sputnik_cache()

    def dst_rigl_step(self, zeta: float, device: str) -> None:
        """RiGL: magnitude prune + grow by |X^T dY| on inactive connections."""
        pg = ZetaPrunerGrower(self.snack, zeta=zeta)
        n_drop = pg.nz
        if n_drop <= 0:
            return
        pg.prune()
        if self._last_x is None or self._last_gy is None:
            pg.regrow(init=UniformInitializer, device=device)
        else:
            x = self._last_x
            gy = self._last_gy
            if x.size(0) != gy.size(0):
                n = min(x.size(0), gy.size(0))
                x, gy = x[:n], gy[:n]
            score = (x.transpose(0, 1) @ gy).abs()  # [in, out]
            ia = self.snack.indices_a.detach().to(dtype=torch.int64)
            ib = self.snack.indices_b.detach().to(dtype=torch.int64)
            score[ia, ib] = -1.0
            flat = score.reshape(-1)
            grow_flat = torch.topk(flat, k=n_drop, largest=True).indices
            new_a = (grow_flat // self.output_size).to(dtype=torch.uint16)
            new_b = (grow_flat % self.output_size).to(dtype=torch.uint16)
            bound = math.sqrt(6.0 / float(self.input_size + self.output_size))
            new_vals = torch.empty(n_drop, device=device, dtype=self.snack.values.dtype).uniform_(-bound, bound)
            self.snack.indices_a.data = torch.cat([self.snack.indices_a.data.to(dtype=torch.uint16), new_a])
            self.snack.indices_b.data = torch.cat([self.snack.indices_b.data.to(dtype=torch.uint16), new_b])
            self.snack.values.data = torch.cat([self.snack.values.data, new_vals])
        self._cast_indices_uint16()
        if self.backend == "sputnik":
            self._refresh_sputnik_cache()


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------


def iter_target_linears(
    model: VisionTransformer, sparse_attn: bool
) -> Iterable[Tuple[str, nn.Module, str]]:
    """Yield (layer_id, parent_module, attr_name) for sparsifiable Linear layers."""
    for bi, block in enumerate(model.blocks):
        yield f"blocks.{bi}.mlp.fc1", block.mlp, "fc1"
        yield f"blocks.{bi}.mlp.fc2", block.mlp, "fc2"
        if sparse_attn:
            yield f"blocks.{bi}.attn.qkv", block.attn, "qkv"
            yield f"blocks.{bi}.attn.proj", block.attn, "proj"


def count_linear_params(model: VisionTransformer) -> Tuple[int, int]:
    total = 0
    linear = 0
    for p in model.parameters():
        total += p.numel()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear += m.weight.numel()
            if m.bias is not None:
                linear += m.bias.numel()
    return total, linear


def sparsify_model(
    model: VisionTransformer,
    variant: str,
    sparsity: float,
    device: str,
    sparse_attn: bool,
    topology_seed: int,
) -> Dict[str, float]:
    """Replace target Linear layers. Returns sparsity bookkeeping stats."""
    if variant == "dense":
        return {"target_weight_params": 0, "kept_weight_params": 0, "realized_sparsity": 0.0}

    backend = "sputnik" if variant == "snack_sputnik" else "sparse_tensor" if variant == "snack_coo" else None
    target_weight = 0
    kept_weight = 0

    # Deterministic ER topology per layer id, shared across backends when the
    # same topology_seed / sparsity are used.
    for layer_id, parent, attr in iter_target_linears(model, sparse_attn=sparse_attn):
        linear: nn.Linear = getattr(parent, attr)
        out_f, in_f = linear.weight.shape
        target_weight += in_f * out_f

        layer_key = f"{layer_id}:{in_f}x{out_f}:{sparsity}:{topology_seed}"
        layer_seed = int(hashlib.md5(layer_key.encode()).hexdigest()[:8], 16)
        torch.manual_seed(layer_seed)
        indices = erdos_renyi_indices(in_f, out_f, sparsity=sparsity, device=device)
        kept_weight += int(indices.size(0))

        # Shared nonzero init for Dense+Mask and SNACK under the same topology seed.
        torch.manual_seed(layer_seed + 1)
        values = torch.randn(indices.size(0), device=device, dtype=linear.weight.dtype)
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        if variant == "dense_mask":
            mask = indices_to_mask(indices, in_f, out_f, device=device)
            dm = DenseMaskLinear(linear, mask)
            with torch.no_grad():
                dm.weight.zero_()
                dm.weight[indices[:, 1].long(), indices[:, 0].long()] = values
            setattr(parent, attr, dm)
        elif variant in {"snack_coo", "snack_sputnik"}:
            assert backend is not None
            setattr(
                parent,
                attr,
                SnackLinear(
                    in_features=in_f,
                    out_features=out_f,
                    indices=indices,
                    values=values,
                    bias=bias,
                    device=device,
                    backend=backend,
                ),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

    realized = 1.0 - (kept_weight / max(1, target_weight))
    return {
        "target_weight_params": float(target_weight),
        "kept_weight_params": float(kept_weight),
        "realized_sparsity": float(realized),
    }


def fraction_sparsified(model: VisionTransformer, sparse_attn: bool) -> float:
    total, _ = count_linear_params(model)
    target = 0
    for _layer_id, parent, attr in iter_target_linears(model, sparse_attn=sparse_attn):
        linear: nn.Linear = getattr(parent, attr)
        target += linear.weight.numel()
    # Recompute total params after counting targets from a fresh dense model would
    # be ideal; here we approximate using current module tree (still dense).
    n_params = sum(p.numel() for p in model.parameters())
    return target / max(1, n_params)


# ---------------------------------------------------------------------------
# Data / train / eval
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SyntheticCIFAR(torch.utils.data.Dataset):
    """Tiny random dataset for smoke tests when CIFAR-10 is unavailable."""

    def __init__(self, n: int, img_size: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, img_size, img_size, generator=g)
        self.targets = torch.randint(0, 10, (n,), generator=g)

    def __len__(self) -> int:
        return int(self.images.size(0))

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.targets[idx])


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
    synthetic: bool = False,
    synthetic_train_size: int = 256,
    synthetic_eval_size: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    if synthetic:
        train_ds = SyntheticCIFAR(synthetic_train_size, img_size, seed=0)
        eval_ds = SyntheticCIFAR(synthetic_eval_size, img_size, seed=1)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
        eval_ds = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, eval_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += int(targets.numel())
    return total_loss / max(1, total), total_correct / max(1, total)


def apply_dst_(model: nn.Module, dst_algo: str, zeta: float, device: str) -> bool:
    """Apply one SET/RiGL update to all sparse layers. Returns True if any update ran."""
    if dst_algo in {"none", "", None}:
        return False
    touched = False
    for m in model.modules():
        if isinstance(m, DenseMaskLinear):
            if dst_algo == "set":
                m.dst_set_step(zeta)
            elif dst_algo == "rigl":
                m.dst_rigl_step(zeta)
            else:
                raise ValueError(f"Unknown dst_algo: {dst_algo}")
            m.apply_mask_()
            touched = True
        elif isinstance(m, SnackLinear):
            if dst_algo == "set":
                m.dst_set_step(zeta, device=device)
            elif dst_algo == "rigl":
                m.dst_rigl_step(zeta, device=device)
            else:
                raise ValueError(f"Unknown dst_algo: {dst_algo}")
            touched = True
    return touched


def rebuild_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_steps: int = 0,
    dst_algo: str = "none",
    dst_zeta: float = 0.05,
    do_dst_at_end: bool = False,
) -> Tuple[float, float, int, torch.optim.Optimizer]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    steps = 0
    n_batches = len(loader) if max_steps <= 0 else min(len(loader), max_steps)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        for m in model.modules():
            if isinstance(m, DenseMaskLinear):
                m.apply_mask_()

        # DST after the step so RiGL can use this batch's gradients / activations.
        is_last = steps + 1 >= n_batches
        if do_dst_at_end and is_last and dst_algo in {"set", "rigl"}:
            if apply_dst_(model, dst_algo=dst_algo, zeta=dst_zeta, device=device):
                optimizer = rebuild_optimizer(
                    model,
                    lr=optimizer.param_groups[0]["lr"],
                    weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
                )

        with torch.no_grad():
            total_loss += float(loss.item()) * int(targets.numel())
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break
    return total_loss / max(1, total), total_correct / max(1, total), steps, optimizer


@dataclass
class RunResult:
    variant: str
    dst_algo: str
    sparsity: float
    realized_sparsity: float
    fraction_params_sparsified: float
    seed: int
    epochs: int
    best_eval_acc: float
    best_eval_loss: float
    final_train_loss: float
    final_train_acc: float
    final_eval_loss: float
    final_eval_acc: float
    wall_time_s: float
    peak_mem_mb: float


def build_model(args: argparse.Namespace) -> VisionTransformer:
    preset = MODEL_PRESETS[args.model]
    return VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=10,
        dropout=args.dropout,
        **preset,
    )


def scheduled_zeta(epoch: int, epochs: int, zeta0: float, dst_end_frac: float) -> float:
    """Cosine-decay drop fraction; stop DST after dst_end_frac of training."""
    end_epoch = max(1, int(round(epochs * dst_end_frac)))
    if epoch >= end_epoch:
        return 0.0
    progress = float(epoch) / float(end_epoch)
    return float(zeta0 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def run_single(
    args: argparse.Namespace,
    variant: str,
    sparsity: float,
    dst_algo: str,
    train_loader: DataLoader,
    eval_loader: DataLoader,
) -> RunResult:
    set_seed(args.seed)
    device = args.device
    model = build_model(args).to(device)

    # Dense never uses DST.
    effective_dst = "none" if variant == "dense" else dst_algo

    # Measure sparsifiable fraction on the dense model before surgery.
    frac = fraction_sparsified(model, sparse_attn=args.sparse_attn)
    stats = sparsify_model(
        model,
        variant=variant,
        sparsity=0.0 if variant == "dense" else sparsity,
        device=device,
        sparse_attn=args.sparse_attn,
        topology_seed=args.seed,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    best_acc = -1.0
    best_loss = float("inf")
    final_train_loss = float("nan")
    final_train_acc = float("nan")
    final_eval_loss = float("nan")
    final_eval_acc = float("nan")

    for epoch in range(args.epochs):
        zeta = scheduled_zeta(epoch, args.epochs, args.dst_zeta, args.dst_end_frac)
        do_dst = (
            effective_dst in {"set", "rigl"}
            and args.dst_interval_epochs > 0
            and ((epoch + 1) % args.dst_interval_epochs == 0)
            and zeta > 0.0
        )
        train_loss, train_acc, _, optimizer = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_steps=args.max_train_steps_per_epoch,
            dst_algo=effective_dst,
            dst_zeta=zeta,
            do_dst_at_end=do_dst,
        )
        # Keep scheduler tied to the (possibly rebuilt) optimizer.
        if len(scheduler.optimizer.param_groups) != len(optimizer.param_groups):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs), last_epoch=epoch - 1
            )
        else:
            scheduler.optimizer = optimizer
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        scheduler.step()
        final_train_loss, final_train_acc = train_loss, train_acc
        final_eval_loss, final_eval_acc = eval_loss, eval_acc
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_loss = eval_loss
        dst_tag = f" dst={effective_dst}" if variant != "dense" else ""
        zeta_tag = f" ζ={zeta:.4f}" if do_dst else ""
        print(
            f"[{variant} η={sparsity:.2f}{dst_tag}] epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}{zeta_tag}",
            flush=True,
        )

    wall = time.perf_counter() - t0
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
    return RunResult(
        variant=variant,
        dst_algo=effective_dst,
        sparsity=0.0 if variant == "dense" else sparsity,
        realized_sparsity=stats["realized_sparsity"],
        fraction_params_sparsified=frac,
        seed=args.seed,
        epochs=args.epochs,
        best_eval_acc=best_acc,
        best_eval_loss=best_loss,
        final_train_loss=final_train_loss,
        final_train_acc=final_train_acc,
        final_eval_loss=final_eval_loss,
        final_eval_acc=final_eval_acc,
        wall_time_s=wall,
        peak_mem_mb=peak_mb,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViT ER sparsity benchmark with SNACK backends.")
    p.add_argument("--model", type=str, default="vit_tiny", choices=sorted(MODEL_PRESETS.keys()))
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random tensors instead of CIFAR-10 (smoke tests only).",
    )
    p.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 0.85, 0.90, 0.95],
    )
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["dense", "dense_mask", "snack_sputnik"],
        choices=["dense", "dense_mask", "snack_sputnik", "snack_coo"],
    )
    p.add_argument(
        "--dst-algos",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", "set", "rigl"],
        help="DST algorithms for sparse variants. Dense always uses none. "
        "SET = magnitude prune + random regrow; RiGL = magnitude prune + grad grow.",
    )
    p.add_argument("--dst-interval-epochs", type=int, default=1, help="Apply DST every N epochs.")
    p.add_argument("--dst-zeta", type=float, default=0.05, help="Initial fraction of connections to rewire.")
    p.add_argument(
        "--dst-end-frac",
        type=float,
        default=0.8,
        help="Stop DST after this fraction of epochs (cosine-decay zeta until then).",
    )
    p.add_argument(
        "--sparse-attn",
        action="store_true",
        help="Also sparsify attention qkv/proj Linear layers (MLP is always sparsified).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-train-steps-per-epoch",
        type=int,
        default=0,
        help="If >0, cap training steps per epoch (useful for smoke tests).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/vit_er_snack_benchmark/results/vit_tiny_cifar10"),
    )
    p.add_argument("--disable-output-hash", action="store_true")
    return p.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.disable_output_hash:
        out = args.output_dir
    else:
        payload = json.dumps(vars(args), sort_keys=True, default=str)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
        out = args.output_dir / f"run_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_summary_table(rows: List[RunResult], path: Path) -> None:
    """Wide CSV: one row per sparsity, columns per (variant, dst_algo) metric."""
    keys = sorted({(r.variant, r.dst_algo) for r in rows})
    sparsities = sorted({r.sparsity for r in rows if r.variant != "dense"})
    dense = next((r for r in rows if r.variant == "dense"), None)

    def col(v: str, d: str, metric: str) -> str:
        return f"{v}_{d}_{metric}" if v != "dense" else f"dense_{metric}"

    fieldnames = ["sparsity"]
    for v, d in keys:
        for metric in ("eval_acc", "eval_loss", "train_acc", "train_loss", "peak_mem_mb"):
            name = col(v, d, metric)
            if name not in fieldnames:
                fieldnames.append(name)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if dense is not None:
            row = {"sparsity": 0.0, "dense_eval_acc": dense.best_eval_acc, "dense_eval_loss": dense.best_eval_loss,
                   "dense_train_acc": dense.final_train_acc, "dense_train_loss": dense.final_train_loss,
                   "dense_peak_mem_mb": dense.peak_mem_mb}
            w.writerow(row)
        for sp in sparsities:
            row: Dict[str, object] = {"sparsity": sp}
            for v, d in keys:
                if v == "dense":
                    continue
                match = next(
                    (r for r in rows if r.variant == v and r.dst_algo == d and abs(r.sparsity - sp) < 1e-9),
                    None,
                )
                if match is None:
                    continue
                row[col(v, d, "eval_acc")] = match.best_eval_acc
                row[col(v, d, "eval_loss")] = match.best_eval_loss
                row[col(v, d, "train_acc")] = match.final_train_acc
                row[col(v, d, "train_loss")] = match.final_train_loss
                row[col(v, d, "peak_mem_mb")] = match.peak_mem_mb
            w.writerow(row)


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if "snack_coo" in args.variants:
        print(
            "WARNING: snack_coo (SNACK-COO) is often prohibitively slow for ViT "
            "training batches; consider omitting it or restricting to η>=0.90.",
            flush=True,
        )

    out_dir = resolve_output_dir(args)
    with (out_dir / "cfg.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"Output directory: {out_dir}", flush=True)
    train_loader, eval_loader = build_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.img_size,
        synthetic=args.synthetic,
    )

    # Report how much of the model we sparsify (on a fresh dense model).
    probe = build_model(args)
    frac = fraction_sparsified(probe, sparse_attn=args.sparse_attn)
    n_params = sum(p.numel() for p in probe.parameters())
    print(
        f"Model={args.model} params={n_params/1e6:.2f}M "
        f"fraction_weight_params_sparsified≈{frac*100:.1f}% "
        f"(sparse_attn={args.sparse_attn})",
        flush=True,
    )
    del probe

    results: List[RunResult] = []
    variants = list(args.variants)
    sparse_variants = [v for v in variants if v != "dense"]
    dst_algos = list(args.dst_algos)

    if "dense" in variants:
        print("=== Dense baseline ===", flush=True)
        results.append(run_single(args, "dense", 0.0, "none", train_loader, eval_loader))

    for sparsity in args.sparsities:
        for variant in sparse_variants:
            for dst_algo in dst_algos:
                print(f"=== {variant} sparsity={sparsity} dst={dst_algo} ===", flush=True)
                results.append(
                    run_single(args, variant, sparsity, dst_algo, train_loader, eval_loader)
                )

    # Persist
    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    write_summary_table(results, out_dir / "summary_by_sparsity.csv")
    with (out_dir / "results.json").open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print("\n=== Summary (best eval acc) ===", flush=True)
    print(
        f"{'variant':<16} {'dst':<6} {'η':>6} {'eval_acc':>10} {'eval_loss':>10} {'mem_mb':>10}",
        flush=True,
    )
    for r in results:
        print(
            f"{r.variant:<16} {r.dst_algo:<6} {r.sparsity:6.2f} {r.best_eval_acc:10.4f} "
            f"{r.best_eval_loss:10.4f} {r.peak_mem_mb:10.1f}",
            flush=True,
        )
    print(f"\nWrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
