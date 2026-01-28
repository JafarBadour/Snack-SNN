import torch
import typing
import math

from DST.layers.Snack import Snack
import DST.initializers.uniform_initializer as uni_init
from DST.initializers.grand import SparseInitializer


class ZetaPrunerGrower:

    def __init__(self, layer: Snack, zeta=0.1):
        self.zeta = zeta
        self.layer = layer
        self.nz = int(self.zeta * self.layer.values.size(0))

    def prune(self):
        indices_a = self.layer.indices_a
        indices_b = self.layer.indices_b
        values = self.layer.values

        # we decrease indices by zeta and then regrow by zeta

        _, idxs = torch.topk(values, self.nz, largest=False)
        msk = torch.ones_like(indices_a, dtype=torch.bool)

        msk[idxs] = False
        values.data = values[msk]
        indices_a.data = indices_a.to(dtype=torch.int32)[msk].to(dtype=torch.uint16).data
        indices_b.data = indices_b.to(dtype=torch.int32)[msk].to(dtype=torch.uint16).data
        # Note: This operation could be optimized with CUDA implementation for better performance

    def regrow(self, init: typing.Type[SparseInitializer] = None, device="cpu"):
        indices_a = self.layer.indices_a
        indices_b = self.layer.indices_b
        values = self.layer.values

        # we decrease indices by zeta and then regrow by zeta

        nz = self.nz
        if not issubclass(init, SparseInitializer):
            raise TypeError("""initializer Must implement SparseInitializer""")

        new_indices = init.initialize(
            self.layer.size[0],
            self.layer.size[1],
            nz=nz,
            built_edges=torch.concat([indices_a.reshape(-1, 1).cpu(), indices_b.reshape(-1, 1).cpu()], dim=1),
        ).to(dtype=torch.uint16)

        new_indices = new_indices.to(device)

        indices_a.data = torch.concat([indices_a, new_indices[:, 0]])
        indices_b.data = torch.concat([indices_b, new_indices[:, 1]])
        # v_mean = values.mean()
        a = -1 * math.sqrt(6) / math.sqrt(sum(self.layer.size)) # Xavier Initialization
        b = a * -1
        values.data = torch.concat([values.data, a + (b-a) * torch.rand(nz, device=device) ])
