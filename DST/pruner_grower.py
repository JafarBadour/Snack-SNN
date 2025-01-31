import torch
from DST.layers.Snack import Snack
import DST.initializers.uniform_initializer as uni_init

class ZetaPrunerGrower:

    def __init__(self, model : Snack, zeta=0.1):
        self.zeta = zeta
        self.model = model


    def prune(self):
        indices = self.model.indices
        values = self.model.values

        # we decrease indices by zeta and then regrow by zeta

        nz = int(self.zeta * self.indices.size(0))
        _, idxs = torch.topk(indices, nz, largest=False)
        msk = torch.ones_like(indices, dtype=torch.bool)
        msk[idxs, :] = False
        values = values[msk]
        indices = indices[msk]
        return indices, values

    def regrow(self, init='uni_init', device='cpu'):
        indices = self.model.indices
        values = self.model.values

        # we decrease indices by zeta and then regrow by zeta

        nz = int(self.zeta * self.indices.size(0))
        if init != "uni_init":
            raise NotImplementedError(f"Initializer of type {init} is not implemented yet")

        new_indices = uni_init.init_without_replacement(self.model.size[0], self.model.size[1], nz, indices.cpu())
        new_indices = new_indices.to(device)
        _, idxs = torch.topk(indices, nz, largest=False)
        msk = torch.ones_like(indices, dtype=torch.bool)
        msk[idxs, :] = False
        values = values[msk]
        indices = indices[msk]
        return indices, values