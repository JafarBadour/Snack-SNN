from torch import nn
import torch


class MaskedDense(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None, device="cuda", mask_enabled=True):
        super(MaskedDense, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias).to(device)  # Linear layer
        self.weight_mask = torch.ones_like(self.linear.weight).to(device) # Initially all ones
        if mask_enabled:
            self.weight_mask = torch.nn.Parameter(self.weight_mask.to(device), requires_grad=False)
        self.weight_mask[0, :] = 0
        self.mask_enabled = mask_enabled
        self.activation = activation

    def forward(self, x):
        if self.mask_enabled:
            masked_weight = self.linear.weight * self.weight_mask
        else:
            masked_weight = self.linear.weight

        # Perform the linear transformation with the masked weight
        out = torch.matmul(x, masked_weight.T) + self.linear.bias

        return out
