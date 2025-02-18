from torch import nn
import torch


class Dense(nn.Module):
    def __init__(
        self, input_size, output_size, bias=True, activation=None, device="cuda"
    ):
        super(Dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias).to(
            device
        )  # Linear layer
        self.weight_mask = torch.ones_like(self.linear.weight).to(
            device
        )  # Initially all ones
        self.weight_mask = self.weight_mask.to(device)
        self.weight_mask[0, :] = 0

        self.activation = activation

    def forward(self, x):
        masked_weight = self.linear.weight * self.weight_mask

        # Perform the linear transformation with the masked weight
        out = torch.matmul(x, masked_weight.T) + self.linear.bias

        return out
