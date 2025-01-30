"""
Sparse network active cuda kernel
"""

from sparse.mult.tensor import SparseTensor
import torch
import torch.nn.functional as F


class Snack(torch.nn.Module):
    def __init__(self, input_size, output_size, activation=F.relu, bias=True, device='cuda'):
        super(Snack, self).__init__()

        self.activation = activation
        self.layer = SparseTensor(*self._erdos_renyi_sparse_weights(input_size, output_size)).to(device)
        self.bias = torch.nn.Parameter(torch.zeros(output_size)) if bias else None

    def _erdos_renyi_sparse_weights(self, in_features, out_features, sparsity=0.1):
        """Generates a sparse weight matrix using Erdos-Renyi initialization."""
        num_elements = int(in_features * out_features * sparsity)

        # Generate random connections using Erdos-Renyi model
        indices = torch.vstack((torch.randint(0, in_features, (num_elements,)),
                                torch.randint(0, out_features, (num_elements,))))
        values = torch.randn(num_elements, requires_grad=True) / torch.sqrt(
            torch.tensor(in_features, dtype=torch.float))

        return indices, values, (in_features, out_features)

    def forward(self, x):
        x = self.layer @ (x)
        if self.bias is not None:
            x += self.bias
        return self.activation(x)

    def backward(self, grad_output):
        grad = grad_output
        if self.bias is not None:
            self.bias.grad = grad.sum(dim=0)
        print('backward', grad.shape, self.layer.shape)
        grad = self.layer @ grad  # Backpropagate through sparse layer
        return grad




