"""
Sparse network active cuda kernel
"""

from sparse.mult.tensor import SparseTensor, create_random_sparse_matrix
import torch
import torch.nn.functional as F
from torch.autograd import Function


class SparseFunc(Function):
    @staticmethod
    def forward(ctx, input, indices, values, input_shape, output_shape):

        ctx.save_for_backward(input, indices, values, torch.Tensor((input_shape, output_shape)))
        output = SparseTensor(indices, values, (input_shape, output_shape)) @ input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, indices, values, size = ctx.saved_tensors

        input_shape, output_shape = list(map(int, size))
        grad_input = None
        grad_indices = None  # Important: Return None for indices (not differentiable)
        grad_values = None
        # print(ctx.needs_input_grad, "######")
        sparse_tensor = SparseTensor(indices, values, (input_shape, output_shape))
        if ctx.needs_input_grad[0]:  # Check if gradient w.r.t input is needed
            grad_input = sparse_tensor.t() @ grad_output

        if ctx.needs_input_grad[2]:  # Check if gradient w.r.t values is needed
            if grad_input is None:
                grad_input = sparse_tensor.t() @ grad_output

            # grad_values = (grad_output.t() @ input).view(-1)  # Correct gradient calculation for values bruh?


            indices_x = indices[:, 0]
            indices_y = indices[:, 1]
            # print(dict(indices_x=indices_x, indices_y=indices_y, grad_output=grad_output, grad_input=grad_input ))
            # print(indices.shape)
            # print('=========')
            # print(grad_output.shape, grad_input.shape)
            # print(grad_output)
            # print(grad_input)
            #
            # print(indices)
            # print(sparse_tensor.t())
            # print(f"{grad_output[indices_y]} * {grad_input[indices_x]}")

            grad_values = grad_output[indices_y] * grad_input[indices_x]

            # print(grad_values)


        return grad_input, None, grad_values, None, None

class Snack(torch.nn.Module):
    def __init__(self, input_size, output_size, sparsity, device='cuda'):
        super(Snack, self).__init__()

        self.indices, self.values = self._erdos_renyi_sparse_weights(input_size, output_size, sparsity)
        self.indices = self.indices.to(device)
        self.size = (input_size, output_size)
        self.values = self.values.to(device).float()
        self.values = torch.nn.Parameter(self.values)
        self.bias = None # add in future
        self.device = device
        self.sparsity = sparsity

    def _erdos_renyi_sparse_weights(self, in_features, out_features, sparsity=0.1):
        """Generates a sparse weight matrix using Erdos-Renyi initialization."""

        sp = create_random_sparse_matrix(in_features, out_features, sparsity)
        return sp.indices, sp.values

    def forward(self, x):
        return (SparseFunc.apply(x, self.indices, self.values, self.size[0], self.size[1])
                + (self.bias if self.bias is not None else 0))

    def sparse_hash(self):
        return SparseTensor(self.values, self.indices, self.size).hash()