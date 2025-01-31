"""
Sparse network active cuda kernel
"""
import typing

from sparse.mult.tensor import SparseTensor, create_random_sparse_matrix
import torch
import torch.nn.functional as F
from torch.autograd import Function
import DST.initializers.uniform_initializer as uni_init

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
            # if grad_input is None:
            #     grad_input = sparse_tensor.t() @ grad_output

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

            grad_values = grad_output[indices_y] * input[indices_x]

            # print(grad_values)


        return grad_input, None, grad_values, None, None

class Snack(torch.nn.Module):
    def __init__(self, input_size, output_size, sparsity, init: typing.Union[str, None], device='cuda'):
        """

        :param input_size:
        :param output_size:
        :param sparsity:
        :param init: "uniform_initializer" or None
        :param device:
        """
        super(Snack, self).__init__()
        self.size = (input_size, output_size)
        if init == "uniform_initializer":
            self.indices, self.values = self.__uniform__init__weights(input_size, output_size, sparsity)
            self.indices = self.indices.to(device)

            self.values = self.values.to(device).float()
            self.values = torch.nn.Parameter(self.values)

        self.bias = None # add in future
        self.device = device
        self.sparsity = sparsity

    def __uniform__init__weights(self, in_features, out_features, sparsity=0.1):
        """Generates a sparse weight matrix using Erdos-Renyi initialization."""

        indices = uni_init.init(in_features, out_features, sparsity=sparsity, device=self.device)
        values = torch.rand(indices.size(0)).float()
        sp = SparseTensor(indices=indices, values=values, matrix_shape=(in_features, out_features))
        return sp.indices, sp.values

    def forward(self, x):
        return (SparseFunc.apply(x, self.indices, self.values, self.size[0], self.size[1])
                + (self.bias if self.bias is not None else 0))

    def sparse_hash(self):
        return SparseTensor(self.values, self.indices, self.size).hash()
