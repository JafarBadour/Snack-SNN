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
    def forward(ctx, input, indices_a, indices_b, values, input_shape, output_shape):

        ctx.save_for_backward(
            input,
            indices_a,
            indices_b,
            values,
            torch.Tensor((input_shape, output_shape)),
        )

        output = (
            SparseTensor(
                indices_a=indices_a,
                indices_b=indices_b,
                values=values,
                matrix_shape=(input_shape, output_shape),
            )
            @ input
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, indices_a, indices_b, values, size = ctx.saved_tensors

        input_shape, output_shape = list(map(int, size))
        grad_input = None
        grad_indices = None  # Important: Return None for indices (not differentiable)
        grad_values = None
        # print(ctx.needs_input_grad, "######")
        sparse_tensor = SparseTensor(
            indices_a=indices_a,
            indices_b=indices_b,
            values=values,
            matrix_shape=(input_shape, output_shape),
        )
        if ctx.needs_input_grad[0]:  # Check if gradient w.r.t input is needed
            grad_input = sparse_tensor.t() @ grad_output

        if ctx.needs_input_grad[2]:  # Check if gradient w.r.t values is needed
            # if grad_input is None:
            #     grad_input = sparse_tensor.t() @ grad_output

            # grad_values = (grad_output.t() @ input).view(-1)  # Correct gradient calculation for values bruh?

            grad_values = grad_output[indices_a] * input[indices_b]

            # print(grad_values)

        return grad_input, None, None, grad_values, None, None


class Snack(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        sparsity,
        init: typing.Union[str] = "uniform_initializer",
        device="cuda",
    ):
        """

        :param input_size:
        :param output_size:
        :param sparsity:
        :param init: "uniform_initializer" or None
        :param device:
        """
        super(Snack, self).__init__()
        self.size = (input_size, output_size)
        self.bias = None  # add in future
        self.device = device
        self.sparsity = sparsity
        if init == "uniform_initializer":
            self.indices_a, self.indices_b, self.values = self.__uniform__init__weights(
                input_size, output_size, sparsity
            )
            self.indices_a = self.indices_a.to(device)
            self.indices_b = self.indices_b.to(device)

            self.values = self.values.to(device).float()
            self.values = torch.nn.Parameter(self.values)

    def __uniform__init__weights(self, in_features, out_features, sparsity=0.1):
        """Generates a sparse weight matrix using Erdos-Renyi initialization."""

        indices = uni_init.init(
            in_features, out_features, sparsity=sparsity, device=self.device
        )
        indices_a, indices_b = indices[:, 0], indices[:, 1]

        values = torch.rand(indices.size(0)).float()
        sp = SparseTensor(
            indices_a=indices_a,
            indices_b=indices_b,
            values=values,
            matrix_shape=(in_features, out_features),
        )
        return sp.indices_a, sp.indices_b, sp.values

    def forward(self, x):
        return SparseFunc.apply(
            x, self.indices_a, self.indices_b, self.values, self.size[0], self.size[1]
        ) + (self.bias if self.bias is not None else 0)

    def sparse_hash(self):
        return SparseTensor(
            values=self.values,
            indices_a=self.indices_a,
            indices_b=self.indices_b,
            matrix_shape=self.size,
        ).hash()
