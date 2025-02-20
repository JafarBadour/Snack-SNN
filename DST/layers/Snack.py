"""
Sparse network active cuda kernel
"""

import typing

from sparse.mult.tensor import SparseTensor, sparse_outer_join
import torch
from torch.autograd import Function
from DST.initializers.uniform_initializer import UniformInitializer
from DST.initializers.grand import SparseInitializer


class SparseFunc(Function):
    @staticmethod
    def forward(ctx, input, indices_a, indices_b, values, bias, input_shape, output_shape):

        ctx.save_for_backward(
            input,
            indices_a,
            indices_b,
            values,
            bias,
            torch.tensor([input_shape, output_shape]),
        )

        output = (
            SparseTensor(
                indices_a=indices_a,
                indices_b=indices_b,
                values=values,
                matrix_shape=(input_shape, output_shape),
            )
            @ input
        ) + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, indices_a, indices_b, values, bias, size = ctx.saved_tensors

        input_shape, output_shape = list(map(int, size))
        grad_input = None
        grad_bias = None
        grad_values = None
        # print(ctx.needs_input_grad, "######")
        sparse_tensor = SparseTensor(
            indices_a=indices_a,
            indices_b=indices_b,
            values=values,
            matrix_shape=(input_shape, output_shape),
        )
        if ctx.needs_input_grad[0]:
            grad_input = sparse_tensor.t() @ grad_output

        if ctx.needs_input_grad[3]:
            # grad_values = grad_output[:, indices_b] * input[:, indices_a]
            grad_values = sparse_outer_join(grad_output, indices_b, input, indices_a)

        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(dim=0)

        # print(ctx.needs_input_grad, grad_input)
        return grad_input, None, None, grad_values, grad_bias, None, None


class Snack(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        sparsity,
        values: torch.Tensor = None,
        initializer: typing.Type[SparseInitializer] = None,
        device="cuda",
        debug=False,
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
        self.bias = torch.randn(output_size).to(device).float()
        self.device = device
        if not (1 > sparsity >= 0):
            raise ValueError("Sparsity is out of range [0, 1[")
        self.sparsity = sparsity
        if initializer is None:
            raise TypeError(
                """initializer cannot be None you can use 
            `DST.initializers.uniform_initializer.UniformInitializer` or others in the initializers subdirectory"""
            )

        if not issubclass(initializer, SparseInitializer):
            raise TypeError("""initializer Must implement SparseInitializer""")

        self.indices_a, self.indices_b, self.values = self.__init__weights(
            initializer, input_size, output_size, sparsity
        )
        self.indices_a = torch.nn.Parameter(self.indices_a.to(device), requires_grad=False)
        self.indices_b = torch.nn.Parameter(self.indices_b.to(device), requires_grad=False)
        if values is not None:
            self.values = values
        self.values = self.values.to(device).float()
        self.values = torch.nn.Parameter(self.values)
        self.bias = torch.nn.Parameter(self.bias)
        self.debug = debug

    def __init__weights(
        self,
        init_cls__: typing.Type[SparseInitializer],
        in_features,
        out_features,
        sparsity=0.1,
    ):
        """Generates a sparse weight matrix using Erdos-Renyi initialization."""

        indices = init_cls__.initialize(in_features, out_features, sparsity=sparsity, device=self.device)
        indices_a, indices_b = indices[:, 0], indices[:, 1]

        values = torch.randn(indices.size(0)).float()
        sp = SparseTensor(
            indices_a=indices_a,
            indices_b=indices_b,
            values=values,
            matrix_shape=(in_features, out_features),
        )
        return sp.indices_a, sp.indices_b, sp.values

    def forward(self, x):
        return SparseFunc.apply(x, self.indices_a, self.indices_b, self.values, self.bias, self.size[0], self.size[1])

    def sparse_hash(self):
        return self.get_sp().hash()

    def get_sp(self):
        return SparseTensor(
            values=self.values,
            indices_a=self.indices_a,
            indices_b=self.indices_b,
            matrix_shape=self.size,
        )

    def __str__(self):
        return f"""
        Snack(indices_a={self.indices_a}, indices_b={self.indices_b}, values={self.values})
        """

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.values
        del self.indices_b
        del self.indices_a
        del self.bias
        torch.cuda.empty_cache()  # Clears unreferenced memory (optional)
