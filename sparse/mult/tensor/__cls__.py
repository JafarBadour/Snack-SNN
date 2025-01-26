import torch
import typing as ty
from .exceptions import  CastingError
import warnings

from sparse_tensor_multiply import sparse_multiply as sparse_multiply_

IntegerTensor = ty.Type[torch.Tensor]




def sparse_multiply(indices : IntegerTensor, values : torch.Tensor, oned_tensor : torch.Tensor, output_shape : int):
    """

    :param indices: (-1, 2) tensor that contains X,Y where X corresponds to Neuron index from first layer and
    Y corresponds to Neuron index from the second layer
    :param values: values of the W matrix
    :param oned_tensor: activations from the first layer
    :param output_shape: for now it is 1D maybe we can create nD in the future size of activations of the second layer
    :return: result of multiplication of shape output_shape
    """
    return sparse_multiply_(oned_tensor, values, indices, output_shape)

class SparseTensor:

    def __init__(self, indices : IntegerTensor, values : torch.Tensor, output_shape : ty.Union[int, ty.Tuple[int]]) -> None:
        if indices.dtype != torch.int:
            warnings.warn("""Casting indices tensor to integer is a costly operation to be handled by SparseTensor it is 
            better to create this tensor with integer values upfront""")
            try:
                indices = indices.to(dtype=torch.int32)
            except Exception as e:
                raise CastingError(inp_dtype=str(indices.dtype), e=e)
        self.values = values
        self.indices = indices
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        if len(output_shape) > 1:
            raise NotImplementedError("""We support having only 1-D tensor output please raise a PR if you want to 
            contribute: https://github.com/JafarBadour/Parallel-Dynamic-Sparse-Training/pull/""")

        self.output_shape = output_shape

    def __matmul__(self, other):
        """
            A = S @ B where S is the sparse tensor.
        :param other: B in the above case
        :return:
        """
        if len(other.shape) > 1:
            raise NotImplementedError("""Only 1-D tensor to be multiplied with the matrix""")
        return sparse_multiply(other, self.values, self.indices, self.output_shape[0])







