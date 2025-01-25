import torch
import typing as ty
from .exceptions import  CastingError
import warnings

from sparse_tensor_multiply import sparse_multiply as sparse_multiply_

IntegerTensor = ty.Type[torch.Tensor]




def sparse_multiply(indices : IntegerTensor, values : torch.Tensor, oned_tensor : torch.Tensor):
    return sparse_multiply_(oned_tensor, values, indices)

class SparseTensor:

    def __init__(self, indices : IntegerTensor, values : torch.Tensor) -> None:
        if indices.dtype != torch.int:
            warnings.warn("""Casting indices tensor to integer is a costly operation to be handled by SparseTensor it is 
            better to create this tensor with integer values upfront""")
            try:
                indices = indices.to(torch.int32)
            except Exception as e:
                raise CastingError(inp_dtype=str(indices.dtype), e=e)
        self.values = values
        self.indices = indices

    def __matmul__(self, other):
        """
            A = S @ B where S is the sparse tensor.
        :param other: B in the above case
        :return:
        """
        if len(other.shape) > 1:
            raise NotImplementedError("""Only 1-D tensor to be multiplied with the matrix""")
        return sparse_multiply(other, self.values, self.indices)







