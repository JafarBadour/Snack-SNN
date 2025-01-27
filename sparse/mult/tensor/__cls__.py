import torch
import typing as ty
from sparse.mult.tensor.exceptions import  CastingError
import warnings

from sparse_tensor_multiply import sparse_multiply as sparse_multiply_

IntegerTensor = ty.Type[torch.Tensor]




def sparse_multiply(oned_tensor : torch.Tensor, indices : IntegerTensor, values : torch.Tensor, output_shape : int):
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

    def __init__(self, indices : IntegerTensor, values : torch.Tensor, matrix_shape : ty.Union[int, ty.Tuple[int]]) -> None:

        if indices.dtype != torch.int:
            warnings.warn("""Casting indices tensor to integer is a costly operation to be handled by SparseTensor it is 
            better to create this tensor with integer values upfront""")
            try:
                indices = indices.to(dtype=torch.int32)
            except Exception as e:
                raise CastingError(inp_dtype=str(indices.dtype), e=e)
        self.values = values
        self.indices = indices

        if len(matrix_shape) > 2:
            raise NotImplementedError("""We support having only 1-D tensor output please raise a PR if you want to 
            contribute: https://github.com/JafarBadour/Parallel-Dynamic-Sparse-Training/pull/""")

        self.matrix_shape = matrix_shape
        self.shape = self.shape_calc()

    def __matmul__(self, other):
        """
            A = S @ B where S is the sparse tensor.
        :param other: B in the above case
        :return:
        """

        if len(other.shape) > 1:
            raise NotImplementedError("""Only 1-D tensor to be multiplied with the matrix""")
        return sparse_multiply(other, self.indices, self.values, self.matrix_shape[1])

    def dense(self) -> torch.Tensor:
        device = self.indices.device
        indices = self.indices# .cpu()
        values = self.values# .cpu()
        res = torch.zeros(self.matrix_shape).to(device)
        res[tuple(indices.T)] = values
        return res.to(device)

    @classmethod
    def __new_obj__(cls, indices, values, output_shape):
        return cls(indices, values, output_shape)

    def to(self, device):

        indices = self.indices.to(device)
        values = self.values.to(device)
        return SparseTensor.__new_obj__(indices, values, self.matrix_shape)

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    def __str__(self):
        return f"""SparseTensor(indices={self.indices}, \nvalues={self.values}, \n, matrix_shape={self.matrix_shape})"""

    def shape_calc(self):
        return f"""SparseTensor(indices={self.indices.shape}, \nvalues={self.values.shape}, \n, matrix_shape={self.matrix_shape})"""

    def __repr__(self):
        return self.__str__()










