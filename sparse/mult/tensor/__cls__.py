import torch
import hashlib
import pickle
import typing as ty
from sparse.mult.tensor.exceptions import  CastingError
import warnings

from sparse_tensor_multiply import sparse_multiply as sparse_multiply_


UnsignedShortIntegerTensor = ty.Type[torch.Tensor]



def sparse_multiply_cpu_(oned_tensor : torch.Tensor, indices : UnsignedShortIntegerTensor,
                    values : torch.Tensor, output_shape : int):
    raise NotImplementedError("Current CPU implementation is too slow")

    result = torch.zeros((oned_tensor.size(0), output_shape))
    #print(indices.shape, values.shape, oned_tensor.shape, result.shape)
    indices_x = indices[:, 0]
    indices_y = indices[:, 1]
    result[:, indices_y] = values * oned_tensor[:, indices_x]
    return result

def sparse_multiply(oned_tensor : torch.Tensor, indices : UnsignedShortIntegerTensor,
                    values : torch.Tensor, output_shape : int, device : str):

    """

    :param indices: (-1, 2) tensor that contains X,Y where X corresponds to Neuron index from first layer and
    Y corresponds to Neuron index from the second layer
    :param values: values of the W matrix
    :param oned_tensor: activations from the first layer
    :param output_shape: for now it is 1D maybe we can create nD in the future size of activations of the second layer
    :param device:
    :return: result of multiplication of shape output_shape
    """
    if device=="cpu":
        return sparse_multiply_cpu_(oned_tensor, indices, values, output_shape)
    return sparse_multiply_(oned_tensor, values, indices, output_shape)

class SparseTensor(torch.nn.Module):

    def __init__(self, indices : UnsignedShortIntegerTensor,
                 values : torch.Tensor, matrix_shape : ty.Union[int, ty.Tuple[int]], device : str = None) -> None:
        if device and device.startswith('cuda') and not  indices.is_cuda:
            raise NotImplementedError(f'device has to be None or cpu not {device}. to move to cuda use .cuda afterwards')
        # indices = indices.int()[(indices.float()[:,0]*indices.size(0)+indices[:,1]).sort().indices]
        if indices.dtype != torch.uint16 and device =='cuda':
            warnings.warn("""Casting indices tensor to uint16 is a costly operation to be handled by SparseTensor it is 
            better to create this tensor with integer values upfront""")
            try:
                indices = indices.to(dtype=torch.uint16)
            except Exception as e:
                raise CastingError(inp_dtype=str(indices.dtype), e=e)



        if len(matrix_shape) > 3:
            raise NotImplementedError("""We support having only 2-D tensor output please raise a PR if you want to 
            contribute: https://github.com/JafarBadour/Parallel-Dynamic-Sparse-Training/pull/""")
        super(SparseTensor, self).__init__()

        self.indices = indices
        self.matrix_shape = matrix_shape

        self.values = values
        self.shape = self.shape_calc()
        self.device = device
    def matmul(self, other : torch.Tensor):
        return self @ other
    def __matmul__(self, other):
        """
            A = S @ B where S is the sparse tensor.
        :param other: B in the above case
        :return:
        """

        if len(other.shape) > 2:
            raise NotImplementedError("""Only 2-D tensor to be multiplied with the matrix""")

        return sparse_multiply(other, self.indices, self.values, self.matrix_shape[1], self.device)

    def dense(self) -> torch.Tensor:
        device = self.indices.device
        indices = self.indices# .cpu()
        values = self.values# .cpu()
        res = torch.zeros(self.matrix_shape).to(device)
        res[tuple(indices.to(dtype=torch.int).T)] = values
        return res.to(device)

    @classmethod
    def __new_obj__(cls, indices, values, output_shape, device):
        return cls(indices, values, output_shape, device)

    def to(self, device):

        indices = self.indices.to(device)
        values = self.values.to(device)
        self.device=device
        return SparseTensor.__new_obj__(indices, values, self.matrix_shape, device)

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
    def t(self):
        indices = self.indices.clone()

        indices[:, 0], indices[:, 1] = indices[:, 1].clone(), indices[:, 0].clone()
        return SparseTensor(indices, self.values.clone(), tuple(reversed(self.matrix_shape)))
    def hash(self):

        serialized_tensor_values = pickle.dumps(self.values.cpu())  # or torch.save to BytesIO for large tensors
        tensor_hash_values = hashlib.sha256(serialized_tensor_values).hexdigest()
        serialized_tensor_indices =  pickle.dumps(self.indices.cpu())
        tensor_hash_indices = hashlib.sha256(serialized_tensor_indices).hexdigest()
        return f"indices={tensor_hash_values},values={tensor_hash_indices}"










