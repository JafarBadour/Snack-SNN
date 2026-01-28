import torch
import hashlib
import pickle
import typing as ty
from sparse.mult.tensor.exceptions import CastingError
import warnings

from sparse_tensor_multiply import sparse_multiply as sparse_multiply_


UnsignedShortIntegerTensor = ty.Type[torch.Tensor]


def sparse_multiply_cpu_(
    oned_tensor: torch.Tensor,
    indices_a: UnsignedShortIntegerTensor,
    indices_b: UnsignedShortIntegerTensor,
    values: torch.Tensor,
    output_shape: int,
):
    raise NotImplementedError("Current CPU implementation is too slow")

    result = torch.zeros((oned_tensor.size(0), output_shape))
    # print(indices.shape, values.shape, oned_tensor.shape, result.shape)

    result[:, indices_b] = values * oned_tensor[:, indices_a]
    return result


def sparse_multiply(
    oned_tensor: torch.Tensor,
    indices_a: UnsignedShortIntegerTensor,
    indices_b,
    values: torch.Tensor,
    output_shape: int,
    device: str,
):
    """

    :param indices: (-1, 2) tensor that contains X,Y where X corresponds to Neuron index from first layer and
    Y corresponds to Neuron index from the second layer
    :param values: values of the W matrix
    :param oned_tensor: activations from the first layer
    :param output_shape: for now it is 1D maybe we can create nD in the future size of activations of the second layer
    :param device:
    :return: result of multiplication of shape output_shape
    """
    if device == "cpu":
        return sparse_multiply_cpu_(oned_tensor, indices_a, indices_b, values, output_shape)

    # print(oned_tensor.shape)
    # print(values.shape)
    # print(indices_a.shape)
    # print(indices_b.shape)
    return sparse_multiply_(oned_tensor, values, indices_a, indices_b, output_shape)


class SparseTensor(torch.nn.Module):

    def __init__(
        self,
        *args,
        indices_a: torch.Tensor = None,
        indices_b: torch.Tensor = None,
        indices: UnsignedShortIntegerTensor = None,
        values: torch.Tensor = None,
        matrix_shape: ty.Union[int, ty.Tuple[int]] = None,
        device: str = None,
        backend: str = "COO",
    ) -> None:
        if len(args) != 0:
            raise ValueError("SparseTensor accepts only keywords arguments")
        if device and device.startswith("cuda") and not (indices_a.is_cuda and indices_b.is_cuda and values.is_cuda):
            raise NotImplementedError(
                f"device has to be None or cpu not {device}. to move to cuda use .cuda afterwards"
            )
        # indices = indices.int()[(indices.float()[:,0]*indices.size(0)+indices[:,1]).sort().indices]

        if indices is not None:
            warnings.warn(
                """Casting indices tensor to uint16 is a costly operation to be handled by SparseTensor it is 
            better to create this tensor with integer values upfront"""
            )
            try:
                indices_a = indices[:, 0].to(dtype=torch.uint16)
                indices_b = indices[:, 1].to(dtype=torch.uint16)
            except Exception as e:
                raise CastingError(inp_dtype=str(indices.dtype), e=e)

        if not (indices_a.dtype == indices_b.dtype == torch.uint16):
            try:
                indices_a = indices_a.to(dtype=torch.uint16)
                indices_b = indices_b.to(dtype=torch.uint16)
            except Exception as e:
                raise CastingError(inp_dtype=str(indices.dtype), e=e)
        if len(matrix_shape) > 3:
            raise NotImplementedError(
                """We support having only 2-D tensor output please raise a PR if you want to 
            contribute: #URL DELETED FOR BLIND REVIEW"""
            )
        super(SparseTensor, self).__init__()
        if indices_a is None:
            indices_a = indices[:, 0]
            indices_b = indices[:, 1]
        self.indices_a = indices_a
        self.indices_b = indices_b
        self.matrix_shape = matrix_shape

        self.values = values
        self.shape = self.shape_calc()
        self.device = device
        if backend == "COO":
            self.spmm = sparse_multiply
        if backend == "CSR":
            self.spmm = None

    def matmul(self, other: torch.Tensor):
        return self @ other

    def __matmul__(self, other):
        """
            A = S @ B where S is the sparse tensor.
        :param other: B in the above case
        :return:
        """

        # if len(other.shape) > 2:
        #     raise NotImplementedError(f"""Only 2-D tensor to be multiplied with the matrix: given {other.shape}""")
        if len(other.shape) > 2:
            # Note: 3D tensor multiplication is currently slower; optimization recommended for production use
            b, h, _ = other.shape
            other = other.reshape(-1, other.shape[-1])
            res = self.spmm(
                other,
                self.indices_a,
                self.indices_b,
                self.values,
                self.matrix_shape[1],
                self.device,
            )
            return res.reshape(b, h, -1)

        return self.spmm(
            other,
            self.indices_a,
            self.indices_b,
            self.values,
            self.matrix_shape[1],
            self.device,
        )

    def dense(self) -> torch.Tensor:
        if not (self.indices_a.device == self.indices_b.device == self.values.device):
            raise ValueError("Indices and values should be on the same device")

        device = self.indices_a.device
        indices_a = self.indices_a.int()
        indices_b = self.indices_b.int()
        values = self.values  # .cpu()
        res = torch.zeros(self.matrix_shape).to(device)

        res[(indices_a, indices_b)] = values
        return res.to(device)

    @classmethod
    def __new_obj__(cls, indices_a, indices_b, values, matrix_shape, device):
        return cls(
            indices_a=indices_a,
            indices_b=indices_b,
            values=values,
            matrix_shape=matrix_shape,
            device=device,
        )

    def to(self, device):

        indices_a = self.indices_a.to(device)
        indices_b = self.indices_b.to(device)
        values = self.values.to(device)

        return SparseTensor.__new_obj__(
            indices_a=indices_a,
            indices_b=indices_b,
            values=values,
            matrix_shape=self.matrix_shape,
            device=device,
        )

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def __str__(self):
        indices = torch.concat((self.indices_a.reshape(-1, 1), self.indices_b.reshape(-1, 1)), dim=1)
        return f"""SparseTensor(indices={indices}, \nvalues={self.values}, \n, matrix_shape={self.matrix_shape})"""

    def shape_calc(self):
        return f"""SparseTensor(\n
                    indices_a={self.indices_a.shape},\n
                    indices_b={self.indices_b.shape}, 
                    \nvalues={self.values.shape}, 
                    \n, matrix_shape={self.matrix_shape})
        """

    def __repr__(self):
        return self.__str__()

    def t(self):

        return SparseTensor(
            indices_a=self.indices_b.clone(),
            indices_b=self.indices_a.clone(),
            values=self.values.clone(),
            matrix_shape=tuple(reversed(self.matrix_shape)),
        )

    def hash(self):

        serialized_tensor_values = pickle.dumps(self.values.cpu())  # or torch.save to BytesIO for large tensors
        tensor_hash_values = hashlib.sha256(serialized_tensor_values).hexdigest()
        serialized_tensor_indices = pickle.dumps(self.indices_a.cpu())
        tensor_hash_indices = hashlib.sha256(serialized_tensor_indices).hexdigest()
        return f"indices={tensor_hash_values},values={tensor_hash_indices}"
