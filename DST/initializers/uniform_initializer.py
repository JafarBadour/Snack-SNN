import torch


def init(input_shape : int, output_shape : int, device='cpu', sparsity=0.9) -> torch.Tensor:
    """

    :param input_shape:
    :param output_shape:
    :param sparsity: here is how empty the tensor is
    :return:
    """
    if sparsity < 0.2:
        raise ValueError("Sparsity is less 0.2 which is inefficient for the uniform initializer")

    nz = int((1 - sparsity) * input_shape * output_shape)
    built_edges = torch.empty((0,2))
    return random_init_without_replacement(input_shape, output_shape, nz, built_edges).to(device)



def init_without_replacement(input_shape : int, output_shape : int, nz : int, excluded_edges : torch.Tensor, device='cpu'):
    ee = excluded_edges.to(dtype=torch.int32)
    return random_init_without_replacement(input_shape, output_shape, nz, ee)


if __name__ == "__main__":
    # r = init(5000, 5000, 'cuda', 0.2)
    # #, device='cuda')
    # print(r)
    from sparse_tensor_multiply import random_init_without_replacement

    ee = torch.tensor([[0, 1]], dtype=torch.int32)
    r = random_init_without_replacement(2, 2, 3, ee)
    print(r.shape)
    print(r.unique(dim=1).shape)
    print(r)