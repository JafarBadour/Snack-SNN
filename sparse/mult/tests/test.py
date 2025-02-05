
from sparse.mult.tensor import SparseTensor, create_random_sparse_matrix
import torch


def tst1():

    def get_sparse_tensor(sp):

        import torch
        indices = torch.concat((sp.indices[:, 0].reshape(1, -1), sp.indices[:, 1].reshape(1, -1)), axis=0)

        return torch.sparse_coo_tensor(indices, sp.values, sp.matrix_shape, device='cuda')


    sp = create_random_sparse_matrix(10000, 10000, 88)
    sp = sp.cuda()

    ones = torch.ones(sp.matrix_shape[1]).cuda()

    r1 = sp @ ones

    st = get_sparse_tensor(sp)

    r2 = ones @ st

    print(torch.abs(r2 - r1).max())


def tst2():
    torch.set_printoptions(sci_mode=False)
    sp = SparseTensor(indices=torch.tensor([[0, 1], [1, 0], [1, 1]]), values=torch.tensor([0.1, 10, 100]),
                      matrix_shape=(3, 3)).cuda()
    activations = torch.tensor([[1, 1, 8], [10, 10,10]]).float().cuda()

    print((sp @ activations) )

    print(activations @ sp.dense())

    print(activations)
    print(sp.dense())

def tst3():
    torch.set_printoptions(sci_mode=False)
    sp = create_random_sparse_matrix(3,2, 50).cuda()
    activations = torch.rand((2, 3)).cuda()
    activations = torch.tensor([[1,2,3], [10, 20, 30]]).float().cuda()
    print(sp)
    r1 = sp @ activations
    r2 = activations @ sp.dense()
    print(r1)
    print(r2)
    print((r1 - r2).abs().max())



if __name__ == "__main__":
    tst3()