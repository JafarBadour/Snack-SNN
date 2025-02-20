import torch
from sparse_tensor_multiply import sparse_outer_product_multiply


def tst1():
    left = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    right = left * 10
    indices_left = torch.tensor([0, 1, 0, 0], dtype=int)
    indices_right = torch.tensor([0, 1, 2, 2], dtype=int)

    r1 = left[:, indices_left] * right[:, indices_right]

    indices_left = indices_left.to(dtype=torch.uint16).cuda()
    indices_right = indices_right.to(dtype=torch.uint16).cuda()

    r2 = sparse_outer_product_multiply(left.cuda(), indices_left, right.cuda(), indices_right)

    print(r2)
    print(r1.mean(dim=0))


if __name__ == "__main__":
    tst1()
