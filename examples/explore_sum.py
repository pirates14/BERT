"""
torch.sum()은 특정 차원에 대해서 더하는 것인가? 아니면 다 더하는 것인가?
"""
import torch


def main():
    N = 10
    L = 30
    A = torch.rand(size=(N, L))
    print(A)
    print(A.sum())
    print(A.shape)
    print(A.sum().shape)
    print(torch.einsum("nl->n", A))
    B = torch.Tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(B.sum())

if __name__ == '__main__':
    main()