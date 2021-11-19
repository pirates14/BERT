import torch


def main():
    x = torch.Tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(x.view(-1, 2))   # -1 = inferred from other dimensions.
    print(x.view(-1, 3))   # -1 = inferred from other dimensions.
    print(x.view(-1, 4))   # no dimension can be inferred for this case


if __name__ == '__main__':
    main()
