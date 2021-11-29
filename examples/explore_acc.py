import torch
import torchmetrics


def main():
    preds = torch.Tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    targets = torch.Tensor([0, 1]).long()
    metric = torchmetrics.Accuracy()
    acc = metric(preds, targets)
    print(acc)
    print(metric.correct)  # this should be a bug


if __name__ == '__main__':
    main()