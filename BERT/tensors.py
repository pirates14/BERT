import torch
from transformers import BertTokenizer


class TensorBuilder:
    """
    Whatever a tensor builder does, it outputs a tensor
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class InputsBuilder(TensorBuilder):

    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self) -> torch.Tensor:
        # TODO: 입력행렬 (X)를 출력하는 빌더를 구현하기
        pass


class TargetsBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self) -> torch.Tensor:
        # TODO: 정답행렬 (y)를 출력하는 빌더를 구현하기
        pass
