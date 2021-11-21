from typing import Tuple, List

import torch
from transformers import BertTokenizer


class TensorBuilder:
    """
    Whatever a tensor builder does, it outputs a tensor
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class InputsBuilder(TensorBuilder):

    def __init__(self, tokenizer: BertTokenizer, petite: List[List[Tuple[str, str, str]]]):
        self.tokenizer = tokenizer
        self.petite = petite

    def __call__(self) -> torch.Tensor:
        # TODO: 입력행렬 (X)를 출력하는 빌더를 구현하기
        # input_ids, attention_mask
        sent2tokens = [
            [word for word, anm_label, ner_label in sent]
            for sent in self.petite
        ]
        encoded = self.tokenizer(sent2tokens,
                                 is_split_into_words=True,
                                 add_special_tokens=False,
                                 padding=True,
                                 truncation=True)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']
        attention_mask = encoded['attention_mask']
        inputs = ...  # (N, 3, L)
        # torch.stack([tensor_1, tensor_2, tensor_3])
        return inputs


class TargetsBuilder(TensorBuilder):
    def __init__(self, petite: List[List[Tuple[str, str, str]]]):
        """
        Tuple[str=단어, int=ANM 레이블, int=NER 레이블]
        :param petite:
        """
        self.petite = petite
        pass

    def __call__(self) -> torch.LongTensor:
        """
        # N = 데이터셋의 크기
        # L = 문장의 길이 -  (패딩된 길이) (단어의 개수)
        # 2 + 1 = ANM 개체명 클래스 개수
        # 12 + 1 = NER 개체명 클래스 개수
        :return: (N, L, 2)
        """
        # cls  ----  sep
        targets = ...  # (N, 2, L)
        anm_targets = targets[:, :, 0]
        ner_targets = targets[:, :, 1]
        return targets
