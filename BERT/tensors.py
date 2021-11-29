import torch
from typing import List, Tuple
from transformers import BertTokenizer
from BERT.labels import ANM_LABELS, SOURCE_LABELS
from keras_preprocessing.sequence import pad_sequences


class TensorBuilder:
    """
    Whatever a tensor builder does, it outputs a tensor
    """
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class InputsBuilder(TensorBuilder):

    def __init__(self, tokenizer: BertTokenizer, sentences: List[List[Tuple[str, str, str]]], max_len: int):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_len = max_len

    def __call__(self) -> torch.Tensor:
        sent2tokens = [
            [word for word, _, _ in sentence]
            for sentence in self.sentences
        ]
        encoded = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in sent2tokens]
        input_ids = torch.LongTensor(pad_sequences(encoded, maxlen=self.max_len, dtype=int,
                                                   # 패딩 토큰을 토크나이저에서 가져오기
                                                   padding="post", value=self.tokenizer.pad_token_id))
        token_type_ids = torch.zeros(size=(len(sent2tokens), self.max_len))  # 어차피 첫문장만 있음
        attention_mask = torch.where(input_ids != self.tokenizer.pad_token_id, 1, 0)  # 패딩인 것은 필요 없음
        inputs = torch.stack([input_ids, token_type_ids, attention_mask], dim=1)

        return inputs.long()


class TargetsBuilder(TensorBuilder):
    def __init__(self, tokenizer, sentences: List[List[Tuple[str, str, str]]], max_len: int):
        """
        Tuple[str=단어, int=ANM 레이블, int=NER 레이블]
        :param sentences:
        """
        self.sentences = sentences
        self.max_len = max_len
        self.tokenizer = tokenizer
        pass

    def __call__(self) -> torch.LongTensor:
        """
        # N = 데이터셋의 크기
        # L = 문장의 길이 -  (패딩된 길이) (단어의 개수)
        # 2 + 1 = ANM 개체명 클래스 개수
        # 12 + 1 = NER 개체명 클래스 개수
        # 2 -> label 개수 (anm + ner)
        :return: (N, L, 2)
        """

        # type : list of list of int
        anm_targets = [
            [ANM_LABELS.index(anm_label) for word, anm_label, source_label in sentence]
            for sentence in self.sentences
        ]

        source_targets = [
            [SOURCE_LABELS.index(source_label) for word, anm_label, source_label in sentence]
            for sentence in self.sentences
        ]

        anm_targets_p = pad_sequences(anm_targets, maxlen=self.max_len,
                                      padding='post', value=ANM_LABELS.index("O"))
        source_targets_p = pad_sequences(source_targets, maxlen=self.max_len,
                                         padding='post', value=SOURCE_LABELS.index("O"))

        # (N, 2, L)
        targets = torch.stack([torch.Tensor(anm_targets_p),
                               torch.Tensor(source_targets_p)], dim=1)

        return targets.long()
