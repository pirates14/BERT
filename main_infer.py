"""
main_infer.py는 모델의 예측값을 정성적으로 살펴보기 위한 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_infer.py
"""
from typing import List, Tuple

import torch
from transformers import BertTokenizer, BertModel, AutoConfig, AutoModel

from BERT.loaders import load_config
from BERT.models import MultiLabelNER
from BERT.paths import SOURCE_ANM_NER_CKPT
from BERT.tensors import InputsBuilder
from BERT.classes import NER_CLASSES, ANM_CLASSES


def main():

    config = load_config()
    tokenizer = BertTokenizer.from_pretrained(config['bert'])

    text = '''
    삼성전자 관계자는 “코로나19 방역 지침에 따라 행사를 조용하게 치렀다”고 밝혔다
    '''
    bert = AutoModel.from_config(AutoConfig.from_pretrained(config['bert']))
    model = MultiLabelNER.load_from_checkpoint(SOURCE_ANM_NER_CKPT,
                                               bert=bert)
    tokens: List[str] = tokenizer.tokenize(text)
    sentences: List[List[Tuple[str, str, str]]] = [
        [
            (token, "", "")
            for token in tokens
        ]
    ]
    inputs_builder = InputsBuilder(tokenizer, sentences, config['max_length'])

    inputs = inputs_builder()  # (N, 3, L)

    # 가중치가 변하지 않음
    # 1. pl 문서를 보면 이렇게 하라. (best practice)
    # 2. 더 보기좋잖아요
    model.eval()
    model.freeze()

    # 원하는 결과?
    predictions: List[List[Tuple[str, int, int]]] = model.predict(inputs)

    print(NER_CLASSES[0])  # O

    # with torch.no_grad():
    #     # 어떤 일이 일어나던 절대로 가중치 업데이트는 하지 않는다


if __name__ == '__main__':
    main()


