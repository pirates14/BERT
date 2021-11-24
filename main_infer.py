"""
main_infer.py는 모델의 예측값을 정성적으로 살펴보기 위한 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_infer.py
"""
from typing import List, Tuple

from transformers import BertTokenizer, AutoConfig, AutoModel

from BERT.loaders import load_config
from BERT.models import MultiLabelNER
from BERT.paths import SOURCE_ANM_NER_CKPT
from BERT.tensors import InputsBuilder
from BERT.classes import SOURCE_CLASSES, ANM_CLASSES


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

    # 원하는 결과
    anm_labels, source_labels = model.predict(inputs)  # (N, 3, L) -> (1, L), (1, L)
    anm_labels = anm_labels[0]  # (1, L) -> (L,)
    source_labels = source_labels[0]  # (1, L) -> (L,)

    for word, anm_label, source_label in zip(tokens, anm_labels, source_labels):
        #  관계자는, B-ANM, B-PERSON
        print(word, ANM_CLASSES[anm_label], SOURCE_CLASSES[source_label])

    # for i in predictions:
    #     if i[0] != 0:
    #         word = tokenizer.decode(i[0])
    #         anm = ANM_CLASSES[i[1]]
    #         ner = NER_CLASSES[i[2]]
    #         print('단어: {}, anm: {}, ner: {}'.format(word, anm, ner))

    # with torch.no_grad():
    # 어떤 일이 일어나도 절대로 가중치 업데이트는 하지 않는다


if __name__ == '__main__':
    main()


