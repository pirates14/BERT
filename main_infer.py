"""
main_infer.py는 모델의 예측값을 정성적으로 살펴보기 위한 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_infer.py
"""
import torch
from transformers import BertTokenizer, BertModel, AutoConfig, AutoModel

from BERT.loaders import load_config
from BERT.models import MultiLabelNER
from BERT.paths import SOURCE_ANM_NER_CKPT


def main():

    config = load_config()
    tokenizer = BertTokenizer.from_pretrained(config['bert'])

    text = '''
    삼성전자 관계자는 “코로나19 방역 지침에 따라 행사를 조용하게 치렀다”고 밝혔다
    '''

    model = MultiLabelNER.load_from_checkpoint(SOURCE_ANM_NER_CKPT,
                                               bert= AutoModel.from_config(AutoConfig.from_pretrained(config['bert'])),
                                               lr=config['lr'])

    tokenized_sentence = tokenizer.encode(text=text)
    input_ids = torch.tensor([tokenized_sentence])

    with torch.no_grad():
        out = model(input_ids)




if __name__ == '__main__':
    main()


