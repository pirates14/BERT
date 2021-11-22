"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

from BERT.dataset import NERDataset, SentenceGetter
from BERT.loaders import load_petite, load_config
from BERT.models import MultiLabelNER
from BERT.tensors import InputsBuilder, TargetsBuilder
from BERT.paths import SOURCE_ANM_NER_CKPT


def main():
    config = load_config()
    sentences = SentenceGetter(load_petite()).sentences
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = BertModel.from_pretrained(config['bert'])
    # TODO: config 사용하기
    inputs = InputsBuilder(tokenizer, sentences=sentences, max_len=100)
    targets = TargetsBuilder(tokenizer, sentences=sentences, max_len=100)
    dataset = NERDataset(inputs=inputs(), targets=targets())
    dataloader = DataLoader(dataset)

    multi_label_ner = MultiLabelNER(bert=bert, lr=5e-6)

    # 파라미터를 보고 싶다: ctrl + p
    # 문서를 보고싶다: fn + 1
    trainer = pl.Trainer(max_epochs=10,
                         enable_checkpointing=False)
    # 학습을 진행한다
    trainer.fit(model=multi_label_ner, train_dataloader=dataloader)
    # 모델학습이 진행이된다.
    trainer.save_checkpoint(filepath=SOURCE_ANM_NER_CKPT)


if __name__ == '__main__':
    main()
