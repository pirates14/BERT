"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""

import pytorch_lightning as pl
import torch
from transformers import BertTokenizer, BertModel
from BERT.dataset import NERDataModule
from BERT.loaders import load_config
from BERT.models import MultiLabelNER


def main():
    # 1. wandb login
    # 2. Wandb Logger 만들기
    # https://wandb.ai/pirates14/BERT

    config = load_config()
    torch.manual_seed(config['seed'])

    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = BertModel.from_pretrained(config['bert'])
    datamodule = NERDataModule(config, tokenizer)

    multi_label_ner = MultiLabelNER(bert=bert, lr=float(config['lr']))

    # 파라미터를 보고 싶다: ctrl + p
    # 문서를 보고싶다: fn + 1

    # early_stopping_callback = EarlyStopping(monitor="val_loss",
    #                                         mode="min", patience=2)

    trainer = pl.Trainer(fast_dev_run=True,
                         gpus=torch.cuda.device_count())
    # 학습을 진행한다
    trainer.fit(model=multi_label_ner, datamodule=datamodule)


if __name__ == '__main__':
    main()
