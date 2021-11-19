"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""
import pytorch_lightning as pl
from BERT.models import MultiLabelNER


def main():

    multi_label_ner = MultiLabelNER(...)
    datamodule = ...

    trainer = pl.Trainer(...)
    trainer.fit(model=multi_label_ner, datamodule=datamodule)


if __name__ == '__main__':
    main()
