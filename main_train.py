"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from BERT.loaders import load_petite
from BERT.models import MultiLabelNER
from BERT.tensors import InputsBuilder, TargetsBuilder
from BERT.paths import SOURCE_ANM_NER_CKPT


def main():
    petite = load_petite()
    tokenizer = ...
    inputs = InputsBuilder(tokenizer, petite)()
    targets = TargetsBuilder(petite)()
    dataset = Dataset(inputs, targets)
    dataloader = DataLoader(dataset)

    multi_label_ner = MultiLabelNER(...)

    # 파라미터를 보고 싶다: ctrl + p
    # 문서를 보고싶다: fn + 1
    trainer = pl.Trainer()
    # 학습을 진행한다
    trainer.fit(model=multi_label_ner, train_dataloader=dataloader)
    # 모델학습이 진행이된다.
    trainer.save_checkpoint(filepath=SOURCE_ANM_NER_CKPT)


if __name__ == '__main__':
    main()
