"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""

import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from pytorch_lightning.callbacks import EarlyStopping
from BERT.dataset import NERDataset, SentenceGetter, NERDataModule
from BERT.loaders import load_petite, load_config
from BERT.models import MultiLabelNER
from BERT.tensors import InputsBuilder, TargetsBuilder
from BERT.paths import SOURCE_ANM_NER_CKPT
from pytorch_lightning.loggers import WandbLogger

def main():
    # TODO: 시드 고정
    # 1. wandb login
    # 2. Wandb Logger 만들기
    # https://wandb.ai/pirates14/BERT

    config = load_config()
    sentences = SentenceGetter(load_petite()).sentences
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = BertModel.from_pretrained(config['bert'])

    # inputs = InputsBuilder(tokenizer, sentences=sentences, max_len=config['max_length'])
    # targets = TargetsBuilder(tokenizer, sentences=sentences, max_len=config['max_length'])
    # dataset = NERDataset(inputs=inputs(), targets=targets())
    # dataloader = DataLoader(dataset)
    datamodule = NERDataModule()

    multi_label_ner = MultiLabelNER(bert=bert, lr=float(config['lr']))

    # 파라미터를 보고 싶다: ctrl + p
    # 문서를 보고싶다: fn + 1

    # early_stopping_callback = EarlyStopping(monitor="val_loss",
    #                                         mode="min", patience=2)
    early_stopping_callback = ...

    logger = WandbLogger(log_model=False)
    with wandb.init(project="BERT") as run:
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             callbacks=[early_stopping_callback],
                             enable_checkpointing=False,
                             logger=logger)
        # 학습을 진행한다
        trainer.fit(model=multi_label_ner, datamodule=datamodule)

    # 모델학습이 진행이된다.
    trainer.save_checkpoint(filepath=SOURCE_ANM_NER_CKPT)

    # main_eval.py
    # trainer.test()

if __name__ == '__main__':
    main()
