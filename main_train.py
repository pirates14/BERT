"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""

import pytorch_lightning as pl
import torch

import wandb
from transformers import BertTokenizer, BertModel
from pytorch_lightning.callbacks import EarlyStopping
from BERT.dataset import NERDataModule
from BERT.loaders import load_config
from BERT.models import MultiLabelNER
from BERT.paths import SOURCE_ANM_NER_CKPT
from pytorch_lightning.loggers import WandbLogger

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

    logger = WandbLogger(log_model=False)
    with wandb.init(project="BERT") as run:
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             gpus=torch.cuda.device_count(),  # cpu 밖에 없으면 0, gpu가 n개이면 n
                             # callbacks=[early_stopping_callback],
                             auto_lr_find=True,
                             enable_checkpointing=False,
                             logger=logger)
        # 학습을 진행한다
        trainer.fit(model=multi_label_ner, datamodule=datamodule)

    # 모델학습이 진행이된다.
    trainer.save_checkpoint(filepath=SOURCE_ANM_NER_CKPT)

    # main_eval.py
    # trainer.test()
    # TODO:  오버피팅이 언제 일어나는지 파악을해서, early stopping 을 해볼 것!
    # Option: COllab -> Ainize

if __name__ == '__main__':
    main()
