"""
main_eval.py 스크립트는 모델을 평가하기 위한 스크립트입니다.
e.g.: https://github.com/wisdomify/wisdomify/blob/main/main_eval.py
지표를 계산 (acc, f1_score)
"""
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertModel

import wandb
from BERT.dataset import NERDataModule
from BERT.loaders import load_config
from BERT.models import MultiLabelNER


def main():
    config = load_config()
    torch.manual_seed(config['seed'])

    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = BertModel.from_pretrained(config['bert'])
    datamodule = NERDataModule(config, tokenizer)

    multi_label_ner = MultiLabelNER(bert=bert, lr=float(config['lr']))
    logger = WandbLogger(log_model=False)
    with wandb.init(project="BERT") as run:
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             gpus=torch.cuda.device_count(),  # cpu 밖에 없으면 0, gpu가 n개이면 n
                             # callbacks=[early_stopping_callback],
                             enable_checkpointing=False,
                             logger=logger)
        trainer.test(model=multi_label_ner, datamodule=datamodule)
        # test_step 실행.

if __name__ == '__main__':
    main()
