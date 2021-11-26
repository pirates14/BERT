"""
main_eval.py 스크립트는 모델을 평가하기 위한 스크립트입니다.
e.g.: https://github.com/wisdomify/wisdomify/blob/main/main_eval.py
지표를 계산 (acc, f1_score)
"""
import torch
import wandb
import argparse
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, BertModel
from BERT.datamodules import AnmSourceNERDataModule
from BERT.loaders import load_config
from BERT.models import BiLabelNER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ver", type=str)
    args = parser.parse_args()
    config = load_config(args.ver)
    config.update(vars(args))  # command-line arguments 도 기록하기!
    # --- fix random seeds -- #
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = BertModel.from_pretrained(config['bert'])
    datamodule = AnmSourceNERDataModule(config, tokenizer)
    multi_label_ner = BiLabelNER(bert=bert, lr=float(config['lr']))
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
