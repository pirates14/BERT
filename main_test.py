"""
main_eval.py 스크립트는 모델을 평가하기 위한 스크립트입니다.
e.g.: https://github.com/wisdomify/wisdomify/blob/main/main_eval.py
지표를 계산 (acc, f1_score)
"""
from os import path

import torch
import wandb
import argparse
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer, AutoModel, AutoConfig
from BERT.datamodules import AnmSourceNERDataModule, SourceNERDataModule, AnmNERDataModule
from BERT.loaders import load_config
from BERT.models import BiLabelNER, BiLabelNERWithBiLSTM, MonoLabelNER


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mono_label_ner")
    parser.add_argument("--ver", type=str, default="test_anm")
    args = parser.parse_args()
    config = load_config(args.model, args.ver)
    config.update(vars(args))  # command-line arguments 도 기록하기!
    # --- fix random seeds -- #
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = AutoModel.from_config(AutoConfig.from_pretrained(config['bert']))
    datamodule = AnmSourceNERDataModule(config, tokenizer)
    with wandb.init(project="BERT", config=config) as run:
        # download a pre-trained model from wandb
        logger = WandbLogger(log_model=False)
        artifact = run.use_artifact(f"{BiLabelNER.name}:{config['ver']}")
        model_path = artifact.checkout()
        model = BiLabelNER.load_from_checkpoint(path.join(model_path, "ner.ckpt"), bert=bert)
        # test_step 실행.
        trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                             enable_checkpointing=False,
                             logger=logger)
        trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
