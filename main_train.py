"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""
import random
import torch
import wandb
import argparse
import numpy as np
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
from BERT.datamodules import AnmSourceNERDataModule, AnmNERDataModule, SourceNERDataModule
from BERT.loaders import load_config
from BERT.models import BiLabelNER, MonoLabelNER, BiLabelNERWithBiLSTM
from BERT.labels import ANM_LABELS, SOURCE_LABELS
from BERT.paths import bi_label_ner_ckpt
from pytorch_lightning.loggers import WandbLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mono_label_ner")
    parser.add_argument("--ver", type=str, default="test_anm")
    args = parser.parse_args()
    config = load_config(args.model, args.ver)
    # command-line arguments 도 기록하기!
    config.update(vars(args))
    # --- fix random seeds -- #
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    # --- prepare the model and the datamodule --- #
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = BertModel.from_pretrained(config['bert'])

    if config['model'] == BiLabelNER.name:
        model = BiLabelNER(bert=bert, lr=float(config['lr']), num_labels_pair=(len(ANM_LABELS), len(SOURCE_LABELS)))
        datamodule = AnmSourceNERDataModule(config, tokenizer)
    elif config['model'] == BiLabelNERWithBiLSTM.name:
        model = BiLabelNERWithBiLSTM(bert=bert, lr=float(config['lr']), num_labels_pair=(len(ANM_LABELS),
                                                                                       len(SOURCE_LABELS)))
        datamodule = AnmSourceNERDataModule(config, tokenizer)
    elif config['model'] == MonoLabelNER.name:
        if config['label_type'] == "anm":
            model = MonoLabelNER(bert=bert, lr=float(config['lr']), num_labels=len(ANM_LABELS),
                                 hidden_size=bert.config.hidden_size)
            datamodule = AnmNERDataModule(config, tokenizer)
        elif config['label_type'] == "source":
            model = MonoLabelNER(bert=bert, lr=float(config['lr']), num_labels=len(SOURCE_LABELS),
                                 hidden_size=bert.config.hidden_size)
            datamodule = SourceNERDataModule(config, tokenizer)
        else:
            raise ValueError(f"Invalid label_type: {config['label_type']}")
    else:
        raise ValueError(f"Invalid model: {config['model']}")

    # --- instantiate the trainer  --- #
    with wandb.init(project="BERT", config=config) as run:
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             log_every_n_steps=config['log_every_n_steps'],
                             gpus=torch.cuda.device_count(),
                             enable_checkpointing=False,
                             logger=logger)
        try:
            trainer.fit(model=model, datamodule=datamodule)
        except Exception as e:
            raise e
        else:
            # --- save the model locally, as push it to wandb as an artifact --- #
            # 오류 없이 학습이 완료되었을 때만 모델을 저장하기!
            model_path = bi_label_ner_ckpt()
            trainer.save_checkpoint(model_path)
            artifact = wandb.Artifact(name=model.name, type="model", metadata=config)
            artifact.add_file(model_path, "ner.ckpt")
            run.log_artifact(artifact, aliases=["latest", config['ver']])


if __name__ == '__main__':
    main()
