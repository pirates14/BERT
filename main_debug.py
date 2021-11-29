"""
main_train.py은 모델 학습을 진행하는 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_train.py
"""
import torch
import random
import argparse
import numpy as np
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
from BERT.datamodules import AnmSourceNERDataModule, SourceNERDataModule, AnmNERDataModule
from BERT.labels import ANM_LABELS, SOURCE_LABELS
from BERT.loaders import load_config
from BERT.models import BiLabelNER, MonoLabelNER, BiLabelNERWithBiLSTM


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

    trainer = pl.Trainer(fast_dev_run=True,  # 에폭을 한번만 돈다. 모델 저장도 안함. 디버깅으로 제격
                         gpus=torch.cuda.device_count())
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
