"""
https://github.com/wisdomify/wisdomify/blob/main/main_tune.py
"""
import argparse
import random
import numpy as np
import pytorch_lightning as pl
import torch.cuda
from transformers import BertTokenizer, BertModel
import os
import wandb
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

    with wandb.init(project="BERT", config=config) as run:
        torch.manual_seed(config['seed'])

        tokenizer = BertTokenizer.from_pretrained(config['bert'])
        bert = BertModel.from_pretrained(config['bert'])
        datamodule = AnmSourceNERDataModule(config, tokenizer)
        model = BiLabelNER(bert=bert, lr=float(config['lr']))
        # Run learning rate finder
        trainer = pl.Trainer(auto_lr_find=True,
                             gpus=torch.cuda.device_count(),
                             enable_checkpointing=False,
                             logger=False)
        try:
            results = trainer.tune(model, datamodule=datamodule)
        except Exception as e:
            # whatever exception occurs, make sure to delete the cash
            os.system("rm lr_find*")  # just remove the file
            raise e
        else:
            lr_finder = results['lr_find']
            run.log({"results": lr_finder.results, "suggested_lr": lr_finder.suggestion()})


if __name__ == '__main__':
    main()
