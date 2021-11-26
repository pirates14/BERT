"""
https://github.com/wisdomify/wisdomify/blob/main/main_tune.py
"""
import pytorch_lightning as pl
import torch.cuda
from transformers import BertTokenizer, BertModel
import os
import wandb
from BERT.dataset import NERDataModule
from BERT.loaders import load_config
from BERT.models import MultiLabelNER


def main():
    with wandb.init(project="BERT") as run:
        config = load_config()
        torch.manual_seed(config['seed'])

        tokenizer = BertTokenizer.from_pretrained(config['bert'])
        bert = BertModel.from_pretrained(config['bert'])
        datamodule = NERDataModule(config, tokenizer)
        multi_label_ner = MultiLabelNER(bert=bert, lr=float(config['lr']))
        # Run learning rate finder
        trainer = pl.Trainer(auto_lr_find=True,
                             gpus=torch.cuda.device_count(),
                             enable_checkpointing=False,
                             logger=False)
        try:
            results = trainer.tune(multi_label_ner, datamodule=datamodule)
        except Exception as e:
            # whatever exception occurs, make sure to delete the cash
            os.system("rm lr_find*")  # just remove the file
            raise e
        else:
            lr_finder = results['lr_find']
            run.log({"results": lr_finder.results, "suggested_lr": lr_finder.suggestion()})


if __name__ == '__main__':
    main()
