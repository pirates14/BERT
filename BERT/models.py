import torch
import pytorch_lightning as pl
from typing import Tuple
from argparse import Namespace
from transformers import BertForTokenClassification
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class MultiLabelNER(pl.LightningModule):
    """
    글자당 여러개의 개체명을 인식하는 모델.
    """
    def __init__(self, bert_tc: BertForTokenClassification, lr: float):
        super().__init__()
        self.bert_tc = bert_tc
        self.save_hyperparameters(Namespace(lr=lr))

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        labels_1 = targets['labels_1']
        labels_2 = targets['labels_2']
        H_all = ...   # ... -> (N, L, H)
        logits = ...  # ... -> (N, L, T)
        logits_1 = ...  # ... -> (N, L,
        loss_1 = ...
        loss_2 = ...
        # multitask learning
        loss = loss_1 + loss_2
        return {
            "loss": loss
        }

    def configure_optimizers(self):
        # 옵티마이저 설정은 여기에서
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # boilerplate - 필요는 없는데 구현은 해야해서 그냥 여기에 둠.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class NER(pl.LightningModule):
    """
    글자당 하나의 개체명만을 인식하는 모델
    """
    def __init__(self, bert_tc: BertForTokenClassification, lr: float):
        super().__init__()
        self.bert_tc = bert_tc
        self.save_hyperparameters(Namespace(lr=lr))

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        X, Y = batch
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        labels_source = Y[:, 0]
        ...
        loss = ...
        # multitask learning
        return {
            "loss": loss
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
