import torch
import pytorch_lightning as pl
from typing import Tuple
from argparse import Namespace
from transformers import BertModel
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class MultiLabelNER(pl.LightningModule):
    """
    글자당 여러개의 개체명을 인식하는 모델.
    """
    def __init__(self, bert: BertModel, lr: float):
        super().__init__()
        self.bert = bert
        self.W_1 = torch.nn.Linear(..., ...)
        self.W_2 = torch.nn.Linear(..., ...)
        self.save_hyperparameters(Namespace(lr=lr))

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        labels_1 = targets['labels_1']
        labels_2 = targets['labels_2']
        H_all = self.bert(...)   # ... -> (N, L, H)
        # H_all로 부터 각 레이블에 해당하는 로짓값을 구하기
        logits_1 = self.W_1(H_all)  # ... -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수
        logits_2 = self.W_2(H_all)  # ... -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수
        # 각 로짓값으로부터 로스를 계산하기.
        loss_1 = ...  # cross entropy with  labels_1
        loss_2 = ...  # cross entropy with labels_2
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
    def __init__(self, bert: BertModel, lr: float):
        super().__init__()
        self.bert = bert
        self.W_labels = torch.nn.Linear(..., ...)
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
