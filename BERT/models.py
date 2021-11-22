import torch
import pytorch_lightning as pl
from typing import Tuple
from argparse import Namespace

from torch import nn
from transformers import BertModel
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.nn import functional as F

class MultiLabelNER(pl.LightningModule):
    """
    글자당 여러개의 개체명을 인식하는 모델.
    """
    def __init__(self, bert: BertModel, lr: float):
        super().__init__()
        self.bert = bert
        self.W_1 = nn.Linear(self.bert.config.hidden_size, 3)
        self.drop = nn.Dropout(p=0.3)
        self.W_2 = nn.Linear(self.bert.config.hidden_size, 13)
        self.lr = lr
        # self.save_hyperparameters(Namespace(lr=lr))

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch  # (N, L, 3), (N, L, 2)
        input_ids = inputs[:, 0]  # (N, 3, L) -> (N, L)
        token_type_ids = inputs[:, 1]  # (N, 3, L) -> (N, L)
        attention_mask = inputs[:, 2]  # (N, 3, L) -> (N, L)
        labels_1 = targets[:, 0]  # (N, 2, L) -> (N, L)
        labels_2 = targets[:, 1]  # (N, 2, L) -> (N, L)
        H_all = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)[0]   # (N, L, H)

        # H_all로 부터 각 레이블에 해당하는 로짓값을 구하기
        # a = H_all.size()    -> 1, 100, 768
        logits_1 = self.W_1(H_all)  # (N, L, H) -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수 / 3
        logits_2 = self.W_2(H_all)  # (N, L, H) -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수 / 13

        logits_1 = torch.einsum("nlc->ncl",logits_1)    # (N, L, T_1) -> (N, T_1, L)
        logits_2 = torch.einsum("nlc->ncl",logits_2)    # (N, L, T_2) -> (N, T_2, L)

        # 각 로짓값으로부터 로스를 계산하기.
        # logits_1 = F.softmax(logits_1, dim=2)
        # # logits_1 = torch.argmax(logits_1, dim=2)
        # logits_2 = F.softmax(logits_2, dim=2)
        # # logits_2 = torch.argmax(logits_2, dim=2)

        # test_1 = len(logits_1)
        # test_2 = labels_anm.view(-1)
        # batch 별 loss 값??
        # logits_1 -> (1, 100, 3)
        # (100, 3), (100,)은 cross entropy가 돌아가긴함 (그럼 배치 하나만 하는거 아닌가?)
        # for문을 써서 평균을 내야하나

        loss_1 = F.cross_entropy(logits_1, labels_1)    # (N, T_1, L), (N, L) -> (N, L)
        loss_2 = F.cross_entropy(logits_2, labels_2)    # (N, T_2, L), (N, L) -> (N, L)

        loss_1 = loss_1.sum()   # (N, L) -> 1
        loss_2 = loss_2.sum()   # (N, L) -> 1

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

#
# class NER(pl.LightningModule):
#     """
#     글자당 하나의 개체명만을 인식하는 모델
#     """
#     def __init__(self, bert: BertModel, lr: float):
#         super().__init__()
#         self.bert = bert
#         self.W_labels = torch.nn.Linear(..., ...)
#         self.save_hyperparameters(Namespace(lr=lr))
#
#     def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
#         X, Y = batch
#         input_ids = X[:, 0]
#         token_type_ids = X[:, 1]
#         attention_mask = X[:, 2]
#         labels_source = Y[:, 0]
#         ...
#         loss = ...
#         # multitask learning
#         return {
#             "loss": loss
#         }
#
#     # boilerplate - 필요는 없는데 구현은 해야해서 그냥 여기에 둠.
#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         pass
#
#     def test_dataloader(self) -> EVAL_DATALOADERS:
#         pass
#
#     def val_dataloader(self) -> EVAL_DATALOADERS:
#         pass
#
#     def predict_dataloader(self) -> EVAL_DATALOADERS:
#         pass

