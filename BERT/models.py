import torch
import pytorch_lightning as pl
from typing import Tuple, List
from argparse import Namespace

import torchmetrics
from torch import nn
from transformers import BertModel, BertTokenizer
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
        self.W_2 = nn.Linear(self.bert.config.hidden_size, 15)
        self.train_acc_1 = torchmetrics.Accuracy()
        self.train_acc_2 = torchmetrics.Accuracy()
        self.val_acc_1 = torchmetrics.Accuracy()
        self.val_acc_2 = torchmetrics.Accuracy()
        self.test_acc_1 = torchmetrics.Accuracy()
        self.test_acc_2 = torchmetrics.Accuracy()
        self.save_hyperparameters(Namespace(lr=lr))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_ids = inputs[:, 0]  # (N, 3, L) -> (N, L)
        token_type_ids = inputs[:, 1]  # (N, 3, L) -> (N, L)
        attention_mask = inputs[:, 2]  # (N, 3, L) -> (N, L)
        H_all = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)[0]   # (N, L, H)
        return H_all

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch  # (N, L, 3), (N, L, 2)
        H_all = self.forward(inputs)  # (N, 3, L) -> (N, L, H)
        # H_all = H_all[:, 1:]  # (N, L, H) -> (N, L-1, H)
        # H_all로 부터 각 레이블에 해당하는 로짓값을 구하기
        logits_1 = self.W_1(H_all)  # (N, L, H) -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수 / 3
        logits_2 = self.W_2(H_all)  # (N, L, H) -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수 / 13

        logits_1 = torch.einsum("nlc->ncl", logits_1)    # (N, L, T_1) -> (N, T_1, L)
        logits_2 = torch.einsum("nlc->ncl", logits_2)    # (N, L, T_2) -> (N, T_2, L)

        labels_1 = targets[:, 0]  # (N, 2, L) -> (N, L)
        labels_2 = targets[:, 1]  # (N, 2, L) -> (N, L)

        loss_1 = F.cross_entropy(logits_1, labels_1)    # (N, T_1, L), (N, L) -> (N, L)
        loss_2 = F.cross_entropy(logits_2, labels_2)    # (N, T_2, L), (N, L) -> (N, L)

        loss_1 = loss_1.sum()   # (N, L) -> 1
        loss_2 = loss_2.sum()   # (N, L) -> 1

        # 정확도 계산 - 배치의 accuracy 수집
        self.train_acc_1.update(logits_1, labels_1)
        self.train_acc_2.update(logits_2, labels_2)

        # multitask learning
        loss = loss_1 + loss_2
        self.log("Train/loss", loss)
        return {
            "loss": loss
        }

    def on_train_epoch_end(self) -> None:
        acc_1 = self.train_acc_1.compute()
        acc_2 = self.train_acc_2.compute()
        self.train_acc_1.reset()
        self.train_acc_2.reset()
        self.log("Train/acc_1", acc_1)
        self.log("Train/acc_2", acc_2)

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: (N, 3, L)
        :return:
        """
        H_all = self.forward(inputs)  # (N,3, L) -> (N, L, H)

        logits_1 = self.W_1(H_all)  # (N, L, H) -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수 / 3
        logits_2 = self.W_2(H_all)  # (N, L, H) -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수 / 13

        probs_1 = torch.softmax(logits_1, 2)  # -> (N, L, T_1)
        probs_2 = torch.softmax(logits_2, 2)  # -> (N, L, T_2)

        labels_1 = torch.argmax(probs_1, 2)  # (N, L, T_1) -> (N, L)
        labels_2 = torch.argmax(probs_2, 2)  # (N, L, T_2) -> (N, L)

        return labels_1, labels_2

    def configure_optimizers(self):
        # 옵티마이저 설정은 여기에서
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    def validation_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        inputs, targets = batch  # (N, L, 3), (N, L, 2)
        H_all = self.forward(inputs)  # (N, 3, L) -> (N, L, H)

        logits_1 = self.W_1(H_all)  # (N, L, H) -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수 / 3
        logits_2 = self.W_2(H_all)  # (N, L, H) -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수 / 13

        logits_1 = torch.einsum("nlc->ncl", logits_1)    # (N, L, T_1) -> (N, T_1, L)
        logits_2 = torch.einsum("nlc->ncl", logits_2)    # (N, L, T_2) -> (N, T_2, L)

        labels_1 = targets[:, 0]  # (N, 2, L) -> (N, L)
        labels_2 = targets[:, 1]  # (N, 2, L) -> (N, L)

        loss_1 = F.cross_entropy(logits_1, labels_1)    # (N, T_1, L), (N, L) -> (N, L)
        loss_2 = F.cross_entropy(logits_2, labels_2)    # (N, T_2, L), (N, L) -> (N, L)

        loss_1 = loss_1.sum()   # (N, L) -> 1
        loss_2 = loss_2.sum()   # (N, L) -> 1

        # 정확도 계산
        self.val_acc_1.update(logits_1, labels_1)
        self.val_acc_2.update(logits_2, labels_2)

        # multitask learning
        loss = loss_1 + loss_2
        self.log("Validation/loss", loss)
        return {
            'loss': loss
        }

    def on_validation_epoch_end(self) -> None:
        acc_1 = self.val_acc_1.compute()
        acc_2 = self.val_acc_2.compute()
        self.val_acc_1.reset()
        self.val_acc_2.reset()
        self.log("Validation/acc_1", acc_1)
        self.log("Validation/acc_2", acc_2)

    def test_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        inputs, targets = batch  # (N, L, 3), (N, L, 2)
        H_all = self.forward(inputs)  # (N, 3, L) -> (N, L, H)

        logits_1 = self.W_1(H_all)  # (N, L, H) -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수 / 3
        logits_2 = self.W_2(H_all)  # (N, L, H) -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수 / 13

        logits_1 = torch.einsum("nlc->ncl", logits_1)    # (N, L, T_1) -> (N, T_1, L)
        logits_2 = torch.einsum("nlc->ncl", logits_2)    # (N, L, T_2) -> (N, T_2, L)

        labels_1 = targets[:, 0]  # (N, 2, L) -> (N, L)
        labels_2 = targets[:, 1]  # (N, 2, L) -> (N, L)

        # 정확도 계산
        # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#id3
        # probs/logits 다 상관없음.
        self.test_acc_1.update(logits_1, labels_1)
        self.test_acc_2.update(logits_2, labels_2)

    def on_test_end(self) -> None:
        # TODO: accuracy logging 하기.
        acc_1 = self.test_acc_1.compute()
        acc_2 = self.test_acc_2.compute()
        self.test_acc_1.reset()
        self.test_acc_2.reset()
        self.log("Test/acc_1", acc_1)
        self.log("Test/acc_2", acc_2)

    # boilerplate - 필요는 없는데 구현은 해야해서 그냥 여기에 둠.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
