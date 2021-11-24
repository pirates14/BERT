import torch
import pytorch_lightning as pl
from typing import Tuple, List
from argparse import Namespace

import torchmetrics
from torch import nn
from transformers import BertModel, BertTokenizer
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.nn import functional as F

from BERT.tensors import InputsBuilder


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
        self.accuracy = torchmetrics.Accuracy()
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

        # H_all로 부터 각 레이블에 해당하는 로짓값을 구하기
        # a = H_all.size()    -> 1, 100, 768
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
        acc1 = self.accuracy(logits_1, labels_1)
        self.log("anm_acc", acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        acc2 = self.accuracy(logits_2, labels_2)
        self.log("ner_acc", acc2, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # multitask learning
        loss = loss_1 + loss_2
        return {
            "loss": loss
        }

    def on_train_epoch_end(self) -> None:
        # TODO: 이걸 왜하는지 주석 달아주세요! (태형님)
        # reset() : resets internal variables and accumulators
        """
        reset()

        We imported necessary classes as Metric, NotComputableError
        and decorators to adapt the metric for distributed setting. In reset method,
        we reset internal variables _num_correct and _num_examples which are used to compute the custom metric.
        In updated method we define how to update the internal variables.
        And finally in compute method, we compute metric value.

        Notice that _num_correct is a tensor,
        since in update we accumulate tensor values. _num_examples is a python scalar since we accumulate normal integers.
        For differentiable metrics, you must detach the accumulated values before adding them to the internal variables.
        """
        # 사용자 지정 메트릭을 계산하는 내부 변수 _num_correct 및 _num_예제를 초기화
        # 업데이트에서 텐서 값을 누적하므로 누적 값을 내부 변수에 추가하기 전에 분리 해야함
        # 솔직히 무슨소리인지 모르겠음

        self.accuracy.reset()

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: (N, 3, L)
        :return:
        """
        # TODO: inference 진행하기! (은정님, 태형님)
        H_all = self.forward(inputs)  # (N,3, L) -> (N, L, H)

        logits_1 = self.W_1(H_all)  # (N, L, H) -> (N, L, T_1)  T_1 =  W_1이 분류하는 토큰의 개수 / 3
        logits_2 = self.W_2(H_all)  # (N, L, H) -> (N, L, T_2)  T_2 = W_2가 분류하는 토큰의 개수 / 13

        probs_1 = torch.softmax(logits_1, 2)  #  -> (N, L, T_1)
        probs_2 = torch.softmax(logits_2, 2)  #  -> (N, L, T_2)

        labels_1 = torch.argmax(probs_1, 2)  # (N, L, T_1) -> (N, L)
        labels_2 = torch.argmax(probs_2, 2)  # (N, L, T_2) -> (N, L)

        return labels_1, labels_2

    def configure_optimizers(self):
        # 옵티마이저 설정은 여기에서
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    # boilerplate - 필요는 없는데 구현은 해야해서 그냥 여기에 둠.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

