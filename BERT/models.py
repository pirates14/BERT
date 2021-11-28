import torch
import torchmetrics
from torch import nn
from typing import Tuple, Optional
import pytorch_lightning as pl
from torch.nn import functional as F
from argparse import Namespace
from transformers import BertModel
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class MonoLabelNER(pl.LightningModule):
    name: str = "mono_label_ner"

    def __init__(self, lr: float, num_labels: int, hidden_size: int, bert: BertModel = None):
        """
        :param lr:
        :param num_labels:
        :param hidden_size:
        :param bert: 이 부분이 None이 될 수 있도록 해야 BiLabelNER 의 구성폼으로 사용가능
        """
        super().__init__()
        self.bert = bert
        self.bilstm = nn.LSTM(hidden_size, bidirectional=True, hidden_size=hidden_size // 2, batch_first=True)
        self.W = nn.Linear(hidden_size, num_labels)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(num_classes=num_labels, mdmc_average='global')
        self.val_f1 = torchmetrics.F1(num_classes=num_labels, mdmc_average='global')
        self.test_f1 = torchmetrics.F1(num_classes=num_labels, mdmc_average='global')
        self.attention_mask: Optional[torch.Tensor] = None
        self.save_hyperparameters(Namespace(lr=lr, num_labels=num_labels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_ids = inputs[:, 0]  # (N, 3, L) -> (N, L)
        token_type_ids = inputs[:, 1]  # (N, 3, L) -> (N, L).
        self.attention_mask = inputs[:, 2]  # (N, 3, L) -> (N, L)
        H_all = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=self.attention_mask)[0]  # (N, L, H)
        hidden_stats, _ = self.bilstm(H_all)
        return hidden_stats

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (N, 3, L)
        :return:
        """
        hidden_stats = self.forward(inputs)
        return self.predict_given_forward(hidden_stats)

    def predict_given_forward(self, hidden_stats: torch.Tensor):
        logits = self.W(hidden_stats)  # (N, L, H) -> (N, L, T)
        probs = torch.softmax(logits, -1)  # -> (N, L, T)
        labels = torch.argmax(probs, -1)  # (N, L, T) -> (N, L)
        return labels

    def configure_optimizers(self):
        # 옵티마이저 설정은 여기에서
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch  # (N, 3, L), (N, L)
        # H_all = H_all[:, 1:]  # (N, L, H) -> (N, L-1, H)
        # H_all 로 부터 각 레이블에 해당하는 로짓값을 구하기
        hidden_stats = self.forward(inputs)
        return self.training_step_given_forward(hidden_stats, self.attention_mask, targets)

    def training_step_given_forward(self, hidden_stats: torch.Tensor,
                                    attention_mask: torch.Tensor, targets: torch.Tensor) -> dict:
        logits = self.W(hidden_stats)  # (N, L, H) -> (N, L, T)
        logits = torch.einsum("nlc->ncl", logits)  # (N, L, T_1) -> (N, T_1, L)
        loss = F.cross_entropy(logits, targets)  # (N, T_1, L), (N, L) -> (N, L)
        # loss = torch.masked_fill(loss, mask=attention_mask == 0, value=0)
        loss = loss.sum()
        # 정확도 계산 - 배치의 accuracy 수집
        return {
            "loss": loss,
            # 이미 학습에 사용했으니, 굳이 기울기를 유지할필요가 없다.
            "logits": logits.detach(),
        }

    def on_train_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        loss = outputs["loss"]
        logits = outputs["logits"]
        acc = self.train_acc(logits, targets)
        f1 = self.train_f1(logits, targets)
        self.log("Train/loss", loss, on_step=True)
        self.log("Train/f1", f1, on_step=True)
        self.log("Train/acc", acc, on_step=True)

    def on_train_epoch_end(self) -> None:
        self.train_f1.reset()
        self.train_acc.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_validation_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        loss = outputs["loss"]
        logits = outputs["logits"]
        f1 = self.val_f1(logits, targets)
        acc = self.val_acc(logits, targets)
        self.log("Validation/loss", loss, on_step=True)
        self.log("Validation/f1", f1, on_step=True)
        self.log("Validation/acc", acc, on_step=True)

    def on_validation_epoch_end(self) -> None:
        self.val_f1.reset()
        self.val_acc.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_test_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        _, targets = batch
        logits = outputs["logits"]
        acc = self.test_acc(logits, targets)
        f1 = self.test_f1(logits, targets)
        self.log("Test/acc", acc, on_step=True)
        self.log("Test/f1", f1, on_step=True)


    def on_test_epoch_end(self):
        self.test_f1.reset()
        self.test_acc.reset()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class BiLabelNER(pl.LightningModule):
    """
    글자당 두개의 레이블을 동시에 에측하는 모델.
    """
    name: str = "bi_label_ner"

    def __init__(self, bert: BertModel, lr: float, num_labels_pair: Tuple[int, int]):
        super().__init__()
        self.bert = bert
        self.bilstm = nn.LSTM(bert.config.hidden_size, bidirectional=True, hidden_size=bert.config.hidden_size// 2, batch_first=True)
        self.mono_1 = MonoLabelNER(lr=lr, num_labels=num_labels_pair[0], hidden_size=bert.config.hidden_size)
        self.mono_2 = MonoLabelNER(lr=lr, num_labels=num_labels_pair[1], hidden_size=bert.config.hidden_size)
        self.attention_mask: Optional[torch.Tensor] = None
        self.save_hyperparameters(Namespace(lr=lr, num_labels_pair=num_labels_pair))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_ids = inputs[:, 0]  # (N, 3, L) -> (N, L)
        token_type_ids = inputs[:, 1]  # (N, 3, L) -> (N, L)
        self.attention_mask = inputs[:, 2]  # (N, 3, L) -> (N, L)
        H_all = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=self.attention_mask)[0]  # (N, L, H)
        hidden_states, _ = self.bilstm(H_all)
        return hidden_states

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: (N, 3, L)
        :return:
        """
        hidden_states = self.forward(inputs)
        labels_1 = self.mono_1.predict_given_forward(hidden_states)  # (N, 3, L) -> (N, L)
        labels_2 = self.mono_2.predict_given_forward(hidden_states)  # (N, 3, L) -> (N, L)
        return labels_1, labels_2

    def configure_optimizers(self):
        # 옵티마이저 설정은 여기에서
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch
        hidden_states = self.forward(inputs)
        outputs_1 = self.mono_1.training_step_given_forward(hidden_states, self.attention_mask, targets[:, 0])
        outputs_2 = self.mono_2.training_step_given_forward(hidden_states, self.attention_mask, targets[:, 1])
        loss = outputs_1["loss"] + outputs_2["loss"]  # unweighted multi-task learning
        return {
            "loss": loss,
            "logits_1": outputs_1["logits"],
            "logits_2": outputs_2["logits"]
        }

    def on_train_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        f1_1 = self.mono_1.train_f1(outputs["logits_1"], targets[:, 0])
        f1_2 = self.mono_2.train_f1(outputs["logits_2"], targets[:, 1])
        acc_1 = self.mono_1.train_acc(outputs["logits_1"], targets[:, 0])
        acc_2 = self.mono_2.train_acc(outputs["logits_2"], targets[:, 1])
        self.log("Train/loss", outputs['loss'], on_step=True)  # 로스는 각 배치별로 로깅
        self.log("Train/f1_1", f1_1, on_step=True)
        self.log("Train/f1_2", f1_2, on_step=True)
        self.log("Train/acc_1", acc_1, on_step=True)
        self.log("Train/acc_2", acc_2, on_step=True)
        # acc_all = (self.mono_1.train_acc.correct + self.mono_2.train_acc.correct) \
        #     / (self.mono_1.train_acc.total + self.mono_2.train_acc.total)
        # self.log("Train/acc_all", acc_all, on_step=True)

    def on_train_epoch_end(self) -> None:
        self.mono_1.train_f1.reset()
        self.mono_2.train_f1.reset()
        self.mono_1.train_acc.reset()
        self.mono_2.train_acc.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_validation_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        f1_1 = self.mono_1.val_f1(outputs["logits_1"], targets[:, 0])
        f1_2 = self.mono_2.val_f1(outputs["logits_2"], targets[:, 1])
        acc_1 = self.mono_1.val_acc(outputs["logits_1"], targets[:, 0])
        acc_2 = self.mono_2.val_acc(outputs["logits_2"], targets[:, 1])
        self.log("Validation/loss", outputs['loss'], on_step=True)  # 로스는 각 배치별로 로깅
        self.log("Validation/f1_1", f1_1, on_step=True)
        self.log("Validation/f1_2", f1_2, on_step=True)
        self.log("Validation/acc_1", acc_1, on_step=True)
        self.log("Validation/acc_2", acc_2, on_step=True)
        # acc_all = (self.mono_1.val_acc.correct + self.mono_2.val_acc.correct) \
        #     / (self.mono_1.val_acc.total + self.mono_2.val_acc.total)
        # self.log("Validation/acc_all", acc_all)

    def on_validation_epoch_end(self) -> None:
        self.mono_1.val_f1.reset()
        self.mono_2.val_f1.reset()
        self.mono_1.val_acc.reset()
        self.mono_2.val_acc.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_test_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        _, targets = batch
        logits_1 = outputs["logits_1"]
        logits_2 = outputs["logits_2"]
        self.mono_1.test_f1.update(logits_1, targets[:, 0])
        self.mono_2.test_f1.update(logits_2, targets[:, 1])
        self.mono_1.test_acc.update(logits_1, targets[:, 0])
        self.mono_2.test_acc.update(logits_2, targets[:, 1])

        # acc_all = (self.mono_1.test_acc.correct + self.mono_2.test_acc.correct) \
        #     / (self.mono_1.test_acc.total + self.mono_2.test_acc.val_acc.total)
        # self.log("Test/acc_all", acc_all, on_step=True)

    def on_test_epoch_end(self) -> None:
        f1_1 = self.mono_1.test_f1.compute()
        f1_2 = self.mono_2.test_f1.compute()
        acc_1 = self.mono_1.test_acc.compute()
        acc_2 = self.mono_2.test_acc.compute()
        self.mono_1.test_f1.reset()
        self.mono_2.test_f1.reset()
        self.mono_1.test_acc.reset()
        self.mono_2.test_acc.reset()
        self.log("Test/f1_1", f1_1, on_epoch=True)
        self.log("Test/f1_2", f1_2, on_epoch=True)
        self.log("Test/acc_1", acc_1, on_epoch=True)
        self.log("Test/acc_2", acc_2, on_epoch=True)

    # boilerplate - 필요는 없는데 구현은 해야해서 그냥 여기에 둠.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
