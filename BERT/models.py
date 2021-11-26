import torch
import torchmetrics
from torch import nn
from typing import Tuple
import pytorch_lightning as pl
from torch.nn import functional as F
from argparse import Namespace
from transformers import BertModel
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class MonoLabelNER(pl.LightningModule):
    # TODO: 레이블을 하나만 예측하는 모델도 구현하기
    # ignored
    def __init__(self, bert: BertModel, lr: float, num_labels: int):
        super().__init__()
        self.bert = bert
        self.W = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.save_hyperparameters(Namespace(lr=lr, num_labels=num_labels))

    @property
    def name(self):
        """
        wandb artifact 이름
        """
        return "mono_label_ner"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_ids = inputs[:, 0]  # (N, 3, L) -> (N, L)
        token_type_ids = inputs[:, 1]  # (N, 3, L) -> (N, L)
        attention_mask = inputs[:, 2]  # (N, 3, L) -> (N, L)
        H_all = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)[0]  # (N, L, H)
        return H_all

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (N, 3, L)
        :return:
        """
        H_all = self.forward(inputs)
        return self.predict_forward_given(H_all)

    def predict_forward_given(self, H_all: torch.Tensor):
        logits = self.W(H_all)  # (N, L, H) -> (N, L, T)
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
        H_all = self.forward(inputs)
        return self.training_step_forward_given(H_all, targets)

    def training_step_forward_given(self, H_all: torch.Tensor, targets: torch.Tensor) -> dict:
        logits = self.W(H_all)  # (N, L, H) -> (N, L, T)
        logits = torch.einsum("nlc->ncl", logits)  # (N, L, T_1) -> (N, T_1, L)
        loss = F.cross_entropy(logits, targets).sum()  # (N, T_1, L), (N, L) -> (N, L) -> ()
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
        self.log("Train/loss", loss)
        self.log("Train/acc", acc)

    def on_train_epoch_end(self) -> None:
        self.train_acc.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_validation_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        loss = outputs["loss"]
        logits = outputs["logits"]
        acc = self.val_acc(logits, targets)
        self.log("Validation/loss", loss)
        self.log("Validation/acc", acc)

    def on_validation_epoch_end(self) -> None:
        self.val_acc.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_test_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        _, targets = batch
        logits = outputs["logits"]
        acc = self.test_acc(logits, targets)
        self.log("Test/acc", acc)

    def on_test_epoch_end(self):
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

    def __init__(self, bert: BertModel, lr: float, num_labels_pair: Tuple[int, int]):
        super().__init__()
        self.bert = bert
        self.mono_1 = MonoLabelNER(bert, lr=lr, num_labels=num_labels_pair[0])
        self.mono_2 = MonoLabelNER(bert, lr=lr, num_labels=num_labels_pair[1])
        self.save_hyperparameters(Namespace(lr=lr, num_labels_pair=num_labels_pair))

    @property
    def name(self):
        """
        wandb artifact 이름
        """
        return "bi_label_ner"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_ids = inputs[:, 0]  # (N, 3, L) -> (N, L)
        token_type_ids = inputs[:, 1]  # (N, 3, L) -> (N, L)
        attention_mask = inputs[:, 2]  # (N, 3, L) -> (N, L)
        H_all = self.bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask)[0]  # (N, L, H)
        return H_all

    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: (N, 3, L)
        :return:
        """
        H_all = self.forward(inputs)
        labels_1 = self.mono_1.predict_forward_given(H_all)  # (N, 3, L) -> (N, L)
        labels_2 = self.mono_2.predict_forward_given(H_all)  # (N, 3, L) -> (N, L)
        return labels_1, labels_2

    def configure_optimizers(self):
        # 옵티마이저 설정은 여기에서
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    def training_step(self, batch: Tuple[torch.Tensor, torch.tensor]) -> dict:
        inputs, targets = batch
        H_all = self.forward(inputs)
        outputs_1 = self.mono_1.training_step_forward_given(H_all, targets[:, 0])
        outputs_2 = self.mono_2.training_step_forward_given(H_all, targets[:, 1])
        loss = outputs_1["loss"] + outputs_2["loss"]  # unweighted multi-task learning
        return {
            "loss": loss,
            "logits_1": outputs_1["logits"],
            "logits_2": outputs_2["logits"]
        }

    def on_train_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        acc_1 = self.mono_1.train_acc(outputs["logits_1"], targets[:, 0])
        acc_2 = self.mono_2.train_acc(outputs["logits_2"], targets[:, 1])
        self.log("Train/loss", outputs['loss'])  # 로스는 각 배치별로 로깅
        self.log("Train/acc_1", acc_1)
        self.log("Train/acc_2", acc_2)
        acc_all = (self.mono_1.train_acc.correct + self.mono_2.train_acc.correct) \
            / (self.mono_1.train_acc.total + self.mono_2.train_acc.total)
        self.log("Train/acc_all", acc_all)

    def on_train_epoch_end(self) -> None:
        self.mono_1.train_acc.reset()
        self.mono_2.train_acc.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_validation_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> None:
        _, targets = batch
        acc_1 = self.mono_1.val_acc(outputs["logits_1"], targets[:, 0])
        acc_2 = self.mono_2.val_acc(outputs["logits_2"], targets[:, 1])
        self.log("Validation/loss", outputs['loss'])  # 로스는 각 배치별로 로깅
        self.log("Validation/acc_1", acc_1)
        self.log("Validation/acc_2", acc_2)
        acc_all = (self.mono_1.val_acc.correct + self.mono_2.val_acc.correct) \
            / (self.mono_1.val_acc.total + self.mono_2.val_acc.total)
        self.log("Validation/acc_all", acc_all)

    def on_validation_epoch_end(self) -> None:
        self.mono_1.val_acc.reset()
        self.mono_2.val_acc.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.tensor], *args) -> dict:
        return self.training_step(batch)

    def on_test_batch_end(self, outputs: dict, batch: Tuple[torch.Tensor, torch.Tensor], *args):
        _, targets = batch
        logits_1 = outputs["logits_1"]
        logits_2 = outputs["logits_2"]
        acc_1 = self.test_acc.update(logits_1, targets[:, 0])
        acc_2 = self.test_acc.update(logits_2, targets[:, 1])
        self.log("Test/acc_1", acc_1)
        self.log("Test/acc_2", acc_2)
        acc_all = (self.mono_1.test_acc.correct + self.mono_2.test_acc.correct) \
            / (self.mono_1.test_acc.total + self.mono_2.test_acc.val_acc.total)
        self.log("Test/acc_all", acc_all)

    def on_test_epoch_end(self) -> None:
        self.mono_1.test_acc.reset()
        self.mono_2.test_acc.reset()

    # boilerplate - 필요는 없는데 구현은 해야해서 그냥 여기에 둠.
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
