from typing import Tuple, Optional

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from BERT.loaders import load_petite, load_config
from BERT.tensors import InputsBuilder, TargetsBuilder


class NERDataset(Dataset):

    def __init__(self,  inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs  # (N, 3, L)
        self.targets = targets  # (N, 2, L)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # e.g.
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        # return item
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        data['Sentence #'] = data['Sentence #'].fillna(method='ffill')
        data.dropna(axis=0, inplace=True)

        # w(word), p(pos),t(tag) 튜플로
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["ANM"].values.tolist(),
                                                           s["NER"].values.tolist())]
        # 문장별로 w, p, t로 된 튜플이 리스트로 묶인다. -> 시리즈
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        # 위 시리즈 -> 리스트
        self.sentences = [s for s in self.grouped]

    # 다음 문장을 뽑는다.
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s

        except:
          # 문장이 없을 때 예외 처리
            return None


class NERDataModule(LightningDataModule):

    def __init__(self, config: dict, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def setup(self, stage: Optional[str] = None) -> None:
        petite = load_petite()

        # 메모리에서 스플릿을 하는 것도 괜찮다.
        # TODO: 단, random seed 반드시 고정하기
        train, test = train_test_split(petite, test_size=0.2, shuffle=True, random_state=self.config['seed'])
        train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=self.config['seed'])

        train = SentenceGetter(train).sentences
        test = SentenceGetter(test).sentences
        val = SentenceGetter(val).sentences

        train_inputs = InputsBuilder(self.tokenizer, sentences=train, max_len=self.config['max_length'])
        train_targets = TargetsBuilder(self.tokenizer, sentences=train, max_len=self.config['max_length'])
        test_inputs = InputsBuilder(self.tokenizer, sentences=test, max_len=self.config['max_length'])
        test_targets = TargetsBuilder(self.tokenizer, sentences=test, max_len=self.config['max_length'])
        val_inputs = InputsBuilder(self.tokenizer, sentences=val, max_len=self.config['max_length'])
        val_targets = TargetsBuilder(self.tokenizer, sentences=val, max_len=self.config['max_length'])

        self.train_dataset = NERDataset(inputs=train_inputs(), targets=train_targets())
        self.test_dataset = NERDataset(inputs=test_inputs(), targets=test_targets())
        self.val_dataset = NERDataset(inputs=val_inputs(), targets=val_targets())

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'],
                                shuffle=True, num_workers=2)
        return dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataloader = DataLoader(self.val_dataset,
                                shuffle=False, batch_size=self.config['batch_size'], num_workers=2)
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataloader = DataLoader(self.test_dataset,  shuffle=True,
                                batch_size=self.config['batch_size'], num_workers=2)
        return dataloader

    # 무시
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

