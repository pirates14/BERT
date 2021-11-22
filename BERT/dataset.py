from typing import Optional,  Dict, Tuple

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

from BERT.loaders import load_petite


# https://github.com/wisdomify/wisdomify/blob/main/wisdomify/datamodules.py


class NERDataset(Dataset):

    def __init__(self,  inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs  # (N, 3, L)
        self.targets = targets  # (N, 2, L)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: X, Y의 차원을 고려하여, 배치를 출력하기.
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

#
# getter = SentenceGetter(load_petite())
# sentences = getter.sentences        # list[list[tuple(word,anm,ner)]]


