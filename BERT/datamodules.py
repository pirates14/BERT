import torch
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from BERT.loaders import load_dataset
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


class AnmSourceNERDataModule(LightningDataModule):
    """
    (token, anm, source)
    """
    def __init__(self, config: dict, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        # to be built
        self.dataset = None
        self.train: Optional[NERDataset] = None
        self.val: Optional[NERDataset] = None
        self.test: Optional[NERDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # --- load the dataset, and preprocess it to get the sentences --- #
        dataset_df = load_dataset()  # noqa
        dataset_df['Sentence #'] = dataset_df['Sentence #'].fillna(method='ffill')
        sentences = dataset_df.dropna(axis=0) \
            .groupby("Sentence #") \
            .apply(
            lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                    s["ANM"].values.tolist(),
                                                    s["NER"].values.tolist())]) \
            .tolist()
        # -- build the tensors, and thereby the dataset --- #
        inputs = InputsBuilder(self.tokenizer, sentences, self.config['max_length'])()  # (N, 3, L)
        targets = TargetsBuilder(self.tokenizer, sentences, self.config['max_length'])()  # (N, 2, L)
        self.dataset = NERDataset(inputs, targets)
        # --- split the dataset into train,val and test --- #
        val_size = int(len(self.dataset) * self.config["val_ratio"])
        test_size = int(len(self.dataset) * self.config["test_ratio"])
        train_size = len(self.dataset) - val_size - test_size
        self.train, self.val, self.test = \
            random_split(self.dataset, lengths=(train_size, val_size, test_size),
                         generator=torch.Generator().manual_seed(self.config['seed']))

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.train, batch_size=self.config['batch_size'],
                                num_workers=self.config['num_workers'], shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.val, batch_size=self.config['batch_size'],
                                num_workers=self.config['num_workers'], shuffle=False)
        return dataloader

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(self.test, batch_size=self.config['batch_size'],
                                num_workers=self.config['num_workers'], shuffle=False)
        return dataloader

    # 무시하기
    def predict_dataloader(self) -> DataLoader:
        pass


class AnmNERDataModule(AnmSourceNERDataModule):
    """
    (token, anm)
    """
    def setup(self, stage: Optional[str] = None) -> None:
        super(AnmNERDataModule, self).setup()
        self.dataset.targets = self.dataset.targets[:, 0]

class SourceNERDataModule(AnmSourceNERDataModule):
    """
    (token, source)
    """
    def setup(self, stage: Optional[str] = None) -> None:
        super(SourceNERDataModule, self).setup()
        self.dataset.targets = self.dataset.targets[:, 1]