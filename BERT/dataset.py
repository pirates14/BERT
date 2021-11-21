from typing import Optional, Tuple, Dict
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch


class NERDataset(Dataset):

    def __init__(self,  inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs  # (N, L, 3)
        self.targets = targets  # (N, L, 2)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: X, Y의 차원을 고려하여, 배치를 출력하기.
        # e.g.        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #         item['labels'] = torch.tensor(self.labels[idx])
        #         return item
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.targets.shape[0]


# include this under setup method
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import LabelEncoder
# from transformers import AutoTokenizer
# from BERT.data import BERTDataset
#
# df1 = pd.read_csv('petite_data_set.csv', encoding='utf-8')
#
#
# df1['Sentence #'] = df1['Sentence #'].fillna(method='ffill')
# Label_encoder = LabelEncoder()
# df1["NER"] = Label_encoder.fit_transform(df1["NER"])
# print(list(Label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7])))
#
#
# df1.dropna(axis=0, inplace=True)
#
#
# # 문장별로 그룹화
# sentences = list(df1.groupby("Sentence #")["Word"].apply(list).reset_index()['Word'].values)
# vals = list(df1.groupby("Sentence #")["NER"].apply(list).reset_index()['NER'].values)
# sentence = [" ".join(s) for s in sentences]
#
# # from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
# tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')
#
# dataset = NERDataset(sentence, vals, tokenizer, 100)
#
# train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)