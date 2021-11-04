import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from torch.utils.data import DataLoader

from transformers.optimization import get_cosine_schedule_with_warmup, AdamW


class anonymousclf(torch.nn.Module):
    def __init__(self, bert: BertModel, num_class: int, device: torch.device):
        super().__init__()
        self.bert = bert
        self.H = bert.config.hidden_size
        self.W_hy = torch.nn.Linear(self.H, num_class)  # (H, 3)
        self.to(device)

    def forward(self, X: torch.Tensor):
        """
        :param X:
        :return:
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert(input_ids, token_type_ids, attention_mask)[0]

        return H_all

    def predict(self, X):
        H_all = self.forward(X)  # N, L, H
        H_cls = H_all[:, 0]  # 첫번째(cls)만 가져옴 (N,H)
        # N,H  H,3 -> N,3

        y_hat = self.W_hy(H_cls)
        return y_hat  # N,3

    def training_step(self, X, y):
        '''
        :param X:
        :param y:
        :return: loss
        '''
        # y = torch.LongTensor(y)
        y_pred = self.predict(X)
        y_pred = F.softmax(y_pred, dim=1)
        # loss
        loss = F.cross_entropy(y_pred, y).sum()
        return loss

def Build_X (sents, tokenizer):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1)

# USE_CUDA = torch.cuda.is_available()
# print(USE_CUDA)


# device = torch.device('cuda:0')
# print('학습을 진행하는 기기:', device)


def predict(DATA):
  bertmodel = BertModel.from_pretrained("monologg/kobert")
  tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
  # device = torch.device('cpu')
  # model = torch.load("/content/drive/MyDrive/epoch10.pth")
  model = torch.load(r'C:\Users\KimTaehyeong\PycharmProjects\model_in_django\epoch20.pth', map_location=torch.device('cpu'))
  # print(model)
  # model.eval()
  X = Build_X(DATA, tokenizer)
  y_hat = model.predict(X)
  y_hat = F.softmax(y_hat, dim=1)
  print(y_hat[0])
  [A, B] = y_hat[0].tolist()
  x = [A*100, B*100]
  x = np.round(x, 2)
  print(DATA)
  return x[0], x[1]
# DATA = ["회담 내막에 밝은 고위 외교 소식통은 \"그때 코언 청문회가 없었다면 하노이 회담 결과와 한반도 정세도 달랐을 것\" 이라고 한탄했다."]
# predict(DATA)