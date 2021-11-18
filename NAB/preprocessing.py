import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from dataset import NerDataset

df1 = pd.read_csv('petite_data_set.csv', encoding='utf-8')

df1['Sentence #'] = df1['Sentence #'].fillna(method='ffill')
df1.dropna(axis=0, inplace=True)

Label_encoder1 = LabelEncoder()
Label_encoder2 = LabelEncoder()

# # df1["NER"]를 이용 피팅하고 라벨숫자로 변환한다
df1["ANM"] = Label_encoder1.fit_transform(df1["ANM"])
df1["NER"] = Label_encoder2.fit_transform(df1["NER"])
# print(set(df1[["NER","ANM"]]))

# 문장별로 그룹화
sentences = list(df1.groupby("Sentence #")["Word"].apply(list).reset_index()['Word'].values)

# 위와 맞게 문장별로 tag를 묶음 ( 패딩 ㄴㄴ )
vals = list(df1.groupby("Sentence #")["NER"].apply(list).reset_index()['NER'].values)
avals = list(df1.groupby("Sentence #")["ANM"].apply(list).reset_index()['ANM'].values)

# 각 그룹화된 단어들을 합쳐 문장으로 만들어줌
sentence = [" ".join(s) for s in sentences]

# from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

dataset = NerDataset(sentence, vals, avals, tokenizer, 100)

train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)