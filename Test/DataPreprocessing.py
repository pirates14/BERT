import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from dataset import ner_dataset

df1 = pd.read_csv('petite_data_set.csv', encoding='utf-8')


df1['Sentence #'] = df1['Sentence #'].fillna(method='ffill')
Label_encoder = LabelEncoder()
df1["NER"] = Label_encoder.fit_transform(df1["NER"])
print(list(Label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7])))


df1.dropna(axis=0, inplace=True)


# 문장별로 그룹화
sentences = list(df1.groupby("Sentence #")["Word"].apply(list).reset_index()['Word'].values)
vals = list(df1.groupby("Sentence #")["NER"].apply(list).reset_index()['NER'].values)
sentence = [" ".join(s) for s in sentences]

# from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

dataset = ner_dataset(sentence, vals, tokenizer, 100)

train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)