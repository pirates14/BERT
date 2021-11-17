from torch.utils.data import Dataset
import torch


class ner_dataset(Dataset):

    def __init__(self, sentences, vals, tokenizer, max_len):
        # 문장
        self.sentences = sentences
        # NER(tag)
        self.vals = vals
        # 토크나이저
        self.tokenizer = tokenizer
        # 최대 길이
        self.max_len = max_len

    def __getitem__(self, idx):

        # 해당 문자를 " "로 나눈다. (한국어에서도 이렇게 해도 되나??)
        s = self.sentences[idx].split(" ")
        # 그 문장에 맞는 tag를 가져옴
        v = self.vals[idx]
        d = {'input_ids': [], 'attention_mask': [], 'labels': []}
        # 리스트안에 인코딩된 값들이 각각 들어간다.
        text, labels, mask = [], [], []
        for w in range(len(s)):
            # 0 부터 len(s)까지 각 단어를 토큰화
            i, l = self.align_labels(self.tokenizer, s[w], v[w])
            text.extend(i['input_ids'])
            labels.extend(l)
            mask.extend(i['attention_mask'])
        # 문장 앞뒤에 [101], [102] 추가
        d['input_ids'] = [101] + self.pad(text + [102], self.max_len)
        # tag 앞뒤에 [0], [1] 추가
        d['labels'] = [0] + self.pad(labels + [0], self.max_len)
        # mask 앞뒤에 1 추가
        d['attention_mask'] = [1] + self.pad(mask + [1], self.max_len)

        # tensor
        d['input_ids'] = torch.tensor(d['input_ids'])
        d['labels'] = torch.tensor(d['labels'])
        d['attention_mask'] = torch.tensor(d['attention_mask'])

        return d

    def __len__(self):
        return len(self.sentences)

    def align_labels(self, tokenizer, word, label):
        # 각 단어를 토큰화
        word = tokenizer(word, add_special_tokens=False)
        labels = []
        # 워드서브 일때
        # '안녕' -> '안', '##녕'
        # 이에 맞는 라벨 값이 각각 들어가야함
        for i in range(len(word['input_ids'])):
            labels.append(label)
        return word, labels

    # 정해준 길이 만큼 pad 추가
    def pad(self, s, max_len):
        pad_len = max_len - len(s)
        if pad_len > 0:
            for i in range(pad_len):
                s.append(0)
        return s[:max_len - 1]