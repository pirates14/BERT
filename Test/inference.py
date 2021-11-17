import torch
from transformers import AutoTokenizer

from Test.DataPreprocessing import Label_encoder
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

model = torch.load('total_model.tar')

test_sentence = """
국내에서는 탈 원전을 추진하면서 해외에는 원전을 팔려는 것은 모순 아니냐는 비판이 일각에서 나왔다
"""

# print(tokenizer.tokenize(test_sentence))
# print(list(Label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7])))

ner = pipeline("ner", model=model.to('cpu'), tokenizer=tokenizer)
for entity in ner(test_sentence):
    # print(entity)
    print(Label_encoder.inverse_transform([int(entity['entity'][-1])]), entity['word'])








