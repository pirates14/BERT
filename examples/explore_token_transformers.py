from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizer
import torch

# model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# sequence = "안전상의 이유로 익명을 요구한 북한 내 소식통은 “지금보다도 더 먹을 것을 줄여가며 2020년대 중반까지 버티라는 일방적 지시에 주민들의 불신과 원성은 그 어느 때보다 팽배해 있다”고 RFA에 전했다"
# tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
# print(tokens)

#['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '“', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '”', '[UNK]', '[UNK]', '[UNK]', '[SEP]']
#^^..?
#-----------------------------------------------------------------

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

sent_ko = "안전상의 이유로 익명을 요구한 북한 내 소식통은 “지금보다도 더 먹을 것을 줄여가며 2020년대 중반까지 버티라는 일방적 지시에 주민들의 불신과 원성은 그 어느 때보다 팽배해 있다”고 RFA에 전했다"
corpus = tokenizer(sent_ko)
print(corpus)

#{'input_ids': [2, 0, 0, 0, 0, 6415, 5678, 0, 501, 0, 5837, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5538, 0, 0, 0, 7143, 502, 5439, 0, 0, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
#아니 저한테 이러시는 이유가 있을 거 아니에요.
