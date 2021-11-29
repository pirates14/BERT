import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer


def main():
    tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
    # https://github.com/huggingface/transformers/issues/8217
    MAX_LENGTH = 30
    # 중요: [CLS] & [SEP]가 문장 속에 포함되어 있어야 함.
    sents = ["[CLS] 모순 ##아니 ##냐는 비판이 일각 ##에서 나왔다 [SEP]".split(),
             "[CLS] 모순 ##아니 ##냐는 비판이 일각 ##에서 [SEP]".split()]
    encoded = [tokenizer.convert_tokens_to_ids(tokens) for tokens in sents]
    input_ids = torch.LongTensor(pad_sequences(encoded, maxlen=MAX_LENGTH, dtype=int,
                                 padding="post", value=tokenizer.pad_token_id))  # 패딩 토큰을 토크나이저에서 가져오기
    token_type_ids = torch.zeros(size=(len(sents), MAX_LENGTH))   # 어차피 첫문장만 있음
    attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1, 0)  # 패딩인 것은 필요 없음
    print(input_ids)
    print(token_type_ids)
    print(attention_mask)
    inputs = torch.stack([input_ids, token_type_ids, attention_mask], dim=1)
    print(inputs.shape)  # (N, 3, L)


if __name__ == '__main__':
    main()