from transformers import BertTokenizer


def main():
    sent = "모순아니냐는 비판이 일각에서 나왔다"
    tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
    encoded = tokenizer(sent)
    print(encoded['input_ids'])
    print([tokenizer.decode(token_id) for token_id in encoded['input_ids']])


    # konlpy등을 사용해서 미리 토큰화하기. tokens = okt.morphs(sent)
    tokens = ["모순", "##아니", "##냐는", "비판이", "일각", "##에서", "나왔다"]
    encoded = tokenizer(tokens,
                        is_split_into_words=True)
    print(encoded['input_ids'])
    print([tokenizer.decode(token_id) for token_id in encoded['input_ids']])


if __name__ == '__main__':
    main()