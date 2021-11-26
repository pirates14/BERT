import torch
from transformers import BertTokenizer, AutoTokenizer


def main():
    sent = "중국 베이징에서 근무했던 한 외교 소식통은 \"중국에서는 관료나 기업인, 유명인사 누구든지 쥐도 새도 모르게 당국에 끌려갈 수 있다\"며 \"중국 공산당이 통치를 유지하기 위한 공포정치의 한 수단\"이라고 귀띔했다."
    tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base")
    encoded = tokenizer(sent)
    print(encoded)
    print(encoded['input_ids'])
    print([tokenizer.decode(token_id) for token_id in encoded['input_ids']])


    # konlpy등을 사용해서 미리 토큰화하기. tokens = okt.morphs(sent)
    tokens = [["모순", "##아니", "##냐는", "비판이", "일각", "##에서", "나왔다"],
              ["모순", "##아니", "##냐는", "비판이", "일각", "##에서"]]
    encoded = tokenizer(tokens,
                        is_split_into_words=True,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True)
    print(encoded['input_ids'])
    print([tokenizer.decode(token_id) for token_id in encoded['input_ids']])

if __name__ == '__main__':
    main()