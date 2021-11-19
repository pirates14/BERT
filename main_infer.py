"""
main_infer.py는 모델의 예측값을 정성적으로 살펴보기 위한 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_infer.py
"""

def main():
    tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

    test_sentence = """
    군 관계자는 "큰 건물이 없는 도서 지역은 긴급 상황이 발생해도 주민들이 대피할 만한 곳이 없다"며 "대피 시설이 생기면 주민의 불안감이 어느정도는 해소될 것" 이라고 기대했다.
    """

    model = torch.load('total_model.tar')
    # print(model.ner_model)

    ner = pipeline("ner", model.ner_model, tokenizer=tokenizer)
    anm = pipeline("ner", model.anm_model, tokenizer=tokenizer)

    for entity1, entity2 in zip(anm(test_sentence), ner(test_sentence)):
        print([Label_encoder1.inverse_transform([int(entity1['entity'][-1])])][0], [Label_encoder2.inverse_transform([int(entity2['entity'][-1])])][0], entity1['word'])



if __name__ == '__main__':
    main()
