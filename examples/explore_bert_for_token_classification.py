from transformers import BertForTokenClassification


def main():
    ner_model = BertForTokenClassification.from_pretrained('dslim/bert-base-NER')
    print(ner_model.config.num_labels)  # num_labels = 개체명의 개수
    print(ner_model.config.id2label)  # 개체명을 여기서 확인할 수 있음
    print(ner_model.config)  # 모델에 대한 거의 모든 정보를 여기서 얻을 수 있음


if __name__ == '__main__':
    main()
