
from BERT.classes import ANM_LABELS



def main():
    data = [("익명의", "B-ANM"), ("관계자는", "I-ANM"), ("말했다", "o")]
    print([(word, ANM_LABELS.index(cls)) for word, cls in data])



if __name__ == '__main__':
    main()