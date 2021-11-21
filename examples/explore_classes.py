
from BERT.classes import ANM_CLASSES



def main():
    data = [("익명의", "B-ANM"), ("관계자는", "I-ANM"), ("말했다", "o")]
    print([(word, ANM_CLASSES.index(cls)) for word, cls in data])



if __name__ == '__main__':
    main()