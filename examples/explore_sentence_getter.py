from BERT.dataset import SentenceGetter
from BERT.loaders import load_dataset


def main():
    dataset_df = load_dataset()
    train = SentenceGetter(dataset_df).sentences
    for sent in train:
        print( " ".join([token for token, _, _ in sent]))



if __name__ == '__main__':
    main()