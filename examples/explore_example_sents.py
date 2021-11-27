from BERT.loaders import load_dataset


def main():
    dataset_df = load_dataset()  # noqa
    dataset_df['Sentence #'] = dataset_df['Sentence #'].fillna(method='ffill')
    sentences = dataset_df.dropna(axis=0) \
        .groupby("Sentence #") \
        .apply(
        lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                s["ANM"].values.tolist(),
                                                s["NER"].values.tolist())]) \
        .tolist()
    for sent in sentences:
        tokens = [token for token, _, _ in sent]
        print(" ".join(tokens))

        
if __name__ == '__main__':
    main()