from BERT.loaders import load_dataset


def main():
    dataset_df = load_dataset()
    for idx, row in dataset_df.iterrows():
        print(row[1], row[2], row[3])
        if idx > 100:
            break


if __name__ == '__main__':
    main()