"""
로컬에서 메모리로 무엇인가를 로드한다 -> 전부 loaders에 함수로 정의하기.
"""
import yaml
import pandas as pd
from BERT.paths import CONFIG_JSON, DATASET_CSV


# exactly the same as
def load_config(ver: str) -> dict:
    with open(CONFIG_JSON, encoding="utf-8") as fh:
        return yaml.safe_load(fh)[ver]


# load the dataset to train a model
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_CSV, encoding='utf-8')
