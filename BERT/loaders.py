"""
로컬에서 메모리로 무엇인가를 로드한다 -> 전부 loaders에 함수로 정의하기.
"""

import pandas as pd
import yaml
from BERT.paths import CONFIG_JSON, DATASET_CSV


# exactly the same as:
def load_config() -> dict:
    with open(CONFIG_JSON, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_CSV, encoding='cp949')
