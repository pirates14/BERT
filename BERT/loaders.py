"""
로컬에서 메모리로 무엇인가를 로드한다 -> 전부 loaders에 함수로 정의하기.
"""

import pandas as pd
import yaml
import torch
from BERT.paths import CONFIG_JSON, PETITE_CSV


# exactly the same as:
# https://github.com/wisdomify/wisdomify/blob/main/wisdomify/loaders.py
def load_config() -> dict:
    with open(CONFIG_JSON, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        if not torch.cuda.is_available():
            raise ValueError("cuda is unavailable")
        else:
            return torch.device("cuda")
    return torch.device("cpu")


def load_petite() -> pd.DataFrame:
    return pd.read_csv(PETITE_CSV)
