from pathlib import Path
from os import path
from os import makedirs
from BERT.models import BiLabelNER, MonoLabelNER
# directories
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = path.join(ROOT_DIR, "artifacts")
# files
CONFIG_JSON = path.join(ROOT_DIR, "config.yaml")  # 실험을 위한 하이퍼파라미터 저장
DATASET_CSV = path.join(ROOT_DIR, "dataset.csv")  # 학습데이터


# 모델 체크포인트 저장 paths
def bi_label_ner_ckpt() -> str:
    artifact_path = path.join(ARTIFACTS_DIR, BiLabelNER.name)
    makedirs(artifact_path, exist_ok=True)
    return path.join(artifact_path, "ner.ckpt")


def mono_label_ner_ckpt() -> str:
    artifact_path = path.join(ARTIFACTS_DIR, MonoLabelNER.name)
    makedirs(artifact_path, exist_ok=True)
    return path.join(artifact_path, "ner.ckpt")
