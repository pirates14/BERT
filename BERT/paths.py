from pathlib import Path
from os import path

# directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(ROOT_DIR, "data")

# files
CONFIG_JSON = path.join(ROOT_DIR, "config.yaml")  # 실험을 위한 하이퍼파라미터 저장
PETITE_CSV = path.join(DATA_DIR, "petite.csv")  # 학습데이터
# 모델 체크포인트 저장 paths
SOURCE_ANM_NER_CKPT = path.join(DATA_DIR, "source_anm_ner.ckpt")  # 취재원 + 익명 동시 예측모델
SOURCE_NER_CKPT = path.join(DATA_DIR, "source_ner.ckpt")  # 취재원 예측 모델
ANM_NER_CKPT = path.join(DATA_DIR, "anm_ner.ckpt")  # 익명성 에측 모델
