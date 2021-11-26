"""
main_infer.py는 모델의 예측값을 정성적으로 살펴보기 위한 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_infer.py
"""
from typing import List, Tuple
import argparse
import torch
import random
import numpy as np
import wandb
from transformers import BertTokenizer, AutoConfig, AutoModel
from BERT.loaders import load_config
from BERT.models import BiLabelNER
from BERT.tensors import InputsBuilder
from BERT.labels import SOURCE_LABELS, ANM_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ver", type=str)
    parser.add_argument("--text", type=str, default="삼성전자 관계자는 “코로나19 방역 지침에 따라 행사를 조용하게 치렀다”고 밝혔다")
    args = parser.parse_args()
    config = load_config(args.ver)
    config.update(vars(args))  # command-line arguments 도 기록하기!
    # --- fix random seeds -- #
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = AutoModel.from_config(AutoConfig.from_pretrained(config['bert']))

    with wandb.init(project="BERT", config=config) as run:
        model_path = run.use_artifact(f"{BiLabelNER.name}:{config['ver']}")
        model = BiLabelNER.load_from_checkpoint(model_path, bert=bert)
        tokens: List[str] = tokenizer.tokenize(config)
        sentences: List[List[Tuple[str, str, str]]] = [[(token, "", "") for token in tokens]]
        inputs = InputsBuilder(tokenizer, sentences, config['max_length'])()
        model.freeze()
        # 원하는 결과
        anm_labels, source_labels = model.predict(inputs)  # (N, 3, L) -> (1, L), (1, L)
        anm_labels = anm_labels.squeeze()  # (1, L) -> (L,)
        source_labels = source_labels.squeeze()  # (1, L) -> (L,)
        for word, anm_label, source_label in zip(tokens, anm_labels, source_labels):
            #  관계자는, B-ANM, B-PERSON
            print(word, ANM_LABELS[anm_label], SOURCE_LABELS[source_label])


if __name__ == '__main__':
    main()


