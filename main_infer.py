"""
main_infer.py는 모델의 예측값을 정성적으로 살펴보기 위한 스크립트입니다.
e.g. https://github.com/wisdomify/wisdomify/blob/main/main_infer.py
"""
import argparse
import torch
import random
import wandb
import numpy as np
from os import path
from typing import List, Tuple
from transformers import BertTokenizer, AutoConfig, AutoModel
from BERT.loaders import load_config
from BERT.models import BiLabelNER
from BERT.tensors import InputsBuilder
from BERT.labels import SOURCE_LABELS, ANM_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mono_label_ner")
    parser.add_argument("--ver", type=str, default="overfit")
    parser.add_argument("--text", type=str, default="[CLS] 폴리티코는 소식통 3명을 인용해 바이든 당선인이 미 육군에서 흑인 최초 기록을 여러 차례 세운 오스틴을 국방장관에 지명할 예정이라고 전했다 [SEP]")
    args = parser.parse_args()
    config = load_config(args.model, args.ver)
    config.update(vars(args))  # command-line arguments 도 기록하기!
    # --- fix random seeds -- #
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    tokenizer = BertTokenizer.from_pretrained(config['bert'])
    bert = AutoModel.from_config(AutoConfig.from_pretrained(config['bert']))

    with wandb.init(project="BERT", config=config) as run:
        # download a pre-trained model from wandb
        artifact = run.use_artifact(f"{BiLabelNER.name}:{config['ver']}")
        model_path = artifact.checkout()
        model = BiLabelNER.load_from_checkpoint(path.join(model_path, "ner.ckpt"), bert=bert)
        tokens: List[str] = tokenizer.tokenize(config['text'])
        sentences: List[List[Tuple[str, str, str]]] = [[(token, "", "") for token in tokens]]
        inputs = InputsBuilder(tokenizer, sentences, config['max_length'])()
        model.freeze()
        # 원하는 결과
        anm_labels, source_labels = model.predict(inputs)  # (N
        input_ids = inputs[:, 0].squeeze()   # (N, 3, L) -> (1, L) -> (L)
        anm_labels = anm_labels.squeeze()  # (1, L) -> (L,)
        source_labels = source_labels.squeeze()  # (1, L) -> (L,)
        assert input_ids.shape[0] == anm_labels.shape[0] == source_labels.shape[0]
        for input_id, anm_label, source_label in zip(input_ids, anm_labels, source_labels):
            #  관계자는, B-ANM, B-PERSON
            print(tokenizer.decode(input_id), ANM_LABELS[anm_label], SOURCE_LABELS[source_label])


if __name__ == '__main__':
    main()


