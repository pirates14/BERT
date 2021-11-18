import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import BertForTokenClassification, AdamW
from tqdm import tqdm, trange
from DataPreprocessing import df1, train_dataloader
from fi.NerAnmBert import NerAnmBert


def main():
    ner_num_label = df1['NER'].nunique()
    anm_num_label = df1['ANM'].nunique()

    ner_model = BertForTokenClassification.from_pretrained('beomi/kcbert-base',
                                                           num_labels=ner_num_label)
    anm_model = BertForTokenClassification.from_pretrained('beomi/kcbert-base',
                                                           num_labels=anm_num_label)

    nab = NerAnmBert(ner_model, anm_model)
    epochs = 30

    # 옵티마이저
    optimizer = AdamW(
        nab.parameters(),
        lr=3e-5,
        eps=1e-8
    )

    tag_values = list(set(df1["NER"].values))

    atag_values = list(set(df1["ANM"].values))

    for _ in trange(epochs, desc="Epoch"):

        nab.train()
        total_loss = 0
        predictions, true_labels = [], []
        a_predictions, a_true_labels = [], []

        for step, batch in enumerate(train_dataloader):
            nab.zero_grad()

            loss_sum, logit_1, logit_2, label_ids, alabel_ids = nab.training_step(batch)

            loss_sum.backward()

            total_loss += loss_sum.item()
            predictions.extend([list(p) for p in np.argmax(logit_1, axis=2)])
            a_predictions.extend([list(p) for p in np.argmax(logit_2, axis=2)])

            true_labels.extend(label_ids)
            a_true_labels.extend(alabel_ids)

            optimizer.step()

        print('\n total_loss:', total_loss/len(train_dataloader))

        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if l_i != 0]

        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if l_i != 0]

        a_pred_tags = [atag_values[p_i] for p, l in zip(a_predictions, a_true_labels)
                     for p_i, l_i in zip(p, l) if l_i != 0]

        a_valid_tags = [atag_values[l_i] for l in a_true_labels
                      for l_i in l if l_i != 0]

        # 매크로 F1 점수는 클래스별/레이블별 F1 점수의 평균으로 정의됩니다.
        ner_f1 = f1_score(pred_tags, valid_tags, average='macro')
        anm_f1 = f1_score(a_pred_tags, a_valid_tags, average='macro')

        # print("NER-F1-Score: {}".format(ner_f1))
        # print("ANM-F1-Score: {}".format(anm_f1))
        print("F1-Score: {}".format((ner_f1 + anm_f1) / 2))

        torch.save(nab, 'total_model.tar')

if __name__ == '__main__':
    main()