
import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertForTokenClassification, AdamW

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from DataPreprocessing import df1, train_dataloader


def main():
    num_label = df1['NER'].nunique()
    print(num_label)

    model = BertForTokenClassification.from_pretrained(
        'beomi/kcbert-base',
        num_labels=num_label,
        # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForTokenClassification
        output_attentions=False,
        output_hidden_states=False
    )

    FULL_FINETUNING = True
    if FULL_FINETUNING:

        # model.named_parameters : 매개 변수 자체인 이름(name)과 파라미터(param)를 반환 여기서 반환되는 파라미터는 일종의 텐서로, torch.nn.Parameter 클래스입니다.
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    # 옵티마이저
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    epochs = 30
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    # 총 훈련 스탭 = 배치 * 에폭
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # 학습률 스케줄러 생성
    # schduler : 미리 정의된 schduler에 따라 학습률을 줄여 훈련 중 학습률을 조정하려고 합니다 (래요)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,  #
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    tag_values = list(set(df1["NER"].values))
    print(tag_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # 학습
    loss_values, validation_loss_values = [], []

    # 에폭만큼 진행률 프로세스바 영차
    for _ in trange(epochs, desc="Epoch"):  # tqdm(range(i)) = trange
        # ========================================
        #               Training
        #              학 습 시 작
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0
        predictions, true_labels = [], []

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            # Always clear any previously calculated gradients before performing a backward pass.
            # 이전에 계산한 기울기 지우기
            model.zero_grad()

            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            # loss 값 반환
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

            # get the loss
            loss = outputs[0]
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Perform a backward pass to calculate the gradients.
            # 기울기 계산을 위해 역전파
            loss.backward()

            # track train loss
            total_loss += loss.item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            # Clip the norm 의 실시 이유 : 기울기 폭주 방지
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

            # update parameters
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        # 평균 loss 계산
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        # 시각화 위해 loss 값 저장
        loss_values.append(avg_train_loss)

        # 예측값과 실제 값을 가져와 실제 인덱스가 pad가 아닐때 예측 인덱스에 해당하는 tag 추출
        # pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     # for p_i, l_i in zip(p, l) if l_i != 'PAD']
        # print(pred_tags)
        # valid_tags = [tag_values[l_i] for l in true_labels
                      # for l_i in l if l_i != 'PAD']
        # print(valid_tags)
        #
        # print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        # print("F1-Score: {}".format(f1_score(pred_tags, valid_tags, average='macro')))
    torch.save(model, 'total_model.tar')


if __name__ == '__main__':
    main()