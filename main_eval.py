"""
main_eval.py 스크립트는 모델을 평가하기 위한 스크립트입니다.
e.g.: https://github.com/wisdomify/wisdomify/blob/main/main_eval.py
지표를 계산 (acc, f1_score)
"""
import pytorch_lightning as pl
import wandb


def main():
    targets_builder = ...

    # 참고:
    # just use.. datamodule & metric.
    # as for this, you would need more data.
    pass
    # print('\n total_loss:', total_loss / len(train_dataloader))
    #
    # pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
    #              for p_i, l_i in zip(p, l) if l_i != 0]
    #
    # valid_tags = [tag_values[l_i] for l in true_labels
    #               for l_i in l if l_i != 0]
    #
    # a_pred_tags = [atag_values[p_i] for p, l in zip(a_predictions, a_true_labels)
    #                for p_i, l_i in zip(p, l) if l_i != 0]
    #
    # a_valid_tags = [atag_values[l_i] for l in a_true_labels
    #                 for l_i in l if l_i != 0]
    #
    # # 매크로 F1 점수는 클래스별/레이블별 F1 점수의 평균으로 정의됩니다.
    # ner_f1 = f1_score(pred_tags, valid_tags, average='macro')
    # anm_f1 = f1_score(a_pred_tags, a_valid_tags, average='macro')
    #
    # # print("NER-F1-Score: {}".format(ner_f1))
    # # print("ANM-F1-Score: {}".format(anm_f1))
    # print("F1-Score: {}".format((ner_f1 + anm_f1) / 2))
    #
    # torch.save(nab, 'total_model.tar')
    config = ...
    with wandb.init(project="BERT") as run:
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             gpus=torch.cuda.device_count(),  # cpu 밖에 없으면 0, gpu가 n개이면 n
                             # callbacks=[early_stopping_callback],
                             enable_checkpointing=False,
                             logger=logger)
        # 학습을 진행한다
        trainer.test(model=..., datamodule=...)
        # test_step 실행.
        # TODO: 테스트 데이터에서 정확도 평가를 하는 것도 구현

if __name__ == '__main__':
    main()
