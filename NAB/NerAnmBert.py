import pytorch_lightning as pl

class NerAnmBert(pl.LightningModule):

    def __init__(self, ner_model, anm_model):
        super().__init__()
        self.ner_model = ner_model
        self.anm_model = anm_model

    def training_step(self, batch):
        b_input_ids, b_input_mask, b_labels, b_alabels = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['alabels']

        output_1 = self.ner_model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
        output_2 = self.anm_model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_alabels)
        loss_1 = output_1[0]
        loss_2 = output_2[0]

        logit_1 = output_1[1].detach().cpu().numpy()
        logit_2 = output_2[1].detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()
        alabel_ids = b_alabels.to('cpu').numpy()

        return loss_1 + loss_2, logit_1, logit_2, label_ids, alabel_ids