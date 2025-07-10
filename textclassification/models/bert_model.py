from transformers import BertForSequenceClassification, BertConfig
from transformers import BertModel, BertPreTrainedModel
import torch

class BertClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)  # 显式定义分类层
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits
        }