"""Cabeza de clasificacion de secuencias: lo que se entrena en el fine-tuning."""

import torch.nn as nn

from .config import BertConfig
from .model import BertModel


class BertForSequenceClassification(nn.Module):
    """
    BERT + dropout + una capa lineal sobre el [CLS] pooleado.

    Seccion 3.2 del paper ("Fine-tuning BERT"): para clasificacion de una
    sola secuencia se usa el vector [CLS] como representacion agregada.

    El `classifier` se inicializa al azar: es la UNICA parte que no viene
    pre-entrenada, y es la que aprende la tarea concreta. Todo lo demas
    parte de los pesos de Google y se ajusta suavemente.
    """

    def __init__(self, config: BertConfig, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        _, pooled_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
        return loss, logits
