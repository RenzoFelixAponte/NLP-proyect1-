"""
Cabeza de clasificacion, replicando la de `DistilBertForSequenceClassification`
de HuggingFace: pre_classifier (Linear + ReLU) -> dropout -> classifier (Linear).
Se aplica sobre el estado oculto del token [CLS] (posicion 0).
"""

import torch
import torch.nn as nn

from src.custom_distilbert.domain.config import DistilBertConfig


class ClassificationHead(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, cls_hidden_state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.pre_classifier(cls_hidden_state))
        x = self.dropout(x)
        return self.classifier(x)
