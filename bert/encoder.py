"""Pila de capas Transformer y pooler del token [CLS]."""

import torch
import torch.nn as nn

from .config import BertConfig
from .layer import BertLayer


class BertEncoder(nn.Module):
    """Pila de N capas identicas en forma, distintas en pesos (12 en BERT-base)."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask=None, output_hidden_states=False):
        # `all_hidden` sirve para inspeccionar capa por capa. Es lo que usa
        # el script de verificacion para localizar en que bloque falla algo.
        all_hidden = [hidden_states] if output_hidden_states else None

        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden.append(hidden_states)

        return hidden_states, all_hidden


class BertPooler(nn.Module):
    """
    Toma el token [CLS] (posicion 0) y lo pasa por dense + tanh.

    Se entreno originalmente para la tarea NSP, por eso "resume" la
    secuencia entera en un solo vector. DistilBERT NO tiene esta capa,
    porque se entreno sin NSP.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        cls_token = hidden_states[:, 0]
        return torch.tanh(self.dense(cls_token))
