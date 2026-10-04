"""Pila de N bloques transformer (N=6 en DistilBERT, la mitad de BERT)."""

from typing import List

import torch
import torch.nn as nn

from src.custom_distilbert.domain.config import DistilBertConfig
from src.custom_distilbert.domain.transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """
        Devuelve el ultimo estado oculto y, si se pide, la lista de estados
        ocultos de TODAS las capas (incluida la entrada). Esa lista es la
        que necesita la destilacion con perdida coseno entre capas
        intermedias (ver application/distillation_loss.py).
        """
        all_hidden_states: List[torch.Tensor] = [hidden_states] if output_hidden_states else []

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states
