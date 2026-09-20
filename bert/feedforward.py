"""Bloque feed-forward posicional (FFN) de cada capa Transformer."""

import torch.nn as nn

from .config import BertConfig
from .activations import gelu


class BertIntermediate(nn.Module):
    """
    Expansion 768 -> 3072 con GELU.

    Se aplica de forma independiente a cada posicion de la secuencia
    (de ahi lo de "position-wise"): es la misma transformacion para todos
    los tokens, no mezcla informacion entre ellos. Ese mezclado ya lo hizo
    la atencion.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        return gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    """Contraccion 3072 -> 768 + conexion residual + LayerNorm."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        return self.LayerNorm(hidden_states + input_tensor)
