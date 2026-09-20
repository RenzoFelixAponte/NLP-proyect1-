"""Un bloque Transformer completo."""

import torch.nn as nn

from .config import BertConfig
from .attention import BertAttention
from .feedforward import BertIntermediate, BertOutput


class BertLayer(nn.Module):
    """
    Un bloque Transformer: atencion -> feed-forward.

    Ambos sub-bloques llevan conexion residual y LayerNorm. BERT-base
    apila 12 de estos, todos con la misma forma pero pesos distintos.

        x -> atencion -> (+x) -> LayerNorm -> FFN -> (+) -> LayerNorm
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attn = self.attention(hidden_states, attention_mask)
        inter = self.intermediate(attn)
        return self.output(inter, attn)
