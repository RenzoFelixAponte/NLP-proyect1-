"""
Un bloque transformer = self-attention + FFN, cada uno con conexion
residual y post-layernorm (norma DESPUES de sumar el residuo), igual
que en BERT/DistilBERT originales (no es pre-norm como GPT-2).
"""

import torch
import torch.nn as nn

from src.custom_distilbert.domain.attention import MultiHeadSelfAttention
from src.custom_distilbert.domain.config import DistilBertConfig
from src.custom_distilbert.domain.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attn_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

        self.ffn = FeedForward(config)
        self.output_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_norm(attn_out + hidden_states)  # residual + norm

        ffn_out = self.ffn(hidden_states)
        hidden_states = self.output_norm(ffn_out + hidden_states)  # residual + norm

        return hidden_states
