"""
Self-attention multi-cabeza, implementada a mano (sin nn.MultiheadAttention)
para que cada proyeccion sea un `nn.Linear` explicito y mapeable 1 a 1
contra los pesos publicados de DistilBERT.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.custom_distilbert.domain.config import DistilBertConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)

        self.dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # (batch, seq, dim) -> (batch, n_heads, seq, head_dim)
        x = x.view(batch_size, -1, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)

        q = self._split_heads(self.q_proj(hidden_states), batch_size)
        k = self._split_heads(self.k_proj(hidden_states), batch_size)
        v = self._split_heads(self.v_proj(hidden_states), batch_size)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask: (batch, seq) con 1 = token real, 0 = padding.
            # Se expande a (batch, 1, 1, seq) para poder sumarse a `scores`
            # (batch, n_heads, seq, seq) por broadcasting, y se castea a
            # -inf donde hay padding para que el softmax lo anule.
            mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)  # (batch, n_heads, seq, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.dim)

        return self.out_proj(context)
