"""Self-attention multi-cabeza: el corazon del Transformer."""

import math

import torch
import torch.nn as nn

from .config import BertConfig


class BertSelfAttention(nn.Module):
    """
    Atencion multi-cabeza bidireccional.

    "Bidireccional" significa simplemente que NO hay mascara causal: cada
    token puede atender a todos los demas, a izquierda y derecha. Es la
    diferencia central con GPT, y lo que el paper llama "deep bidirectional".
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) debe ser divisible por "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """(batch, seq, hidden) -> (batch, heads, seq, head_size)"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        # scores = Q K^T / sqrt(d_k)
        # El sqrt evita que los productos punto crezcan con la dimension y
        # saturen el softmax.
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)

        # La mascara llega ya en forma ADITIVA: 0 para tokens reales y un
        # numero muy negativo para el padding, de modo que el softmax les
        # asigne probabilidad ~0. (La conversion se hace en BertModel.)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)                       # (b, h, s, hd)
        context = context.permute(0, 2, 1, 3).contiguous()     # (b, s, h, hd)
        new_shape = context.size()[:-2] + (self.all_head_size,)
        return context.view(new_shape)                         # (b, s, hidden)


class BertSelfOutput(nn.Module):
    """Proyeccion de salida de la atencion + conexion residual + LayerNorm."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.dense(hidden_states))
        # Residual: se suma la ENTRADA del bloque, y el LayerNorm va DESPUES
        # (post-norm, como en el Transformer original).
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttention(nn.Module):
    """Agrupa la atencion y su proyeccion de salida."""

    def __init__(self, config: BertConfig):
        super().__init__()
        # El atributo se llama "self" porque asi lo nombra el checkpoint
        # oficial: bert.encoder.layer.N.attention.self.query.weight
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_out = self.self(hidden_states, attention_mask)
        return self.output(self_out, hidden_states)
