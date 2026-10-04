"""
Capa de embeddings de entrada.

A diferencia de BERT, DistilBERT NO tiene token_type_embeddings (los
embeddings de segmento A/B para pares de oraciones): se elimino en la
destilacion porque casi ninguna tarea de clasificacion de una sola
oracion los necesita. Por eso aqui solo hay dos tablas: token y posicion.
"""

import torch
import torch.nn as nn

from src.custom_distilbert.domain.config import DistilBertConfig


class DistilBertEmbeddings(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        # Posicionales APRENDIDOS (una fila por posicion 0..511), no
        # senoidales fijas como el Transformer original.
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        self.layer_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)  # (1, seq_len), se expande por broadcasting al batch

        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(
            position_ids
        )
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
