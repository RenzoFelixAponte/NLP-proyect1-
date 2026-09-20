"""Capa de embeddings de BERT (Figura 2 del paper)."""

import torch
import torch.nn as nn

from .config import BertConfig


class BertEmbeddings(nn.Module):
    """
    Entrada de BERT = suma de TRES embeddings:

        token embeddings    : que palabra es (WordPiece)
        position embeddings : en que posicion esta. APRENDIDOS, no senoidales
                              como en el Transformer original de Vaswani.
        segment embeddings  : si el token pertenece a la frase A o a la B.
                              Existen por la tarea NSP del pre-entrenamiento.

    Las tres se SUMAN (no se concatenan), luego LayerNorm + dropout.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)

        # Posiciones 0, 1, 2, ..., seq_len-1
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Si no se indican segmentos, todo es frase A (caso de una sola frase,
        # que es justamente el de clasificacion de texto).
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = (self.word_embeddings(input_ids)
                      + self.position_embeddings(position_ids)
                      + self.token_type_embeddings(token_type_ids))

        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)
