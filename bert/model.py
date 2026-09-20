"""Modelo BERT base (sin cabeza de tarea)."""

import torch
import torch.nn as nn

from .config import BertConfig
from .embeddings import BertEmbeddings
from .encoder import BertEncoder, BertPooler


class BertModel(nn.Module):
    """
    BERT completo: embeddings -> encoder de 12 capas -> pooler.

    Devuelve:
        sequence_output : (batch, seq, hidden) un vector por token
        pooled_output   : (batch, hidden)      el [CLS] pasado por el pooler
        all_hidden      : lista de salidas intermedias, o None
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    @staticmethod
    def get_extended_attention_mask(attention_mask, dtype):
        """
        Convierte la mascara 0/1 de forma (batch, seq) en una mascara
        ADITIVA de forma (batch, 1, 1, seq):

            token real (1) -> 0.0
            padding    (0) -> el valor minimo del dtype (~ -inf)

        Se suma a los scores ANTES del softmax, asi el padding recibe
        probabilidad ~0. Las dos dimensiones intermedias son para que
        haga broadcast sobre (heads, query_positions).
        """
        extended = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                output_hidden_states=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        ext_mask = self.get_extended_attention_mask(attention_mask,
                                                    embedding_output.dtype)

        sequence_output, all_hidden = self.encoder(
            embedding_output, ext_mask, output_hidden_states
        )
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output, all_hidden
