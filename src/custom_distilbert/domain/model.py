"""
Ensambla las piezas en los dos modelos que se usan desde afuera:

    DistilBertModel                    embeddings + encoder (sin cabeza)
    DistilBertForSequenceClassification  lo anterior + cabeza de clasificacion

Ambos son `nn.Module` de PyTorch puro: no heredan de nada de `transformers`.
El unico punto de contacto con HuggingFace es externo, en
`infrastructure/hf_weight_loader.py`, que copia los NUMEROS (pesos) desde
un checkpoint descargado hacia los `nn.Parameter` de estas clases.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from src.custom_distilbert.domain.classification_head import ClassificationHead
from src.custom_distilbert.domain.config import DistilBertConfig
from src.custom_distilbert.domain.embeddings import DistilBertEmbeddings
from src.custom_distilbert.domain.encoder import TransformerEncoder


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None


@dataclass
class ClassifierOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[List[torch.Tensor]] = None


class DistilBertModel(nn.Module):
    """Encoder puro: texto tokenizado -> secuencia de vectores contextuales."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = DistilBertEmbeddings(config)
        self.encoder = TransformerEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> EncoderOutput:
        x = self.embeddings(input_ids)
        last_hidden_state, hidden_states = self.encoder(
            x, attention_mask, output_hidden_states=output_hidden_states
        )
        return EncoderOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None,
        )


class DistilBertForSequenceClassification(nn.Module):
    """Encoder + cabeza. Es el modelo "estudiante" que se destila y/o se
    entrena para clasificacion de texto."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.config = config
        self.distilbert = DistilBertModel(config)
        self.head = ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> ClassifierOutput:
        encoder_out = self.distilbert(
            input_ids, attention_mask, output_hidden_states=output_hidden_states
        )
        cls_hidden_state = encoder_out.last_hidden_state[:, 0]  # token [CLS]
        logits = self.head(cls_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return ClassifierOutput(
            logits=logits,
            loss=loss,
            hidden_states=encoder_out.hidden_states,
        )
