"""
BERT implementado desde cero en PyTorch puro.

Arquitectura del paper "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding" (Devlin et al., 2018), en
`paper/BERT_paper_1810.04805.pdf`.

Los nombres de los tensores coinciden EXACTAMENTE con los del checkpoint
oficial `bert-base-uncased`, de modo que se le pueden cargar los pesos
pre-entrenados reales de Google.

No se importa nada de `transformers` en el modelo. Esa libreria se usa
solo en `verify.py`, como oraculo para comprobar que esta implementacion
produce los mismos numeros.

Organizacion (una responsabilidad por archivo):

    config.py         BertConfig
    activations.py    gelu
    embeddings.py     BertEmbeddings
    attention.py      BertSelfAttention, BertSelfOutput, BertAttention
    feedforward.py    BertIntermediate, BertOutput
    layer.py          BertLayer
    encoder.py        BertEncoder, BertPooler
    model.py          BertModel
    classification.py BertForSequenceClassification
    weights.py        load_pretrained, find_checkpoint, count_parameters
    verify.py         verificacion contra HuggingFace

Uso:

    from bert import BertConfig, BertForSequenceClassification, load_pretrained

    model = BertForSequenceClassification(BertConfig.base(), num_labels=2)
    load_pretrained(model)
"""

from .config import BertConfig
from .activations import gelu
from .embeddings import BertEmbeddings
from .attention import BertSelfAttention, BertSelfOutput, BertAttention
from .feedforward import BertIntermediate, BertOutput
from .layer import BertLayer
from .encoder import BertEncoder, BertPooler
from .model import BertModel
from .classification import BertForSequenceClassification
from .weights import load_pretrained, find_checkpoint, count_parameters

__all__ = [
    "BertConfig",
    "gelu",
    "BertEmbeddings",
    "BertSelfAttention",
    "BertSelfOutput",
    "BertAttention",
    "BertIntermediate",
    "BertOutput",
    "BertLayer",
    "BertEncoder",
    "BertPooler",
    "BertModel",
    "BertForSequenceClassification",
    "load_pretrained",
    "find_checkpoint",
    "count_parameters",
]
