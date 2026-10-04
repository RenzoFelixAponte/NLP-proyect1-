"""
Hiperparametros de la arquitectura.

Los valores por defecto son EXACTAMENTE los de `distilbert-base-uncased`
(ver su `config.json` en el Hub): asi el tensor de cada capa tiene la
forma correcta para recibir los pesos pre-entrenados sin transformar nada.
"""

from dataclasses import dataclass


@dataclass
class DistilBertConfig:
    vocab_size: int = 30522        # mismo vocabulario WordPiece que BERT
    dim: int = 768                 # dimension oculta
    n_layers: int = 6              # DistilBERT se queda con la mitad de BERT (12 -> 6)
    n_heads: int = 12
    hidden_dim: int = 3072         # dimension interna del FFN (4 * dim)
    max_position_embeddings: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    num_labels: int = 2            # cabeza de clasificacion: se define por tarea

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) debe ser divisible entre n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads
