"""Configuracion de la arquitectura BERT."""

from dataclasses import dataclass


@dataclass
class BertConfig:
    """
    Hiperparametros de la arquitectura.

    Los valores por defecto son los de `bert-base-uncased`, es decir los
    del "BERT-BASE" del paper: L=12, H=768, A=12, 110M de parametros.
    """

    vocab_size: int = 30522          # tamano del vocabulario WordPiece
    hidden_size: int = 768           # d_model
    num_hidden_layers: int = 12      # capas del encoder (L)
    num_attention_heads: int = 12    # cabezas de atencion (768 / 12 = 64 c/u)
    intermediate_size: int = 3072    # tamano del FFN (4 * hidden_size)
    max_position_embeddings: int = 512
    type_vocab_size: int = 2         # segmentos A / B (para NSP)
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    @classmethod
    def base(cls):
        """BERT-BASE: 12 capas, 110M parametros."""
        return cls()

    @classmethod
    def large(cls):
        """BERT-LARGE: 24 capas, 340M parametros (referencia, no se usa aqui)."""
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )
