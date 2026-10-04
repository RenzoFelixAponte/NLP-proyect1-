"""
Puente entre el checkpoint publicado en el HuggingFace Hub y nuestra
arquitectura propia (`domain/`).

Lo que hace esto NO es "usar el modelo de HuggingFace": es copiar, tensor
por tensor, los NUMEROS de sus pesos hacia los `nn.Parameter` de nuestros
propios `nn.Module`. Una vez copiados, `transformers` deja de estar en el
camino de ejecucion: el forward que corre es el nuestro (domain/*.py), y
lo unico que quedo prestado son los valores iniciales de los pesos -- que
es exactamente lo que hariamos si hubieramos re-entrenado su destilacion
desde cero y guardado el resultado.

Por eso, de aca en mas, entrenar/destilar "nuestro" DistilBert-custom es
tan legitimo como entrenar el `distilbert-base-uncased` original: la
arquitectura que corre el gradiente es la nuestra.
"""

from typing import Dict

import torch

from src.custom_distilbert.domain.config import DistilBertConfig
from src.custom_distilbert.domain.model import (
    DistilBertForSequenceClassification,
    DistilBertModel,
)

# HuggingFace usa un nombre por peso; nosotros usamos otro. Este diccionario
# es la unica pieza de codigo que conoce ambos vocabularios a la vez -- si
# HuggingFace cambiara sus nombres internos, solo hay que tocar aqui.
_STATIC_KEY_MAP = {
    "embeddings.word_embeddings.weight": "embeddings.token_embeddings.weight",
    "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
    "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
    "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
}

# Mapa de nombres DENTRO de cada bloque transformer (se repite por capa).
_PER_LAYER_KEY_MAP = {
    "attention.q_lin.weight": "attention.q_proj.weight",
    "attention.q_lin.bias": "attention.q_proj.bias",
    "attention.k_lin.weight": "attention.k_proj.weight",
    "attention.k_lin.bias": "attention.k_proj.bias",
    "attention.v_lin.weight": "attention.v_proj.weight",
    "attention.v_lin.bias": "attention.v_proj.bias",
    "attention.out_lin.weight": "attention.out_proj.weight",
    "attention.out_lin.bias": "attention.out_proj.bias",
    "sa_layer_norm.weight": "attn_norm.weight",
    "sa_layer_norm.bias": "attn_norm.bias",
    "ffn.lin1.weight": "ffn.fc1.weight",
    "ffn.lin1.bias": "ffn.fc1.bias",
    "ffn.lin2.weight": "ffn.fc2.weight",
    "ffn.lin2.bias": "ffn.fc2.bias",
    "output_layer_norm.weight": "output_norm.weight",
    "output_layer_norm.bias": "output_norm.bias",
}


def _build_full_key_map(n_layers: int) -> Dict[str, str]:
    """Genera el diccionario completo hf_key -> nuestra_key para N capas."""
    mapping = dict(_STATIC_KEY_MAP)
    for i in range(n_layers):
        for hf_suffix, own_suffix in _PER_LAYER_KEY_MAP.items():
            hf_key = f"transformer.layer.{i}.{hf_suffix}"
            own_key = f"encoder.layers.{i}.{own_suffix}"
            mapping[hf_key] = own_key
    return mapping


def _copy_encoder_weights(
    encoder_model: DistilBertModel, hf_state_dict: Dict[str, torch.Tensor]
) -> None:
    key_map = _build_full_key_map(encoder_model.config.n_layers)
    own_state = encoder_model.state_dict()

    copiados, faltantes = [], []
    for hf_key, own_key in key_map.items():
        if hf_key not in hf_state_dict:
            faltantes.append(hf_key)
            continue
        tensor = hf_state_dict[hf_key]
        if tensor.shape != own_state[own_key].shape:
            raise ValueError(
                f"Forma incompatible para {own_key}: "
                f"esperado {tuple(own_state[own_key].shape)}, "
                f"recibido {tuple(tensor.shape)} desde {hf_key}"
            )
        own_state[own_key] = tensor.clone()
        copiados.append(own_key)

    if faltantes:
        raise KeyError(
            f"Faltaron {len(faltantes)} pesos en el checkpoint de HuggingFace: "
            f"{faltantes[:5]}{'...' if len(faltantes) > 5 else ''}"
        )

    encoder_model.load_state_dict(own_state)
    print(f"[hf_weight_loader] copiados {len(copiados)} tensores del encoder "
          f"({sum(v.numel() for v in own_state.values()):,} parametros)")


def load_pretrained_encoder(
    config: DistilBertConfig, hf_id: str = "distilbert-base-uncased"
) -> DistilBertModel:
    """
    Construye un `DistilBertModel` (nuestro) y le carga dentro los pesos
    pre-entrenados publicados en `hf_id`. El objeto que devuelve NO tiene
    ninguna dependencia de `transformers` en su forward: la libreria solo
    se uso aca adentro, como si fuera un lector de un archivo de pesos.
    """
    from transformers import AutoModel

    print(f"[hf_weight_loader] descargando/leyendo pesos de '{hf_id}' "
          f"(cache local de HuggingFace)...")
    hf_model = AutoModel.from_pretrained(hf_id)
    hf_state_dict = hf_model.state_dict()

    own_model = DistilBertModel(config)
    _copy_encoder_weights(own_model, hf_state_dict)
    return own_model


def load_pretrained_into_custom_model(
    model: DistilBertForSequenceClassification,
    hf_id: str = "distilbert-base-uncased",
) -> DistilBertForSequenceClassification:
    """
    Version "in place": recibe un `DistilBertForSequenceClassification` ya
    construido y le inyecta los pesos del encoder pre-entrenado. La cabeza
    de clasificacion (`model.head`) se deja con su inicializacion aleatoria
    -- el checkpoint base de DistilBERT no trae cabeza de clasificacion,
    igual que le pasa a `AutoModelForSequenceClassification.from_pretrained`
    en `src/models.py`.
    """
    from transformers import AutoModel

    print(f"[hf_weight_loader] descargando/leyendo pesos de '{hf_id}' "
          f"(cache local de HuggingFace)...")
    hf_model = AutoModel.from_pretrained(hf_id)
    hf_state_dict = hf_model.state_dict()

    _copy_encoder_weights(model.distilbert, hf_state_dict)
    print("[hf_weight_loader] cabeza de clasificacion (pre_classifier/classifier) "
          "queda con inicializacion aleatoria: no existe en el checkpoint base.")
    return model
