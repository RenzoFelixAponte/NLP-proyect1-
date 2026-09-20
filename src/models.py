"""
Fabrica de modelos, con los pesos pre-entrenados de HuggingFace.

Todos los modelos se construyen con la MISMA implementacion (transformers)
a proposito: asi, cuando se comparen entre si, las diferencias de latencia
y memoria seran de la arquitectura y no de la calidad del codigo.
"""

# Este import parece sobrar (nada de aqui usa `torch` directamente) pero NO
# se puede quitar: fuerza el orden de carga. En este entorno (Anaconda +
# torch 2.5.1 en Windows), si numpy carga antes que torch, torch revienta
# con "Error loading fbgemm.dll". Ver la misma nota en train.py.
import torch  # noqa: F401

MODELOS = {
    "bert": {
        "hf_id": "bert-base-uncased",
        "nombre": "BERT-base",
        "capas": 12,
    },
}


def list_models():
    return list(MODELOS.keys())


def build_model(nombre_modelo: str, num_labels: int):
    """
    Construye un modelo de clasificacion con los pesos pre-entrenados.

    El encoder viene pre-entrenado; la cabeza de clasificacion se
    inicializa al azar y es lo que aprende la tarea. El aviso de
    "newly initialized" que imprime transformers es esperado y correcto.

    Returns:
        (model, info) donde info es la entrada de MODELOS correspondiente.
    """
    from transformers import AutoModelForSequenceClassification

    if nombre_modelo not in MODELOS:
        raise ValueError(
            f"Modelo '{nombre_modelo}' desconocido. Disponibles: {list_models()}"
        )

    info = MODELOS[nombre_modelo]
    model = AutoModelForSequenceClassification.from_pretrained(
        info["hf_id"], num_labels=num_labels
    )
    return model, info


def build_tokenizer(nombre_modelo: str):
    """
    Tokenizador correspondiente al modelo.

    Cada checkpoint trae SU vocabulario: usar el de otro modelo rompe la
    correspondencia token <-> embedding, y el modelo predice basura sin
    dar ningun error. Por eso el tokenizador se deriva siempre del modelo.
    """
    from transformers import AutoTokenizer

    if nombre_modelo not in MODELOS:
        raise ValueError(f"Modelo '{nombre_modelo}' desconocido.")
    return AutoTokenizer.from_pretrained(MODELOS[nombre_modelo]["hf_id"])


def count_parameters(model, only_trainable: bool = False):
    """Numero total de parametros. Es una de las metricas de eficiencia."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
