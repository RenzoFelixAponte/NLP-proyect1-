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
    # DistilBERT: destilado de bert-base-uncased. Conserva 6 de las 12 capas
    # (se queda con una de cada dos) y elimina el pooler y los embeddings de
    # segmento (token_type). Misma dimension oculta (768) y mismo vocabulario,
    # por lo que el tokenizador es intercambiable con el de BERT.
    "distilbert": {
        "hf_id": "distilbert-base-uncased",
        "nombre": "DistilBERT-base",
        "capas": 6,
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


# ---------------------------------------------------------------------------
# Cabeza de clasificacion configurable (para el ablation study)
# ---------------------------------------------------------------------------
# El ablation pide variar la cabeza (numero de capas y de neuronas) y congelar
# partes del transformer. La clase de HuggingFace no deja hacerlo: su cabeza
# es fija y ademas es DISTINTA en cada arquitectura (BERT pasa por el pooler
# con tanh; DistilBERT usa pre_classifier + ReLU). Comparar cabezas encima de
# dos preprocesos distintos mezclaria dos variables.
#
# Por eso el ablation usa esta envoltura: encoder desnudo (AutoModel) + una
# cabeza construida aqui. Es identica para BERT y para DistilBERT -- toma el
# estado oculto del token [CLS] -- asi que la unica diferencia entre dos
# corridas es lo que el experimento cambia a proposito.

import torch.nn as nn


class ClasificadorConCabeza(nn.Module):
    """
    Encoder pre-entrenado + MLP de clasificacion sobre el token [CLS].

    Args:
        encoder:     un AutoModel ya cargado (BERT o DistilBERT).
        hidden_size: dimension de salida del encoder (768 en ambos).
        num_labels:  numero de clases.
        head_hidden: tamanos de las capas ocultas de la cabeza. La lista vacia
                     da un clasificador lineal (768 -> num_labels); [768] imita
                     la cabeza por defecto de DistilBERT; [512, 256] son dos
                     capas ocultas.
        dropout:     probabilidad de dropout, aplicada antes de cada Linear.

    Devuelve un SequenceClassifierOutput, el mismo tipo que devuelven los
    modelos de HuggingFace, para que el bucle de entrenamiento no cambie.
    """

    def __init__(self, encoder, hidden_size, num_labels,
                 head_hidden=(), dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.head_hidden = list(head_hidden)

        capas = []
        dim = hidden_size
        for h in self.head_hidden:
            capas += [nn.Dropout(dropout), nn.Linear(dim, h), nn.ReLU()]
            dim = h
        capas += [nn.Dropout(dropout), nn.Linear(dim, num_labels)]
        self.head = nn.Sequential(*capas)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        from transformers.modeling_outputs import SequenceClassifierOutput

        salida = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Posicion 0 = token [CLS]. Es la representacion de la secuencia
        # completa: es el token con el que ambos modelos fueron pre-entrenados
        # para tareas de frase.
        cls = salida.last_hidden_state[:, 0]
        logits = self.head(cls)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _capas_transformer(encoder):
    """
    Lista de bloques transformer del encoder, sea BERT o DistilBERT.

    Cada arquitectura los guarda en un atributo distinto
    (bert.encoder.layer vs distilbert.transformer.layer), asi que se busca
    en los dos sitios en lugar de asumir uno.
    """
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer          # BERT
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return encoder.transformer.layer      # DistilBERT
    raise AttributeError(
        f"No se encontraron las capas transformer en {type(encoder).__name__}"
    )


def congelar(modelo, freeze_encoder=False, freeze_layers=0):
    """
    Congela parte del modelo: esos pesos no reciben gradiente y no se
    actualizan, solo se entrena lo que queda por encima.

    Args:
        freeze_encoder: congela TODO el encoder (embeddings + todas las capas).
                        El transformer queda como extractor de caracteristicas
                        fijo y solo aprende la cabeza.
        freeze_layers:  congela los embeddings y las primeras N capas. Las
                        capas bajas codifican rasgos mas genericos (lexico,
                        sintaxis local), asi que suelen ser las que menos
                        necesitan adaptarse a la tarea.

    Devuelve una descripcion de lo congelado, para guardarla en el JSON.
    """
    encoder = modelo.encoder if hasattr(modelo, "encoder") else modelo

    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        return {"freeze_encoder": True, "freeze_layers": None}

    if freeze_layers and freeze_layers > 0:
        # Los embeddings van siempre con el primer bloque congelado: dejarlos
        # entrenables mientras las capas de arriba estan fijas obligaria al
        # modelo a mover la entrada de una pila que ya no puede adaptarse.
        if hasattr(encoder, "embeddings"):
            for p in encoder.embeddings.parameters():
                p.requires_grad = False
        capas = _capas_transformer(encoder)
        for capa in capas[:freeze_layers]:
            for p in capa.parameters():
                p.requires_grad = False
        return {"freeze_encoder": False, "freeze_layers": int(freeze_layers)}

    return {"freeze_encoder": False, "freeze_layers": 0}


def build_ablation_model(nombre_modelo: str, num_labels: int,
                         head_hidden=(), head_dropout: float = 0.1,
                         freeze_encoder: bool = False, freeze_layers: int = 0):
    """
    Modelo del ablation study: encoder pre-entrenado + cabeza configurable.

    Returns:
        (model, info, detalle) donde `detalle` resume la configuracion de la
        cabeza y del congelado, para dejar constancia en los resultados.
    """
    from transformers import AutoModel

    if nombre_modelo not in MODELOS:
        raise ValueError(
            f"Modelo '{nombre_modelo}' desconocido. Disponibles: {list_models()}"
        )

    info = MODELOS[nombre_modelo]
    encoder = AutoModel.from_pretrained(info["hf_id"])
    modelo = ClasificadorConCabeza(
        encoder,
        hidden_size=encoder.config.hidden_size,
        num_labels=num_labels,
        head_hidden=head_hidden,
        dropout=head_dropout,
    )
    detalle = {"head_hidden": list(head_hidden), "head_dropout": head_dropout}
    detalle.update(congelar(modelo, freeze_encoder, freeze_layers))
    return modelo, info, detalle
