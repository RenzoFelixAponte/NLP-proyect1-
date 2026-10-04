"""
Implementacion propia de DistilBERT (arquitectura hexagonal).

Este paquete NO usa `transformers.DistilBertModel`: la arquitectura esta
escrita desde cero en `domain/`. Lo unico que se toma prestado de
HuggingFace son los PESOS numericos publicados (el checkpoint
`distilbert-base-uncased`), que se cargan dentro de nuestros propios
`nn.Module` via `infrastructure/hf_weight_loader.py`.

Estructura (hexagonal: el dominio no depende de nada externo):

    domain/            arquitectura pura (PyTorch puro, sin transformers)
        config.py          hiperparametros del modelo
        embeddings.py       embeddings de token + posicion
        attention.py         self-attention multi-cabeza
        feed_forward.py      MLP interno de cada bloque
        transformer_block.py  un bloque = attention + ffn + norms
        encoder.py          pila de bloques
        classification_head.py  cabeza de clasificacion
        model.py            ensambla todo: DistilBertModel / ForSequenceClassification

    infrastructure/    adaptadores hacia el mundo exterior
        hf_weight_loader.py   mapea pesos de HuggingFace -> nuestros modulos
        teacher.py           envoltorio del profesor (BERT) para destilar

    application/        casos de uso (logica de negocio, sin I/O de framework)
        distillation_loss.py   perdida triple de destilacion (Hinton/Sanh)
        distill_trainer.py     bucle de entrenamiento/destilacion

    cli/                puntos de entrada (argparse)
        distill.py

Uso rapido:

    from src.custom_distilbert.domain.config import DistilBertConfig
    from src.custom_distilbert.domain.model import DistilBertForSequenceClassification
    from src.custom_distilbert.infrastructure.hf_weight_loader import (
        load_pretrained_into_custom_model,
    )

    config = DistilBertConfig(num_labels=2)
    model = DistilBertForSequenceClassification(config)
    load_pretrained_into_custom_model(model, "distilbert-base-uncased")
"""
