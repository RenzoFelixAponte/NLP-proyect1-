"""Carga de los pesos pre-entrenados oficiales y utilidades de conteo."""

import os
import glob

from .model import BertModel

MODEL_ID = "bert-base-uncased"


def find_checkpoint(model_id: str = MODEL_ID):
    """
    Localiza el `model.safetensors` en la cache local de HuggingFace.

    Los pesos NO viven en el proyecto: HuggingFace usa una cache global en
    ~/.cache/huggingface para que varios proyectos compartan los mismos
    archivos sin duplicar gigas.
    """
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pattern = os.path.join(
        home, "hub", f"models--{model_id}", "snapshots", "*", "model.safetensors"
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No se encontro el checkpoint de {model_id}.\n"
            f"  Buscado en: {pattern}\n"
            f"  Descargalo con: "
            f"python -c \"from transformers import AutoModel; "
            f"AutoModel.from_pretrained('{model_id}')\""
        )
    return matches[0]


def load_pretrained(model, checkpoint_path: str = None, verbose: bool = True):
    """
    Carga el checkpoint oficial `bert-base-uncased` en esta implementacion.

    Dos ajustes necesarios sobre los nombres del checkpoint:

    1. El checkpoint nombra los LayerNorm con la convencion antigua de
       TensorFlow: 'gamma' / 'beta'. PyTorch los llama 'weight' / 'bias'.
       Sin este remapeo la carga falla.
    2. Las cabezas de pre-entrenamiento ('cls.predictions' para MLM y
       'cls.seq_relationship' para NSP) no se usan en clasificacion: se
       descartan.

    La verificacion es estricta a proposito: si algun tensor esperado falta
    o sobra, se lanza un error en vez de dejar pesos sin inicializar en
    silencio (que daria un modelo que entrena pero rinde mal, sin aviso).
    """
    from safetensors.torch import load_file

    if checkpoint_path is None:
        checkpoint_path = find_checkpoint()

    raw = load_file(checkpoint_path)
    state_dict = {}
    skipped = []

    for key, tensor in raw.items():
        if key.startswith("cls."):          # cabezas de MLM / NSP
            skipped.append(key)
            continue
        new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
        new_key = new_key.replace("LayerNorm.beta", "LayerNorm.bias")
        state_dict[new_key] = tensor

    # Las claves del checkpoint traen el prefijo "bert.". El clasificador
    # lo espera (su atributo se llama self.bert); BertModel pelado no.
    if isinstance(model, BertModel):
        state_dict = {k[len("bert."):]: v for k, v in state_dict.items()
                      if k.startswith("bert.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # El classifier es nuevo y no esta en el checkpoint: eso es esperado.
    real_missing = [k for k in missing if not k.startswith("classifier.")]
    if real_missing or unexpected:
        raise RuntimeError(
            "Desajuste al cargar pesos.\n"
            f"  Faltantes inesperados: {real_missing}\n"
            f"  Sobrantes: {unexpected}"
        )

    if verbose:
        print(f"Pesos cargados desde {checkpoint_path}")
        print(f"  tensores cargados    : {len(state_dict)}")
        print(f"  descartados (MLM/NSP): {len(skipped)}")
        if missing:
            print(f"  inicializados al azar (esperado): {missing}")
    return model


def count_parameters(model, only_trainable: bool = False):
    """Numero total de parametros. Se usa en las metricas de eficiencia."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
