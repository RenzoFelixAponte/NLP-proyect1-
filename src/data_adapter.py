"""
Adaptador de datasets para el Proyecto 1 (BERT vs DistilBERT).

Objetivo: que los 3 datasets (SST-2, AG News, Yelp) se vean EXACTAMENTE IGUAL
para el resto del pipeline. Todos salen normalizados a:

    DatasetDict con splits "train" / "validation" / "test"
    y en cada uno las columnas:  "text" (str)  y  "label" (int)

Asi el codigo de fine-tuning no cambia: solo se cambia el nombre de la tarea.

    from src.data_adapter import load_task
    ds, info = load_task("sst2")
    ds, info = load_task("ag_news")
    ds, info = load_task("yelp", subsample={"train": 50_000, "test": 10_000})
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

from datasets import load_dataset, DatasetDict

SEED = 42
# Fraccion de "train" que se aparta como validacion cuando el dataset
# original no trae un split de validacion con etiquetas.
VAL_FRACTION = 0.1


@dataclass
class TaskInfo:
    """Metadatos de una tarea. Lo que el pipeline necesita saber del dataset."""
    name: str
    hf_id: str                      # identificador en el Hub de HuggingFace
    hf_config: Optional[str]        # sub-configuracion (ej. "sst2" dentro de "glue")
    text_col: str                   # nombre original de la columna de texto
    label_col: str                  # nombre original de la columna de etiqueta
    num_labels: int
    class_names: list
    max_length: int                 # longitud de tokenizacion recomendada
    notes: str = ""
    # Se rellenan despues de cargar:
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


# ---------------------------------------------------------------------------
# Registro de datasets. Agregar uno nuevo = agregar una entrada aqui.
# ---------------------------------------------------------------------------
REGISTRY = {
    "sst2": TaskInfo(
        name="sst2",
        hf_id="nyu-mll/glue",
        hf_config="sst2",
        text_col="sentence",
        label_col="label",
        num_labels=2,
        class_names=["negative", "positive"],
        # Las frases de SST-2 son muy cortas (mediana ~9 tokens). 64 sobra.
        max_length=64,
        notes=(
            "El split 'test' de GLUE viene SIN etiquetas (label = -1) porque se "
            "evalua en el servidor de GLUE. Por eso usamos el 'validation' "
            "oficial (872 ejemplos) como nuestro TEST, y apartamos un 10% del "
            "train como validacion."
        ),
    ),
    "ag_news": TaskInfo(
        name="ag_news",
        hf_id="fancyzhx/ag_news",
        hf_config=None,
        text_col="text",
        label_col="label",
        num_labels=4,
        class_names=["World", "Sports", "Business", "Sci/Tech"],
        max_length=128,
        notes="Trae train (120k) y test (7.6k) etiquetados. Validacion = 10% del train.",
    ),
    "yelp": TaskInfo(
        name="yelp",
        hf_id="fancyzhx/yelp_polarity",
        hf_config=None,
        text_col="text",
        label_col="label",
        num_labels=2,
        class_names=["negative", "positive"],
        # Resenas largas. 256 captura la mayoria sin disparar la memoria.
        max_length=256,
        notes=(
            "560k ejemplos de train. Entrenarlo completo con 2 modelos es "
            "inviable en una GPU de 6GB: usar subsample y documentarlo."
        ),
    ),
}


def list_tasks():
    """Devuelve los nombres de tarea disponibles."""
    return list(REGISTRY.keys())


def load_task(task_name: str,
              subsample: Optional[dict] = None,
              cache_dir: str = "data/raw",
              seed: int = SEED):
    """
    Carga una tarea y la normaliza a columnas "text" / "label".

    Args:
        task_name: una clave de REGISTRY ("sst2", "ag_news", "yelp").
        subsample: dict opcional {"train": n, "validation": n, "test": n} para
                   quedarse con n ejemplos de ese split (muestreo aleatorio
                   con semilla fija, para que sea reproducible).
        cache_dir: donde HuggingFace guarda los archivos descargados.
        seed:      semilla para el split de validacion y el subsampling.

    Returns:
        (DatasetDict, TaskInfo)
    """
    if task_name not in REGISTRY:
        raise ValueError(
            f"Tarea '{task_name}' desconocida. Disponibles: {list_tasks()}"
        )

    info = REGISTRY[task_name]
    raw = load_dataset(info.hf_id, info.hf_config, cache_dir=cache_dir)

    # --- 1. Decidir que split original hace de test -----------------------
    if task_name == "sst2":
        # El "test" de GLUE no tiene etiquetas -> el "validation" oficial
        # pasa a ser nuestro test.
        train_full = raw["train"]
        test_split = raw["validation"]
    else:
        train_full = raw["train"]
        test_split = raw["test"]

    # --- 2. Apartar una validacion del train ------------------------------
    split = train_full.train_test_split(test_size=VAL_FRACTION, seed=seed)
    ds = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": test_split,
    })

    # --- 3. Normalizar nombres de columna ---------------------------------
    def normalize(dataset):
        cols = dataset.column_names
        if info.text_col != "text":
            dataset = dataset.rename_column(info.text_col, "text")
        if info.label_col != "label":
            dataset = dataset.rename_column(info.label_col, "label")
        # Descartar todo lo demas (ej. la columna "idx" de GLUE)
        extra = [c for c in dataset.column_names if c not in ("text", "label")]
        if extra:
            dataset = dataset.remove_columns(extra)
        return dataset

    ds = DatasetDict({k: normalize(v) for k, v in ds.items()})

    # --- 4. Subsampling opcional ------------------------------------------
    if subsample:
        for split_name, n in subsample.items():
            if split_name in ds and n < len(ds[split_name]):
                ds[split_name] = (ds[split_name]
                                  .shuffle(seed=seed)
                                  .select(range(n)))

    # --- 5. Sanidad: ninguna etiqueta puede ser -1 ------------------------
    for split_name, d in ds.items():
        labels = set(d.unique("label"))
        if -1 in labels:
            raise ValueError(
                f"El split '{split_name}' de {task_name} tiene etiquetas -1 "
                f"(no etiquetado). Revisar la logica de splits."
            )

    info.n_train = len(ds["train"])
    info.n_val = len(ds["validation"])
    info.n_test = len(ds["test"])
    return ds, info


def describe(task_name: str, ds=None, info=None):
    """Imprime un resumen legible de la tarea (para el informe)."""
    if ds is None or info is None:
        ds, info = load_task(task_name)
    print(f"=== {info.name} ===")
    print(f"  fuente      : {info.hf_id}" + (f" / {info.hf_config}" if info.hf_config else ""))
    print(f"  clases      : {info.num_labels} -> {info.class_names}")
    print(f"  max_length  : {info.max_length}")
    print(f"  train       : {info.n_train:,}")
    print(f"  validation  : {info.n_val:,}")
    print(f"  test        : {info.n_test:,}")
    if info.notes:
        print(f"  nota        : {info.notes}")
    print(f"  ejemplo     : {ds['train'][0]['text'][:120]!r} -> "
          f"{info.class_names[ds['train'][0]['label']]}")
    return ds, info


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "sst2"
    describe(task)
