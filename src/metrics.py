"""
Metricas de desempeno y de eficiencia del Proyecto 1.

Cubre los dos bloques que pide el enunciado:

  Desempeno : Accuracy, Precision, Recall, F1-score
  Eficiencia: numero de parametros, latencia de inferencia, memoria GPU
"""

# IMPORTANTE: torch antes que numpy (ver nota en train.py sobre fbgemm.dll).
import torch

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Desempeno
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, average="macro"):
    """
    Accuracy, Precision, Recall y F1.

    Se usa average='macro' por defecto: promedia la metrica de cada clase
    dandoles el mismo peso. Es lo correcto cuando las clases estan
    desbalanceadas (SST-2 tiene 56/44), porque 'micro' en clasificacion
    multiclase de una sola etiqueta colapsa a la accuracy y no aporta nada.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def per_class_report(y_true, y_pred, class_names):
    """Metricas desglosadas por clase, para la tabla del informe."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    return [
        {
            "clase": class_names[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "n": int(support[i]),
        }
        for i in range(len(class_names))
    ]


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()


# ---------------------------------------------------------------------------
# Eficiencia
# ---------------------------------------------------------------------------
def measure_latency(model, input_ids, attention_mask, device,
                    n_warmup=10, n_runs=50):
    """
    Latencia de inferencia en milisegundos por batch.

    Dos precauciones que cambian el resultado por completo:

    1. WARM-UP. Las primeras pasadas incluyen la carga de kernels CUDA y
       la asignacion de memoria. Medirlas infla el numero varias veces.
    2. torch.cuda.synchronize(). Las llamadas a CUDA son ASINCRONAS: sin
       sincronizar se mide cuanto tarda en encolarse la operacion, no en
       ejecutarse. Este es el error clasico al medir latencia en GPU.

    Devuelve media, desviacion y mediana. La mediana es mas robusta a
    picos puntuales del sistema operativo.
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids, attention_mask=attention_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        tiempos = []
        for _ in range(n_runs):
            inicio = time.perf_counter()
            model(input_ids, attention_mask=attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tiempos.append((time.perf_counter() - inicio) * 1000.0)

    tiempos = np.array(tiempos)
    return {
        "latencia_ms_media": float(tiempos.mean()),
        "latencia_ms_std": float(tiempos.std()),
        "latencia_ms_mediana": float(np.median(tiempos)),
        "batch_size": int(input_ids.size(0)),
        "seq_len": int(input_ids.size(1)),
        "n_runs": n_runs,
    }


def measure_gpu_memory(model, input_ids, attention_mask, device):
    """
    Pico de memoria GPU durante una pasada de inferencia, en MB.

    Se resetea el contador ANTES de medir; si no, se arrastra el pico de
    cualquier operacion previa (por ejemplo el entrenamiento) y el numero
    no significa nada.
    """
    if device.type != "cuda":
        return {"memoria_mb_pico": None, "nota": "sin GPU disponible"}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    with torch.no_grad():
        model(input_ids.to(device), attention_mask=attention_mask.to(device))
    torch.cuda.synchronize()

    return {
        "memoria_mb_pico": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "memoria_mb_reservada": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    }


def model_size_mb(model):
    """Tamano del modelo en MB, sumando parametros y buffers."""
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (params + buffers) / (1024 ** 2)
