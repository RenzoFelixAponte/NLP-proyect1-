"""
Tablas del informe, en Markdown, a partir de los JSON de results/metrics/.

Se generan desde `python -m src.plots` junto con las figuras; este modulo
existe aparte porque construir texto y dibujar ejes no tienen nada en comun
salvo el origen de los datos, y juntarlos hacia de plots.py el archivo mas
largo del repo.

Todas las funciones siguen la misma forma: reciben las corridas ya
seleccionadas (de `src/runs.py`), escriben un .md y devuelven la ruta, o
None si no habia nada que escribir.
"""

from src.style import ETIQUETA_CONFIG, nombre_tarea, orden_tarea


def _fila(celdas):
    """Una fila de Markdown a partir de una lista de celdas."""
    return "| " + " | ".join(celdas) + " |"


def _cabecera(columnas):
    """Encabezado + separador de una tabla de Markdown."""
    return [_fila(columnas), "|" + "---|" * len(columnas)]


def _minutos(corrida):
    return f"{corrida.get('tiempo_entrenamiento_s', 0) / 60:.1f}"


def _f1_val(corrida):
    """F1 de validacion de la mejor epoca, o 'n/d' si la corrida no lo trae."""
    val = (corrida.get("desempeno_val") or {}).get("mejor_f1")
    return f"{val:.4f}" if val is not None else "n/d"



# ---------------------------------------------------------------------------
# Desempeno y eficiencia de las corridas base
# ---------------------------------------------------------------------------
def tabla_resultados(corridas, destino):
    """Desempeno en test y eficiencia, una fila por (modelo, dataset)."""
    if not corridas:
        return None

    lineas = _cabecera(["Modelo", "Dataset", "Acc", "P", "R", "F1",
                        "Params (M)", "Latencia (ms)", "Mem. GPU (MB)",
                        "Train (min)"])
    for (_, tarea), c in sorted(corridas.items()):
        d, e = c["desempeno_test"], c["eficiencia"]
        mem = e.get("memoria_mb_pico")
        lineas.append(_fila([
            c["modelo_nombre"], nombre_tarea(tarea),
            f"{d['accuracy']:.4f}", f"{d['precision']:.4f}",
            f"{d['recall']:.4f}", f"{d['f1']:.4f}",
            f"{e['n_parametros'] / 1e6:.1f}",
            f"{e['latencia_ms_media']:.2f}",
            f"{mem:.0f}" if mem else "n/d",
            _minutos(c),
        ]))
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------
def tabla_ablation(ablaciones, destino, modelo="distilbert"):
    """
    Tabla del ablation study.

    La columna que decide es **F1 de validacion**: es con la que se elige la
    mejor configuracion. El test aparece al lado solo para comprobar que la
    eleccion no cambia de signo, no para elegir con el.
    """
    filas = {k: v for k, v in ablaciones.items() if k[0] == modelo}
    if not filas:
        return None

    lineas = _cabecera(["Dataset", "Config", "Cabeza", "Congelado",
                        "Params entren.", "F1 val", "Acc test", "F1 test",
                        "Train (min)"])
    # El modelo ya esta filtrado arriba, asi que el desempaquetado lo ignora.
    for (_, tarea, cfg), c in sorted(filas.items(),
                                     key=lambda kv: (kv[0][1], kv[0][2])):
        abl = c.get("ablacion") or {}
        congelado = ("encoder" if abl.get("freeze_encoder")
                     else (f"{abl.get('freeze_layers')} capas"
                           if abl.get("freeze_layers") else "no"))
        entren = c["eficiencia"].get("n_parametros_entrenables")
        lineas.append(_fila([
            nombre_tarea(tarea), cfg,
            str(abl.get("head_hidden") or "lineal"),
            congelado,
            f"{entren / 1e6:.1f} M" if entren else "n/d",
            _f1_val(c),
            f"{c['desempeno_test']['accuracy']:.4f}",
            f"{c['desempeno_test']['f1']:.4f}",
            _minutos(c),
        ]))
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def tabla_mejor_config(ablaciones, destino, config="lineal"):
    """
    La configuracion ganadora del ablation, en los dos modelos.

    Es la tabla que pide el enunciado al final: una sola configuracion, con
    todas las metricas, comparando DistilBERT contra BERT. Devuelve None
    mientras falte alguno de los dos modelos: media tabla induce a error.
    """
    filas = {k: v for k, v in ablaciones.items() if k[2] == config}
    if len({k[0] for k in filas}) < 2:
        return None

    etiqueta = ETIQUETA_CONFIG.get(config, config)
    lineas = [f"Configuracion `{config}`: cabeza {etiqueta}, misma receta en "
              f"los dos modelos.", ""]
    lineas += _cabecera(["Dataset", "Modelo", "Acc", "P", "R", "F1", "F1 val",
                         "Params (M)", "Train (min)"])

    tareas = sorted({k[1] for k in filas}, key=orden_tarea)
    for tarea in tareas:
        for modelo in ("bert", "distilbert"):
            c = filas.get((modelo, tarea, config))
            if not c:
                continue
            d = c["desempeno_test"]
            lineas.append(_fila([
                nombre_tarea(tarea), c["modelo_nombre"],
                f"{d['accuracy']:.4f}", f"{d['precision']:.4f}",
                f"{d['recall']:.4f}", f"{d['f1']:.4f}",
                _f1_val(c),
                f"{c['eficiencia']['n_parametros'] / 1e6:.1f}",
                _minutos(c),
            ]))
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino



# ---------------------------------------------------------------------------
# Benchmark controlado
# ---------------------------------------------------------------------------
def tabla_benchmark(benchmark, destino):
    """Eficiencia medida en la misma rejilla (batch, longitud) para todos."""
    modelos = benchmark.get("modelos", [])
    if not modelos:
        return None

    lineas = [f"Medido en {benchmark.get('dispositivo', '?')} "
              f"({benchmark.get('precision', '?')}).", ""]
    lineas += _cabecera(["Modelo", "Params (M)", "Batch", "Longitud",
                         "Latencia (ms)", "Muestras/s", "Mem. GPU (MB)"])
    for mod in modelos:
        for m in mod["mediciones"]:
            lineas.append(_fila([
                mod["modelo_nombre"], f"{mod['n_parametros'] / 1e6:.1f}",
                str(m["batch_size"]), str(m["seq_len"]),
                f"{m['latencia_ms_media']:.2f}",
                f"{m['muestras_por_s']:.0f}",
                f"{m.get('memoria_mb_pico', 0):.0f}",
            ]))
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino
