"""
Consulta de las corridas guardadas en results/metrics/.

Cada entrenamiento escribe un JSON con nombre unico:

    {modelo}_{tarea}[_{tag}]_{fecha-hora}.json

Nada se sobrescribe, asi que con el tiempo se acumulan varias corridas del
mismo modelo y tarea. Este modulo sirve para elegir cual usar: normalmente
la mas reciente, o la de una etiqueta concreta.

Aqui vive TODA la logica de "que corrida usar": cargarlas, filtrarlas, y
elegir la que representa a cada (modelo, tarea) en el informe. `src/plots.py`
y `src/tables.py` solo dibujan lo que este modulo les entrega -- antes cada
uno repetia su propio filtro y podian discrepar.

Uso:
    python -m src.runs                    # listar todas
    python -m src.runs --tag final        # solo las etiquetadas 'final'

    from src.runs import latest, load_all, corridas_finales
    corrida = latest("bert", "sst2")
"""

import json
from pathlib import Path

from src.paths import METRICS_DIR


def load_all(metrics_dir=METRICS_DIR):
    """Carga todas las corridas, de la mas reciente a la mas antigua."""
    directorio = Path(metrics_dir)
    if not directorio.exists():
        return []

    corridas = []
    for archivo in directorio.glob("*.json"):
        try:
            datos = json.loads(archivo.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  aviso: {archivo.name} no es un JSON valido, se omite")
            continue
        # En results/metrics/ tambien viven los JSON de src/benchmark.py, que
        # no son corridas de entrenamiento: no traen modelo, tarea ni metricas
        # de test, y apareceria una fila de '?' con nan en el resumen.
        if not (datos.get("modelo") and datos.get("tarea")
                and datos.get("desempeno_test")):
            continue
        datos["_archivo"] = archivo.name
        corridas.append(datos)

    # Se ordena por el timestamp guardado dentro del JSON, no por la fecha
    # del archivo: copiar o mover archivos cambia la fecha del sistema, pero
    # no cuando se hizo realmente el experimento.
    return sorted(corridas, key=lambda c: c.get("timestamp", ""), reverse=True)


def filtrar(corridas, modelo=None, tarea=None, tag=None):
    """Filtra una lista de corridas por modelo, tarea y/o etiqueta."""
    salida = corridas
    if modelo:
        salida = [c for c in salida if c.get("modelo") == modelo]
    if tarea:
        salida = [c for c in salida if c.get("tarea") == tarea]
    if tag is not None:
        salida = [c for c in salida if c.get("tag") == tag]
    return salida


def latest(modelo=None, tarea=None, tag=None, metrics_dir=METRICS_DIR):
    """
    Devuelve la corrida mas reciente que cumpla los filtros, o None.

    Es lo que deberian usar los scripts de graficos: asi una prueba rapida
    posterior no contamina el informe si se etiqueta como 'smoke' y se pide
    tag='final'.
    """
    encontradas = filtrar(load_all(metrics_dir), modelo, tarea, tag)
    return encontradas[0] if encontradas else None


def resumen(corridas):
    """Tabla compacta de corridas, para inspeccion rapida."""
    if not corridas:
        print("No hay corridas guardadas.")
        return

    print(f"{'fecha':<20} {'modelo':<12} {'tarea':<10} {'tag':<8} "
          f"{'n_train':>8} {'acc':>7} {'f1':>7}")
    print("-" * 78)
    for c in corridas:
        d = c.get("desempeno_test", {})
        print(f"{c.get('timestamp', '?'):<20} "
              f"{c.get('modelo', '?'):<12} "
              f"{c.get('tarea', '?'):<10} "
              f"{str(c.get('tag') or '-'):<8} "
              f"{c.get('n_train', 0):>8,} "
              f"{d.get('accuracy', float('nan')):>7.4f} "
              f"{d.get('f1', float('nan')):>7.4f}")


# ---------------------------------------------------------------------------
# Seleccion de corridas para el informe
# ---------------------------------------------------------------------------
# Corridas que NO representan un experimento del informe y que hay que apartar
# de las figuras salvo que se pidan a proposito:
#   smoke  pruebas rapidas para ver que el pipeline arranca
#   pilot  corridas exploratorias con el train submuestreado
TAGS_DESCARTADOS = ("smoke", "pilot")


def corridas_finales(tag=None, excluir_tags=TAGS_DESCARTADOS,
                     incluir_ablation=False):
    """
    Una corrida por (modelo, tarea): la mas reciente que cumpla los filtros.

    Se descartan dos cosas: las corridas que no son del informe (ver
    TAGS_DESCARTADOS) y, salvo que se pida lo contrario, las del ablation
    ('abl-...'). Sin este filtro las figuras de la comparacion BERT vs
    DistilBERT acabarian mostrando la ultima configuracion del ablation, que
    es mas reciente, en vez de la corrida base.
    """
    todas = [c for c in load_all() if c.get("tag") not in excluir_tags]
    if not incluir_ablation:
        todas = [c for c in todas
                 if not str(c.get("tag") or "").startswith("abl-")]
    if tag:
        todas = filtrar(todas, tag=tag)

    elegidas = {}
    for c in todas:                      # load_all() ordena de nuevo a viejo
        elegidas.setdefault((c.get("modelo"), c.get("tarea")), c)
    return elegidas


def corridas_ablation(metrics_dir=METRICS_DIR):
    """
    Corridas del ablation, indexadas por (modelo, tarea, config).

    La config sale de la etiqueta: 'abl-h768' -> 'h768'.
    """
    salida = {}
    for c in load_all(metrics_dir):
        tag = str(c.get("tag") or "")
        if not tag.startswith("abl-"):
            continue
        salida.setdefault((c["modelo"], c["tarea"], tag[4:]), c)
    return salida


def ultimo_benchmark(metrics_dir=METRICS_DIR):
    """
    El JSON de `src/benchmark.py` mas reciente, o None si no hay ninguno.

    No pasa por load_all(): los benchmarks no son corridas de entrenamiento
    y ese cargador los descarta a proposito.
    """
    archivos = sorted(Path(metrics_dir).glob("benchmark_*.json"))
    if not archivos:
        return None
    return json.loads(archivos[-1].read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--tarea", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    resumen(filtrar(load_all(), args.modelo, args.tarea, args.tag))
