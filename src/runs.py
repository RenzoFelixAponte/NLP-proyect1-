"""
Consulta de las corridas guardadas en results/metrics/.

Cada entrenamiento escribe un JSON con nombre unico:

    {modelo}_{tarea}[_{tag}]_{fecha-hora}.json

Nada se sobrescribe, asi que con el tiempo se acumulan varias corridas del
mismo modelo y tarea. Este modulo sirve para elegir cual usar: normalmente
la mas reciente, o la de una etiqueta concreta.

Uso:
    python -m src.runs                    # listar todas
    python -m src.runs --tag final        # solo las etiquetadas 'final'

    from src.runs import latest, load_all
    corrida = latest("bert", "sst2")
"""

import json
from pathlib import Path

METRICS_DIR = Path("results/metrics")


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--tarea", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    resumen(filtrar(load_all(), args.modelo, args.tarea, args.tag))
