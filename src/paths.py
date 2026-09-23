"""
Rutas del repositorio, ancladas a la raiz.

Antes cada modulo escribia `Path("results/metrics")` por su cuenta. Eso
obliga a ejecutar siempre desde la raiz del repo -- desde cualquier otro
directorio, los scripts no encuentran nada y fallan o, peor, crean un
`results/` vacio donde toque -- y ademas repite la misma ruta en varios
sitios, que es como empiezan a divergir.

Aqui se calcula una sola vez a partir de la ubicacion de ESTE archivo, asi
que funciona desde donde sea.
"""

from pathlib import Path

# src/paths.py -> src/ -> raiz del repo
RAIZ = Path(__file__).resolve().parent.parent

RESULTS_DIR = RAIZ / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

DATA_DIR = RAIZ / "data"
RAW_DIR = DATA_DIR / "raw"

DOCS_DIR = RAIZ / "docs"
