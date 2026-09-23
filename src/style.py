"""
Estilo comun de las figuras y nombres legibles del proyecto.

Vive aparte de `src/plots.py` para que las figuras y las tablas de
`src/tables.py` compartan exactamente los mismos nombres de dataset y de
configuracion. Cuando la paleta o una etiqueta estaban duplicadas en los dos
sitios, cambiar una y olvidar la otra producia un informe donde la leyenda de
la figura y el encabezado de la tabla decian cosas distintas del mismo dato.

Criterio de diseno (las figuras van a un PDF, y ademas a PNG para el README):
  - Una sola escala por eje. Nunca dos ejes Y en la misma figura.
  - Cada serie lleva su etiqueta directa o una leyenda; el color nunca es el
    unico portador de identidad.
  - Rejilla tenue y por detras de los datos; sin marcos superfluos.
"""

import matplotlib
matplotlib.use("Agg")           # sin pantalla: se ejecuta en un nodo de calculo
import matplotlib.pyplot as plt

# Paleta categorica validada para vision con deficiencia de color
# (separacion deutan/protan suficiente entre pares adyacentes).
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_TENUE = "#1a1a19", "#6b6a66"

COLOR_TAREA = {"sst2": AZUL, "ag_news": NARANJA, "yelp": AQUA}
COLOR_MODELO = {"bert": AZUL, "distilbert": NARANJA}
MARCA_MODELO = {"bert": "o", "distilbert": "s"}

NOMBRE_TAREA = {"sst2": "SST-2", "ag_news": "AG News", "yelp": "Yelp"}

# Configuraciones del ablation, en el orden en que se leen en el informe:
# primero las variaciones de la cabeza, y el encoder congelado al final
# porque es el unico que cambia de eje.
ORDEN_CONFIG = ["lineal", "h128", "h768", "h2048", "c2", "frz"]
ETIQUETA_CONFIG = {
    "lineal": "lineal (0 ocultas)",
    "h128": "128",
    "h768": "768",
    "h2048": "2048",
    "c2": "512+256 (2 ocultas)",
    "frz": "768, encoder congelado",
}


def aplicar_estilo():
    """Fija el estilo global de matplotlib. Se llama una vez, desde el main."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": TINTA_TENUE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e2de",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,          # la rejilla, siempre por detras
        "legend.frameon": False,
        "text.color": TINTA,
        "xtick.color": TINTA_TENUE,
        "ytick.color": TINTA_TENUE,
    })


def nombre_tarea(tarea):
    """Nombre legible de un dataset; si es desconocido, se devuelve tal cual."""
    return NOMBRE_TAREA.get(tarea, tarea)


def orden_tarea(tarea):
    """
    Posicion de un dataset en el orden del informe (el de NOMBRE_TAREA).

    Figuras y tablas lo comparten: si cada una ordenara por su cuenta, la
    tabla y la figura de al lado listarian los datasets al reves.
    """
    nombres = list(NOMBRE_TAREA)
    return nombres.index(tarea) if tarea in nombres else len(nombres)


def nombre_corto(modelo_nombre):
    """'BERT-base' -> 'BERT'. Para etiquetas dentro de una figura."""
    return modelo_nombre.split("-")[0]


# Sin esto, matplotlib estampa la fecha de generacion dentro de cada PDF y de
# cada PNG. El archivo cambia byte a byte aunque los datos sean identicos, asi
# que regenerar las figuras ensucia el `git status` con 11 archivos modificados
# y el diff no dice nada. Con la fecha fuera, la salida es reproducible: si un
# archivo aparece modificado, es porque los numeros cambiaron.
_SIN_FECHA = {"pdf": {"CreationDate": None}, "png": {"Software": None}}


def guardar(fig, destino, formatos=("pdf", "png")):
    """
    Guarda una figura en varios formatos y devuelve la ruta principal.

    Se generan los dos a proposito: el PDF es vectorial y es el que va al
    informe; el PNG es el que GitHub sabe renderizar dentro del README. Antes
    solo salia el PDF, asi que el README hablaba de figuras que nadie veia.

    `destino` lleva la extension del formato principal (la primera de la
    lista); las demas se derivan cambiandosela.
    """
    principal = None
    for fmt in formatos:
        ruta = destino.with_suffix(f".{fmt}")
        fig.savefig(ruta, format=fmt, metadata=_SIN_FECHA.get(fmt))
        if principal is None:
            principal = ruta
    plt.close(fig)
    return principal
