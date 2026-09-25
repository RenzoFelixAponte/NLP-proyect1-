"""
Estilo comun de las figuras y nombres legibles del proyecto.

Vive aparte de `src/plots.py` para que las figuras y las tablas de
`src/tables.py` compartan exactamente los mismos nombres de dataset y de
configuracion. Cuando la paleta o una etiqueta estaban duplicadas en los dos
sitios, cambiar una y olvidar la otra producia un informe donde la leyenda de
la figura y el encabezado de la tabla decian cosas distintas del mismo dato.

Referencia visual: las figuras y tablas de ChangeTitans (IEEE TGRS 2025, en
docs/papers/). De ahi salen la tipografia con serifa (combina con el Times del
template de NeurIPS), los ejes en caja con ticks hacia dentro, las burbujas en
tono pastel con borde oscuro del mismo tono y la franja lila que marca la
configuracion elegida.

Criterio de diseno (las figuras van a un PDF, y ademas a PNG para el README):
  - Una sola escala por eje. Nunca dos ejes Y en la misma figura.
  - Cada serie lleva su etiqueta directa o una leyenda; el color nunca es el
    unico portador de identidad (ademas cambia la forma del marcador).
  - Rejilla tenue, discontinua y por detras de los datos.
  - Las etiquetas van en ingles porque las figuras entran al informe, que el
    enunciado pide en ingles.
"""

import matplotlib
matplotlib.use("Agg")           # sin pantalla: se ejecuta en un nodo de calculo
import matplotlib.pyplot as plt

# Paleta categorica validada con el script del skill de dataviz sobre fondo
# blanco: separacion CVD de 12,9 (deutan) en el peor par y 17,6 con vision
# normal. El azul y el ambar quedan por debajo de 3:1 de contraste, asi que
# todas las marcas llevan borde oscuro y etiqueta directa.
MORADO, AZUL, AMBAR = "#7b5cc4", "#4f9fcf", "#c9793a"

# Variante oscura de cada tono, para bordes y texto junto a la marca.
BORDE = {MORADO: "#43307f", AZUL: "#1f5f8b", AMBAR: "#7f4413"}

TINTA, TINTA_TENUE = "#1a1a19", "#5f5e5a"
REJILLA = "#d9d8e6"
# Relleno de la franja/fila destacada: el lila de las tablas de referencia.
LILA = "#e6e1f5"
CABECERA = "#dcdcef"

COLOR_TAREA = {"sst2": AZUL, "ag_news": MORADO, "yelp": AMBAR}
MARCA_TAREA = {"sst2": "o", "ag_news": "s", "yelp": "D"}
# DistilBERT hace el papel del "ours" de la referencia: morado.
COLOR_MODELO = {"bert": AZUL, "distilbert": MORADO}
MARCA_MODELO = {"bert": "o", "distilbert": "s"}

NOMBRE_TAREA = {"sst2": "SST-2", "ag_news": "AG News", "yelp": "Yelp"}

# Configuraciones del ablation, en el orden en que se leen en el informe:
# primero las variaciones de la cabeza, y el encoder congelado al final
# porque es el unico que cambia de eje.
ORDEN_CONFIG = ["lineal", "h128", "h768", "h2048", "c2", "frz"]
ETIQUETA_CONFIG = {
    "lineal": "linear",
    "h128": "1×128",
    "h768": "1×768",
    "h2048": "1×2048",
    "c2": "512+256",
    "frz": "1×768 frozen",
}
# La configuracion ganadora del ablation (se elige por F1 de validacion).
CONFIG_ELEGIDA = "lineal"


def aplicar_estilo():
    """Fija el estilo global de matplotlib. Se llama una vez, desde el main."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        # STIXGeneral viene dentro de matplotlib (no depende del sistema) y es
        # un clon de Times, la letra del template de NeurIPS.
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.labelcolor": TINTA,
        "axes.edgecolor": TINTA,
        "axes.linewidth": 0.7,
        # Caja completa con ticks hacia dentro, como en la referencia.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "axes.grid": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.5,
        "grid.linestyle": (0, (4, 3)),
        "axes.axisbelow": True,          # la rejilla, siempre por detras
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#c9c8d6",
        "legend.fancybox": False,
        "legend.fontsize": 7.5,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.4,
        "text.color": TINTA,
        # Tipo 42: el texto del PDF queda seleccionable y no se rasteriza.
        "pdf.fonttype": 42,
    })


def relleno(color, alpha=0.45):
    """Tono pastel de un color de la paleta (mezcla con blanco, sin alpha).

    Se mezcla a mano en lugar de usar alpha para que dos burbujas que se tocan
    no produzcan un tercer color en la interseccion.
    """
    c = color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(alpha * v + (1 - alpha) for v in rgb)


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
        fig.savefig(ruta, format=fmt, metadata=_SIN_FECHA.get(fmt),
                    facecolor="white")
        if principal is None:
            principal = ruta
    plt.close(fig)
    return principal
