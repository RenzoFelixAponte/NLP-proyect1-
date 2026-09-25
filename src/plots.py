"""
Figuras del informe, a partir de los JSON de results/metrics/.

    python -m src.plots                 # todas las figuras y tablas
    python -m src.plots --tag base      # solo las corridas etiquetadas 'base'

No hace falta GPU ni torch: solo lee los JSON que ya estan en el repo.

    pip install matplotlib numpy
    python -m src.plots

Genera en results/figures/, cada figura en PDF (vectorial, para el informe)
y en PNG (que es lo que GitHub renderiza dentro del README):

    curvas_<modelo>_<tarea>   iteraciones vs train loss / val loss, una corrida
    curvas_todas              las seis corridas base, un panel por dataset
    curvas_mejor_config       lo mismo con la configuracion ganadora del ablation
    burbujas_params_acc       parametros vs accuracy (area = latencia)
    ablation                  F1 de validacion de cada configuracion
    benchmark_eficiencia      latencia y memoria en la misma rejilla
    tabla_*                   tablas, en Markdown y como imagen (src/tables.py)

El estilo y la paleta estan en `src/style.py`; las tablas, en
`src/tables.py`; que corrida representa a cada celda, en `src/runs.py`.
Este modulo solo dibuja.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.paths import FIG_DIR
from src.runs import (corridas_finales, corridas_ablation, ultimo_benchmark,
                      medicion_benchmark)
from src.style import (
    AZUL, MORADO, TINTA, TINTA_TENUE, BORDE, LILA,
    COLOR_TAREA, MARCA_TAREA, COLOR_MODELO, MARCA_MODELO,
    ORDEN_CONFIG, ETIQUETA_CONFIG, CONFIG_ELEGIDA,
    aplicar_estilo, guardar, relleno, nombre_tarea, nombre_corto, orden_tarea,
)
import src.tables as tablas


def _borde(color):
    return BORDE.get(color, TINTA)


# ---------------------------------------------------------------------------
# 1. Curvas de entrenamiento: iteraciones vs loss
# ---------------------------------------------------------------------------
def _serie_loss(corrida):
    """(pasos, train_loss) y los puntos (paso, val_loss) donde hay medida."""
    historia = corrida.get("historia", [])
    pasos = [h["paso"] for h in historia]
    train = [h["train_loss"] for h in historia]
    # La validacion se evalua cada --eval-every pasos, no en cada registro:
    # se filtran los puntos donde realmente hay medida.
    val = [(h["paso"], h["val_loss"])
           for h in historia if h.get("val_loss") is not None]
    return pasos, train, val


def _dibujar_loss(ax, corrida, color):
    pasos, train, val = _serie_loss(corrida)
    ax.plot(pasos, train, color=color, linewidth=1.0, alpha=0.9)
    if val:
        ax.plot([p for p, _ in val], [v for _, v in val], color=_borde(color),
                linewidth=1.2, linestyle=(0, (4, 2)),
                marker=MARCA_MODELO.get(corrida["modelo"], "o"), markersize=4,
                markerfacecolor=relleno(color, 0.6),
                markeredgecolor=_borde(color), markeredgewidth=0.8)


def figura_curvas(corrida, destino):
    """Train loss y validation loss de una corrida frente a las iteraciones."""
    if not corrida.get("historia"):
        return None
    color = COLOR_MODELO.get(corrida["modelo"], AZUL)
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    _dibujar_loss(ax, corrida, color)
    ax.set_xlabel("Iteration (optimizer step)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{corrida['modelo_nombre']} on "
                 f"{nombre_tarea(corrida['tarea'])}", loc="left")
    ax.legend(handles=[
        Line2D([], [], color=color, linewidth=1.0, label="Training loss"),
        Line2D([], [], color=_borde(color), linestyle=(0, (4, 2)),
               marker=MARCA_MODELO.get(corrida["modelo"], "o"), markersize=4,
               markerfacecolor=relleno(color, 0.6), label="Validation loss"),
    ], loc="upper right")
    return guardar(fig, destino)


def figura_curvas_todas(corridas, destino):
    """
    Las seis corridas base en una fila: un panel por dataset, los dos modelos
    superpuestos. Asi se lee a la vez la forma de cada curva y la distancia
    entre modelos, que con seis figuras sueltas habria que ir a buscar.

    Cada panel tiene su propio eje Y: la loss de Yelp y la de SST-2 viven en
    rangos distintos y compartir escala aplastaria las dos.
    """
    tareas = sorted({t for (_, t) in corridas}, key=orden_tarea)
    if not tareas:
        return None
    fig, axes = plt.subplots(1, len(tareas), figsize=(7.0, 1.8))
    if len(tareas) == 1:
        axes = [axes]

    for ax, tarea in zip(axes, tareas):
        for modelo in ("bert", "distilbert"):
            c = corridas.get((modelo, tarea))
            if c and c.get("historia"):
                _dibujar_loss(ax, c, COLOR_MODELO[modelo])
        ax.set_title(nombre_tarea(tarea), loc="left")
        ax.set_xlabel("Iteration")
    axes[0].set_ylabel("Cross-entropy loss")

    manejadores = []
    for modelo, nombre in (("bert", "BERT"), ("distilbert", "DistilBERT")):
        color = COLOR_MODELO[modelo]
        manejadores += [
            Line2D([], [], color=color, linewidth=1.0,
                   label=f"{nombre} train"),
            Line2D([], [], color=_borde(color), linestyle=(0, (4, 2)),
                   marker=MARCA_MODELO[modelo], markersize=4,
                   markerfacecolor=relleno(color, 0.6),
                   markeredgecolor=_borde(color), label=f"{nombre} validation"),
        ]
    fig.legend(handles=manejadores, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.tight_layout(w_pad=1.2)
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 2. Burbujas: numero de parametros vs accuracy
# ---------------------------------------------------------------------------
def _latencia_controlada(corrida, bench):
    """
    Latencia (ms, batch 1) con la longitud de secuencia del dataset.

    Sale del benchmark controlado si existe; si no, del campo de la corrida,
    que es peor medida pero es lo unico que hay (y se avisa en la etiqueta).
    """
    punto = medicion_benchmark(bench, corrida["modelo"],
                               corrida.get("max_length")
                               or corrida["eficiencia"].get("seq_len"))
    if punto:
        return punto["latencia_ms_media"], True
    return corrida["eficiencia"]["latencia_ms_media"], False


def figura_burbujas(corridas, destino, bench=None):
    """
    Parametros (eje X) vs accuracy de test (eje Y); el area de la burbuja es
    la latencia de inferencia. Resume las tres metricas que importan en una
    sola figura: cuanto ocupa, cuanto acierta, cuanto tarda.

    Imita la figura 1 de la referencia: burbujas pastel con borde oscuro,
    etiqueta directa al lado y una leyenda de tamanos con circulos grises.
    """
    if not corridas:
        return None

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    escala = 55                       # puntos^2 por milisegundo

    xs = {m: c["eficiencia"]["n_parametros"] / 1e6
          for (m, _), c in corridas.items()}
    todas_controladas = True
    por_tarea = {}
    for (modelo, tarea), c in sorted(corridas.items()):
        params_m = c["eficiencia"]["n_parametros"] / 1e6
        acc = 100 * c["desempeno_test"]["accuracy"]
        lat, controlada = _latencia_controlada(c, bench)
        todas_controladas &= controlada
        por_tarea.setdefault(tarea, {})[modelo] = (params_m, acc)
        color = COLOR_MODELO.get(modelo, AZUL)
        # Area proporcional a la latencia (el area, no el radio: el ojo
        # compara areas, y escalar el radio exagera las diferencias).
        area = escala * lat
        ax.scatter(params_m, acc, s=area, color=relleno(color, 0.5),
                   edgecolor=_borde(color), linewidth=1.1, zorder=3,
                   marker=MARCA_MODELO.get(modelo, "o"))
        # La etiqueta sale HACIA AFUERA: DistilBERT (izquierda) la lleva a su
        # izquierda y BERT a su derecha. Hacia dentro, las dos columnas se
        # solaparian en el hueco del centro.
        radio = (area / 3.14159) ** 0.5
        izquierda = modelo == "distilbert"
        ax.annotate(f"{nombre_tarea(tarea)}\n{lat:.2f} ms",
                    (params_m, acc), textcoords="offset points",
                    xytext=(-(radio + 5) if izquierda else radio + 5, 0),
                    ha="right" if izquierda else "left", va="center",
                    fontsize=7, color=TINTA, linespacing=1.05)

    # Linea discontinua entre los dos modelos de cada dataset, con la brecha
    # de accuracy escrita encima: es la lectura que la figura quiere provocar.
    for tarea, pares in por_tarea.items():
        if len(pares) < 2:
            continue
        (x0, y0), (x1, y1) = pares["distilbert"], pares["bert"]
        ax.plot([x0, x1], [y0, y1], color=TINTA_TENUE, linewidth=0.7,
                linestyle=(0, (3, 2)), zorder=2)
        ax.annotate(f"{y1 - y0:+.2f} pt", ((x0 + x1) / 2, (y0 + y1) / 2),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=7, color=TINTA_TENUE, style="italic")

    # Cabecera de cada columna, como el "ChangeTitans (ours, 27M)" de la
    # referencia: nombre del modelo y tamano, en el color de sus burbujas.
    for (modelo, _), c in corridas.items():
        color = COLOR_MODELO.get(modelo, AZUL)
        ax.annotate(f"{nombre_corto(c['modelo_nombre'])} "
                    f"({xs[modelo]:.1f}M)",
                    (xs[modelo], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -9), textcoords="offset points", ha="center",
                    va="top", fontsize=7.5, fontweight="bold",
                    color=_borde(color))

    # Leyenda de tamanos, como los "Params=7M" de la referencia: circulos
    # grises dibujados en coordenadas de datos, abajo a la derecha, que es el
    # unico hueco que no cruza ninguna etiqueta ni la linea de SST-2. Una
    # leyenda de matplotlib no sirve: no reserva sitio para marcadores grandes
    # y el circulo de 5 ms se montaba sobre el titulo.
    for x, ms in ((127, 2.5), (140, 5.0)):
        ax.scatter(x, 89.75, s=escala * ms, facecolor="#eeeeee",
                   edgecolor=TINTA_TENUE, linewidth=0.8, zorder=3)
        ax.annotate(f"{ms:g} ms", (x, 89.75), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=6.5,
                    color=TINTA_TENUE)
    ax.text(133.5, 91.2, "Latency (batch 1)", ha="center", fontsize=7,
            color=TINTA)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(35, 150)
    ax.set_ylim(88.7, 98.8)
    if not todas_controladas:
        ax.set_title("Latency from the per-run measurement (no benchmark)",
                     loc="left", fontsize=7, color=TINTA_TENUE)
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 3. Benchmark controlado: latencia y memoria en la misma rejilla
# ---------------------------------------------------------------------------
def figura_benchmark(benchmark, destino):
    """
    Latencia de cada modelo en la misma rejilla (batch, longitud), y el pico
    de memoria en el ultimo panel.

    Un panel por batch size en vez de un solo grafico con dos escalas: las
    latencias de batch 1 y batch 32 se diferencian en un orden de magnitud y
    juntarlas en un eje aplastaria la curva de abajo. Cada panel tiene su
    propia escala de Y.
    """
    modelos = benchmark.get("modelos", [])
    if not modelos:
        return None

    batches = sorted({m["batch_size"] for m in modelos[0]["mediciones"]})
    # Las longitudes salen de la rejilla completa, no del ultimo modelo que
    # toco el bucle: si un modelo no tuviera mediciones para este batch, los
    # xticks se quedarian fijados con los de otro, o directamente sin definir.
    longitudes = sorted({m["seq_len"] for mod in modelos
                         for m in mod["mediciones"]})

    fig, axes = plt.subplots(1, len(batches) + 1,
                             figsize=(1.8 * (len(batches) + 1) + 0.2, 1.85))

    def puntos(mod, bs):
        return sorted([m for m in mod["mediciones"] if m["batch_size"] == bs],
                      key=lambda m: m["seq_len"])

    def linea(ax, mod, ps, campo, estilo="-", etiqueta=None):
        color = COLOR_MODELO.get(mod["modelo"], TINTA_TENUE)
        ax.plot([m["seq_len"] for m in ps], [m[campo] for m in ps],
                color=_borde(color), linestyle=estilo, linewidth=1.2,
                marker=MARCA_MODELO.get(mod["modelo"], "o"), markersize=4.5,
                markerfacecolor=relleno(color, 0.55),
                markeredgecolor=_borde(color), markeredgewidth=0.8,
                label=etiqueta)

    for ax, bs in zip(axes, batches):
        por_modelo = {}
        for mod in modelos:
            ps = puntos(mod, bs)
            if ps:
                linea(ax, mod, ps, "latencia_ms_media",
                      etiqueta=nombre_corto(mod["modelo_nombre"]))
                por_modelo[mod["modelo"]] = ps
        # Aceleracion media del panel, escrita dentro: evita ir a la tabla.
        if {"bert", "distilbert"} <= set(por_modelo):
            ratios = [b["latencia_ms_media"] / d["latencia_ms_media"]
                      for b, d in zip(por_modelo["bert"],
                                      por_modelo["distilbert"])]
            ax.text(0.04, 0.95, f"{sum(ratios) / len(ratios):.2f}$\\times$ "
                    "faster", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, style="italic", color=_borde(MORADO))
        ax.set_title(f"Latency, batch {bs}", loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(longitudes, [str(l) for l in longitudes])
        ax.minorticks_off()
        ax.set_xlabel("Sequence length")
        ax.margins(y=0.25)
    axes[0].set_ylabel("Milliseconds")

    # Memoria: batch 1 (dominado por los pesos) frente al batch mayor
    # (dominado por las activaciones). Es donde se ve que la ventaja de
    # memoria de DistilBERT se estrecha al crecer el lote.
    ax = axes[-1]
    b_min, b_max = batches[0], batches[-1]
    for mod in modelos:
        linea(ax, mod, puntos(mod, b_max), "memoria_mb_pico", "-")
        linea(ax, mod, puntos(mod, b_min), "memoria_mb_pico", (0, (3, 2)))
    ax.set_title("Peak GPU memory", loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(longitudes, [str(l) for l in longitudes])
    ax.minorticks_off()
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("MB")
    ax.legend(handles=[
        Line2D([], [], color=TINTA_TENUE, linestyle="-", label=f"batch {b_max}"),
        Line2D([], [], color=TINTA_TENUE, linestyle=(0, (3, 2)),
               label=f"batch {b_min}"),
    ], loc="upper left", fontsize=6.8)
    ax.margins(y=0.12)

    fig.legend(handles=axes[0].get_legend_handles_labels()[0],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.99),
               frameon=False)
    fig.tight_layout(w_pad=0.8)
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 4. Ablation study
# ---------------------------------------------------------------------------
def figura_ablation(ablaciones, destino, modelo="distilbert"):
    """
    F1 de validacion de cada configuracion: una fila por configuracion, un
    marcador por dataset.

    El eje empieza por debajo del encoder congelado y no se recorta alrededor
    de las cabezas: recortarlo haria parecer enormes unas diferencias de
    centesimas. Lo que se ve asi es lo que realmente pasa -- las cinco cabezas
    apiladas en la misma vertical y el encoder congelado descolgado -- que es
    la conclusion del estudio. La fila elegida lleva la franja lila con la que
    la referencia marca su configuracion por defecto.
    """
    # El ablation es solo de DistilBERT. Las corridas de la configuracion
    # ganadora con BERT usan la misma etiqueta 'abl-lineal' y se colarian
    # aqui: se filtra por modelo, o la figura mezclaria los dos.
    filas = {k: v for k, v in ablaciones.items() if k[0] == modelo}
    if not filas:
        return None

    tareas = sorted({t for (_, t, _) in filas}, key=orden_tarea)
    configs = [c for c in ORDEN_CONFIG if any(k[2] == c for k in filas)]
    fig, ax = plt.subplots(figsize=(3.2, 0.3 * len(configs) + 0.8))

    medias = {}
    for y, cfg in enumerate(configs):
        if cfg == CONFIG_ELEGIDA:
            ax.axhspan(y - 0.45, y + 0.45, color=LILA, zorder=0, linewidth=0)
        valores = []
        for tarea in tareas:
            c = filas.get((modelo, tarea, cfg))
            val = ((c or {}).get("desempeno_val") or {}).get("mejor_f1")
            if val is None:
                continue
            valores.append(100 * val)
            color = COLOR_TAREA.get(tarea, TINTA_TENUE)
            ax.scatter(100 * val, y, s=26, zorder=3,
                       marker=MARCA_TAREA.get(tarea, "o"),
                       color=relleno(color, 0.6), edgecolor=_borde(color),
                       linewidth=0.9)
        if valores:
            medias[y] = sum(valores) / len(valores)

    # La media de los tres datasets, que es con lo que se elige, en una
    # columna a la derecha del eje.
    for y, m in medias.items():
        ax.annotate(f"{m:.2f}", (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=TINTA,
                    fontweight="bold" if configs[y] == CONFIG_ELEGIDA
                    else "normal")
    ax.annotate("Mean", (1.0, 1.0), xycoords="axes fraction", xytext=(6, 2),
                textcoords="offset points", va="bottom", fontsize=7.5,
                fontweight="bold")

    ax.set_yticks(range(len(configs)),
                  [ETIQUETA_CONFIG.get(c, c) for c in configs])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(configs) - 0.5, -0.5)
    ax.set_xlim(84, 97.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Validation F1 (%)")
    ax.set_ylabel("Hidden layers of the head")
    ax.legend(handles=[
        Line2D([], [], linestyle="", marker=MARCA_TAREA.get(t, "o"),
               markersize=5.5, markerfacecolor=relleno(COLOR_TAREA[t], 0.6),
               markeredgecolor=_borde(COLOR_TAREA[t]), label=nombre_tarea(t))
        for t in tareas], loc="lower center", ncol=len(tareas),
        bbox_to_anchor=(0.5, 1.0), frameon=False)
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Genera las figuras y tablas del informe.")
    parser.add_argument("--tag", default=None,
                        help="Usar solo corridas con esta etiqueta")
    parser.add_argument("--out", default=str(FIG_DIR))
    args = parser.parse_args()

    aplicar_estilo()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corridas = corridas_finales(tag=args.tag)
    if not corridas:
        print("No hay corridas que graficar. Entrena algo primero.")
        return

    # El benchmark controlado es opcional: solo existe si ya se corrio
    # `python -m src.benchmark`. Sin el, las figuras caen a la latencia de
    # cada corrida y las tablas dejan esas columnas en blanco.
    bench = ultimo_benchmark()

    generadas = []
    for (modelo, tarea), c in sorted(corridas.items()):
        generadas.append(figura_curvas(c, out / f"curvas_{modelo}_{tarea}.pdf"))

    generadas += [
        figura_curvas_todas(corridas, out / "curvas_todas.pdf"),
        figura_burbujas(corridas, out / "burbujas_params_acc.pdf", bench),
        *tablas.tabla_resultados(corridas, out / "tabla_resultados", bench),
    ]

    ablaciones = corridas_ablation()
    if ablaciones:
        generadas += [
            figura_ablation(ablaciones, out / "ablation.pdf"),
            # Las curvas de la ganadora en los dos modelos: el enunciado pide
            # todas las metricas para esa comparacion, tambien las de loss.
            figura_curvas_todas(
                {(m, t): c for (m, t, cfg), c in ablaciones.items()
                 if cfg == CONFIG_ELEGIDA},
                out / "curvas_mejor_config.pdf"),
            *tablas.tabla_ablation(ablaciones, out / "tabla_ablation"),
            *tablas.tabla_mejor_config(ablaciones, out / "tabla_mejor_config",
                                       bench),
        ]

    if bench:
        generadas += [
            figura_benchmark(bench, out / "benchmark_eficiencia.pdf"),
            *tablas.tabla_benchmark(bench, out / "tabla_benchmark"),
        ]

    generadas = [g for g in generadas if g]
    print(f"{len(corridas)} corridas -> {len(generadas)} archivos en {out}/")
    for g in generadas:
        print(f"  {g.name}")


if __name__ == "__main__":
    main()
