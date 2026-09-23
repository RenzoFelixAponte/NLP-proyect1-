"""
Figuras del informe, a partir de los JSON de results/metrics/.

    python -m src.plots                 # todas las figuras y tablas
    python -m src.plots --tag base      # solo las corridas etiquetadas 'base'

No hace falta GPU ni torch: solo lee los JSON que ya estan en el repo.

    pip install matplotlib numpy
    python -m src.plots

Genera en results/figures/, cada figura en PDF (vectorial, para el informe)
y en PNG (que es lo que GitHub renderiza dentro del README):

    curvas_<modelo>_<tarea>   iteraciones vs train loss / val loss
    burbujas_params_acc       parametros vs accuracy (area = latencia)
    latencia / memoria        comparacion de eficiencia por corrida
    ablation                  F1 de validacion de cada configuracion
    benchmark_latencia        latencia en la misma rejilla (batch, longitud)

El estilo y la paleta estan en `src/style.py`; las tablas en Markdown, en
`src/tables.py`; que corrida representa a cada celda, en `src/runs.py`.
Este modulo solo dibuja.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.paths import FIG_DIR
from src.runs import corridas_finales, corridas_ablation, ultimo_benchmark
from src.style import (
    AZUL, NARANJA, TINTA, TINTA_TENUE,
    COLOR_TAREA, COLOR_MODELO, MARCA_MODELO,
    ORDEN_CONFIG, ETIQUETA_CONFIG,
    aplicar_estilo, guardar, nombre_tarea, nombre_corto, orden_tarea,
)
import src.tables as tablas


# ---------------------------------------------------------------------------
# 1. Curvas de entrenamiento: iteraciones vs loss
# ---------------------------------------------------------------------------
def figura_curvas(corrida, destino):
    """Train loss y validation loss frente al numero de iteraciones."""
    historia = corrida.get("historia", [])
    if not historia:
        return None

    pasos = [h["paso"] for h in historia]
    train = [h["train_loss"] for h in historia]
    # La validacion se evalua cada --eval-every pasos, no en cada registro:
    # se filtran los puntos donde realmente hay medida.
    val = [(h["paso"], h["val_loss"])
           for h in historia if h.get("val_loss") is not None]

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(pasos, train, color=AZUL, linewidth=2, label="Training loss")
    if val:
        ax.plot([p for p, _ in val], [v for _, v in val], color=NARANJA,
                linewidth=2, marker="o", markersize=4, label="Validation loss")

    ax.set_xlabel("Iteración (paso de optimización)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{corrida['modelo_nombre']} · {nombre_tarea(corrida['tarea'])}",
                 loc="left")
    ax.legend(loc="upper right")
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 2. Burbujas: numero de parametros vs accuracy
# ---------------------------------------------------------------------------
def figura_burbujas(corridas, destino):
    """
    Parametros (eje X) vs accuracy (eje Y); el area de la burbuja es la
    latencia de inferencia. Resume las tres metricas de eficiencia que
    importan en una sola figura: cuanto ocupa, cuanto acierta, cuanto tarda.
    """
    if not corridas:
        return None

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    latencias = [c["eficiencia"]["latencia_ms_media"] for c in corridas.values()]
    lat_max = max(latencias) if latencias else 1.0

    xs = [c["eficiencia"]["n_parametros"] / 1e6 for c in corridas.values()]
    x_medio = (min(xs) + max(xs)) / 2 if xs else 0

    for (modelo, tarea), c in sorted(corridas.items()):
        params_m = c["eficiencia"]["n_parametros"] / 1e6
        acc = c["desempeno_test"]["accuracy"]
        lat = c["eficiencia"]["latencia_ms_media"]
        # Area proporcional a la latencia (el area, no el radio: el ojo
        # compara areas, y escalar el radio exagera las diferencias).
        area = 80 + 900 * (lat / lat_max)

        ax.scatter(params_m, acc, s=area,
                   color=COLOR_TAREA.get(tarea, TINTA_TENUE),
                   marker=MARCA_MODELO.get(modelo, "o"),
                   alpha=0.75, edgecolor="white", linewidth=1.5, zorder=3)

        # Etiqueta directa: el color no es el unico portador de identidad.
        # Va al COSTADO, no encima: los modelos de igual tamano caen en la
        # misma vertical y una etiqueta arriba se solapa con la burbuja de
        # al lado. El desplazamiento sale del radio de la burbuja (el area
        # esta en puntos^2) mas un margen fijo.
        radio = (area / 3.14159) ** 0.5
        # La etiqueta sale HACIA AFUERA: los puntos de la columna izquierda la
        # llevan a su izquierda y los de la derecha a su derecha. Al reves --
        # que es lo que se hacia antes -- las dos columnas empujan sus
        # etiquetas hacia el centro y se solapan entre si.
        izquierda = params_m < x_medio
        dx = -(radio + 6) if izquierda else (radio + 6)
        ax.annotate(f"{nombre_corto(c['modelo_nombre'])} · {nombre_tarea(tarea)}\n"
                    f"{lat:.1f} ms",
                    (params_m, acc), textcoords="offset points",
                    xytext=(dx, 0), va="center",
                    ha="right" if izquierda else "left",
                    fontsize=7, color=TINTA_TENUE)

    ax.set_xlabel("Parámetros (millones)")
    ax.set_ylabel("Accuracy en test")
    ax.set_title("Tamaño vs desempeño (área = latencia por muestra)", loc="left")

    # Los limites del eje X se fijan a mano en vez de con margins(): las
    # etiquetas se desplazan en PUNTOS, no en unidades de dato, asi que un
    # margen relativo no sabe cuanto texto tiene que acomodar y las de la
    # columna izquierda -- las mas largas, por el prefijo 'DistilBERT' -- se
    # salian del lienzo. El hueco de la izquierda es mayor por ese motivo.
    if xs:
        span = (max(xs) - min(xs)) or max(xs) or 1.0
        ax.set_xlim(min(xs) - 0.85 * span, max(xs) + 0.55 * span)
    ax.margins(y=0.22)
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 3. Barras de eficiencia
# ---------------------------------------------------------------------------
def figura_barras(corridas, campo, etiqueta, titulo, destino):
    """Barras horizontales de una metrica de eficiencia, por modelo y tarea."""
    filas = []
    for (_, tarea), c in sorted(corridas.items()):
        valor = c["eficiencia"].get(campo)
        if valor is not None:
            filas.append((f"{nombre_corto(c['modelo_nombre'])} · "
                          f"{nombre_tarea(tarea)}", valor, tarea))
    if not filas:
        return None

    fig, ax = plt.subplots(figsize=(4.6, 0.45 * len(filas) + 1.2))
    y = range(len(filas))
    ax.barh(list(y), [f[1] for f in filas],
            color=[COLOR_TAREA.get(f[2], TINTA_TENUE) for f in filas],
            height=0.6, zorder=3)
    ax.set_yticks(list(y), [f[0] for f in filas])
    ax.invert_yaxis()
    ax.set_xlabel(etiqueta)
    ax.set_title(titulo, loc="left")
    ax.grid(axis="y", visible=False)
    # Valor al final de cada barra: se lee el numero exacto sin volver al eje.
    for i, (_, valor, _) in enumerate(filas):
        ax.text(valor, i, f"  {valor:,.1f}", va="center", fontsize=8,
                color=TINTA_TENUE)
    ax.margins(x=0.18)
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 4. Benchmark controlado: latencia frente a longitud de secuencia
# ---------------------------------------------------------------------------
def figura_benchmark(benchmark, destino):
    """
    Latencia de cada modelo en la misma rejilla (batch, longitud).

    Un panel por batch size en vez de un solo grafico con dos escalas: las
    latencias de batch 1 y batch 32 se diferencian en un orden de magnitud y
    juntarlas en un eje aplastaria la curva de abajo. Cada panel comparte el
    eje X (longitud) y tiene su propia escala de Y, indicada en el titulo.
    """
    modelos = benchmark.get("modelos", [])
    if not modelos:
        return None

    batches = sorted({m["batch_size"] for m in modelos[0]["mediciones"]})
    fig, axes = plt.subplots(1, len(batches),
                             figsize=(2.3 * len(batches) + 1.2, 2.7))
    if len(batches) == 1:
        axes = [axes]

    # Las longitudes salen de la rejilla completa, no del ultimo modelo que
    # toco el bucle: si un modelo no tuviera mediciones para este batch, los
    # xticks se quedarian fijados con los de otro, o directamente sin definir.
    longitudes = sorted({m["seq_len"] for mod in modelos
                         for m in mod["mediciones"]})

    for ax, bs in zip(axes, batches):
        for mod in modelos:
            puntos = sorted([m for m in mod["mediciones"]
                             if m["batch_size"] == bs],
                            key=lambda m: m["seq_len"])
            if not puntos:
                continue
            ax.plot([m["seq_len"] for m in puntos],
                    [m["latencia_ms_media"] for m in puntos],
                    color=COLOR_MODELO.get(mod["modelo"], TINTA_TENUE),
                    marker="o", markersize=4, linewidth=2,
                    label=mod["modelo_nombre"])
        ax.set_title(f"batch = {bs}", loc="left")
        ax.set_xlabel("Longitud de secuencia")
        ax.set_xticks(longitudes)
    axes[0].set_ylabel("Latencia (ms)")
    axes[-1].legend(loc="upper left")

    fig.suptitle("Latencia de inferencia en la misma GPU", x=0.02, ha="left")
    fig.tight_layout()
    return guardar(fig, destino)


# ---------------------------------------------------------------------------
# 5. Ablation study
# ---------------------------------------------------------------------------
def figura_ablation(ablaciones, destino, modelo="distilbert"):
    """
    F1 de validacion de cada configuracion, por dataset.

    Se grafica con el eje completo desde 0,80 y no recortado alrededor de las
    diferencias: recortarlo haria parecer enormes unas diferencias de milesimas
    entre las cabezas. Lo que se ve asi es lo que realmente pasa -- todas las
    cabezas apiladas en el mismo punto y el encoder congelado descolgado --
    que es la conclusion del estudio.
    """
    # El ablation es solo de DistilBERT. Las corridas de la configuracion
    # ganadora con BERT usan la misma etiqueta 'abl-lineal' y se colarian
    # aqui: se filtra por modelo, o la figura mezclaria los dos.
    filas = {k: v for k, v in ablaciones.items() if k[0] == modelo}
    if not filas:
        return None

    tareas = sorted({t for (_, t, _) in filas}, key=orden_tarea)
    fig, ax = plt.subplots(
        figsize=(5.4, 0.42 * len(tareas) * len(ORDEN_CONFIG) / 2 + 1.4))

    # Se guarda la posicion Y de cada punto junto con su etiqueta: entre
    # bloques de dataset se deja un hueco, asi que las posiciones NO son
    # 0, 1, 2... y usar range() desalinearia las etiquetas de los puntos.
    y, etiquetas, posiciones = 0, [], []
    for tarea in tareas:
        for cfg in ORDEN_CONFIG:
            corrida = filas.get((modelo, tarea, cfg))
            if corrida is None:
                continue
            val = (corrida.get("desempeno_val") or {}).get("mejor_f1")
            if val is None:
                continue
            congelado = cfg == "frz"
            ax.scatter(val, y, s=70, zorder=3,
                       color=COLOR_TAREA.get(tarea, TINTA_TENUE),
                       marker="X" if congelado else "o",
                       edgecolor="white", linewidth=1.2)
            # El valor va a la derecha del punto, no encima: encima queda a
            # media altura entre dos filas y se lee como si fuera de la fila
            # de arriba.
            ax.text(val + 0.006, y, f"{val:.3f}", ha="left", va="center",
                    fontsize=6.5, color=TINTA_TENUE)
            etiquetas.append(f"{nombre_tarea(tarea)} · "
                             f"{ETIQUETA_CONFIG.get(cfg, cfg)}")
            posiciones.append(y)
            y += 1
        y += 0.6                       # separacion entre bloques de dataset

    if not posiciones:
        plt.close(fig)
        return None

    ax.set_yticks(posiciones)
    ax.set_yticklabels([])
    for pos, etq in zip(posiciones, etiquetas):
        ax.text(-0.012, pos, etq, ha="right", va="center", fontsize=7.5,
                transform=ax.get_yaxis_transform(), color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0.80, 1.0)
    ax.set_xlabel("F1 de validación")
    # Titulo neutro: la interpretacion va en el texto del informe, no dentro
    # de la figura, y un titulo largo se sale del ancho de columna.
    ax.set_title("Estudio de ablación · F1 de validación", loc="left")
    ax.grid(axis="y", visible=False)
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

    generadas = []
    for (modelo, tarea), c in sorted(corridas.items()):
        generadas.append(figura_curvas(c, out / f"curvas_{modelo}_{tarea}.pdf"))

    generadas += [
        figura_burbujas(corridas, out / "burbujas_params_acc.pdf"),
        figura_barras(corridas, "latencia_ms_media",
                      "Milisegundos por muestra (batch=1)",
                      "Latencia de inferencia", out / "latencia.pdf"),
        figura_barras(corridas, "memoria_mb_pico", "MB",
                      "Pico de memoria GPU en inferencia", out / "memoria.pdf"),
        tablas.tabla_resultados(corridas, out / "tabla_resultados.md"),
    ]

    ablaciones = corridas_ablation()
    if ablaciones:
        generadas += [
            tablas.tabla_ablation(ablaciones, out / "tabla_ablation.md"),
            figura_ablation(ablaciones, out / "ablation.pdf"),
            tablas.tabla_mejor_config(ablaciones, out / "tabla_mejor_config.md"),
        ]

    # El benchmark controlado es opcional: solo existe si ya se corrio
    # `python -m src.benchmark`.
    bench = ultimo_benchmark()
    if bench:
        generadas += [
            figura_benchmark(bench, out / "benchmark_latencia.pdf"),
            tablas.tabla_benchmark(bench, out / "tabla_benchmark.md"),
        ]

    generadas = [g for g in generadas if g]
    print(f"{len(corridas)} corridas -> {len(generadas)} archivos en {out}/")
    for g in generadas:
        print(f"  {g.name}")


if __name__ == "__main__":
    main()
