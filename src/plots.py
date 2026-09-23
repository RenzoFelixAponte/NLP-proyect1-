"""
Figuras y tablas del informe, a partir de los JSON de results/metrics/.

    python -m src.plots                 # todas las figuras
    python -m src.plots --tag base      # solo las corridas etiquetadas 'base'

Genera en results/figures/:
    curvas_<modelo>_<tarea>.pdf   iteraciones vs train loss / val loss
    burbujas_params_acc.pdf       parametros vs accuracy (tamano = latencia)
    latencia.pdf / memoria.pdf    comparacion de eficiencia
    tabla_resultados.md           tabla de desempeno lista para el informe

Criterio de diseno (las figuras van a un PDF en blanco y negro-friendly):
  - Una sola escala por eje. Nunca dos ejes Y en la misma figura.
  - Cada serie lleva su etiqueta directamente encima o una leyenda; el color
    nunca es el unico portador de identidad.
  - Rejilla tenue y por detras de los datos; sin marcos superfluos.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # sin pantalla: se ejecuta en un nodo de calculo
import matplotlib.pyplot as plt

from src.paths import FIG_DIR, METRICS_DIR
from src.runs import load_all, filtrar

# Paleta categorica validada para vision con deficiencia de color
# (separacion deutan/protan suficiente entre pares adyacentes).
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_TAREA = {"sst2": AZUL, "ag_news": NARANJA, "yelp": AQUA}
MARCA_MODELO = {"bert": "o", "distilbert": "s"}
NOMBRE_TAREA = {"sst2": "SST-2", "ag_news": "AG News", "yelp": "Yelp"}

TINTA, TINTA_TENUE = "#1a1a19", "#6b6a66"


def _estilo():
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


def corridas_finales(tag=None, excluir_tags=("smoke",), incluir_ablation=False):
    """
    Una corrida por (modelo, tarea): la mas reciente.

    Se descartan dos cosas: las pruebas rapidas ('smoke') y, salvo que se pida
    lo contrario, las corridas del ablation ('abl-...'). Sin este filtro las
    figuras de la comparacion BERT vs DistilBERT acabarian mostrando la ultima
    configuracion del ablation, que es mas reciente, en vez de la corrida base.
    """
    # Solo corridas de entrenamiento: en results/metrics/ tambien viven los
    # JSON del benchmark, que no tienen ni modelo ni tarea ni metricas de test.
    todas = [c for c in load_all()
             if c.get("modelo") and c.get("tarea") and c.get("desempeno_test")]
    todas = [c for c in todas if c.get("tag") not in excluir_tags]
    if not incluir_ablation:
        todas = [c for c in todas
                 if not str(c.get("tag") or "").startswith("abl-")]
    if tag:
        todas = filtrar(todas, tag=tag)

    elegidas = {}
    for c in todas:                      # load_all() ordena de nuevo a viejo
        clave = (c.get("modelo"), c.get("tarea"))
        elegidas.setdefault(clave, c)
    return elegidas


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
    val = [(h["paso"], h["val_loss"]) for h in historia if h.get("val_loss") is not None]

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(pasos, train, color=AZUL, linewidth=2, label="Training loss")
    if val:
        ax.plot([p for p, _ in val], [v for _, v in val], color=NARANJA,
                linewidth=2, marker="o", markersize=4, label="Validation loss")

    ax.set_xlabel("Iteracion (paso de optimizacion)")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{corrida['modelo_nombre']} · {NOMBRE_TAREA.get(corrida['tarea'], corrida['tarea'])}",
                 loc="left")
    ax.legend(loc="upper right")
    fig.savefig(destino)
    plt.close(fig)
    return destino


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
        izquierda = params_m > x_medio       # si esta a la derecha, etiqueta a la izquierda
        dx = -(radio + 6) if izquierda else (radio + 6)
        ax.annotate(f"{c['modelo_nombre'].split('-')[0]} · "
                    f"{NOMBRE_TAREA.get(tarea, tarea)}\n{lat:.1f} ms",
                    (params_m, acc), textcoords="offset points",
                    xytext=(dx, 0), va="center",
                    ha="right" if izquierda else "left",
                    fontsize=7, color=TINTA_TENUE)

    ax.set_xlabel("Parametros (millones)")
    ax.set_ylabel("Accuracy en test")
    ax.set_title("Tamano vs desempeno (area = latencia por muestra)", loc="left")
    ax.margins(x=0.32, y=0.22)
    fig.savefig(destino)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# 3. Barras de eficiencia
# ---------------------------------------------------------------------------
def figura_barras(corridas, campo, etiqueta, titulo, destino):
    """Barras horizontales de una metrica de eficiencia, por modelo y tarea."""
    filas = []
    for (modelo, tarea), c in sorted(corridas.items()):
        valor = c["eficiencia"].get(campo)
        if valor is not None:
            filas.append((f"{c['modelo_nombre'].split('-')[0]} · "
                          f"{NOMBRE_TAREA.get(tarea, tarea)}", valor, tarea))
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
    fig.savefig(destino)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# 4. Benchmark controlado: latencia frente a longitud de secuencia
# ---------------------------------------------------------------------------
COLOR_MODELO = {"bert": AZUL, "distilbert": NARANJA}


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
    fig, axes = plt.subplots(1, len(batches), figsize=(2.3 * len(batches) + 1.2, 2.7))
    if len(batches) == 1:
        axes = [axes]

    # Las longitudes salen de la rejilla completa, no del ultimo modelo que
    # toco el bucle: si un modelo no tuviera mediciones para este batch, los
    # xticks se quedarian fijados con los de otro, o directamente sin definir.
    longitudes = sorted({m["seq_len"] for mod in modelos
                         for m in mod["mediciones"]})

    for ax, bs in zip(axes, batches):
        for mod in modelos:
            puntos = sorted([m for m in mod["mediciones"] if m["batch_size"] == bs],
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
    fig.savefig(destino)
    plt.close(fig)
    return destino


def tabla_benchmark(benchmark, destino):
    """Tabla de eficiencia controlada, en Markdown."""
    modelos = benchmark.get("modelos", [])
    if not modelos:
        return None

    lineas = [f"Medido en {benchmark.get('dispositivo', '?')} "
              f"({benchmark.get('precision', '?')}).", "",
              "| Modelo | Params (M) | Batch | Longitud | Latencia (ms) | "
              "Muestras/s | Mem. GPU (MB) |",
              "|" + "---|" * 7]
    for mod in modelos:
        for m in mod["mediciones"]:
            lineas.append(
                f"| {mod['modelo_nombre']} | {mod['n_parametros']/1e6:.1f} "
                f"| {m['batch_size']} | {m['seq_len']} "
                f"| {m['latencia_ms_media']:.2f} | {m['muestras_por_s']:.0f} "
                f"| {m.get('memoria_mb_pico', 0):.0f} |")
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def ultimo_benchmark(metrics_dir=METRICS_DIR):
    """El JSON de benchmark mas reciente, o None si no hay ninguno."""
    import json
    archivos = sorted(Path(metrics_dir).glob("benchmark_*.json"))
    if not archivos:
        return None
    return json.loads(archivos[-1].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 5. Tabla de resultados
# ---------------------------------------------------------------------------
def tabla_markdown(corridas, destino):
    """Tabla de desempeno y eficiencia, en Markdown, para pasar al informe."""
    cab = ("| Modelo | Dataset | Acc | P | R | F1 | Params (M) | "
           "Latencia (ms) | Mem. GPU (MB) | Train (min) |")
    sep = "|" + "---|" * 10
    lineas = [cab, sep]
    for (modelo, tarea), c in sorted(corridas.items()):
        d, e = c["desempeno_test"], c["eficiencia"]
        mem = e.get("memoria_mb_pico")
        mem_txt = f"{mem:.0f}" if mem else "n/d"
        lineas.append(
            f"| {c['modelo_nombre']} | {NOMBRE_TAREA.get(tarea, tarea)} "
            f"| {d['accuracy']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} "
            f"| {d['f1']:.4f} | {e['n_parametros']/1e6:.1f} "
            f"| {e['latencia_ms_media']:.2f} | {mem_txt} "
            f"| {c.get('tiempo_entrenamiento_s', 0) / 60:.1f} |")
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def corridas_ablation():
    """Todas las corridas del ablation, agrupadas por (modelo, tarea, config)."""
    salida = {}
    for c in load_all():
        tag = str(c.get("tag") or "")
        if not tag.startswith("abl-") or not c.get("desempeno_test"):
            continue
        salida.setdefault((c["modelo"], c["tarea"], tag[4:]), c)
    return salida


ORDEN_CONFIG = ["lineal", "h128", "h768", "h2048", "c2", "frz"]
ETIQUETA_CONFIG = {
    "lineal": "lineal (0 ocultas)",
    "h128": "128",
    "h768": "768",
    "h2048": "2048",
    "c2": "512+256 (2 ocultas)",
    "frz": "768, encoder congelado",
}


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
    ablaciones = {k: v for k, v in ablaciones.items() if k[0] == modelo}
    if not ablaciones:
        return None

    tareas = sorted({t for (_, t, _) in ablaciones}, key=lambda t: list(NOMBRE_TAREA).index(t)
                    if t in NOMBRE_TAREA else 99)
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(tareas) * len(ORDEN_CONFIG) / 2 + 1.4))

    # Se guarda la posicion Y de cada punto junto con su etiqueta: entre
    # bloques de dataset se deja un hueco, asi que las posiciones NO son
    # 0, 1, 2... y usar range() desalinearia las etiquetas de los puntos.
    y, etiquetas, posiciones = 0, [], []
    for tarea in tareas:
        for cfg in ORDEN_CONFIG:
            clave = (modelo, tarea, cfg)
            if clave not in ablaciones:
                continue
            val = (ablaciones[clave].get("desempeno_val") or {}).get("mejor_f1")
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
            etiquetas.append(f"{NOMBRE_TAREA.get(tarea, tarea)} · "
                             f"{ETIQUETA_CONFIG.get(cfg, cfg)}")
            posiciones.append(y)
            y += 1
        y += 0.6                       # separacion entre bloques de dataset

    ax.set_yticks(posiciones)
    ax.set_yticklabels([])
    for pos, etq in zip(posiciones, etiquetas):
        ax.text(-0.012, pos, etq, ha="right", va="center", fontsize=7.5,
                transform=ax.get_yaxis_transform(), color=TINTA)
    ax.invert_yaxis()
    ax.set_xlim(0.80, 1.0)
    ax.set_xlabel("F1 de validacion")
    # Titulo neutro: la interpretacion va en el texto del informe, no dentro
    # de la figura, y un titulo largo se sale del ancho de columna.
    ax.set_title("Ablation study · F1 de validacion", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(destino)
    plt.close(fig)
    return destino


def tabla_ablation(ablaciones, destino, modelo="distilbert"):
    """
    Tabla del ablation study.

    La columna que decide es **F1 de validacion**: es con la que se elige la
    mejor configuracion. El test aparece al lado solo para comprobar que la
    eleccion no cambia de signo, no para elegir con el.
    """
    ablaciones = {k: v for k, v in ablaciones.items() if k[0] == modelo}
    if not ablaciones:
        return None

    lineas = ["| Dataset | Config | Cabeza | Congelado | Params entren. | "
              "F1 val | Acc test | F1 test | Train (min) |",
              "|" + "---|" * 9]
    # El desempaquetado usa `_` para el modelo: ya se filtro por el argumento
    # `modelo` arriba, y nombrarlo igual pisaba el parametro de la funcion.
    for (_, tarea, cfg), c in sorted(ablaciones.items(),
                                     key=lambda kv: (kv[0][1], kv[0][2])):
        abl = c.get("ablacion") or {}
        val = (c.get("desempeno_val") or {}).get("mejor_f1")
        cabeza = abl.get("head_hidden") or "lineal"
        congelado = ("encoder" if abl.get("freeze_encoder")
                     else (f"{abl.get('freeze_layers')} capas"
                           if abl.get("freeze_layers") else "no"))
        entren = c["eficiencia"].get("n_parametros_entrenables")
        entren_txt = f"{entren / 1e6:.1f} M" if entren else "n/d"
        val_txt = f"{val:.4f}" if val is not None else "n/d"
        lineas.append(
            f"| {NOMBRE_TAREA.get(tarea, tarea)} | {cfg} | {cabeza} "
            f"| {congelado} | {entren_txt} | {val_txt} "
            f"| {c['desempeno_test']['accuracy']:.4f} "
            f"| {c['desempeno_test']['f1']:.4f} "
            f"| {c.get('tiempo_entrenamiento_s', 0) / 60:.1f} |")
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def tabla_mejor_config(ablaciones, destino, config="lineal"):
    """
    La configuracion ganadora del ablation, en los dos modelos.

    Es la tabla que pide el enunciado al final: una sola configuracion, con
    todas las metricas, comparando DistilBERT contra BERT.
    """
    filas = {k: v for k, v in ablaciones.items() if k[2] == config}
    modelos = {k[0] for k in filas}
    if len(modelos) < 2:
        return None          # todavia no esta corrida con los dos modelos

    lineas = [f"Configuracion `{config}`, misma receta en los dos modelos.", "",
              "| Dataset | Modelo | Acc | P | R | F1 | F1 val | Params (M) | "
              "Train (min) |",
              "|" + "---|" * 9]
    for tarea in [t for t in NOMBRE_TAREA if any(k[1] == t for k in filas)]:
        for modelo in ("bert", "distilbert"):
            c = filas.get((modelo, tarea, config))
            if not c:
                continue
            d = c["desempeno_test"]
            val = (c.get("desempeno_val") or {}).get("mejor_f1")
            val_txt = f"{val:.4f}" if val is not None else "n/d"
            lineas.append(
                f"| {NOMBRE_TAREA[tarea]} | {c['modelo_nombre']} "
                f"| {d['accuracy']:.4f} | {d['precision']:.4f} "
                f"| {d['recall']:.4f} | {d['f1']:.4f} | {val_txt} "
                f"| {c['eficiencia']['n_parametros'] / 1e6:.1f} "
                f"| {c.get('tiempo_entrenamiento_s', 0) / 60:.1f} |")
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None,
                        help="Usar solo corridas con esta etiqueta")
    parser.add_argument("--out", default=str(FIG_DIR))
    args = parser.parse_args()

    _estilo()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corridas = corridas_finales(tag=args.tag)
    if not corridas:
        print("No hay corridas que graficar. Entrena algo primero.")
        return

    generadas = []
    for (modelo, tarea), c in sorted(corridas.items()):
        g = figura_curvas(c, out / f"curvas_{modelo}_{tarea}.pdf")
        if g:
            generadas.append(g)

    generadas += [g for g in [
        figura_burbujas(corridas, out / "burbujas_params_acc.pdf"),
        figura_barras(corridas, "latencia_ms_media", "Milisegundos por muestra (batch=1)",
                      "Latencia de inferencia", out / "latencia.pdf"),
        figura_barras(corridas, "memoria_mb_pico", "MB",
                      "Pico de memoria GPU en inferencia", out / "memoria.pdf"),
        tabla_markdown(corridas, out / "tabla_resultados.md"),
    ] if g]

    # El benchmark controlado es opcional: solo existe si ya se corrio
    # `python -m src.benchmark`.
    ablaciones = corridas_ablation()
    if ablaciones:
        generadas += [g for g in [
            tabla_ablation(ablaciones, out / "tabla_ablation.md"),
            figura_ablation(ablaciones, out / "ablation.pdf"),
            tabla_mejor_config(ablaciones, out / "tabla_mejor_config.md"),
        ] if g]

    bench = ultimo_benchmark()
    if bench:
        generadas += [g for g in [
            figura_benchmark(bench, out / "benchmark_latencia.pdf"),
            tabla_benchmark(bench, out / "tabla_benchmark.md"),
        ] if g]

    print(f"{len(corridas)} corridas -> {len(generadas)} archivos en {out}/")
    for g in generadas:
        print(f"  {g}")


if __name__ == "__main__":
    main()
