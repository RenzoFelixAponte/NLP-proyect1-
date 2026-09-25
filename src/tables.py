"""
Tablas del informe, a partir de los JSON de results/metrics/.

Se generan desde `python -m src.plots` junto con las figuras. Cada tabla sale
en dos formatos a partir de la MISMA especificacion:

    tabla_<nombre>.md    Markdown, para diffs y para copiar numeros
    tabla_<nombre>.png   imagen con el estilo de las tablas de la referencia
                         (ChangeTitans, IEEE TGRS 2025): cabecera sombreada,
                         reglas horizontales, grupos de columnas con cmidrule,
                         el mejor valor en negrita y la fila elegida en lila
    tabla_<nombre>.tex   el mismo tabular en LaTeX, que el informe incluye
                         con \\input: ninguna cifra del informe se copia a mano

GitHub no deja dar estilo a una tabla de Markdown, asi que el README muestra
la imagen y enlaza el .md. Construir las dos desde una sola especificacion es
lo que evita que la imagen, el texto y el informe digan cosas distintas.

Todas las funciones publicas siguen la misma forma: reciben las corridas ya
seleccionadas (de `src/runs.py`), escriben los archivos y devuelven la lista
de rutas escritas (vacia si no habia nada que escribir).
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.runs import medicion_benchmark
from src.style import (CABECERA, LILA, TINTA, ETIQUETA_CONFIG, ORDEN_CONFIG,
                       CONFIG_ELEGIDA, guardar, nombre_tarea, nombre_corto,
                       orden_tarea)


# ---------------------------------------------------------------------------
# Especificacion comun y los dos renderizadores
# ---------------------------------------------------------------------------
@dataclass
class Tabla:
    columnas: list
    filas: list = field(default_factory=list)
    # (fila, columna) de las celdas en negrita.
    negrita: set = field(default_factory=set)
    # Filas con fondo lila (la configuracion elegida).
    resaltar: set = field(default_factory=set)
    # Filas antes de las que se traza una regla fina (cambio de bloque).
    separar: set = field(default_factory=set)
    # Encabezados de grupo: (texto, primera_columna, ultima_columna).
    grupos: list = field(default_factory=list)
    nota: str = ""
    # Cuantas columnas de la izquierda van alineadas a la izquierda.
    n_texto: int = 2


def _md(tabla, destino):
    def fila(celdas):
        return "| " + " | ".join(celdas) + " |"

    # Markdown no tiene encabezados de grupo: el grupo se antepone a la
    # columna para que el nombre siga siendo inequivoco.
    columnas = list(tabla.columnas)
    for texto, c0, c1 in tabla.grupos:
        for c in range(c0, c1 + 1):
            columnas[c] = f"{texto}: {columnas[c]}"

    lineas = ([tabla.nota, ""] if tabla.nota else [])
    lineas += [fila(columnas), "|" + "---|" * len(columnas)]
    for i, celdas in enumerate(tabla.filas):
        lineas.append(fila([f"**{v}**" if (i, j) in tabla.negrita else v
                            for j, v in enumerate(celdas)]))
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def _png(tabla, destino):
    """Dibuja la tabla con matplotlib, en unidades de pulgada."""
    ancho_car, pad, alto = 0.058, 0.16, 0.2
    anchos = []
    for j, col in enumerate(tabla.columnas):
        textos = [col] + [f[j] for f in tabla.filas]
        anchos.append(max(len(t) for t in textos) * ancho_car + pad)
    # Un grupo mas ancho que sus columnas las ensancha por igual.
    for texto, c0, c1 in tabla.grupos:
        falta = len(texto) * ancho_car + pad - sum(anchos[c0:c1 + 1])
        if falta > 0:
            for c in range(c0, c1 + 1):
                anchos[c] += falta / (c1 - c0 + 1)

    n_cab = 2 if tabla.grupos else 1
    W = sum(anchos)
    H = alto * (n_cab + len(tabla.filas)) + 0.08
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    x0 = [sum(anchos[:j]) for j in range(len(anchos))]
    y_top, y_cab = 0.04, 0.04 + alto * n_cab

    def celda(texto, j, y, negrita=False, ancho=None, centro=None):
        izq = j < tabla.n_texto and centro is None
        if centro is None:
            centro = x0[j] + (0.08 if izq else anchos[j] / 2)
        ax.text(centro, y + alto / 2, texto, ha="left" if izq else "center",
                va="center", fontsize=8, color=TINTA,
                fontweight="bold" if negrita else "normal")

    ax.add_patch(Rectangle((0, y_top), W, y_cab - y_top, color=CABECERA,
                           linewidth=0, zorder=0))
    for i in tabla.resaltar:
        ax.add_patch(Rectangle((0, y_cab + alto * i), W, alto, color=LILA,
                               linewidth=0, zorder=0))

    if tabla.grupos:
        for texto, c0, c1 in tabla.grupos:
            xa, xb = x0[c0], x0[c1] + anchos[c1]
            celda(texto, c0, y_top, negrita=True, centro=(xa + xb) / 2)
            ax.plot([xa + 0.05, xb - 0.05], [y_top + alto - 0.02] * 2,
                    color=TINTA, linewidth=0.5)
    for j, col in enumerate(tabla.columnas):
        celda(col, j, y_cab - alto, negrita=True)

    for i, fila in enumerate(tabla.filas):
        for j, v in enumerate(fila):
            celda(v, j, y_cab + alto * i, negrita=(i, j) in tabla.negrita)
        if i in tabla.separar:
            ax.plot([0, W], [y_cab + alto * i] * 2, color=TINTA,
                    linewidth=0.35)

    y_fin = y_cab + alto * len(tabla.filas)
    for y, lw in ((y_top, 1.1), (y_cab, 0.6), (y_fin, 1.1)):
        ax.plot([0, W], [y, y], color=TINTA, linewidth=lw)
    return guardar(fig, destino.with_suffix(".png"), formatos=("png",))


_TEX_ESCAPES = [("\\", r"\textbackslash{}"), ("%", r"\%"), ("_", r"\_"),
                ("&", r"\&"), ("#", r"\#"), ("→", r"$\to$"),
                ("×", r"$\times$")]


def _tex_texto(texto):
    for original, escapado in _TEX_ESCAPES:
        texto = texto.replace(original, escapado)
    return texto


def _tex(tabla, destino):
    """
    Solo el entorno tabular, sin table ni caption: el informe decide donde va
    y que dice el pie. Requiere booktabs y colortbl (\\rowcolor) y los colores
    `cabecera` y `lila`, que define el preambulo del informe.
    """
    n = len(tabla.columnas)
    alineacion = "l" * tabla.n_texto + "c" * (n - tabla.n_texto)
    lineas = [f"\\begin{{tabular}}{{{alineacion}}}", "\\toprule"]
    if tabla.grupos:
        celdas, reglas, j = [], [], 0
        for texto, c0, c1 in sorted(tabla.grupos, key=lambda g: g[1]):
            celdas += [""] * (c0 - j)
            celdas.append(f"\\multicolumn{{{c1 - c0 + 1}}}{{c}}"
                          f"{{\\textbf{{{_tex_texto(texto)}}}}}")
            reglas.append(f"\\cmidrule(lr){{{c0 + 1}-{c1 + 1}}}")
            j = c1 + 1
        celdas += [""] * (n - j)
        lineas += ["\\rowcolor{cabecera}" + " & ".join(celdas) + " \\\\",
                   "".join(reglas)]
    lineas.append("\\rowcolor{cabecera}" + " & ".join(
        f"\\textbf{{{_tex_texto(c)}}}" for c in tabla.columnas) + " \\\\")
    lineas.append("\\midrule")
    for i, fila in enumerate(tabla.filas):
        if i in tabla.separar:
            lineas.append("\\midrule")
        celdas = [f"\\textbf{{{_tex_texto(v)}}}" if (i, j) in tabla.negrita
                  else _tex_texto(v) for j, v in enumerate(fila)]
        prefijo = "\\rowcolor{lila}" if i in tabla.resaltar else ""
        lineas.append(prefijo + " & ".join(celdas) + " \\\\")
    lineas += ["\\bottomrule", "\\end{tabular}"]
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return destino


def _escribir(tabla, base):
    return [_md(tabla, base.with_suffix(".md")), _png(tabla, base),
            _tex(tabla, base.with_suffix(".tex"))]


# ---------------------------------------------------------------------------
# Ayudas de formato
# ---------------------------------------------------------------------------
def _pct(x):
    return f"{100 * x:.2f}"


def _minutos(segundos):
    return f"{segundos / 60:.1f}"


def _f1_val(corrida):
    """F1 de validacion de la mejor epoca, o None si la corrida no lo trae."""
    return (corrida.get("desempeno_val") or {}).get("mejor_f1")


def _mejor_por_bloque(tabla, bloques, columnas_max, columnas_min=()):
    """
    Marca en negrita el mejor valor de cada columna dentro de cada bloque de
    filas (p. ej. las dos filas de un dataset). Los empates se marcan todos.
    """
    for filas in bloques:
        for cols, mejor in ((columnas_max, max), (columnas_min, min)):
            for j in cols:
                valores = {}
                for i in filas:
                    try:
                        valores[i] = float(tabla.filas[i][j].rstrip("×"))
                    except ValueError:
                        continue
                if valores:
                    tope = mejor(valores.values())
                    tabla.negrita |= {(i, j) for i, v in valores.items()
                                      if v == tope}


def _eficiencia_controlada(corrida, bench):
    """(latencia ms, memoria MB) del benchmark a batch 1 y la longitud del
    dataset, como texto; 'n/a' si no hay benchmark."""
    punto = medicion_benchmark(bench, corrida["modelo"],
                               corrida.get("max_length"))
    if not punto:
        return "n/a", "n/a"
    return (f"{punto['latencia_ms_media']:.2f}",
            f"{punto['memoria_mb_pico']:.0f}")


def _filas_modelos(tabla, filas_por_tarea, bench, con_val):
    """Rellena una tabla de (dataset, modelo) con desempeno y eficiencia."""
    bloques = []
    for tarea in sorted(filas_por_tarea, key=orden_tarea):
        bloque = []
        if tabla.filas:
            tabla.separar.add(len(tabla.filas))
        for modelo in ("bert", "distilbert"):
            c = filas_por_tarea[tarea].get(modelo)
            if not c:
                continue
            d, e = c["desempeno_test"], c["eficiencia"]
            lat, mem = _eficiencia_controlada(c, bench)
            val = _f1_val(c)
            fila = [nombre_tarea(tarea), nombre_corto(c["modelo_nombre"]),
                    _pct(d["accuracy"]), _pct(d["precision"]),
                    _pct(d["recall"]), _pct(d["f1"])]
            if con_val:
                fila.append(_pct(val) if val is not None else "n/a")
            fila += [f"{e['n_parametros'] / 1e6:.1f}", lat, mem,
                     _minutos(c.get("tiempo_entrenamiento_s", 0))]
            bloque.append(len(tabla.filas))
            tabla.filas.append(fila)
        bloques.append(bloque)
    n_des = 5 if con_val else 4
    _mejor_por_bloque(tabla, bloques, range(2, 2 + n_des),
                      range(2 + n_des, 2 + n_des + 4))


# ---------------------------------------------------------------------------
# Desempeno y eficiencia de las corridas base
# ---------------------------------------------------------------------------
def tabla_resultados(corridas, base, bench=None):
    """Desempeno en test y eficiencia, una fila por (modelo, dataset)."""
    if not corridas:
        return []
    tabla = Tabla(
        columnas=["Dataset", "Model", "Acc", "Prec", "Rec", "F1",
                  "Params (M)", "Latency (ms)", "Memory (MB)", "Train (min)"],
        grupos=[("Test (%)", 2, 5), ("Efficiency", 6, 9)],
        nota="Corridas base. Precision, recall y F1 son macro. Latencia y "
             "memoria: benchmark controlado a batch 1 con la longitud del "
             "dataset. En negrita, el mejor de cada dataset.")
    por_tarea = {}
    for (modelo, tarea), c in corridas.items():
        por_tarea.setdefault(tarea, {})[modelo] = c
    _filas_modelos(tabla, por_tarea, bench, con_val=False)
    return _escribir(tabla, base)


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------
def _cabeza(abl):
    ocultas = abl.get("head_hidden") or []
    return " → ".join(["768", *map(str, ocultas), "C"])


def tabla_ablation(ablaciones, base, modelo="distilbert"):
    """
    Resumen del ablation study: F1 de validacion de cada configuracion en los
    tres datasets y su media.

    La media es la columna que decide: con ella se elige la configuracion.
    El test no aparece a proposito, para que nadie elija mirandolo.
    """
    filas = {k: v for k, v in ablaciones.items() if k[0] == modelo}
    if not filas:
        return []
    tareas = sorted({k[1] for k in filas}, key=orden_tarea)
    configs = [c for c in ORDEN_CONFIG if any(k[2] == c for k in filas)]

    tabla = Tabla(
        columnas=["Config", "Head", "Encoder",
                  *[nombre_tarea(t) for t in tareas], "Mean",
                  "Trainable (M)", "Train (min)"],
        grupos=[("Validation F1 (%)", 3, 3 + len(tareas))],
        nota=f"Ablation sobre {modelo}. La fila resaltada es la "
             "configuracion elegida (mayor F1 medio de validacion).",
        n_texto=3)
    for cfg in configs:
        corridas = [filas.get((modelo, t, cfg)) for t in tareas]
        presentes = [c for c in corridas if c]
        abl = presentes[0].get("ablacion") or {}
        vals = [_f1_val(c) if c else None for c in corridas]
        conocidos = [v for v in vals if v is not None]
        entren = presentes[0]["eficiencia"].get("n_parametros_entrenables")
        if cfg == "frz":
            tabla.separar.add(len(tabla.filas))
        if cfg == CONFIG_ELEGIDA:
            tabla.resaltar.add(len(tabla.filas))
        tabla.filas.append([
            cfg, _cabeza(abl),
            "frozen" if abl.get("freeze_encoder") else "trained",
            *[_pct(v) if v is not None else "n/a" for v in vals],
            _pct(sum(conocidos) / len(conocidos)) if conocidos else "n/a",
            f"{entren / 1e6:.1f}" if entren else "n/a",
            _minutos(sum(c.get("tiempo_entrenamiento_s", 0)
                         for c in presentes)),
        ])
    n = len(tareas)
    _mejor_por_bloque(tabla, [range(len(tabla.filas))],
                      range(3, 4 + n), range(4 + n, 6 + n))
    return _escribir(tabla, base)


def tabla_mejor_config(ablaciones, base, bench=None, config=CONFIG_ELEGIDA):
    """
    La configuracion ganadora del ablation, en los dos modelos y con todas
    las metricas: desempeno en test, F1 de validacion y eficiencia.

    Es la tabla que pide el enunciado al final. Devuelve [] mientras falte
    alguno de los dos modelos: media tabla induce a error.
    """
    filas = {k: v for k, v in ablaciones.items() if k[2] == config}
    if len({k[0] for k in filas}) < 2:
        return []
    tabla = Tabla(
        columnas=["Dataset", "Model", "Acc", "Prec", "Rec", "F1", "Val F1",
                  "Params (M)", "Latency (ms)", "Memory (MB)", "Train (min)"],
        grupos=[("Performance (%)", 2, 6), ("Efficiency", 7, 10)],
        nota=f"Configuracion `{config}` ({ETIQUETA_CONFIG.get(config)}) en "
             "los dos modelos, misma receta. En negrita, el mejor de cada "
             "dataset.")
    por_tarea = {}
    for (modelo, tarea, _), c in filas.items():
        por_tarea.setdefault(tarea, {})[modelo] = c
    _filas_modelos(tabla, por_tarea, bench, con_val=True)
    return _escribir(tabla, base)


# ---------------------------------------------------------------------------
# Benchmark controlado
# ---------------------------------------------------------------------------
def tabla_benchmark(benchmark, base):
    """Latencia y memoria de los dos modelos en la misma rejilla."""
    mods = {m["modelo"]: m for m in benchmark.get("modelos", [])}
    if not {"bert", "distilbert"} <= set(mods):
        return []

    def indice(mod):
        return {(m["batch_size"], m["seq_len"]): m for m in mod["mediciones"]}

    b, d = indice(mods["bert"]), indice(mods["distilbert"])
    tabla = Tabla(
        columnas=["Batch", "Length", "BERT", "DistilBERT", "Speedup",
                  "BERT", "DistilBERT", "Ratio"],
        grupos=[("Latency (ms)", 2, 4), ("Peak memory (MB)", 5, 7)],
        nota=f"Medido en {benchmark.get('dispositivo', '?')} "
             f"({benchmark.get('precision', '?')}), mismo proceso.")
    lote_previo = None
    for clave in sorted(set(b) & set(d)):
        if lote_previo is not None and clave[0] != lote_previo:
            tabla.separar.add(len(tabla.filas))
        lote_previo = clave[0]
        mb, md = b[clave], d[clave]
        tabla.filas.append([
            str(clave[0]), str(clave[1]),
            f"{mb['latencia_ms_media']:.2f}", f"{md['latencia_ms_media']:.2f}",
            f"{mb['latencia_ms_media'] / md['latencia_ms_media']:.2f}×",
            f"{mb['memoria_mb_pico']:.0f}", f"{md['memoria_mb_pico']:.0f}",
            f"{md['memoria_mb_pico'] / mb['memoria_mb_pico']:.2f}×",
        ])
    return _escribir(tabla, base)
