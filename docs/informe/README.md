# Informe del Proyecto 1

Informe técnico en inglés con el template de **NeurIPS 2025**
(`neurips_2025.sty`, copiado de `Styles/`). Tiene las seis secciones del
enunciado y ocupa 4 páginas sin contar las referencias. Se compila con
[tectonic](https://tectonic-typesetting.github.io/), que descarga solo los
paquetes que necesita, sin TeX Live.

```bash
conda install -c conda-forge tectonic   # o: brew install tectonic
make                                     # genera main.pdf
```

## Estructura

```
main.tex              preambulo, estilo de tablas y orden de las secciones
neurips_2025.sty      estilo oficial de NeurIPS 2025
secciones/
  00-abstract.tex
  01-introduction.tex
  02-approach.tex
  03-experiments.tex        <- figuras y tablas
  04-analysis.tex           <- incluye las limitaciones
  05-conclusion.tex
referencias.bib
Makefile
```

## Figuras y tablas

Ninguna vive aquí. `main.tex` lee de `../../results/figures/`, donde las
escribe `python -m src.plots`:

- las figuras, en PDF vectorial.
- Las tablas, como `tabla_*.tex` (solo el `tabular`). `src/tables.py` las
  genera desde la misma especificación que el Markdown y el PNG del README.
  El pie y la posición de cada tabla se escriben en el informe.

Las tablas usan `\rowcolor` con dos colores del preámbulo: `cabecera` para el
encabezado y `lila` para la configuración elegida, como en las tablas de
ChangeTitans (la referencia de estilo).

| Objetivo | Qué hace |
|---|---|
| `make` | compila `main.pdf` con las figuras actuales |
| `make figuras` | regenera figuras y tablas desde los JSON y luego compila |
| `make standalone` | copia figuras y tablas a `figuras/` para enviar solo esta carpeta |
| `make limpiar` | borra el PDF, los auxiliares y `figuras/` |

`figuras/` no se versiona (está en el `.gitignore`). Duplicaría
`results/figures/` y quedaría vieja en cuanto alguien regenerara el pipeline.
