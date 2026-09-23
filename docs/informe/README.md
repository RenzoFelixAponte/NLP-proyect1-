# Informe del Proyecto 1

Fuente LaTeX del informe. Se compila con [tectonic](https://tectonic-typesetting.github.io/),
que descarga solo los paquetes que necesita (no hace falta instalar TeX Live).

```bash
brew install tectonic      # una vez
make                       # genera main.pdf
```

## Estructura

```
main.tex              preambulo, paquetes y orden de las secciones
secciones/
  00-resumen.tex
  01-introduccion.tex
  02-datos.tex
  03-metodologia.tex        <- incluye el tratamiento de los pesos
  04-resultados.tex
  05-ablation.tex
  06-eficiencia.tex
  07-discusion.tex          <- mejor configuracion + limitaciones
  08-conclusiones.tex
  A-reproducibilidad.tex    <- apendice
referencias.bib
Makefile
```

## Las figuras

No se copian aqui. `main.tex` las lee de `../../results/figures/` en su
version PDF, que es donde las escribe `python -m src.plots`. Asi el informe
no puede quedar desincronizado del pipeline.

| Objetivo | Que hace |
|---|---|
| `make` | compila `main.pdf` con las figuras actuales |
| `make figuras` | regenera las figuras desde los JSON y luego compila |
| `make standalone` | copia las figuras a `figuras/` para poder enviar solo esta carpeta |
| `make limpiar` | borra el PDF, los auxiliares y `figuras/` |

## Si hay que reenviar solo el informe

`make standalone` deja la carpeta autocontenida: copia las figuras dentro de
`figuras/`, que es el segundo camino de `\graphicspath`. Esa copia **no se
versiona** (esta en el `.gitignore`), porque duplicaria las figuras en el repo
y se quedaria vieja en cuanto alguien regenerara el pipeline.
