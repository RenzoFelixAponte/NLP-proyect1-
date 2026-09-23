# Informe del Proyecto 1. Requiere tectonic (brew install tectonic).
#
# Las figuras NO viven aqui: se leen de ../../results/figures/, que es donde
# las escribe `python -m src.plots`. El objetivo `figuras` las regenera.

TEX      := tectonic
MAIN     := main
FIGURAS  := ../../results/figures

.PHONY: all figuras standalone limpiar

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex secciones/*.tex referencias.bib
	$(TEX) -X compile $(MAIN).tex

# Regenera las figuras desde los JSON antes de compilar.
figuras:
	cd ../.. && python -m src.plots
	$(MAKE) all

# Copia las figuras dentro de la carpeta para poder enviar solo docs/informe/.
# \graphicspath busca primero en ../../results/figures/ y luego en figuras/.
standalone:
	mkdir -p figuras
	cp $(FIGURAS)/*.pdf figuras/
	$(TEX) -X compile $(MAIN).tex

limpiar:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg $(MAIN).toc
	rm -rf figuras
