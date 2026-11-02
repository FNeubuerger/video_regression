# Makefile for LaTeX Paper

PROJECT = main
TEX = pdflatex
BIB = bibtex
BUILD_DIR = build

PAPER_INPUTS = $(filter-out $(PROJECT).tex,$(wildcard *.tex))
TABLES = $(wildcard tables/*.tex)
FIGURES = $(wildcard figures/*.png)

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).tex references.bib $(PAPER_INPUTS) $(TABLES) $(FIGURES)
	mkdir -p $(BUILD_DIR)
	$(TEX) -output-directory=$(BUILD_DIR) $(PROJECT).tex
	cp references.bib $(BUILD_DIR)/
	cd $(BUILD_DIR) && $(BIB) $(PROJECT)
	$(TEX) -output-directory=$(BUILD_DIR) $(PROJECT).tex
	$(TEX) -output-directory=$(BUILD_DIR) $(PROJECT).tex
	cp $(BUILD_DIR)/$(PROJECT).pdf .

clean:
	rm -rf $(BUILD_DIR) $(PROJECT).pdf

.PHONY: all clean
