# Makefile for LaTeX Paper

PROJECT = main
TEX = pdflatex
BIB = bibtex
BUILD_DIR = build

all: $(PROJECT).pdf

$(PROJECT).pdf: $(PROJECT).tex references.bib
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
