#!/usr/bin/env bash
# compile.sh - Helper script to build the LaTeX manuscript

DOC="main"

echo "=> Running pdflatex (1/3)..."
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" > /dev/null
if [ $? -ne 0 ]; then
    echo "Error during first pdflatex run. Check $DOC.log for details."
    exit 1
fi

echo "=> Running bibtex (2/3)..."
bibtex "$DOC" > /dev/null

echo "=> Running pdflatex (3/3)..."
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$DOC.tex" > /dev/null

echo "Compilation successful. Output generated: $DOC.pdf"
