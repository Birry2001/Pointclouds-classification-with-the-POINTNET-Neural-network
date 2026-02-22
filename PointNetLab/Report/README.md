# Rapport LaTeX - PointNetLab

## Compilation
Depuis `PointNetLab/Report`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

(ou `latexmk -pdf main.tex` si disponible)
