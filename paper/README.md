# Paper (IEEE format)

| File | Contents |
|---|---|
| `main.tex` | Full paper in the IEEE conference format (`IEEEtran`), with the ENSO (El Niño) extension. Red `[TODO: ...]` markers show what still depends on the final experiments. |
| `snippet.tex` | Short extract for review: title, authors, abstract, introduction and proposed method. It contains no results and no TODO markers. |
| `framework_fig.tex` | Framework diagram (TikZ), used by both files. |
| `references.bib` | Bibliography. Entries marked `% VERIFY` are incomplete and must be replaced with the full references before submission. |
| `main.pdf`, `snippet.pdf` | Compiled versions of the two files. |

## Opening in Overleaf

1. Download this `paper/` folder as a zip (or the whole repository from GitHub: **Code → Download ZIP**, then zip the `paper/` folder only).
2. In Overleaf: **New Project → Upload Project**, and select the zip.
3. Set the main document (**Menu → Main document**) to `snippet.tex` or `main.tex`.
4. Compile with pdfLaTeX. `IEEEtran` is installed in Overleaf by default.

With an Overleaf plan that includes GitHub sync, **New Project → Import from GitHub** imports the repository directly; then set the main document to `paper/main.tex` or `paper/snippet.tex`.

## Building locally

```
latexmk -pdf snippet.tex
latexmk -pdf main.tex
```
