# Research Proposal (ACM `acmart`)

Self-contained LaTeX source for the proposal
*"Any-Modality Cervical Cancer Screening by Calibrated Decision-Level Fusion of Two Unpaired Experts."*

## Files
- `main.tex` — the proposal (single-column `acmart` **manuscript** format).
- `refs.bib` — bibliography (21 references, all verified against primary sources).
- `acmart.cls`, `ACM-Reference-Format.bst` — the ACM class + BibTeX style (from the ACM SIGPLAN template zip).

## How to compile
**Overleaf (recommended):** create a project, upload all four files, set `main.tex` as the main document, and compile with **pdfLaTeX**. Overleaf runs BibTeX automatically. `acmart` and all packages used (`tikz`, `tcolorbox`, `algorithm`, `algpseudocode`) are pre-installed there.

**Locally (needs a TeX distribution, e.g. TeX Live / MiKTeX):**
```
latexmk -pdf main.tex          # runs pdflatex + bibtex + reruns automatically
# or manually:
pdflatex main ; bibtex main ; pdflatex main ; pdflatex main
```

## Format note
`main.tex` uses `\documentclass[manuscript,screen]{acmart}` — the single-column ACM manuscript style, chosen for readability and to prevent wide diagrams/tables from overflowing. To produce the two-column conference look, change the first line to `\documentclass[sigconf]{acmart}` (wide figures already fit; the `\resizebox` wrappers keep them within the column).

## Verification status
- LaTeX structure checked programmatically: all environments and braces balanced; all 21 citations resolve to bib entries; 5 figures, 10 tables, 1 algorithm.
- Citations were verified against primary sources (venue/year/DOI).
- The document was **not** compiled in this environment (no TeX engine present here); compile on Overleaf or a local TeX install as above.
