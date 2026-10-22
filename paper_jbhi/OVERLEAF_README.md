# Overleaf package

A self-contained Overleaf project for the IEEE JBHI submission: the paper and its supplement.

## Upload
1. **overleaf.com → New Project → Upload Project** and drop `paper_jbhi_overleaf.zip`.
2. **Menu (top left)** → set **Main document** to `main.tex`, **TeX Live** 2023 or newer.
3. **Recompile.**

To build the supplement, set **Main document** to `supplement.tex` and recompile. Overleaf compiles one
document at a time, so switch back afterwards.

## Expected output

| Document | Pages | Floats |
|---|---|---|
| `main.tex` | **14** | 12 figures, 7 tables |
| `supplement.tex` | **12** | 17 figures, 9 tables |

If you get fewer pages than this, check the two things below before anything else.

**1. Missing references (costs ~2 pages).** If the citations render as `[?]` and the REFERENCES section
is empty or absent, BibTeX has not run yet. `main.bbl` is included so this should not happen, but if it
does, press **Recompile** a second time, or use **Logs → Clear cached files** and recompile.

**2. Missing supplementary tables (costs ~3 pages).** `supplement.tex` pulls its nine tables from the
`supp/` folder via `\input`. If that folder did not upload, the tables vanish. Confirm `supp/` is present
in the file tree on the left.

## Layout notes
The paper is two-column IEEE format. The **supplement is single-column on purpose**: it is figure-heavy,
and a two-column layout cannot place a full-width figure beside text, which forced float-only pages with
large blank areas. In one column the figures and their discussion share pages.

Both documents load `\usepackage[T1]{fontenc}`. That is what makes the body text render in Times
(`NimbusRomNo9L`) rather than a substituted font — without it some engines silently fall back to Latin
Modern, which is wider and adds roughly a page over a document this long. Do not remove it; the page
counts above assume it.

## What's inside

| File | Purpose |
|---|---|
| `main.tex` | The paper (IEEEtran, journal mode) |
| `supplement.tex` | Supplementary material, numbered S1, S2, … independently |
| `supp/*.tex` | The nine supplementary tables, generated from the result files |
| `main.bbl` | Pre-built bibliography, so references appear on the first compile |
| `Mybib.bib` | The original bibliography, unchanged |
| `refs_added.bib` | References added during the revision (EfficientNet, distillation, prior shift, conformal prediction, calibration, decision curves, Grad-CAM, the co-test cohort, products of experts, …) |
| `IEEEtran.cls` | Official IEEE class file |
| `figures/*.pdf` | All 26 figures as vector PDFs |

## Regenerating

Nothing here is hand-typed. The figures come from `scripts/make_figures.py` and
`scripts/make_figures2.py`, and the supplementary tables from `scripts/make_supp_tables.py`, all reading
the JSON files in `results/`. Figures are authored at their printed size (7.16 in across both columns,
3.45 in for one) and are never rescaled in LaTeX, so `\includegraphics` widths should not be changed.
