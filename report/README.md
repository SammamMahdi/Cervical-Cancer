# Legacy-model report (2-column arXiv style)

`legacy_report.pdf` — 4-page, two-column arXiv-style report of the **legacy** SIPaKMeD
B4→B0 knowledge-distilled model: in-distribution results → calibration (ECE) → Grad-CAM →
cross-dataset (Herlev/APACC/HiCervix) → a deep diagnosis of *why* cross-dataset fails.

## Files
- `legacy_report.tex` — the report source (uses `arxiv.sty`, `twocolumn`).
- `references.bib`, `arxiv.sty`, `orcid.pdf` — template assets.
- `figures/` — all figures (house SciencePlots style), incl. the two generated diagnostics:
  `cross_separation.png` (P(abnormal) overlap per dataset) and `cm_cross_combined.png`.
- `_tools/tectonic.exe` (+ `_tools/cache/`) — the LaTeX engine used to build it.

## Recompile
No system LaTeX is installed; build with the bundled Tectonic engine:

```bash
cd report
export TECTONIC_CACHE_DIR="$PWD/_tools/cache"     # PowerShell: $env:TECTONIC_CACHE_DIR="$PWD\_tools\cache"
./_tools/tectonic.exe legacy_report.tex
```

(Tectonic runs the bib + multiple passes automatically and writes `legacy_report.pdf`.)

## Numbers reported (all from `Legacy_src/results/`)
- In-dist student: acc 0.966, macro-F1 0.967, AUROC 0.998, ECE 0.015 (beats the teacher).
- Cross-dataset (binary): Herlev 0.639 / AUROC 0.73, HiCervix 0.605 / 0.59, APACC 0.532 / 0.53.
- Diagnosis: positive P(abnormal) separation everywhere (no label flip) → failure is domain shift
  + operating-point miscalibration, amplified by an optimistic leaky internal split.
