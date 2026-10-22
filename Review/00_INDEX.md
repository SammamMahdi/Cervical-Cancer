# Systematic Review & Research Proposal — Deliverables Index

**Topic:** *Flexible Multimodal Cervical Cancer Screening: Integrating Pap-Smear Image Analysis and Patient Risk Factors Under Missing-Modality Conditions.*

**Evidence base:** **155 verified papers** — 45 in-house cervical-AI (2023–2026, each re-verified against its live source) + **110** from an 11-cluster systematic search, every one independently checked by an anti-fabrication verifier.

> **Integrity headline: 110/110 papers CONFIRMED to exist — zero invented.** The verifiers did catch **21 records needing fixes**, and all were enforced **in code**, not prose: **2 fabricated evidence quotes removed**, **3 unsupported number-sets blanked**, **3 metadata errors flagged** (e.g. CerviFusionNet author `Qiang Zhang` → `Qingyue Zhang`).

---

## Deliverables map

| # | Deliverable | Where |
|--:|---|---|
| 1 | Systematic literature review | `01_TAXONOMY_AND_LANDSCAPE.md` + `Literature/` |
| 2 | Local PDF repository (where legally available) | `Literature/PDFs/` — **64 OA PDFs** (260 MB) |
| 3 | Extracted Markdown for every paper | `Literature/Markdown/` — **155 files** |
| 4 | BibTeX database | `Literature/BibTeX/literature.bib` — 155 entries |
| 5 | Master CSV of all papers | `Literature/CSV/papers.csv` — **155 × 39** |
| 6 | Taxonomy of AI methods | `01_TAXONOMY_AND_LANDSCAPE.md` §1 |
| 7 | Evidence-based landscape review | `01_TAXONOMY_AND_LANDSCAPE.md` §2–4 |
| 8 | **Gap analysis** | `02_GAP_ANALYSIS.md` — 12 retained gaps, **6 rejected**, 8 absence findings |
| 9 | Ranked research opportunities | `03_OPPORTUNITIES_RQ_HYPOTHESES.md` (Phase 7) |
| 10 | Research questions | `03_…` (Phase 8) — RQ1–RQ5 |
| 11 | Testable hypotheses | `03_…` (Phase 9) — H1a–H5 + **global falsifier** |
| 12 | Novel methodology | `04_METHODOLOGY_AND_MATH.md` (Phase 10) — modules M1–M9 |
| 13 | Mathematical formulation | `04_…` (Phase 11) — Eq. (1)–(7) + complexity |
| 14 | End-to-end architecture | `04_…` §1 |
| 15 | Recommended models & implementation | `05_MODELS_EXPERIMENTS_ABLATIONS.md` (Phase 12) |
| 16 | Experimental protocol | `05_…` (Phase 13) |
| 17 | Ablation plan | `05_…` (Phase 14) — A1–A14 |
| 18 | Statistical analysis plan | `05_…` (Deliverable 18) |
| 19 | Risk assessment | `06_RISKS_REPRODUCIBILITY.md` (Phase 15) |
| 20 | Reproducibility checklist | `06_…` |
| 21 | **Explicit novelty statement** | `10_NOVELTY_STATEMENT.md` |

**Supporting data:** `Literature/JSON/slr_clusters.json` (raw cluster results + verifier verdicts) · `slr_gaps_critic.json` (gap synthesis + adversarial novelty critic) · `Literature/Metadata/acquisition_log.csv` (status **and reason** per paper) · `Literature/build_database.py`, `Literature/acquire_sources.py` (rebuild).

---

## The headline result of this review

**The proposal we were asked to evaluate was rejected under the stated quality standard, and replaced with a smaller, defensible one.**

*Original framing* — "two experts, shared axis, evidence-weighted fusion, conformal abstention" — **fails three kill criteria**:
1. **Novelty = assembly.** The fusion rule is **ProbEn** (ECCV 2022 oral); one level up, a **weighted opinion pool** (Genest & Zidek 1986). Per-subset calibration is **AECF** (which *measures* it). Modality dropout for image+tabular is **Gu et al. (MICCAI 2025)**. Prior correction is **Saerens 2002 / Alexandari 2020**.
2. **The headline path is unvalidatable.** **No public dataset pairs a cervical image with a risk questionnaire for the same patient** — verified dataset-by-dataset; the decisive miss is **UCI 384 vs UCI 383** (same authors, hospital, year — yet no questionnaire, no patient ID, 287 vs 858).
3. **Construct + unit mismatch.** Cytomorphology labels vs a biopsy target; cell-level vs patient-level. Calling their fusion "P(abnormal)" is a **definitional** error, not a calibration one.

*What survives* — **CO-PRIOR**:
> **The two-prior composition operator** — prior-*stripping* as a precondition of pooling experts trained on disjoint cohorts. `λ_k = logit(p̃_k) − logit(π_k)`, then `logit P = logit(τ) + Σ w_k λ_k`.
> The **only** axis where adversarial search found no collision: prior-correction papers retarget **one** classifier to **one** population; combination papers pool **several** under a **common** prior. **Nobody strips priors as a precondition of pooling.**
> Decisively, **it is validatable per-expert on public data with no paired cohort** — which is what rescues the programme.

Supported by two standalone results:
- **Leakage-audited UCI baseline:** Biopsy **AUROC 0.656 [0.584, 0.720]** (nested CV) vs a published **0.98–1.00** that is fully explained by leakage (`Dx:Cancer` from other `Dx` columns = **0.998**; from legitimate features = **0.592**; 8-family tuned ceiling ≈ 0.65; best single feature 0.572).
- **The price of the forced assumption:** dependence ρ is **non-identified** at zero overlap, so independence is forced; **Grill et al. (2017)** predict directional over-estimation **up to 56%** — pre-registered as a falsifiable property of our own output.

**Pre-registered falsifier:** if composition does not beat `max(image-only, tabular-only)` on **both** discrimination and calibration, the **negative result is the finding** (precedent: Kelebet et al. 2025 — fused 92% < tabular alone 96%).

---

## Evidence hygiene

**Quarantined — located but NOT verified; must be retrieved before citing:** MC2 (Research Square; **do not cite its 0.73-vs-0.90 figure**), Uddin et al. 2024 (ScienceDirect 403), the "88.9–92.6% use UCI" review (403), UAD-FM (Nature auth wall), and **Poh & Bengio's chimeric-user conclusions** (publisher-elided — existence HIGH confidence, content unknown).

**Contrary evidence honoured (claims narrowed accordingly):** AECF (per-subset calibration is claimed *and measured*), Gu et al. MICCAI 2025 (modality dropout not novel), Poh & Bengio (pseudo-pairing *is* critiqued — our absence-finding #13 was **withdrawn**), Genest & Zidek (opinion pooling is 40 years old), Podkopaev & Ramdas (conformal-under-shift needs unlabeled target data we lack).
