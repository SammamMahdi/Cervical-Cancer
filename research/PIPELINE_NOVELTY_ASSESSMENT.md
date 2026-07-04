# Grounding Verdict — CerviWISE vs FICCD (+ a buildable alternative)

**Method:** 17 live web-grounded prior-art / feasibility checks + 2 skeptical per-pipeline assessments (verified 2026-06-21). Question answered: *which proposed pipeline is genuinely novel, and which will actually work?*

## One-line answer
**Neither is genuinely novel** — every component of both is established prior art, and a real 2026 paper (**SiCal-CytoNet**) already fuses the "trustworthy stack" both rely on. **CerviWISE will actually run** (cell-level, real public data, fits your GPU) but isn't a novel method; **FICCD will not run as described** (its WSI/MIL core has no compatible public data, its result tables are invented, and its "guaranteed safety" claim is unsound). The buildable, defensibly-novel path is a **third option** (below) built on your existing baseline.

---

## Scorecard

| | **CerviWISE** (ChatGPT) | **FICCD** (Gemini) |
|---|---|---|
| Novelty | **3/10** | **3/10** |
| Feasibility (real data + RTX 5060 Ti) | **7/10** | **2/10** |
| Will it work? | **Partly** | **No (as specified)** |
| Fatal issue | Not novel + clinically mislabeled task | No public data for its core + fabricated results |

---

## Why neither is novel (every block is prior art, much of it in cervical cytology)

- **Domain-adversarial training (DANN)** — established in histopathology (Lafarge 2017) and cervical cytology (STRIDE 2024, LSA 2025, stain-norm 1909.05184). *Not novel.*
- **Temperature-scaling / ECE calibration** — Guo 2017; already reported on SIPaKMeD cells (PMC12550697: ECE 0.030; SiCal-CytoNet 2026). *Not novel.*
- **Fourier Domain Adaptation + Beer-Lambert stain** — Yang & Soatto 2020; histopathology FDA (AIDA, F-SN/F-SA, Curri-AFDA). *Not novel.*
- **Evidential Dirichlet head** — Sensoy 2018; already benchmarked on SIPaKMeD cervical cells (BEDL 2026). *Not novel.*
- **Gated-attention MIL** — Ilse 2018; routine baseline in cervical-WSI work (UniCAS, Smart-CCS, arXiv 2407.11486). *Not novel.*
- **LoRA on EfficientNet MBConv** — `loralib` ships Conv2d; ConvLoRA/CoLoRA in medical imaging. Technically fine but **marginal savings on a 5.3M-param net** (full fine-tuning is already cheap). *Not novel.*
- **Mondrian conformal triage (release/defer/flag)** — Cohen 2024 "Conformal Triage"; Narendra (Mondrian CP on CRIC); Hagos (cervical-atypia CP). *Not novel.*

### The decisive prior-art hit: **SiCal-CytoNet (2026, real, verified from publisher PDF)**
*"SiCal-CytoNet: A Stain-Invariant, Calibrated Multi-Scale Network for Trustworthy Cytology on SIPaKMeD"* (Science and Engineering Connect, Vol 49(1):28–52, 2026) **already combines** optical-density/Beer-Lambert stain invariance + **Fourier Domain Adaptation** + MixStyle + **evidential Dirichlet** + Brier + **temperature-scaling calibration** + **uncertainty-based abstention** on SIPaKMeD cells (98.9% acc, ECE 1.7%, automates 88%). This **pre-empts the entire "trustworthy/calibrated/stain-robust" headline of both proposals.** Both must cite it; CerviWISE's calibration-novelty claim is dead, and FICCD's trustworthy-stack novelty is gone.
*(Note: two of my search agents initially failed to find SiCal-CytoNet and called it fabricated; a deeper agent fetched the actual galley PDF and confirmed it is real. It is real — treat it as the key prior art.)*

---

## CerviWISE — feasible, not novel, and clinically mislabeled

**Works:** cell-level → trains and externally evaluates on real public data (SIPaKMeD/Mendeley LBC → Herlev/CRIC) on a single RTX 5060 Ti; your B4→B0 KD baseline already exists.

**But:**
- ⚠️ **"ASC-US+" is a misnomer.** SIPaKMeD has **no ASC-US/atypical class** — its lesion cells are overt LSIL (koilocytotic) and HSIL (dyskeratotic). And **metaplastic (~793 cells) is clinically NILM**, not abnormal, so any binary collapse mis-assigns ~20% of the data. Merging LSIL+HSIL discards the low- vs high-grade distinction that drives clinical management.
- ⚠️ **DANN is statistically underpowered** with only **2 source domains** (SIPaKMeD + Mendeley) — the domain discriminator just overfits a 2-way split. Prior work on these exact datasets (Akhras 2026) *deliberately declined* learned DA for this reason and used Reinhard normalization instead.
- ⚠️ **No headroom:** binary SIPaKMeD accuracy is already ~99.5% (A2SDNet121).
- ⚠️ Your existing KD has the **double-softmax / missing-T² bug** (KL term ≈0.005), so distillation barely contributes — a no-teacher ablation is mandatory.

**Verdict:** feasible but a reject-on-novelty as a "new method." Salvageable as a *rigorous external-generalization + measured-efficiency benchmark* (which the field actually lacks).

---

## FICCD — sophisticated, but not buildable as described

**Fatal flaws:**
1. ❌ **No public cervical-cytology WSI data exists.** FICCD is WSI-tiling → MIL → slide-level conformal triage, which needs gigapixel WSIs with **slide-level labels** and many patches per slide. **All five datasets it names (SIPaKMeD, Herlev, CRIC, Cx22, Mendeley LBC) are isolated cell / small-FOV crops with no slide labels and no MIL bags.** The only *open* cervical WSI set is **H&E histology** (wrong modality); the cytology WSI corpora that would work (CSD 2,625 WSIs; STRIDE 341,889 WSIs) are **proprietary**. → its distinguishing core **can't be trained or validated** on available data.
2. ❌ **The result tables are invented.** No "FICCD" paper exists, so 95.8% acc / ECE 1.3% / AUROC 0.997 / RS-FNR 0.0% / 4.13 GB / 7.02 ms / Friedman χ²=142.6 / Wilcoxon p<0.0001 are **design targets presented as measured results** (the 0.997 AUROC even spuriously matches an unrelated SIPaKMeD cell-level KD paper).
3. ❌ **"Mathematically guaranteed safety" is unsound** in FICCD's own regime. Conformal coverage needs exchangeability, which FICCD *breaks by design* (it applies stain-DA precisely because distributions differ). The 2026 audit **arXiv 2605.20956** shows that under prevalence shift, marginal coverage stays nominal while **abnormal-class (release-side) coverage collapses (~56%) and true positives get auto-released (~39% event-release risk)** — exactly FICCD's failure mode; Mondrian only fixes it at ~96% deferral.
4. ⚠️ **Factual errors:** calls UniCAS "ViT-H" (it's **ViT-L**: 1024-dim/24-layer); conflates **APACC** (4 coarse classes) with **APCData** (the real 6-class Bethesda set).

**Verdict:** will not work as a Q1 contribution and cannot be honestly executed on the stated data/resources. Compute is the *only* thing that's fine.

---

## My recommendation — a pipeline that will actually work (and has a defensible, novel angle)

Stay **cell/FOV-level** (forced by data reality), build on your **existing B4→B0 baseline**, and put the novelty where there is a *real, un-closed, buildable* gap: **release-side safety auditing under prevalence/domain shift** — the one thing SiCal-CytoNet, the conformal-cervical papers, and both proposals do **not** do.

**Working title:** *Externally-Validated, Release-Audited Lightweight Cervical Cytology Triage.*

**Data (all real & downloadable):** train on SIPaKMeD + Mendeley LBC + CRIC; **leave-one-dataset-out external test** on the held-out set; use **APCData** for a true 6-class Bethesda evaluation. No WSI dependency.

**Pipeline:**
1. **Backbone/student:** distill a lightweight **EfficientNet-B0** student from your existing B4 teacher (or from frozen UNI/UniCAS cytology features) — **fix the T²/double-softmax bug**, run the **no-teacher ablation**.
2. **Stain robustness (compared, not claimed novel):** Macenko/Reinhard normalization **vs** MixStyle **vs** color-adversarial augmentation — report which actually closes the external gap.
3. **Calibration:** temperature scaling + reliability/ECE (framed as good practice, citing SiCal-CytoNet).
4. **★ The novel contribution — release-side conformal risk audit:** cell/tile-level **Mondrian (class-conditional) conformal prediction** with **explicit prevalence-shift stress tests**, reporting **release-side abnormal-class coverage (FNrel / Prel)**, not just marginal coverage. This operationalizes the arXiv 2605.20956 critique for cervical cytology for the first time — a genuine, citable, buildable contribution.
5. **Measured edge efficiency:** params / FLOPs / latency / peak VRAM of the student on the **RTX 5060 Ti** (the field's unvalidated "lightweight" gap).

**Honest novelty claim (defensible at Q1/Q2):** *"The first cervical-cytology screening study to combine leave-one-dataset-out external validation, measured edge efficiency of a distilled student, and **release-side conformal risk auditing under prevalence shift** — converting the field's documented generalization, calibration, and safety-claim gaps into a rigorously evaluated, deployable cell-level system."* Integration + rigor + the release-side-audit angle = real, and **buildable on your hardware and on data that actually exists.**

**What to drop:** WSI/MIL/slide-level ambitions (no public data), "ASC-US+" naming, any "guaranteed safety" wording, and any claim that calibration/stain-invariance/evidential heads are novel.
