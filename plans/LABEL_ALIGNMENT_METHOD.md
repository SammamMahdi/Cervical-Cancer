# How to Map Other Datasets INTO SIPaKMeD's 5-Class Morphology Space

Grounded methods review (real techniques + cervical precedents, verified June 2026). Goal: keep SIPaKMeD's 5 cytomorphology classes **fixed**, and project Herlev/CRIC/RIVA/APCData/HiCervix/APACC/Mendeley-LBC/BMT into them.

## The core difficulty (why it isn't a simple lookup table)
You're projecting **diagnosis → morphology**, which is **one-to-many and often undefined**:
- `NILM → {Superficial-Intermediate, Parabasal, Metaplastic}` (a NILM label can't name the maturation type)
- `LSIL → {Koilocytotic, (mild) Dyskeratotic}`
- `HSIL non-keratinizing / ASC-H / glandular(AGC/AIS/adeno) / invasive SCC / Herlev columnar → NO HOME` in SIPaKMeD
- `Parabasal` and `Metaplastic` are labeled by **no other dataset at all**.

So a flat forced crosswalk injects label noise exactly where it matters. The grounded answer is a **hybrid of three routes**, not one.

## The three routes (and the verdict on each)

| Route | What it is | Role | Verdict |
|---|---|---|---|
| **A — Deterministic expert crosswalk** | MSeg-style lookup table; every edge typed EQUIVALENT / BROADER / NARROWER / **NO-MAP** | Semantic backbone + the explicit "discard" gate | **Necessary but insufficient alone** — can't invent Parabasal/Metaplastic, can't place HSIL/glandular |
| **B — Model pseudo-labeling + open-set reject** | SIPaKMeD-teacher labels foreign cells; OpenMatch / Mahalanobis / energy rejects no-home cells | The engine that **grows** data and the only way to positively populate orphan classes from foreign cells | **Powerful but biased** — must be gated by A's allowed edges + stain normalization |
| **C — Partial-label / multi-head training** | Marginal+exclusion loss, PRODEN/PiCO, hierarchical conditional softmax, per-dataset heads | Lets coarse labels contribute gradient **without forcing a leaf** | **The right training formalism** — pairs naturally with B |

They're complementary: **A supplies the clinically-vetted candidate sets and no-map list → C consumes them as a partial-label loss → B disambiguates within candidate sets and grows the orphan classes**, with SSL pretraining underneath.

## Recommended pipeline (hybrid A→C→B)
0. **SSL pretrain, label-free** on *all* cervical crops pooled (no unified labels needed) — CytoFM/UniCAS/Deep-Cervix style. **Stain-normalize (Macenko/Reinhard or MixStyle) first** so the later OOD gate detects *semantic* novelty, not scanner identity.
1. **Author the crosswalk (Route A)** following the WHO-FIC/SNOMED crosswalk protocol: type each edge; anchor the clean ones — `Herlev superficial+intermediate = EQUIVALENT → Superficial-Intermediate`; `LSIL / CIN1 mild-dysplasia = BROADER → Koilocytotic` (the single clean bridge); `NILM = BROADER → {SI, Parabasal, Metaplastic} candidate set`; `ASC-H / HSIL-NK / AGC / SCC / Herlev columnar = NO-MAP`. **Do not route any single Bethesda class into Dyskeratotic** (it's a non-TBS descriptor). Output **candidate sets, not hard labels**. Two cytopathologists code independently; track kappa.
2. **Train with a partial-label loss (Route C):** shared encoder + 5-logit SIPaKMeD head (+ optional native heads for representation benefit). SIPaKMeD cells → exact 5-way CE; a NILM cell supervises `P(SI)+P(Parabasal)+P(Metaplastic)` as one **marginal mass** (no leaf penalized); an LSIL cell supervises `P(Koilo)+P(Dysk)`. Optionally wrap as PRODEN/PiCO or a hierarchical conditional softmax (`root→{NILM, SIL, out-of-space}`).
3. **Pseudo-label + reject (Route B):** use the Step-2 model as a teacher, **constrained by the source diagnostic label** (a CRIC-LSIL cell can only become Koilo/Dysk). Run an **open-set/OOD gate** (OpenMatch / Mahalanobis / energy) to send HSIL-NK/glandular/SCC to an out-of-space bin; keep only **per-class adaptive-threshold** (FreeMatch) pseudo-labels; iterate Noisy-Student with a **rising** threshold — never without a manually-verified target-domain validation set.
4. **Image-level FOVs → cells:** point-labeled FOVs (CRIC, RIVA, APCData) → fixed-radius nucleus patches with the point label as a partial constraint; genuinely image-level sets (BMT, Mendeley-LBC, APACC tiles) → class-agnostic cell detector + attention-MIL/CLAM top-k attribution (inherit the image label only for top-attention cells). *(No public cervical WSI exists → this is FOV-level attribution, not slide-level MIL.)*
5. **Orphan-class rescue (Parabasal, Metaplastic):** SIPaKMeD stays the sole positive source; under marginal loss their logits aren't forced negative; recover candidates via **prototype-seeded coarse-to-fine discovery** against SIPaKMeD exemplars + confidence-gated teacher labels; **inspect HiCervix leaves** (the only realistic cell-level donor); rebalance with **conditional generative synthesis** (CellGAN) incl. stain-translation. Validate gains on **real** held-out cells.
6. **Freeze + audit** (validation plan below). Document the space honestly as **NILM/LSIL-centric squamous morphology** that **cannot detect high-grade squamous or any glandular lesion** — a safety statement.

## Per-class population strategy
| SIPaKMeD class | How it gets populated |
|---|---|
| **Superficial-Intermediate** | Well-supplied: EQUIVALENT from Herlev superficial+intermediate; the normal slice of every NILM source (after Step 2/3 disambiguation) |
| **Koilocytotic** | **The one clean bridge** — BROADER inflow from every LSIL / CIN1. Read strictly as LSIL/HPV; do **not** let it absorb HSIL |
| **Dyskeratotic** | **Riskiest** — SIPaKMeD's own labels + keratinizing phenotype detected by the teacher *within LSIL-constrained cells*; never auto-map a Bethesda class here; rebalance via CellGAN |
| **Parabasal** | **Orphan** — SIPaKMeD only + marginal-loss non-penalty + coarse-to-fine discovery + HiCervix leaf check + synthesis |
| **Metaplastic** | **Orphan** — same stack; even harder (SIPaKMeD-internal distinction; confuses experts) |
| *No-home cells (HSIL-NK, ASC-H, glandular, SCC, columnar)* | Explicit **out-of-space / reject** node (Route A NO-MAP + Route B OOD); optionally give the bin structure via Generalized Category Discovery |

## Validation plan
1. **Crosswalk validation (pre-training):** 2+ cytopathologists assign target/NO-MAP per source label *and* per sampled cropped cell; blinded double-read + consensus; report per-source-label **Cohen/Fleiss kappa** (expect high on Herlev superficial/intermediate, low on HSIL-NK/ASC-US/glandular; *no* kappa for Parabasal/Metaplastic — report that absence). STAPLE/Dawid-Skene for soft consensus.
2. **Confusion-matrix refinement under Leave-One-Dataset-Out** — confirm benign confusions (Koilo↔SI, Dysk↔Koilo); a mapped HSIL spread across multiple classes **confirms it has no home → exclude**. Stain-normalize first to separate model error from mapping error.
3. **Mapping-as-hyperparameter sensitivity sweep** (the single most important check): rerun under each defensible variant (`LSIL→Koilo` vs `→{Koilo,Dysk}` vs excluded; `HSIL→no-map` vs `→Dysk`; `ASC-US→SI` vs `no-map`); report the metric spread, **always with and without the no-map classes** (excluding hard classes inflates accuracy).
4. **Prevalence sanity** — χ²/KS on source-vs-mapped histograms; confirm SI/NILM stays dominant and Parabasal/Metaplastic receive ~zero foreign contributions (expected structural zeros).
5. **Pseudo-label audit** — cytopathologist spot-review; PLL model selection on a manually-verified **target-domain** validation set (per PLENCH).
6. **OOD-gate calibration** on domain-matched, stain-normalized data — verify it flags semantic novelty, not scanner identity.

## Honest limits (state these in any paper)
- **Parabasal & Metaplastic can never get positive supervision from non-SIPaKMeD data** — structural label-space mismatch; their ground truth stays SIPaKMeD alone (~4,049 cells, both minority).
- **Dyskeratotic** is a non-TBS descriptor spanning benign parakeratosis → keratinizing SCC; no diagnostic label routes into it cleanly.
- The result **cannot detect HSIL non-keratinizing, glandular lesions, ASC-H, or invasive SCC** — a screening-safety limitation, not a footnote.
- **Open-set rejection is confounded by domain shift** (stain-normalize or it just detects "not-SIPaKMeD-scanner").
- **Pseudo-labeling carries confirmation bias** (a 5-class teacher confidently mislabels foreign cells).
- **Almost none of this machinery has a cervical-cytology precedent** — applying it to SIPaKMeD projection would be *novel but unvalidated* (HiCervix+HierSwin and Coskun 2026 are the only close exemplars; Coskun maps the *opposite* direction, into Bethesda).
- **Generative augmentation creates no new clinical information** — validate on real held-out cells only.
- Validation proves label **consistency**, not biological **correctness** (two experts can share a wrong bias).

## Key methods / citations to use
- **Crosswalk:** MSeg (Lambert, CVPR 2020); WHO-FIC/SNOMED crosswalk protocol (Vikström, BMC MIDM 2007); Coskun et al., *Bioengineering* 2026 13(3):289 (closest cervical precedent — reverse direction; read its repo before citing a table).
- **Clinical mapping:** Nayar & Wilbur, *The Bethesda System* 3rd ed. (Springer 2015); Pangarkar, *CytoJournal* 2022;19:28 (koilocyte = LSIL); CytoJournal PMC8326095 (dyskeratosis not in TBS; ASC indeterminate).
- **Partial-label / hierarchy:** Shi et al. marginal+exclusion loss (*MedIA* 2021); PRODEN (ICML 2020), PiCO (ICLR 2022), PLENCH (ICLR 2025); HXE/WordTree (Bertinetto CVPR 2020; Redmon CVPR 2017); **HiCervix + HierSwin (IEEE TMI 2024)** — direct cervical precedent + key cell-level donor.
- **Pseudo-label + open-set:** OpenMatch (NeurIPS 2021), FixMatch (NeurIPS 2020), FreeMatch (ICLR 2023); Mahalanobis OOD (NeurIPS 2018), Energy OOD (NeurIPS 2020), OpenMax (CVPR 2016); Noisy Student (CVPR 2020); confirmation-bias (Arazo, IJCNN 2020).
- **Image→cell MIL:** ABMIL (ICML 2018), CLAM (Nat. Biomed. Eng. 2021), STRIDE-AI (arXiv 2407.19512).
- **Synthesis / SSL / discovery:** CellGAN (MICCAI 2023); CytoFM (CVPR-W 2025), UniCAS (Cell Rep. Med. 2025); Generalized Category Discovery (CVPR 2022).
- **Multi-head:** UniDet/Simple Multi-Dataset Detection (CVPR 2022); LENS partial-label (IEEE TMI 2021).

---
*Bottom line: it can be done — as a hybrid of an expert NO-MAP crosswalk (A) + partial-label training (C) + teacher pseudo-labeling with open-set rejection (B), over an SSL-pretrained, stain-normalized backbone. But it is intrinsically a **partial, NILM/LSIL-centric projection**: two classes stay SIPaKMeD-only and all high-grade/glandular signal is, by clinical fact, unmappable.*
