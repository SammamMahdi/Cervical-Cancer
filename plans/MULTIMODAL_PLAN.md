# Multimodal Plan — Using Numerical Data Alongside Image Data

Grounded answer to "can the model use numerical data as well as image data?" (verified against primary sources, June 2026). Companion to [DATA_PLAN.md](DATA_PLAN.md) and [PIPELINE_DOCUMENTATION.md](../PIPELINE_DOCUMENTATION.md).

## The key correction first
Multimodal fusion needs **both modalities for the *same* instance**. So:
- ✅ You **can** fuse image + numerical features **within one dataset** (the numbers describe the same cell as the image).
- ❌ You **cannot** fuse **Herlev's numbers with SIPaKMeD's images** — they're different, unpaired cells, with **different feature schemas** (Herlev 20-D vs SIPaKMeD ~52-D) and **different label spaces** (7-class vs 5-class).
- ❌ You **cannot** fuse **UCI risk-factor data** with either — it's *patient-level*, different patients, no images, no shared key (fusing it would be an invalid join / data-leakage red flag).

## What "numerical data" actually exists (verified)
| Source | Numerical data | Dim | Paired to image? |
|---|---|---|---|
| **SIPaKMeD** | `Features_CELL.7z` (download **separately** from the UoI site — not in Kaggle mirrors): intensity (avg intensity, contrast) + texture (smoothness, uniformity, 3rd moment, entropy) ×3 RGB channels + shape (area, axes, eccentricity, orientation, equiv-diameter, solidity, extent) = **26 features**, computed for **nucleus AND cytoplasm** ≈ **52 per cell** | ~52 | ✅ via `image#`+`cell#` join keys |
| **Herlev** | Jantzen-2005 feature table (ARFF/CSV): nucleus area, cytoplasm area, **N/C ratio**, nucleus/cytoplasm brightness, short/long diameters, elongation, roundness, perimeter, position, intensity maxima/minima = **20 features** | 20 | ✅ 1:1 row↔cell |
| **UCI Risk Factors** | 858 patients × ~32 demographic/behavioural/HPV attributes | ~32 | ❌ patient-level, no images |

These are **clinically interpretable** (N/C ratio, nuclear area, chromatin-texture proxies) — exactly the morphology that distinguishes the SIPaKMeD classes — and they're tiny, so a tabular branch adds **near-zero VRAM** on your RTX 5060 Ti.

## What you can actually do (3 fusion options, ranked for your situation)
Your model overfits at ~65% val on a small dataset — that pushes you toward the **low-parameter, overfit-resistant** end.

1. **Late / decision fusion — recommended first move.** Keep your image CNN → 5-class probabilities; train a **GBDT (XGBoost/CatBoost) or TabPFN** on the per-cell features → 5-class probabilities; **stack both probability vectors into a tiny logistic-regression / RF meta-learner.** Few parameters → barely overfits; each branch tuned independently; trivial compute. Best fit for small data per the medical-fusion reviews.
2. **Early / feature fusion.** Project the ~52-D (SIPaKMeD) feature vector through a small MLP (e.g. 52→32) to balance magnitude against the B0 embedding (1280-D) or B4 (1792-D), **concatenate**, then dropout + linear head. Strong baseline, but **must dim-balance** (a raw 30–52-D vector concatenated to a ~1300-D embedding gets ignored) — use PCA/projection on the image side.
3. **Joint / intermediate fusion (FiLM / DAFT / GMU / cross-attention).** A tabular MLP modulates the CNN's feature channels (**DAFT** is the medical FiLM variant, public PyTorch). Most expressive, **most overfit-prone** — needs end-to-end training, re-exposing the 65% overfitting. Treat as a *later* experiment, not the first.

**Tabular encoder choice:** GBDT/XGBoost or **TabPFN** (a foundation model for ≤10k-row tables, no tuning) — **not** TabNet/FT-Transformer (overkill, overfit at this N).

## Does it actually help? (grounded expectation)
Yes, but **modestly** — credible cervical ablations show **+3 to +5%**, not 30:
- Frontiers in Oncology 2025: 71 radiomics + **EfficientNet-B4** features → fused **94.6%** vs image-only baselines of **~85%** (end-to-end) to **~90%** (feature-based) — a several-point gain, with a clean ablation and external validation.
- Kupas & Harangi (IEEE EMBC 2022): handcrafted features added **+3.2% F1** to a CNN on Pap-smear cells.
- MDHDN (Biomed. Signal Process. Control 2022): decision-level handcrafted+deep fusion, 98.7% on Herlev.
- **The gain is largest exactly when the CNN is data-starved/overfitting — your situation** — because interpretable numeric priors (N/C ratio, nuclear area) act as a **regularizer**. *Caveat:* the features are derived **from the image**, so they're partly redundant with what a well-trained CNN learns; fusion **complements** but won't *replace* fixing the overfitting (augmentation, the cluster-grouped split, LR scheduler).

## Bonus: it gives you the explainability you asked about
The same per-cell features feed directly into **"classified X because N/C ratio high, nuclear area large, chromatin texture irregular"** via SHAP on the tabular branch — turning the fusion model into an interpretable one (see the explainability discussion).

## The UCI two-tier option (separate, not fused)
No published work fuses UCI risk factors with cytology images, and you can't here (patient-level vs cell-level; no shared cohort; SIPaKMeD/Herlev have no patient IDs). The *valid* design is a **decision funnel / cascade**, not feature fusion: a standalone **Tier-1 tabular risk model** (RF/LR/XGBoost on UCI) triages patients → **Tier-2 cytology model** diagnoses. To *fuse* at patient level you'd first **aggregate cell predictions to one patient vector** (mean/max/top-k or attention-MIL), then late-fuse the risk vector — but that needs a cohort with **both** modalities per patient, which no public set provides. Keep UCI as a separate deliverable.

## Recommended path for your setup
1. **SIPaKMeD image branch** (your existing B0 student) + **late fusion** with a **GBDT/TabPFN on the shipped ~52 per-cell features** via a small meta-learner. Cheapest, most overfit-resistant, gives interpretability.
2. **Critical prerequisite:** download `Features_CELL.7z` from the **original UoI site** (Kaggle mirrors omit it), and join features→crops via the `image#`/`cell#` keys; **split by source-cluster image** (not per-cell) so fusion isn't evaluated on leaked data.
3. Only after that, try early-fusion (dim-balanced concat) or DAFT/FiLM as ablation arms.
4. Report fusion's marginal contribution with a **clean ablation** (image-only vs tabular-only vs fused) on the **cluster-grouped + external** split — and don't expect more than a few points.

## Honest caveats
- Fusion is **within-dataset only** (SIPaKMeD-image + SIPaKMeD-features, or Herlev+Herlev) — never cross-dataset, never UCI+image.
- Features are **image-derived → partly redundant**; expected gain is small and is mostly **regularization + interpretability**, not a headline accuracy jump.
- The two feature schemas differ — to use SIPaKMeD's image model with Herlev-style features you'd have to **recompute** equivalent features from SIPaKMeD crops, not reuse Herlev's table.
- Joint/attention/tensor fusion is **data-hungry** and will likely overfit at ~4k cells — prefer late/early fusion first.
- Shape features are in raw pixel units (segmentation/magnification-dependent) and intensity is uncalibrated across slides — **standardize/normalize per feature** before an MLP.

## Key references
- Huang et al., *Fusion of medical imaging and EHR using DL: a systematic review and implementation guidelines*, npj Digital Medicine 2020 (early/late preferred for small data).
- Pölsterl et al., **DAFT** (medical FiLM), MICCAI 2021 / NeuroImage 2022 (`github.com/ai-med/DAFT`); Perez et al., FiLM, AAAI 2018.
- Hollmann et al., **TabPFN**, Nature 2025; Grinsztajn et al., *Why tree-based models still outperform DL on tabular data*, NeurIPS 2022.
- Frontiers in Oncology 2025, *Hybrid feature fusion in cervical cancer cytology* (radiomics + EfficientNet-B4: fused ~94.6% vs image-only ~85–90%); Kupas & Harangi, IEEE EMBC 2022 (+3.2% F1); MDHDN, Biomed. Signal Process. Control 2022 (Herlev).
- Plissiti et al., **SIPaKMeD**, IEEE ICIP 2018 (feature + image dataset; the 26×2 features); Jantzen et al., **Herlev** Pap-smear benchmark, NiSIS 2005 (the 20 features).
