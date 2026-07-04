# Research Prompt — Design a Novel, Gap-Grounded Cervical Cancer Screening AI Pipeline

> Copy everything in the block below into a research-capable AI agent (one with live web search/fetch).
> It is self-contained, but if you can attach `Cervical Cancer Literature Review.xlsx`,
> `LITERATURE_REVIEW.md`, and `PIPELINE_DOCUMENTATION.md`, the agent will be better grounded.

---

```
ROLE
You are a senior machine-learning research scientist specializing in computational pathology and
cervical cancer screening AI. You think like a Q1-journal reviewer: skeptical of saturated
benchmarks, allergic to data leakage, and focused on clinical deployability and generalization.

OBJECTIVE
Design ONE novel, concrete, implementable pipeline methodology for cervical cancer screening whose
novelty is GROUNDED IN DOCUMENTED, UNSOLVED GAPS of the 2023–2026 state of the art — NOT in squeezing
another decimal point of accuracy out of SIPaKMeD/Herlev. The contribution must be defensible to
reviewers as "this closes a real gap that current SOTA does not."

CONTEXT — THE VERIFIED SOTA LANDSCAPE (as of mid-2026; verify and update before you rely on it)
- Single-cell cytology classification is SATURATED: SIPaKMeD 5-class sits at ~98–99.1% and Herlev
  near-ceiling across CNNs, ViTs, and hybrids (e.g. CCanNet 98.53% with ~1.27M params; MaxCerVixT
  99.02%; CICNN/CausalCervixNet 99.14%). These are almost all INTERNAL, same-dataset splits.
- The frontier has moved to FOUNDATION MODELS on whole-slide images with EXTERNAL/PROSPECTIVE
  validation: UniCAS (Cell Reports Medicine, 2026) reports external AUC ~0.90 across six unseen
  scanners; Smart-CCS (2025) reports external AUC ~0.95 and prospective multi-center AUCs. These set
  the new bar = generalization, not internal accuracy.
- KNOWLEDGE DISTILLATION / lightweight models exist (CCanNet, MaxCerVixT, CerviSegNet-DistillPlus)
  but rarely report validated edge efficiency (FLOPs/params/latency on real hardware) or
  student-vs-teacher fidelity.
- FEDERATED LEARNING is emerging (FedDP; federated SIPaKMeD) but collapses under non-IID conditions
  (e.g. ~94% IID -> ~78% non-IID) and seldom couples privacy budgets with heterogeneity testing.
- TABULAR risk-factor models routinely report ~99.99% — a hallmark of resampling-before-splitting
  data leakage.
- CALIBRATION is essentially absent (a rare exception applies temperature scaling); MULTIMODAL fusion
  is brittle (near-1.0 AUC signals joint-distribution overfitting) and assumes complete, co-registered
  modalities rarely available in practice.

GROUNDED GAPS (your novelty must attack a subset of these — these are the documented openings)
1.  External / cross-site validation is the exception, not the default; internal metrics are reported
    as if deployable.
2.  Severe (10–30 pt) performance collapse under scanner / stain / population domain shift, rarely
    measured or mitigated.
3.  Dataset monoculture: clean pre-segmented single cells stand in for messy whole-slide reality
    (overlapping cells, debris, inflammation, rare aggressive subtypes).
4.  Slide-level spatial context & rare-event detection done properly (modern MIL: CLAM/TransMIL/DSMIL
    baselines routinely omitted).
5.  No probability calibration or pre-specified clinical operating points (ASC-US+/HSIL+).
6.  Data leakage + weak benchmarking rigor (no CIs, seeds, shared protocol, public code).
7.  Lightweight/KD claims unvalidated for real edge/low-resource deployment.
8.  Federated / privacy-preserving training untested under realistic non-IID heterogeneity.
9.  Annotation quality, inter-observer variability, and label-noise robustness unaddressed.
10. Untrustworthy synthetic-data (GAN/diffusion) and self-supervised pretraining (fidelity/leakage
    not validated).
11. Brittle multimodal fusion under missing / mismatched modalities.
12. Almost no prospective, reader-in-the-loop, workflow-integrated, or regulatory/equity evaluation.

YOUR STARTING ASSETS (the team's existing, reproducible baseline to build on or supersede)
- A working SIPaKMeD pipeline: EfficientNet-B4 teacher (~97% internal val acc) knowledge-distilled into
  an EfficientNet-B0 student (~97.5% internal val acc), framed as Tier 2 of a two-tier
  (tabular risk + cytology image) screening funnel.
- Known limitations of that baseline (shared with the field): no external/cross-dataset validation,
  no calibration, the validation set doubles as the test set, and edge efficiency
  (FLOPs/latency/params) is claimed but never measured.
- Compute for new work is a SINGLE NVIDIA RTX 5060 Ti (assume the 16 GB GDDR7 / Blackwell variant; if
  the 8 GB variant, halve batch size and use gradient accumulation) — ONE local consumer GPU, NOT a
  multi-GPU cluster. The original baseline was trained on Kaggle 2x Tesla T4, but every proposed method
  MUST train and run on this single 16 GB GPU: use mixed precision (BF16/FP16), gradient accumulation,
  and patch-based (not whole-slide-in-memory) processing where needed; if a foundation model is used,
  rely on frozen features or parameter-efficient fine-tuning (e.g. LoRA/adapters) rather than full
  fine-tuning. The deployment target remains low-resource / edge clinical settings, so this single-GPU
  budget doubles as a realistic proxy for that constraint.

TASKS (do these in order; show your work)
1.  VERIFY & UPDATE. Run live searches; re-confirm the gaps above against the most recent literature
    (through the current date). Add any genuinely new SOTA. Explicitly flag any gap that has since been
    closed, with citations. Do not trust my summary blindly.
2.  SELECT. Choose 2–3 gaps that are HIGH-VALUE and CO-ADDRESSABLE by one coherent system (e.g.
    generalization + calibration + validated efficiency). Justify why these, not others.
3.  PROPOSE ONE PIPELINE. Specify end to end:
    - Clinical problem framing & intended use (screening triage? slide-level? cell-level?).
    - Data plan: training corpus and, critically, a HELD-OUT EXTERNAL test set (name real public
      datasets — SIPaKMeD, Herlev, Mendeley LBC, CRIC, ISBI/CNSeg/Cx22, colposcopy sets, etc. — and
      which is train vs external test). State realistic access constraints.
    - Architecture & data flow (give a step-by-step / box-and-arrow description).
    - Training procedure (losses, objectives, the specific MECHANISM that closes each chosen gap).
    - Why it is NOVEL vs named recent SOTA (UniCAS, Smart-CCS, CCanNet, MaxCerVixT, etc.) — be concrete.
4.  EVALUATION PROTOCOL THAT FIXES THE FIELD'S OWN EVALUATION GAPS:
    - External / leave-one-site-out validation; report the INTERNAL-vs-EXTERNAL delta as the headline.
    - Calibration (reliability diagrams, ECE) and pre-specified operating thresholds with sens/spec.
    - Efficiency: params, FLOPs, memory, latency on accessible field hardware; accuracy-retained-vs-teacher.
    - Statistical rigor: multiple seeds, confidence intervals, significance tests, a standardized taxonomy.
    - Fair, modern baselines (incl. proper MIL and a foundation-model comparison where relevant).
5.  ABLATIONS. Define experiments that isolate the contribution of EACH novel component (including a
    leakage-safe sanity check, e.g. the distillation/SSL/fusion component removed).
6.  FEASIBILITY & RISK. Compute budget on a SINGLE RTX 5060 Ti (16 GB; BF16 mixed precision + gradient
    accumulation; state batch size, resolution, and VRAM headroom); data accessibility; top 3–5 risks
    (e.g. external data unavailable, gap already closed, negative result) and mitigations.
7.  NOVELTY DEFENSE & THREATS TO VALIDITY. State the one-sentence novelty claim, then argue how a
    hostile reviewer would attack it and how the design pre-empts that.

HARD CONSTRAINTS
- GROUNDED & REAL: cite real papers (title + venue + year) and real, named, publicly accessible
  datasets. If a fact, metric, or dataset is uncertain or you could not verify it, SAY SO explicitly —
  never invent citations, numbers, or datasets.
- The novelty must close a documented GAP; a new architecture purely for its own sake is rejected.
- Must train and run on a SINGLE 16 GB consumer GPU (RTX 5060 Ti) and be oriented to low-resource / edge deployment.
- Must define a head-to-head benchmark protocol against current SOTA, not just internal numbers.
- Prefer one deep, well-justified pipeline over a buffet of shallow ideas.

OUTPUT FORMAT
A. Executive summary + a single-sentence novelty statement.
B. Updated gap analysis (what's still open / newly closed, with citations).
C. The proposed pipeline (with a textual architecture diagram and a data-flow walkthrough).
D. A table mapping: chosen gap -> mechanism that closes it -> metric that proves it.
E. Evaluation & ablation plan.
F. Feasibility, risks & mitigations, threats to validity.
G. References (real, verifiable; flag any low-confidence items).
```

---

### How to use / tune this prompt
- **Narrow the angle** if you want a specific direction — replace the open selection in Task 2 with, e.g., *"focus on cross-scanner domain generalization + calibrated, distilled edge inference."*
- **Anchor to your work** — the *Starting Assets* block already points the agent at your EfficientNet B4→B0 two-tier baseline, so it will propose something that extends/beats it rather than starting from scratch.
- **Force comparability** — keep the external-validation and head-to-head-baseline requirements; they are what turn "another high-accuracy model" into a defensible Q1 contribution.

