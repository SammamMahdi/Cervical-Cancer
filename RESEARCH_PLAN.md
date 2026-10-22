# Research Plan — Flexible Multimodal Cervical Screening (CO-PRIOR)

*Synthesised from all results to date: the 155-paper systematic review, the leakage-safe tabular
characterization, the implemented fusion, and the adversarial novelty critique. Every number below is
measured in this repository or verified from a primary source.*

---

## 0. The one decision this plan rests on

The adversarial review **rejected** the obvious framing ("we built a flexible image+questionnaire
screener") on three grounds we cannot engineer away: the fusion rule is already published
(ProbEn / opinion pooling), the both-modality path **cannot be validated** (no public dataset pairs a
cervical image with a questionnaire for the same patient), and the two labels are different constructs on
different units. **So the deliverable is not a screening product — it is a method + an honest audit.**

> **Recommended contribution shape (what the paper/thesis actually claims):**
> 1. **CO-PRIOR** — the *two-prior composition operator* (strip each expert's own training prevalence
>    *before* pooling). The single novel piece; validatable *per expert* without paired data.
> 2. **A leakage-audited honest re-benchmark** of the UCI questionnaire — nobody has published one.
> 3. **An impossibility/honesty result** — precisely what a disjoint-cohort screener can and cannot be
>    shown to do, with a pre-registered falsifier.

Everything else (the CNN, the GBDT, the cascade, conformal, evidential conflict) is **deliberately
borrowed and named as such**.

---

## 1. Where we are — asset inventory (what already exists)

| Asset | Location | State |
|---|---|---|
| Systematic review (155 verified papers) | `Literature/`, `Review/` | ✅ done; 0 fabricated; gap analysis + novelty critic |
| Frozen image expert (EfficientNet-B0 student) | `legacy_src/results/checkpoints/student_b0.pt` | ⚠️ trained on a **leaky random split** — must be redone group-aware |
| Frozen tabular experts (5 calibrated heads) | `legacy_src/results/checkpoints/tabular_*.joblib` | ✅ leakage-safe; Biopsy AUROC 0.656 [0.584, 0.720] |
| Tabular characterization (13 figures) | `legacy_src/results/figures/tabular/` | ✅ ceiling ~0.66 proven; leakage demos |
| Fusion implementation | `legacy_src/src/{fusion,run_fusion,eval_fusion}.py` | ✅ cascade + decide + any-subset; runs end-to-end |
| Dependence-sensitivity envelope | `legacy_src/src/eval_fusion.py` | ✅ the honest "does fusion help?" analysis |
| Research proposal (ACM LaTeX, 12 pp, compiles) | `proposal/main.tex` | ✅ CO-PRIOR framing, Bethesda handled, 0 errors |

**Grounded constraints that shape everything:**
- Tabular ceiling **≈ 0.66 AUROC** — data-imposed (max univariate AUROC 0.57, MI < 0.01); no model breaks it.
- Image expert: **0.998 in-distribution (leaky) → 0.73 external (Herlev)** — the external number is the real one.
- **No paired cohort exists** (verified dataset-by-dataset, incl. the decisive UCI 383 vs 384 miss).
- Published UCI 0.98–1.00 = leakage (oversample-before-split **0.994**; `Dx:Cancer`-from-`Dx` **0.998**).

---

## 2. The phased plan

Each task lists **owner** (▶ = code/analysis I can do here · ◆ = your decision/collection),
**effort**, **dependency**, and **deliverable/artifact**.

### Phase 1 — Honest per-expert foundations *(the validatable core; ~3 weeks)*
The one place real, defensible numbers come from. Both experts must be **frozen, calibrated, and reported
with CIs on their own held-out + external data**.

| # | Task | Owner | Effort | Deliverable |
|---|---|---|---|---|
| 1.1 | **Re-train the image expert with a GROUP-AWARE split** (by source cluster/slide, 70/15/15) to kill the leaky-random-split number | ▶ | M | honest in-dist AUROC/PR-AUC/ECE + CIs |
| 1.2 | **External validation** of the image expert on Herlev + HiCervix (Bethesda-collapsed, glandular excluded) + APACC | ▶ | S | cross-dataset transfer matrix (figure) |
| 1.3 | **Calibrate the image expert** (temperature → BCTS/MLLS); reliability diagram + ECE | ▶ | S | calibrated `P_img`, ECE table |
| 1.4 | **Confirm the tabular ceiling** by adding TabPFN + LightGBM + CatBoost to the bake-off | ▶ | S | updated `11_model_bakeoff` (expect ≤ +0.02) |
| 1.5 | Finalise the tabular expert (Biopsy primary; report all 4 + any-of-4) with PR-AUC, sens@spec, ECE, CIs | ▶ | done | per-target metrics table |

**Exit criterion:** two frozen, calibrated experts, each with an honest external number and a reliability
diagram. *This is the strongest, safest result in the whole project.*

### Phase 2 — The CO-PRIOR operator + its per-expert validation *(the novelty; ~3 weeks)*

| # | Task | Owner | Effort | Deliverable |
|---|---|---|---|---|
| 2.1 | Formalise & implement prior-stripping (already in `fusion.prior_shift` / `log_lr`) as the M4 operator | ▶ | done | operator code |
| 2.2 | **RQ2/H-C:** stripped vs raw pooling vs single post-hoc correction — ΔECE across τ ∈ {0.02,0.06,0.10,0.20} | ▶ | M | ΔECE bar + CI |
| 2.3 | **H2c (the key test, needs NO paired data):** re-anchor one expert to a *known* target prevalence on a resampled cohort → does ECE improve? | ▶ | M | per-expert re-anchoring result |
| 2.4 | Ablations: A6 stripping on/off · A7 π mis-spec ±50% · A9 sum vs product · A11 calibration on/off | ▶ | M | ablation grid (table) |

**Exit criterion:** a clean answer to "does prior-stripping-before-pooling help, and is it distinct from a
single post-hoc correction?" — validated per expert, no pairing required. **If A6 shows ΔECE ≈ 0, the
operator collapses to a re-parameterisation → demote to the diagnostic study (Phase 3 + the audit).**

### Phase 3 — Fusion mechanism + the honesty result *(~2 weeks)*

| # | Task | Owner | Effort | Deliverable |
|---|---|---|---|---|
| 3.1 | **Dependence-sensitivity envelope** at the real marginals (done) — the honest "fusion helps?" answer | ▶ | done | `fusion_dependence_envelope` figure |
| 3.2 | **Any-subset degradation** — verify `{img}` path ≡ image-only exactly; confidence monotone across subsets | ▶ | S | confidence-vs-availability plot |
| 3.3 | **Abstention** — per-path conformal coverage + risk–coverage (AURC) curve | ▶ | M | AURC figure, coverage table |
| 3.4 | **Leakage re-benchmark write-up** (honest 0.65 vs 0.994/0.998) — a standalone contribution | ▶ | done | `10_leakage_demo` + section |
| 3.5 | Pre-register the **falsifier (H-E)**: fusion must beat `max(image, tabular)` on discrimination *and* calibration, else report the null | ▶ | S | pre-registration doc |

**Exit criterion:** the "what can and cannot be claimed" section is fully backed by figures.

### Phase 4 — The decisive (optional) experiment: a prospective paired pilot *(◆ your call; long-lead)*

The **only** route to a real joint-accuracy claim. Even ~50–100 patients with *image + questionnaire +
biopsy* enables a DeLong test of fused-vs-best-single.

| # | Task | Owner | Effort | Deliverable |
|---|---|---|---|---|
| 4.1 | Scope ethics/consent + de-identification (the barrier that blocks every public set) | ◆ | L | protocol |
| 4.2 | Collect the pilot cohort (split by patient, test-only) | ◆ | XL | paired test set |
| 4.3 | If collected: fused-vs-`max(single)` with DeLong + CI | ▶ | S | the one legitimate joint result |

**If Phase 4 is out of scope, say so in the paper** — the impossibility becomes the finding, not the gap.

### Phase 5 — Explainability & clinical framing *(~1 week)*

| # | Task | Owner | Effort | Deliverable |
|---|---|---|---|---|
| 5.1 | Grad-CAM on the image expert (nucleus vs background) | ▶ | S | Grad-CAM panels |
| 5.2 | SHAP on the tabular expert (done) + modality attribution (M9: which expert drove the call) | ▶ | done | SHAP + attribution figure |
| 5.3 | Frame the system as a **triage cascade with safety asymmetry** (questionnaire raises, never lowers) | ▶ | S | clinical-semantics section |

### Phase 6 — Write-up & release *(~2 weeks)*

| # | Task | Owner | Effort | Deliverable |
|---|---|---|---|---|
| 6.1 | Turn `proposal/main.tex` into the full paper (drop in Phase 1–3 results tables/figures) | ▶ | M | camera-ready `.pdf` |
| 6.2 | Reproducibility release: code + literature DB + figures + seeds | ▶ | S | tagged repo + README |
| 6.3 | Final honesty pass (claims ↔ evidence; the falsifier reported) | ▶ | S | claim-audit checklist |

---

## 3. Timeline (indicative, ~11 working weeks)

```
Wk  1  2  3  4  5  6  7  8  9 10 11
P1  ███████████                          honest experts (image re-split, calibration, external)
P2        ███████████                    CO-PRIOR operator + ablations
P3              ████████                 mechanism + honesty result
P5                    █████              explainability + clinical framing
P6                       ████████        write-up + release
P4  (long-lead, parallel).............  paired pilot (only if pursued; the sole route to a joint claim)
```

---

## 4. What we will and will not claim (bind this before running anything)

- ✅ **Will:** each expert's own + external performance, with CIs; a leakage-audited honest UCI baseline; a
  calibrated, uncertainty-aware, any-modality *interface* that degrades gracefully and abstains; the
  prior-stripping operator, validated per expert.
- ✅ **Will (only with Phase 4):** fusion beats the best single expert (DeLong on the paired pilot).
- ❌ **Will not:** a joint patient-level accuracy number from the two unpaired public sets; a joint conformal
  guarantee (per-path marginal only); novelty in the fusion rule, calibration-across-subsets, or modality
  dropout (all borrowed and cited); any "99%" produced by oversampling-before-split or `Dx:*`.

---

## 5. Risks & mitigations

| Risk | Evidence | Mitigation |
|---|---|---|
| **Operator adds nothing** (A6 ΔECE≈0) | novelty rests on M4 | pre-registered demotion to the diagnostic-study paper (audit + impossibility still stand) |
| **Image re-split drops the number** | 0.998→0.73 already seen | report external honestly; it is the deployable truth, and the safety-asymmetry protects sensitivity |
| **Tabular stays weak** (~0.66) | ceiling is data-imposed | the tabular arm is *designed* as a weak triage flag, not a diagnostician |
| **No paired cohort** | verified absence | reframe as the finding; cost the pilot |
| **Reviewer: "it's just combining two models"** | the exact kill criterion | novelty centred on M4 only; borrowings declared in the abstract |

---

## 6. Decisions needed from you (blocking Phase 1 kickoff)

1. **Scope of Phase 4** — pursue a prospective paired pilot, or explicitly scope the paper to the
   per-expert + operator + impossibility contribution? *(Recommend: scope it out for v1; note it as future work.)*
2. **Primary target** — Biopsy (histology gold standard, recommended) vs Schiller vs any-of-4.
3. **Venue** — method venue (the operator + audit) vs clinical-informatics venue (the honest screener) —
   changes emphasis, not content.
4. **Green-light Phase 1.1** — re-train the image expert group-aware (the single most important fix; the
   current 0.998 is not usable).

---

### One-paragraph summary
We have two frozen experts (a strong cytology image model, a weak-but-honest risk-factor model), a working
any-modality fusion, a 155-paper evidence base, and an adversarially-verified novelty (prior-stripping
before pooling). The plan is: **(P1)** make both experts honest and calibrated with external validation;
**(P2)** validate the CO-PRIOR operator per expert; **(P3)** report the fusion mechanism and, candidly, the
limits of what disjoint cohorts allow; **(P5–P6)** add explainability and write it up — with an optional
**(P4)** prospective paired pilot as the only route to a joint-accuracy claim. The contribution is a
method and an honest audit, not a screening product, and every borrowed part is named.
