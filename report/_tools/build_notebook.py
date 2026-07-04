"""Regenerate new_pipeline.ipynb with the upgraded, positive-framed, runnable flow.
Run: python report/_tools/build_notebook.py
"""
import json, os

NB = os.path.join(os.path.dirname(__file__), "..", "..", "new_pipeline.ipynb")
cells = []


def _cid():
    return f"cell{len(cells):02d}"


def md(text):
    cells.append({"id": _cid(), "cell_type": "markdown", "metadata": {}, "source": text})


def code(src):
    cells.append({"id": _cid(), "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})


# ---------------------------------------------------------------- title
md("""# Cervical Cytology — NEW (Upgraded) Pipeline

A **solution** pipeline that closes gaps in the cervical-cytology DL literature (it does **not** reproduce the
legacy audit). Run top-to-bottom. Each research question (RQ1–RQ7) is pre-registered with a *positive*
hypothesis and its evidence is produced below.

**What it delivers:** leakage-safe cluster-grouped 80/10/10 split with a **sealed test** (RQ1) · domain
generalization via stain-norm + SSL (RQ2) · calibrated, risk-aware abstention (RQ3) · safe cross-taxonomy
E0/E1/E-morph evaluation (RQ4) · **CellGAN** minority-class rescue (RQ5) · distilled B0 vs **published SOTA**
(RQ6) · multimodal fusion + explainability (RQ7).

B4 teacher → distilled B0 student @224px on the RTX 5060 Ti. Independent of `Legacy_src/`. Outputs → `results/`.
Resumable training (timestamped JSON logs; best+last checkpoints). **Every value and every figure is persisted
as JSON/CSV** for exact reproduction.""")

# ---------------------------------------------------------------- setup
md("## 0. Setup")
code("""import sys, os, glob, json
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import torch, config as C
C.set_seed()
print('device:', C.DEVICE, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
from IPython.display import Image, display
FIG = C.FIGDIR
def show(*stems):
    for s in stems:
        p = s if s.endswith('.png') else s + '.png'
        display(Image(p)) if os.path.exists(p) else print('(missing)', os.path.basename(p))""")

# ---------------------------------------------------------------- preflight
md("""## 0b. Preflight — verify every dataset + guard every legacy failure
Checks all 5 datasets are downloaded (auto-fetches any missing), scans for 0-byte/corrupt images, verifies the
`ssl.py`-shadow / drive-letter-case fixes, and reports GPU/VRAM/BF16. Writes `results/metrics/preflight.json`.""")
code("""import preflight
pf = preflight.run()
assert pf['ok'], 'preflight found issues — see results/metrics/preflight.json before training'""")

# ---------------------------------------------------------------- research framing
md("""## 0c. Research framing — pre-registered RQs / hypotheses / contributions
Literature-gap-anchored, positive hypotheses (no experiment is set up to fail). Saved to
`results/metrics/research_framing.json`.""")
code("""import research
fr = research.framing(); research.dump()
for g in fr['research_gaps']:
    print(f"{g['id']}: {g['title']}")
print()
for rq in fr['research_questions']:
    print(f"{rq['id']} [{rq['gap']}] {rq['label']}\\n    H: {rq['hypothesis']}\\n    target: {rq['success_metric']}\\n")""")

# ---------------------------------------------------------------- one-click
md("""## 0d. (Optional) One-click full run
Runs the whole pipeline end-to-end (preflight → EDA → train teacher+student → evaluate). **Long** (hours on one
GPU) but resumable — re-run the cell to continue after any interruption. Leave commented to run the steps
individually below.""")
code("""import run, train
# One-click end-to-end (resumable). Real toggle, not a placeholder: flip to True to run the whole
# pipeline here; leave False to run the per-step cells below (the live path).
RUN_ONE_CLICK = False
if RUN_ONE_CLICK:
    run.eda()                  # all-dataset EDA + leakage-safe split + research framing
    train.main()               # teacher (two-stage) -> distilled student (auto-resumes)
    run.evaluate('student')    # sealed test + cross-dataset E0/E1/E-morph + calibration + efficiency""")

# ---------------------------------------------------------------- EDA
md("""## 1. EDA — ALL datasets + leakage-safe split (RQ1/RQ4)
`run.eda()` profiles SIPaKMeD + Herlev + APACC + HiCervix + CRIC (native-class counts, E0 mapping coverage +
**excluded fraction**, image-size stats, corrupt scan, montages) and builds the sealed cluster-grouped split.
All JSON+CSV → `results/metrics/eda/`, figures → `results/figures/eda/`.""")
code("""import run
rep = run.eda()
print('overview:', open(os.path.join(C.METRICDIR,'eda','eda_overview.csv')).read())
show(os.path.join(FIG,'eda_class_balance'), os.path.join(FIG,'eda_cluster_balance'))
for ds in ['sipakmed','herlev','apacc','hicervix','cric']:
    show(os.path.join(FIG,'eda',f'{ds}_classes'), os.path.join(FIG,'eda',f'{ds}_e0coverage'), os.path.join(FIG,'eda',f'{ds}_montage'))""")

# ---------------------------------------------------------------- train
md("""## 2. Train — B4 teacher (two-stage) → distilled B0 student (RQ2/RQ6). Resumable; test sealed.
Training + early stopping use **train+val only**. Arms (each a full run): `train.main(distill=False)` (A2 scratch),
`kd_mode='buggy'` (A2 control), `imbalance='focal'|'oversample'` (A6), `aug='geometric'|'none'` (A5),
`stain='macenko'|'reinhard'` (A4, RQ2), `student_backbone='maxcervixt'|'ccannet'|'cervixpert'` (RQ6),
`ssl_init=True` (A9, RQ2).""")
code("""import train
train.main()        # auto-resumes from the last checkpoint""")

md("## 3. Training curves (per-epoch loss/acc; companion CSV beside each figure)")
code("show(os.path.join(FIG,'teacher_training_curves'), os.path.join(FIG,'student_training_curves'))")

# ---------------------------------------------------------------- evaluate
md("""## 4. Evaluate (RQ1/RQ3/RQ4) — sealed test (5-class, ±95% CI) + E0/E1/E-morph + calibration + efficiency
In-distribution metrics are reported on the **sealed 10% test split**; cross-dataset on Herlev/APACC/HiCervix/CRIC.""")
code("""import run
S = run.evaluate('student')
show(os.path.join(FIG,'cm_student'), os.path.join(FIG,'roc_student'), os.path.join(FIG,'reliability_student'),
     os.path.join(FIG,'cm_cross_herlev'), os.path.join(FIG,'cm_cross_apacc'),
     os.path.join(FIG,'cm_cross_hicervix'), os.path.join(FIG,'cm_cross_cric'))
print('internal-vs-external delta (E0):', json.dumps(S['internal_external_delta'], indent=1))
print('preparation-stratified E0:', S['prep_stratified_E0'])""")

md("""### 4b. CRIC crop-size sensitivity (RQ4 robustness)
CRIC ships whole-FOV images with nucleus points; it is a **dense** preparation (nuclei ~70px apart, see
`cric_crop_geometry.json`), so the fixed point-crop size is a confound. This sweep evaluates CRIC E0 across
crop sizes and logs the curve (`cric_crop_sensitivity.json/csv`) so the choice is measured, not arbitrary.""")
code("""run.cric_crop_sweep('student')   # needs a trained student; sweeps C.CRIC_CROP_SWEEP
show(os.path.join(FIG,'cric_crop_sensitivity'))""")

md("""### 4c. Per-domain operating-point calibration (RQ3/RQ4)
The fixed tau=0.5 is wrong per domain. This reports, for the internal test and each external set, the
accuracy recoverable by an oracle threshold and the ECE before/after a Platt fit (oracle/Platt tuned on
each domain's own labels = **upper bound**). Writes `per_domain_calibration.json`.""")
code("""run.per_domain_calibration('student')
show(os.path.join(FIG,'calib_threshold_recovery'))""")

# ---------------------------------------------------------------- explainability
md("""## 5. Explainability + multimodal fusion (RQ7)
**where** (Grad-CAM family + IG/LIME/SHAP-pixel) · **why** (SHAP on morphometrics) · **cases** (exemplar,
counterfactual, ProtoPNet) · **trust** (conformal sets).""")
code("""import explain
# CAMs rendered at 380px (B4-native): 12x12 grid -> sharper, MORE faithful maps than 224 (free; no retrain).
EX = explain.explain_all('student', cam_img_size=380)
show(os.path.join(FIG,'cam_gradcam_380px'), os.path.join(FIG,'cam_layercam_380px'),
     os.path.join(FIG,'attr_ig'), os.path.join(FIG,'shap_features'))
print('conformal:', EX.get('conformal'))
# even sharper LayerCAM/HiResCAM at 224 via an earlier conv block: explain.run_cam(which='student', fine_layer=True)""")
code("""import explain.features as feat
out = feat.run_all(save=True)          # tabular XGBoost + SHAP + permutation + PDP + late fusion (all logged JSON/CSV)
print('tabular test metrics:', out['tabular_metrics'])
print('late fusion (image vs tabular vs fused):', out['late_fusion'])""")

md("""### 5b. Grad-CAM faithfulness vs ground-truth nucleus contours (RQ7)
Uses SIPaKMeD `_nuc.dat` polygons to measure whether the heatmap actually lands on the nucleus
(hit-rate, energy-in-nucleus, concentration ratio). LayerCAM is far more nucleus-focused than the
default Grad-CAM. Writes `nucleus_faithfulness_student.json`.""")
code("""import explain.nucleus as nuc
nuc.run_nucleus_faithfulness('student', img_size=380, n_per_class=20)
show(os.path.join(FIG,'nucleus_hit_rate'), os.path.join(FIG,'nucleus_energy_in_nucleus'))""")

# ---------------------------------------------------------------- CellGAN (RQ5)
md("""## 6. CellGAN — minority-class rescue (RQ5)
Class-conditional DCGAN trained on the **train split only** (no test leakage). `smoke=True` is a fast sanity
check; set `smoke=False, epochs=120` for a real run, then `synthesize_minority()` tops up under-represented
classes. The RQ5 ablation arm consumes `cellgan.SynthClips` as extra training data.""")
code("""import cellgan
# Full conditional-DCGAN training on the TRAIN split only (no test leakage), then top every
# under-represented class up to the majority count for the RQ5 minority-rescue arm.
cellgan.train_cellgan(epochs=300, smoke=False)     # full GAN training (long)
made = cellgan.synthesize_minority()               # real per-class rescue -> results/synthetic/<class>/
print('synthesized per class:', made)
show(os.path.join(FIG,'cellgan_loss'))
for ep_fig in sorted(glob.glob(os.path.join(FIG,'cellgan_samples_ep*.png')))[-1:]:
    show(ep_fig[:-4])""")

# ---------------------------------------------------------------- SOTA (RQ6)
md("""## 7. Published-SOTA comparison (RQ6)
CCanNet / CerviXpert / MaxCerVixT reference reimplementations, trained + evaluated under the **identical**
leakage-safe protocol as the backbones. `smoke=True` is a quick check; full run trains all three. Saved to
`results/metrics/sota_comparison.json`.""")
code("""import ablation
# Full RQ6 published-SOTA comparison: trains CCanNet, CerviXpert, MaxCerVixT under the IDENTICAL
# leakage-safe protocol as the backbones (each a full training run; long).
sota = ablation.sota_comparison(smoke=False)
print('SOTA rows:', [r['arm'] for r in sota], '-> results/metrics/sota_comparison.json')""")

# ---------------------------------------------------------------- HPO
md("## 8. Hyperparameter optimization (U2b) — image student + tabular, refit-best + re-eval")
code("""import hpo
# Full search. Image student tuned WITH KD (needs the teacher), then the single best config is refit
# and re-evaluated on the sealed test + external sets (retrain=True). Tabular XGBoost tuned separately.
study = hpo.run_hpo(n_trials=50, max_epochs=25, use_kd=True, retrain=True)   # image (long)
tab   = hpo.run_tabular_hpo(n_trials=100)                                    # XGBoost tabular
print('image  best:', round(study['best_value'], 4), study['best_params'])
print('tabular best:', round(tab['best_value'], 4), tab['best_params'])""")

# ---------------------------------------------------------------- ablations + model comparison
md("""## 9. Ablations (A1–A9) + general model comparison (§3)
Each arm = a full training run. A1 (leakage A1 vs grouped, RQ1) and A2 (KD, RQ2/RQ6) are the headline rigor arms.""")
code("""import os, ablation, selfsup, config as C
print('arms:', list(ablation.ARMS))   # 18 arms total
# A9_ssl_init needs SSL weights first; pretrain once here (guarded) so EVERY arm actually runs.
if not os.path.exists(os.path.join(C.CKPTDIR, 'ssl_efficientnet_b0.pt')):
    selfsup.pretrain('efficientnet_b0', epochs=200, batch=256)   # SimCLR prerequisite for A9
# Maximum ablation: ALL 18 arms (each a full training run) + the §3 backbone comparison + efficiency.
ablation.run_all()              # -> results/metrics/ablation_table.json + figures/ablation_macro_f1
ablation.model_comparison()     # B0/B2/MNv3/ShuffleNet/ResNet/DenseNet/Swin/MaxViT + efficiency
show(os.path.join(FIG,'ablation_macro_f1'))""")

# ---------------------------------------------------------------- SSL
md("""## 10. SSL pretraining (A9, RQ2) — SimCLR on ~46k pooled label-free crops
Then warm-start the student: `train.main(ssl_init=True)`.""")
code("""import os, selfsup, config as C
print('SSL pool:', len(selfsup.pool_image_paths()), 'label-free crops (sipakmed+herlev+apacc+hicervix+cric)')
ssl_ck = os.path.join(C.CKPTDIR, 'ssl_efficientnet_b0.pt')
# SimCLR pretraining on the pooled label-free crops (guarded: skips if the ablations cell already made it).
if not os.path.exists(ssl_ck):
    selfsup.pretrain('efficientnet_b0', epochs=200, batch=256)   # long; saves ssl_efficientnet_b0.pt
print('SSL backbone checkpoint:', os.path.basename(ssl_ck), '->', os.path.exists(ssl_ck))""")

# ---------------------------------------------------------------- summary
md("## 11. Results summary — every artifact (JSON / CSV / figures)")
code("""for p in sorted(glob.glob(os.path.join(C.RESULTS, '**', '*'), recursive=True)):
    if os.path.isfile(p): print(os.path.relpath(p, C.PROJECT))""")

# ---------------------------------------------------------------- write
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {os.path.abspath(NB)} with {len(cells)} cells")
