"""Atomically patch the GAN->ablation experiment cells in new_pipeline.ipynb to real/maximal runs.

Matches each target code cell by a UNIQUE substring of its current source and replaces the source
(idempotent: re-running is a no-op once patched). One atomic write; cell outputs are preserved.
"""
import json, os

NB = os.path.join(os.path.dirname(__file__), "..", "..", "new_pipeline.ipynb")

ONE_CLICK = """import run, train
# One-click end-to-end (resumable). Real toggle, not a placeholder: flip to True to run the whole
# pipeline here; leave False to run the per-step cells below (the live path).
RUN_ONE_CLICK = False
if RUN_ONE_CLICK:
    run.eda()                  # all-dataset EDA + leakage-safe split + research framing
    train.main()               # teacher (two-stage) -> distilled student (auto-resumes)
    run.evaluate('student')    # sealed test + cross-dataset E0/E1/E-morph + calibration + efficiency"""

CELLGAN = """import cellgan
# Full conditional-DCGAN training on the TRAIN split only (no test leakage), then top every
# under-represented class up to the majority count for the RQ5 minority-rescue arm.
cellgan.train_cellgan(epochs=300, smoke=False)     # full GAN training (long)
made = cellgan.synthesize_minority()               # real per-class rescue -> results/synthetic/<class>/
print('synthesized per class:', made)
show(os.path.join(FIG,'cellgan_loss'))
for ep_fig in sorted(glob.glob(os.path.join(FIG,'cellgan_samples_ep*.png')))[-1:]:
    show(ep_fig[:-4])"""

SOTA = """import ablation
# Full RQ6 published-SOTA comparison: trains CCanNet, CerviXpert, MaxCerVixT under the IDENTICAL
# leakage-safe protocol as the backbones (each a full training run; long).
sota = ablation.sota_comparison(smoke=False)
print('SOTA rows:', [r['arm'] for r in sota], '-> results/metrics/sota_comparison.json')"""

HPO = """import hpo
# Full search. Image student tuned WITH KD (needs the teacher), then the single best config is refit
# and re-evaluated on the sealed test + external sets (retrain=True). Tabular XGBoost tuned separately.
study = hpo.run_hpo(n_trials=50, max_epochs=25, use_kd=True, retrain=True)   # image (long)
tab   = hpo.run_tabular_hpo(n_trials=100)                                    # XGBoost tabular
print('image  best:', round(study['best_value'], 4), study['best_params'])
print('tabular best:', round(tab['best_value'], 4), tab['best_params'])"""

ABLATION = """import os, ablation, selfsup, config as C
print('arms:', list(ablation.ARMS))   # 18 arms total
# A9_ssl_init needs SSL weights first; pretrain once here (guarded) so EVERY arm actually runs.
if not os.path.exists(os.path.join(C.CKPTDIR, 'ssl_efficientnet_b0.pt')):
    selfsup.pretrain('efficientnet_b0', epochs=200, batch=256)   # SimCLR prerequisite for A9
# Maximum ablation: ALL 18 arms (each a full training run) + the §3 backbone comparison + efficiency.
ablation.run_all()              # -> results/metrics/ablation_table.json + figures/ablation_macro_f1
ablation.model_comparison()     # B0/B2/MNv3/ShuffleNet/ResNet/DenseNet/Swin/MaxViT + efficiency
show(os.path.join(FIG,'ablation_macro_f1'))"""

SSL = """import os, selfsup, config as C
print('SSL pool:', len(selfsup.pool_image_paths()), 'label-free crops (sipakmed+herlev+apacc+hicervix+cric)')
ssl_ck = os.path.join(C.CKPTDIR, 'ssl_efficientnet_b0.pt')
# SimCLR pretraining on the pooled label-free crops (guarded: skips if the ablations cell already made it).
if not os.path.exists(ssl_ck):
    selfsup.pretrain('efficientnet_b0', epochs=200, batch=256)   # long; saves ssl_efficientnet_b0.pt
print('SSL backbone checkpoint:', os.path.basename(ssl_ck), '->', os.path.exists(ssl_ck))"""

# (unique substring that identifies the target cell) -> replacement source
RULES = [
    (lambda s: ("Uncomment to run everything in one go" in s) or ("RUN_ONE_CLICK" in s), ONE_CLICK, "one-click"),
    (lambda s: "train_cellgan" in s, CELLGAN, "cellgan"),
    (lambda s: "sota_comparison" in s, SOTA, "sota"),
    (lambda s: "run_hpo" in s, HPO, "hpo"),
    (lambda s: "list(ablation.ARMS)" in s, ABLATION, "ablation"),
    (lambda s: "pool_image_paths" in s, SSL, "ssl"),
]


def _src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def main():
    nb_path = os.path.abspath(NB)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = []
    used = set()
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = _src(cell)
        for i, (match, new, name) in enumerate(RULES):
            if i in used:
                continue
            if match(src):
                if src.strip() != new.strip():
                    cell["source"] = new
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    changed.append(name)
                used.add(i)
                break

    tmp = nb_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    os.replace(tmp, nb_path)
    matched = [name for i, (_, _, name) in enumerate(RULES) if i in used]
    print("matched cells:", matched)
    print("changed cells:", changed or "(none — already patched)")
    missing = [name for i, (_, _, name) in enumerate(RULES) if i not in used]
    if missing:
        print("WARNING: no cell matched for:", missing)


if __name__ == "__main__":
    main()
