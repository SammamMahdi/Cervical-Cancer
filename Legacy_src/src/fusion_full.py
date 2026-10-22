"""Full result suite for the FUSED model, using the new (SMOTE/ExtraTrees) tabular expert.

Produces well-labelled figures for the paper's fused-model section:
  fusx_marginals   per-expert + fused AUROC (bar)
  fusx_roc         fused ROC vs each single expert
  fusx_coprior     CO-PRIOR calibration (ECE vs deployment prevalence, with vs without re-anchoring)
  fusx_coverage    split-conformal coverage per expert (empirical vs target)
Metrics -> metrics/fusion_full.json.  Run AFTER nhanes_tabular_full.py (which refreshes the tabular OOF cache).
Run:  python fusion_full.py
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import fusion_experiments as FE
FIG, MET = FE.FIG, FE.MET
DEPLOY = FE.DEPLOY_PREV
INK, GRN, RED, BLU = "#1a1a1a", "#1a6", "#c0392b", "#2c5aa0"


def _save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, name + "." + ext), bbox_inches="tight")
    plt.close(fig)


def main():
    pi, yi = FE.image_expert("indist")            # image expert scores + labels
    pt, yt, prt = FE.tabular_expert()             # NEW tabular expert (from refreshed cache)
    pri = float(yi.mean())
    img_sc, tab_sc = FE._by_class(pi, yi), FE._by_class(pt, yt)

    # ---- simulate a deployment cohort (near-independent, optimistic) and fuse ----
    y, si, st = FE.simulate(img_sc, tab_sc, DEPLOY, rho=0.0)
    ai = FE.anchor(si, pri, DEPLOY); at = FE.anchor(st, prt, DEPLOY)
    pf = FE.fuse(si, st, pri, prt, DEPLOY, "cascade")
    au_img, au_tab, au_fused = roc_auc_score(y, ai), roc_auc_score(y, at), roc_auc_score(y, pf)
    print(f"image={au_img:.3f} tabular={au_tab:.3f} fused={au_fused:.3f}")

    out = dict(auroc_image=float(au_img), auroc_tabular=float(au_tab), auroc_fused=float(au_fused),
               fused_ece=FE.ece(y, pf))

    # ---- Fig marginals ----
    fig, ax = plt.subplots(figsize=(5.2, 4))
    names = ["image\nexpert", "tabular\nexpert", "fused\nsystem"]; vals = [au_img, au_tab, au_fused]
    ax.bar(names, vals, color=[BLU, "#e08a1e", GRN])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("AUROC"); ax.set_title("Per-expert and fused discrimination")
    _save(fig, "fusx_marginals")

    # ---- Fig fused ROC vs singles ----
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for p, lab, col in [(ai, f"image ({au_img:.3f})", BLU), (at, f"tabular ({au_tab:.3f})", "#e08a1e"),
                        (pf, f"fused ({au_fused:.3f})", GRN)]:
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, lw=2, color=col, label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right"); ax.set_title("Fused system ROC vs single experts")
    _save(fig, "fusx_roc")

    # ---- Fig CO-PRIOR calibration ----
    taus = [0.02, 0.06, 0.10, 0.20]; noshift, shift = [], []
    for tau in taus:
        pr, yr = FE._resample_to_prev(pt, yt, tau)
        noshift.append(FE.ece(yr, pr)); shift.append(FE.ece(yr, FE.anchor(pr, prt, tau)))
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.plot(taus, noshift, "o-", color=RED, label="without CO-PRIOR")
    ax.plot(taus, shift, "o-", color=GRN, label="with CO-PRIOR")
    ax.set_xlabel("deployment prevalence $\\tau$"); ax.set_ylabel("calibration error (ECE)")
    ax.set_title("CO-PRIOR restores calibration"); ax.legend()
    _save(fig, "fusx_coprior")
    out["coprior_ece_no_shift"] = dict(zip(map(str, taus), map(float, noshift)))
    out["coprior_ece_with_shift"] = dict(zip(map(str, taus), map(float, shift)))

    # ---- Fig conformal coverage ----
    fig, ax = plt.subplots(figsize=(5.2, 4))
    alphas = [0.05, 0.1, 0.2]; targets = [1 - a for a in alphas]
    cov_t = [FE._split_conformal_coverage(pt, yt, a) for a in alphas]
    cov_i = [FE._split_conformal_coverage(pi, yi, a) for a in alphas]
    ax.plot(targets, cov_t, "o-", color="#e08a1e", label="tabular expert")
    ax.plot(targets, cov_i, "o-", color=BLU, label="image expert")
    ax.plot([0.8, 0.95], [0.8, 0.95], "k--", lw=0.8, label="target")
    ax.set_xlabel("target coverage $1-\\alpha$"); ax.set_ylabel("empirical coverage")
    ax.set_title("Conformal abstention: coverage guarantee"); ax.legend(loc="upper left")
    _save(fig, "fusx_coverage")
    out["conformal_coverage_tabular"] = dict(zip(map(str, targets), map(float, cov_t)))
    out["conformal_coverage_image"] = dict(zip(map(str, targets), map(float, cov_i)))

    json.dump(out, open(os.path.join(MET, "fusion_full.json"), "w"), indent=2)
    print("saved figures fusx_* , metrics/fusion_full.json")


if __name__ == "__main__":
    main()
