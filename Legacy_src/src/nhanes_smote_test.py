"""Split-then-SMOTE vs SMOTE-before-split — the controlled experiment.

Holds the classifier fixed (the tuned LightGBM from nhanes_hpo) and varies ONLY how SMOTE is applied,
under the SAME repeated stratified 5-fold protocol, reporting AUROC / PR-AUC / accuracy / balanced-accuracy:

  A  honest baseline        impute -> LGBM(class_weight)          no resampling               (reference)
  B  split-then-SMOTE       impute -> SMOTE(train fold only) -> LGBM   CORRECT: SMOTE fits inside the fold,
                                                                       test fold stays 100% real
  C  SMOTE-before-split     SMOTE(whole data) -> 5-fold CV        LEAKY: synthetic points leak into test

The question: does correctly-applied SMOTE beat the honest baseline, or was the boost purely the leak?

Outputs -> Tabular results/NHANES/figures/nhanes_smote_comparison.*  and  metrics/nhanes_smote_test.json
Run:  python nhanes_smote_test.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import nhanes_data as ND
from nhanes_pipeline import NUM, CAT
RNG = 42
FIG, MET = ND.FIG, ND.MET


def tuned_lgbm(class_weight=None):
    from lightgbm import LGBMClassifier
    p = os.path.join(MET, "nhanes_hpo_results.json")
    params = {}
    if os.path.exists(p):
        params = json.load(open(p))["models"]["LightGBM"]["best_params"]
    return LGBMClassifier(**params, class_weight=class_weight, random_state=RNG, verbose=-1, n_jobs=-1)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(auroc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                f1=float(f1_score(y, pred)))


def repeated_oof(build_est, X, y, repeats=3):
    """Repeated 5-fold OOF for a pipeline that handles its own impute/resample INSIDE fit()."""
    Xin = X[NUM + CAT]
    accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        sk = StratifiedKFold(5, shuffle=True, random_state=RNG + r)
        for tr, te in sk.split(Xin, y):
            est = build_est()
            est.fit(Xin.iloc[tr], y[tr])
            accum[te] += est.predict_proba(Xin.iloc[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


def leaky_before_split(X, y, repeats=3):
    """SMOTE the WHOLE dataset first, THEN cross-validate — metrics computed on the augmented test folds,
    exactly as the flawed protocol reports them (that is the leak)."""
    Xi = SimpleImputer(strategy="median").fit_transform(X[NUM + CAT])
    Xr, yr = SMOTE(random_state=RNG).fit_resample(Xi, y)                 # synthetic points made from ALL data
    accum = np.zeros(len(yr)); cnt = np.zeros(len(yr))
    for r in range(repeats):
        sk = StratifiedKFold(5, shuffle=True, random_state=RNG + r)
        for tr, te in sk.split(Xr, yr):
            m = clone(tuned_lgbm(None)); m.fit(Xr[tr], yr[tr])
            accum[te] += m.predict_proba(Xr[te])[:, 1]; cnt[te] += 1
    return yr, accum / np.maximum(cnt, 1)


def main():
    X, y, feat, meta = ND.load(verbose=False)
    print(f"NHANES n={len(y)} HPV+={y.sum()} ({y.mean():.3f})  minority baseline acc={1-y.mean():.3f}\n")
    res = {}

    # A honest baseline (class weights, no resampling)
    pA = repeated_oof(lambda: ImbPipeline([("imp", SimpleImputer(strategy="median")),
                                           ("clf", tuned_lgbm("balanced"))]), X, y)
    res["A_honest_classweight"] = metrics(y, pA)

    # B split-then-SMOTE (SMOTE only fits on the training fold; test fold stays real)
    pB = repeated_oof(lambda: ImbPipeline([("imp", SimpleImputer(strategy="median")),
                                           ("smote", SMOTE(random_state=RNG)),
                                           ("clf", tuned_lgbm(None))]), X, y)
    res["B_split_then_SMOTE"] = metrics(y, pB)

    # C SMOTE-before-split (leaky) — evaluated on the augmented set, as the flawed papers do
    yr, pC = leaky_before_split(X, y)
    res["C_SMOTE_before_split_LEAKY"] = metrics(yr, pC)

    hdr = f'{"method":30s}{"AUROC":>8s}{"PR-AUC":>8s}{"accuracy":>10s}{"bal-acc":>9s}{"F1":>7s}'
    print(hdr); print("-" * len(hdr))
    for k, m in res.items():
        print(f'{k:30s}{m["auroc"]:8.3f}{m["pr_auc"]:8.3f}{m["accuracy"]:10.3f}'
              f'{m["balanced_accuracy"]:9.3f}{m["f1"]:7.3f}')

    json.dump({"meta": meta, "results": res}, open(os.path.join(MET, "nhanes_smote_test.json"), "w"), indent=2)
    make_fig(res)
    print("\nsaved -> metrics/nhanes_smote_test.json , figures/nhanes_smote_comparison.*")
    return res


def make_fig(res):
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 600,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = ["A honest\n(class-weight)", "B split-then-SMOTE\n(correct)", "C SMOTE-before-split\n(leaky)"]
    keys = list(res)
    auroc = [res[k]["auroc"] for k in keys]; acc = [res[k]["accuracy"] for k in keys]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - w / 2, auroc, w, label="AUROC", color="#1a1a1a")
    b2 = ax.bar(x + w / 2, acc, w, label="accuracy@0.5", color="#c0392b")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_ylabel("score"); ax.legend(loc="upper left")
    ax.set_title("SMOTE applied correctly (B) ≈ honest baseline (A); only the leaky order (C) inflates")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "nhanes_smote_comparison.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "nhanes_smote_comparison.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
