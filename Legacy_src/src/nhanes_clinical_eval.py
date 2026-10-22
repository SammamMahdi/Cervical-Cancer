"""Clinical-prediction evaluation additions (TRIPOD+AI style) for the NHANES HPV-risk model.

Adds the four rigour pieces on top of the discrimination bake-off:
  1. Feature-set ablation (Model A demographics -> B questionnaire -> C +HSV-2 biomarker)
  2. Secondary endpoint: ANY detectable HPV (alongside primary high-risk HPV)
  3. Calibration slope + intercept (the standard complement to Brier/ECE)
  4. Decision-curve analysis (net benefit) — clinical usefulness, not just AUC

Model = tuned LightGBM + isotonic calibration (fit inside folds). One consolidated figure +
metrics JSON.  Run:  python nhanes_clinical_eval.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.optimize import brentq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import nhanes_data as ND
RNG = 42
MET, FIG = ND.MET, ND.FIG

DEMOG = ["age", "race", "education", "marital", "poverty_ratio"]


def lgbm():
    from lightgbm import LGBMClassifier
    p = os.path.join(MET, "nhanes_hpo_results.json")
    params = json.load(open(p))["models"]["LightGBM"]["best_params"] if os.path.exists(p) else \
        dict(n_estimators=500, learning_rate=0.03, num_leaves=31)
    return LGBMClassifier(**params, class_weight="balanced", random_state=RNG, verbose=-1, n_jobs=-1)


def cal_oof(cols, X, y, repeats=2):
    """Isotonic-calibrated out-of-fold probabilities (calibration fit inside each training fold)."""
    Xc = X[cols]; accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        sk = StratifiedKFold(5, shuffle=True, random_state=RNG + r)
        for tr, te in sk.split(Xc, y):
            m = CalibratedClassifierCV(lgbm(), method="isotonic", cv=3)
            m.fit(Xc.iloc[tr], y[tr]); accum[te] += m.predict_proba(Xc.iloc[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


# ---- calibration slope + intercept ----
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    lp = _logit(p)
    lr = LogisticRegression(penalty=None, solver="lbfgs").fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])                                   # ideal 1.0
    f = lambda a: np.mean(1 / (1 + np.exp(-(a + lp)))) - y.mean()   # slope fixed at 1 -> intercept-in-large
    intercept = float(brentq(f, -15, 15))                          # ideal 0.0
    return slope, intercept


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def full_metrics(y, p):
    slope, intercept = calibration_slope_intercept(y, p)
    return dict(auroc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), ece=ece(y, p),
                cal_slope=slope, cal_intercept=intercept)


# ---- decision-curve analysis ----
def net_benefit(y, p, pt):
    pred = p >= pt; n = len(y)
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    return tp / n - fp / n * (pt / (1 - pt))


def dca(y, p, thresholds):
    prev = y.mean()
    model = np.array([net_benefit(y, p, t) for t in thresholds])
    treat_all = np.array([prev - (1 - prev) * (t / (1 - t)) for t in thresholds])
    return model, treat_all


def main():
    out = {}
    oof_primary = {}
    for target in ["high_risk", "any"]:
        X, y, feat, meta = ND.load(target=target, verbose=False)
        clean = [c for c in feat if c not in ND.ADJACENT]
        B = [c for c in clean if c != "hsv2"]                       # questionnaire only (no lab biomarker)
        C = clean                                                    # questionnaire + HSV-2 biomarker
        sets = {"A_demographics": DEMOG, "B_questionnaire": B, "C_plus_biomarker": C}
        print(f"\n=== target={target}  n={meta['n']} pos={meta['n_pos']} ({meta['pos_rate']:.1%}) ===")
        out[target] = {"meta": meta, "sets": {}}
        for name, cols in sets.items():
            p = cal_oof(cols, X, y)
            m = full_metrics(y, p); out[target]["sets"][name] = m
            print(f"  {name:18s} AUROC={m['auroc']:.3f} PR={m['pr_auc']:.3f} Brier={m['brier']:.3f} "
                  f"ECE={m['ece']:.3f} cal-slope={m['cal_slope']:.2f} cal-int={m['cal_intercept']:+.2f}")
            if target == "high_risk":
                oof_primary[name] = (y, p)

    json.dump(out, open(os.path.join(MET, "nhanes_clinical_eval.json"), "w"), indent=2)
    figure(out, oof_primary)
    print("\nsaved -> metrics/nhanes_clinical_eval.json , figures/nhanes_clinical_eval.*")


def figure(out, oof_primary):
    plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 400,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15, 5.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)

    # panel 1: ablation AUROC across feature sets x endpoints
    ax1 = fig.add_subplot(gs[0, 0])
    sets = ["A_demographics", "B_questionnaire", "C_plus_biomarker"]
    x = np.arange(len(sets)); w = 0.38
    for i, tgt in enumerate(["high_risk", "any"]):
        vals = [out[tgt]["sets"][s]["auroc"] for s in sets]
        ax1.bar(x + (i - 0.5) * w, vals, w, label=f"{tgt} HPV")
        for j, v in enumerate(vals):
            ax1.text(x[j] + (i - 0.5) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax1.set_xticks(x); ax1.set_xticklabels(["A\ndemog.", "B\nquestionnaire", "C\n+HSV-2"], fontsize=8.5)
    ax1.set_ylim(0.6, 0.78); ax1.set_ylabel("AUROC (isotonic-calibrated OOF)")
    ax1.set_title("Ablation: demographics -> questionnaire -> +biomarker"); ax1.legend(fontsize=8)

    # panel 2: calibration curve (Model B, high-risk) with slope/intercept
    ax2 = fig.add_subplot(gs[0, 1])
    y, p = oof_primary["B_questionnaire"]
    edges = np.linspace(0, 1, 11); xs, ys = [], []
    for i in range(10):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() > 30:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax2.plot(xs, ys, "o-", color="#1a6", label="Model B (questionnaire)")
    mm = out["high_risk"]["sets"]["B_questionnaire"]
    ax2.text(0.04, 0.9, f"slope={mm['cal_slope']:.2f} (ideal 1.0)\nintercept={mm['cal_intercept']:+.2f} "
             f"(ideal 0)\nECE={mm['ece']:.3f}  Brier={mm['brier']:.3f}", fontsize=8,
             va="top", bbox=dict(boxstyle="round", fc="white", ec="#999"))
    ax2.set_xlabel("predicted P(HPV+)"); ax2.set_ylabel("observed frequency")
    ax2.set_title("Calibration — high-risk HPV, Model B"); ax2.legend(fontsize=8, loc="lower right")

    # panel 3: decision-curve analysis (Model B, high-risk)
    ax3 = fig.add_subplot(gs[0, 2])
    thr = np.linspace(0.05, 0.6, 60)
    model, treat_all = dca(y, p, thr)
    ax3.plot(thr, model, color="#1a6", lw=2, label="Model B (questionnaire)")
    ax3.plot(thr, treat_all, color="#c0392b", lw=1.2, ls="--", label="treat all")
    ax3.axhline(0, color="k", lw=1, label="treat none")
    ax3.set_ylim(-0.05, max(0.05, y.mean() * 1.05)); ax3.set_xlabel("threshold probability")
    ax3.set_ylabel("net benefit"); ax3.set_title("Decision-curve analysis (clinical usefulness)")
    ax3.legend(fontsize=8)

    fig.suptitle("NHANES HPV-risk — clinical evaluation (ablation · calibration slope/intercept · decision curve)",
                 fontsize=12.5, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"nhanes_clinical_eval.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
