"""NHANES diagnostics — the three 'understand the ceiling' experiments (not AUROC chases) + CI quantification.

  1. Genotype subgroups: where does the questionnaire signal live?  any / high-risk / 16-18 / other-high-risk.
  2. Monotonic constraints: does epidemiologically-plausible structure help stability/calibration (not AUROC)?
  3. Native-categorical CatBoost: does true categorical handling change anything vs integer codes?
  4. CI quantification: bootstrap 95% CIs on the top candidates -> is any ordering meaningful?

Same leakage-safe repeated-5-fold + LOCO + isotonic-calibration protocol throughout. Base = CLEAN questionnaire
(feat minus HPV-adjacent warts). Outputs -> figures/nhanes_diagnostics_*  and  metrics/nhanes_diagnostics.json.
Run:  python nhanes_diagnostics.py
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

import nhanes_data as ND
from nhanes_pipeline import boot_ci
from nhanes_clinical_eval import calibration_slope_intercept, ece
RNG = 42
MET, FIG = ND.MET, ND.FIG

# exposure features with an epidemiologically well-established monotone direction (others left free)
MONO = {"n_partners_life": 1, "n_partners_year": 1, "n_vaginal_partners_life": 1,
        "n_oral_partners_life": 1, "condomless_year": 1, "partners_per_active_year": 1,
        "years_active": 1, "age_first_sex": -1}


def lgbm_params():
    p = os.path.join(MET, "nhanes_hpo_results.json")
    return json.load(open(p))["models"]["LightGBM"]["best_params"] if os.path.exists(p) else \
        dict(n_estimators=500, learning_rate=0.03, num_leaves=31)


def lgbm(cols=None):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**lgbm_params(), class_weight="balanced", random_state=RNG, verbose=-1, n_jobs=-1)


def lgbm_mono(cols):
    from lightgbm import LGBMClassifier
    mc = [MONO.get(c, 0) for c in cols]
    return LGBMClassifier(**lgbm_params(), class_weight="balanced", monotone_constraints=mc,
                          random_state=RNG, verbose=-1, n_jobs=-1)


def oof(build, X, cols, y, repeats=2, calibrate=False):
    Xv = X[cols].to_numpy(float); accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG + r).split(Xv, y):
            m = CalibratedClassifierCV(build(cols), method="isotonic", cv=3) if calibrate else build(cols)
            m.fit(Xv[tr], y[tr]); accum[te] += m.predict_proba(Xv[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


def loco(build, X, cols, y, cycles):
    Xv = X[cols].to_numpy(float); per = []
    for c in sorted(set(cycles)):
        te = (cycles == c); tr = ~te
        m = build(cols); m.fit(Xv[tr], y[tr]); p = m.predict_proba(Xv[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            per.append(roc_auc_score(y[te], p))
    return float(np.mean(per))


def base_cols(feat):
    return [c for c in feat if c not in ND.ADJACENT]


# ---- 1. genotype subgroups ----
def genotype_subgroups():
    print("=== 1. Genotype subgroups (where is the signal?) ===")
    res = {}
    for target in ["any", "high_risk", "16_18", "other_hr"]:
        X, y, feat, meta = ND.load(target=target, verbose=False)
        cols = base_cols(feat)
        p = oof(lgbm, X, cols, y, repeats=2)
        au = roc_auc_score(y, p); lo, hi = boot_ci(y, p, roc_auc_score)
        res[target] = dict(prevalence=float(y.mean()), n_pos=int(y.sum()), auroc=au, ci=[lo, hi])
        print(f"  {target:10s} prev={y.mean():.3f} n+={int(y.sum()):4d}  AUROC={au:.3f} [{lo:.3f},{hi:.3f}]")
    return res


# ---- 2. monotonic constraints ----
def monotonic():
    print("\n=== 2. Monotonic constraints (stability/calibration, not AUROC) ===")
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    cols = base_cols(feat); cycles = pd.Series(X.attrs["cycle"]).to_numpy()
    res = {}
    for name, build in [("unconstrained", lgbm), ("monotonic", lgbm_mono)]:
        p = oof(build, X, cols, y, repeats=2, calibrate=True)
        sl, ic = calibration_slope_intercept(y, p)
        res[name] = dict(auroc=float(roc_auc_score(y, p)), loco=loco(build, X, cols, y, cycles),
                         brier=float(brier_score_loss(y, p)), ece=ece(y, p), cal_slope=sl, cal_intercept=ic)
        r = res[name]
        print(f"  {name:14s} AUROC={r['auroc']:.4f} LOCO={r['loco']:.4f} Brier={r['brier']:.4f} "
              f"ECE={r['ece']:.3f} slope={r['cal_slope']:.2f} int={r['cal_intercept']:+.2f}")
    return res


# ---- 3. native-categorical CatBoost ----
def catboost_native():
    print("\n=== 3. CatBoost: native categorical vs integer-coded ===")
    from catboost import CatBoostClassifier
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    cols = base_cols(feat); cycles = pd.Series(X.attrs["cycle"]).to_numpy()
    cat_cols = [c for c in cols if c in ND.CAT]

    def frame(Xsub, native):
        Xp = Xsub[cols].copy()
        if native:
            for c in cat_cols:
                Xp[c] = Xp[c].map(lambda v: "nan" if pd.isna(v) else str(int(v)))
        return Xp

    def make():
        return CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6,
                                  auto_class_weights="Balanced", random_seed=RNG, verbose=False, thread_count=-1)

    res = {}
    for native in [False, True]:
        Xf = frame(X, native); cf = cat_cols if native else None
        accum = np.zeros(len(y)); cnt = np.zeros(len(y))
        for r in range(2):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG + r).split(Xf, y):
                m = make(); m.fit(Xf.iloc[tr], y[tr], cat_features=cf)
                accum[te] += m.predict_proba(Xf.iloc[te])[:, 1]; cnt[te] += 1
        p = accum / np.maximum(cnt, 1)
        # LOCO
        per = []
        for c in sorted(set(cycles)):
            te = (cycles == c); tr = ~te
            m = make(); m.fit(Xf[tr], y[tr], cat_features=cf); pp = m.predict_proba(Xf[te])[:, 1]
            if len(np.unique(y[te])) > 1:
                per.append(roc_auc_score(y[te], pp))
        key = "native_categorical" if native else "integer_coded"
        res[key] = dict(auroc=float(roc_auc_score(y, p)), loco=float(np.mean(per)))
        print(f"  {key:20s} AUROC={res[key]['auroc']:.4f}  LOCO={res[key]['loco']:.4f}")
    return res


# ---- 4. CI quantification across top candidates ----
def ci_quantification():
    print("\n=== 4. CI quantification (is any ordering meaningful?) ===")
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    cols = base_cols(feat)
    res = {}
    from lightgbm import LGBMClassifier  # noqa
    builds = {"LightGBM": lgbm, "LightGBM_monotonic": lgbm_mono}
    for name, b in builds.items():
        p = oof(b, X, cols, y, repeats=3)
        au = roc_auc_score(y, p); lo, hi = boot_ci(y, p, roc_auc_score)
        res[name] = dict(auroc=au, ci=[lo, hi]); print(f"  {name:20s} AUROC={au:.3f} [{lo:.3f},{hi:.3f}]")
    # EBM
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        eb = lambda c: Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("ebm", ExplainableBoostingClassifier(interactions=8, outer_bags=4,
                                                                       random_state=RNG, n_jobs=-1))])
        p = oof(eb, X, cols, y, repeats=1)
        au = roc_auc_score(y, p); lo, hi = boot_ci(y, p, roc_auc_score)
        res["EBM"] = dict(auroc=au, ci=[lo, hi]); print(f"  {'EBM':20s} AUROC={au:.3f} [{lo:.3f},{hi:.3f}]")
    except Exception as e:
        print(f"  EBM skipped ({type(e).__name__})")
    return res


def figure(sub, mono, cb, ci):
    plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 400,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    # genotype subgroups with CI
    order = ["any", "high_risk", "16_18", "other_hr"]
    labels = ["any HPV", "high-risk", "HPV-16/18", "other high-risk"]
    au = [sub[t]["auroc"] for t in order]
    lo = [au[i] - sub[t]["ci"][0] for i, t in enumerate(order)]
    hi = [sub[t]["ci"][1] - au[i] for i, t in enumerate(order)]
    axs[0].bar(labels, au, color="#1a6", yerr=[lo, hi], capsize=4)
    for i, t in enumerate(order):
        axs[0].text(i, au[i] + max(hi) + 0.006, f"{au[i]:.3f}\n({sub[t]['prevalence']:.0%})", ha="center", fontsize=8)
    axs[0].axhline(0.5, color="k", lw=0.8, ls="--"); axs[0].set_ylim(0.5, 0.82)
    axs[0].set_ylabel("AUROC (95% CI)"); axs[0].set_title("Genotype subgroups — where is the signal?")
    plt.setp(axs[0].get_xticklabels(), rotation=15, ha="right")
    # CI dot plot for candidates
    names = list(ci); yv = np.arange(len(names))
    a = [ci[n]["auroc"] for n in names]
    el = [a[i] - ci[n]["ci"][0] for i, n in enumerate(names)]
    eh = [ci[n]["ci"][1] - a[i] for i, n in enumerate(names)]
    axs[1].errorbar(a, yv, xerr=[el, eh], fmt="o", color="#111", ecolor="#999", capsize=4)
    axs[1].set_yticks(yv); axs[1].set_yticklabels(names)
    axs[1].axvspan(min(ci[n]["ci"][0] for n in names), max(ci[n]["ci"][1] for n in names),
                   color="#eee", zorder=0)
    axs[1].set_xlabel("AUROC (95% CI)"); axs[1].set_title("Candidates — overlapping CIs = no meaningful ordering")
    fig.suptitle("NHANES diagnostics — subgroups · monotonic · CatBoost · CI", y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"nhanes_diagnostics.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    sub = genotype_subgroups()
    mono = monotonic()
    cb = catboost_native()
    ci = ci_quantification()
    out = dict(genotype_subgroups=sub, monotonic=mono, catboost=cb, ci=ci)
    json.dump(out, open(os.path.join(MET, "nhanes_diagnostics.json"), "w"), indent=2)
    figure(sub, mono, cb, ci)
    print("\nsaved -> metrics/nhanes_diagnostics.json , figures/nhanes_diagnostics.*")


if __name__ == "__main__":
    main()
