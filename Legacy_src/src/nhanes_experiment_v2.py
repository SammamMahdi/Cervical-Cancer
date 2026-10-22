"""NHANES Experiment v2 — representation & heterogeneity (not more tuning/SMOTE/ensembling).

Tests, under the SAME leakage-safe repeated-5-fold + leave-one-cycle-out (LOCO) protocol:
  1. Feature blocks, added incrementally to the CLEAN base, each kept ONLY if it helps BOTH random-CV and LOCO:
       temporal  : recent vs cumulative sexual-exposure features
       timing    : sexual-debut timing & exposure interactions
       nonlinear : log/sqrt of partner counts
       age_int   : age × exposure interactions
       missing   : explicit missingness indicators
       warts_int : genital-warts interaction-only signal
  2. EBM (Explainable Boosting Machine, a GAM with selected interactions) vs tuned LightGBM on the best set.
  3. Age-stratified models vs one global model.
  4. TabPFN v2 benchmark (attempted; skipped if license/limits block it).

Model held fixed = tuned LightGBM (from HPO) unless stated. Outputs ->
Tabular results/NHANES/figures/nhanes_expv2_*  and  metrics/nhanes_expv2.json.   Run: python nhanes_experiment_v2.py
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

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import nhanes_data as ND
RNG = 42
MET, FIG = ND.MET, ND.FIG


def lgbm():
    from lightgbm import LGBMClassifier
    p = os.path.join(MET, "nhanes_hpo_results.json")
    params = json.load(open(p))["models"]["LightGBM"]["best_params"] if os.path.exists(p) else \
        dict(n_estimators=500, learning_rate=0.03, num_leaves=31)
    return LGBMClassifier(**params, class_weight="balanced", random_state=RNG, verbose=-1, n_jobs=-1)


# ---------- feature blocks ----------
def _c(X, c):
    return X[c] if c in X.columns else pd.Series(np.nan, index=X.index)


def block_temporal(X):
    life, yr, ya = _c(X, "n_partners_life"), _c(X, "n_partners_year"), _c(X, "years_active")
    return pd.DataFrame({
        "recent_partner_fraction": yr / (life + 1),
        "historical_partner_burden": (life - yr).clip(lower=0),
        "partner_rate_recent": yr / ya.clip(lower=1),
        "recent_partners_per_active_year": yr / (ya + 1)}, index=X.index)


def block_timing(X):
    a, afs, life, ya = _c(X, "age"), _c(X, "age_first_sex"), _c(X, "n_partners_life"), _c(X, "years_active")
    return pd.DataFrame({
        "age_first_sex_ratio": afs / a,
        "age_first_sex_x_partners": afs * life,
        "years_active_x_partners": ya * life}, index=X.index)


def block_nonlinear(X):
    life, yr = _c(X, "n_partners_life"), _c(X, "n_partners_year")
    return pd.DataFrame({
        "log1p_partners_life": np.log1p(life),
        "log1p_partners_year": np.log1p(yr),
        "sqrt_partners_life": np.sqrt(life)}, index=X.index)


def block_age_int(X):
    a, life, yr = _c(X, "age"), _c(X, "n_partners_life"), _c(X, "n_partners_year")
    afs, mar, smk = _c(X, "age_first_sex"), _c(X, "marital"), _c(X, "smoking_status")
    return pd.DataFrame({
        "age_x_partners_life": a * life, "age_x_partners_year": a * yr, "age_x_first_sex": a * afs,
        "marital_x_partners": mar * life, "smoking_x_partners": smk * life}, index=X.index)


def block_missing(X):
    cols = ["n_partners_life", "n_partners_year", "age_first_sex", "education", "poverty_ratio",
            "hsv2", "age_menarche", "n_pregnancies", "condomless_year"]
    return pd.DataFrame({c + "_missing": _c(X, c).isna().astype(int) for c in cols}, index=X.index)


def block_warts_int(X):
    gw = (_c(X, "genital_warts_hx") == 1).astype(float)
    a, life, afs, smk = _c(X, "age"), _c(X, "n_partners_life"), _c(X, "age_first_sex"), _c(X, "smoking_status")
    return pd.DataFrame({"gw": gw, "gw_x_partners": gw * life, "gw_x_age": gw * a,
                         "gw_x_first_sex": gw * afs, "gw_x_smoking": gw * smk}, index=X.index)


BLOCKS = {"temporal": block_temporal, "timing": block_timing, "nonlinear": block_nonlinear,
          "age_int": block_age_int, "missing": block_missing, "warts_int": block_warts_int}


# ---------- evaluation ----------
def oof_cv(build, Xa, y, repeats=2):
    Xv = Xa.to_numpy(float); accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG + r).split(Xv, y):
            m = build(); m.fit(Xv[tr], y[tr]); accum[te] += m.predict_proba(Xv[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


def loco_auc(build, Xa, y, cycles):
    Xv = Xa.to_numpy(float); per = []
    for c in sorted(set(cycles)):
        te = (cycles == c); tr = ~te
        m = build(); m.fit(Xv[tr], y[tr]); p = m.predict_proba(Xv[te])[:, 1]
        if len(np.unique(y[te])) > 1:
            per.append(roc_auc_score(y[te], p))
    return float(np.mean(per))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    X, y, feat, meta = ND.load(verbose=False)
    cycles = pd.Series(X.attrs["cycle"]).to_numpy()
    base_cols = [c for c in feat if c not in ND.ADJACENT]
    base = X[base_cols].copy()
    print(f"n={len(y)} HPV+={y.sum()} ({y.mean():.3f})  base features={len(base_cols)}")

    out = {"meta": meta}

    # --- 1. incremental feature blocks ---
    p0 = oof_cv(lgbm, base, y); cv0 = roc_auc_score(y, p0); lc0 = loco_auc(lgbm, base, y, cycles)
    print(f"\nBASE                cv={cv0:.4f}  loco={lc0:.4f}")
    block_res = {"BASE": dict(cv=cv0, loco=lc0)}
    keep = []
    for name, fn in BLOCKS.items():
        Xa = pd.concat([base, fn(X)], axis=1)
        cv = roc_auc_score(y, oof_cv(lgbm, Xa, y)); lc = loco_auc(lgbm, Xa, y, cycles)
        dcv, dlc = cv - cv0, lc - lc0
        helped = (dcv > 5e-4) and (dlc > -5e-4)
        keep.append(name) if helped else None
        print(f"  +{name:10s}     cv={cv:.4f} ({dcv:+.4f})  loco={lc:.4f} ({dlc:+.4f})  {'KEEP' if helped else 'drop'}")
        block_res[name] = dict(cv=cv, loco=lc, dcv=dcv, dloco=dlc, kept=helped)

    # cumulative winners
    Xfinal = base.copy()
    for name in keep:
        Xfinal = pd.concat([Xfinal, BLOCKS[name](X)], axis=1)
    cvF = roc_auc_score(y, oof_cv(lgbm, Xfinal, y)); lcF = loco_auc(lgbm, Xfinal, y, cycles)
    print(f"\nFINAL (base + {keep})  cv={cvF:.4f}  loco={lcF:.4f}   (base cv={cv0:.4f})")
    out["blocks"] = block_res; out["kept_blocks"] = keep
    out["final"] = dict(cv=cvF, loco=lcF, n_features=Xfinal.shape[1], vs_base=cvF - cv0)

    # --- 2. EBM (GAM + interactions) vs LightGBM on the final set ---
    print("\n=== EBM vs LightGBM (final feature set) ===")
    out["models"] = {}
    for name, build in [("LightGBM", lgbm), ("EBM", ebm)]:
        p = oof_cv(build, Xfinal, y, repeats=1)         # uncalibrated OOF (EBM is natively calibrated)
        au = roc_auc_score(y, p); ec = ece(y, p); lc = loco_auc(build, Xfinal, y, cycles)
        print(f"  {name:10s} AUROC={au:.4f}  LOCO={lc:.4f}  ECE={ec:.3f}")
        out["models"][name] = dict(auroc=au, loco=lc, ece=ec)

    # --- 3. age-stratified vs global ---
    print("\n=== age-stratified vs global (final set) ===")
    glob = roc_auc_score(y, oof_cv(lgbm, Xfinal, y))
    strat = strat_oof(Xfinal, y, X["age"].to_numpy())
    print(f"  global      cv={glob:.4f}\n  age-strat   cv={strat:.4f}  ({strat-glob:+.4f})")
    out["age_stratified"] = dict(global_cv=glob, stratified_cv=strat, delta=strat - glob)

    # --- 4. TabPFN benchmark ---
    out["tabpfn"] = tabpfn_benchmark(Xfinal, y)

    json.dump(out, open(os.path.join(MET, "nhanes_expv2.json"), "w"), indent=2)
    figure(block_res, out)
    print("\nsaved -> metrics/nhanes_expv2.json , figures/nhanes_expv2_*")


def cal_oof(build, Xa, y, repeats=1):
    Xv = Xa.to_numpy(float); accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG + r).split(Xv, y):
            m = CalibratedClassifierCV(build(), method="isotonic", cv=3)
            m.fit(Xv[tr], y[tr]); accum[te] += m.predict_proba(Xv[te])[:, 1]; cnt[te] += 1
    return accum / np.maximum(cnt, 1)


def ebm():
    from interpret.glassbox import ExplainableBoostingClassifier
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("ebm", ExplainableBoostingClassifier(interactions=8, outer_bags=4,
                                                           random_state=RNG, n_jobs=-1))])


def strat_oof(Xa, y, age, repeats=2):
    bands = np.digitize(age, [25, 35, 45])            # 0:18-24 1:25-34 2:35-44 3:45-59
    Xv = Xa.to_numpy(float); accum = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG + r).split(Xv, y):
            for b in np.unique(bands):
                trb = tr[bands[tr] == b]; teb = te[bands[te] == b]
                if len(np.unique(y[trb])) < 2 or len(teb) == 0:
                    continue
                m = lgbm(); m.fit(Xv[trb], y[trb]); accum[teb] += m.predict_proba(Xv[teb])[:, 1]; cnt[teb] += 1
    ok = cnt > 0
    return float(roc_auc_score(y[ok], (accum / np.maximum(cnt, 1))[ok]))


def tabpfn_benchmark(Xa, y):
    try:
        from tabpfn import TabPFNClassifier
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), min(9000, len(y)), replace=False)          # <=10k cap
        Xv = SimpleImputer(strategy="median").fit_transform(Xa)
        oof = np.full(len(idx), np.nan); ys = y[idx]; Xs = Xv[idx]
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RNG).split(Xs, ys):
            clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
            clf.fit(Xs[tr], ys[tr]); oof[te] = clf.predict_proba(Xs[te])[:, 1]
        au = float(roc_auc_score(ys, oof)); print(f"\nTabPFN (subsample {len(idx)}) AUROC={au:.4f}")
        return dict(auroc=au, n=len(idx))
    except Exception as e:
        print(f"\nTabPFN skipped ({type(e).__name__}: {str(e)[:70]})")
        return dict(skipped=str(type(e).__name__))


def figure(block_res, out):
    plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 400,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    names = [k for k in block_res if k != "BASE"]
    dcv = [block_res[n]["dcv"] for n in names]; dlc = [block_res[n]["dloco"] for n in names]
    x = np.arange(len(names)); w = 0.38
    axs[0].bar(x - w / 2, dcv, w, label="Δ random-CV", color="#1a1a1a")
    axs[0].bar(x + w / 2, dlc, w, label="Δ LOCO", color="#c0392b")
    axs[0].axhline(0, color="k", lw=0.8); axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=25, ha="right")
    axs[0].set_ylabel("Δ AUROC vs base"); axs[0].legend(); axs[0].set_title("Feature blocks (kept only if both > 0)")
    labels = ["BASE", "FINAL"] + list(out.get("models", {})) + (["TabPFN"] if "auroc" in out.get("tabpfn", {}) else [])
    vals = [block_res["BASE"]["cv"], out["final"]["cv"]] + [out["models"][m]["auroc"] for m in out.get("models", {})]
    if "auroc" in out.get("tabpfn", {}):
        vals.append(out["tabpfn"]["auroc"])
    axs[1].bar(labels, vals, color="#1a6")
    for i, v in enumerate(vals):
        axs[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    axs[1].set_ylim(0.66, 0.76); axs[1].axhline(block_res["BASE"]["cv"], color="#888", ls="--", lw=1)
    axs[1].set_ylabel("AUROC"); axs[1].set_title("Best feature set — models"); plt.setp(axs[1].get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Experiment v2 — representation & heterogeneity", y=1.02, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"nhanes_expv2_summary.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
