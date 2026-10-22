"""Revised questionnaire-expert analysis addressing the statistical review.

Changes from the first version
  * nominal variables (race, marital status, sexual orientation) are one-hot encoded instead of being left in
    their released integer codes; yes/no and serology items are recoded to 0/1 so coefficients are readable;
    genuinely ordinal items (education, smoking status) stay numeric;
  * missingness indicators are added for items with >5% missing, because missingness here is structural
    (several questions were only introduced in the 2009 cycle);
  * NHANES survey design (examination weights, strata, primary sampling units) is used for a weighted
    sensitivity analysis of prevalence and performance;
  * model selection is nested inside the outer cross-validation, so the reported score is not the maximum over
    six families on the same folds;
  * calibration is reported as intercept and slope in addition to the expected calibration error;
  * temporal validation: train on the 2003-2010 cycles, test on 2011-2016;
  * performance and calibration are broken out by age group, race/ethnicity and survey cycle.

Run: python paper_jbhi/scripts/eval_tabular_v2.py   -> ../results/tabular_v2.json, tabular_v2_oof.npz
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "Legacy_src", "src"))
import nhanes_data as ND
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RES = os.path.join(HERE, "..", "results"); SEED = 42
NOMINAL = ["race", "marital", "orientation"]
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def build():
    X, y, feat, meta = ND.load(verbose=False)
    cyc = X.attrs["cycle"]
    kind = {q[1]: q[3] for q in ND.QUESTIONNAIRE}
    df = X[feat].copy()
    for f in feat:                                     # yes/no 1,2 and serology 1,2 -> 1,0
        if kind.get(f) in ("bin", "sero"):
            df[f] = (df[f] == 1).astype(float).where(df[f].notna())
    miss_cols = [f for f in feat if df[f].isna().mean() > 0.05]
    for f in miss_cols:
        df[f + "_missing"] = df[f].isna().astype(float)
    df = pd.get_dummies(df, columns=NOMINAL, dummy_na=False, dtype=float)
    return df, y, cyc, feat, meta, miss_cols


def survey_weights(n_expected):
    """NHANES examination weights, strata and PSUs, in the same row order as ND.load()."""
    seqn, wt, psu, strat = [], [], [], []
    for cycle in ND.ALL_CYCLES:
        lab = ND.hpv_swab(cycle, "high_risk")
        if lab is None:
            continue
        rows = lab.dropna(subset=["high_risk_hpv"])
        year, suf = ND.CYCLES[cycle]
        demo = ND._download(f"DEMO_{suf}", year)
        cols = {c: c for c in ("SEQN", "WTMEC2YR", "SDMVPSU", "SDMVSTRA") if c in demo.columns}
        d = demo[list(cols)].drop_duplicates("SEQN")
        m = rows[["SEQN"]].merge(d, on="SEQN", how="left")
        seqn += m["SEQN"].tolist()
        wt += (m["WTMEC2YR"] if "WTMEC2YR" in m else pd.Series([np.nan] * len(m))).tolist()
        psu += (m["SDMVPSU"] if "SDMVPSU" in m else pd.Series([np.nan] * len(m))).tolist()
        strat += (m["SDMVSTRA"] if "SDMVSTRA" in m else pd.Series([np.nan] * len(m))).tolist()
    w = np.array(wt, float) / 7.0                      # pooling 7 two-year cycles
    ok = len(seqn) == n_expected
    print(f"survey weights: {len(seqn)} rows (expected {n_expected}) -> {'aligned' if ok else 'MISALIGNED'}; "
          f"{np.isnan(w).sum()} missing")
    return (w, np.array(psu, float), np.array(strat, float)) if ok else (None, None, None)


# ------------------------------------------------------------------ metrics
def ece(y, p, bins=10, w=None):
    w = np.ones_like(p) if w is None else w
    e = 0.0; edges = np.linspace(0, 1, bins + 1); W = w.sum()
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any() and w[m].sum() > 0:
            e += (w[m].sum() / W) * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return float(e)


def calib_curve_params(y, p):
    """Calibration intercept and slope from a logistic fit of the outcome on the predicted log-odds."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    slope = LogisticRegression(penalty=None, max_iter=1000).fit(z, y)
    inter = LogisticRegression(penalty=None, max_iter=1000, fit_intercept=True).fit(np.zeros_like(z), y)
    return dict(intercept=float(inter.intercept_[0] - np.mean(z)), slope=float(slope.coef_[0][0]))


def scores(y, p, w=None):
    kw = dict(sample_weight=w) if w is not None else {}
    return dict(auroc=float(roc_auc_score(y, p, **kw)), pr_auc=float(average_precision_score(y, p, **kw)),
                brier=float(brier_score_loss(y, p, **kw)), ece=ece(y, p, w=w), n=int(len(y)),
                prevalence=float(np.average(y, weights=w) if w is not None else y.mean()))


# ------------------------------------------------------------------ models
def families(spw):
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Logistic regression": Pipeline([("i", imp()), ("s", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "k-NN": Pipeline([("i", imp()), ("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
        "RandomForest": Pipeline([("i", imp()), ("m", RandomForestClassifier(n_estimators=600, class_weight="balanced",
                                                                            random_state=SEED, n_jobs=-1))]),
        "ExtraTrees": Pipeline([("i", imp()), ("m", ExtraTreesClassifier(n_estimators=600, class_weight="balanced",
                                                                        random_state=SEED, n_jobs=-1))]),
        "LightGBM": Pipeline([("i", imp()), ("m", LGBMClassifier(n_estimators=600, learning_rate=0.05,
                                                                 class_weight="balanced", random_state=SEED,
                                                                 verbose=-1, n_jobs=-1))]),
        "XGBoost": Pipeline([("i", imp()), ("m", XGBClassifier(n_estimators=600, learning_rate=0.05,
                                                               scale_pos_weight=spw, tree_method="hist",
                                                               eval_metric="logloss", random_state=SEED, n_jobs=-1))]),
    }


def main():
    t0 = time.time()
    df, y, cyc, feat, meta, miss_cols = build()
    Xv = df.to_numpy(float); spw = float((y == 0).sum() / (y == 1).sum())
    print(f"n={len(y)} positives={y.sum()} ({y.mean():.3f}) | features {Xv.shape[1]} "
          f"(from {len(feat)}; +{len(miss_cols)} missingness flags, one-hot {NOMINAL})")
    w, psu, strat = survey_weights(len(y))
    sk = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = dict(design=dict(n=len(y), n_features=int(Xv.shape[1]), base_features=len(feat),
                           missing_indicator_features=miss_cols, one_hot=NOMINAL,
                           unweighted_prevalence=float(y.mean())))

    # 1. families under the new encoding
    out["families"] = {}
    for name, clf in families(spw).items():
        p = cross_val_predict(clf, Xv, y, cv=sk, method="predict_proba")[:, 1]
        out["families"][name] = scores(y, p)
        print(f"  {name:20s} AUROC {out['families'][name]['auroc']:.4f}  ECE {out['families'][name]['ece']:.3f}")

    # 2. nested model selection (no peeking: the family is chosen inside each outer fold)
    picked, oof = [], np.zeros(len(y))
    for tr, te in sk.split(Xv, y):
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
        best, best_auc = None, -1
        for name, clf in families(spw).items():
            pi = cross_val_predict(clf, Xv[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
            a = roc_auc_score(y[tr], pi)
            if a > best_auc:
                best, best_auc = name, a
        picked.append(best)
        m = CalibratedClassifierCV(families(spw)[best], method="isotonic", cv=3).fit(Xv[tr], y[tr])
        oof[te] = m.predict_proba(Xv[te])[:, 1]
    out["nested_selection"] = dict(picked_per_fold=picked, calibrated=scores(y, oof),
                                   calibration=calib_curve_params(y, oof))
    print(f"  nested selection picked {picked}; calibrated AUROC {out['nested_selection']['calibrated']['auroc']:.4f} "
          f"ECE {out['nested_selection']['calibrated']['ece']:.3f} "
          f"slope {out['nested_selection']['calibration']['slope']:.2f}")

    # 3. survey-weighted sensitivity analysis
    if w is not None:
        ok = ~np.isnan(w)
        out["survey_weighted"] = dict(
            weighted_prevalence=float(np.average(y[ok], weights=w[ok])), unweighted_prevalence=float(y.mean()),
            weighted=scores(y[ok], oof[ok], w=w[ok]), unweighted=scores(y[ok], oof[ok]),
            n_psu=int(np.unique(psu[~np.isnan(psu)]).size), n_strata=int(np.unique(strat[~np.isnan(strat)]).size))
        s = out["survey_weighted"]
        print(f"  weighted prevalence {s['weighted_prevalence']:.3f} vs unweighted {s['unweighted_prevalence']:.3f}; "
              f"weighted AUROC {s['weighted']['auroc']:.4f} vs {s['unweighted']['auroc']:.4f}")

    # 4. temporal validation: fit on the earlier cycles, test on the later ones
    early = np.isin(cyc, ["2003-2004", "2005-2006", "2007-2008", "2009-2010"])
    mdl = CalibratedClassifierCV(families(spw)["Logistic regression"], method="isotonic", cv=3).fit(Xv[early], y[early])
    p_late = mdl.predict_proba(Xv[~early])[:, 1]
    out["temporal"] = dict(train_cycles=sorted(set(cyc[early])), test_cycles=sorted(set(cyc[~early])),
                           n_train=int(early.sum()), n_test=int((~early).sum()),
                           test=scores(y[~early], p_late), calibration=calib_curve_params(y[~early], p_late),
                           train_prevalence=float(y[early].mean()), test_prevalence=float(y[~early].mean()))
    print(f"  temporal: train {out['temporal']['n_train']} -> test {out['temporal']['n_test']}; "
          f"AUROC {out['temporal']['test']['auroc']:.4f} ECE {out['temporal']['test']['ece']:.3f} "
          f"(prevalence {out['temporal']['train_prevalence']:.3f} -> {out['temporal']['test_prevalence']:.3f})")

    # 5. subgroup performance and calibration
    X0, _, _, _ = ND.load(verbose=False), None, None, None
    raw = X0[0]
    age = raw["age"].to_numpy(float); race = raw["race"].to_numpy(float)
    groups = {"age 18-29": (age < 30), "age 30-39": (age >= 30) & (age < 40),
              "age 40-49": (age >= 40) & (age < 50), "age 50-59": (age >= 50)}
    race_names = {1: "Mexican American", 2: "Other Hispanic", 3: "White", 4: "Black", 5: "Other"}
    for code, nm in race_names.items():
        groups[f"race: {nm}"] = race == code
    for c in sorted(set(cyc)):
        groups[f"cycle {c}"] = cyc == c
    out["subgroups"] = {}
    for g, m in groups.items():
        if m.sum() > 200 and len(np.unique(y[m])) == 2:
            out["subgroups"][g] = scores(y[m], oof[m])
    print("  subgroup AUROC range:",
          f"{min(v['auroc'] for v in out['subgroups'].values()):.3f}-"
          f"{max(v['auroc'] for v in out['subgroups'].values()):.3f}; "
          f"ECE max {max(v['ece'] for v in out['subgroups'].values()):.3f}")

    # 6. raw (uncalibrated) predictions of the deployed family, for the reliability figure
    p_raw = cross_val_predict(families(spw)["Logistic regression"], Xv, y, cv=sk, method="predict_proba")[:, 1]
    out["raw_logistic"] = dict(**scores(y, p_raw), calibration=calib_curve_params(y, p_raw))
    print(f"  raw logistic: AUROC {out['raw_logistic']['auroc']:.4f} ECE {out['raw_logistic']['ece']:.3f}")

    # 7. permutation importance of the deployed expert, measured on held-out folds
    names = list(df.columns)
    drops = {n: [] for n in names}
    for tr, te in sk.split(Xv, y):
        m = families(spw)["Logistic regression"].fit(Xv[tr], y[tr])
        pi = permutation_importance(m, Xv[te], y[te], scoring="roc_auc", n_repeats=5,
                                    random_state=SEED, n_jobs=1)
        for n, v in zip(names, pi.importances_mean):
            drops[n].append(float(v))
    full = families(spw)["Logistic regression"].fit(Xv, y)
    coefs = dict(zip(names, full.named_steps["m"].coef_[0].astype(float)))
    imp = sorted(({"feature": n, "auroc_drop": float(np.mean(v)), "sd": float(np.std(v)),
                   "coef": coefs[n]} for n, v in drops.items()), key=lambda d: -d["auroc_drop"])
    out["permutation_importance"] = imp

    # 8. are the missingness flags carrying the signal, or only inflating permutation importance?
    keep = [c for c in names if not c.endswith("_missing")]
    Xk = df[keep].to_numpy(float)
    pk = cross_val_predict(families(spw)["Logistic regression"], Xk, y, cv=sk, method="predict_proba")[:, 1]
    out["missingness_ablation"] = dict(with_indicators=out["raw_logistic"]["auroc"],
                                       without_indicators=float(roc_auc_score(y, pk)),
                                       n_indicators=len(names) - len(keep))
    print(f"  missingness ablation: {out['missingness_ablation']['with_indicators']:.4f} -> "
          f"{out['missingness_ablation']['without_indicators']:.4f} without the flags")
    print("  top features:", ", ".join(f"{d['feature']} {d['auroc_drop']:.3f}" for d in imp[:5]))

    np.savez_compressed(os.path.join(RES, "tabular_v2_oof.npz"), y=y, p_cal=oof, p_raw=p_raw, cycle=cyc,
                        weights=(w if w is not None else np.zeros(len(y))), age=age, race=race)
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(RES, "tabular_v2.json"), "w"), indent=2)
    print(f"saved results/tabular_v2.json ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
