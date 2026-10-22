"""Dataset characterization for the NHANES HPV-risk table — the counterpart to the UCI
`tabular_characterize.py`, but here the signal is real, so the figures show separation instead of overlap.

Figures -> Tabular results/NHANES/figures/char_*   |   stats -> metrics/nhanes_characterization.json
  char_01 missingness            char_06 PCA (variance + 2D scatter)
  char_02 prevalence (cycle/age) char_07 t-SNE (2D, coloured by HPV)
  char_03 distributions by class char_08 class separability (RF risk-score split)
  char_04 feature correlation    char_09 leakage demo (why published 0.94 is oversample-before-split)
  char_05 univariate signal + MI

Run:  python nhanes_characterize.py
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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

import nhanes_data as ND
from nhanes_pipeline import NUM, CAT, _load_cycles

FIG, MET = ND.FIG, ND.MET
INK = "#1a1a1a"; POS = "#c0392b"; NEG = "#2c3e50"
plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 600,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _design(X):
    """Impute+scale numeric, impute+one-hot categorical -> dense matrix for PCA/t-SNE/MI."""
    ct = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])
    return ct.fit_transform(X[NUM + CAT])


def main():
    X, y, feat, meta = ND.load(verbose=False)
    cyc = _load_cycles()
    stats = {"meta": meta}
    print(f"n={len(y)}  HPV+={y.sum()} ({y.mean():.3f})")

    # --- char_01 missingness ---
    miss = (X[feat].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(miss.index, miss.values, color=INK)
    ax.set_xlabel("% missing"); ax.set_title("NHANES: feature missingness")
    _save(fig, "char_01_missingness")

    # --- char_02 prevalence by cycle and by age band ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    pc = pd.Series(y, name="y").groupby(cyc.values).mean()
    axs[0].bar(pc.index, pc.values, color=INK); axs[0].axhline(y.mean(), color=POS, ls="--", lw=1)
    axs[0].set_ylabel("HPV+ rate"); axs[0].set_title("by survey cycle")
    plt.setp(axs[0].get_xticklabels(), rotation=30, ha="right")
    ageband = pd.cut(X["age"], [17, 24, 29, 34, 39, 44, 49, 59])
    pa = pd.Series(y).groupby(ageband.values, observed=True).mean()
    axs[1].bar([str(i) for i in pa.index], pa.values, color=INK); axs[1].axhline(y.mean(), color=POS, ls="--", lw=1)
    axs[1].set_ylabel("HPV+ rate"); axs[1].set_title("by age band (peaks young, declines with age)")
    plt.setp(axs[1].get_xticklabels(), rotation=30, ha="right")
    _save(fig, "char_02_prevalence")
    stats["prevalence_by_cycle"] = {k: float(v) for k, v in pc.items()}

    # --- char_03 key feature distributions split by HPV status ---
    keys = ["n_partners_life", "age", "age_first_sex", "poverty_ratio"]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for ax, k in zip(axs.ravel(), keys):
        v = X[k].to_numpy(float)
        lo, hi = np.nanpercentile(v, [1, 99]); bins = np.linspace(lo, hi, 30)
        ax.hist(v[(y == 0)], bins=bins, density=True, alpha=0.55, color=NEG, label="HPV-")
        ax.hist(v[(y == 1)], bins=bins, density=True, alpha=0.55, color=POS, label="HPV+")
        ax.set_title(k); ax.legend(fontsize=8)
    fig.suptitle("Feature distributions by HPV status (visible shift = real signal)", y=1.02)
    _save(fig, "char_03_dist_by_class")

    # --- char_04 correlation (numeric + label) ---
    corr = X[NUM].assign(HPV=y).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, fraction=0.046); ax.set_title("Pearson correlation (numeric + HPV)")
    _save(fig, "char_04_correlation")

    # --- char_05 univariate AUROC + mutual information ---
    uni = {}
    for c in feat:
        v = X[c].to_numpy(float); m = ~np.isnan(v)
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], v[m]); uni[c] = float(max(a, 1 - a))
    D = _design(X)
    mi = mutual_info_classif(D, y, random_state=0)
    # aggregate one-hot MI back is messy; report MI on the raw imputed features instead
    Ximp = SimpleImputer(strategy="median").fit_transform(X[feat])
    mi_raw = mutual_info_classif(Ximp, y, discrete_features=False, random_state=0)
    mi_raw = dict(zip(feat, mi_raw))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    u = dict(sorted(uni.items(), key=lambda kv: kv[1]))
    axs[0].barh(list(u), list(u.values()), color=INK); axs[0].axvline(0.5, color=POS, ls="--", lw=1)
    axs[0].set_xlabel("univariate AUROC"); axs[0].set_title("single-feature discrimination")
    mr = dict(sorted(mi_raw.items(), key=lambda kv: kv[1]))
    axs[1].barh(list(mr), list(mr.values()), color=INK)
    axs[1].set_xlabel("mutual information (nats)")
    axs[1].set_title(f"MI with HPV (max {max(mi_raw.values()):.3f} vs UCI <0.01)")
    _save(fig, "char_05_signal")
    stats["univariate_auroc"] = uni; stats["mutual_info"] = {k: float(v) for k, v in mi_raw.items()}

    # --- char_06 PCA ---
    p = PCA(n_components=10, random_state=0).fit(D)
    Z = p.transform(D)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    axs[0].plot(np.arange(1, 11), np.cumsum(p.explained_variance_ratio_), "o-", color=INK)
    axs[0].set_xlabel("component"); axs[0].set_ylabel("cumulative variance"); axs[0].set_title("PCA variance")
    idx = np.random.default_rng(0).choice(len(y), min(4000, len(y)), replace=False)
    axs[1].scatter(Z[idx, 0], Z[idx, 1], c=[POS if t else NEG for t in y[idx]], s=6, alpha=0.4)
    axs[1].set_xlabel("PC1"); axs[1].set_ylabel("PC2"); axs[1].set_title("PC1–PC2 (red=HPV+)")
    _save(fig, "char_06_pca")

    # --- char_07 t-SNE ---
    idx = np.random.default_rng(1).choice(len(y), min(3000, len(y)), replace=False)
    emb = TSNE(n_components=2, perplexity=35, init="pca", random_state=0).fit_transform(D[idx])
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.scatter(emb[:, 0], emb[:, 1], c=[POS if t else NEG for t in y[idx]], s=8, alpha=0.5)
    ax.set_title("t-SNE (red=HPV+): gradient, not the total overlap UCI showed")
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, "char_07_tsne")

    # --- char_08 class separability via RF OOF risk score ---
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    oof = cross_val_predict(rf, np.nan_to_num(Ximp), y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 30)
    ax.hist(oof[y == 0], bins=bins, density=True, alpha=0.55, color=NEG, label="HPV-")
    ax.hist(oof[y == 1], bins=bins, density=True, alpha=0.55, color=POS, label="HPV+")
    ax.axvline(oof[y == 0].mean(), color=NEG, ls="--"); ax.axvline(oof[y == 1].mean(), color=POS, ls="--")
    ax.set_xlabel("RF OOF risk score"); ax.legend()
    ax.set_title(f"Class separability (AUROC {auc:.3f}) — distributions genuinely differ")
    _save(fig, "char_08_separability")

    # --- char_09 leakage demo: why the published 0.94 is oversample-before-split ---
    from sklearn.metrics import roc_auc_score as _auc
    honest = auc                                                # RF OOF, resampling-free
    # ADASYN/SMOTE applied to the WHOLE dataset BEFORE the split (the paper's likely error)
    from imblearn.over_sampling import SMOTE
    Xr, yr = SMOTE(random_state=0).fit_resample(np.nan_to_num(Ximp), y)
    leak = cross_val_predict(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
                             Xr, yr, cv=5, method="predict_proba")[:, 1]
    leak_auc = _auc(yr, leak)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ["honest\n(RF OOF, no resample)", "SMOTE BEFORE split\n(leakage)"]
    vals = [honest, leak_auc]
    ax.bar(bars, vals, color=[INK, POS])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.5, color="k", lw=0.8, ls="--"); ax.set_ylim(0.5, 1.02); ax.set_ylabel("AUROC")
    ax.set_title("Reproducing the published inflation: oversample-before-split")
    _save(fig, "char_09_leakage_demo")
    stats["honest_rf_oof_auroc"] = float(honest); stats["smote_before_split_auroc"] = float(leak_auc)

    json.dump(stats, open(os.path.join(MET, "nhanes_characterization.json"), "w"), indent=2)
    print(f"honest OOF AUROC={honest:.3f} | SMOTE-before-split={leak_auc:.3f} (inflated)")
    print("saved figures -> Tabular results/NHANES/figures/char_*")


if __name__ == "__main__":
    main()
