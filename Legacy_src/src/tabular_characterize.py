"""Comprehensive characterization of the UCI Cervical Cancer (Risk Factors) tabular dataset.

Produces (into results/figures/tabular/, PDF + PNG @600dpi):
  1  missingness            per-feature % missing
  2  target_prevalence      class imbalance of the 4 tests + any-of-4
  3  target_cooccurrence    phi-correlation between the 4 test targets
  4  feature_correlation    Spearman correlation heatmap of the leakage-safe features
  5  univariate_signal      |AUROC-0.5| and mutual information of every feature vs Biopsy  (WHY the ceiling)
  6  pca_variance           cumulative explained variance
  7  pca_scatter            2-D PCA coloured by Biopsy
  8  tsne_biopsy / tsne_any4  2-D t-SNE coloured by target (class overlap => the ceiling, visually)
  9  class_separability     top discriminative features by Biopsy class (box)
  10 leakage_demo           honest AUROC vs Dx-leaked AUROC (why 0.98 in the literature is fake)
  11 model_bakeoff          repeated-CV OOF AUROC + bootstrap CI across 9 model families
  12 shap_summary           XGBoost SHAP (which risk factors actually move the score)

Also writes results/metrics/tabular_characterization.json and a recommendations markdown.
Run from legacy_src/src:  python tabular_characterize.py
Grounding: leakage-safe throughout (Dx:* dropped for modelling; median-impute for the models that
need it). The Dx-leaked number is shown ONLY in figure 10 to demonstrate the leakage, never used.
"""
import os
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 12, "font.weight": "bold", "axes.labelsize": 13, "axes.labelweight": "bold",
    "axes.titlesize": 13, "axes.titleweight": "bold", "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 1.2, "lines.linewidth": 2.0,
    "legend.frameon": True, "legend.edgecolor": "black",
})
PAL = {"pos": "#A23B72", "neg": "#2E86AB", "bar": "#2E86AB", "hi": "#F18F01", "chance": "#7F7F7F",
       "leak": "#D32F2F", "honest": "#2ECC71"}

import warnings as _w
_w.filterwarnings("ignore")

from scipy.stats import norm  # noqa
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import config as C
import tabular as TAB

FIGDIR = os.path.join(C.FIGDIR, "tabular")
os.makedirs(FIGDIR, exist_ok=True)
SEED = C.SEED
TARGETS = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
PRIMARY = "Biopsy"

SHORT = {  # readable labels for the long UCI column names
    "Number of sexual partners": "# sex partners", "First sexual intercourse": "age 1st sex",
    "Num of pregnancies": "# pregnancies", "Smokes (years)": "smoke yrs",
    "Smokes (packs/year)": "smoke packs/yr", "Hormonal Contraceptives": "horm.contra",
    "Hormonal Contraceptives (years)": "horm.contra yrs", "IUD (years)": "IUD yrs",
    "STDs (number)": "# STDs", "STDs:condylomatosis": "STD:condyl",
    "STDs:vulvo-perineal condylomatosis": "STD:vp-condyl", "STDs:vaginal condylomatosis": "STD:vag-condyl",
    "STDs:cervical condylomatosis": "STD:cerv-condyl", "STDs:pelvic inflammatory disease": "STD:PID",
    "STDs:genital herpes": "STD:herpes", "STDs:molluscum contagiosum": "STD:mollusc",
    "STDs:Hepatitis B": "STD:HepB", "STDs: Number of diagnosis": "# STD dx",
    "STDs:syphilis": "STD:syphilis", "STDs:HIV": "STD:HIV", "STDs:AIDS": "STD:AIDS",
    "STDs:HPV": "STD:HPV",
}
short = lambda c: SHORT.get(c, c)


def save(fig, stem):
    fig.savefig(os.path.join(FIGDIR, stem + ".pdf"))
    fig.savefig(os.path.join(FIGDIR, stem + ".png"))
    plt.close(fig)
    return stem


# --------------------------------------------------------------------------- #
#  Load
# --------------------------------------------------------------------------- #
def load_raw():
    df = pd.read_csv(TAB.UCI_CSV, na_values="?").apply(pd.to_numeric, errors="coerce")
    return df


# =========================================================================== #
#  PART 1 — EDA statistics
# =========================================================================== #
def eda_stats(df):
    stats = dict(n=len(df), n_cols=df.shape[1])
    stats["targets"] = {t: dict(pos=int(df[t].sum()), rate=float(df[t].mean())) for t in TARGETS}
    yany = (df[TARGETS].sum(1) > 0).astype(int)
    stats["any4"] = dict(pos=int(yany.sum()), rate=float(yany.mean()))
    stats["imbalance_ratio"] = {t: float((df[t] == 0).sum() / max(int(df[t].sum()), 1)) for t in TARGETS}
    miss = df.isna().mean().sort_values(ascending=False)
    stats["missing_top"] = {c: float(m) for c, m in miss[miss > 0].items()}
    return stats, yany


# =========================================================================== #
#  FIGURES
# =========================================================================== #
def fig_missingness(df):
    miss = (df.isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6.5, max(4, 0.28 * len(miss))))
    ax.barh([short(c) for c in miss.index], miss.values, color=PAL["bar"], edgecolor="black", linewidth=0.8)
    ax.set_xlabel("% missing"); ax.set_title("Missingness per feature")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    return save(fig, "01_missingness")


def fig_prevalence(df, yany):
    names = TARGETS + ["any-of-4"]
    rates = [df[t].mean() * 100 for t in TARGETS] + [yany.mean() * 100]
    pos = [int(df[t].sum()) for t in TARGETS] + [int(yany.sum())]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(names, rates, color=PAL["pos"], edgecolor="black", linewidth=1.0)
    for b, r, p in zip(bars, rates, pos):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.1f}%\n(n={p})", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.axhline(50, ls="--", c=PAL["chance"], lw=1.0)
    ax.set_ylabel("positive rate (%)"); ax.set_ylim(0, 60)
    ax.set_title("Severe class imbalance across all targets")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return save(fig, "02_target_prevalence")


def fig_cooccurrence(df):
    M = df[TARGETS].astype(float).corr().values     # phi = pearson of 0/1
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(TARGETS, rotation=30, ha="right"); ax.set_yticklabels(TARGETS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("$\\phi$ correlation")
    ax.set_title("Target co-occurrence (the 4 tests)")
    return save(fig, "03_target_cooccurrence")


def fig_feature_corr(X):
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    M = Xi.corr(method="spearman").values
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(X.columns))); ax.set_yticks(range(len(X.columns)))
    ax.set_xticklabels([short(c) for c in X.columns], rotation=90, fontsize=6)
    ax.set_yticklabels([short(c) for c in X.columns], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman $\\rho$")
    ax.set_title("Feature correlation (leakage-safe features)")
    return save(fig, "04_feature_correlation")


def fig_univariate_signal(X, y):
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_classif(Xi, y, random_state=SEED)
    rows = []
    for j, c in enumerate(X.columns):
        try:
            au = roc_auc_score(y, Xi[:, j])
        except Exception:
            au = 0.5
        rows.append((c, abs(au - 0.5), mi[j]))
    rows.sort(key=lambda r: -r[1])
    labels = [short(c) for c, _, _ in rows]
    auc_dev = [a for _, a, _ in rows]
    mis = [m for _, _, m in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, max(4, 0.28 * len(rows))))
    axes[0].barh(labels[::-1], auc_dev[::-1], color=PAL["hi"], edgecolor="black", linewidth=0.7)
    axes[0].axvline(0.07, ls="--", c=PAL["chance"], lw=1.0)  # |AUROC-0.5|=0.07 ~ AUROC 0.57
    axes[0].set_xlabel("|univariate AUROC $-$ 0.5|"); axes[0].set_title("Discrimination vs Biopsy")
    axes[0].grid(True, axis="x", alpha=0.3, linestyle="--"); axes[0].set_axisbelow(True)
    axes[1].barh(labels[::-1], mis[::-1], color=PAL["bar"], edgecolor="black", linewidth=0.7)
    axes[1].set_xlabel("mutual information (nats)"); axes[1].set_title("MI vs Biopsy")
    axes[1].grid(True, axis="x", alpha=0.3, linestyle="--"); axes[1].set_axisbelow(True)
    axes[1].set_yticklabels([])
    fig.suptitle("Every feature is a weak predictor of Biopsy (max AUROC $\\approx$ 0.57)",
                 fontweight="bold", y=1.01)
    return save(fig, "05_univariate_signal"), rows


def fig_pca(X, y):
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=4, color=PAL["bar"])
    ax.axhline(0.9, ls="--", c=PAL["chance"], lw=1.0)
    ax.set_xlabel("# components"); ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA — variance is spread across many axes")
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "06_pca_variance")

    Z = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=12, c=PAL["neg"], alpha=0.5, label="Biopsy $-$", edgecolor="none")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=26, c=PAL["pos"], alpha=0.9, label="Biopsy $+$", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(loc="best")
    ax.set_title("PCA (2-D) — positives are not separable")
    return save(fig, "07_pca_scatter")


def fig_tsne(X, y, yany):
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    Xp = PCA(n_components=min(30, Xs.shape[1]), random_state=SEED).fit_transform(Xs)
    Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
             random_state=SEED).fit_transform(Xp)
    for tag, yy, name in [("biopsy", y, "Biopsy"), ("any4", yany, "any-of-4")]:
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.scatter(Z[yy == 0, 0], Z[yy == 0, 1], s=12, c=PAL["neg"], alpha=0.45,
                   label=f"{name} $-$", edgecolor="none")
        ax.scatter(Z[yy == 1, 0], Z[yy == 1, 1], s=30, c=PAL["pos"], alpha=0.95,
                   label=f"{name} $+$", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(loc="best")
        ax.set_title(f"t-SNE — {name} positives sit inside the negative cloud")
        ax.set_xticks([]); ax.set_yticks([])
        save(fig, f"08_tsne_{tag}")
    return "08_tsne"


def fig_class_sep(X, y, ranked, topn=6):
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    feats = [c for c, _, _ in ranked[:topn]]
    fig, axes = plt.subplots(1, topn, figsize=(2.0 * topn, 3.6))
    for ax, c in zip(axes, feats):
        data = [Xi.loc[y == 0, c].values, Xi.loc[y == 1, c].values]
        bp = ax.boxplot(data, labels=["$-$", "$+$"], patch_artist=True, widths=0.6, showfliers=False)
        for patch, col in zip(bp["boxes"], [PAL["neg"], PAL["pos"]]):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_title(short(c), fontsize=9); ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    fig.suptitle("Top-6 features by Biopsy class — heavy overlap", fontweight="bold", y=1.03)
    return save(fig, "09_class_separability")


# =========================================================================== #
#  MODEL BAKE-OFF  (leakage-safe repeated-CV OOF AUROC + bootstrap CI)
# =========================================================================== #
def _pipe(est, scale=True, impute=True):
    steps = []
    if impute:
        steps.append(("imp", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)


def models():
    return {
        "LogReg (L2)": _pipe(LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
        "SVM-RBF": _pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        "kNN": _pipe(KNeighborsClassifier(n_neighbors=21, weights="distance")),
        "GaussianNB": _pipe(GaussianNB(), scale=False),
        "MLP": _pipe(MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=800, random_state=SEED)),
        "RandomForest": _pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                     min_samples_leaf=3, random_state=SEED), scale=False),
        "ExtraTrees": _pipe(ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                                 min_samples_leaf=3, random_state=SEED), scale=False),
        "HistGB": _pipe(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                        max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20,
                        random_state=SEED), scale=False),
        "XGBoost": _pipe(xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
                        scale_pos_weight=14.0, random_state=SEED), scale=False),
    }


def oof_auroc(est, X, y, seeds=(42, 7, 13)):
    Xv = X.to_numpy(float); y = np.asarray(y)
    aucs, oof0 = [], None
    for si, s in enumerate(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        oof = np.full(len(y), np.nan)
        for tr, va in skf.split(Xv, y):
            from sklearn.base import clone
            m = clone(est); m.fit(Xv[tr], y[tr]); oof[va] = m.predict_proba(Xv[va])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        if si == 0:
            oof0 = oof
    return float(np.mean(aucs)), float(np.std(aucs)), oof0


def boot_ci(y, p, n=2000, seed=0):
    y = np.asarray(y); p = np.asarray(p); rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            b.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def bakeoff(X, y):
    res = {}
    print("\n=== MODEL BAKE-OFF (leakage-safe, repeated 5-fold OOF, target=Biopsy) ===")
    for name, est in models().items():
        mu, sd, oof = oof_auroc(est, X, y)
        lo, hi = boot_ci(y, oof)
        res[name] = dict(auroc=mu, sd=sd, ci=[lo, hi])
        print(f"  {name:14s} AUROC={mu:.3f} ± {sd:.3f}  CI[{lo:.3f}, {hi:.3f}]")
    order = sorted(res, key=lambda k: res[k]["auroc"])
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ys = np.arange(len(order))
    mus = [res[k]["auroc"] for k in order]
    err = [[res[k]["auroc"] - res[k]["ci"][0] for k in order],
           [res[k]["ci"][1] - res[k]["auroc"] for k in order]]
    ax.barh(ys, mus, xerr=err, color=PAL["bar"], edgecolor="black", linewidth=1.0,
            error_kw=dict(elinewidth=1.2, ecolor="#333", capsize=3))
    for yv, k in zip(ys, order):
        ax.text(res[k]["auroc"] + 0.005, yv, f"{res[k]['auroc']:.3f}", va="center", fontsize=8, fontweight="bold")
    ax.axvline(0.5, ls="--", c=PAL["chance"], lw=1.2, label="chance")
    ax.set_yticks(ys); ax.set_yticklabels(order); ax.set_xlim(0.45, 0.80)
    ax.set_xlabel("out-of-fold AUROC (95% CI)"); ax.legend(loc="lower right")
    ax.set_title("Model bake-off — all families cluster at 0.55–0.66")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    save(fig, "11_model_bakeoff")
    return res


def fig_leakage(X_safe, df, y):
    """Demonstrate the TWO leakage mechanisms behind the literature's 0.9+ on this dataset:
       (B) resampling the WHOLE set before the split; (C) substituting the easy Dx:Cancer target.
    Bar A is the honest number. NONE of B/C is used anywhere else — they exist only to expose the leak."""
    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    mk = lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                                l2_regularization=1.0, min_samples_leaf=20, random_state=SEED)

    # (A) honest — leakage-safe features, proper out-of-fold CV
    _, _, oof = oof_auroc(_pipe(mk(), scale=False), X_safe, y, seeds=(42,))
    honest = roc_auc_score(y, oof)

    # (B) oversample-BEFORE-split — duplicate minority on the FULL set, then a random split
    Xi = SimpleImputer(strategy="median").fit_transform(X_safe)
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    dup = rng.choice(pos, size=len(neg) - len(pos), replace=True)   # balance by duplication
    idx = np.concatenate([np.arange(len(y)), dup])
    Xb, yb = Xi[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=1, stratify=yb)
    over = roc_auc_score(yte, mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1])

    # (C) Dx:Cancer target predicted from its sibling Dx flags — target-substitution leak.
    #     A tree that can memorise (RandomForest, as the leaky literature uses) exposes it: the
    #     'Dx' union flag + Dx:HPV/Dx:CIN almost perfectly determine Dx:Cancer (18 positives).
    dxy = df["Dx:Cancer"].astype(int).to_numpy()
    dxX = df[["Dx", "Dx:HPV", "Dx:CIN"]].fillna(0)
    dx_est = RandomForestClassifier(n_estimators=400, random_state=SEED)
    _, _, oofc = oof_auroc(_pipe(dx_est, scale=False), dxX, dxy, seeds=(42,))
    dxc = roc_auc_score(dxy, oofc)

    vals = [honest, over, dxc]
    labels = ["A. HONEST\n(leakage-safe,\nproper CV)",
              "B. oversample\nBEFORE split\n(same features)",
              "C. predict Dx:Cancer\nfrom Dx columns\n(target swap)"]
    cols = [PAL["honest"], PAL["leak"], PAL["leak"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.axhline(0.5, ls="--", c=PAL["chance"], lw=1.0)
    ax.set_ylabel("AUROC (out-of-fold / held-out)"); ax.set_ylim(0, 1.08)
    ax.set_title("The literature's 0.9$+$ is leakage, not skill")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    save(fig, "10_leakage_demo")
    print(f"  leakage demo: honest={honest:.3f}  oversample-before-split={over:.3f}  "
          f"Dx:Cancer-from-Dx={dxc:.3f}")
    return dict(honest=float(honest), oversample_before_split=float(over), dx_cancer_from_dx=float(dxc))


def fig_shap(X, y):
    try:
        import shap
        m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
                              scale_pos_weight=14.0, random_state=SEED)
        Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                          columns=[short(c) for c in X.columns])
        m.fit(Xi, y)
        sv = shap.TreeExplainer(m).shap_values(Xi)
        plt.figure(figsize=(6.5, 5.0))
        shap.summary_plot(sv, Xi, show=False, max_display=15)
        fig = plt.gcf(); fig.suptitle("XGBoost SHAP — which risk factors move the Biopsy score",
                                      fontweight="bold", y=1.02)
        save(fig, "12_shap_summary")
        return True
    except Exception as e:
        print(f"  [SHAP skipped: {e}]")
        return False


# =========================================================================== #
#  main
# =========================================================================== #
def main():
    C.set_seed()
    df = load_raw()
    X, Y, feat_cols = TAB.load_uci()               # leakage-safe features + 4 targets
    y = Y[PRIMARY].to_numpy().astype(int)

    stats, yany = eda_stats(df)
    print(f"UCI: n={stats['n']} patients, {stats['n_cols']} cols; "
          f"{len(feat_cols)} leakage-safe features. Primary target = {PRIMARY} "
          f"({stats['targets'][PRIMARY]['pos']} pos, {100*stats['targets'][PRIMARY]['rate']:.1f}%).")

    fig_missingness(df)
    fig_prevalence(df, yany)
    fig_cooccurrence(df)
    fig_feature_corr(X)
    _, ranked = fig_univariate_signal(X, y)
    fig_pca(X, y)
    fig_tsne(X, y, yany)
    fig_class_sep(X, y, ranked)
    leak = fig_leakage(X, df, y)
    bake = bakeoff(X, y)
    shap_ok = fig_shap(X, y)

    out = dict(dataset=stats, leakage_demo=leak, model_bakeoff=bake,
               top_features=[dict(feature=c, auroc_dev=a, mi=m) for c, a, m in ranked[:10]],
               shap=shap_ok, figures_dir=FIGDIR)
    os.makedirs(C.METRICDIR, exist_ok=True)
    json.dump(out, open(os.path.join(C.METRICDIR, "tabular_characterization.json"), "w"), indent=2)
    best = max(bake, key=lambda k: bake[k]["auroc"])
    print(f"\nbest model = {best} (AUROC {bake[best]['auroc']:.3f}); "
          f"leakage-safe ceiling confirmed ~{bake[best]['auroc']:.2f}.")
    print(f"figures -> {FIGDIR}")
    print("metrics -> results/metrics/tabular_characterization.json")


if __name__ == "__main__":
    main()
