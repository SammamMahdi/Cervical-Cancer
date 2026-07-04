"""Class characterization: what defines each 5-class morphology and each binary (Normal/Abnormal) group.

Runs the analyses a morphometric/cytology paper uses to answer "what does each class have in common
and what separates it", on the 53 ground-truth handcrafted features (explain.features.load_features):

  1. Class PROTOTYPE PROFILES   z-scored per-class centroids -> heatmap (class x feature). The shared
     numeric signature of each class (what its cells are characteristically high/low on).
  2. DISCRIMINATIVE features    one-vs-rest Cohen's d per feature per class -> ranked defining features
     + |d| heatmap. ("What makes this class different from all others.")
  3. UNIVARIATE RELEVANCE       Kruskal-Wallis H (5-class) / Mann-Whitney U (binary) per feature, with
     Benjamini-Hochberg FDR correction (53 tests) and effect size (epsilon^2 / rank-biserial). Which
     features are significantly class-dependent.
  4. SEPARABILITY               standardized feature space projected by PCA / LDA / t-SNE coloured by
     class + silhouette coefficient (global and per-class). How cleanly the classes separate.
  5. VISUAL PROTOTYPES          the mean cell image per class / per binary group ("average appearance").
  6. TOP-FEATURE DISTRIBUTIONS  violins of the most discriminative features split by class.

Methodology is non-parametric (morphometric features are non-normal) with multiple-comparison control.
Everything is persisted: results/metrics/class_analysis/*.json|csv and results/figures/class_analysis/*.
Reproducible (seed). Run: `python -m class_analysis` or `class_analysis.run()`.

References: Cohen 1988 (d); Kruskal & Wallis 1952; Benjamini & Hochberg 1995 (FDR); Fisher 1936 (LDA);
van der Maaten & Hinton 2008 (t-SNE); Rousseeuw 1987 (silhouette).
"""
import os
import numpy as np
from PIL import Image

import config as C
import runlog
import plotting as P

MDIR = os.path.join(C.METRICDIR, "class_analysis")
FDIR = os.path.join(C.FIGDIR, "class_analysis")
for _d in (MDIR, FDIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- stats helpers
def _zscore(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def _cohens_d(a, b):
    """Standardized mean difference (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


# --------------------------------------------------------------------------- analyses
def class_profiles(Xz, y, names, class_names):
    """Z-scored per-class centroid matrix (classes x features). Heatmap + CSV."""
    cents = np.vstack([Xz[y == k].mean(0) for k in range(len(class_names))])
    P.heatmap(cents, class_names, names, os.path.join(FDIR, "profiles_" + _tag(class_names)),
              title=f"Class prototype profiles (z-scored means) — {_tag(class_names)}",
              cmap="coolwarm", center0=True, cbar_label="z-score")
    return cents


def discriminative(X, y, names, class_names, topk=8):
    """One-vs-rest Cohen's d per feature per class; ranked defining features + |d| heatmap."""
    K = len(class_names)
    D = np.zeros((K, X.shape[1]))
    for k in range(K):
        a = X[y == k]; b = X[y != k]
        D[k] = [_cohens_d(a[:, j], b[:, j]) for j in range(X.shape[1])]
    P.heatmap(np.abs(D), class_names, names, os.path.join(FDIR, "discriminative_absD_" + _tag(class_names)),
              title=f"|Cohen's d| one-vs-rest — {_tag(class_names)}", cmap="magma", center0=False,
              cbar_label="|d|")
    top = {}
    for k, cn in enumerate(class_names):
        idx = np.argsort(-np.abs(D[k]))[:topk]
        top[cn] = [{"feature": names[j], "cohens_d": round(float(D[k, j]), 3),
                    "direction": "higher" if D[k, j] > 0 else "lower"} for j in idx]
    P.write_csv(os.path.join(MDIR, "discriminative_" + _tag(class_names)),
                ["class", "rank", "feature", "cohens_d", "direction"],
                [[cn, r + 1, t["feature"], t["cohens_d"], t["direction"]]
                 for cn, ts in top.items() for r, t in enumerate(ts)])
    return D, top


def univariate_relevance(X, y, names, class_names):
    """Per-feature class-dependence test with BH-FDR + effect size. KW for >2 classes, MWU for binary."""
    from scipy import stats
    K = len(class_names); n = len(y); F = X.shape[1]
    rows = []
    pvals = []
    for j in range(F):
        groups = [X[y == k, j] for k in range(K)]
        if K > 2:
            H, p = stats.kruskal(*groups)
            eff = max(0.0, (H - K + 1) / (n - K))          # epsilon^2
            stat_name, eff_name = "kruskal_H", "epsilon_sq"
            val = float(H)
        else:
            U, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            n0, n1 = len(groups[0]), len(groups[1])
            eff = abs(1 - 2 * U / (n0 * n1))               # |rank-biserial|
            stat_name, eff_name = "mannwhitney_U", "rank_biserial"
            val = float(U)
        rows.append([names[j], val, float(p), eff]); pvals.append(p)
    q = _bh_fdr(pvals)
    for i in range(len(rows)):
        rows[i].append(float(q[i]))
    rows.sort(key=lambda r: -r[3])                          # by effect size
    P.write_csv(os.path.join(MDIR, "univariate_relevance_" + _tag(class_names)),
                ["feature", stat_name, "p_value", eff_name, "q_value_BH"], rows)
    return rows


def separability(Xz, y, class_names, seed=C.SEED):
    """PCA / LDA / t-SNE 2-D projections + silhouette (global & per-class)."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, silhouette_samples
    tag = _tag(class_names)
    pca = PCA(n_components=2, random_state=seed).fit_transform(Xz)
    P.embedding_scatter(pca, y, os.path.join(FDIR, f"pca_{tag}"), label_names=class_names,
                        title=f"PCA — {tag}", xlabel="PC1", ylabel="PC2")
    ncomp = min(2, len(class_names) - 1)
    lda = LDA(n_components=ncomp).fit_transform(Xz, y)
    if ncomp == 1:
        lda = np.c_[lda[:, 0], np.zeros(len(lda))]
    P.embedding_scatter(lda, y, os.path.join(FDIR, f"lda_{tag}"), label_names=class_names,
                        title=f"LDA (supervised) — {tag}", xlabel="LD1", ylabel="LD2")
    ts = TSNE(n_components=2, random_state=seed, init="pca",
              perplexity=30, max_iter=1000).fit_transform(Xz)
    P.embedding_scatter(ts, y, os.path.join(FDIR, f"tsne_{tag}"), label_names=class_names,
                        title=f"t-SNE — {tag}", xlabel="t-SNE 1", ylabel="t-SNE 2")
    sil = float(silhouette_score(Xz, y))
    ss = silhouette_samples(Xz, y)
    per = {class_names[k]: round(float(ss[y == k].mean()), 4) for k in range(len(class_names))}
    return dict(silhouette=round(sil, 4), per_class_silhouette=per)


def average_images(paths, y, class_names, size=128, cap=400, seed=C.SEED):
    """Mean cell image per class (visual prototype montage)."""
    rng = np.random.default_rng(seed)
    imgs, titles = [], []
    for k, cn in enumerate(class_names):
        idx = np.where(y == k)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        acc = np.zeros((size, size, 3), np.float64); m = 0
        for i in idx:
            try:
                with Image.open(paths[i]) as im:
                    acc += np.asarray(im.convert("RGB").resize((size, size)), np.float64) / 255.0
                m += 1
            except Exception:
                pass
        if m:
            imgs.append(acc / m); titles.append(f"{cn}\n(n={m})")
    if imgs:
        P.montage(imgs, titles, os.path.join(FDIR, "avg_cell_" + _tag(class_names)),
                  suptitle=f"Average cell — {_tag(class_names)}", ncol=len(imgs))


def top_feature_violins(X, y, names, class_names, D, n=3):
    """Violins of the globally most discriminative features, split by class."""
    rank = np.argsort(-np.abs(D).max(0))[:n]               # features with the largest any-class |d|
    for j in rank:
        groups = {class_names[k]: X[y == k, j] for k in range(len(class_names))}
        P.violins(groups, os.path.join(FDIR, f"dist_{_tag(class_names)}_{names[j]}"),
                  ylabel=names[j], title=f"{names[j]} by class — {_tag(class_names)}")


def _tag(class_names):
    return "binary" if len(class_names) == 2 else "5class"


def _run_one(X, Xz, y, names, class_names, paths):
    cents = class_profiles(Xz, y, names, class_names)
    D, top = discriminative(X, y, names, class_names)
    rel = univariate_relevance(X, y, names, class_names)
    sep = separability(Xz, y, class_names)
    average_images(paths, y, class_names)
    top_feature_violins(X, y, names, class_names, D)
    sig = sum(1 for r in rel if r[-1] < 0.05)
    report = dict(
        tag=_tag(class_names), n=int(len(y)), n_features=len(names), classes=class_names,
        silhouette=sep["silhouette"], per_class_silhouette=sep["per_class_silhouette"],
        n_significant_features_q05=sig,
        top_relevant_features=[{"feature": r[0], "effect_size": round(r[3], 3), "q": round(r[-1], 4)} for r in rel[:10]],
        defining_features_per_class={cn: [t["feature"] + (" (+)" if t["direction"] == "higher" else " (-)")
                                          for t in ts[:5]] for cn, ts in top.items()},
    )
    runlog.dump_json(os.path.join(MDIR, f"class_analysis_{_tag(class_names)}.json"), report)
    return report


def run(split_kind="grouped"):
    """Full class characterization for the 5-class and the binary (Normal/Abnormal) groupings."""
    import explain.features as feat
    data = feat.load_features(split_kind=split_kind, standardize=False)
    X = np.asarray(data["X"], float); y = np.asarray(data["y"], int); names = data["names"]
    paths = data["paths"]
    Xz, _, _ = _zscore(X)
    out = {}
    print("[class-analysis] 5-class ...")
    out["5class"] = _run_one(X, Xz, y, names, list(C.CLASSES), paths)
    print(f"  silhouette={out['5class']['silhouette']}  "
          f"sig features (q<.05)={out['5class']['n_significant_features_q05']}/{len(names)}")
    print("[class-analysis] binary ...")
    yb = np.isin(y, C.ABNORMAL_IDX).astype(int)
    out["binary"] = _run_one(X, Xz, yb, names, ["Normal", "Abnormal"], paths)
    print(f"  silhouette={out['binary']['silhouette']}  "
          f"sig features (q<.05)={out['binary']['n_significant_features_q05']}/{len(names)}")
    runlog.dump_json(os.path.join(MDIR, "class_analysis_summary.json"), out)
    print(f"[class-analysis] JSON+CSV -> {MDIR} | figures -> {FDIR}")
    return out


if __name__ == "__main__":
    run()
