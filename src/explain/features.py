"""Axis-2 feature explainability + U7 late fusion.

The SIPaKMeD release ships, per class, two handcrafted morphometry tables in
``data/raw/sipakmed_features/Features_CELL/<CLASS>_{NUCLEI,CYTOPLASM}_FEAT.dat`` where ``<CLASS>``
is one of ``DYSKERATOTIC, KOILOCYTOTIC, METAPLASTIC, PARABASAL, SUP_INT`` (SUP_INT ==
Superficial-Intermediate). Each line is ``fov,cell,<26 floats>`` with NO header; columns 0,1 are
integer join keys ``(fov, cell)`` and column 2 (the first float) is the segmented region AREA.

This module turns those tables into an interpretable, *non-deep* axis of the study:

  * :func:`load_features` joins Nuclei(26)+Cytoplasm(26) per cell via ``(class, fov, cell)`` to the
    cropped single-cell images (filename ``<FOV>_<cell>.bmp`` -> fov int, cell int), derives the
    nucleus/cytoplasm AREA ratio ``nc_ratio`` (53 features total), standardizes the columns, and
    aligns every row to the leakage-safe ``data.splits.make_split`` membership (per-cell path).
  * :func:`train_tabular` fits an XGBoost 5-class model on the TRAIN cells and reports val/test
    accuracy + macro-F1; the model is persisted.
  * :func:`shap_report` runs ``shap.TreeExplainer`` for a global beeswarm
    (``results/figures/shap_features``) and a human "abnormal because nc_ratio high (+), ..."
    reason string per cell (top-k signed SHAP values).
  * :func:`permutation_importance` and :func:`partial_dependence` add the usual tabular-XAI figures.
  * :func:`late_fusion` (U7 ablation) stacks tabular-XGBoost probabilities with the image model's
    5-class probabilities through a LogisticRegression meta-learner and contrasts image-only vs
    tabular-only vs FUSED on the SEALED test split. It degrades to tabular-only when the trained
    student checkpoint (``results/checkpoints/student.pt``) is absent.

All numbers are collected into ``results/metrics/feature_explain.json``. The ``__main__`` path runs
the CPU-only chain (load + train + SHAP), never touching the image model.
"""
import os
import json

import numpy as np

import config as C

# --------------------------------------------------------------------------- #
#  Paths / constants
# --------------------------------------------------------------------------- #
FEATURE_DIR = os.path.join(C.DATA_RAW, "sipakmed_features", "Features_CELL")

# SIPaKMeD CLASS name  ->  .dat filename stem
CLASS_TO_DAT = {
    "Dyskeratotic": "DYSKERATOTIC",
    "Koilocytotic": "KOILOCYTOTIC",
    "Metaplastic": "METAPLASTIC",
    "Parabasal": "PARABASAL",
    "Superficial-Intermediate": "SUP_INT",
}

# The 26 morphometry columns (cols 2..27 of each .dat) in SIPaKMeD release order. Col 0 (the first
# float, dat-column 2) is AREA; the remaining names follow the published per-region descriptor set
# (geometric shape descriptors, then three intensity-channel statistic blocks). They are used purely
# as readable labels for the figures/reasons; the join + model are index-based so a label mismatch
# never corrupts the math.
_RAW_FEAT_NAMES = [
    "area", "majoraxis", "minoraxis", "eccentricity", "orientation", "equivdiameter",
    "solidity", "extent", "perimeter",
    "i0_mean", "i0_var", "i0_skew", "i0_kurt", "i0_entropy",
    "i1_mean", "i1_var", "i1_skew", "i1_kurt", "i1_entropy",
    "i2_mean", "i2_var", "i2_skew", "i2_kurt", "i2_entropy",
    "ratio_a", "ratio_b",
]
N_RAW = len(_RAW_FEAT_NAMES)   # 26

SHAP_FIG = os.path.join(C.FIGDIR, "shap_features")
PERM_FIG = os.path.join(C.FIGDIR, "feature_permutation_importance")
PDP_FIG = os.path.join(C.FIGDIR, "feature_partial_dependence")
MODEL_PATH = os.path.join(C.CKPTDIR, "tabular_xgb.json")
METRICS_PATH = os.path.join(C.METRICDIR, "feature_explain.json")
STUDENT_CKPT = os.path.join(C.CKPTDIR, "student.pt")


# --------------------------------------------------------------------------- #
#  .dat parsing + join
# --------------------------------------------------------------------------- #
def _read_dat(path):
    """Parse a SIPaKMeD ``*_FEAT.dat`` -> dict ``(fov:int, cell:int) -> np.ndarray(26 floats)``.

    Lines are ``fov,cell,<26 floats>``; malformed/short lines are skipped. The ``(fov,cell)`` keys
    are cast through float-then-int so a ``"1.0"`` style integer never breaks the join.
    """
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2 + N_RAW:
                continue
            try:
                fov = int(float(parts[0]))
                cell = int(float(parts[1]))
                vals = np.asarray([float(v) for v in parts[2:2 + N_RAW]], dtype=np.float64)
            except ValueError:
                continue
            out[(fov, cell)] = vals
    return out


def _key_from_path(path):
    """Cropped filename ``<FOV>_<cell>.bmp`` (e.g. ``001_03``) -> ``(fov:int, cell:int)`` join key."""
    stem = os.path.splitext(os.path.basename(path))[0]
    fov_s, cell_s = stem.split("_")[:2]
    return int(fov_s), int(cell_s)


def feature_names():
    """Structured 53-name list: ``nucleus_*`` (26), ``cytoplasm_*`` (26), then ``nc_ratio``."""
    return (
        [f"nucleus_{n}" for n in _RAW_FEAT_NAMES]
        + [f"cytoplasm_{n}" for n in _RAW_FEAT_NAMES]
        + ["nc_ratio"]
    )


def _build_matrix(standardize=True):
    """Join Nuclei+Cytoplasm tables to the cropped cells; return raw (un-split) arrays.

    Returns ``(X[n,53], y[n], paths[n], names[53], mean[53], std[53])`` over EVERY cell that has both
    a nucleus row and a cytoplasm row (the SIPaKMeD tables cover all 4,049 cells 1:1). When
    ``standardize`` is True the columns are z-scored using the returned ``mean``/``std`` (population
    statistics over all matched cells; train_tabular re-fits standardization on TRAIN only).
    """
    from data.clustering import load_cells

    cells = load_cells()
    # cache parsed .dat per class so we read each file once
    nuc_cache, cyt_cache = {}, {}
    rows, ys, paths = [], [], []
    for c in cells:
        dat = CLASS_TO_DAT.get(c.class_name)
        if dat is None:
            continue
        if dat not in nuc_cache:
            nuc_cache[dat] = _read_dat(os.path.join(FEATURE_DIR, f"{dat}_NUCLEI_FEAT.dat"))
            cyt_cache[dat] = _read_dat(os.path.join(FEATURE_DIR, f"{dat}_CYTOPLASM_FEAT.dat"))
        key = _key_from_path(c.path)
        nv = nuc_cache[dat].get(key)
        cv = cyt_cache[dat].get(key)
        if nv is None or cv is None:
            continue
        nuc_area = nv[0]
        cyt_area = cv[0]
        nc_ratio = float(nuc_area / cyt_area) if cyt_area != 0 else 0.0
        rows.append(np.concatenate([nv, cv, [nc_ratio]]))
        ys.append(c.class_idx)
        paths.append(c.path)

    if not rows:
        raise RuntimeError(f"no cells joined to handcrafted features under {FEATURE_DIR}")

    X = np.vstack(rows).astype(np.float64)
    y = np.asarray(ys, dtype=np.int64)
    names = feature_names()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)   # guard constant columns
    if standardize:
        X = (X - mean) / std
    return X, y, paths, names, mean, std


def load_features(split_kind="grouped", standardize=True):
    """Handcrafted per-cell features aligned to the leakage-safe split.

    Joins the SIPaKMeD Nuclei(26)+Cytoplasm(26) ``.dat`` tables to the cropped cells via
    ``(class, fov, cell)``, appends the derived nucleus/cytoplasm AREA ratio ``nc_ratio`` (53
    features), standardizes the columns (z-score over all matched cells) and tags every row with its
    ``data.splits.make_split`` membership.

    Returns a dict with:
        X        : float64 ``[n, 53]`` (standardized when ``standardize``)
        y        : int64 ``[n]`` class index (NORMAL-first)
        paths    : list[str] cropped-cell image paths (row-aligned)
        names    : list[str] 53 structured feature names (``nucleus_*`` / ``cytoplasm_*`` / nc_ratio)
        split    : dict ``{'train','val','test'}`` -> int row-index arrays into X/y/paths
        mean,std : the standardization stats (``[53]``) so callers can de/standardize new rows
    """
    from data.splits import get_split

    X, y, paths, names, mean, std = _build_matrix(standardize=standardize)
    # normcase: Windows abspath drive-letter case (F: vs f:) varies by how python was launched, so the
    # split JSON and the freshly-globbed feature paths can differ only in case -> match case-insensitively.
    path_to_row = {os.path.normcase(p): i for i, p in enumerate(paths)}

    sp = get_split(split_kind)
    split = {}
    for part in ("train", "val", "test"):
        idx = [path_to_row[k] for k in (os.path.normcase(p) for p in sp[part]) if k in path_to_row]
        split[part] = np.asarray(idx, dtype=np.int64)
    return dict(X=X, y=y, paths=paths, names=names, split=split, mean=mean, std=std)


# --------------------------------------------------------------------------- #
#  Tabular model (XGBoost 5-class)
# --------------------------------------------------------------------------- #
def _train_split_standardized(data=None, split_kind="grouped"):
    """Build TRAIN-fitted-standardized arrays from a (possibly raw) load_features dict.

    Returns ``(Xs, y, names, split, mean, std)`` where ``Xs`` is standardized using TRAIN-only
    mean/std (no val/test leakage into the scaler). Re-standardizes regardless of how ``data`` was
    loaded so the function is safe to call on a standardized or raw dict.
    """
    if data is None:
        data = load_features(split_kind=split_kind, standardize=False)
    # recover raw matrix: if it was standardized, un-standardize using its stored stats
    X = data["X"]
    # Detect standardization by checking column std ~ 1; safest is to rebuild raw.
    Xraw, y, paths, names, _m, _s = _build_matrix(standardize=False)
    # align: load_features and _build_matrix iterate identically -> same row order
    if Xraw.shape == X.shape:
        Xuse = Xraw
    else:  # pragma: no cover - defensive
        Xuse = X
    split = data["split"]
    tr = split["train"]
    mean = Xuse[tr].mean(axis=0)
    std = Xuse[tr].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    Xs = (Xuse - mean) / std
    return Xs, data["y"], names, split, mean, std


def train_tabular(data=None, split_kind="grouped", save=True, n_estimators=400,
                  max_depth=4, learning_rate=0.05, persist_path=MODEL_PATH):
    """Fit an XGBoost 5-class classifier on the TRAIN cells; report val/test acc + macro-F1.

    The features are standardized on TRAIN only. The fitted booster is persisted to
    ``persist_path`` (JSON). Returns a dict with the model, the TRAIN-standardization stats, the
    feature names, the split, and a ``metrics`` block (train/val/test accuracy + macro_f1).
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score

    if data is None:
        data = load_features(split_kind=split_kind, standardize=False)
    Xs, y, names, split, mean, std = _train_split_standardized(data, split_kind)

    tr, va, te = split["train"], split["val"], split["test"]
    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", num_class=C.NUM_CLASSES,
        tree_method="hist", eval_metric="mlogloss",
        random_state=C.SEED, n_jobs=0,
    )
    eval_set = [(Xs[va], y[va])] if len(va) else None
    clf.fit(Xs[tr], y[tr], eval_set=eval_set, verbose=False)

    def _scores(idx):
        if len(idx) == 0:
            return dict(accuracy=float("nan"), macro_f1=float("nan"), n=0)
        pred = clf.predict(Xs[idx])
        return dict(
            accuracy=float(accuracy_score(y[idx], pred)),
            macro_f1=float(f1_score(y[idx], pred, labels=list(range(C.NUM_CLASSES)),
                                    average="macro", zero_division=0)),
            n=int(len(idx)),
        )

    metrics = {"train": _scores(tr), "val": _scores(va), "test": _scores(te)}

    if save:
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        clf.save_model(persist_path)
        # stash the standardization stats alongside the booster for reuse at fusion time
        with open(persist_path + ".scaler.json", "w", encoding="utf-8") as fh:
            json.dump({"mean": mean.tolist(), "std": std.tolist(), "names": names}, fh)

    return dict(model=clf, Xs=Xs, y=y, names=names, split=split,
                mean=mean, std=std, metrics=metrics)


# --------------------------------------------------------------------------- #
#  SHAP global + per-cell reasons
# --------------------------------------------------------------------------- #
def _shap_values_for_class(explainer, X, class_idx):
    """Return signed SHAP values ``[n, n_feat]`` for one class from a TreeExplainer output.

    Handles both shap return layouts: a list-per-class, or a 3-D array ``[n, n_feat, n_class]``.
    """
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.asarray(sv[class_idx])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, class_idx]
    return sv  # binary / single-output fallback


def reasons(i, trained=None, data=None, split_kind="grouped", k=4):
    """Human-readable signed-SHAP reason string for cell row ``i`` (index into X/paths).

    Uses the SHAP values for the cell's PREDICTED class; picks the top-``k`` features by absolute
    contribution and renders e.g. ``"abnormal because nc_ratio high (+), nucleus_area large (+),
    cytoplasm_solidity low (-)"``. ``trained`` may be a :func:`train_tabular` dict (reused); else a
    model is trained on the fly.
    """
    import shap

    if trained is None:
        trained = train_tabular(data=data, split_kind=split_kind, save=False)
    clf, Xs, names = trained["model"], trained["Xs"], trained["names"]

    pred = int(clf.predict(Xs[i:i + 1])[0])
    explainer = shap.TreeExplainer(clf)
    sv = _shap_values_for_class(explainer, Xs[i:i + 1], pred)[0]   # [n_feat]

    order = np.argsort(np.abs(sv))[::-1][:k]
    # value level relative to the (standardized) population: >0 -> "high/large", <0 -> "low/small"
    parts = []
    for j in order:
        sign = "+" if sv[j] >= 0 else "-"
        level = "high" if Xs[i, j] >= 0 else "low"
        parts.append(f"{names[j]} {level} ({sign})")

    cls_name = C.CLASSES[pred]
    verdict = "abnormal" if pred in C.ABNORMAL_IDX else "normal"
    return f"{verdict} ({cls_name}) because " + ", ".join(parts)


def shap_report(trained=None, data=None, split_kind="grouped", max_display=18,
                sample=None, out=SHAP_FIG):
    """Global SHAP beeswarm over the test cells + a few example per-cell reason strings.

    Saves ``results/figures/shap_features.png`` (+ .pdf). Returns a dict with the figure path,
    a ``mean_abs_shap`` per-feature ranking, and ``example_reasons`` for a handful of test cells.
    Uses the abnormal-class SHAP signal for the beeswarm coloring of clinical relevance.
    """
    import shap
    import plotting  # noqa: F401  -- applies the house style globally
    import matplotlib.pyplot as plt

    if trained is None:
        trained = train_tabular(data=data, split_kind=split_kind, save=False)
    clf, Xs, names, split = trained["model"], trained["Xs"], trained["names"], trained["split"]

    # explain on the held-out TEST cells (fall back to all rows if test empty)
    idx = split["test"] if len(split["test"]) else np.arange(len(Xs))
    if sample is not None and len(idx) > sample:
        rng = np.random.default_rng(C.SEED)
        idx = rng.choice(idx, size=sample, replace=False)
    Xe = Xs[idx]

    explainer = shap.TreeExplainer(clf)

    # mean |SHAP| across ALL classes for a robust global ranking
    sv_all = explainer.shap_values(Xe)
    if isinstance(sv_all, list):
        stacked = np.stack([np.abs(np.asarray(s)) for s in sv_all], axis=0)  # [cls,n,feat]
        mean_abs = stacked.mean(axis=(0, 1))
        beeswarm_vals = np.asarray(sv_all[C.ABNORMAL_IDX[0]])
    else:
        sv_all = np.asarray(sv_all)
        if sv_all.ndim == 3:                       # [n, feat, cls]
            mean_abs = np.abs(sv_all).mean(axis=(0, 2))
            beeswarm_vals = sv_all[:, :, C.ABNORMAL_IDX[0]]
        else:
            mean_abs = np.abs(sv_all).mean(axis=0)
            beeswarm_vals = sv_all

    ranking = sorted(
        ({"feature": names[j], "mean_abs_shap": float(mean_abs[j])} for j in range(len(names))),
        key=lambda d: d["mean_abs_shap"], reverse=True,
    )

    # global beeswarm (abnormal-class signal)
    fig = plt.figure(figsize=(8.4, 8.0))
    try:
        shap.summary_plot(beeswarm_vals, Xe, feature_names=names, max_display=max_display,
                          show=False, plot_size=None)
    except Exception:
        # ultra-defensive fallback: bar of mean |SHAP|
        plt.close(fig)
        fig = plt.figure(figsize=(8.4, 8.0))
        top = ranking[:max_display][::-1]
        plt.barh([d["feature"] for d in top], [d["mean_abs_shap"] for d in top],
                 color="#A23B72", edgecolor="black")
        plt.xlabel("mean |SHAP|")
    fig = plt.gcf()
    fig.suptitle("Handcrafted-feature SHAP (abnormal class)", fontweight="bold", y=1.01)
    png = plotting.save(fig, out)

    # a few example reasons from the test set
    paths = _paths_of(trained, data, split_kind)
    examples = []
    for i in idx[:6]:
        examples.append({"path": os.path.basename(paths[int(i)]),
                         "reason": reasons(int(i), trained=trained)})

    return dict(figure=png, mean_abs_shap=ranking, example_reasons=examples)


def _paths_of(trained, data, split_kind):
    """Recover the row-aligned image paths for a trained dict (load_features carries them)."""
    if data is not None and "paths" in data:
        return data["paths"]
    return load_features(split_kind=split_kind, standardize=False)["paths"]


# --------------------------------------------------------------------------- #
#  Permutation importance + partial dependence
# --------------------------------------------------------------------------- #
def permutation_importance(trained=None, data=None, split_kind="grouped", n_repeats=10,
                           out=PERM_FIG, top=18):
    """Permutation importance of each handcrafted feature on the TEST split (accuracy drop).

    Saves a horizontal bar figure and returns the ranked list of
    ``{feature, importance_mean, importance_std}``.
    """
    from sklearn.inspection import permutation_importance as sk_perm
    import plotting  # noqa: F401
    import matplotlib.pyplot as plt

    if trained is None:
        trained = train_tabular(data=data, split_kind=split_kind, save=False)
    clf, Xs, y, names, split = (trained["model"], trained["Xs"], trained["y"],
                                trained["names"], trained["split"])
    idx = split["test"] if len(split["test"]) else split["val"]
    if len(idx) == 0:
        idx = np.arange(len(Xs))

    r = sk_perm(clf, Xs[idx], y[idx], n_repeats=n_repeats, random_state=C.SEED,
                scoring="accuracy", n_jobs=1)
    ranking = sorted(
        ({"feature": names[j], "importance_mean": float(r.importances_mean[j]),
          "importance_std": float(r.importances_std[j])} for j in range(len(names))),
        key=lambda d: d["importance_mean"], reverse=True,
    )

    show = ranking[:top][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.barh([d["feature"] for d in show], [d["importance_mean"] for d in show],
            xerr=[d["importance_std"] for d in show], color="#2E86AB", edgecolor="black",
            error_kw=dict(ecolor="#222222", lw=1.2, capsize=2.5))
    ax.set_xlabel("Accuracy drop when permuted")
    ax.set_title("Permutation importance (test)")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    png = plotting.save(fig, out)
    return dict(figure=png, ranking=ranking)


def partial_dependence(trained=None, data=None, split_kind="grouped", features=None,
                       target_class=None, out=PDP_FIG):
    """Partial-dependence curves of the top features for one class (default: first abnormal class).

    ``features`` may be a list of feature names (default: top-4 by permutation importance). Saves a
    small multi-panel figure; returns the chosen features + the target class.
    """
    from sklearn.inspection import PartialDependenceDisplay
    import plotting  # noqa: F401
    import matplotlib.pyplot as plt

    if trained is None:
        trained = train_tabular(data=data, split_kind=split_kind, save=False)
    clf, Xs, names, split = trained["model"], trained["Xs"], trained["names"], trained["split"]
    target_class = C.ABNORMAL_IDX[0] if target_class is None else target_class

    if features is None:
        perm = permutation_importance(trained=trained, split_kind=split_kind, n_repeats=5,
                                      out=PERM_FIG)
        features = [d["feature"] for d in perm["ranking"][:4]]
    feat_idx = [names.index(f) for f in features]

    idx = split["train"] if len(split["train"]) else np.arange(len(Xs))
    Xbg = Xs[idx]

    n = len(feat_idx)
    ncol = min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.6 * nrow), squeeze=False)
    try:
        PartialDependenceDisplay.from_estimator(
            clf, Xbg, feat_idx, feature_names=names, target=target_class,
            ax=axes.ravel()[:n], grid_resolution=30,
        )
    except Exception:
        # manual 1-D PDP fallback (average predicted prob as one feature is swept)
        for ax, fj in zip(axes.ravel(), feat_idx):
            grid = np.linspace(np.percentile(Xbg[:, fj], 5), np.percentile(Xbg[:, fj], 95), 30)
            ys = []
            for g in grid:
                Xt = Xbg.copy(); Xt[:, fj] = g
                ys.append(clf.predict_proba(Xt)[:, target_class].mean())
            ax.plot(grid, ys, color="#A23B72")
            ax.set_xlabel(names[fj]); ax.set_ylabel(f"P({C.CLASSES[target_class]})")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Partial dependence — {C.CLASSES[target_class]}", fontweight="bold", y=1.02)
    png = plotting.save(fig, out)
    return dict(figure=png, features=features, target_class=int(target_class))


# --------------------------------------------------------------------------- #
#  U7 late fusion (image-model probs + tabular probs -> meta-learner)
# --------------------------------------------------------------------------- #
def _image_probs_for(paths, ckpt=None, batch_size=C.BATCH_SIZE):
    """Run the trained student over ``paths`` -> (probs[n,5], available:bool).

    Builds the student per ``C.STUDENT`` and loads the LATEST 'student' checkpoint (ckpt root or a
    run_*/ snapshot subfolder; an explicit path overrides), then evaluates over a CellDataset +
    eval-transform DataLoader via ``eval.metrics.predict``. Returns ``(None, False)`` if no student
    checkpoint exists so the caller can gate to tabular-only.
    """
    import ckpt as CK
    path = ckpt if (ckpt and os.path.isfile(ckpt)) else CK.latest_path("student")
    if not path or not os.path.isfile(path):
        return None, False
    from torch.utils.data import DataLoader
    from models.backbones import build
    from data.dataset import CellDataset, build_transform
    from eval.metrics import predict

    model = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"],
                  C.STUDENT["dropout"], pretrained=False)
    state, _meta, _p = CK.load(path, map_location="cpu")   # unwraps payload OR bare state_dict
    model.load_state_dict(state, strict=False)
    model.to(C.DEVICE).eval()

    ds = CellDataset(paths, build_transform(train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=False)
    _y, probs = predict(model, loader)   # probs aligned to paths order (shuffle=False)
    return probs, True


def late_fusion(data=None, split_kind="grouped", ckpt=None, save=True):
    """U7 ablation: image-only vs tabular-only vs FUSED on the SEALED test split.

    Tabular probabilities come from the XGBoost model; image probabilities from the trained student
    (gated out if ``student.pt`` is absent). A LogisticRegression meta-learner is fit on the
    concatenated VAL probabilities and scored on TEST. Returns a dict of acc/macro_f1 for each arm
    (``fused`` present only when the image model is available).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if data is None:
        data = load_features(split_kind=split_kind, standardize=False)
    trained = train_tabular(data=data, split_kind=split_kind, save=False)
    clf, Xs, y, split = trained["model"], trained["Xs"], trained["y"], trained["split"]
    paths = data["paths"]

    va, te = split["val"], split["test"]

    def _sc(yt, yp):
        return dict(
            accuracy=float(accuracy_score(yt, yp)),
            macro_f1=float(f1_score(yt, yp, labels=list(range(C.NUM_CLASSES)),
                                    average="macro", zero_division=0)),
            n=int(len(yt)),
        )

    tab_va = clf.predict_proba(Xs[va]) if len(va) else np.empty((0, C.NUM_CLASSES))
    tab_te = clf.predict_proba(Xs[te]) if len(te) else np.empty((0, C.NUM_CLASSES))

    out = {
        "split_kind": split_kind,
        "n_val": int(len(va)), "n_test": int(len(te)),
        "tabular_only": _sc(y[te], tab_te.argmax(1)) if len(te) else None,
        "image_model_available": False,
    }

    img_va, ok_va = _image_probs_for([paths[i] for i in va], ckpt=ckpt) if len(va) else (None, False)
    img_te, ok_te = _image_probs_for([paths[i] for i in te], ckpt=ckpt) if len(te) else (None, False)

    if ok_va and ok_te and len(va) and len(te):
        out["image_model_available"] = True
        out["image_only"] = _sc(y[te], img_te.argmax(1))

        meta = LogisticRegression(max_iter=2000, C=1.0)
        Zva = np.hstack([tab_va, img_va])
        Zte = np.hstack([tab_te, img_te])
        meta.fit(Zva, y[va])
        out["fused"] = _sc(y[te], meta.predict(Zte))
    else:
        out["note"] = ("student.pt absent — tabular-only (U7 fusion gated)"
                       if not (ok_va and ok_te) else "empty val/test split")

    if save:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        existing = {}
        if os.path.isfile(METRICS_PATH):
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except Exception:
                existing = {}
        existing["late_fusion"] = out
        with open(METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
    return out


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def run_all(split_kind="grouped", with_fusion=True, save=True):
    """Full axis-2 chain: load -> train -> SHAP -> permutation -> PDP (+ optional fusion).

    Writes ``results/metrics/feature_explain.json`` and the figures; returns the assembled dict.
    """
    data = load_features(split_kind=split_kind, standardize=False)
    trained = train_tabular(data=data, split_kind=split_kind, save=save)
    shap_out = shap_report(trained=trained, data=data, split_kind=split_kind)
    perm_out = permutation_importance(trained=trained, data=data, split_kind=split_kind)
    pdp_out = partial_dependence(trained=trained, data=data, split_kind=split_kind)

    result = dict(
        n_cells=int(len(data["y"])),
        n_features=len(data["names"]),
        feature_names=data["names"],
        split_sizes={k: int(len(v)) for k, v in data["split"].items()},
        tabular_metrics=trained["metrics"],
        shap_top=shap_out["mean_abs_shap"][:15],
        shap_figure=shap_out["figure"],
        shap_example_reasons=shap_out["example_reasons"],
        permutation_top=perm_out["ranking"][:15],
        permutation_figure=perm_out["figure"],
        pdp_figure=pdp_out["figure"],
        pdp_features=pdp_out["features"],
    )
    if with_fusion:
        result["late_fusion"] = late_fusion(data=data, split_kind=split_kind, save=False)

    if save:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        # companion CSVs next to each value-plot (full provenance: every diagram regenerable)
        import plotting as P
        def _stem(fig):
            return os.path.splitext(fig)[0] if fig else None
        if shap_out.get("figure"):
            P.write_csv(_stem(shap_out["figure"]), ["feature", "mean_abs_shap"],
                        [[r["feature"], r["mean_abs_shap"]] for r in shap_out["mean_abs_shap"]])
        if perm_out.get("figure"):
            P.write_csv(_stem(perm_out["figure"]), ["feature", "importance_mean", "importance_std"],
                        [[r["feature"], r["importance_mean"], r.get("importance_std", "")] for r in perm_out["ranking"]])
        if pdp_out.get("figure"):
            P.write_csv(_stem(pdp_out["figure"]), ["pdp_feature"], [[f] for f in pdp_out["features"]])
    return result


if __name__ == "__main__":
    # CPU-only chain (no image model): load + train + SHAP, plus the saved metrics JSON.
    C.set_seed()
    data = load_features(split_kind="grouped", standardize=False)
    print(f"loaded {len(data['y'])} cells x {len(data['names'])} features; "
          f"split sizes = {{train:{len(data['split']['train'])}, "
          f"val:{len(data['split']['val'])}, test:{len(data['split']['test'])}}}")

    trained = train_tabular(data=data, save=True)
    m = trained["metrics"]
    print("tabular XGBoost  "
          f"train acc={m['train']['accuracy']:.3f}  "
          f"val acc={m['val']['accuracy']:.3f} F1={m['val']['macro_f1']:.3f}  "
          f"test acc={m['test']['accuracy']:.3f} F1={m['test']['macro_f1']:.3f}")

    sh = shap_report(trained=trained, data=data)
    print("SHAP beeswarm ->", sh["figure"])
    print("top features:", [d["feature"] for d in sh["mean_abs_shap"][:6]])
    if sh["example_reasons"]:
        print("example reason:", sh["example_reasons"][0]["reason"])

    # persist the __main__ summary so feature_explain.json exists after a CPU run
    summary = dict(
        tabular_metrics=trained["metrics"],
        shap_top=sh["mean_abs_shap"][:15],
        shap_figure=sh["figure"],
        shap_example_reasons=sh["example_reasons"],
    )
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("metrics ->", METRICS_PATH)
