"""House SciencePlots style for the new pipeline (same style guide as Legacy_src/plotting.py).

publication-figures house style: science+ieee+no-latex, bold serif, semantic palette, PDF+PNG@600dpi.
Figure helpers return saved paths; never restyle per-figure.
"""
import os
from datetime import datetime as _dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
    SCIENCEPLOTS = True
except Exception:
    SCIENCEPLOTS = False

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 16, "font.weight": "bold",
    "axes.labelsize": 18, "axes.labelweight": "bold", "axes.titlesize": 18, "axes.titleweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "legend.title_fontsize": 14, "figure.titlesize": 20,
    "axes.linewidth": 1.5, "grid.linewidth": 1.0, "lines.linewidth": 2.4, "patch.linewidth": 1.2,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5, "xtick.major.size": 6, "ytick.major.size": 6,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "black", "legend.fancybox": True,
})

PALETTE = {
    "teacher": "#2E86AB", "student": "#A23B72", "train": "#2E86AB", "val": "#F18F01", "test": "#D32F2F",
    "baseline": "#7F7F7F", "random": "#7F7F7F", "oracle": "#222222",
    "Superficial-Intermediate": "#2E86AB", "Parabasal": "#5DA9C7", "Metaplastic": "#2ECC71",
    "Koilocytotic": "#F18F01", "Dyskeratotic": "#A23B72",
    "Normal": "#2E86AB", "Abnormal": "#D32F2F", "normal": "#2E86AB", "abnormal": "#D32F2F",
    "low": "#F18F01", "high": "#D32F2F",
}
_CYCLE = ["#2E86AB", "#A23B72", "#F18F01", "#2ECC71", "#5DA9C7", "#D32F2F", "#7B6D8D", "#222222"]

def color(name, i=0):
    return PALETTE.get(name, _CYCLE[i % len(_CYCLE)])

PRETTY = {"Superficial-Intermediate": "Superf.-Interm.", "loss": "Loss", "accuracy": "Accuracy",
          "val_loss": "Val loss", "ece": "ECE", "auroc": "AUROC", "macro_f1": "Macro-F1"}
def pretty(k): return PRETTY.get(k, k)

# --------------------------------------------------------------------------
# Figure provenance: every saved figure is stamped with WHEN it was generated and
# WHICH checkpoint it came from. A driver that loads a model checkpoint binds the
# provenance once (ckpt.load_into does this automatically) so every figure it emits
# carries a footer + a <fig>.meta.json sidecar, and an append-only MANIFEST.jsonl in
# the figures dir records the full history. Filenames stay STABLE (timestamp lives in
# the footer/sidecar, not the name) so report/notebook references never break.
# --------------------------------------------------------------------------
_PROVENANCE = {}

def set_provenance(checkpoint=None, checkpoint_saved_at=None, **extra):
    """Bind provenance applied to every figure saved afterwards (footer + sidecar).
    Drivers call this right after loading a checkpoint; ckpt.load_into() calls it for you."""
    _PROVENANCE.clear()
    if checkpoint is not None: _PROVENANCE["checkpoint"] = os.path.basename(str(checkpoint))
    if checkpoint_saved_at is not None: _PROVENANCE["checkpoint_saved_at"] = str(checkpoint_saved_at)
    _PROVENANCE.update({k: v for k, v in extra.items() if v is not None})

def clear_provenance():
    """Forget the bound checkpoint (e.g. before EDA figures that come from no model)."""
    _PROVENANCE.clear()

def _stamp_footer(fig, generated_at, prov):
    bits = [f"generated {generated_at}"]
    ck = prov.get("checkpoint")
    if ck:
        sa = prov.get("checkpoint_saved_at")
        bits.append(f"ckpt: {ck}" + (f" @ {sa}" if sa else ""))
    fig.text(0.997, 0.003, "   |   ".join(bits), ha="right", va="bottom",
             fontsize=5.5, color="#9A9A9A", style="italic", zorder=1000)

def _write_meta_and_manifest(out_noext, rec):
    with open(out_noext + ".meta.json", "w", encoding="utf-8") as f:
        _json.dump(rec, f, indent=2)
    man = os.path.join(os.path.dirname(out_noext) or ".", "MANIFEST.jsonl")
    with open(man, "a", encoding="utf-8") as f:
        f.write(_json.dumps(rec) + "\n")

def save(fig, out_noext, meta=None):
    """Save fig as PDF+PNG, stamp a generated-at + source-checkpoint footer, and write a
    <out>.meta.json provenance sidecar + append to MANIFEST.jsonl. `meta` overrides the
    globally-bound provenance (set_provenance) for this one figure."""
    os.makedirs(os.path.dirname(out_noext) or ".", exist_ok=True)
    generated_at = _dt.now().isoformat(timespec="seconds")
    prov = dict(_PROVENANCE)
    if meta: prov.update({k: v for k, v in meta.items() if v is not None})
    try:
        _stamp_footer(fig, generated_at, prov)
    except Exception:
        pass
    fig.savefig(out_noext + ".pdf"); fig.savefig(out_noext + ".png"); plt.close(fig)
    rec = {"figure": os.path.basename(out_noext), "generated_at": generated_at}
    rec.update(prov)
    try:
        _write_meta_and_manifest(out_noext, rec)
    except Exception:
        pass
    return out_noext + ".png"


# --------------------------------------------------------------------------
# Provenance: EVERY figure persists the exact data it plots, so each diagram is
# regenerable from a .csv (tabular) or .json (structured) sitting beside it.
# --------------------------------------------------------------------------
import csv as _csv
import json as _json


def write_csv(out_noext, header, rows):
    """Write rows (list of sequences) to <out_noext>.csv with the given header."""
    os.makedirs(os.path.dirname(out_noext) or ".", exist_ok=True)
    with open(out_noext + ".csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(list(r))
    return out_noext + ".csv"


def write_json(out_noext, obj):
    os.makedirs(os.path.dirname(out_noext) or ".", exist_ok=True)
    def _d(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(out_noext + ".json", "w", encoding="utf-8") as f:
        _json.dump(obj, f, indent=2, default=_d)
    return out_noext + ".json"

def _smooth(ys, k=3):
    """Centered rolling mean (edge-aware) for the jagged, small-val curves."""
    ys = np.asarray(ys, float)
    if len(ys) < 2 or k <= 1:
        return ys
    half = k // 2
    return np.array([ys[max(0, i - half):min(len(ys), i + half + 1)].mean() for i in range(len(ys))])

def training_curves(history, out, title="Training history", smooth_k=3):
    """Train/val curves. Val series are drawn raw-but-faded plus a bold rolling-mean overlay (the honest
    leakage-safe val set is small, so per-epoch val is jagged; the smoothed line shows the trend)."""
    has_acc = "accuracy" in history or "val_accuracy" in history
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12.6, 5.2) if has_acc else (7.4, 5.6))
    axes = np.atleast_1d(axes)
    def _ep(seq): return range(1, len(seq) + 1)          # per-series x-axis (handles unequal lengths)
    ax = axes[0]
    if "loss" in history:     ax.plot(_ep(history["loss"]), history["loss"], color=color("train"), marker="o", ms=4, label="train loss")
    if "val_loss" in history:
        ax.plot(_ep(history["val_loss"]), history["val_loss"], color=color("val"), lw=1.1, alpha=0.30)     # raw (faded)
        ax.plot(_ep(history["val_loss"]), _smooth(history["val_loss"], smooth_k), color=color("val"), marker="s", ms=4, label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="best")
    if has_acc:
        ax = axes[1]
        if "accuracy" in history:     ax.plot(_ep(history["accuracy"]), history["accuracy"], color=color("train"), marker="o", ms=4, label="train acc")
        if "val_accuracy" in history:
            ax.plot(_ep(history["val_accuracy"]), history["val_accuracy"], color=color("val"), lw=1.1, alpha=0.30)   # raw (faded)
            ax.plot(_ep(history["val_accuracy"]), _smooth(history["val_accuracy"], smooth_k), color=color("val"), marker="s", ms=4, label="val acc")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy")
        ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="best")
    fig.suptitle(title, fontweight="bold", y=1.03)
    # provenance: dump the full per-epoch history as CSV (raw + smoothed val columns)
    keys = [k for k in ("loss", "val_loss", "accuracy", "val_accuracy", "lr", "epoch_time") if k in history]
    n = max((len(history[k]) for k in keys), default=0)
    sm = {f"{k}_smooth": _smooth(history[k], smooth_k) for k in ("val_loss", "val_accuracy") if k in history}
    smk = list(sm)
    rows = [[e + 1] + [history[k][e] if e < len(history[k]) else "" for k in keys]
            + [sm[k][e] if e < len(sm[k]) else "" for k in smk] for e in range(n)]
    write_csv(out, ["epoch"] + keys + smk, rows)
    return save(fig, out)

_CM_CMAPS = {"teacher": "Greens", "student": "Blues", "herlev": "Oranges",
             "apacc": "Purples", "hicervix": "Reds", "cric": "YlGnBu"}

def confusion_matrix(cm, labels, out, title="Confusion matrix", normalize=False, cmap=None):
    cm = np.asarray(cm, dtype=float)
    if normalize: cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    if cmap is None:                                   # distinct colormap per figure (by output name)
        base = os.path.basename(out).lower()
        cmap = next((v for k, v in _CM_CMAPS.items() if k in base), "Blues")
    fig, ax = plt.subplots(figsize=(7.2, 6.4)); im = ax.imshow(cm, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([pretty(l) for l in labels], rotation=40, ha="right")
    ax.set_yticklabels([pretty(l) for l in labels]); thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]; fg = "white" if v > thr else "black"
            ax.text(j, i, (f"{v:.2f}" if normalize else f"{int(v)}"), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=fg,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="black" if fg == "white" else "white")])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    # provenance: dump the matrix as CSV (row=actual, col=predicted)
    write_csv(out, ["actual\\predicted"] + [str(l) for l in labels],
              [[str(labels[i])] + list(cm[i]) for i in range(cm.shape[0])])
    return save(fig, out)

def class_balance(counts, out, ylabel="cells", title="Class balance"):
    names = list(counts); vals = [counts[n] for n in names]; xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.bar(xs, vals, color=[color(n, i) for i, n in enumerate(names)], edgecolor="black", linewidth=1.0)
    for x_, v in zip(xs, vals):
        ax.text(x_, v, f"{int(v)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([pretty(n) for n in names], rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    write_csv(out, ["class", ylabel], [[n, counts[n]] for n in names])
    return save(fig, out)

def roc_curves(curves, out, title="ROC"):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=color(name, i), label=f"{pretty(name)} (AUC={auc:.3f})", linewidth=2.6)
    ax.plot([0, 1], [0, 1], ls="--", color="#7F7F7F", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="lower right")
    # provenance: full fpr/tpr arrays per curve (json) + a compact AUC table (csv)
    write_json(out, {name: {"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr)), "auc": float(auc)}
                     for name, (fpr, tpr, auc) in curves.items()})
    write_csv(out, ["curve", "auc"], [[name, float(auc)] for name, (_, _, auc) in curves.items()])
    return save(fig, out)

def reliability(prob_pred, prob_true, out, title="Reliability", label="model"):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], ls=":", color="#222222", label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", ms=6, color=color("teacher", 0), label=pretty(label))
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title(title)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="best")
    write_csv(out, ["bin_predicted_prob", "observed_frequency"],
              [[float(a), float(b)] for a, b in zip(prob_pred, prob_true)])
    return save(fig, out)

def bars(data, out, ylabel="value", title=""):
    names = list(data); vals = [data[n] for n in names]; xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.bar(xs, vals, color=[color(n, i) for i, n in enumerate(names)], edgecolor="black", linewidth=1.0)
    for x_, v in zip(xs, vals):
        ax.text(x_, v, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([pretty(n) for n in names], rotation=18, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    write_csv(out, ["name", ylabel], [[n, data[n]] for n in names])
    return save(fig, out)

def line_plot(xs, series, out, xlabel="x", ylabel="value", title="", baseline=None):
    """Generic metric-vs-parameter line plot (e.g. CRIC metric vs crop size). `series` maps name->y-list.
    Optional `baseline` draws a dashed grey reference line. Writes a companion CSV."""
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(xs, ys, color=color(name, i), marker="o", ms=6, linewidth=2.6, label=pretty(name))
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="#7F7F7F", label=f"chance ({baseline})")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="best")
    names = list(series)
    write_csv(out, [xlabel] + names, [[xs[j]] + [series[n][j] for n in names] for j in range(len(xs))])
    return save(fig, out)

def heatmap(M, row_labels, col_labels, out, title="", cmap="coolwarm", center0=True, annot=False,
            cbar_label="", xrot=90):
    """Annotated matrix heatmap (class profiles / effect sizes). Diverging+centered by default."""
    M = np.asarray(M, float)
    fig, ax = plt.subplots(figsize=(min(0.30 * len(col_labels) + 3, 22), 0.55 * len(row_labels) + 2.2))
    vmax = float(np.nanmax(np.abs(M))) if center0 and np.isfinite(M).any() else None
    im = ax.imshow(M, cmap=cmap, aspect="auto",
                   vmin=(-vmax if center0 and vmax else None), vmax=(vmax if center0 and vmax else None))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    if cbar_label: cb.set_label(cbar_label, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels([pretty(c) for c in col_labels], rotation=xrot, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels([pretty(r) for r in row_labels], fontsize=10, fontweight="bold")
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    write_csv(out, ["row"] + [str(c) for c in col_labels], [[row_labels[i]] + list(M[i]) for i in range(M.shape[0])])
    return save(fig, out)


def embedding_scatter(coords, labels, out, label_names=None, title="", xlabel="dim 1", ylabel="dim 2"):
    """2-D embedding (PCA/LDA/t-SNE) coloured by class label."""
    coords = np.asarray(coords, float); labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for i, u in enumerate(sorted(set(labels.tolist()))):
        m = labels == u
        nm = (label_names[u] if label_names is not None else str(u))
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.55, color=color(nm, i),
                   label=pretty(nm), edgecolors="none")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="best", fontsize=9, markerscale=1.6)
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    write_csv(out, [xlabel, ylabel, "label"], [[coords[k, 0], coords[k, 1], int(labels[k])] for k in range(len(labels))])
    return save(fig, out)


def violins(groups, out, ylabel="value", title="", xlabel=""):
    """Per-group distribution violins (e.g. a feature split by class). groups: dict name -> 1-D array."""
    names = list(groups); data = [np.asarray(groups[n], float) for n in names]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(names) + 2), 5.6))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(color(names[i], i)); b.set_alpha(0.6); b.set_edgecolor("black")
    if "cmeans" in parts: parts["cmeans"].set_color("black")
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels([pretty(n) for n in names], rotation=18, ha="right")
    ax.set_ylabel(ylabel); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    write_csv(out, ["group", "n", "mean", "std", "median"],
              [[n, len(g), float(np.mean(g)), float(np.std(g)), float(np.median(g))] for n, g in zip(names, data)])
    return save(fig, out)

def montage(images, titles, out, suptitle="", ncol=None):
    """Grid of RGB sample images (EDA). `images` are HxWx3 float[0,1] arrays."""
    n = len(images)
    if n == 0: return None
    ncol = ncol or min(n, 6); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.7 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1))
            ax.set_title(titles[i] if i < len(titles) else "", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    if suptitle: fig.suptitle(suptitle, fontweight="bold", y=1.02)
    return save(fig, out)
