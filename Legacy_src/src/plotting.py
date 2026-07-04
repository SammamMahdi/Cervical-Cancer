"""House SciencePlots style for the cervical-cytology pipeline.

Implements the `publication-figures` skill style (SciencePlots science+ieee+no-latex,
bold serif, thick axes, semantic palette, PDF+PNG @600 dpi). Import this module and use
its figure helpers; never restyle per-figure. All figures land under Legacy_src/results/figures/.
"""
import os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless/batch-safe; before pyplot
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---- 1. style stack (fail gracefully) --------------------------------------
try:
    import scienceplots  # noqa: F401  (registers styles)
    plt.style.use(["science", "ieee", "no-latex"])
    SCIENCEPLOTS = True
except Exception:
    SCIENCEPLOTS = False

# ---- 2. rcParams (bold serif, thick axes, high-DPI) ------------------------
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16, "font.weight": "bold",
    "axes.labelsize": 18, "axes.labelweight": "bold",
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13, "legend.title_fontsize": 14, "figure.titlesize": 20,
    "axes.linewidth": 1.5, "grid.linewidth": 1.0, "lines.linewidth": 2.4,
    "patch.linewidth": 1.2,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "black", "legend.fancybox": True,
})

# ---- 3. semantic palette (color = meaning) ---------------------------------
PALETTE = {
    # pipeline roles
    "teacher":   "#2E86AB",   # EfficientNet-B4 teacher
    "student":   "#A23B72",   # distilled EfficientNet-B0 student
    "train":     "#2E86AB",   # training curve
    "val":       "#F18F01",   # validation curve
    "test":      "#D32F2F",   # held-out / external test
    "baseline":  "#7F7F7F",
    "random":    "#7F7F7F",
    "oracle":    "#222222",
    # SIPaKMeD 5 morphology classes (fixed)
    "Superficial-Intermediate": "#2E86AB",
    "Parabasal":                "#5DA9C7",
    "Metaplastic":              "#2ECC71",
    "Koilocytotic":             "#F18F01",
    "Dyskeratotic":             "#A23B72",
    # binary
    "Normal":   "#2E86AB",
    "Abnormal": "#D32F2F",
}
_CYCLE = ["#2E86AB", "#A23B72", "#F18F01", "#2ECC71", "#5DA9C7", "#D32F2F", "#7B6D8D", "#222222"]

def color(name, i=0):
    return PALETTE.get(name, _CYCLE[i % len(_CYCLE)])

PRETTY = {
    "Superficial-Intermediate": "Superf.-Interm.",
    "loss": "Loss", "accuracy": "Accuracy", "val_loss": "Val loss", "train_loss": "Train loss",
    "ece": "ECE", "auroc": "AUROC", "macro_f1": "Macro-F1", "bal_acc": "Balanced acc.",
}
def pretty(k):
    return PRETTY.get(k, k)

# ---- 4. save (PDF + PNG, always close) -------------------------------------
def save(fig, out_noext):
    os.makedirs(os.path.dirname(out_noext) or ".", exist_ok=True)
    fig.savefig(out_noext + ".pdf")
    fig.savefig(out_noext + ".png")
    plt.close(fig)
    return out_noext + ".png"

# ---- figure helpers --------------------------------------------------------
def training_curves(history, out, title="Training history"):
    """history: dict with keys like 'loss','val_loss','accuracy','val_accuracy' (lists)."""
    has_acc = "accuracy" in history or "val_accuracy" in history
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(12.6, 5.2) if has_acc else (7.4, 5.6))
    axes = np.atleast_1d(axes)
    ep = range(1, len(history.get("loss", history.get("val_loss", []))) + 1)
    ax = axes[0]
    if "loss" in history:     ax.plot(ep, history["loss"], color=color("train"), marker="o", ms=4, label="train loss")
    if "val_loss" in history: ax.plot(ep, history["val_loss"], color=color("val"), marker="s", ms=4, label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="best")
    if has_acc:
        ax = axes[1]
        if "accuracy" in history:     ax.plot(ep, history["accuracy"], color=color("train"), marker="o", ms=4, label="train acc")
        if "val_accuracy" in history: ax.plot(ep, history["val_accuracy"], color=color("val"), marker="s", ms=4, label="val acc")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy")
        ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True); ax.legend(loc="best")
    fig.suptitle(title, fontweight="bold", y=1.03)
    return save(fig, out)

# distinct colormap per figure so confusion matrices are visually distinguishable at a glance
_CM_CMAPS = {"teacher": "Greens", "student": "Blues", "herlev": "Oranges",
             "apacc": "Purples", "hicervix": "Reds", "cric": "YlGnBu"}

def confusion_matrix(cm, labels, out, title="Confusion matrix", normalize=False, cmap=None):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    if cmap is None:                                   # pick by output name -> each CM a different color
        base = os.path.basename(out).lower()
        cmap = next((v for k, v in _CM_CMAPS.items() if k in base), "Blues")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cm, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([pretty(l) for l in labels], rotation=40, ha="right")
    ax.set_yticklabels([pretty(l) for l in labels])
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            fg = "white" if v > thr else "black"
            # opposite-color halo so the number is legible on ANY cell shade
            ax.text(j, i, (f"{v:.2f}" if normalize else f"{int(v)}"),
                    ha="center", va="center", fontsize=14, fontweight="bold", color=fg,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="black" if fg == "white" else "white")])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    return save(fig, out)

def class_balance(counts, out, ylabel="cells", title="Class balance"):
    """counts: {class_name: int}."""
    names = list(counts); vals = [counts[n] for n in names]; xs = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.bar(xs, vals, color=[color(n, i) for i, n in enumerate(names)],
           edgecolor="black", linewidth=1.0)
    for x_, v in zip(xs, vals):
        ax.text(x_, v, f"{int(v)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([pretty(n) for n in names], rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    return save(fig, out)

def roc_curves(curves, out, title="ROC"):
    """curves: {name: (fpr, tpr, auc)}."""
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=color(name, i), label=f"{pretty(name)} (AUC={auc:.3f})", linewidth=2.6)
    ax.plot([0, 1], [0, 1], ls="--", color="#7F7F7F", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    return save(fig, out)

def reliability(prob_pred, prob_true, out, title="Reliability", label="model"):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], ls=":", color="#222222", label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", ms=6, color=color("primary", 0), label=pretty(label))
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title(title)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3, linestyle="--"); ax.set_axisbelow(True)
    ax.legend(loc="best")
    return save(fig, out)

def image_panels(images, titles, out, suptitle=""):
    """images: list of HxWx3 (RGB, 0..1) or HxW (scalar). 1xN panel.
    constrained_layout + wrapped small titles so long 'true -> pred' labels don't overlap neighbours."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, img, t in zip(axes, images, titles):
        if img.ndim == 3:
            ax.imshow(np.clip(img, 0, 1))
        else:
            im = ax.imshow(img, cmap="magma"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("\n".join(textwrap.wrap(str(t), 20)), fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    return save(fig, out)


if __name__ == "__main__":
    # smoke test
    h = {"loss": [1.0, 0.6, 0.4, 0.3], "val_loss": [1.1, 0.7, 0.55, 0.5],
         "accuracy": [0.6, 0.8, 0.88, 0.92], "val_accuracy": [0.55, 0.75, 0.8, 0.83]}
    p = training_curves(h, "results/figures/_smoke_curves")
    class_balance({"Superficial-Intermediate": 813, "Parabasal": 787, "Metaplastic": 793,
                   "Koilocytotic": 825, "Dyskeratotic": 831}, "results/figures/_smoke_balance",
                  title="SIPaKMeD class balance")
    print("plotting OK ->", p, "scienceplots=", SCIENCEPLOTS)
