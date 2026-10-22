"""Publication figure house style (publication-figures skill): SciencePlots base, bold serif,
semantic palette, vector PDF + 600-dpi PNG export. Import once; figure functions only place data."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
except Exception:
    pass

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "font.weight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold", "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "legend.title_fontsize": 8, "figure.titlesize": 10,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5, "lines.linewidth": 1.5, "patch.linewidth": 0.7,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "black", "legend.fancybox": True,
    "legend.borderpad": 0.35, "legend.labelspacing": 0.3, "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.4, "legend.columnspacing": 1.0,
    "figure.constrained_layout.use": True, "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.wspace": 0.06,
    "figure.constrained_layout.hspace": 0.06,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# IEEEtran journal geometry: figures are authored at the size they are printed at, never rescaled.
WIDE, COL = 7.16, 3.45


def size(w, h):
    """Map an authored (w, h) in inches onto the printed column or text width, keeping the aspect."""
    k = (WIDE if w > 8 else COL) / w
    return (w * k, h * k)

# colour = meaning, fixed across every figure in the paper
PALETTE = {
    "image": "#2E86AB", "teacher": "#5DA9C7", "student": "#2E86AB",
    "tabular": "#A23B72", "fused": "#F18F01", "coprior": "#2ECC71",
    "train": "#2E86AB", "val": "#A23B72", "test": "#F18F01",
    "baseline": "#D32F2F", "random": "#7F7F7F", "oracle": "#222222", "muted": "#7B6D8D",
}
_CYCLE = ["#2E86AB", "#A23B72", "#F18F01", "#2ECC71", "#7B6D8D", "#D32F2F", "#7F7F7F", "#222222"]
CLASS_COLORS = ["#D32F2F", "#F18F01", "#2ECC71", "#2E86AB", "#7B6D8D"]   # Dysk, Koilo, Metap, Parab, Sup-Int


def color(name, i=0):
    return PALETTE.get(name, _CYCLE[i % len(_CYCLE)])


FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, stem)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png")
    plt.close(fig)
    return out + ".png"


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
