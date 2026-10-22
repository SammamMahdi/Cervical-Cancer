"""Figures for the revision: assumption violations, external validation, paired-data requirement, subgroups.

Run: python paper_jbhi/scripts/make_figures2.py [robust|external|subgroups|all]
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pubstyle import plt, save, color, grid, size  # noqa: E402
from sklearn.metrics import roc_curve, auc  # noqa: E402

RES = os.path.join(HERE, "..", "results")
J = lambda f: json.load(open(os.path.join(RES, f)))
PROP, IMG, ORC = "proposed", "image only", "oracle (paired)"
COL = {PROP: color("fused"), IMG: color("image"), ORC: color("oracle"), "sum rule": "#7B6D8D",
       "naive mean": color("baseline"), "questionnaire only": color("tabular")}


def _series(block, key, method, metric="auroc"):
    xs = [r["setting"].get(key, 0.0) for r in block]
    ys = [r["methods"][method][metric] for r in block]
    return np.array(xs), np.array(ys)


def fig_robust():
    R = J("fusion_robust.json")
    fig, axs = plt.subplots(1, 3, figsize=size(11, 4.1))

    ax = axs[0]                                   # V1: the assumption that matters
    for m in (ORC, IMG, PROP, "sum rule"):
        x, y = _series(R["V1_dependence"], "rho", m)
        ax.plot(x, y, marker="o", ms=5, color=COL[m], linewidth=2.4 if m == PROP else 1.9,
                linestyle=":" if m == ORC else "-", label=m)
    ax.set_xlabel("residual dependence $\\rho$"); ax.set_ylabel("AUROC")
    ax.set_title("Dependence given the\nlatent state", fontsize=8.7)
    ax.set_ylim(0.86, 1.02)
    ax.legend(loc="upper left", framealpha=0.95); grid(ax)

    ax = axs[1]                                   # where the proposed rule stops paying
    x, y = _series(R["V1_dependence"], "rho", PROP)
    _, yi = _series(R["V1_dependence"], "rho", IMG)
    _, yo = _series(R["V1_dependence"], "rho", ORC)
    ax.axhline(0, color=color("random"), linestyle="--", linewidth=1.5)
    ax.plot(x, y - yi, marker="o", ms=5, color=COL[PROP], linewidth=2.4, label="proposed $-$ image")
    ax.plot(x, yo - yi, marker="s", ms=5, color=COL[ORC], linestyle=":", linewidth=2.0, label="oracle $-$ image")
    cross = x[np.argmax((y - yi) < 0)] if np.any((y - yi) < 0) else None
    if cross is not None:
        ax.axvline(cross, color=color("baseline"), linestyle="-.", linewidth=1.6)
        ax.text(cross + 0.03, 0.045, f"fusion stops\nhelping at\n$\\rho\\approx${cross:.1f}",
                fontsize=8, color=color("baseline"))
    ax.set_xlabel("residual dependence $\\rho$"); ax.set_ylabel("$\\Delta$AUROC vs image only")
    ax.set_title("When fusion stops\nhelping", fontsize=8.7)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.32 * (hi - lo), hi)
    ax.legend(loc="lower left"); grid(ax)

    ax = axs[2]                                   # calibration: what re-anchoring can and cannot fix
    base = R["V1_dependence"][0]["methods"][PROP]["ece"]
    mis = [r for r in R["V5_miscalibration"] if r["setting"].get("mis_i") == 2.0][0]["methods"][PROP]["ece"]
    dom = [r for r in R["V6_domain_prevalence"] if r["setting"].get("img_bias") == 1.0
           and abs(r["setting"].get("prev") - R["margins"]["p_A"]) < 1e-9][0]["methods"][PROP]["ece"]
    domp = [r for r in R["V6_domain_prevalence"] if r["setting"].get("img_bias") == 1.0
            and r["setting"].get("prev") == 0.02][0]["methods"][PROP]["ece"]
    labels = ["none", "miscal.\nimage", "domain\nshift", "domain\n+ prev."]
    vals = [base, mis, dom, domp]
    ax.bar(range(4), vals, color=[color("coprior"), color("tabular"), color("baseline"), "#7B1FA2"],
           edgecolor="black", linewidth=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("violation applied"); ax.set_ylabel("ECE of the fused output")
    ax.set_title("What re-anchoring\ncannot fix", fontsize=8.7)
    grid(ax, "y"); ax.set_ylim(0, max(vals) * 1.28)
    save(fig, "fus_robust")


def fig_paired_need():
    R = J("fusion_robust.json")
    n = [r["n_paired"] for r in R["paired_data_requirement"]]
    d = [r["oracle_minus_proposed"]["mean"] for r in R["paired_data_requirement"]]
    s = [r["oracle_minus_proposed"]["sd"] for r in R["paired_data_requirement"]]
    fig, ax = plt.subplots(figsize=size(5.9, 3.9))
    ax.axhline(0, color=color("random"), linestyle="--", linewidth=1.6)
    ax.errorbar(n, d, yerr=s, marker="o", ms=6, color=color("oracle"), linewidth=2.4, capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("paired patients available to the stacker")
    ax.set_ylabel("stacker $-$ rule ($\\Delta$AUROC)")
    ax.annotate("stacker worse than\nthe unpaired rule", xy=(n[0], d[0]), xytext=(4e2, min(d) * 1.9),
                fontsize=8, color=color("baseline"), ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=color("baseline"), lw=0.9,
                                connectionstyle="arc3,rad=0.2"))
    grid(ax); save(fig, "fus_paired_need")


def fig_external():
    E = J("external_eval.json")
    d = np.load(os.path.join(RES, "external_protocolB.npz"), allow_pickle=True)
    y, p, cls = d["y"], d["p"], np.array([str(c) for c in d["cls"]])
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.0))

    ax = axs[0]
    for tag, key, col, ls in (("in-distribution (SIPaKMeD)", None, color("image"), "-"),
                              ("external (Herlev)", "p", color("baseline"), "-")):
        if key is None:
            g = np.load(os.path.join(RES, "grouped_preds.npz"))
            yy = np.isin(g["y_test"], [0, 1]).astype(int); pp = g["P_student"][:, [0, 1]].sum(1)
        else:
            yy, pp = y, p
        fpr, tpr, _ = roc_curve(yy, pp)
        ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=2.6, label=f"{tag} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=color("random"), linestyle="--", linewidth=1.5)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Same model, two data sets", fontsize=9.3); ax.legend(loc="lower right", fontsize=8); grid(ax)

    ax = axs[1]
    order = ["normal_superficiel", "normal_intermediate", "normal_columnar",
             "light_dysplastic", "moderate_dysplastic", "severe_dysplastic", "carcinoma_in_situ"]
    short = ["superf.", "interm.", "columnar", "light\ndyspl.", "moderate\ndyspl.", "severe\ndyspl.", "carcinoma\nin situ"]
    data = [p[cls == c] for c in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black", linewidth=1.6))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(color("image") if i < 3 else color("baseline")); b.set_edgecolor("black")
    ax.axhline(0.5, color=color("oracle"), linestyle=":", linewidth=1.8)
    ax.set_xticks(range(1, 8)); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("predicted P(abnormal)"); ax.set_title("Scores by Herlev class", fontsize=9.3); grid(ax, "y")
    save(fig, "img_external")


def fig_subgroups():
    T = J("tabular_v2.json")
    sub = {k: v for k, v in T["subgroups"].items() if not k.startswith("cycle")}
    names = list(sub)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.2))
    overall = T["nested_selection"]["calibrated"]
    for ax, met, lab in ((axs[0], "auroc", "AUROC"), (axs[1], "ece", "ECE")):
        v = [sub[n][met] for n in names]
        ax.barh(names, v, color=color("tabular"), edgecolor="black", linewidth=0.6)
        lo = min(v) - 0.06 if met == "auroc" else 0.0
        ax.set_xlim(lo, max(v) * 1.16)
        ax.axvline(overall[met], color=color("oracle"), linestyle="--", linewidth=1.2)
        # values sit inside the bar so they never collide with the overall reference line
        span = ax.get_xlim()[1] - lo
        for i, val in enumerate(v):
            inside = (val - lo) > 0.28 * span
            ax.text(val - 0.012 * span if inside else val + 0.012 * span, i, f"{val:.3f}",
                    va="center", ha="right" if inside else "left", fontsize=8,
                    color="white" if inside else "black")
        ax.annotate(f"overall {overall[met]:.3f}", xy=(overall[met], len(v) - 0.75),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, va="center",
                    color=color("oracle"), bbox=dict(fc="white", ec="none", pad=0.8))
        ax.set_xlabel(lab); grid(ax, "x")
        ax.tick_params(axis="y", labelsize=8)
    save(fig, "tab_subgroups")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("robust", "all"):
        fig_robust(); fig_paired_need()
    if what in ("external", "all"):
        fig_external()
    if what in ("subgroups", "all") and os.path.exists(os.path.join(RES, "tabular_v2.json")):
        fig_subgroups()
    print("done")
