"""All paper figures from the saved results (publication-figures house style).

Run (Windows python):  python paper_jbhi/scripts/make_figures.py [image|tabular|fusion|grouped|all]
Writes ../figures/<stem>.pdf + .png
"""
import os, sys, json, glob, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pubstyle import plt, save, color, grid, size, CLASS_COLORS, FIGDIR  # noqa: E402
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support  # noqa: E402

RES = os.path.join(HERE, "..", "results"); ROOT = r"E:\Cervaical Cancer"
CLS = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-\nIntermediate"]
CLS_SHORT = ["Dysk.", "Koilo.", "Metap.", "Parab.", "Sup.-Int."]
J = lambda f: json.load(open(os.path.join(RES, f)))


# =============================================================== IMAGE
def fig_sipakmed_counts():
    full, crop = [], []
    for c in ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]:
        d = os.path.join(ROOT, "data", "sipakmed", f"im_{c}", f"im_{c}")
        full.append(len(glob.glob(os.path.join(d, "*.bmp")))); crop.append(len(glob.glob(os.path.join(d, "CROPPED", "*.bmp"))))
    fig, ax = plt.subplots(figsize=size(6.1, 3.69)); x = np.arange(5)
    ax.bar(x, crop, 0.7, color=color("image"), edgecolor="black", linewidth=1.0, label="cropped single cells")
    ax.bar(x, full, 0.7, bottom=crop, color=color("muted"), edgecolor="black", linewidth=1.0, label="full-field images")
    for i in range(5):
        ax.text(i, crop[i] + full[i] + 12, f"{crop[i] + full[i]}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i, crop[i] / 2, f"{crop[i]}", ha="center", va="center", fontsize=8.7, fontweight="bold", color="white")
    ax.set_xticks(x); ax.set_xticklabels(CLS_SHORT, fontsize=9); ax.set_ylabel("images")
    ax.set_ylim(0, max(np.add(crop, full)) * 1.15); ax.legend(loc="lower right", fontsize=8.7); grid(ax, "y")
    save(fig, "img_dataset_counts"); return dict(full=full, crop=crop)


def _square(img):
    """Centre-crop to a square so every panel of the grid has the same aspect and no gaps appear."""
    a = np.asarray(img.convert("RGB")); h, w = a.shape[:2]; k = min(h, w)
    return a[(h - k) // 2:(h - k) // 2 + k, (w - k) // 2:(w - k) // 2 + k]


def fig_sipakmed_samples():
    from PIL import Image
    fig, axs = plt.subplots(2, 5, figsize=size(11, 4.6))
    for j, c in enumerate(["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]):
        d = os.path.join(ROOT, "data", "sipakmed", f"im_{c}", f"im_{c}")
        crops = sorted(glob.glob(os.path.join(d, "CROPPED", "*.bmp"))); fulls = sorted(glob.glob(os.path.join(d, "*.bmp")))
        axs[0, j].imshow(_square(Image.open(crops[len(crops) // 3])))
        axs[1, j].imshow(_square(Image.open(fulls[len(fulls) // 3])))
        axs[0, j].set_title(CLS[j], fontsize=8.7)
        for a in axs[:, j]:
            a.set_xticks([]); a.set_yticks([])
    axs[0, 0].set_ylabel("cropped cell", fontsize=8.7); axs[1, 0].set_ylabel("full field", fontsize=8.7)
    save(fig, "img_dataset_samples")


def _conf(ax, cm, title):
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="cividis", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8.4, fontweight="bold",
                    color="black" if cmn[i, j] > 0.6 else "white")
    ax.set_xticks(range(5)); ax.set_xticklabels(CLS_SHORT, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(5)); ax.set_yticklabels(CLS_SHORT, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return im


def fig_img_confusion():
    E = J("image_eval.json")
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.87))
    _conf(axs[0], np.array(E["teacher"]["confusion"]), f"Teacher B4 (acc {E['teacher']['accuracy']:.3f})")
    im = _conf(axs[1], np.array(E["student"]["confusion"]), f"Student B0 (acc {E['student']['accuracy']:.3f})")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("row-normalised", fontweight="bold")
    save(fig, "img_confusion")


def fig_img_roc():
    d = np.load(os.path.join(RES, "image_preds.npz")); y = d["y_test"]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 5.03))
    P = d["P_student"]; grid_ = np.linspace(0, 1, 400); mean_tpr = np.zeros_like(grid_)
    for i in range(5):
        fpr, tpr, _ = roc_curve(y == i, P[:, i]); mean_tpr += np.interp(grid_, fpr, tpr)
        axs[0].plot(fpr, tpr, color=CLASS_COLORS[i], linewidth=2.2, label=f"{CLS_SHORT[i]} ({auc(fpr, tpr):.3f})")
    mean_tpr /= 5
    axs[0].plot(grid_, mean_tpr, color=color("oracle"), linewidth=2.6, linestyle="-", label=f"macro ({auc(grid_, mean_tpr):.3f})")
    axs[0].set_title("Student B0: one-vs-rest")
    yb = np.isin(y, [0, 1]).astype(int)
    for tag, key in (("teacher", "P_teacher"), ("student", "P_student")):
        pb = d[key][:, [0, 1]].sum(1); fpr, tpr, _ = roc_curve(yb, pb)
        axs[1].plot(fpr, tpr, color=color(tag), linewidth=2.6, linestyle="-" if tag == "student" else "--",
                    label=f"{'Teacher B4' if tag == 'teacher' else 'Student B0'} ({auc(fpr, tpr):.4f})")
    axs[1].set_title("Normal vs abnormal (Bethesda collapse)")
    for a in axs:
        a.plot([0, 1], [0, 1], color=color("random"), linestyle="--", linewidth=1.5)
        a.set_xlabel("false-positive rate"); a.set_ylabel("true-positive rate"); a.legend(loc="lower right", fontsize=8.7)
        a.set_xlim(-0.01, 1); a.set_ylim(0, 1.01); grid(a)
    ins = axs[0].inset_axes([0.42, 0.42, 0.35, 0.3])
    for i in range(5):
        fpr, tpr, _ = roc_curve(y == i, P[:, i]); ins.plot(fpr, tpr, color=CLASS_COLORS[i], linewidth=1.6)
    ins.set_xlim(0, 0.1); ins.set_ylim(0.85, 1.0); ins.tick_params(labelsize=12); ins.set_title("zoom", fontsize=8.4)
    save(fig, "img_roc")


def _reliab(ax, y, P, lab, col, marker):
    conf, pred = P.max(1), P.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, 11); xs, ys, ns = [], [], []
    for i in range(10):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(acc[m].mean()); ns.append(m.sum())
    ax.plot(xs, ys, marker=marker, ms=8, color=col, linewidth=2.4, label=lab)


def _binary_ece(y, p, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def fig_img_reliability():
    """Reliability of the binary P(abnormal) = P(Dysk.) + P(Koilo.) that the fusion consumes; equal-count bins."""
    d = np.load(os.path.join(RES, "image_preds.npz")); yb = np.isin(d["y_test"], [0, 1]).astype(int)
    fig, ax = plt.subplots(figsize=size(6.4, 6.01))
    ax.plot([0, 1], [0, 1], color=color("oracle"), linestyle=":", linewidth=2, label="perfect calibration")
    for key, lab, col, mk in (("P_teacher", "Teacher B4", color("teacher"), "s"), ("P_student", "Student B0", color("student"), "o")):
        p = d[key][:, [0, 1]].sum(1); order = np.argsort(p); chunks = np.array_split(order, 12)
        xs = [p[c].mean() for c in chunks]; ys = [yb[c].mean() for c in chunks]
        ax.plot(xs, ys, marker=mk, ms=8, color=col, linewidth=2.4, linestyle="--" if "Teacher" in lab else "-",
                label=f"{lab} (ECE {_binary_ece(yb, p):.3f})")
    ax.set_xlabel("predicted P(abnormal)"); ax.set_ylabel("observed frequency"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8.7); grid(ax)
    save(fig, "img_reliability")


def fig_img_report():
    d = np.load(os.path.join(RES, "image_preds.npz")); y = d["y_test"]
    fig, axs = plt.subplots(1, 2, figsize=size(11.5, 4.14))
    for ax, key, title in ((axs[0], "P_teacher", "Teacher B4"), (axs[1], "P_student", "Student B0")):
        pred = d[key].argmax(1); p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
        M = np.vstack([p, r, f]).T
        im = ax.imshow(M, cmap="cividis", vmin=0.85, vmax=1.0, aspect="auto")
        for i in range(5):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="black" if M[i, j] > 0.94 else "white")
        ax.set_xticks(range(3)); ax.set_xticklabels(["precision", "recall", "F1"], fontsize=9)
        ax.set_yticks(range(5)); ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(CLS_SHORT, s)], fontsize=9)
        ax.set_title(title)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    save(fig, "img_report")


def fig_training():
    """Learning curves re-drawn in house style from the per-epoch metrics printed in the notebook outputs."""
    import re
    nb = json.load(open(os.path.join(ROOT, "Legacy_src", "efficientnetb4-sipakmed.ipynb"), encoding="utf-8"))

    def out(i):
        return "".join("".join(o.get("text", [])) for o in nb["cells"][i].get("outputs", []))

    def epochs(i):
        """One record per epoch: Keras prints metrics repeatedly while the bar advances, so keep only the
        summary line of each epoch (the one that also carries the validation metrics)."""
        recs = []
        for line in out(i).splitlines():
            if "val_accuracy:" not in line:
                continue
            d = {k: float(v) for k, v in re.findall(r"([a-z_]+): ([0-9.e+-]+)", line)}
            if "accuracy" in d:
                recs.append(d)
        return recs

    def col(recs, k):
        return [r[k] for r in recs if k in r]

    r1, r2, rs = epochs(8), epochs(10), epochs(19)
    t_acc = col(r1, "accuracy") + col(r2, "accuracy"); t_val = col(r1, "val_accuracy") + col(r2, "val_accuracy")
    t_los = col(r1, "loss") + col(r2, "loss"); t_vlo = col(r1, "val_loss") + col(r2, "val_loss")
    n1 = len(r1)
    s_acc = col(rs, "accuracy"); s_val = col(rs, "val_accuracy")
    s_los = col(rs, "student_loss"); s_vlo = col(rs, "val_student_loss"); s_kd = col(rs, "distillation_loss")

    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.5))
    ep = lambda v: np.arange(1, len(v) + 1)
    axs[0].plot(ep(t_acc), t_acc, color=color("teacher"), linestyle="-", linewidth=2.2, label="teacher, train")
    axs[0].plot(ep(t_val), t_val, color=color("teacher"), linestyle="--", linewidth=2.0, marker="s", ms=4, markevery=3, label="teacher, validation")
    axs[0].plot(ep(s_acc), s_acc, color=color("fused"), linestyle="-", linewidth=2.2, label="student, train")
    axs[0].plot(ep(s_val), s_val, color=color("fused"), linestyle="--", linewidth=2.0, marker="o", ms=4, markevery=3, label="student, validation")
    axs[0].axvline(n1 + 0.5, color=color("random"), linestyle=":", linewidth=1.8)
    axs[0].text(n1 + 1.0, 0.655, "backbone\nunfrozen", fontsize=8, color="#555")
    axs[0].set_ylabel("accuracy"); axs[0].set_ylim(0.62, 1.0); axs[0].set_title("Accuracy")
    axs[1].plot(ep(t_los), t_los, color=color("teacher"), linestyle="-", linewidth=2.2, label="teacher, train")
    axs[1].plot(ep(t_vlo), t_vlo, color=color("teacher"), linestyle="--", linewidth=2.0, marker="s", ms=4, markevery=3, label="teacher, validation")
    axs[1].plot(ep(s_los), s_los, color=color("fused"), linestyle="-", linewidth=2.2, label="student, train")
    axs[1].plot(ep(s_vlo), s_vlo, color=color("fused"), linestyle="--", linewidth=2.0, marker="o", ms=4, markevery=3, label="student, validation")
    if s_kd:
        axs[1].plot(ep(s_kd), s_kd, color=color("tabular"), linestyle="-.", linewidth=2.0, label="distillation (KL)")
    axs[1].set_yscale("log"); axs[1].set_ylabel("loss"); axs[1].set_title("Loss")
    for a in axs:
        a.set_xlabel("epoch"); grid(a)
    h, l = axs[1].get_legend_handles_labels()          # one shared legend below, so neither panel is covered
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.1, bbox_to_anchor=(0.5, -0.16), frameon=True)
    save(fig, "img_training")
    print(f"training curves: teacher {len(t_acc)} epochs (stage 1: {n1}), student {len(s_acc)} epochs")


# =============================================================== TABULAR
ONEHOT = {
    "race_1.0": "Mexican American", "race_2.0": "Other Hispanic", "race_3.0": "White",
    "race_4.0": "Black", "race_5.0": "other race",
    "marital_1.0": "married", "marital_2.0": "widowed", "marital_3.0": "divorced",
    "marital_4.0": "separated", "marital_5.0": "never married", "marital_6.0": "living with partner",
    "orientation_1.0": "heterosexual", "orientation_2.0": "homosexual",
    "orientation_3.0": "bisexual", "orientation_4.0": "other orientation",
}
PRETTY = {**ONEHOT, "marital": "marital status", "smoking_status": "smoking status", "partners_per_active_year": "partners / active yr",
          "smoked_100": "smoked >=100 cig.", "age": "age", "ever_sex": "ever had sex", "n_partners_life": "lifetime partners",
          "n_partners_year": "partners (12 mo)", "age_first_sex": "age at first sex", "race": "race/ethnicity",
          "education": "education", "poverty_ratio": "income/poverty ratio", "hsv2": "HSV-2 serostatus",
          "hpv_vaccine": "HPV vaccinated", "years_active": "years sexually active", "n_pregnancies": "pregnancies",
          "ever_pregnant": "ever pregnant", "condomless_year": "condomless sex (12 mo)", "new_partner_year": "new partner (12 mo)",
          "chlamydia_hx": "chlamydia history", "herpes_hx": "herpes history", "gonorrhea_hx": "gonorrhea history",
          "genital_warts_hx": "genital warts history", "age_menarche": "age at menarche", "orientation": "sexual orientation",
          "ever_oral_man": "ever oral sex", "ever_anal_man": "ever anal sex", "age_first_oral": "age first oral sex",
          "n_vaginal_partners_life": "lifetime vaginal partners", "n_oral_partners_life": "lifetime oral partners",
          "pregnancies_per_active_year": "pregnancies / active yr"}
MODEL_ORDER = ["Logistic regression", "RandomForest", "LightGBM", "XGBoost", "ExtraTrees", "k-NN"]
MODEL_KEY = {m: "p_" + m.replace(" ", "_").replace("-", "") for m in MODEL_ORDER}


def fig_tab_cohort():
    sys.path.insert(0, os.path.join(ROOT, "Legacy_src", "src")); import nhanes_data as ND
    X, y, feat, meta = ND.load(verbose=False)
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.24))
    bins = [18, 25, 30, 35, 40, 45, 50, 60]; lab = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-59"]
    a = X["age"].to_numpy(); g = np.digitize(a, bins[1:-1])
    rate = [y[g == i].mean() for i in range(len(lab))]; n = [(g == i).sum() for i in range(len(lab))]
    axs[0].bar(range(len(lab)), rate, color=color("tabular"), edgecolor="black", linewidth=1.0)
    for i, (r, k) in enumerate(zip(rate, n)):
        axs[0].text(i, r + 0.006, f"{r:.2f}", ha="center", va="bottom", fontsize=8.7, fontweight="bold")
    axs[0].set_xticks(range(len(lab))); axs[0].set_xticklabels(lab, rotation=30, fontsize=9)
    axs[0].set_xlabel("age group (years)"); axs[0].set_ylabel("high-risk HPV prevalence"); axs[0].set_title("By age")
    pl = X["n_partners_life"].to_numpy(); cats = [(0, 1, "0-1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 14, "7-14"), (15, 1e9, "15+")]
    rate2 = [y[(pl >= lo) & (pl <= hi)].mean() for lo, hi, _ in cats]
    axs[1].bar(range(len(cats)), rate2, color=color("tabular"), edgecolor="black", linewidth=1.0)
    for i, r in enumerate(rate2):
        axs[1].text(i, r + 0.006, f"{r:.2f}", ha="center", va="bottom", fontsize=8.7, fontweight="bold")
    axs[1].set_xticks(range(len(cats))); axs[1].set_xticklabels([c[2] for c in cats], fontsize=9)
    axs[1].set_xlabel("lifetime male partners"); axs[1].set_title("By lifetime partners")
    for ax in axs:
        ax.axhline(y.mean(), color=color("random"), linestyle="--", linewidth=1.6, label=f"overall {y.mean():.3f}")
        ax.legend(loc="upper left", fontsize=8.7); grid(ax, "y"); ax.set_ylim(0, max(max(rate), max(rate2)) * 1.25)
    save(fig, "tab_cohort")


def fig_tab_models():
    T = J("tabular_eval.json")["results"]
    fig, ax = plt.subplots(figsize=size(11, 4.3)); x = np.arange(len(MODEL_ORDER)); w = 0.38
    for k, (met, col, off) in enumerate((("auroc", color("tabular"), -w / 2), ("pr_auc", color("muted"), w / 2))):
        v = [T[m][met] for m in MODEL_ORDER]; lo = [T[m][met + "_ci95"][0] for m in MODEL_ORDER]; hi = [T[m][met + "_ci95"][1] for m in MODEL_ORDER]
        ax.bar(x + off, v, w, color=col, edgecolor="black", linewidth=1.0, label="AUROC" if met == "auroc" else "PR-AUC",
               yerr=[np.subtract(v, lo), np.subtract(hi, v)], capsize=3, error_kw=dict(elinewidth=1.2, ecolor="#333"))
        for i, val in enumerate(v):
            ax.text(i + off, hi[i] + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax.axhline(0.246, color=color("random"), linestyle="--", linewidth=1.5, label="PR-AUC of chance (prevalence)")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("Logistic regression", "Logistic\nregression") for m in MODEL_ORDER], fontsize=9)
    ax.set_ylim(0.2, 0.83); ax.set_ylabel("5-fold CV score (95% CI)"); ax.legend(loc="upper center", fontsize=8.4, ncol=3); grid(ax, "y")
    save(fig, "tab_models")


def fig_tab_roc():
    d = np.load(os.path.join(RES, "tabular_oof.npz")); y = d["y"]
    fig, ax = plt.subplots(figsize=size(5.8, 5.16))
    for i, m in enumerate(MODEL_ORDER):
        fpr, tpr, _ = roc_curve(y, d[MODEL_KEY[m]])
        ax.plot(fpr, tpr, color=color("tabular") if i == 0 else [None, "#2E86AB", "#F18F01", "#2ECC71", "#7B6D8D", "#7F7F7F"][i],
                linewidth=2.8 if i == 0 else 1.8, label=f"{m} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=color("random"), linestyle="--", linewidth=1.5)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.legend(loc="lower right", fontsize=8.4)
    ax.set_aspect("equal"); grid(ax); save(fig, "tab_roc")


def fig_tab_calibration():
    """Reliability of the revised (one-hot, missingness-aware) questionnaire expert, out of fold."""
    d = np.load(os.path.join(RES, "tabular_v2_oof.npz")); T = J("tabular_v2.json"); y = d["y"]
    fig, ax = plt.subplots(figsize=size(5.0, 5.0))
    ax.plot([0, 1], [0, 1], color=color("oracle"), linestyle=":", linewidth=1.4, label="perfect calibration")
    for p, lab, col, mk in ((d["p_raw"], f"class-weighted, raw (ECE {T['raw_logistic']['ece']:.3f})", color("baseline"), "s"),
                            (d["p_cal"], f"nested + isotonic (ECE {T['nested_selection']['calibrated']['ece']:.3f})", color("tabular"), "o")):
        edges = np.quantile(p, np.linspace(0, 1, 11)); xs, ys = [], []
        for i in range(10):
            m = (p >= edges[i]) & (p <= edges[i + 1])
            if m.sum() > 20:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, marker=mk, ms=4, color=col, linewidth=1.6, label=lab)
    ax.set_xlabel("predicted P(high-risk HPV)"); ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9); ax.set_aspect("equal")
    ax.legend(loc="upper left"); grid(ax); save(fig, "tab_calibration")


def fig_tab_importance():
    """Permutation importance of the revised expert, coloured by the sign of its own coefficient.

    The 27 missingness flags are excluded: they are collinear (all driven by survey cycle) and carry large
    offsetting coefficients, so permuting one breaks a cancellation and overstates its contribution.  Dropping
    all of them costs only 0.004 AUROC, which is reported in the text.
    """
    from matplotlib.patches import Patch
    T = J("tabular_v2.json")
    items = [d for d in T["permutation_importance"] if not d["feature"].endswith("_missing")][:12][::-1]
    up, down = color("baseline"), color("image")
    fig, ax = plt.subplots(figsize=size(5.9, 4.4))
    ax.barh([PRETTY.get(d["feature"], d["feature"].replace("_", " ")) for d in items],
            [d["auroc_drop"] for d in items], xerr=[d["sd"] for d in items],
            color=[up if d["coef"] >= 0 else down for d in items],
            edgecolor="black", linewidth=0.6, capsize=2, error_kw=dict(elinewidth=0.8, ecolor="#333"))
    ax.legend(handles=[Patch(facecolor=up, edgecolor="black", label=r"higher value $\rightarrow$ higher risk"),
                       Patch(facecolor=down, edgecolor="black", label=r"higher value $\rightarrow$ lower risk")],
              loc="lower right")
    ax.set_xlabel("AUROC drop when permuted"); grid(ax, "x")
    save(fig, "tab_importance")
    print("importance:", [(d["feature"], round(d["auroc_drop"], 3)) for d in items[::-1]])


def fig_tab_shap():
    p = os.path.join(RES, "tabular_shap.npz")
    if not os.path.exists(p):
        return
    d = np.load(p, allow_pickle=True); sv, Xs, feat = d["shap"], d["X"], [str(f) for f in d["features"]]
    order = np.argsort(np.abs(sv).mean(0))[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=size(6.8, 5.38)); rng = np.random.default_rng(0)
    for r, j in enumerate(order):
        v = Xs[:, j]; vv = (v - np.nanpercentile(v, 5)) / (np.nanpercentile(v, 95) - np.nanpercentile(v, 5) + 1e-9)
        sc = ax.scatter(sv[:, j], r + rng.normal(0, 0.12, len(v)), c=np.clip(vv, 0, 1), cmap="coolwarm", s=9, alpha=0.8, linewidths=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([PRETTY.get(feat[j], feat[j]) for j in order], fontsize=9)
    ax.axvline(0, color=color("oracle"), linewidth=1.2); ax.set_xlabel("SHAP value (impact on log-odds of HPV+)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.set_label("feature value", fontweight="bold"); grid(ax, "x"); save(fig, "tab_shap")


def fig_tab_decision():
    d = np.load(os.path.join(RES, "tabular_oof.npz")); y = d["y"]; p = d["p_cal"]
    thr = np.linspace(0.05, 0.6, 60); n = len(y); prev = y.mean()
    nb = [np.sum((p >= t) & (y == 1)) / n - np.sum((p >= t) & (y == 0)) / n * t / (1 - t) for t in thr]
    nb_all = [prev - (1 - prev) * t / (1 - t) for t in thr]
    fig, ax = plt.subplots(figsize=size(6.8, 5.15))
    ax.plot(thr, nb, color=color("tabular"), linewidth=2.6, label="questionnaire expert")
    ax.plot(thr, nb_all, color=color("random"), linestyle="--", linewidth=1.8, label="test everyone")
    ax.axhline(0, color=color("oracle"), linestyle=":", linewidth=2, label="test no one")
    ax.set_ylim(-0.03, prev + 0.02); ax.set_xlabel("risk threshold for HPV testing"); ax.set_ylabel("net benefit")
    ax.legend(loc="upper right", fontsize=8.7); grid(ax); save(fig, "tab_decision")


# =============================================================== FUSION
SHORT = {"Image only (raw)": "image, raw", "Image only (CO-PRIOR)": "image only", "Questionnaire only (pre-test)": "questionnaire only",
         "Naive mean": "naive mean", "Naive product": "naive product", "Sum rule (CO-PRIOR)": "sum rule",
         "CO-PRIOR Bayes (proposed)": "proposed rule", "CO-PRIOR Bayes + safety floor": "proposed + safety floor",
         "Paired-data oracle (stacking)": "paired-data oracle"}
MCOL = {"Image only (CO-PRIOR)": color("image"), "Questionnaire only (pre-test)": color("tabular"), "Sum rule (CO-PRIOR)": "#7B6D8D",
        "Naive mean": color("baseline"), "CO-PRIOR Bayes (proposed)": color("fused"), "Paired-data oracle (stacking)": color("oracle")}


def fig_fus_coprior_real():
    F = J("fusion_eval.json")["coprior_real"]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.24))
    for ax, key, title, train in ((axs[0], "tabular", "Questionnaire expert (target: HPV+)", 0.246),
                                  (axs[1], "image", "Image expert (target: abnormal cell)", 0.413)):
        t = sorted(F[key], key=float); x = [float(v) for v in t]
        ax.errorbar(x, [F[key][k]["raw"] for k in t], yerr=[F[key][k]["raw_sd"] for k in t], marker="s", ms=7,
                    color=color("baseline"), linewidth=2.4, capsize=3, label="without re-anchoring")
        ax.errorbar(x, [F[key][k]["coprior"] for k in t], yerr=[F[key][k]["coprior_sd"] for k in t], marker="o", ms=7,
                    color=color("coprior"), linewidth=2.4, capsize=3, label="with re-anchoring")
        ax.axvline(train, color=color("random"), linestyle="--", linewidth=1.5, label=f"training prevalence ({train:.3f})")
        ax.set_xscale("log"); ax.set_xlabel("deployment prevalence $\\tau$"); ax.set_ylabel("ECE"); ax.set_title(title, fontsize=9.3)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.42 * (hi - lo))      # headroom so the legend clears the data
        ax.legend(loc="upper center"); grid(ax)
    save(fig, "fus_coprior_real")


def fig_fus_conformal_real():
    F = J("fusion_eval.json")["conformal_real"]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.24))
    for key, col, lab, mk in (("tabular", color("tabular"), "questionnaire expert", "s"), ("image", color("image"), "image expert", "o")):
        a = sorted(F[key], key=float); tgt = [1 - float(v) for v in a]
        axs[0].errorbar(tgt, [F[key][k]["coverage"]["mean"] for k in a], yerr=[F[key][k]["coverage"]["sd"] for k in a],
                        marker=mk, ms=8, color=col, linewidth=2.4, capsize=3, label=lab)
        axs[1].plot(tgt, [F[key][k]["abstention"]["mean"] for k in a], marker=mk, ms=8, color=col, linewidth=2.4, label=lab)
    axs[0].plot([0.78, 1], [0.78, 1], color=color("oracle"), linestyle=":", linewidth=2, label="target")
    axs[0].set_ylabel("empirical coverage"); axs[1].set_ylabel("abstention rate (set size $\\neq$ 1)")
    for ax in axs:
        ax.set_xlabel("target coverage $1-\\alpha$"); ax.legend(loc="upper left", fontsize=8.7); grid(ax)
    save(fig, "fus_conformal_real")


def fig_fus_gain_vs_or():
    F = J("fusion_eval.json"); E = F["by_expert"]["random-split student"]; ors = F["design"]["ORs"]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.40))
    lv = [("as-is", "0.999 (real)", "o"), ("0.95", "0.95", "s"), ("0.90", "0.90", "^"), ("0.80", "0.80", "D")]
    cols = ["#222222", color("image"), color("fused"), color("tabular")]
    for (lvl, lab, mk), c in zip(lv, cols):
        X = E[f"image_{lvl}"]
        g = [X[str(o)]["CO-PRIOR Bayes (proposed)"]["auroc"]["mean"] - X[str(o)]["Image only (CO-PRIOR)"]["auroc"]["mean"] for o in ors]
        go = [X[str(o)]["Paired-data oracle (stacking)"]["auroc"]["mean"] - X[str(o)]["Image only (CO-PRIOR)"]["auroc"]["mean"] for o in ors]
        gs = [X[str(o)]["Sum rule (CO-PRIOR)"]["auroc"]["mean"] - X[str(o)]["Image only (CO-PRIOR)"]["auroc"]["mean"] for o in ors]
        axs[0].plot(ors, g, marker=mk, ms=7, color=c, linewidth=2.6, label=f"image AUROC {lab}")
        axs[0].plot(ors, go, color=c, linestyle=":", linewidth=1.8)
        axs[1].plot(ors, gs, marker=mk, ms=7, color=c, linewidth=2.2, label=f"image AUROC {lab}")
    axs[0].plot([], [], color="#555", linestyle=":", linewidth=1.8, label="paired-data oracle")
    axs[0].set_title("Proposed rule"); axs[1].set_title("Naive sum rule")
    for ax in axs:
        ax.axhline(0, color=color("random"), linestyle="--", linewidth=1.5); ax.set_xscale("log")
        # the estimated link is named in the legend, so no text can land on the curves or the axis
        ax.axvline(17.9, color=color("baseline"), linestyle="-.", linewidth=1.2,
                   label="estimated OR 17.9")
        ax.set_xlabel("HPV $\\rightarrow$ cytology odds ratio"); ax.set_ylabel("$\\Delta$AUROC vs image only")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.22 * (hi - lo))   # room for the legend
        ax.legend(loc="upper left", fontsize=8); grid(ax)
    save(fig, "fus_gain_vs_or")


def fig_fus_roc_headline(lvl="0.90"):
    C = np.load(os.path.join(RES, "fusion_curves.npz")); F = J("fusion_eval.json")
    X = F["by_expert"]["random-split student"][f"image_{lvl}"][str(F["design"]["headline_OR"])]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.87))
    for m in ("Questionnaire only (pre-test)", "Image only (CO-PRIOR)", "Sum rule (CO-PRIOR)", "CO-PRIOR Bayes (proposed)", "Paired-data oracle (stacking)"):
        k = m.replace(" ", "_"); ls = ":" if "oracle" in m else "-"
        axs[0].plot(C[f"{lvl}_roc_{k}_fpr"], C[f"{lvl}_roc_{k}_tpr"], color=MCOL[m], linestyle=ls,
                    linewidth=2.8 if "proposed" in m else 2.0, label=f"{SHORT[m]} ({X[m]['auroc']['mean']:.3f})")
    axs[0].plot([0, 1], [0, 1], color=color("random"), linestyle="--", linewidth=1.4)
    axs[0].set_xlabel("false-positive rate"); axs[0].set_ylabel("true-positive rate"); axs[0].legend(loc="lower right", fontsize=8.4)
    axs[0].set_title("ROC (synthetic cohort)"); grid(axs[0])
    y = C[f"{lvl}_y"]; axs[1].plot([0, 1], [0, 1], color=color("oracle"), linestyle=":", linewidth=2, label="perfect")
    for m, mk in (("Image only (raw)", "s"), ("Sum rule (CO-PRIOR)", "^"), ("CO-PRIOR Bayes (proposed)", "o")):
        p = C[f"{lvl}_p_{m.replace(' ', '_')}"]; edges = np.quantile(p, np.linspace(0, 1, 11)); xs, ys = [], []
        for i in range(10):
            msk = (p >= edges[i]) & (p <= edges[i + 1])
            if msk.sum() > 30:
                xs.append(p[msk].mean()); ys.append(y[msk].mean())
        axs[1].plot(xs, ys, marker=mk, ms=7, linewidth=2.4, color=MCOL.get(m, color("baseline")),
                    label=f"{SHORT[m]} (ECE {X[m]['ece']['mean']:.3f})")
    axs[1].set_xscale("symlog", linthresh=0.05); axs[1].set_yscale("symlog", linthresh=0.05)
    axs[1].set_xlim(0, 1); axs[1].set_ylim(0, 1); axs[1].set_xlabel("predicted P(abnormal)"); axs[1].set_ylabel("observed frequency")
    axs[1].legend(loc="upper left", fontsize=8.4); axs[1].set_title("Calibration at 7.1% prevalence"); grid(axs[1])
    save(fig, f"fus_headline_{lvl.replace('.', '')}")


def fig_fus_missing(lvl="0.90"):
    F = J("fusion_eval.json")["by_expert"]["random-split student"]["missing"][lvl]
    mi = [0.0, 0.25, 0.5, 0.75, 1.0]; mt = [0.0, 0.25, 0.5, 1.0]
    M = np.full((len(mt), len(mi)), np.nan)
    for a, i_ in enumerate(mi):
        for b, t_ in enumerate(mt):
            k = f"img{i_}_tab{t_}"
            if k in F:
                M[b, a] = F[k]["auroc"]["mean"]
    fig, ax = plt.subplots(figsize=size(6.0, 3.91))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=np.nanmin(M), vmax=np.nanmax(M))
    for b in range(len(mt)):
        for a in range(len(mi)):
            if not np.isnan(M[b, a]):
                ax.text(a, b, f"{M[b, a]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="black" if M[b, a] > np.nanmean(M) else "white")
    ax.set_xticks(range(len(mi))); ax.set_xticklabels([f"{int(v * 100)}%" for v in mi])
    ax.set_yticks(range(len(mt))); ax.set_yticklabels([f"{int(v * 100)}%" for v in mt])
    ax.set_xlabel("patients without an image"); ax.set_ylabel("patients without a questionnaire")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("AUROC", fontweight="bold")
    save(fig, f"fus_missing_{lvl.replace('.', '')}")


def fig_fus_misspec():
    F = J("fusion_eval.json"); ors = F["design"]["ORs"]; E = F["by_expert"]["random-split student"]["misspecification"]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.24))
    for lvl, c, mk in (("as-is", "#222222", "o"), ("0.90", color("fused"), "^")):
        axs[0].plot(ors, [E[lvl][str(o)]["auroc"]["mean"] for o in ors], marker=mk, ms=7, color=c, linewidth=2.4,
                    label=f"image AUROC {'0.999' if lvl == 'as-is' else lvl}")
        axs[1].plot(ors, [E[lvl][str(o)]["ece"]["mean"] for o in ors], marker=mk, ms=7, color=c, linewidth=2.4,
                    label=f"image AUROC {'0.999' if lvl == 'as-is' else lvl}")
    for ax, yl in ((axs[0], "AUROC"), (axs[1], "ECE")):
        ax.axvline(F["design"]["headline_OR"], color=color("random"), linestyle="--", linewidth=1.5, label="true OR = 10")
        ax.set_xscale("log"); ax.set_xlabel("odds ratio assumed by the fusion"); ax.set_ylabel(yl); ax.legend(fontsize=8.7); grid(ax)
    save(fig, "fus_misspec")


def fig_fus_conformal_synth(lvl="0.90"):
    F = J("fusion_eval.json")["by_expert"]["random-split student"]["conformal"][lvl]
    alphas = ["0.01", "0.05", "0.1"]; ms = ["Image only (CO-PRIOR)", "CO-PRIOR Bayes (proposed)"]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.24)); x = np.arange(3); w = 0.36
    for k, m in enumerate(ms):
        cov = [F[f"{a}|{m}"]["coverage"]["mean"] for a in alphas]; ab = [F[f"{a}|{m}"]["abstention"]["mean"] for a in alphas]
        axs[0].bar(x + (k - 0.5) * w, cov, w, color=MCOL[m], edgecolor="black", linewidth=1.0, label=SHORT[m])
        axs[1].bar(x + (k - 0.5) * w, ab, w, color=MCOL[m], edgecolor="black", linewidth=1.0, label=SHORT[m])
        for i in range(3):
            axs[0].text(i + (k - 0.5) * w, cov[i] + 0.003, f"{cov[i]:.3f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
            axs[1].text(i + (k - 0.5) * w, ab[i] + 0.005, f"{ab[i]:.2f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    for i, a in enumerate(alphas):
        axs[0].plot([i - 0.45, i + 0.45], [1 - float(a)] * 2, color=color("oracle"), linestyle=":", linewidth=2)
    axs[0].set_ylim(0.85, 1.05); axs[0].set_ylabel("coverage"); axs[1].set_ylabel("abstention rate")
    axs[1].set_ylim(0, max(F[f"0.01|{m}"]["abstention"]["mean"] for m in ms) * 1.42)
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([f"$\\alpha$={a}" for a in alphas])
        ax.legend(loc="upper center", ncol=2, fontsize=8.7); grid(ax, "y")
    save(fig, f"fus_conformal_synth_{lvl.replace('.', '')}")


def fig_fus_decision(lvl="0.90"):
    C = np.load(os.path.join(RES, "fusion_curves.npz")); thr = C["nb_thr"]; y = C[f"{lvl}_y"]; prev = y.mean()
    fig, ax = plt.subplots(figsize=size(6.8, 5.15))
    for m in ("Questionnaire only (pre-test)", "Image only (CO-PRIOR)", "CO-PRIOR Bayes (proposed)"):
        ax.plot(thr, C[f"{lvl}_nb_{m.replace(' ', '_')}"], color=MCOL[m], linewidth=2.8 if "proposed" in m else 2.0, label=SHORT[m])
    ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), color=color("random"), linestyle="--", linewidth=1.6, label="refer everyone")
    ax.axhline(0, color=color("oracle"), linestyle=":", linewidth=2, label="refer no one")
    ax.set_ylim(-0.01, prev + 0.01); ax.set_xlabel("referral threshold"); ax.set_ylabel("net benefit")
    ax.legend(loc="upper right", fontsize=8.7); grid(ax); save(fig, f"fus_decision_{lvl.replace('.', '')}")


# =============================================================== GROUPED (robustness)
def fig_grouped():
    p = os.path.join(RES, "grouped_preds.npz")
    if not os.path.exists(p):
        print("grouped results not ready"); return
    g = np.load(p); E = J("image_eval.json"); yg = g["y_test"]
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    rows = {}
    for tag, P, y in (("random split\n(full + crops)", np.load(os.path.join(RES, "image_preds.npz"))["P_student"], E and np.load(os.path.join(RES, "image_preds.npz"))["y_test"]),
                      ("slide-grouped\n(crops only)", g["P_student"], yg)):
        rows[tag] = [accuracy_score(y, P.argmax(1)), f1_score(y, P.argmax(1), average="macro"),
                     roc_auc_score(y, P, multi_class="ovr"), roc_auc_score(np.isin(y, [0, 1]), P[:, [0, 1]].sum(1))]
    fig, axs = plt.subplots(1, 2, figsize=size(11, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28))
    ax = axs[0]; x = np.arange(4); w = 0.38
    for k, (tag, v) in enumerate(rows.items()):
        ax.bar(x + (k - 0.5) * w, v, w, color=[color("image"), color("fused")][k], edgecolor="black", linewidth=1.0, label=tag.replace("\n", " "))
        for i, val in enumerate(v):
            ax.text(i + (k - 0.5) * w, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["accuracy", "macro-F1", "macro\nAUROC", "abnormal\nAUROC"], fontsize=9)
    lo = min(min(v) for v in rows.values()); ax.set_ylim(max(0, lo - 0.08), 1.03)
    ax.set_ylabel("student B0, test set"); ax.legend(loc="lower left", fontsize=8.4); grid(ax, "y")
    cm = confusion_matrix(yg, g["P_student"].argmax(1), labels=range(5))
    _conf(axs[1], cm, f"Slide-grouped test set (acc {np.mean(g['P_student'].argmax(1) == yg):.3f})")
    save(fig, "img_grouped")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("image", "all"):
        fig_sipakmed_counts(); fig_sipakmed_samples(); fig_img_confusion(); fig_img_roc(); fig_img_reliability(); fig_img_report(); fig_training()
    if what in ("tabular", "all"):
        fig_tab_cohort(); fig_tab_models(); fig_tab_roc(); fig_tab_calibration(); fig_tab_importance(); fig_tab_shap(); fig_tab_decision()
    if what in ("fusion", "all"):
        fig_fus_coprior_real(); fig_fus_conformal_real(); fig_fus_gain_vs_or(); fig_fus_roc_headline("0.90"); fig_fus_roc_headline("as-is")
        fig_fus_missing("0.90"); fig_fus_misspec(); fig_fus_conformal_synth("0.90"); fig_fus_decision("0.90")
    if what in ("grouped", "all"):
        fig_grouped()
    print("figures ->", os.path.abspath(FIGDIR)); print(sorted(os.path.basename(f) for f in glob.glob(os.path.join(FIGDIR, "*.png"))))
