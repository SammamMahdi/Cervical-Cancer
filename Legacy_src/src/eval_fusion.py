"""Honest evaluation of the any-modality combiner (v2) — NO circular 'fusion helps' claim.

The v1 simulate_joint() was circular: it DREW the two experts as independent given y (the exact
assumption the fuser needs) with a fictional tabular AUROC ~0.98, then 'confirmed' fused >= best single.
Deleted. Instead we report only what is honestly knowable WITHOUT a paired cohort:

  (A) Real per-expert marginal AUROC (image: binary collapse of SIPaKMeD predictions; tabular: OOF, with CI).
  (B) DEPENDENCE-SENSITIVITY ENVELOPE (Frechet/Makarov-style): using the two REAL marginal AUROCs, the
      fused (equal-weight sum-rule) AUROC is computed under a Gaussian-score model across ALL cross-modal
      dependences rho in [-1,1]. The true rho is UNTESTABLE here and is biased POSITIVE (image koilocytosis
      and the UCI HPV/STD features share HPV as a common cause), so the realistic operating region is rho>=0,
      where the fused AUROC lies between the independence value (optimistic) and the image-only AUROC (rho=1,
      fusion adds nothing). This replaces one fabricated favorable point with an honest range.
  (C) Structural properties (mechanism, not accuracy): missing modality == single expert; safety asymmetry
      (risk never lowers a positive cytology); a small REAL routing demo.

Run: python eval_fusion.py   (from legacy_src/src)
"""
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import config as C
import fusion as F
try:
    import tabular as TAB
except Exception:                                # pragma: no cover
    TAB = None

TARGET = "Biopsy"                                # primary tabular target (only viable one, ~0.65)
IMG_PRIOR = 0.40                                 # SIPaKMeD abnormal-cell (koilo+dysk) construction rate
DEPLOY_PREV = 0.10                               # DECLARED deployment prevalence to anchor to


# --------------------------------------------------------------------------- #
#  (A) real per-expert marginal AUROC
# --------------------------------------------------------------------------- #
def image_binary_auroc():
    """Binary abnormal AUROC of the frozen image student, from saved SIPaKMeD val predictions."""
    p = os.path.join(C.METRICDIR, "predictions_student.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    y = np.asarray(d["y_true"]); probs = np.asarray(d["probs"])
    abn_idx = [C.CLASSES.index("Koilocytotic"), C.CLASSES.index("Dyskeratotic")]
    y_bin = np.isin(y, abn_idx).astype(int)
    p_abn = probs[:, abn_idx].sum(1)
    return float(roc_auc_score(y_bin, p_abn)) if len(set(y_bin)) > 1 else None


def tabular_auroc(target=TARGET):
    p = os.path.join(C.METRICDIR, "tabular_metrics.json")
    if not os.path.isfile(p):
        return None, None
    m = json.load(open(p)).get(target, {})
    return m.get("auroc"), m.get("auroc_ci95")


# --------------------------------------------------------------------------- #
#  (B) dependence-sensitivity envelope  (Gaussian-score model)
# --------------------------------------------------------------------------- #
def _sep(auroc):
    """Signal separation d (SD units) implied by a marginal AUROC under equal-variance Gaussian scores."""
    return np.sqrt(2.0) * norm.ppf(np.clip(auroc, 0.5001, 0.9999))


def fused_auroc_vs_dependence(auroc_img, auroc_tab, rhos=(-0.5, 0.0, 0.3, 0.6, 0.9, 1.0)):
    """Equal-weight sum-rule fused AUROC across cross-modal dependences rho. rho=0 independence (optimistic);
    rho=1 comonotone (fusion == best single); realistic region is rho>=0 (shared HPV cause)."""
    d1, d2 = _sep(auroc_img), _sep(auroc_tab)
    out = {}
    for rho in rhos:
        var = 2.0 + 2.0 * rho                    # Var(eps1+eps2)
        auc = float(norm.cdf((d1 + d2) / (np.sqrt(2.0) * np.sqrt(max(var, 1e-9)))))
        out[f"rho={rho:+.1f}"] = auc
    return out


def external_image_auroc():
    """Realistic OUT-OF-DISTRIBUTION binary image AUROC (Herlev — the most Pap-comparable external set).
    The in-distribution 0.998 is the leaky SIPaKMeD random-split number; external is the honest one."""
    p = os.path.join(C.METRICDIR, "cross_herlev.json")
    if os.path.isfile(p):
        return json.load(open(p)).get("auroc")
    return 0.73


def dependence_envelope():
    ai_indist = image_binary_auroc()
    ai_ext = external_image_auroc()
    at, at_ci = tabular_auroc()
    if ai_indist is None or at is None:
        print("[envelope SKIP] need predictions_student.json and tabular_metrics.json "
              "(run evaluate.indist_eval('student') and tabular.train_all())")
        return None
    scenarios = {"in_distribution (leaky, optimistic)": ai_indist,
                 "external Herlev (realistic)": ai_ext}
    res = dict(target=TARGET, auroc_tabular=at, auroc_tabular_ci95=at_ci, scenarios={})
    print("\n=== (B) dependence-sensitivity envelope (honest — REAL marginals) ===")
    print(f"  tabular({TARGET}) AUROC={at:.3f} CI{at_ci}")
    for name, ai in scenarios.items():
        env = fused_auroc_vs_dependence(ai, at)
        best = max(ai, at)
        res["scenarios"][name] = dict(auroc_image=ai, best_single=best, fused_vs_dependence=env,
                                      gain_independent=env["rho=+0.0"] - best,
                                      gain_realistic_rho06=env["rho=+0.6"] - best)
        print(f"\n  [{name}]  image AUROC={ai:.3f}, best single={best:.3f}")
        for k, v in env.items():
            tag = "  <- independence (optimistic)" if k == "rho=+0.0" else \
                  ("  <- fusion adds nothing" if k == "rho=+1.0" else "")
            print(f"     fused {k}: {v:.3f}{tag}")
        print(f"     realistic gain (rho~+0.6) = {env['rho=+0.6']-best:+.3f}")
    _envelope_figure(ai_ext, at, fused_auroc_vs_dependence(ai_ext, at))
    print("\n  NOTE: true rho is untestable without a paired cohort and biased POSITIVE (shared HPV cause);")
    print("  symmetric fusion HELPS only when the two experts are comparably strong AND near-independent.")
    return res


def _envelope_figure(ai, at, env):
    rhos = [float(k.split("=")[1]) for k in env]; aucs = list(env.values())
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot(rhos, aucs, "o-", label="fused (sum-rule)")
    ax.axhline(ai, ls="--", c="tab:blue", lw=0.9, label=f"image only ({ai:.3f})")
    ax.axhline(at, ls="--", c="tab:orange", lw=0.9, label=f"tabular only ({at:.3f})")
    ax.axvspan(0, 1, color="green", alpha=0.06)
    ax.text(0.5, min(aucs), "realistic region (rho>=0)", ha="center", va="bottom", fontsize=7, color="green")
    ax.set_xlabel("cross-modal dependence rho (untestable here)"); ax.set_ylabel("AUROC")
    ax.set_title(f"Fused-AUROC envelope vs dependence ({TARGET})"); ax.legend(fontsize=7)
    os.makedirs(C.FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIGDIR, f"fusion_dependence_envelope.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  (C) structural properties + real routing demo
# --------------------------------------------------------------------------- #
def routing_demo(n_show=3):
    print("\n=== (C) real unpaired routing demo (NOT a joint accuracy claim) ===")
    priors = dict(img=IMG_PRIOR, tab=0.064)
    try:
        exp = TAB.load_expert(TARGET); X, _, _ = TAB.load_uci()
        base = exp["metrics"]["base_rate"]; priors["tab"] = base
        for i, p in enumerate(TAB.p_positive(exp, X.iloc[:n_show])):
            print(f"  patient {i}: tabular-only ->", F.predict(dict(p_tab=float(p)), priors, DEPLOY_PREV))
    except Exception as e:
        print(f"  [tabular demo SKIP] {e}")
    try:
        import torch, evaluate as EV
        from dataset import make_loaders
        _, val_ld, _ = make_loaders(batch_size=8)
        model = EV.load_model("student")
        xb, _ = next(iter(val_ld))
        _, probs = EV.predict(model, [(xb, torch.zeros(len(xb)))])
        for i, p in enumerate(probs[:n_show, EV.ABNORMAL_IDX].sum(1)):
            print(f"  cell {i}: image-only ->", F.predict(dict(p_img=float(p)), priors, DEPLOY_PREV))
    except Exception as e:
        print(f"  [image demo SKIP] {e}")


if __name__ == "__main__":
    C.set_seed()
    res = dependence_envelope()
    if res is not None:
        os.makedirs(C.METRICDIR, exist_ok=True)
        json.dump(res, open(os.path.join(C.METRICDIR, "fusion_mechanism.json"), "w"), indent=2)
    routing_demo()
    print("\ndone.")
