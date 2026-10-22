"""Full fused-model experiment suite (honest; no paired cohort exists, so joint claims are simulation- or
per-expert-based and every assumption is stated).

Uses the REAL marginals of BOTH frozen experts:
  * image  : binary-abnormal posterior from the EfficientNet student — in-distribution SIPaKMeD (AUROC ~0.998,
             leaky random split) AND external Herlev (the honest out-of-distribution number).
  * tabular: the frozen NHANES questionnaire expert (monotonic LightGBM + isotonic), calibrated OOF (~0.713).

Experiments (from RESEARCH_PLAN Phase 2-3 / FUSION_METHOD):
  E1  per-expert marginals + calibration (AUROC, ECE, reliability)
  E2  dependence-sensitivity envelope + the pre-registered FALSIFIER (fused vs max(single), AUROC & ECE),
      using a Gaussian-COPULA coupling of the two REAL per-class score distributions across dependence rho
  E3  CO-PRIOR operator: prior-stripping restores calibration across deployment prevalence tau; pi-misspec robustness
  E4  combiner ablation: sum vs naive-Bayes(product) vs cascade under dependence
  E5  any-subset degradation: {img},{tab},{both}; missing==single; safety asymmetry
  E6  split-conformal abstention: marginal coverage + risk-coverage (AURC), in-dist vs external
  E8  falsifier verdict + honesty statement

Outputs -> legacy_src/results/figures/fusion_*  and  results/metrics/fusion_experiments.json  and  ../FUSION_RESULTS.md
Run (from legacy_src/src):  python fusion_experiments.py
"""
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import config as C
import fusion as F

FIG, MET = C.FIGDIR, C.METRICDIR
RNG = 42
DEPLOY_PREV = 0.10
OP_THR = 0.30
ABN = [C.CLASSES.index("Dyskeratotic"), C.CLASSES.index("Koilocytotic")]


# --------------------------------------------------------------------------- #
#  expert data (REAL)
# --------------------------------------------------------------------------- #
def image_expert(which="herlev"):
    if which == "indist":
        d = json.load(open(os.path.join(MET, "predictions_student.json")))
        y = np.asarray(d["y_true"]); probs = np.asarray(d["probs"])
        return probs[:, ABN].sum(1), np.isin(y, ABN).astype(int)
    d = json.load(open(os.path.join(MET, "predictions_cross_herlev.json")))
    return np.asarray(d["p_abnormal"], float), np.asarray(d["y_true"], int)


def tabular_expert():
    """Frozen NHANES expert: isotonic-calibrated monotonic-LightGBM OOF (cached)."""
    cache = os.path.join(MET, "nhanes_tabular_oof.json")
    if os.path.exists(cache):
        d = json.load(open(cache)); return np.asarray(d["p"]), np.asarray(d["y"]), d["prior"]
    import nhanes_data as ND
    from nhanes_diagnostics import lgbm_mono, base_cols, oof
    X, y, feat, meta = ND.load(target="high_risk", verbose=False)
    cols = base_cols(feat)
    p = oof(lgbm_mono, X, cols, y, repeats=2, calibrate=True)
    json.dump(dict(p=p.tolist(), y=y.astype(int).tolist(), prior=float(y.mean())), open(cache, "w"))
    return p, y.astype(int), float(y.mean())


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def ece(y, p, bins=12):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


# --------------------------------------------------------------------------- #
#  vectorised fusion (matches fusion.predict)
# --------------------------------------------------------------------------- #
def _lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def _sg(z):
    return 1 / (1 + np.exp(-z))


def anchor(p, src_prior, tau):
    return _sg(_lg(p) - _lg(src_prior) + _lg(tau))


def fuse(pi, pt, prior_i, prior_t, tau, mode="cascade", op_thr=OP_THR):
    ai, at = anchor(pi, prior_i, tau), anchor(pt, prior_t, tau)
    s = 0.5 * (ai + at)
    if mode == "sum":
        return s
    if mode == "nb":
        return _sg(_lg(tau) + (_lg(pi) - _lg(prior_i)) + (_lg(pt) - _lg(prior_t)))
    p = s.copy(); hot = ai >= op_thr; p[hot] = np.maximum(s[hot], ai[hot])   # cascade safety floor
    return p


# --------------------------------------------------------------------------- #
#  Gaussian-copula joint simulator with REAL per-class marginals
# --------------------------------------------------------------------------- #
def _by_class(p, y):
    return {0: np.sort(p[y == 0]), 1: np.sort(p[y == 1])}


def simulate(img_sc, tab_sc, prev, rho, n=40000, seed=RNG):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prev).astype(int)
    cov = [[1.0, rho], [rho, 1.0]]
    z = rng.multivariate_normal([0, 0], cov, size=n)
    u = norm.cdf(z)
    pi = np.empty(n); pt = np.empty(n)
    for c in (0, 1):
        m = y == c
        pi[m] = np.quantile(img_sc[c], u[m, 0])
        pt[m] = np.quantile(tab_sc[c], u[m, 1])
    return y, pi, pt


# =========================================================================== #
#  E1 — per-expert marginals + calibration
# =========================================================================== #
def E1():
    out = {}
    pin, yin = image_expert("indist"); out["image_indist"] = dict(auroc=auroc(yin, pin), ece=ece(yin, pin),
                                                                  prior=float(yin.mean()), n=len(yin))
    ph, yh = image_expert("herlev"); out["image_herlev"] = dict(auroc=auroc(yh, ph), ece=ece(yh, ph),
                                                                prior=float(yh.mean()), n=len(yh))
    pt, yt, prt = tabular_expert(); out["tabular_nhanes"] = dict(auroc=auroc(yt, pt), ece=ece(yt, pt),
                                                                 prior=prt, n=len(yt))
    print("=== E1 per-expert marginals ===")
    for k, v in out.items():
        print(f"  {k:16s} AUROC={v['auroc']:.3f} ECE={v['ece']:.3f} prior={v['prior']:.3f} n={v['n']}")
    return out


# =========================================================================== #
#  E2 — dependence envelope + falsifier (copula, real marginals)
# =========================================================================== #
def E2(rhos=(-0.3, 0.0, 0.3, 0.6, 0.9, 1.0)):
    pt, yt, prt = tabular_expert()
    tab_sc = _by_class(pt, yt)
    scen = {}
    for name, which in [("image_external_Herlev (realistic)", "herlev"),
                        ("image_indist_SIPaKMeD (optimistic)", "indist")]:
        pi, yi = image_expert(which); pri = float(yi.mean()); img_sc = _by_class(pi, yi)
        env = {}
        for rho in rhos:
            y, si, st = simulate(img_sc, tab_sc, DEPLOY_PREV, rho)
            pf = fuse(si, st, pri, prt, DEPLOY_PREV, "cascade")
            ai = anchor(si, pri, DEPLOY_PREV); at = anchor(st, prt, DEPLOY_PREV)
            best = max(auroc(y, ai), auroc(y, at))
            env[f"{rho:+.1f}"] = dict(fused_auroc=auroc(y, pf), best_single=best,
                                      fused_ece=ece(y, pf), gain=auroc(y, pf) - best)
        scen[name] = dict(image_prior=pri, envelope=env)
        print(f"\n=== E2 [{name}] ===")
        for r, v in env.items():
            print(f"  rho={r}: fused={v['fused_auroc']:.3f} best_single={v['best_single']:.3f} "
                  f"gain={v['gain']:+.3f} fused_ECE={v['fused_ece']:.3f}")
    return scen


# =========================================================================== #
#  E3 — CO-PRIOR: prior-stripping restores calibration across tau; pi-misspec
# =========================================================================== #
def _resample_to_prev(p, y, tau, n=8000, seed=RNG):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    npos = int(round(n * tau)); nneg = n - npos
    idx = np.concatenate([rng.choice(pos, npos, replace=True), rng.choice(neg, nneg, replace=True)])
    return p[idx], y[idx]


def E3():
    pt, yt, prt = tabular_expert()
    taus = [0.02, 0.06, 0.10, 0.20]
    res = {"prior_src": prt, "no_shift": {}, "with_shift": {}, "misspec": {}}
    print("\n=== E3 CO-PRIOR (per-expert; no pairing) ===")
    for tau in taus:
        pr, yr = _resample_to_prev(pt, yt, tau)
        res["no_shift"][f"{tau}"] = ece(yr, pr)                                   # deployed as-is (wrong prior)
        res["with_shift"][f"{tau}"] = ece(yr, anchor(pr, prt, tau))              # prior-stripped -> re-anchored
    # pi mis-specification at tau=0.10: shift with a wrong source prior
    tau = 0.10; pr, yr = _resample_to_prev(pt, yt, tau)
    for f in [0.5, 0.75, 1.0, 1.5, 2.0]:
        res["misspec"][f"{f:.2f}x"] = ece(yr, anchor(pr, np.clip(prt * f, 1e-3, 0.99), tau))
    for tau in taus:
        print(f"  tau={tau}: ECE no-shift={res['no_shift'][str(tau)]:.3f}  "
              f"with-shift={res['with_shift'][str(tau)]:.3f}")
    print("  pi-misspec ECE:", {k: round(v, 3) for k, v in res["misspec"].items()})
    return res


# =========================================================================== #
#  E4 — combiner ablation (sum / nb / cascade) under dependence
# =========================================================================== #
def E4(rhos=(0.0, 0.3, 0.6, 0.9)):
    pi, yi = image_expert("herlev"); pri = float(yi.mean()); img_sc = _by_class(pi, yi)
    pt, yt, prt = tabular_expert(); tab_sc = _by_class(pt, yt)
    res = {}
    print("\n=== E4 combiner ablation (Herlev image) ===")
    for rho in rhos:
        y, si, st = simulate(img_sc, tab_sc, DEPLOY_PREV, rho)
        row = {}
        for mode in ["sum", "nb", "cascade"]:
            pf = fuse(si, st, pri, prt, DEPLOY_PREV, mode)
            row[mode] = dict(auroc=auroc(y, pf), ece=ece(y, pf))
        res[f"{rho:+.1f}"] = row
        print(f"  rho={rho:+.1f}: " + " | ".join(
            f"{m} AUROC={row[m]['auroc']:.3f}/ECE={row[m]['ece']:.3f}" for m in row))
    return res


# =========================================================================== #
#  E5 — any-subset degradation + structural invariants
# =========================================================================== #
def E5():
    pi, yi = image_expert("herlev"); pri = float(yi.mean()); img_sc = _by_class(pi, yi)
    pt, yt, prt = tabular_expert(); tab_sc = _by_class(pt, yt)
    y, si, st = simulate(img_sc, tab_sc, DEPLOY_PREV, 0.6)
    ai = anchor(si, pri, DEPLOY_PREV); at = anchor(st, prt, DEPLOY_PREV)
    pf = fuse(si, st, pri, prt, DEPLOY_PREV, "cascade")
    res = dict(image_only=auroc(y, ai), tabular_only=auroc(y, at), both=auroc(y, pf))
    print("\n=== E5 any-subset (rho=0.6 realistic) ===")
    print(f"  image-only={res['image_only']:.3f}  tabular-only={res['tabular_only']:.3f}  both={res['both']:.3f}")
    # structural invariants via fusion.predict
    priors = dict(img=pri, tab=prt); kw = dict(op_thr=OP_THR, boundary_margin=0.10)
    inv = {}
    a = F.predict(dict(p_img=0.95), priors, DEPLOY_PREV, **kw)
    b = F.predict(dict(p_img=0.95, p_tab=None), priors, DEPLOY_PREV, **kw)
    inv["missing_eq_single"] = abs(a["p_fused"] - b["p_fused"]) < 1e-9
    c = F.predict(dict(p_img=0.95, p_tab=0.02), priors, DEPLOY_PREV, **kw)
    inv["safety_asymmetry"] = c["p_fused"] >= a["p_fused"] - 1e-9
    inv["lone_tabular_defers"] = F.predict(dict(p_tab=0.8), priors, DEPLOY_PREV, **kw)["decision"] == F.DEFER
    inv["conflict_flags"] = c["decision"] == F.FLAG
    print("  invariants:", inv)
    res["invariants"] = inv
    return res


# =========================================================================== #
#  E6 — conformal abstention: marginal coverage + risk-coverage (AURC)
# =========================================================================== #
def _split_conformal_coverage(p, y, alpha, seed=RNG):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); h = len(y) // 2
    cal, test = idx[:h], idx[h:]
    thr = F.conformal_thresholds(p[cal], y[cal], alpha=alpha)
    covered = 0
    for i in test:
        s = F.conformal_set(p[i], thr)
        covered += (y[i] in s)
    return covered / len(test)


def _risk_coverage(p, y):
    """Confidence = |p-0.5|; sweep coverage by abstaining on the least-confident. Returns (cov, risk, AURC)."""
    conf = np.abs(p - 0.5); order = np.argsort(-conf)
    yhat = (p >= 0.5).astype(int); err = (yhat != y).astype(int)[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    aurc = float(np.sum((risk[:-1] + risk[1:]) / 2 * np.diff(cov)))          # trapezoid (np.trapz removed in numpy 2)
    return cov, risk, aurc


def E6():
    print("\n=== E6 conformal coverage + risk-coverage ===")
    res = {}
    for name, (p, y) in {"tabular": tabular_expert()[:2],
                         "image_indist": image_expert("indist"),
                         "image_herlev": image_expert("herlev")}.items():
        cov = {f"alpha={a}": _split_conformal_coverage(p, y, a) for a in (0.05, 0.1, 0.2)}
        _, _, aurc = _risk_coverage(p, y)
        res[name] = dict(coverage=cov, aurc=aurc)
        print(f"  {name:13s} coverage(target 1-a): " +
              " ".join(f"{k}->{v:.3f}" for k, v in cov.items()) + f"  AURC={aurc:.3f}")
    return res


# =========================================================================== #
#  figures + summary
# =========================================================================== #
def figures(e1, e2, e3, e4, e6):
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6)); axs = axs.ravel()

    # (0) dependence envelope — both scenarios
    ax = axs[0]
    for name, sc in e2.items():
        rhos = [float(k) for k in sc["envelope"]]; fused = [sc["envelope"][k]["fused_auroc"] for k in sc["envelope"]]
        best = [sc["envelope"][k]["best_single"] for k in sc["envelope"]]
        ax.plot(rhos, fused, "o-", label=f"fused — {name.split(' ')[0]}")
        ax.plot(rhos, best, "--", lw=0.9, label=f"max(single) — {name.split(' ')[0]}")
    ax.axvspan(0, 1, color="green", alpha=0.06); ax.set_xlabel("cross-modal dependence ρ (untestable)")
    ax.set_ylabel("AUROC"); ax.set_title("E2 · dependence envelope + falsifier"); ax.legend(fontsize=6.5)

    # (1) E2 gain vs rho (realistic scenario)
    ax = axs[1]; sc = e2["image_external_Herlev (realistic)"]["envelope"]
    rhos = [float(k) for k in sc]; gain = [sc[k]["gain"] for k in sc]
    ax.bar([f"{r:+.1f}" for r in rhos], gain, color=["#1a6" if g > 0 else "#c0392b" for g in gain])
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("ρ"); ax.set_ylabel("fused − max(single) AUROC")
    ax.set_title("E2 · falsifier: fusion gain (Herlev)")

    # (2) E3 CO-PRIOR calibration
    ax = axs[2]; taus = [float(k) for k in e3["no_shift"]]
    ax.plot(taus, [e3["no_shift"][str(t)] for t in taus], "o-", color="#c0392b", label="no prior-shift")
    ax.plot(taus, [e3["with_shift"][str(t)] for t in taus], "o-", color="#1a6", label="CO-PRIOR shift")
    ax.set_xlabel("deployment prevalence τ"); ax.set_ylabel("ECE"); ax.set_title("E3 · prior-stripping restores calibration")
    ax.legend(fontsize=8)

    # (3) E4 combiner ablation ECE vs rho
    ax = axs[3]; rhos = [float(k) for k in e4]
    for mode, col in [("sum", "#1a1a1a"), ("nb", "#c0392b"), ("cascade", "#1a6")]:
        ax.plot(rhos, [e4[k][mode]["ece"] for k in e4], "o-", color=col, label=mode)
    ax.set_xlabel("ρ"); ax.set_ylabel("fused ECE"); ax.set_title("E4 · combiner overconfidence vs dependence")
    ax.legend(fontsize=8)

    # (4) E6 conformal coverage
    ax = axs[4]
    for name in e6:
        alphas = [0.05, 0.1, 0.2]; cov = [e6[name]["coverage"][f"alpha={a}"] for a in alphas]
        ax.plot([1 - a for a in alphas], cov, "o-", label=name)
    ax.plot([0.8, 0.95], [0.8, 0.95], "k--", lw=0.8, label="target")
    ax.set_xlabel("target coverage 1−α"); ax.set_ylabel("empirical coverage")
    ax.set_title("E6 · conformal coverage (in-dist vs external)"); ax.legend(fontsize=7)

    # (5) E1 marginals
    ax = axs[5]
    names = ["image_indist", "image_herlev", "tabular_nhanes"]
    au = [e1[n]["auroc"] for n in names]; ec = [e1[n]["ece"] for n in names]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, au, w, label="AUROC", color="#1a1a1a"); ax.bar(x + w / 2, ec, w, label="ECE", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(["img\nin-dist", "img\nHerlev", "tab\nNHANES"], fontsize=8)
    ax.set_title("E1 · per-expert marginals"); ax.legend(fontsize=8)

    fig.suptitle("Fused-model experiments (honest; unpaired cohorts → simulation/per-expert)", y=1.0, fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fusion_experiments.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    C.set_seed()
    e1 = E1(); e2 = E2(); e3 = E3(); e4 = E4(); e5 = E5(); e6 = E6()
    out = dict(E1=e1, E2=e2, E3=e3, E4=e4, E5=e5, E6=e6)
    json.dump(out, open(os.path.join(MET, "fusion_experiments.json"), "w"), indent=2)
    figures(e1, e2, e3, e4, e6)
    # E8 falsifier verdict
    her = e2["image_external_Herlev (realistic)"]["envelope"]
    beats = {r: her[r]["gain"] > 0.002 for r in her}
    print("\n=== E8 FALSIFIER verdict (fused beats max(single) by >0.002 AUROC?) ===")
    for r, b in beats.items():
        print(f"  rho={r}: {'PASS' if b else 'fail'} (gain {her[r]['gain']:+.3f})")
    print("\nsaved -> results/metrics/fusion_experiments.json , results/figures/fusion_experiments.*")
    return out


if __name__ == "__main__":
    main()
