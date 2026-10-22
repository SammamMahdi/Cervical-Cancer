"""Does CO-PRIOR survive when its assumptions are false?

The main synthetic cohort satisfies the two assumptions used to derive the fusion rule, so matching a paired-data
oracle there is close to guaranteed. This script breaks each assumption in turn and measures the damage, comparing
CO-PRIOR against the image alone, naive fusion and an oracle stacker that is *refitted on the violated data* and
can therefore exploit whatever structure the violation introduces.

Violations
  V1 dependence   residual dependence between the two records given the latent state (Gaussian copula, rho)
  V2 direct path  questionnaire affects cytology directly, not only through HPV (gamma)
  V3 infection    HPV status shifts the image score even at fixed cytology state (delta)
  V4 subgroup     the link (q1,q0) differs by subgroup while the fusion uses pooled values (spread)
  V5 miscalib     one expert is miscalibrated (temperature on its log-odds) without the fusion knowing
  V6 domain+prev  image domain shift (bias on log-odds) together with prevalence shift

Also: uncertainty of the image expert itself (bootstrap over its test cells), how much paired data a stacker
needs to match a method that needs none, and conformal coverage when calibration and deployment differ.

Run: python paper_jbhi/scripts/fusion_robust.py   -> ../results/fusion_robust.json
"""
import os, sys, json, time
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synthetic_fusion as SF
from synthetic_fusion import lg, sg, anchor, ece

RES = os.path.join(HERE, "..", "results")
LINK = json.load(open(os.path.join(RES, "epi_link.json")))
PI_H, P_A = LINK["estimate"]["p_H"], LINK["estimate"]["p_A"]     # KPNC screening margins
Q1, Q0 = LINK["estimate"]["q1"], LINK["estimate"]["q0"]
N, R = 20000, 10
PROP, IMG, ORC = "proposed", "image only", "oracle (paired)"


def fuse(p_t, p_i, prior_t, prior_i, q1=Q1, q0=Q0, pi_h=PI_H, p_a=P_A):
    """The three comparators plus naive fusion, all from the same two expert outputs."""
    i_cp = anchor(p_i, prior_i, p_a)
    t_h = anchor(p_t, prior_t, pi_h)
    pre = q1 * t_h + q0 * (1 - t_h)
    return {IMG: i_cp, "questionnaire only": pre, "naive mean": 0.5 * (p_i + p_t),
            "sum rule": 0.5 * (i_cp + pre), PROP: sg(lg(p_i) - lg(prior_i) + lg(pre))}


def pick(scores, u):
    """Select a record by its rank: u in [0,1) indexes the sorted score list (copula coupling)."""
    idx = np.clip((u * len(scores)).astype(int), 0, len(scores) - 1)
    return np.sort(scores)[idx]


def make_cohort(tab, img, n, rng, rho=0.0, gamma=0.0, delta=0.0, q_spread=0.0,
                mis_t=1.0, mis_i=1.0, img_bias=0.0, prev=None):
    """One synthetic cohort, optionally violating the fusion's assumptions."""
    p_a = P_A if prev is None else prev
    q1, q0 = Q1, Q0
    if prev is not None:                                   # keep the odds ratio, move the prevalence
        q1, q0 = SF.solve_q(PI_H, p_a, LINK["estimate"]["odds_ratio"])
    H = (rng.random(n) < PI_H).astype(int)
    sub = rng.random(n) < 0.5                              # V4: two subgroups with different links
    q1v = np.where(sub, q1 * (1 + q_spread), q1 * (1 - q_spread))
    q0v = np.where(sub, q0 * (1 - q_spread), q0 * (1 + q_spread))
    pA = np.where(H == 1, q1v, q0v)

    t_pos, t_neg = np.sort(tab["p_t"][tab["y_h"] == 1]), np.sort(tab["p_t"][tab["y_h"] == 0])
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)   # V1: coupling given the latent state
    u = norm.cdf(z)
    p_t = np.where(H == 1, pick(t_pos, u[:, 0]), pick(t_neg, u[:, 0]))

    if gamma:                                              # V2: questionnaire acts on cytology directly
        zt = (lg(p_t) - lg(p_t).mean()) / lg(p_t).std()
        pA = sg(lg(np.clip(pA, 1e-6, 1 - 1e-6)) + gamma * zt)
    A = (rng.random(n) < pA).astype(int)

    i_pos, i_neg = np.sort(img["p"][img["a"] == 1]), np.sort(img["p"][img["a"] == 0])
    ui = u[:, 1]
    if delta:                                              # V3: infection shifts the image score at fixed A
        ui = np.clip(ui + delta * (H - PI_H), 1e-6, 1 - 1e-6)
    p_i = np.where(A == 1, pick(i_pos, ui), pick(i_neg, ui))

    if mis_t != 1.0:                                       # V5: unknown miscalibration
        p_t = sg(lg(p_t) / mis_t)
    if mis_i != 1.0:
        p_i = sg(lg(p_i) / mis_i)
    if img_bias:                                           # V6: domain shift on the image expert
        p_i = sg(lg(p_i) + img_bias)
    return H, A, p_t, p_i, p_a


def evaluate(tab, img, reps=R, **kw):
    """Mean AUROC/ECE per method, plus the paired gap to an oracle refitted on the violated data."""
    acc = {}
    for r in range(reps):
        rng = np.random.default_rng(4000 + r)
        H, A, p_t, p_i, p_a = make_cohort(tab, img, N, rng, **kw)
        Htr, Atr, ptr, pitr, _ = make_cohort(tab, img, N, rng, **kw)
        orc = LogisticRegression(max_iter=1000).fit(np.column_stack([lg(pitr), lg(ptr)]), Atr)
        f = fuse(p_t, p_i, tab["prior_t"], img["prior"], p_a=p_a)
        f[ORC] = orc.predict_proba(np.column_stack([lg(p_i), lg(p_t)]))[:, 1]
        for m, p in f.items():
            acc.setdefault(m, {"auroc": [], "ece": []})
            acc[m]["auroc"].append(roc_auc_score(A, p)); acc[m]["ece"].append(ece(A, p))
        acc.setdefault("_gap", {"auroc": [], "ece": []})
        acc["_gap"]["auroc"].append(roc_auc_score(A, f[PROP]) - roc_auc_score(A, f[ORC]))
        acc["_gap"]["ece"].append(ece(A, f[PROP]) - ece(A, f[ORC]))
    return {m: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in d.items()}
            for m, d in acc.items()}


def main():
    t0 = time.time(); tab, experts = SF.load_experts()
    img = experts["slide-grouped student"] if "slide-grouped student" in experts else experts["random-split student"]
    weak = SF.NoisyImage(img, SF.sigma_for_auroc(img, 0.90, np.random.default_rng(3)), np.random.default_rng(4))

    wimg = dict(img); wimg["p"] = weak(img["p"], np.random.default_rng(5))       # a realistically imperfect reader
    if "external (Herlev)" in experts:                                           # and a measured one
        wimg = experts["external (Herlev)"]
        note = "external (Herlev), no retraining"
    else:
        note = "simulated weakening to AUROC 0.90"
    out = dict(link=LINK["estimate"], margins=dict(pi_H=PI_H, p_A=P_A), N=N, replicates=R,
               image_expert=dict(source=note, auroc_in_distribution=float(roc_auc_score(img["a"], img["p"])),
                                 auroc_used=float(roc_auc_score(wimg["a"], wimg["p"])), prior=wimg["prior"]))
    print(f"image expert: in-distribution AUROC {out['image_expert']['auroc_in_distribution']:.4f}; "
          f"used for robustness = {note} (AUROC {out['image_expert']['auroc_used']:.4f}); "
          f"link OR={LINK['estimate']['odds_ratio']:.1f}")

    grids = dict(
        V1_dependence=[dict(rho=v) for v in (0.0, 0.2, 0.4, 0.6, 0.8)],
        V2_direct_path=[dict(gamma=v) for v in (0.0, 0.25, 0.5, 1.0)],
        V3_infection_on_image=[dict(delta=v) for v in (0.0, 0.1, 0.2, 0.4)],
        V4_subgroup_link=[dict(q_spread=v) for v in (0.0, 0.25, 0.5)],
        V5_miscalibration=[dict(mis_t=v) for v in (1.0, 1.5, 2.0)] + [dict(mis_i=v) for v in (1.5, 2.0)],
        V6_domain_prevalence=[dict(img_bias=b, prev=p) for b in (0.0, 1.0) for p in (P_A, 0.02)],
    )
    for name, settings in grids.items():
        out[name] = []
        for kw in settings:
            r = evaluate(tab, wimg, **kw)
            rec = dict(setting=kw, gap_auroc=r["_gap"]["auroc"], gap_ece=r["_gap"]["ece"],
                       methods={m: dict(auroc=r[m]["auroc"]["mean"], ece=r[m]["ece"]["mean"]) for m in r if m != "_gap"})
            out[name].append(rec)
            print(f"  {name:22s} {str(kw):34s} proposed {rec['methods'][PROP]['auroc']:.4f}/"
                  f"{rec['methods'][PROP]['ece']:.4f}  image {rec['methods'][IMG]['auroc']:.4f}  "
                  f"oracle {rec['methods'][ORC]['auroc']:.4f}  gap {rec['gap_auroc']['mean']:+.4f}")

    # how much paired data would a stacker need to match a method that needs none?
    out["paired_data_requirement"] = []
    for n_paired in (50, 100, 200, 500, 1000, 5000, 20000):
        d = []
        for r in range(R):
            rng = np.random.default_rng(6000 + r)
            _, Atr, ptr, pitr, _ = make_cohort(tab, wimg, n_paired, rng)
            H, A, p_t, p_i, _ = make_cohort(tab, wimg, N, rng)
            try:
                o = LogisticRegression(max_iter=1000).fit(np.column_stack([lg(pitr), lg(ptr)]), Atr)
                a_o = roc_auc_score(A, o.predict_proba(np.column_stack([lg(p_i), lg(p_t)]))[:, 1])
            except ValueError:
                continue
            d.append(a_o - roc_auc_score(A, fuse(p_t, p_i, tab["prior_t"], wimg["prior"])[PROP]))
        out["paired_data_requirement"].append(dict(n_paired=n_paired, oracle_minus_proposed=dict(
            mean=float(np.mean(d)), sd=float(np.std(d, ddof=1)))))
        print(f"  paired stacker n={n_paired:5d}: oracle-proposed AUROC {np.mean(d):+.4f}")

    # uncertainty of the image expert itself: resample its test cells between replicates
    boot = []
    for r in range(30):
        rng = np.random.default_rng(7000 + r)
        idx = rng.integers(0, len(img["a"]), len(img["a"]))
        bi = dict(a=img["a"][idx], p=weak(img["p"][idx], rng), prior=img["prior"])
        H, A, p_t, p_i, _ = make_cohort(tab, bi, N, rng)
        f = fuse(p_t, p_i, tab["prior_t"], bi["prior"])
        boot.append([roc_auc_score(A, f[PROP]), roc_auc_score(A, f[IMG])])
    boot = np.array(boot)
    out["image_expert_bootstrap"] = dict(
        proposed=dict(mean=float(boot[:, 0].mean()), sd=float(boot[:, 0].std(ddof=1)),
                      ci95=[float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))]),
        image_only=dict(mean=float(boot[:, 1].mean()), sd=float(boot[:, 1].std(ddof=1)),
                        ci95=[float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))]),
        paired_gain=dict(mean=float((boot[:, 0] - boot[:, 1]).mean()), sd=float((boot[:, 0] - boot[:, 1]).std(ddof=1))))
    print(f"  image-expert bootstrap: proposed {boot[:,0].mean():.4f}+-{boot[:,0].std(ddof=1):.4f}, "
          f"gain {np.mean(boot[:,0]-boot[:,1]):+.4f}+-{np.std(boot[:,0]-boot[:,1],ddof=1):.4f}")

    # conformal coverage when calibration and deployment distributions differ
    out["conformal_under_shift"] = []
    for name, kw in (("matched", {}), ("dependence rho=0.6", dict(rho=0.6)),
                     ("miscalibrated image", dict(mis_i=2.0)), ("domain+prevalence shift", dict(img_bias=1.0, prev=0.02))):
        cov, ab = [], []
        for r in range(R):
            rng = np.random.default_rng(8000 + r)
            _, Ac, ptc, pic, _ = make_cohort(tab, wimg, N, rng)                      # calibration: clean
            _, At, ptt, pit, _ = make_cohort(tab, wimg, N, rng, **kw)                # test: shifted
            pc = fuse(ptc, pic, tab["prior_t"], wimg["prior"])[PROP]
            pt_ = fuse(ptt, pit, tab["prior_t"], wimg["prior"])[PROP]
            c = SF.conformal(pc, Ac, pt_, At, 0.10)
            cov.append(c["coverage"]); ab.append(c["abstention"])
        out["conformal_under_shift"].append(dict(scenario=name, target=0.90,
                                                 coverage=dict(mean=float(np.mean(cov)), sd=float(np.std(cov, ddof=1))),
                                                 abstention=dict(mean=float(np.mean(ab)), sd=float(np.std(ab, ddof=1)))))
        print(f"  conformal {name:24s} coverage {np.mean(cov):.3f} (target 0.90) abstention {np.mean(ab):.3f}")

    json.dump(out, open(os.path.join(RES, "fusion_robust.json"), "w"), indent=1)
    print(f"saved results/fusion_robust.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
