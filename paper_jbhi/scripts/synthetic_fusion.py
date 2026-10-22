"""Fusion evaluation on a semi-synthetic paired multimodal cohort + per-expert CO-PRIOR/conformal on real data.

No real patient has both a SIPaKMeD cell image and an NHANES questionnaire, so the fused system is tested on
SEMI-SYNTHETIC patients. Each synthetic patient i gets a latent state and two REAL held-out records:
    H_i ~ Bernoulli(pi_H)                       high-risk HPV status        (pi_H = 12.6%, ATHENA baseline)
    A_i | H_i ~ Bernoulli(q_{H_i})              abnormal cytology           (P(A)   = 7.1%, ATHENA baseline)
    questionnaire  <- a real NHANES participant with HPV status H_i     (out-of-fold calibrated p_tab)
    cell image     <- a real SIPaKMeD test cell with abnormality A_i    (held-out image-expert p_img)
q_1, q_0 are solved so the cohort reproduces both ATHENA margins for a given HPV->cytology odds ratio OR,
which is swept (OR=1: questionnaire uninformative for cytology). Given (H, A) the two records are drawn
independently, i.e. image _|_ questionnaire | A (the only cross-modal link runs through the latent state).

Target of the fused system = A (abnormal cytology, i.e. needs follow-up).

Run (Windows python):  python paper_jbhi/scripts/synthetic_fusion.py
Outputs: ../results/fusion_eval.json, ../results/fusion_curves.npz
"""
import os, json, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
PI_H, P_A = 0.126, 0.071                     # ATHENA (Wright et al., AJOG 2012): hrHPV+ and abnormal-cytology prevalence
ORS = [1, 2, 5, 10, 20, 50, 100]
HEADLINE_OR = 10
N, N_CAL, R = 20000, 20000, 20              # cohort size, calibration/training cohort size, replicates
OP_THR = 0.5                                 # safety floor: a confident abnormal cytology read is never lowered


# ------------------------------------------------------------------ helpers
def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def anchor(p, src_prior, tau):
    """CO-PRIOR: strip the expert's training prior, re-anchor to the deployment prevalence tau."""
    return sg(lg(p) - lg(src_prior) + lg(tau))


def ece(y, p, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def solve_q(pi_h, p_a, OR):
    """q1=P(A|H+), q0=P(A|H-) with odds ratio OR and pi_h*q1 + (1-pi_h)*q0 = p_a."""
    lo, hi = 1e-9, p_a
    for _ in range(200):
        q0 = (lo + hi) / 2; q1 = OR * q0 / (1 - q0 + OR * q0)
        if pi_h * q1 + (1 - pi_h) * q0 > p_a:
            hi = q0
        else:
            lo = q0
    return q1, q0


def sens_at_spec(y, p, spec=0.90):
    fpr, tpr, _ = roc_curve(y, p); return float(np.interp(1 - spec, fpr, tpr))


def score(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), ece=ece(y, p), sens_at_spec90=sens_at_spec(y, p, 0.90),
                mean_pred=float(p.mean()))


def net_benefit(y, p, thr):
    n = len(y); out = []
    for t in thr:
        pos = p >= t; tp = np.sum(pos & (y == 1)); fp = np.sum(pos & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


# ------------------------------------------------------------------ experts (real held-out predictions)
def load_experts():
    # prefer the revised questionnaire expert (one-hot nominals, missingness flags, nested model selection)
    v2 = os.path.join(RES, "tabular_v2_oof.npz")
    tab = np.load(v2 if os.path.exists(v2) else os.path.join(RES, "tabular_oof.npz"))
    y_h, p_t = tab["y"].astype(int), tab["p_cal"].astype(float)
    img = np.load(os.path.join(RES, "image_preds.npz"))
    experts = {}
    for tag, key_y, key_p, key_prior in (("random-split student", "y_test", "P_student", "y_holdout"),):
        y5 = img[key_y]; P = img[key_p]
        experts[tag] = dict(a=np.isin(y5, [0, 1]).astype(int), p=P[:, [0, 1]].sum(1),
                            prior=float(np.isin(img[key_prior], [0, 1]).mean()))
    # a genuinely weak, miscalibrated expert: the same network applied to another data set (Herlev) with no
    # retraining. Its CO-PRIOR source prior is still the prevalence it was TRAINED on, not Herlev's.
    ep = os.path.join(RES, "external_protocolB.npz")
    if os.path.exists(ep):
        e = np.load(ep)
        experts["external (Herlev)"] = dict(a=e["y"].astype(int), p=e["p"].astype(float), prior=0.401)
    gp = os.path.join(RES, "grouped_preds.npz")
    if os.path.exists(gp):
        g = np.load(gp); split = json.load(open(os.path.join(RES, "grouped_split.json")))
        prior_g = float(np.mean(["/im_Dyskeratotic/" in p or "/im_Koilocytotic/" in p for p in split["train"]]))
        experts["slide-grouped student"] = dict(a=np.isin(g["y_test"], [0, 1]).astype(int),
                                                p=g["P_student"][:, [0, 1]].sum(1), prior=prior_g)
    return dict(y_h=y_h, p_t=p_t, prior_t=float(y_h.mean())), experts


def cohort(tab, img, OR, n, rng):
    q1, q0 = solve_q(PI_H, P_A, OR)
    H = (rng.random(n) < PI_H).astype(int)
    A = (rng.random(n) < np.where(H == 1, q1, q0)).astype(int)
    pos_t, neg_t = np.where(tab["y_h"] == 1)[0], np.where(tab["y_h"] == 0)[0]
    pos_i, neg_i = np.where(img["a"] == 1)[0], np.where(img["a"] == 0)[0]
    it = np.where(H == 1, rng.choice(pos_t, n), rng.choice(neg_t, n))
    ii = np.where(A == 1, rng.choice(pos_i, n), rng.choice(neg_i, n))
    return H, A, tab["p_t"][it], img["p"][ii]


class NoisyImage:
    """Controlled weakening of the image expert (emulates poorer slides / acquisition shift).
    Gaussian noise (sd sigma) is added to the image log-odds, and the noisy expert is then re-calibrated with
    Platt scaling on IMAGE-ONLY labelled data drawn at the expert's native prevalence, so it remains a
    calibrated expert at its own training prior (as CO-PRIOR assumes). sigma=0 -> the real expert, unchanged."""

    def __init__(self, img, sigma, rng, n=40000):
        self.sigma = sigma
        if sigma == 0:
            return
        a = (rng.random(n) < img["prior"]).astype(int)
        pos, neg = np.where(img["a"] == 1)[0], np.where(img["a"] == 0)[0]
        z = lg(img["p"][np.where(a == 1, rng.choice(pos, n), rng.choice(neg, n))]) + rng.normal(0, sigma, n)
        self.platt = LogisticRegression(max_iter=1000).fit(z[:, None], a)

    def __call__(self, p, rng):
        if self.sigma == 0:
            return p
        return self.platt.predict_proba((lg(p) + rng.normal(0, self.sigma, len(p)))[:, None])[:, 1]


def sigma_for_auroc(img, target, rng, n=40000):
    """Noise level giving (approximately) a target image-expert AUROC."""
    pos, neg = np.where(img["a"] == 1)[0], np.where(img["a"] == 0)[0]
    zp, zn = lg(img["p"][rng.choice(pos, n)]), lg(img["p"][rng.choice(neg, n)])
    e1, e0 = rng.normal(0, 1, n), rng.normal(0, 1, n); lo, hi = 0.0, 50.0
    for _ in range(40):
        s = (lo + hi) / 2
        au = roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[zp + s * e1, zn + s * e0])
        lo, hi = (s, hi) if au > target else (lo, s)
    return (lo + hi) / 2


def fuse_all(p_t, p_i, tab, prior_i, OR_assumed, oracle=None):
    q1, q0 = solve_q(PI_H, P_A, OR_assumed)
    i_cp = anchor(p_i, prior_i, P_A)                               # image, re-anchored to deployment
    t_h = anchor(p_t, tab["prior_t"], PI_H)                          # questionnaire -> P(H+) at deployment
    pre = q1 * t_h + q0 * (1 - t_h)                                  # endpoint bridge: pre-test P(A | questionnaire)
    bayes = sg(lg(p_i) - lg(prior_i) + lg(pre))                      # proposed CO-PRIOR fusion
    floor = bayes.copy(); hot = i_cp >= OP_THR; floor[hot] = np.maximum(bayes[hot], i_cp[hot])
    out = {"Image only (raw)": p_i, "Image only (CO-PRIOR)": i_cp, "Questionnaire only (pre-test)": pre,
           "Naive mean": 0.5 * (p_i + p_t), "Naive product": sg(lg(p_i) + lg(p_t)),
           "Sum rule (CO-PRIOR)": 0.5 * (i_cp + pre),
           "CO-PRIOR Bayes (proposed)": bayes, "CO-PRIOR Bayes + safety floor": floor}
    if oracle is not None:
        out["Paired-data oracle (stacking)"] = oracle.predict_proba(np.column_stack([lg(p_i), lg(p_t)]))[:, 1]
    return out


def conformal(p_cal, y_cal, p_te, y_te, alpha):
    """Mondrian (class-conditional) split conformal with finite-sample correction; abstain unless singleton."""
    thr = {}
    for c in (0, 1):
        s = np.sort(1 - (p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c])); n = len(s)
        k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1); thr[c] = s[k]
    in1 = (1 - p_te) <= thr[1]; in0 = p_te <= thr[0]
    covered = np.where(y_te == 1, in1, in0); single = in1 ^ in0
    pred = np.where(in1 & ~in0, 1, 0)
    acc_acc = float((pred[single] == y_te[single]).mean()) if single.any() else float("nan")
    cls_cov = {int(c): float(covered[y_te == c].mean()) for c in (0, 1)}
    return dict(coverage=float(covered.mean()), class_coverage=cls_cov, abstention=float(1 - single.mean()),
                accuracy_on_accepted=acc_acc)


def mean_sd(list_of_dicts):
    keys = list_of_dicts[0].keys()
    return {k: dict(mean=float(np.mean([d[k] for d in list_of_dicts])), sd=float(np.std([d[k] for d in list_of_dicts])))
            for k in keys}


# ------------------------------------------------------------------ experiments
def main():
    t0 = time.time(); tab, experts = load_experts()
    print("experts:", {k: dict(n=len(v["a"]), prior=round(v["prior"], 3), auroc=round(roc_auc_score(v["a"], v["p"]), 4))
                       for k, v in experts.items()}, "| tab prior", round(tab["prior_t"], 3),
          "AUROC", round(roc_auc_score(tab["y_h"], tab["p_t"]), 4))
    out = dict(design=dict(pi_H=PI_H, p_A=P_A, N=N, replicates=R, ORs=ORS, headline_OR=HEADLINE_OR,
                           q={str(o): solve_q(PI_H, P_A, o) for o in ORS}, op_thr=OP_THR,
                           source="ATHENA baseline (Wright et al., AJOG 2012): 12.6% hrHPV+, 7.1% abnormal cytology"),
               experts={k: dict(n=int(len(v["a"])), prior=v["prior"], auroc=float(roc_auc_score(v["a"], v["p"])))
                        for k, v in experts.items()},
               tabular=dict(n=int(len(tab["y_h"])), prior=tab["prior_t"], auroc=float(roc_auc_score(tab["y_h"], tab["p_t"]))))
    curves = {}

    LEVELS = [("as-is", None), ("0.95", 0.95), ("0.90", 0.90), ("0.80", 0.80)]   # image-expert AUROC targets
    for etag, img in experts.items():
        E = out.setdefault("by_expert", {}).setdefault(etag, {})
        noisy = {}
        for lvl, target in LEVELS:
            sig = 0.0 if target is None else sigma_for_auroc(img, target, np.random.default_rng(3))
            noisy[lvl] = NoisyImage(img, sig, np.random.default_rng(4))
        E["noise_sigma"] = {lvl: noisy[lvl].sigma for lvl, _ in LEVELS}

        # E1: OR sweep x image strength (+ paired-data oracle)
        for lvl, _ in LEVELS:
            key = f"image_{lvl}"; E[key] = {}; nz = noisy[lvl]
            for OR in ORS:
                reps = {}
                for r in range(R):
                    rng = np.random.default_rng(1000 * r + OR)
                    _, A_tr, pt_tr, pi_tr = cohort(tab, img, OR, N_CAL, rng); pi_tr = nz(pi_tr, rng)
                    orc = LogisticRegression(max_iter=1000).fit(np.column_stack([lg(pi_tr), lg(pt_tr)]), A_tr)
                    H, A, pt, pi = cohort(tab, img, OR, N, rng); pi = nz(pi, rng)
                    f = fuse_all(pt, pi, tab, img["prior"], OR, orc)
                    for m, p in f.items():
                        reps.setdefault(m, []).append(score(A, p))
                    if etag == "random-split student" and OR == HEADLINE_OR and r == 0:
                        thr = np.linspace(0.01, 0.5, 50); curves["nb_thr"] = thr; curves[f"{lvl}_y"] = A
                        for m, p in f.items():
                            fpr, tpr, _ = roc_curve(A, p); k = m.replace(" ", "_")
                            curves[f"{lvl}_roc_{k}_fpr"], curves[f"{lvl}_roc_{k}_tpr"] = fpr, tpr
                            curves[f"{lvl}_p_{k}"] = p.astype(np.float32)
                            curves[f"{lvl}_nb_{k}"] = net_benefit(A, p, thr)
                E[key][str(OR)] = {m: mean_sd(v) for m, v in reps.items()}
            h = E[key][str(HEADLINE_OR)]
            print(f"[{etag}] image {lvl} (sigma={nz.sigma:.2f}) OR={HEADLINE_OR}: " +
                  ", ".join(f"{m.split(' (')[0][:16]}={h[m]['auroc']['mean']:.4f}"
                            for m in ("Image only (CO-PRIOR)", "Questionnaire only (pre-test)", "Sum rule (CO-PRIOR)",
                                      "CO-PRIOR Bayes (proposed)", "Paired-data oracle (stacking)")) +
                  f" | ECE img-raw={h['Image only (raw)']['ece']['mean']:.3f} fused={h['CO-PRIOR Bayes (proposed)']['ece']['mean']:.3f}"
                  f" ({time.time() - t0:.0f}s)")

        for lvl in ("as-is", "0.90"):
            nz = noisy[lvl]
            # E2: misspecified epidemiological link (true OR=headline, assumed OR varies)
            E.setdefault("misspecification", {})[lvl] = {}
            for OR_a in ORS:
                reps = []
                for r in range(R):
                    rng = np.random.default_rng(5000 + r)
                    H, A, pt, pi = cohort(tab, img, HEADLINE_OR, N, rng); pi = nz(pi, rng)
                    reps.append(score(A, fuse_all(pt, pi, tab, img["prior"], OR_a)["CO-PRIOR Bayes (proposed)"]))
                E["misspecification"][lvl][str(OR_a)] = mean_sd(reps)

            # E3: missing modalities (any-modality output), headline OR
            E.setdefault("missing", {})[lvl] = {}
            for miss_img in (0.0, 0.25, 0.5, 0.75, 1.0):
                for miss_tab in (0.0, 0.25, 0.5, 1.0):
                    if miss_img == 1.0 and miss_tab == 1.0:
                        continue
                    reps = []
                    for r in range(R):
                        rng = np.random.default_rng(7000 + r)
                        H, A, pt, pi = cohort(tab, img, HEADLINE_OR, N, rng); pi = nz(pi, rng)
                        f = fuse_all(pt, pi, tab, img["prior"], HEADLINE_OR)
                        mi = rng.random(N) < miss_img; mt = (rng.random(N) < miss_tab) & ~mi   # never both missing
                        p = f["CO-PRIOR Bayes (proposed)"].copy()
                        p[mi] = f["Questionnaire only (pre-test)"][mi]
                        p[mt] = f["Image only (CO-PRIOR)"][mt]
                        reps.append(score(A, p))
                    E["missing"][lvl][f"img{miss_img}_tab{miss_tab}"] = mean_sd(reps)

            # E4: conformal abstention on the fused output vs image only
            E.setdefault("conformal", {})[lvl] = {}
            for alpha in (0.01, 0.05, 0.10):
                for m in ("Image only (CO-PRIOR)", "CO-PRIOR Bayes (proposed)"):
                    reps = []
                    for r in range(R):
                        rng = np.random.default_rng(9000 + r)
                        _, A_c, pt_c, pi_c = cohort(tab, img, HEADLINE_OR, N_CAL, rng); pi_c = nz(pi_c, rng)
                        _, A_t, pt_t, pi_t = cohort(tab, img, HEADLINE_OR, N, rng); pi_t = nz(pi_t, rng)
                        pc = fuse_all(pt_c, pi_c, tab, img["prior"], HEADLINE_OR)[m]
                        pq = fuse_all(pt_t, pi_t, tab, img["prior"], HEADLINE_OR)[m]
                        c = conformal(pc, A_c, pq, A_t, alpha)
                        reps.append(dict(coverage=c["coverage"], abstention=c["abstention"],
                                         accuracy_on_accepted=c["accuracy_on_accepted"],
                                         cov_pos=c["class_coverage"][1], cov_neg=c["class_coverage"][0]))
                    E["conformal"][lvl][f"{alpha}|{m}"] = mean_sd(reps)
        print(f"[{etag}] all experiments done ({time.time() - t0:.0f}s)")

    # E5 (real data, no pairing): CO-PRIOR calibration per expert under prevalence shift
    rng = np.random.default_rng(11)
    def shift_ece(p, y, prior, taus, n=8000, reps=20):
        res = {}
        for tau in taus:
            raw, cp = [], []
            for _ in range(reps):
                pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; k = int(round(n * tau))
                idx = np.concatenate([rng.choice(pos, k), rng.choice(neg, n - k)])
                raw.append(ece(y[idx], p[idx])); cp.append(ece(y[idx], anchor(p[idx], prior, tau)))
            res[str(tau)] = dict(raw=float(np.mean(raw)), raw_sd=float(np.std(raw)),
                                 coprior=float(np.mean(cp)), coprior_sd=float(np.std(cp)))
        return res
    img0 = experts["random-split student"]
    out["coprior_real"] = dict(
        tabular=shift_ece(tab["p_t"], tab["y_h"], tab["prior_t"], [0.02, 0.05, 0.10, 0.126, 0.20, 0.246, 0.35, 0.50]),
        image=shift_ece(img0["p"], img0["a"], img0["prior"], [0.01, 0.03, 0.071, 0.15, 0.30, 0.42, 0.60]))

    # E6 (real data): per-expert conformal coverage, 50 random calibration/test splits
    def real_conformal(p, y, alphas=(0.01, 0.05, 0.10, 0.20), splits=50):
        res = {}
        for a in alphas:
            reps = []
            for _ in range(splits):
                idx = rng.permutation(len(y)); h = len(y) // 2
                c = conformal(p[idx[:h]], y[idx[:h]], p[idx[h:]], y[idx[h:]], a)
                reps.append(dict(coverage=c["coverage"], abstention=c["abstention"],
                                 cov_pos=c["class_coverage"][1], cov_neg=c["class_coverage"][0]))
            res[str(a)] = mean_sd(reps)
        return res
    out["conformal_real"] = dict(tabular=real_conformal(tab["p_t"], tab["y_h"]), image=real_conformal(img0["p"], img0["a"]))

    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(RES, "fusion_eval.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RES, "fusion_curves.npz"), **curves)
    print(f"saved fusion_eval.json + fusion_curves.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
