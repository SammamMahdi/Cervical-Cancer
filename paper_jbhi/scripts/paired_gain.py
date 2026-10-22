"""Paired uncertainty for the fusion gain.

The fusion comparison is paired: within a replicate every rule scores the same 20,000 women from the same two
expert outputs. Reporting each method's mean +- SD across replicates therefore overstates the uncertainty of the
DIFFERENCE. This recomputes the per-replicate differences and reports their mean, SD, a 95% interval and a
sign test.  Run:  python paper_jbhi/scripts/paired_gain.py  -> ../results/paired_gain.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import synthetic_fusion as SF
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

OUT = os.path.join(HERE, "..", "results")
PROP, IMG, ORC, SUM = ("CO-PRIOR Bayes (proposed)", "Image only (CO-PRIOR)",
                       "Paired-data oracle (stacking)", "Sum rule (CO-PRIOR)")


def main():
    tab, experts = SF.load_experts()
    out = {}
    for etag, img in experts.items():
        noisy = {}
        for lvl, target in [("as-is", None), ("0.95", 0.95), ("0.90", 0.90), ("0.80", 0.80)]:
            sig = 0.0 if target is None else SF.sigma_for_auroc(img, target, np.random.default_rng(3))
            noisy[lvl] = SF.NoisyImage(img, sig, np.random.default_rng(4))
        for lvl in ("as-is", "0.95", "0.90", "0.80"):
            nz = noisy[lvl]
            for OR in (1, 10, 100):
                d_prop, d_sum, d_orc = [], [], []
                for r in range(SF.R):
                    rng = np.random.default_rng(1000 * r + OR)
                    _, A_tr, pt_tr, pi_tr = SF.cohort(tab, img, OR, SF.N_CAL, rng); pi_tr = nz(pi_tr, rng)
                    orc = LogisticRegression(max_iter=1000).fit(
                        np.column_stack([SF.lg(pi_tr), SF.lg(pt_tr)]), A_tr)
                    _, A, pt, pi = SF.cohort(tab, img, OR, SF.N, rng); pi = nz(pi, rng)
                    f = SF.fuse_all(pt, pi, tab, img["prior"], OR, orc)
                    a = {k: roc_auc_score(A, v) for k, v in f.items()}
                    d_prop.append(a[PROP] - a[IMG]); d_sum.append(a[SUM] - a[IMG]); d_orc.append(a[PROP] - a[ORC])
                rec = {}
                for name, d in (("proposed_minus_image", d_prop), ("sum_minus_image", d_sum),
                                ("proposed_minus_oracle", d_orc)):
                    d = np.array(d)
                    rec[name] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                     ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                     positive_replicates=int((d > 0).sum()), n=len(d))
                out.setdefault(etag, {}).setdefault(lvl, {})[str(OR)] = rec
                p = rec["proposed_minus_image"]
                print(f"[{etag[:14]:14s}] image {lvl:6s} OR={OR:3d}  proposed-image "
                      f"{p['mean']:+.4f} +- {p['sd']:.4f} (95% CI {p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}) "
                      f"positive in {p['positive_replicates']}/{p['n']} replicates")
    json.dump(out, open(os.path.join(OUT, "paired_gain.json"), "w"), indent=1)
    print("saved results/paired_gain.json")


if __name__ == "__main__":
    main()
