"""Estimate the HPV -> cytology link from a real screening cohort instead of assuming an odds ratio.

Castle et al., Obstet Gynecol 2009 (Kaiser Permanente Northern California, 812,598 co-tests, women >=30) reports
  P(hrHPV+)                         = 6.27%
  P(hrHPV+ and normal cytology)     = 3.99%
  P(ASC-US or worse)                = 5.18%
  P(ASC-US or worse and hrHPV-)     = 2.87%
The first two give P(HPV+, abnormal) = 2.28%; the last two give 5.18 - 2.87 = 2.31%, an independent check on the
same cell. That fixes the full 2x2, hence q1 = P(A|H+), q0 = P(A|H-) and the odds ratio, with no assumption.

Run: python paper_jbhi/scripts/epi_link.py   -> ../results/epi_link.json
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 812598
P_H, P_H_ANORM, P_A, P_A_HNEG = 0.0627, 0.0399, 0.0518, 0.0287


def two_by_two(p_h, p_h_anorm, p_a):
    a = p_h - p_h_anorm          # P(H+, A+)
    b = p_h_anorm                # P(H+, A-)
    c = p_a - a                  # P(H-, A+)
    d = 1 - a - b - c            # P(H-, A-)
    return a, b, c, d


def summarise(a, b, c, d):
    q1 = a / (a + b); q0 = c / (c + d)
    return dict(q1=q1, q0=q0, odds_ratio=(q1 / (1 - q1)) / (q0 / (1 - q0)),
                risk_ratio=q1 / q0, p_H=a + b, p_A=a + c)


def main():
    a, b, c, d = two_by_two(P_H, P_H_ANORM, P_A)
    est = summarise(a, b, c, d)
    consistency = abs((P_A - P_A_HNEG) - a)          # the paper's other pair implies the same cell
    print(f"2x2 (fractions): HPV+/abn {a:.4f}  HPV+/norm {b:.4f}  HPV-/abn {c:.4f}  HPV-/norm {d:.4f}")
    print(f"cross-check on P(H+,A+): {a:.4f} vs {P_A - P_A_HNEG:.4f}  (difference {consistency:.4f})")
    print(f"q1 = P(A|H+) = {est['q1']:.4f}   q0 = P(A|H-) = {est['q0']:.4f}")
    print(f"odds ratio = {est['odds_ratio']:.1f}   risk ratio = {est['risk_ratio']:.1f}")

    # uncertainty: resample the four cells from the reported counts
    rng = np.random.default_rng(0); counts = np.array([a, b, c, d]) * N
    ors, q1s, q0s = [], [], []
    for _ in range(5000):
        s = rng.multinomial(N, counts / counts.sum()) / N
        e = summarise(*s); ors.append(e["odds_ratio"]); q1s.append(e["q1"]); q0s.append(e["q0"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    print(f"sampling 95% CI: OR {ci(ors)}  q1 {ci(q1s)}  q0 {ci(q0s)}")

    out = dict(source="Castle et al., Obstet Gynecol 2009 (KPNC cotesting, n=812,598, women >=30)",
               reported=dict(p_hpv_pos=P_H, p_hpv_pos_cyt_normal=P_H_ANORM, p_cyt_abnormal=P_A,
                             p_cyt_abnormal_hpv_neg=P_A_HNEG, n=N),
               cells=dict(hpv_pos_abn=a, hpv_pos_norm=b, hpv_neg_abn=c, hpv_neg_norm=d),
               consistency_check=consistency, estimate=est,
               ci95=dict(odds_ratio=ci(ors), q1=ci(q1s), q0=ci(q0s)),
               athena_margins=dict(p_H=0.126, p_A=0.071,
                                   note="ATHENA (Wright et al. 2012, women >=21) used as an alternative, "
                                        "higher-prevalence deployment setting"))
    json.dump(out, open(os.path.join(HERE, "..", "results", "epi_link.json"), "w"), indent=2)
    print("saved results/epi_link.json")


if __name__ == "__main__":
    main()
