"""Any-modality decision combiner for two FROZEN, independently-calibrated cervical experts.

v2 — rebuilt after an adversarial review (see FUSION_METHOD.md §5) found the v1 symmetric
Dempster-Shafer/PoE design statistically and clinically unsound for THIS problem:
  * the two experts sit at different points on the disease pathway (P_tab = UPSTREAM pre-test risk
    from a questionnaire; P_img = DOWNSTREAM cytologic finding) -> they must be combined
    ASYMMETRICALLY (pre-test odds x likelihood-ratio), not as a peer vote;
  * they are calibrated to DIFFERENT priors (SIPaKMeD ~40% abnormal cells vs UCI ~6% biopsy+),
    so a shared base rate cannot be assumed -> re-anchor each to a DECLARED deployment prevalence
    (Saerens, Latinne & Decaestecker, Neural Comp. 2002) before combining;
  * the honest tabular ceiling is AUROC ~0.65 (weak), so fusion is TRIAGE consolidation, not an
    accuracy driver; a low-risk questionnaire must NEVER talk down a positive cytology (safety asymmetry).

Grounded combiner: Kittler, Hatef, Duin & Matas, "On Combining Classifiers", IEEE TPAMI 1998 —
pool prior-corrected log-likelihood-ratios (product/naive-Bayes rule) or the more robust SUM rule.
Abstention via split-conformal risk control (Angelopoulos & Bates, 2023), calibrated per-expert on
each expert's OWN data (no paired cohort needed). Dempster-Shafer is retained ONLY as a conflict FLAG.

Pure numpy.
"""
import numpy as np

RELEASE, DEFER, FLAG = "RELEASE", "DEFER", "FLAG"
_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
#  Prior-shift re-anchoring (Saerens-Latinne-Decaestecker 2002, per-sample form)
# --------------------------------------------------------------------------- #
def prior_shift(p, src_prior, target_prev):
    """Re-anchor a calibrated posterior from its training prior to a DECLARED deployment prevalence.

    Assumes the class-conditional likelihood is unchanged (label shift): logit(p') = logit(p)
    - logit(src_prior) + logit(target_prev). This is what puts the two experts on ONE probability axis.
    """
    return float(_sigmoid(_logit(p) - _logit(src_prior) + _logit(target_prev)))


def log_lr(p, prior):
    """Log-likelihood-ratio an expert contributes = logit(posterior) - logit(prior) (base rate removed)."""
    return float(_logit(p) - _logit(prior))


# --------------------------------------------------------------------------- #
#  Combiners (Kittler et al. 1998)
# --------------------------------------------------------------------------- #
def naive_bayes_lr(p_img, p_tab, prior_img, prior_tab, target_prev):
    """Product / naive-Bayes rule: posterior logit = logit(prev) + sum of per-expert log-LRs.
    A missing expert simply omits its LR term. Powerful but OVERCONFIDENT if the experts' errors are
    correlated (they share HPV as a cause) -> prefer sum_rule as the operating default."""
    z = _logit(target_prev)
    if p_img is not None:
        z += log_lr(p_img, prior_img)
    if p_tab is not None:
        z += log_lr(p_tab, prior_tab)
    return float(_sigmoid(z))


def sum_rule(posteriors):
    """Kittler ROBUST SUM rule (linear opinion pool) over prior-corrected posteriors — the operating
    default: far less sensitive than the product rule to the high-variance tabular posterior."""
    posteriors = [p for p in posteriors if p is not None]
    return float(np.mean(posteriors)) if posteriors else float("nan")


def ds_conflict(p_img, p_tab, r_img=1.0, r_tab=1.0):
    """Dempster-Shafer CONFLICT mass K only (reporting/flagging, NOT the point estimate).
    K = b_img*d_tab + d_img*b_tab with belief discounted by reliability r."""
    if p_img is None or p_tab is None:
        return 0.0
    b1, d1 = r_img * p_img, r_img * (1 - p_img)
    b2, d2 = r_tab * p_tab, r_tab * (1 - p_tab)
    return float(b1 * d2 + d1 * b2)


# --------------------------------------------------------------------------- #
#  Split-conformal abstention (Angelopoulos & Bates 2023), per-expert calibrated
# --------------------------------------------------------------------------- #
def conformal_thresholds(cal_p_pos, cal_labels, alpha=0.1):
    """Mondrian split-conformal thresholds from one expert's OWN calibration probs+labels.
    Nonconformity(true class c) = 1 - p_c ; threshold_c = (1-alpha) quantile of scores for class c."""
    cal_p_pos = np.asarray(cal_p_pos, float); y = np.asarray(cal_labels, int)
    P = np.column_stack([1 - cal_p_pos, cal_p_pos])
    thr = {}
    for c in (0, 1):
        s = 1 - P[y == c, c]
        thr[c] = float(np.quantile(s, 1 - alpha)) if s.size else 1.0
    return thr


def conformal_set(p_pos, thr):
    """Conformal prediction SET at the calibrated alpha. Size!=1 -> ambiguous -> DEFER."""
    P = [1 - p_pos, p_pos]
    return [c for c in (0, 1) if (1 - P[c]) <= thr[c]]


# --------------------------------------------------------------------------- #
#  Decision gate (IDEAL_PIPELINE.md §5.4 flowchart)
# --------------------------------------------------------------------------- #
def _ambiguous(p, conformal_thr, op_thr, margin):
    """Is the call near the boundary / not a confident singleton? (the '§5.4 near boundary?' node)"""
    if conformal_thr is not None:                    # principled option: conformal prediction set
        return len(conformal_set(p, conformal_thr)) != 1
    return abs(p - op_thr) < margin                  # simple option: within a margin of the operating point


def decide(p, conflict_K, *, route, conformal_thr=None, op_thr=0.5,
           boundary_margin=0.0, conflict_flag=0.45, defer_tabular_alone=True):
    """Turn (fused prob, conflict) into RELEASE / DEFER / FLAG, per IDEAL_PIPELINE.md §5.3-§5.4:

      * FLAG  — the two experts strongly disagree (conflict K >= conflict_flag).  [both present only]
      * DEFER — a lone weak expert (questionnaire) is all we have, OR the call sits near the boundary.
      * RELEASE — a confident, non-conflicting call.
    """
    if conflict_K is not None and conflict_K >= conflict_flag:
        return FLAG                                  # experts contradict -> never auto-decide silently
    if route == "tabular_only" and defer_tabular_alone:
        return DEFER                                 # weak lone triage -> recommend the real test
    return DEFER if _ambiguous(p, conformal_thr, op_thr, boundary_margin) else RELEASE


# --------------------------------------------------------------------------- #
#  Any-subset inference API  (stages 6-7: route -> combine -> decide)
# --------------------------------------------------------------------------- #
def predict(inputs, priors, target_prev=0.10, mode="cascade",
            reliabilities=None, conformal=None, op_thr=0.5,
            boundary_margin=0.0, conflict_flag=0.45, defer_tabular_alone=True):
    """Route + combine + decide for whatever modalities are present (IDEAL_PIPELINE.md §5).

    inputs      : {'p_img': float|None, 'p_tab': float|None} (missing key/None = missing modality).
    priors      : {'img': src_prior_img, 'tab': src_prior_tab} — each expert's TRAINING prevalence
                  (SIPaKMeD abnormal-cell rate; UCI target base rate) for prior-shift re-anchoring.
    target_prev : DECLARED deployment prevalence to anchor both experts to (default 0.10).
    mode        : 'cascade' (RECOMMENDED — robust SUM base + SAFETY ASYMMETRY: the questionnaire can
                  RAISE suspicion or trigger abstention but NEVER lower a positive cytology), 'sum'
                  (Kittler symmetric pool), 'nb' (naive-Bayes LR; overconfident under shared-HPV dependence).
    op_thr      : operating threshold on the ANCHORED scale — the point above which the image 'calls
                  positive' (the cascade floor) and the centre of the near-boundary DEFER band.
    conformal   : optional {'img': thr, 'tab': thr} from conformal_thresholds -> DEFER on an ambiguous set.
    boundary_margin : if no conformal thr, DEFER when |p - op_thr| < margin (0 disables it).

    Returns: p_fused, route, decision {RELEASE/DEFER/FLAG}, conflict, prior-anchored per-expert probs.
    """
    reliabilities = reliabilities or {}
    conformal = conformal or {}
    pi, pt = inputs.get("p_img"), inputs.get("p_tab")
    if pi is None and pt is None:
        raise ValueError("need at least one of p_img / p_tab")

    ai = prior_shift(pi, priors["img"], target_prev) if pi is not None else None   # anchored posteriors
    at = prior_shift(pt, priors["tab"], target_prev) if pt is not None else None

    # ---- single modality: that expert alone (prior-anchored) ----
    if (pi is None) != (pt is None):
        img = pi is not None
        p = ai if img else at
        route = "image_only" if img else "tabular_only"
        decision = decide(p, None, route=route, conformal_thr=conformal.get("img" if img else "tab"),
                          op_thr=op_thr, boundary_margin=boundary_margin,
                          defer_tabular_alone=defer_tabular_alone)
        return dict(p_fused=float(p), route=route, decision=decision, conflict=None,
                    p_img=ai, p_tab=at, mode="single")

    # ---- both present: combine ----
    K = ds_conflict(ai, at, reliabilities.get("img", 1.0), reliabilities.get("tab", 1.0))
    if mode == "nb":
        p = naive_bayes_lr(pi, pt, priors["img"], priors["tab"], target_prev)
    elif mode == "sum":
        p = sum_rule([ai, at])
    else:  # cascade (asymmetric): robust Kittler SUM base + SAFETY ASYMMETRY
        p = sum_rule([ai, at])                      # robust to the high-variance tabular posterior
        if ai >= op_thr:                            # image is calling positive -> risk can only RAISE
            p = max(p, ai)                          # ...never lower it (the safety floor)

    decision = decide(p, K, route="fused", conformal_thr=conformal.get("img"),
                      op_thr=op_thr, boundary_margin=boundary_margin, conflict_flag=conflict_flag)
    return dict(p_fused=float(p), route="fused", decision=decision, conflict=float(K),
                p_img=ai, p_tab=at, mode=mode)


# --------------------------------------------------------------------------- #
#  smoke test
# --------------------------------------------------------------------------- #
def _show(tag, r):
    print(f"  {tag:16s} p_fused={r['p_fused']:.3f}  route={r['route']:12s} "
          f"decision={r['decision']:8s} conflict={r['conflict']}")


if __name__ == "__main__":
    priors = dict(img=0.40, tab=0.064)            # SIPaKMeD abnormal-cell rate ; UCI Biopsy base rate
    tp, op = 0.10, 0.30                           # deployment prevalence ; operating threshold (anchored)
    kw = dict(op_thr=op, boundary_margin=0.10)
    print("=== single modality ===")
    _show("image only",   predict(dict(p_img=0.90), priors, tp, **kw))          # confident -> RELEASE
    _show("image borderline", predict(dict(p_img=0.55), priors, tp, **kw))      # near op_thr -> DEFER
    _show("tabular only",  predict(dict(p_tab=0.80), priors, tp, **kw))         # weak lone -> DEFER
    print("=== both present ===")
    _show("both agree +",  predict(dict(p_img=0.90, p_tab=0.80), priors, tp, **kw))
    _show("both agree -",  predict(dict(p_img=0.10, p_tab=0.15), priors, tp, **kw))
    _show("both sum-pool",  predict(dict(p_img=0.90, p_tab=0.80), priors, tp, mode="sum", **kw))
    conflict = predict(dict(p_img=0.95, p_tab=0.02), priors, tp, **kw)          # strong disagree -> FLAG
    _show("img+/risk-",    conflict)

    # ---- invariants from IDEAL_PIPELINE.md §5.2 / §5.5 ----
    hi_img_lo_tab = predict(dict(p_img=0.95, p_tab=0.02), priors, tp, **kw)
    img_alone = predict(dict(p_img=0.95), priors, tp, **kw)
    assert hi_img_lo_tab["p_fused"] >= img_alone["p_fused"] - 1e-9, "safety asymmetry violated"
    assert predict(dict(p_tab=0.80), priors, tp, **kw)["decision"] == DEFER, "lone tabular must DEFER"
    assert conflict["decision"] == FLAG, "strong disagreement must FLAG"
    # missing modality == the single-expert path (graceful degradation)
    both_missing_tab = predict(dict(p_img=0.95, p_tab=None), priors, tp, **kw)
    assert abs(both_missing_tab["p_fused"] - img_alone["p_fused"]) < 1e-9, "None p_tab != image-only"
    print("\nOK — safety asymmetry holds; lone tabular DEFERs; conflict FLAGs; missing==single.")
