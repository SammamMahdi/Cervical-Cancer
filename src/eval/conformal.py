"""Axis-4 / U9 conformal prediction: distribution-free, finite-sample coverage guarantees.

A calibrated softmax tells you HOW confident the model is; split conformal prediction turns that
into a guarantee you can put in front of a clinician. Given a target error rate alpha, it emits a
PREDICTION SET (a subset of the 5 classes) such that, marginally over a fresh test point drawn from
the same distribution as the calibration data, the true class lies in the set with probability at
least 1 - alpha. The only assumption is exchangeability of calibration + test points -- no model
re-training, no distributional form, no asymptotics.

The recipe (split / inductive conformal):
  1. On a held-out VALIDATION set compute a non-conformity score s_i for the TRUE class of each point
     (high score == the point looks unusual / hard under the model).
  2. qhat = the ceil((n+1)(1-alpha))/n empirical quantile of those scores (the finite-sample
     correction (n+1)/n is what buys the exact >=1-alpha guarantee, not >=1-alpha-on-average).
  3. For a test point, include every class whose score would be <= qhat. That set is the prediction.

Two score functions, each with a plain (marginal) and a Mondrian (class-conditional) variant:
  * LAC  (Least Ambiguous set-valued Classifier; aka "softmax"/"threshold"): s = 1 - p_softmax[true].
    Smallest sets at a given coverage, but coverage is only guaranteed marginally.
  * APS  (Adaptive Prediction Sets, Romano et al. 2020): sort softmax descending, s = cumulative mass
    up to and including the true class. Larger but more adaptive sets; coverage is more even across
    the probability range.
  * Mondrian (class-conditional): fit a SEPARATE qhat per (true) class so coverage holds WITHIN each
    class -- important here because the abnormal classes are the clinically costly ones and we do not
    want their coverage silently traded away to flatter the easy normal classes.

Dependency-free: pure numpy (probabilities in, sets out). No torch, no sklearn.

Public API
----------
    fit_conformal(val_probs, val_labels, alpha=0.1, method='aps', mondrian=False)
        -> qhat : float (marginal) or np.ndarray[K] (Mondrian, one threshold per class)
    predict_sets(test_probs, qhat, method='aps', mondrian=False)
        -> list[set[int]] : one prediction set (set of class indices) per test row
    report(val_probs, val_labels, test_probs, test_labels, alpha=0.1, method='aps', mondrian=True)
        -> dict(method, alpha, mondrian, qhat, coverage, avg_set_size, per_class_coverage, n)
           and writes results/metrics/conformal.json
"""
import os
import json

import numpy as np

import config as C

METHODS = ("lac", "aps")


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _as_probs(probs):
    """(N, K) float64 array; defensively renormalise so rows sum to 1 (guards logits/odd input)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be 2-D (N,K), got shape {p.shape}")
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def _as_labels(labels):
    return np.asarray(labels).astype(np.int64).ravel()


def _conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile of 1-D `scores` at level 1-alpha.

    Uses the rank ceil((n+1)(1-alpha)); if that rank exceeds n the threshold is +inf (the set must
    contain every class to honour the guarantee with so few calibration points). Returns a float.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = s.size
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k >= n + 1:        # quantile lands past the largest score -> cover everything
        return np.inf
    return float(s[k - 1])   # k is 1-indexed rank


# --------------------------------------------------------------------------- #
#  non-conformity scores
# --------------------------------------------------------------------------- #
def _lac_true_scores(probs, labels):
    """LAC calibration scores for the TRUE class: s_i = 1 - p_i[y_i]. Shape (N,)."""
    p = _as_probs(probs)
    y = _as_labels(labels)
    return 1.0 - p[np.arange(p.shape[0]), y]


def _lac_all_scores(probs):
    """LAC score of EVERY class for each row: s[i, c] = 1 - p_i[c]. Shape (N, K)."""
    return 1.0 - _as_probs(probs)


def _aps_true_scores(probs, labels):
    """APS calibration scores for the TRUE class.

    Sort each row's softmax descending; the score of the true class is the cumulative probability
    mass of all classes ranked at least as high as the true class, INCLUDING the true class itself
    (the standard non-randomised APS score). Shape (N,).
    """
    p = _as_probs(probs)
    y = _as_labels(labels)
    n = p.shape[0]
    order = np.argsort(-p, axis=1)                 # descending; (N,K) column = rank
    sorted_p = np.take_along_axis(p, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)             # cumulative mass at each rank
    # rank position of the true class within the sorted order, per row
    rank = np.argmax(order == y[:, None], axis=1)  # first (only) match per row
    return csum[np.arange(n), rank]


def _aps_all_scores(probs):
    """APS score of EVERY class for each row, mapped back to ORIGINAL class index order.

    For class c, score = cumulative softmax mass down to and including c when ranked descending.
    Returns (N, K) aligned to class index (not rank order).
    """
    p = _as_probs(probs)
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)             # (N,K) in RANK order
    # scatter cumulative mass back to original class positions
    scores = np.empty_like(csum)
    np.put_along_axis(scores, order, csum, axis=1)
    return scores


def _true_scores(probs, labels, method):
    if method == "lac":
        return _lac_true_scores(probs, labels)
    if method == "aps":
        return _aps_true_scores(probs, labels)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def _all_scores(probs, method):
    if method == "lac":
        return _lac_all_scores(probs)
    if method == "aps":
        return _aps_all_scores(probs)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


# --------------------------------------------------------------------------- #
#  fit / predict
# --------------------------------------------------------------------------- #
def fit_conformal(val_probs, val_labels, alpha=0.1, method="aps", mondrian=False):
    """Calibrate the conformal threshold qhat on the VALIDATION set.

    method   : 'lac' (1 - softmax[true]) or 'aps' (cumulative sorted softmax).
    mondrian : if True, fit a separate qhat PER (true) class -> class-conditional coverage. Classes
               absent from val_labels get +inf (cover them entirely until calibration data exists).

    Returns a float (marginal) or an np.ndarray of shape (K,) (Mondrian).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    scores = _true_scores(val_probs, val_labels, method)
    y = _as_labels(val_labels)

    if not mondrian:
        return _conformal_quantile(scores, alpha)

    K = _as_probs(val_probs).shape[1]
    qhat = np.full(K, np.inf, dtype=np.float64)
    for c in range(K):
        m = (y == c)
        if m.any():
            qhat[c] = _conformal_quantile(scores[m], alpha)
    return qhat


def predict_sets(test_probs, qhat, method="aps", mondrian=False):
    """Build prediction sets for each test row given a calibrated `qhat`.

    A class c is INCLUDED for row i iff its non-conformity score s[i,c] <= the relevant threshold
    (the single qhat for marginal; qhat[c] for Mondrian). Every set is non-empty by construction:
    if a row would otherwise be empty we add its arg-max (lowest-score / most-conformal) class so a
    clinician always sees at least one candidate.

    Returns list[set[int]] of length N (class indices).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    scores = _all_scores(test_probs, method)        # (N,K), lower == more conformal
    n, K = scores.shape

    if mondrian:
        thr = np.asarray(qhat, dtype=np.float64).reshape(1, K)   # broadcast per-class
    else:
        thr = float(qhat)

    include = scores <= thr                          # (N,K) boolean membership
    sets = []
    for i in range(n):
        members = set(np.flatnonzero(include[i]).tolist())
        if not members:                             # guarantee non-empty: keep the most conformal
            members = {int(scores[i].argmin())}
        sets.append(members)
    return sets


# --------------------------------------------------------------------------- #
#  report
# --------------------------------------------------------------------------- #
def report(val_probs, val_labels, test_probs, test_labels,
           alpha=0.1, method="aps", mondrian=True, save=True, out_path=None):
    """Fit on validation, evaluate prediction sets on test, summarise + persist.

    Returns dict with:
      method, alpha, mondrian, qhat (float or list[K]),
      coverage           : fraction of test rows whose set contains the true class (target >=1-alpha),
      avg_set_size       : mean number of classes per prediction set (efficiency; smaller is better),
      per_class_coverage : {class_name: coverage on rows whose TRUE label is that class},
      n                  : number of test rows.
    Writes results/metrics/conformal.json (unless save=False).
    """
    qhat = fit_conformal(val_probs, val_labels, alpha=alpha, method=method, mondrian=mondrian)
    sets = predict_sets(test_probs, qhat, method=method, mondrian=mondrian)

    y = _as_labels(test_labels)
    n = y.size
    hit = np.array([y[i] in sets[i] for i in range(n)], dtype=np.float64)
    size = np.array([len(s) for s in sets], dtype=np.float64)

    K = _as_probs(test_probs).shape[1]
    per_class = {}
    for c in range(K):
        m = (y == c)
        name = C.CLASSES[c] if c < len(C.CLASSES) else str(c)
        per_class[name] = float(hit[m].mean()) if m.any() else float("nan")

    out = dict(
        method=method,
        alpha=float(alpha),
        mondrian=bool(mondrian),
        qhat=(qhat.tolist() if isinstance(qhat, np.ndarray) else float(qhat)),
        coverage=float(hit.mean()) if n else float("nan"),
        avg_set_size=float(size.mean()) if n else float("nan"),
        per_class_coverage=per_class,
        n=int(n),
    )

    if save:
        os.makedirs(C.METRICDIR, exist_ok=True)
        path = out_path or os.path.join(C.METRICDIR, "conformal.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return out


# --------------------------------------------------------------------------- #
#  smoke test: synthetic, well-calibrated logits -> empirical coverage ~ 1-alpha
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(C.SEED)
    K = C.NUM_CLASSES
    alpha = 0.1

    def _make(n):
        """Draw well-calibrated softmax rows + labels sampled FROM those probabilities.

        Sampling y ~ Categorical(p_i) makes p_i the genuine posterior, so the softmax is calibrated
        by construction -- exactly the regime where conformal coverage should land on 1 - alpha.
        """
        logits = rng.normal(0.0, 1.5, size=(n, K))
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(K, p=p[i]) for i in range(n)], dtype=np.int64)
        return p, y

    val_probs, val_labels = _make(5000)
    test_probs, test_labels = _make(5000)

    for method in METHODS:
        for mondrian in (False, True):
            rep = report(val_probs, val_labels, test_probs, test_labels,
                         alpha=alpha, method=method, mondrian=mondrian, save=False)
            cov, sz = rep["coverage"], rep["avg_set_size"]
            tag = f"{method:>3} mondrian={int(mondrian)}"
            print(f"{tag}: coverage={cov:.3f} (target {1 - alpha:.2f})  avg_set_size={sz:.3f}")
            assert abs(cov - (1 - alpha)) <= 0.03, \
                f"{tag}: coverage {cov:.3f} not within 0.03 of {1 - alpha:.2f}"

    # the file write path also works end-to-end
    rep = report(val_probs, val_labels, test_probs, test_labels,
                 alpha=alpha, method="aps", mondrian=True, save=True)
    print("wrote", os.path.join(C.METRICDIR, "conformal.json"))
    print("OK: all methods within 0.03 of nominal coverage")
