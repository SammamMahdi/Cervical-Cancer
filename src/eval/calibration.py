"""U5 Confidence calibration: temperature scaling (Guo et al., ICML 2017) + ECE.

A trained classifier is typically OVER-confident: its softmax peak overstates the true probability
of correctness. For a clinical screening model this matters more than raw accuracy — a downstream
triage threshold (ABNORMAL_THRESHOLD) is only meaningful if the probabilities are honest.

Temperature scaling is the simplest, most reliable post-hoc fix: it learns ONE positive scalar T,
divides the logits by it (softmax(z / T)) and leaves the arg-max — hence the accuracy — untouched.
T is fit by minimising NLL on a held-out VALIDATION set (never the test set), then frozen and applied
to test logits. We report Expected Calibration Error (ECE) before/after plus the reliability-diagram
bins (mean confidence vs. empirical accuracy per bin) so the calibration figure can be drawn elsewhere.

Public API
----------
    TemperatureScaler(nn.Module)         single log_T param; .forward(logits)=logits/T; .fit(...); .T
    fit_temperature(logits, labels)->T   convenience scalar fit (LBFGS on NLL)
    apply_temperature(logits, T)->logits scaled logits (np in -> np out, tensor in -> tensor out)
    ece(probs, labels, n_bins=15)->float Expected Calibration Error
    reliability_bins(probs, labels, n_bins=15)->(bin_conf, bin_acc, bin_count)
    calibrate_and_report(val_logits, val_labels, test_logits, test_labels)->dict
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ----------------------------------------------------------------------------- helpers
def _as_tensor(x, dtype=torch.float32):
    """Accept np.ndarray / list / torch.Tensor -> float32 CPU tensor (detached)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _as_long(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.long)
    return torch.as_tensor(np.asarray(x), dtype=torch.long)


# ----------------------------------------------------------------------------- module
class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaler.

    We parameterise log(T) so T = exp(log_T) is strictly positive without a constraint, which keeps
    the LBFGS optimisation unconstrained and numerically stable. forward divides logits by T.
    """

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(float(init_T))))

    @property
    def T(self) -> float:
        """Current temperature as a python float."""
        return float(self.log_T.detach().exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature: z / T."""
        return logits / self.log_T.exp()

    @torch.enable_grad()
    def fit(self, logits, labels, max_iter: int = 100, lr: float = 0.01):
        """Optimise NLL (cross-entropy) over T with LBFGS on the VALIDATION logits. Returns self."""
        z = _as_tensor(logits)
        y = _as_long(labels)
        opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def _closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(_closure)
        return self


# ----------------------------------------------------------------------------- functional API
def fit_temperature(logits, labels, max_iter: int = 100, lr: float = 0.01) -> float:
    """Fit and return the optimal temperature T (float) on (val) logits via LBFGS NLL."""
    return TemperatureScaler().fit(logits, labels, max_iter=max_iter, lr=lr).T


def apply_temperature(logits, T: float):
    """Divide logits by temperature T. np.ndarray in -> np.ndarray out; tensor in -> tensor out."""
    if isinstance(logits, torch.Tensor):
        return logits / float(T)
    return np.asarray(logits, dtype=np.float64) / float(T)


def _softmax_np(logits) -> np.ndarray:
    """Numerically stable row-wise softmax -> (N, K) float64 probabilities."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- ECE / reliability
def reliability_bins(probs, labels, n_bins: int = 15):
    """Equal-width confidence bins on [0,1].

    For each bin return (mean predicted confidence, empirical accuracy, count). Confidence = max
    softmax prob (top-1). Empty bins yield 0 for conf/acc and 0 count. Returns three length-n_bins
    float arrays: (bin_conf, bin_acc, bin_count).
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64).ravel()
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.float64)

    # bin index in [0, n_bins-1]; include the right edge (conf==1.0) in the last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        bin_count[b] = c
        if c > 0:
            bin_conf[b] = conf[m].mean()
            bin_acc[b] = correct[m].mean()
    return bin_conf, bin_acc, bin_count


def ece(probs, labels, n_bins: int = 15) -> float:
    """Expected Calibration Error: sum_b (n_b / N) * |acc_b - conf_b| (top-1, equal-width bins)."""
    bin_conf, bin_acc, bin_count = reliability_bins(probs, labels, n_bins=n_bins)
    n = bin_count.sum()
    if n <= 0:
        return 0.0
    return float((bin_count / n * np.abs(bin_acc - bin_conf)).sum())


# ----------------------------------------------------------------------------- convenience report
def calibrate_and_report(val_logits, val_labels, test_logits, test_labels, n_bins: int = 15) -> dict:
    """Fit T on validation logits, apply to test logits, and report ECE + reliability before/after.

    Returns dict(T, ece_pre, ece_post, reliability_pre, reliability_post) where each reliability_*
    is the (bin_conf, bin_acc, bin_count) tuple on the TEST set. ECE/reliability are computed on the
    held-out test logits so the numbers reflect generalisation, not the data T was tuned on.
    """
    T = fit_temperature(val_logits, val_labels)

    probs_pre = _softmax_np(test_logits)
    probs_post = _softmax_np(apply_temperature(np.asarray(test_logits, dtype=np.float64), T))

    return dict(
        T=float(T),
        ece_pre=ece(probs_pre, test_labels, n_bins=n_bins),
        ece_post=ece(probs_post, test_labels, n_bins=n_bins),
        reliability_pre=reliability_bins(probs_pre, test_labels, n_bins=n_bins),
        reliability_post=reliability_bins(probs_post, test_labels, n_bins=n_bins),
    )


# ----------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    C.set_seed()
    rng = np.random.default_rng(C.SEED)
    K = C.NUM_CLASSES

    def _make(n):
        # ground-truth class means in logit space, then OVER-confident (scaled-up) logits => miscalibrated
        y = rng.integers(0, K, size=n)
        base = rng.normal(0.0, 1.0, size=(n, K))
        base[np.arange(n), y] += 2.0          # signal so the model is mostly right
        logits = base * 2.5                   # inflate -> over-confident -> high ECE
        return logits, y

    val_logits, val_labels = _make(2000)
    test_logits, test_labels = _make(2000)

    rep = calibrate_and_report(val_logits, val_labels, test_logits, test_labels)
    print(f"T          = {rep['T']:.4f}")
    print(f"ECE  pre   = {rep['ece_pre']:.4f}")
    print(f"ECE  post  = {rep['ece_post']:.4f}")

    bc, ba, bn = rep["reliability_post"]
    print(f"bins(post) conf={np.round(bc, 3)}")
    print(f"bins(post) acc ={np.round(ba, 3)}")
    print(f"bins(post) cnt ={bn.astype(int)}")

    # accuracy must be preserved by temperature scaling (arg-max invariant)
    acc_pre = (_softmax_np(test_logits).argmax(1) == test_labels).mean()
    acc_post = (_softmax_np(apply_temperature(test_logits, rep["T"])).argmax(1) == test_labels).mean()
    assert abs(acc_pre - acc_post) < 1e-9, "temperature scaling must NOT change accuracy"

    # calibration should improve (usually). T>1 means it cooled an over-confident model.
    assert rep["ece_post"] <= rep["ece_pre"] + 1e-6, "ECE should not get worse on synthetic over-conf data"
    print("OK: accuracy preserved, ece_post <= ece_pre, T =", round(rep["T"], 3))


# ---------------------------------------------------------------------------
# Per-domain BINARY recalibration (C): operating-point + score calibration on P(abnormal).
# ---------------------------------------------------------------------------
def binary_ece(p_abn, y_bin, n_bins=15):
    """ECE of a binary score: top-confidence max(p,1-p) vs accuracy, equal-width bins."""
    p = np.asarray(p_abn, float); y = np.asarray(y_bin, int)
    pred = (p >= 0.5).astype(int); conf = np.maximum(p, 1 - p); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def recalibrate_binary(p_abn, y_bin):
    """Per-domain operating-point + score calibration on P(abnormal) (honest UPPER BOUND: tuned on the
    domain's own labels). Returns the fixed-tau=0.5 baseline vs an oracle threshold (max balanced acc),
    and ECE before/after a 1-D Platt fit on logit(P(abnormal))."""
    from sklearn.metrics import accuracy_score
    from eval.metrics import balanced_accuracy as _balacc       # silences the single-class warning
    p = np.clip(np.asarray(p_abn, float), 1e-6, 1 - 1e-6); y = np.asarray(y_bin, int)
    out = dict(n=int(len(y)), base_abnormal_rate=float(y.mean()),
               acc_tau50=float(accuracy_score(y, (p >= 0.5).astype(int))),
               balacc_tau50=_balacc(y, (p >= 0.5).astype(int)),
               ece_pre=binary_ece(p, y))
    if len(np.unique(y)) < 2:                      # degenerate single-class slice -> no threshold to tune
        out.update(tau_oracle=0.5, acc_oracle=out["acc_tau50"], balacc_oracle=out["balacc_tau50"],
                   ece_post=out["ece_pre"], note="single-class domain; threshold/Platt undefined")
        return out
    taus = np.linspace(0.02, 0.98, 97)
    bal = [_balacc(y, (p >= t).astype(int)) for t in taus]
    t_star = float(taus[int(np.argmax(bal))])
    out.update(tau_oracle=round(t_star, 3),
               acc_oracle=float(accuracy_score(y, (p >= t_star).astype(int))),
               balacc_oracle=float(max(bal)))
    # 1-D Platt scaling on the logit of P(abnormal) -> recalibrated probability
    try:
        from sklearn.linear_model import LogisticRegression
        z = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y)
        p_cal = lr.predict_proba(z)[:, 1]
        out["ece_post"] = binary_ece(p_cal, y)
    except Exception as ex:
        out["ece_post"] = out["ece_pre"]; out["platt_error"] = str(ex)
    return out
