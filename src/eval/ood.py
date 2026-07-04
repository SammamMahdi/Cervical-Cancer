"""GATE 3 — Domain-energy OOD premise: a label-free free-energy abstention signal.

Free energy (Liu et al. 2020, "Energy-based OOD detection"):

    E(x) = -T * logsumexp(logits(x) / T)        (here T = 1)

computed from the RAW 5-class LOGITS (not softmax). Lower energy == higher
confidence/in-distribution; higher energy == more out-of-distribution. Energy is
label-free (needs only the logits), so it is a candidate ABSTENTION signal for a
deployed system that cannot see target labels.

This module provides:
  * free_energy(logits)         -> per-sample energy E(x)  [N]
  * energy_logits(model, loader)-> (energy [N], logits [N,5]) via one forward pass
  * ood_auroc(e_in, e_ext)      -> AUROC of (internal-vs-external) using energy as
                                   the OOD score (external = positive class).

The GATE-3 premise: a good OOD score should rank-order the domains by their measured
ECE collapse, i.e. mean energy should rise monotonically with external ECE
(SIPaKMeD-internal < Herlev < HiCervix < APACC < CRIC).
"""
import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

import config as C


def free_energy(logits, T=1.0):
    """Free energy E(x) = -T * logsumexp(logits / T) from RAW logits.

    logits : array-like [N, K] of raw (pre-softmax) class logits.
    Returns: float64 array [N] of per-sample energies (lower = in-distribution).
    """
    logits = np.asarray(logits, dtype=np.float64)
    return (-T * logsumexp(logits / T, axis=1)).astype(np.float64)


@torch.no_grad()
def energy_logits(model, loader, desc="energy", max_n=None):
    """Run `model` over `loader` capturing RAW LOGITS; return (energy [N], logits [N,K]).

    Eval / no_grad / BF16 autocast on CUDA (same inference path as eval.metrics.predict),
    but keeps the RAW logits (no softmax). Stops early once `max_n` samples are collected
    (sub-sampling for speed); returns the exact n captured.
    """
    from tqdm.auto import tqdm
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    n_seen = 0
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        xb = batch[0].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
            lg = model(xb)
        lg = lg.float().cpu().numpy()
        chunks.append(lg)
        n_seen += lg.shape[0]
        if max_n is not None and n_seen >= max_n:
            break
    logits = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, C.NUM_CLASSES))
    if max_n is not None and logits.shape[0] > max_n:
        logits = logits[:max_n]
    energy = free_energy(logits)
    return energy, logits


@torch.no_grad()
def extract_features(model, loader, desc="feats", max_n=None):
    """Run `model.features_vec` over `loader` capturing the POOLED PENULTIMATE embedding.

    Returns (feats [N, D], labels [N]). `feats` is the exact vector fed to the final head
    (models.backbones.Classifier.features_vec). Eval / no_grad / BF16 autocast on CUDA (same
    inference path as energy_logits). Stops early once `max_n` samples are collected.
    Labels are captured when the loader yields (x, y); else filled with -1.
    """
    from tqdm.auto import tqdm
    model.eval()
    device = next(model.parameters()).device
    fchunks, ychunks = [], []
    n_seen = 0
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        xb = batch[0].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
            fv = model.features_vec(xb)
        fchunks.append(fv.float().cpu().numpy())
        if len(batch) > 1 and batch[1] is not None:
            ychunks.append(np.asarray(batch[1]).reshape(-1))
        else:
            ychunks.append(np.full(xb.shape[0], -1))
        n_seen += xb.shape[0]
        if max_n is not None and n_seen >= max_n:
            break
    if not fchunks:
        return np.zeros((0, model.feature_dim)), np.zeros((0,), dtype=int)
    feats = np.concatenate(fchunks, axis=0)
    labels = np.concatenate(ychunks, axis=0)
    if max_n is not None and feats.shape[0] > max_n:
        feats, labels = feats[:max_n], labels[:max_n]
    return feats, labels


class MahalanobisScorer:
    """Class-conditional Mahalanobis OOD scorer in the penultimate feature space.

    Fit per-class means mu_k on SIPaKMeD TRAIN features and ONE SHARED covariance estimated
    by Ledoit-Wolf shrinkage (sklearn.covariance.LedoitWolf) over the within-class deviations
    (features minus their own class mean), i.e. the pooled within-class scatter. The OOD score
    of a sample x is

        score(x) = min_k  (feat(x) - mu_k)^T  Sigma^-1  (feat(x) - mu_k)

    the squared Mahalanobis distance to the NEAREST class Gaussian. Higher score == farther from
    every in-distribution class == more out-of-distribution.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None        # [K, D]
        self.precision_ = None    # [D, D] = Sigma^-1
        self.shrinkage_ = None

    def fit(self, feats, labels):
        from sklearn.covariance import LedoitWolf
        feats = np.asarray(feats, dtype=np.float64)
        labels = np.asarray(labels).reshape(-1)
        self.classes_ = np.unique(labels[labels >= 0])
        means = []
        centered = np.empty_like(feats)
        for k in self.classes_:
            m = labels == k
            mu = feats[m].mean(axis=0)
            means.append(mu)
            centered[m] = feats[m] - mu          # within-class deviation (shared cov)
        self.means_ = np.stack(means, axis=0)
        lw = LedoitWolf(assume_centered=True).fit(centered)
        self.precision_ = lw.precision_.astype(np.float64)
        self.shrinkage_ = float(lw.shrinkage_)
        return self

    def score(self, feats):
        """Per-sample OOD score = min_k squared Mahalanobis distance to class mean. [N]"""
        feats = np.asarray(feats, dtype=np.float64)
        if feats.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        dists = np.empty((feats.shape[0], self.means_.shape[0]), dtype=np.float64)
        for j, mu in enumerate(self.means_):
            d = feats - mu                              # [N, D]
            dists[:, j] = np.einsum("ni,ij,nj->n", d, self.precision_, d)
        return dists.min(axis=1)


def ood_auroc(e_in, e_ext):
    """AUROC of internal-vs-external using energy as the OOD score (external = positive=1).

    Higher energy should mark external (OOD) samples, so we score with raw energy and label
    external as 1, internal as 0. Returns float (nan if a class is empty).
    """
    e_in = np.asarray(e_in, dtype=np.float64)
    e_ext = np.asarray(e_ext, dtype=np.float64)
    if e_in.size == 0 or e_ext.size == 0:
        return float("nan")
    y = np.concatenate([np.zeros(e_in.size), np.ones(e_ext.size)])
    s = np.concatenate([e_in, e_ext])
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def summarize_energy(e):
    """Summary stats for an energy array (mean/std/median/quartiles + n)."""
    e = np.asarray(e, dtype=np.float64)
    if e.size == 0:
        return dict(n=0, mean=float("nan"), std=float("nan"), median=float("nan"))
    return dict(
        n=int(e.size),
        mean=float(e.mean()),
        std=float(e.std()),
        median=float(np.median(e)),
        p25=float(np.percentile(e, 25)),
        p75=float(np.percentile(e, 75)),
    )


if __name__ == "__main__":
    # Dependency-free smoke test of the energy math (no model / no disk).
    rng = np.random.default_rng(0)
    # confident (peaked) logits -> low energy; flat logits -> high energy
    peaked = rng.normal(size=(100, 5)); peaked[:, 0] += 8.0
    flat = rng.normal(scale=0.1, size=(100, 5))
    e_peaked = free_energy(peaked)
    e_flat = free_energy(flat)
    print(f"mean E peaked={e_peaked.mean():.3f}  mean E flat={e_flat.mean():.3f}")
    assert e_peaked.mean() < e_flat.mean(), "peaked logits should have LOWER energy"
    auroc = ood_auroc(e_peaked, e_flat)   # flat = 'external'
    print(f"AUROC(peaked-vs-flat)={auroc:.3f}")
    assert auroc > 0.9
    print("OK")
