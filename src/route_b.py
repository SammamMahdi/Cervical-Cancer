"""Route-B: open-set / OOD gate + constrained pseudo-labeling (LABEL_ALIGNMENT_METHOD).

Route-A (data.labelmap) is a deterministic, hand-built MANY-TO-ONE crosswalk: it never invents a
5-class label for an external cell, only collapses BOTH sides to a shared safe tier (E0/E1) or to a
morphology candidate set. Route-B is the *learned* alternative — it lets the trained 5-class model
itself propose SIPaKMeD labels for external cells — but only behind two safety valves:

  1. an OPEN-SET / OOD gate, so cells that do not look like SIPaKMeD training data are rejected
     rather than force-classified, and
  2. a SOURCE CONSTRAINT, so a kept pseudo-label must lie inside the candidate set allowed by the
     external cell's own native diagnostic label (labelmap.EXTERNAL_CANDIDATES). The model can pick
     *which* SIPaKMeD class within the allowed set, but can NEVER assign a label the native diagnosis
     forbids (e.g. it can never call a confirmed-normal Herlev cell "Dyskeratotic").

Two complementary OOD scores, both standard and training-free given a fitted model:

  * Mahalanobis (Lee, Lee, Lee & Shin, "A Simple Unified Framework for Detecting Out-of-Distribution
    Samples and Adversarial Attacks", NeurIPS 2018). Fit class-conditional Gaussians on the penultimate
    embeddings (model.features_vec) with PER-CLASS means and a SHARED (tied) covariance; the OOD score
    is the min over classes of the Mahalanobis distance to each class mean (lower distance == more
    in-distribution). We expose it as a CONFIDENCE (higher == more in-distribution) = -min_c d_M^2.
  * Energy (Liu, Wang, Owens & Li, "Energy-based Out-of-distribution Detection", NeurIPS 2020). The
    free energy E(x) = -T * logsumexp(logits / T); LOW energy == in-distribution. We expose the energy
    confidence as -E(x) = T * logsumexp(logits / T) (higher == more in-distribution) so both scores
    share the convention "higher == more in-distribution".

The gate threshold is CALIBRATED on the in-distribution validation split: we keep a target fraction
(`keep_quantile`, default 0.95) of val cells as in-distribution, i.e. the threshold is the
`(1-keep_quantile)` empirical quantile of the val confidence. Anything below threshold at test/external
time is flagged OOD.

CONFIRMATION-BIAS CAVEAT
------------------------
Pseudo-labeling is self-training: the model supplies its own future supervision. This is a known
confirmation-bias / error-amplification risk (Arazo et al., "Pseudo-Labeling and Confirmation Bias in
Deep Semi-Supervised Learning", IJCNN 2020) — a confident-but-wrong pseudo-label is rewarded and the
error is reinforced on the next round. Three mitigations are built in and MUST be kept:
  * high confidence threshold (conf_thresh, default 0.95) — only near-certain predictions are kept;
  * OOD gate — cells unlike the training distribution are dropped, not labeled;
  * source constraint — a pseudo-label is discarded unless it lies inside the candidate set the native
    diagnosis allows, so the label is never *outside* what the ground-truth-adjacent diagnosis permits.
Even so, pseudo-labels are WEAK supervision: report kept/dropped counts, never treat them as ground
truth, and prefer Route-A (deterministic crosswalk) for headline cross-dataset numbers. Route-B is for
*augmentation / domain-adaptation experiments*, with the bias caveat stated.
"""
import os
import json

import numpy as np
import torch

import config as C
import runlog
from data import labelmap as L


# --------------------------------------------------------------------------- #
#  Embedding extraction
# --------------------------------------------------------------------------- #
def _embeddings(model, loader, with_logits=False):
    """Run `model.features_vec` over `loader` -> (feats [N,D] float64, y [N] int64[, logits [N,5]]).

    Eval / no_grad / BF16 autocast on CUDA (matching eval.metrics.predict). Labels come from the
    loader's second element when present (in-distribution loaders) else filled with -1.
    """
    model.eval()
    device = next(model.parameters()).device
    feats, ys, lgs = [], [], []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
                f = model.features_vec(xb)
                lg = model.head(f) if with_logits else None
            feats.append(f.float().cpu().numpy())
            if with_logits:
                lgs.append(lg.float().cpu().numpy())
            if len(batch) > 1 and batch[1] is not None:
                ys.append(np.asarray(batch[1]))
            else:
                ys.append(np.full(len(xb), -1, dtype=np.int64))
    F = np.concatenate(feats).astype(np.float64)
    Y = np.concatenate(ys).astype(np.int64)
    if with_logits:
        return F, Y, np.concatenate(lgs).astype(np.float64)
    return F, Y


# --------------------------------------------------------------------------- #
#  Mahalanobis OOD: fit class-conditional Gaussians (tied covariance)
# --------------------------------------------------------------------------- #
def fit_ood(model, train_loader, reg=1e-6, num_classes=C.NUM_CLASSES):
    """Fit class-conditional Gaussians on TRAIN embeddings (Lee et al., NeurIPS 2018).

    Per-class empirical means mu_c and a single SHARED (tied) covariance Sigma pooled across classes;
    `reg` * I is added to Sigma for a stable inverse. Returns a serializable-ish stats dict:
        {means: (K,D) float64, precision: (D,D) float64, classes: K, dim: D, reg: float}.
    The precision (Sigma^{-1}) is stored so mahalanobis_score is a cheap quadratic form.
    """
    F, Y = _embeddings(model, train_loader)
    D = F.shape[1]
    means = np.zeros((num_classes, D), dtype=np.float64)
    # pooled (within-class) scatter for the tied covariance
    cov = np.zeros((D, D), dtype=np.float64)
    n_total = 0
    for c in range(num_classes):
        m = (Y == c)
        if not m.any():
            # no samples for this class in the loader: leave mean at 0 (will simply never be argmin
            # nearest for real cells); keep going so the fit is still well-defined.
            continue
        Fc = F[m]
        mu = Fc.mean(0)
        means[c] = mu
        Xc = Fc - mu
        cov += Xc.T @ Xc
        n_total += Fc.shape[0]
    cov /= max(1, n_total)                       # tied covariance (pooled within-class)
    cov += reg * np.eye(D)                       # ridge for invertibility
    precision = np.linalg.inv(cov)
    return dict(means=means, precision=precision, classes=int(num_classes), dim=int(D), reg=float(reg))


def mahalanobis_score(model, x, stats):
    """In-distribution CONFIDENCE via the Lee et al. (2018) Mahalanobis detector.

    `x` may be a (N,3,H,W) image batch (run through model.features_vec) OR a precomputed (N,D)
    embedding array/tensor. Returns a (N,) float64 array: -min_c (z-mu_c)^T Sigma^{-1} (z-mu_c)
    (HIGHER == more in-distribution). Also matches the literature's confidence form (negative of the
    minimum squared Mahalanobis distance over the class-conditional Gaussians).
    """
    means = np.asarray(stats["means"])
    prec = np.asarray(stats["precision"])
    # ----- obtain embeddings -----
    if isinstance(x, np.ndarray) and x.ndim == 2:
        Z = x.astype(np.float64)
    elif torch.is_tensor(x) and x.dim() == 2:
        Z = x.detach().float().cpu().numpy().astype(np.float64)
    else:
        model.eval()
        device = next(model.parameters()).device
        xb = x.to(device) if torch.is_tensor(x) else torch.as_tensor(x, device=device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.DEVICE == "cuda")):
                Z = model.features_vec(xb).float().cpu().numpy().astype(np.float64)
    # ----- min over classes of squared Mahalanobis distance -----
    # d_c^2 = sum over j,k of diff_ij * prec_jk * diff_ik, computed per class.
    dists = np.empty((Z.shape[0], means.shape[0]), dtype=np.float64)
    for c in range(means.shape[0]):
        diff = Z - means[c]                      # (N,D)
        dists[:, c] = np.einsum("nd,dk,nk->n", diff, prec, diff)
    return -dists.min(1)                          # confidence: higher == in-distribution


def energy_score(logits, T=1.0):
    """Energy-based OOD CONFIDENCE (Liu et al., NeurIPS 2020).

    Free energy E(x) = -T * logsumexp(logits / T); LOW energy == in-distribution. Returns the energy
    CONFIDENCE -E(x) = T * logsumexp(logits / T) as a (N,) float64 array (HIGHER == in-distribution),
    so it shares the convention with mahalanobis_score. Accepts a numpy array or a torch tensor.
    """
    if torch.is_tensor(logits):
        lg = logits.detach().float()
        conf = T * torch.logsumexp(lg / T, dim=1)
        return conf.cpu().numpy().astype(np.float64)
    lg = np.asarray(logits, dtype=np.float64)
    # numerically stable logsumexp(lg / T)
    z = lg / T
    mx = z.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(z - mx).sum(1))
    return (T * lse).astype(np.float64)


# --------------------------------------------------------------------------- #
#  OOD gate: calibrate a threshold on val, flag in/out
# --------------------------------------------------------------------------- #
def calibrate_threshold(scores, keep_quantile=0.95):
    """Threshold = the (1 - keep_quantile) empirical quantile of in-distribution `scores`.

    With keep_quantile=0.95 we set the cut so ~95% of (in-distribution) val cells score ABOVE it; at
    test/external time anything BELOW the threshold is flagged OOD.
    """
    scores = np.asarray(scores, dtype=np.float64)
    q = float(np.quantile(scores, 1.0 - keep_quantile))
    return q


def ood_gate(model, loader, stats, val_loader=None, method="mahalanobis", T=1.0,
             keep_quantile=0.95, threshold=None):
    """Per-sample in-distribution vs OOD flags for every cell in `loader`.

    method: 'mahalanobis' (uses `stats` from fit_ood) or 'energy' (uses the model logits, temp `T`).
    The decision threshold is either supplied (`threshold`) or CALIBRATED on `val_loader`
    (in-distribution) so a `keep_quantile` fraction of val cells pass the gate. Returns a dict:
        {scores: (N,), is_in: (N,) bool, is_ood: (N,) bool, threshold, method, n_in, n_ood, keep_rate}.
    A cell is in-distribution iff score >= threshold (both scores: higher == in-distribution).
    """
    def _score(ld):
        if method == "mahalanobis":
            F, _ = _embeddings(model, ld)
            return mahalanobis_score(model, F, stats)
        elif method == "energy":
            _, _, lg = _embeddings(model, ld, with_logits=True)
            return energy_score(lg, T=T)
        raise ValueError(f"unknown OOD method {method!r}")

    if threshold is None:
        if val_loader is None:
            raise ValueError("ood_gate needs either `threshold` or a `val_loader` to calibrate on")
        threshold = calibrate_threshold(_score(val_loader), keep_quantile=keep_quantile)

    scores = _score(loader)
    is_in = scores >= threshold
    n = int(len(scores))
    return dict(
        scores=scores,
        is_in=is_in,
        is_ood=~is_in,
        threshold=float(threshold),
        method=method,
        n_in=int(is_in.sum()),
        n_ood=int((~is_in).sum()),
        keep_rate=float(is_in.mean()) if n else 0.0,
    )


# --------------------------------------------------------------------------- #
#  Constrained pseudo-labeling
# --------------------------------------------------------------------------- #
def pseudo_label(model, external_loader, stats, val_loader=None, source_constraint=None,
                 conf_thresh=0.95, ood_method="mahalanobis", T=1.0, keep_quantile=0.95,
                 ood_threshold=None, paths=None, candidate_sets=None, save_path=None):
    """Constrained, OOD-gated pseudo-labeling of external cells (LABEL_ALIGNMENT_METHOD, Route-B).

    For every external cell: predict the 5-class softmax, then KEEP its argmax as a pseudo-label ONLY
    if ALL of the following hold:
      (a) confident:  max softmax probability >= `conf_thresh`;
      (b) in-distribution: passes the OOD gate (not flagged OOD);
      (c) source-constrained: the predicted class lies inside the cell's allowed candidate set.

    The candidate set is the SOURCE CONSTRAINT and comes from the external cell's native diagnostic
    label via labelmap.EXTERNAL_CANDIDATES (e.g. a Herlev "normal" folder -> {SI,Parabasal,Metaplastic};
    a confirmed "light dysplasia" -> {Koilocytotic,Dyskeratotic}). High-grade / glandular / indeterminate
    native labels have NO candidate set (None) and are ALWAYS rejected — the model is never allowed to
    invent a SIPaKMeD label for a cell whose diagnosis has no morphology home. This guarantees a kept
    pseudo-label is NEVER outside the allowed set.

    How candidate sets are provided (first match wins):
      * `candidate_sets`: an explicit per-sample list/array of frozensets|None aligned to `external_loader`
        ordering (and to `paths` if given).
      * `source_constraint`: EITHER a single frozenset|None applied to every cell, OR a dataset name in
        labelmap.EXTERNAL_CANDIDATES — in which case `external_loader.dataset` is expected to expose
        `.cand` (list of candidate frozensets) and `.key_idx` (per-sample index into it), exactly as
        eval.metrics.CandidateExternal does; the per-sample candidate set is then `cand[key_idx[i]]`.
      * if neither is given, NO source constraint is applied (only (a)+(b)) and a caveat is logged.

    Returns dict with:
        kept:   list of (path_or_index, pseudo_label int) for cells passing ALL gates,
        counts: {total, kept, drop_lowconf, drop_ood, drop_constraint, drop_no_candidate},
        plus the gate threshold / method / conf_thresh used.
    """
    # ----- predictions, embeddings, logits in one pass -----
    F, _, logits = _embeddings(model, external_loader, with_logits=True)
    # fp32 stable softmax
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    pred = probs.argmax(1)
    conf = probs.max(1)
    N = len(pred)

    # ----- OOD gate (calibrated on val, or explicit threshold) -----
    if ood_method == "mahalanobis":
        scores = mahalanobis_score(model, F, stats)
    elif ood_method == "energy":
        scores = energy_score(logits, T=T)
    else:
        raise ValueError(f"unknown OOD method {ood_method!r}")
    thr = ood_threshold
    if thr is None:
        if val_loader is None:
            raise ValueError("pseudo_label needs either `ood_threshold` or a `val_loader` to calibrate")
        if ood_method == "mahalanobis":
            vF, _ = _embeddings(model, val_loader)
            vscores = mahalanobis_score(model, vF, stats)
        else:
            _, _, vlg = _embeddings(model, val_loader, with_logits=True)
            vscores = energy_score(vlg, T=T)
        thr = calibrate_threshold(vscores, keep_quantile=keep_quantile)
    is_in = scores >= thr

    # ----- resolve the per-sample candidate sets (source constraint) -----
    per_cand = _resolve_candidates(external_loader, N, source_constraint, candidate_sets)
    if per_cand is None:
        runlog.RunLogger  # no-op reference; explicit caveat below
        constrained = False
        per_cand = [None] * N
    else:
        constrained = True

    # ----- resolve paths for output -----
    if paths is None:
        ds = getattr(external_loader, "dataset", None)
        paths = list(getattr(ds, "paths", range(N)))
    if len(paths) != N:
        paths = list(range(N))

    kept = []
    c = dict(total=int(N), kept=0, drop_lowconf=0, drop_ood=0,
             drop_constraint=0, drop_no_candidate=0)
    for i in range(N):
        if conf[i] < conf_thresh:
            c["drop_lowconf"] += 1
            continue
        if not is_in[i]:
            c["drop_ood"] += 1
            continue
        if constrained:
            cand = per_cand[i]
            if cand is None:
                c["drop_no_candidate"] += 1           # native label has no morphology home
                continue
            if int(pred[i]) not in cand:
                c["drop_constraint"] += 1             # confident+in-dist but FORBIDDEN by source
                continue
        kept.append((paths[i], int(pred[i])))
        c["kept"] += 1

    out = dict(
        kept=kept,
        counts=c,
        constrained=bool(constrained),
        ood_method=ood_method,
        ood_threshold=float(thr),
        conf_thresh=float(conf_thresh),
        keep_quantile=float(keep_quantile),
        caveat=("Pseudo-labels are WEAK self-supervision (confirmation-bias risk); kept only when "
                "confident + in-distribution + inside the source candidate set. Report counts; do NOT "
                "treat as ground truth. See module docstring."),
    )
    if not constrained:
        out["warning"] = ("NO source constraint applied (no candidate_sets / source_constraint) — "
                          "confirmation-bias risk is HIGHER; prefer constraining to EXTERNAL_CANDIDATES.")
    if save_path is not None:
        runlog.dump_json(save_path, dict(counts=c, kept=[[str(p), int(l)] for p, l in kept],
                                         meta={k: out[k] for k in
                                               ("constrained", "ood_method", "ood_threshold",
                                                "conf_thresh", "keep_quantile")}))
    return out


def _resolve_candidates(external_loader, N, source_constraint, candidate_sets):
    """Return a length-N list of candidate frozensets|None, or None if no constraint was supplied."""
    # explicit per-sample list wins
    if candidate_sets is not None:
        cs = list(candidate_sets)
        assert len(cs) == N, f"candidate_sets length {len(cs)} != N {N}"
        return cs
    if source_constraint is None:
        return None
    # a single frozenset|None applied to all cells
    if isinstance(source_constraint, (frozenset, set)):
        return [frozenset(source_constraint)] * N
    if source_constraint is L.EXCLUDE:   # None handled above; keep explicit for clarity
        return None
    # a dataset name -> use dataset.cand / dataset.key_idx (eval.metrics.CandidateExternal layout)
    if isinstance(source_constraint, str):
        ds = getattr(external_loader, "dataset", None)
        cand = getattr(ds, "cand", None)
        key_idx = getattr(ds, "key_idx", None)
        if cand is not None and key_idx is not None:
            return [cand[k] for k in key_idx]
        raise ValueError(
            f"source_constraint={source_constraint!r}: the loader's dataset must expose `.cand` and "
            f"`.key_idx` (e.g. eval.metrics.CandidateExternal). Pass `candidate_sets` instead.")
    raise TypeError(f"unsupported source_constraint type {type(source_constraint)!r}")


# --------------------------------------------------------------------------- #
#  CPU smoke
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import torch.utils.data as tud
    from models.backbones import build

    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    dev = "cpu"

    # tiny RANDOM model (no download); CPU
    model = build("efficientnet_b0", num_classes=C.NUM_CLASSES, hidden=32, dropout=0.0,
                  pretrained=False).to(dev).eval()

    # ---- synthetic "in-distribution" train/val and "OOD" test images ----
    def loader_from(imgs, labels=None, bs=8):
        t = torch.as_tensor(imgs, dtype=torch.float32)
        if labels is None:
            ds = tud.TensorDataset(t, torch.full((len(t),), -1, dtype=torch.long))
        else:
            ds = tud.TensorDataset(t, torch.as_tensor(labels, dtype=torch.long))
        return tud.DataLoader(ds, batch_size=bs, shuffle=False)

    n_tr, n_val, n_ood = 40, 24, 24
    train_imgs = rng.normal(0.0, 1.0, size=(n_tr, 3, C.IMG_SIZE, C.IMG_SIZE)).astype("float32")
    train_y = rng.integers(0, C.NUM_CLASSES, n_tr)
    val_imgs = rng.normal(0.0, 1.0, size=(n_val, 3, C.IMG_SIZE, C.IMG_SIZE)).astype("float32")
    # OOD: shifted + scaled distribution -> should score far from the train Gaussians
    ood_imgs = (rng.normal(6.0, 3.0, size=(n_ood, 3, C.IMG_SIZE, C.IMG_SIZE))).astype("float32")

    train_ld = loader_from(train_imgs, train_y)
    val_ld = loader_from(val_imgs)
    ood_ld = loader_from(ood_imgs)

    # ---- fit_ood ----
    stats = fit_ood(model, train_ld)
    assert stats["means"].shape == (C.NUM_CLASSES, model.feature_dim)
    assert stats["precision"].shape == (model.feature_dim, model.feature_dim)
    assert np.isfinite(stats["means"]).all() and np.isfinite(stats["precision"]).all()
    print("fit_ood: means", stats["means"].shape, "precision", stats["precision"].shape, "OK")

    # ---- mahalanobis + energy on a batch (both forms: images and embeddings) ----
    xb = torch.as_tensor(ood_imgs[:8])
    m_img = mahalanobis_score(model, xb, stats)
    F_emb, _, lg = _embeddings(model, ood_ld, with_logits=True)
    m_emb = mahalanobis_score(model, F_emb, stats)
    e = energy_score(lg, T=1.0)
    assert np.isfinite(m_img).all() and np.isfinite(m_emb).all() and np.isfinite(e).all()
    print("mahalanobis(images) mean=%.3f  mahalanobis(emb) mean=%.3f  energy mean=%.3f"
          % (m_img.mean(), m_emb.mean(), e.mean()))

    # ---- ood_gate must split in (val/train-like) vs out (OOD) ----
    g_val = ood_gate(model, val_ld, stats, val_loader=val_ld, method="mahalanobis", keep_quantile=0.95)
    g_ood = ood_gate(model, ood_ld, stats, threshold=g_val["threshold"], method="mahalanobis")
    print("gate threshold=%.2f | val keep_rate=%.2f (n_in=%d) | ood keep_rate=%.2f (n_ood=%d)"
          % (g_val["threshold"], g_val["keep_rate"], g_val["n_in"], g_ood["keep_rate"], g_ood["n_ood"]))
    assert g_val["scores"].shape == (n_val,) and g_ood["scores"].shape == (n_ood,)
    # the OOD batch should be flagged OOD at least as often as the in-distribution val batch is
    assert g_ood["keep_rate"] <= g_val["keep_rate"] + 1e-9, "OOD batch not separated from val"
    assert g_ood["n_ood"] >= 1, "expected at least one OOD flag on the shifted batch"

    # energy gate also runs end-to-end
    ge = ood_gate(model, ood_ld, stats, val_loader=val_ld, method="energy", keep_quantile=0.95)
    assert np.isfinite(ge["scores"]).all()
    print("energy gate threshold=%.3f ood keep_rate=%.2f" % (ge["threshold"], ge["keep_rate"]))

    # ---- pseudo_label: constrained vs unconstrained ----
    # build a small external loader with synthetic paths + per-sample candidate sets
    ext_imgs = rng.normal(0.0, 1.0, size=(16, 3, C.IMG_SIZE, C.IMG_SIZE)).astype("float32")
    ext_ld = loader_from(ext_imgs)
    ext_paths = [f"ext_{i}.png" for i in range(16)]
    # alternate: half allowed normal-only, half allowed abnormal-only, a couple with no home (None)
    cand_list = []
    for i in range(16):
        if i % 5 == 0:
            cand_list.append(None)                       # high-grade-like: no morphology home
        elif i % 2 == 0:
            cand_list.append(L.NORMAL_CAND)
        else:
            cand_list.append(L.LOW_CAND)

    res = pseudo_label(model, ext_ld, stats, val_loader=val_ld, candidate_sets=cand_list,
                       conf_thresh=0.0, paths=ext_paths, ood_method="mahalanobis", keep_quantile=0.95)
    c = res["counts"]
    assert c["total"] == 16
    assert c["kept"] + c["drop_lowconf"] + c["drop_ood"] + c["drop_constraint"] + c["drop_no_candidate"] == 16
    # every kept pseudo-label must lie inside its allowed candidate set (the core safety guarantee)
    cand_by_path = dict(zip(ext_paths, cand_list))
    for p, lbl in res["kept"]:
        assert cand_by_path[p] is not None and lbl in cand_by_path[p], "kept label violates source constraint!"
    assert c["drop_no_candidate"] >= 1, "expected at least one no-candidate drop"
    assert res["constrained"] is True
    print("pseudo_label (constrained): counts=%s" % c)

    # unconstrained path emits a warning and keeps more (only conf+OOD gates)
    res2 = pseudo_label(model, ext_ld, stats, val_loader=val_ld, conf_thresh=0.0,
                        paths=ext_paths, ood_method="mahalanobis", keep_quantile=0.95)
    assert res2["constrained"] is False and "warning" in res2
    assert res2["counts"]["kept"] >= res["counts"]["kept"], "constraint should not increase kept count"
    print("pseudo_label (unconstrained): counts=%s warning-present=%s"
          % (res2["counts"], "warning" in res2))

    print("OK")
