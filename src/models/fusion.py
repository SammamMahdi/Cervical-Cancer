"""U7 multimodal fusion: early/joint image+tabular fusion (MULTIMODAL_PLAN options 2 & 3).

The late-fusion arm (stack tabular-XGBoost probabilities with the image model's 5-class
probabilities through a LogisticRegression meta-learner) already lives in
``explain.features.late_fusion``. This module adds the two *jointly trained* fusion architectures
so the U7 ablation is complete:

  * :class:`EarlyConcatFusion` (option 2 — early/feature concat). The image backbone's pooled
    embedding (``Classifier.features_vec`` -> ``feature_dim``) is projected to a compact image vector
    (default 256) and concatenated with a small MLP encoding (53 -> 32) of the *standardized*
    handcrafted morphometry features. A two-layer head maps the joined vector to the 5 SIPaKMeD
    logits.
  * :class:`DAFTFusion` (option 3 — joint FiLM/DAFT modulation, Poelsterl et al., MICCAI 2021). A
    tabular MLP predicts a per-channel affine transform ``(scale, shift)`` that modulates the CNN
    feature vector *before* the classification head — the tabular stream gates the image features
    rather than being concatenated to them.

Both consume the SAME handcrafted features as ``explain.features``:
``explain.features.load_features() -> {X[n,53], y, paths, names, split, mean, std}``. The tabular
rows are aligned to the image cells **by path** (a per-cell ``path -> row`` map) so each minibatch's
image tensor and tabular vector describe the same cell.

:func:`train_fusion` trains a chosen architecture on the leakage-safe grouped-split TRAIN cells
(image+tabular paired) with early stopping on VAL, then scores the 5-class problem on the SEALED
TEST split. It also reports the two unimodal baselines on the same TEST cells — *image-only* (the
fusion model run with its tabular stream zeroed) and *tabular-only* (an XGBoost model from
``explain.features.train_tabular``) — i.e. the U7 ablation contrast that complements the
late-fusion numbers. Results are written to ``results/metrics/fusion_<kind>.json``. Every external
dependency (handcrafted features on disk, image backbone weights, a CUDA device) is optional: the
trainer gates gracefully and records *why* it could not run instead of crashing.

The ``__main__`` path is a CPU-only smoke: it instantiates both modules on a RANDOM (untrained)
backbone and forwards a random ``(2,3,224,224)`` image batch + ``(2,53)`` tabular batch -> ``(2,5)``.
It never touches disk data, downloads weights, or runs real GPU training.
"""
import os
import json
import time

# --- robustness shim (import-order hazard, NOT a behavioural change) ---------------------------- #
# This package lives on PYTHONPATH at ``src/`` which also contains ``src/ssl.py`` (the SimCLR module
# named ``ssl``). That file shadows the stdlib ``ssl``; on this torch build ``import torch`` lazily
# triggers ``asyncio`` -> ``import ssl``, which would resolve to the shadow and re-enter torch mid-
# init (circular import crash) when this module is run as a script. Force the genuine stdlib ``ssl``
# into ``sys.modules`` first so asyncio finds the real one. No-op if ``ssl`` already resolves to the
# stdlib (the normal case when imported from a host that loaded ssl before adding src/ to the path).
import sys as _sys


def _ensure_stdlib_ssl():
    cur = _sys.modules.get("ssl")
    cur_file = (getattr(cur, "__file__", "") or "").replace("\\", "/")
    if cur is not None and "/src/ssl.py" not in cur_file:
        return  # already the genuine stdlib ssl
    import importlib.util as _ilu
    import sysconfig as _sc
    cand = os.path.join(_sc.get_paths()["stdlib"], "ssl.py")
    if os.path.isfile(cand):
        spec = _ilu.spec_from_file_location("ssl", cand)
        mod = _ilu.module_from_spec(spec)
        _sys.modules["ssl"] = mod
        spec.loader.exec_module(mod)


_ensure_stdlib_ssl()
# ----------------------------------------------------------------------------------------------- #

import numpy as np
import torch
import torch.nn as nn

import config as C
from models.backbones import build, count_params

# Tabular feature width produced by explain.features.load_features (nucleus 26 + cytoplasm 26 +
# nc_ratio). Kept as a module constant so the smoke + heads agree without importing the (heavy,
# disk-touching) feature loader.
N_TAB = 53

METRICS_PATH = os.path.join(C.METRICDIR, "fusion_{kind}.json")


# --------------------------------------------------------------------------- #
#  Architectures
# --------------------------------------------------------------------------- #
def _tab_mlp(in_dim, out_dim, hidden=64, dropout=0.2):
    """Small standardized-tabular encoder: Linear -> BN -> ReLU -> Dropout -> Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class EarlyConcatFusion(nn.Module):
    """Early/feature-level fusion (MULTIMODAL_PLAN option 2).

    image -> backbone.features_vec -> Linear(feature_dim, img_proj) -> ReLU         (image branch)
    tabular(53, standardized) -> MLP(53 -> tab_proj)                                 (tabular branch)
    concat([img_proj, tab_proj]) -> BN -> Linear -> ReLU -> Dropout -> Linear(5)     (fusion head)

    Args:
        backbone:    registry key for :func:`models.backbones.build`.
        num_classes: output logits (default ``C.NUM_CLASSES`` == 5).
        n_tab:       tabular feature width (53).
        img_proj:    projected image-embedding width (default 256).
        tab_proj:    tabular-embedding width (default 32).
        hidden:      fusion-head hidden width.
        dropout:     fusion-head dropout.
        pretrained:  load ImageNet backbone weights (False for the CPU smoke).
    """

    def __init__(self, backbone="efficientnet_b0", num_classes=C.NUM_CLASSES, n_tab=N_TAB,
                 img_proj=256, tab_proj=32, hidden=128, dropout=0.3, pretrained=True):
        super().__init__()
        self.kind = "early"
        # reuse the shared Classifier trunk + pooling; its .head is unused here (we build our own).
        self.encoder = build(backbone, num_classes=num_classes, hidden=hidden,
                             dropout=dropout, pretrained=pretrained)
        self.feature_dim = self.encoder.feature_dim
        self.n_tab = n_tab

        self.img_proj = nn.Sequential(
            nn.Linear(self.feature_dim, img_proj),
            nn.ReLU(inplace=True),
        )
        self.tab_enc = _tab_mlp(n_tab, tab_proj, hidden=max(32, tab_proj * 2), dropout=dropout)

        self.head = nn.Sequential(
            nn.BatchNorm1d(img_proj + tab_proj),
            nn.Linear(img_proj + tab_proj, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def set_backbone_trainable(self, flag):
        self.encoder.set_backbone_trainable(flag)
        return self

    def forward(self, image, tab):
        f = self.encoder.features_vec(image)          # (N, feature_dim)
        fi = self.img_proj(f)                          # (N, img_proj)
        ft = self.tab_enc(tab)                         # (N, tab_proj)
        return self.head(torch.cat([fi, ft], dim=1))   # (N, num_classes)


class DAFTFusion(nn.Module):
    """Joint FiLM/DAFT fusion (MULTIMODAL_PLAN option 3; Poelsterl et al., MICCAI 2021).

    A tabular MLP predicts a per-channel affine ``(scale, shift)`` over the CNN feature vector:

        f      = backbone.features_vec(image)                 # (N, feature_dim)
        s, b   = split( MLP(tab) -> 2*feature_dim )           # (N, feature_dim) each
        f'     = (1 + s) * f + b                              # DAFT modulation
        logits = head(f')                                     # (N, num_classes)

    The ``(1 + s)`` parameterization initializes the modulation near identity so early training is a
    smooth perturbation of the image-only model. With ``tab = 0`` and the tab-MLP's final layer at
    its default init the scale/shift are small -> image-dominated, which is exactly what the
    image-only baseline in :func:`train_fusion` exploits (it forwards zeroed tabular features).

    Args mirror :class:`EarlyConcatFusion` (``img_proj``/``tab_proj`` are unused; DAFT modulates the
    full ``feature_dim``).
    """

    def __init__(self, backbone="efficientnet_b0", num_classes=C.NUM_CLASSES, n_tab=N_TAB,
                 tab_hidden=64, hidden=128, dropout=0.3, pretrained=True):
        super().__init__()
        self.kind = "daft"
        self.encoder = build(backbone, num_classes=num_classes, hidden=hidden,
                             dropout=dropout, pretrained=pretrained)
        self.feature_dim = self.encoder.feature_dim
        self.n_tab = n_tab

        # tab -> (scale, shift) over every channel of the pooled feature vector
        self.film = nn.Sequential(
            nn.Linear(n_tab, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, 2 * self.feature_dim),
        )
        # start near identity: zero the final FiLM layer so scale=0, shift=0 -> f' == f
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def set_backbone_trainable(self, flag):
        self.encoder.set_backbone_trainable(flag)
        return self

    def forward(self, image, tab):
        f = self.encoder.features_vec(image)               # (N, feature_dim)
        scale, shift = self.film(tab).chunk(2, dim=1)      # (N, feature_dim) each
        f = (1.0 + scale) * f + shift                       # DAFT modulation
        return self.head(f)                                 # (N, num_classes)


def build_fusion(kind="early", backbone="efficientnet_b0", pretrained=True, **kw):
    """Factory: ``kind in {'early','daft'}`` -> the corresponding fusion module."""
    if kind == "early":
        return EarlyConcatFusion(backbone=backbone, pretrained=pretrained, **kw)
    if kind == "daft":
        return DAFTFusion(backbone=backbone, pretrained=pretrained, **kw)
    raise ValueError(f"unknown fusion kind {kind!r}; choose 'early' or 'daft'")


# --------------------------------------------------------------------------- #
#  Paired image+tabular dataset (aligned by path)
# --------------------------------------------------------------------------- #
class _PairedDataset(torch.utils.data.Dataset):
    """Yield ``(image_tensor, tab_vector, label)`` for one cell.

    ``paths``/``tab``/``labels`` are row-aligned (caller aligns tabular rows to image paths via the
    ``path -> row`` map from ``load_features``). The image transform is the shared eval/train
    transform from ``data.dataset.build_transform``.
    """
    def __init__(self, paths, tab, labels, transform):
        from PIL import Image
        self._Image = Image
        self.paths = list(paths)
        self.tab = np.asarray(tab, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self._Image.open(self.paths[i]).convert("RGB")
        x = self.transform(img)
        t = torch.from_numpy(self.tab[i])
        return x, t, int(self.labels[i])


# --------------------------------------------------------------------------- #
#  Train / evaluate
# --------------------------------------------------------------------------- #
def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=(C.USE_BF16 and C.DEVICE == "cuda"))


@torch.no_grad()
def _predict_fusion(model, loader, zero_tab=False):
    """Run a fusion model over ``loader`` -> (y_true[N], probs[N,5]).

    ``zero_tab=True`` zeroes the tabular input so the model runs image-only (the image-only baseline
    arm). Eval mode, BF16 autocast on CUDA, fp32 softmax (mirrors ``eval.metrics.predict``).
    """
    model.eval()
    device = next(model.parameters()).device
    ys, ps = [], []
    for x, t, y in loader:
        x = x.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        if zero_tab:
            t = torch.zeros_like(t)
        with _autocast():
            logits = model(x, t)
        probs = torch.softmax(logits.float(), dim=1)
        ps.append(probs.cpu().numpy())
        ys.append(np.asarray(y))
    if not ys:
        return np.zeros(0, np.int64), np.zeros((0, C.NUM_CLASSES), np.float64)
    return np.concatenate(ys).astype(np.int64), np.concatenate(ps).astype(np.float64)


def _scores(y, probs):
    """5-class accuracy + macro-F1 from probabilities (matches eval.metrics conventions)."""
    from sklearn.metrics import accuracy_score, f1_score
    if len(y) == 0:
        return dict(accuracy=float("nan"), macro_f1=float("nan"), n=0)
    pred = np.asarray(probs).argmax(1)
    labels = list(range(C.NUM_CLASSES))
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        n=int(len(y)),
    )


def train_fusion(kind="early", backbone="efficientnet_b0", epochs=30, lr=5e-4,
                 weight_decay=1e-4, batch_size=C.BATCH_SIZE, patience=8, aug="photometric",
                 split_kind="grouped", pretrained=True, num_workers=C.NUM_WORKERS,
                 save=True, smoke=False):
    """Train a joint image+tabular fusion model; report 5-class TEST acc/macro-F1 vs unimodal arms.

    Pipeline:
      1. ``explain.features.load_features`` -> standardized 53-D tabular X aligned to image paths.
      2. Build paired (image, tab, label) loaders for the grouped TRAIN / VAL / TEST splits.
      3. Train the fusion model (AdamW, label-smoothed CE) with early stopping on VAL accuracy.
      4. On the SEALED TEST split, score FUSED, IMAGE-ONLY (tab zeroed) and TABULAR-ONLY (XGBoost
         from ``explain.features.train_tabular``).

    Gates gracefully: if the handcrafted features cannot be loaded (missing ``.dat`` tables) the
    function returns a dict with ``status='gated'`` and the reason, writing nothing destructive.
    Returns the results dict; with ``save=True`` it is also written to
    ``results/metrics/fusion_<kind>.json``.
    """
    if kind not in ("early", "daft"):
        raise ValueError(f"unknown fusion kind {kind!r}; choose 'early' or 'daft'")

    from torch.utils.data import DataLoader

    out = {
        "kind": kind, "backbone": backbone, "split_kind": split_kind,
        "epochs": int(epochs), "lr": float(lr), "smoke": bool(smoke),
        "device": C.DEVICE,
    }

    # ---- 1. handcrafted features aligned to image paths (gated if absent) ----
    try:
        from explain.features import load_features, train_tabular
        feats = load_features(split_kind=split_kind, standardize=True)
    except Exception as e:  # missing .dat tables, etc. -> gate gracefully
        out["status"] = "gated"
        out["reason"] = f"handcrafted features unavailable: {type(e).__name__}: {e}"
        if save:
            _dump(kind, out)
        return out

    X, y_all, paths, names = feats["X"], feats["y"], feats["paths"], feats["names"]
    split = feats["split"]
    X = np.asarray(X, dtype=np.float32)
    n_tab = X.shape[1]

    tr_idx, va_idx, te_idx = split["train"], split["val"], split["test"]
    out["n_features"] = int(n_tab)
    out["split_sizes"] = {k: int(len(split[k])) for k in ("train", "val", "test")}

    if len(tr_idx) == 0 or len(te_idx) == 0:
        out["status"] = "gated"
        out["reason"] = "empty train or test split"
        if save:
            _dump(kind, out)
        return out

    # ---- 2. paired loaders ----
    from data.dataset import build_transform

    def _loader(idx, train):
        ds = _PairedDataset(
            [paths[i] for i in idx], X[idx], y_all[idx],
            build_transform(train=train, aug=aug),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=train,
                          num_workers=num_workers, pin_memory=False, drop_last=False)

    train_ld = _loader(tr_idx, True)
    val_ld = _loader(va_idx, False) if len(va_idx) else None
    test_ld = _loader(te_idx, False)

    # ---- 3. build + train the fusion model ----
    C.set_seed()
    model = build_fusion(kind, backbone=backbone, n_tab=n_tab, pretrained=pretrained).to(C.DEVICE)
    out["params_total"], out["params_trainable"] = (int(v) for v in count_params(model))

    label_smoothing = float(C.STUDENT.get("label_smoothing", 0.0))
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    n_epochs = 2 if smoke else int(epochs)
    history = []
    best = {"score": -1.0, "epoch": -1, "state": None}
    wait = 0
    for ep in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_corr, n = 0.0, 0, 0
        for x, t, yb in train_ld:
            x = x.to(C.DEVICE, non_blocking=True)
            t = t.to(C.DEVICE, non_blocking=True)
            yb = yb.to(C.DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast():
                logits = model(x, t)
                loss = crit(logits.float(), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_corr += (logits.argmax(1) == yb).sum().item()
            n += x.size(0)

        # validation accuracy (early-stopping monitor); fall back to train acc if no val
        if val_ld is not None:
            yv, pv = _predict_fusion(model, val_ld)
            val_acc = float((pv.argmax(1) == yv).mean()) if len(yv) else float("nan")
        else:
            val_acc = tr_corr / max(1, n)
        rec = dict(epoch=ep, loss=tr_loss / max(1, n), accuracy=tr_corr / max(1, n),
                   val_accuracy=val_acc, epoch_time=time.time() - t0)
        history.append(rec)

        if val_acc > best["score"]:
            best = {"score": val_acc, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out["history"] = history
    out["best_epoch"] = int(best["epoch"])
    out["best_val_accuracy"] = float(best["score"])

    # ---- 4. SEALED-TEST evaluation: fused vs image-only vs tabular-only ----
    yte, p_fused = _predict_fusion(model, test_ld, zero_tab=False)
    _, p_img = _predict_fusion(model, test_ld, zero_tab=True)
    out["fused"] = _scores(yte, p_fused)
    out["image_only"] = _scores(yte, p_img)

    # tabular-only: XGBoost on the SAME standardized features (gate if xgboost missing)
    try:
        trained = train_tabular(split_kind=split_kind, save=False)
        clf, Xs, split_t = trained["model"], trained["Xs"], trained["split"]
        te_t = split_t["test"]
        if len(te_t):
            p_tab = clf.predict_proba(Xs[te_t])
            out["tabular_only"] = _scores(trained["y"][te_t], p_tab)
        else:
            out["tabular_only"] = None
    except Exception as e:
        out["tabular_only"] = None
        out["tabular_only_note"] = f"{type(e).__name__}: {e}"

    out["status"] = "ok"
    # honest delta of the joint fusion over the image-only branch
    if np.isfinite(out["fused"]["macro_f1"]) and np.isfinite(out["image_only"]["macro_f1"]):
        out["delta_fused_vs_image"] = dict(
            accuracy=out["fused"]["accuracy"] - out["image_only"]["accuracy"],
            macro_f1=out["fused"]["macro_f1"] - out["image_only"]["macro_f1"],
        )

    if save:
        _dump(kind, out)
    return out


def _dump(kind, obj):
    path = METRICS_PATH.format(kind=kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
#  CPU smoke
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # CPU-only: instantiate both fusion heads on a RANDOM (untrained, no-download) backbone and
    # forward a random image+tabular batch -> (2,5). No disk data, no GPU, no real training.
    torch.manual_seed(C.SEED)
    img = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    tab = torch.randn(2, N_TAB)

    for kind in ("early", "daft"):
        model = build_fusion(kind, backbone="efficientnet_b0", n_tab=N_TAB,
                             pretrained=False).eval()
        with torch.no_grad():
            logits = model(img, tab)
            # image-only path (tabular zeroed) used by the train_fusion baseline arm
            logits_img = model(img, torch.zeros_like(tab))
        total, trainable = count_params(model)
        assert logits.shape == (2, C.NUM_CLASSES), f"{kind}: bad output {tuple(logits.shape)}"
        assert logits_img.shape == (2, C.NUM_CLASSES)
        print(f"{kind:5s}  out={tuple(logits.shape)}  img_only={tuple(logits_img.shape)}  "
              f"params={total/1e6:.2f}M (trainable {trainable/1e6:.2f}M)  "
              f"feat_dim={model.feature_dim}")
    print("OK")
