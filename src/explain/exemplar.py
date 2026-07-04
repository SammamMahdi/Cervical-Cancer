"""Axis-3 case-based ("this looks like that") explanations for the cytology student.

Dependency-free interpretability built entirely on the model's pooled embedding
(``model.features_vec`` -> (N, feat)). Four complementary, post-hoc tools:

* :func:`build_index`        embed the TRAIN cells once -> (embeddings, labels, paths) +
                              an in-process cache (and a small ``.npz`` on disk).
* :func:`nearest_exemplars`  cosine k-NN over the train embeddings: surface the k most
                              similar training cells for a query ("this looks like that"),
                              and save a query+neighbours panel.
* :func:`class_prototypes` /
  :func:`prototype_similarity`  per-class mean embedding and the query's cosine similarity
                              to each prototype — a lightweight, ProtoPNet-flavoured view
                              (NOT full ProtoPNet; see the docstring note).
* :func:`counterfactual`     gradient ascent on a *copy* of the input, with an L2 proximity
                              penalty, to flip the prediction to a target class; returns the
                              perturbed input + delta + final probabilities and saves a
                              before / after / |delta| panel.

All routines accept ANY model exposing the SHARED CONTRACT API
(``features_vec``/``forward``); they never assume the trained checkpoint exists. Figures use
the house plotting style and are written under ``results/figures``.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from data.dataset import build_transform
from data.splits import get_split

# ----------------------------------------------------------------------------- paths / cache
INDEX_PATH = os.path.join(C.METRICDIR, "exemplar_index.npz")
_INDEX_CACHE = None          # in-process cache: dict(embeddings, labels, paths, model_id)
_PROTO_CACHE = {}            # model_id -> (protos (K,feat), present_mask (K,))


def _model_id(model):
    """A cheap identity key so the cache invalidates when a different model is passed."""
    return id(model)


def _device_of(model):
    try:
        return next(model.parameters()).device
    except StopIteration:                     # pragma: no cover - model with no params
        return torch.device("cpu")


def _class_of(path):
    """Class index from the path substring ``im_<ClassName>`` (SHARED CONTRACT)."""
    for ci, cname in enumerate(C.CLASSES):
        if f"im_{cname}" in path:
            return ci
    raise ValueError(f"cannot determine class for {path}")


def _load_tensor(path, device):
    """Read one image -> normalized eval tensor (1, 3, H, W) on ``device``."""
    tf = build_transform(train=False)
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def _l2norm(emb, eps=1e-8):
    """Row-wise L2 normalize a (N, feat) tensor; rows with norm < eps map to exactly 0.

    More robust than ``F.normalize`` for degenerate (near-zero) embeddings, whose norm can
    straddle ``F.normalize``'s default eps and yield a non-unit, non-zero row. A clean zero
    row gives a well-defined cosine of 0 against everything.
    """
    norm = emb.norm(dim=1, keepdim=True)
    safe = norm.clamp_min(eps)
    out = emb / safe
    out = torch.where(norm < eps, torch.zeros_like(out), out)
    return out


def _denorm(t):
    """Undo ImageNet normalization on a (3,H,W) tensor -> displayable (H,W,3) in [0,1]."""
    mean = torch.tensor(C.IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD, device=t.device).view(3, 1, 1)
    img = (t.detach() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


# ----------------------------------------------------------------------------- index
@torch.no_grad()
def _embed_paths(model, paths, batch_size=32):
    """Embed a list of image paths -> (N, feat) float32 numpy (L2-normalized rows)."""
    device = _device_of(model)
    model.eval()
    tf = build_transform(train=False)
    vecs = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        emb = model.features_vec(batch).float()
        emb = _l2norm(emb)                    # cosine == dot product on normalized rows
        vecs.append(emb.cpu().numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)


def build_index(model, paths=None, batch_size=32, use_disk_cache=True):
    """Embed the TRAIN cells once and cache (embeddings, labels, paths).

    Args:
        model:          any model with ``features_vec`` (random or trained — graceful either way).
        paths:          image paths to index; default = ``get_split('grouped')['train']``.
        batch_size:     embedding batch size.
        use_disk_cache: if True and ``paths`` is None, load/save an ``.npz`` under results/metrics
                        keyed by the path list so repeated default calls are fast. A disk cache is
                        only trusted when the path list matches exactly (the embeddings still depend
                        on the model weights, so the in-process cache is keyed by model identity).

    Returns:
        dict(embeddings=(N,feat) float32 L2-normalized, labels=(N,) int64, paths=list[str]).
    """
    global _INDEX_CACHE
    default = paths is None
    if default:
        paths = get_split("grouped")["train"]
    paths = list(paths)

    mid = _model_id(model)
    if _INDEX_CACHE is not None and _INDEX_CACHE["model_id"] == mid and _INDEX_CACHE["paths"] == paths:
        return {k: _INDEX_CACHE[k] for k in ("embeddings", "labels", "paths")}

    embeddings = labels = None
    if default and use_disk_cache and os.path.exists(INDEX_PATH):
        try:
            z = np.load(INDEX_PATH, allow_pickle=True)
            cached_paths = list(z["paths"])
            if cached_paths == paths and int(z["model_id"]) == mid:
                embeddings = z["embeddings"].astype(np.float32)
                labels = z["labels"].astype(np.int64)
        except Exception:
            embeddings = labels = None

    if embeddings is None:
        embeddings = _embed_paths(model, paths, batch_size=batch_size)
        labels = np.array([_class_of(p) for p in paths], dtype=np.int64)
        if default and use_disk_cache:
            os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
            np.savez(INDEX_PATH, embeddings=embeddings, labels=labels,
                     paths=np.array(paths, dtype=object), model_id=np.int64(mid))

    _INDEX_CACHE = {"embeddings": embeddings, "labels": labels, "paths": paths, "model_id": mid}
    return {"embeddings": embeddings, "labels": labels, "paths": paths}


# ----------------------------------------------------------------------------- nearest exemplars
def nearest_exemplars(model, query_path, k=5, index=None, save=True, out_dir=None):
    """Find the k most similar TRAIN cells to ``query_path`` (cosine sim on embeddings).

    "This looks like that": returns the neighbours and, by default, saves a panel
    ``results/figures/exemplar_<stem>.png`` (query + k neighbours, titled with class names
    and cosine similarity).

    Returns:
        dict(query_path, query_class, neighbors=list of dict(path, class_idx, class_name, sim),
             fig_path=str|None).
    """
    idx = index if index is not None else build_index(model)
    emb, labels, paths = idx["embeddings"], idx["labels"], idx["paths"]

    device = _device_of(model)
    model.eval()
    with torch.no_grad():
        q = model.features_vec(_load_tensor(query_path, device)).float()
        q = _l2norm(q).cpu().numpy()[0]                            # (feat,)

    sims = emb @ q                                                 # (N,) cosine, rows are L2-normalized
    # exclude the query itself if it is present in the index (same path)
    order = np.argsort(-sims)
    neighbors = []
    for j in order:
        if paths[j] == query_path:
            continue
        neighbors.append({
            "path": paths[j],
            "class_idx": int(labels[j]),
            "class_name": C.CLASSES[int(labels[j])],
            "sim": float(sims[j]),
        })
        if len(neighbors) >= k:
            break

    q_cls = _class_of(query_path)
    result = {
        "query_path": query_path,
        "query_class": C.CLASSES[q_cls],
        "neighbors": neighbors,
        "fig_path": None,
    }
    if save:
        result["fig_path"] = _save_exemplar_panel(query_path, q_cls, neighbors, out_dir=out_dir)
    return result


def _save_exemplar_panel(query_path, q_cls, neighbors, out_dir=None):
    import plotting as P                                           # applies house style globally
    import matplotlib.pyplot as plt

    tf = build_transform(train=False)
    n = len(neighbors)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.4 * (n + 1), 2.9))
    axes = np.atleast_1d(axes)

    qimg = _denorm(tf(Image.open(query_path).convert("RGB")))
    axes[0].imshow(qimg)
    axes[0].set_title(f"query\n{P.pretty(C.CLASSES[q_cls])}", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_edgecolor(P.color("oracle")); spine.set_linewidth(2.5)

    for ax, nb in zip(axes[1:], neighbors):
        img = _denorm(tf(Image.open(nb["path"]).convert("RGB")))
        ax.imshow(img)
        match = "(hit)" if nb["class_idx"] == q_cls else "(miss)"
        ax.set_title(f"{P.pretty(nb['class_name'])} {match}\nsim={nb['sim']:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        col = P.color(nb["class_name"], nb["class_idx"])
        for spine in ax.spines.values():
            spine.set_edgecolor(col); spine.set_linewidth(2.0)

    fig.suptitle("Nearest training exemplars (cosine)", fontweight="bold", y=1.04)
    out_dir = out_dir or C.FIGDIR
    stem = os.path.splitext(os.path.basename(query_path))[0]
    return P.save(fig, os.path.join(out_dir, f"exemplar_{stem}"))


# ----------------------------------------------------------------------------- prototypes
def class_prototypes(model, index=None):
    """Per-class mean (L2-normalized) embedding -> (K, feat) prototypes + a present-mask.

    Lightweight, ProtoPNet-*flavoured*: prototypes are class-mean train embeddings in the
    backbone's feature space. NOTE: this is NOT full ProtoPNet — genuine ProtoPNet learns a
    bank of part prototypes with a dedicated training regime (a prototype layer + push step),
    which is out of scope for this post-hoc, training-free module.

    Returns:
        dict(prototypes=(K,feat) float32 L2-normalized, present=(K,) bool mask of classes seen,
             classes=C.CLASSES).
    """
    global _PROTO_CACHE
    idx = index if index is not None else build_index(model)
    mid = _model_id(model)
    if mid in _PROTO_CACHE and _PROTO_CACHE[mid][2] == idx["paths"]:
        protos, present, _ = _PROTO_CACHE[mid]
        return {"prototypes": protos, "present": present, "classes": C.CLASSES}

    emb, labels = idx["embeddings"], idx["labels"]
    feat = emb.shape[1]
    protos = np.zeros((C.NUM_CLASSES, feat), dtype=np.float32)
    present = np.zeros(C.NUM_CLASSES, dtype=bool)
    for ci in range(C.NUM_CLASSES):
        mask = labels == ci
        if mask.any():
            m = emb[mask].mean(axis=0)
            norm = np.linalg.norm(m)
            protos[ci] = m / norm if norm > 0 else m
            present[ci] = True
    _PROTO_CACHE[mid] = (protos, present, list(idx["paths"]))
    return {"prototypes": protos, "present": present, "classes": C.CLASSES}


def prototype_similarity(model, query, index=None):
    """Cosine similarity of a query to each class prototype (lightweight ProtoPNet-style score).

    Args:
        query: an image path (str) or a normalized input tensor (1,3,H,W) / (3,H,W).

    Returns:
        dict(sims=(K,) float cosine sims (NaN for absent classes), classes=C.CLASSES,
             pred_class=index of the most similar present class, pred_name=str).
    """
    protos_d = class_prototypes(model, index=index)
    protos, present = protos_d["prototypes"], protos_d["present"]

    device = _device_of(model)
    model.eval()
    with torch.no_grad():
        if isinstance(query, str):
            x = _load_tensor(query, device)
        else:
            x = query if query.dim() == 4 else query.unsqueeze(0)
            x = x.to(device)
        q = model.features_vec(x).float()
        q = _l2norm(q).cpu().numpy()[0]

    sims = protos @ q                                              # (K,) cosine (rows L2-normalized)
    sims = np.where(present, sims, np.nan)
    valid = np.where(present)[0]
    pred = int(valid[np.argmax(sims[valid])]) if valid.size else -1
    return {
        "sims": sims.astype(np.float32),
        "classes": C.CLASSES,
        "pred_class": pred,
        "pred_name": C.CLASSES[pred] if pred >= 0 else None,
    }


# ----------------------------------------------------------------------------- counterfactual
def counterfactual(model, x, target_class, steps=200, lr=0.02, l2=0.1, save=True,
                   out_dir=None, tag=None):
    """Gradient-ascent counterfactual: minimally perturb ``x`` to be classified ``target_class``.

    Optimizes a clone of the input (grads flow to the *input*, NOT the weights) to maximize the
    target-class log-probability while an L2 proximity penalty keeps the perturbation small:

        maximize  log p(target | x')  -  l2 * ||x' - x||^2

    Runs in float32 with grads enabled (no autocast / no_grad), per the SHARED CONTRACT for
    gradient-based explanations. Saves a before / after / |delta| panel by default.

    The model's train/eval mode is left as the caller set it (NOT forced): a TRAINED student
    should be in ``eval()`` (deterministic BN/dropout) and its eval-mode input gradients are
    well-behaved. A freshly RANDOM deep trunk has vanishing eval-mode input gradients, so the
    ``__main__`` smoke test primes the model in ``train()`` mode to exercise real gradient flow.

    Args:
        x:            input tensor (N,3,H,W) or (3,H,W), ImageNet-normalized (typically N=1).
        target_class: desired class index in [0, NUM_CLASSES).
        steps, lr, l2: optimization budget, step size, proximity weight.

    Returns:
        dict(x_cf=(N,3,H,W) perturbed tensor (detached, on CPU), delta=(N,3,H,W) = x_cf - x,
             probs_before=(K,), probs_after=(K,) for the FIRST sample, orig_class, target_class,
             success=bool (first sample), fig_path=str|None).
    """
    if not (0 <= target_class < C.NUM_CLASSES):
        raise ValueError(f"target_class {target_class} out of range [0,{C.NUM_CLASSES})")

    device = _device_of(model)
    x0 = (x if x.dim() == 4 else x.unsqueeze(0)).detach().to(device).float()
    n = x0.shape[0]

    with torch.no_grad():
        probs_before = F.softmax(model(x0), dim=1)[0]
        orig_class = int(probs_before.argmax())

    x_cf = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x_cf], lr=lr)
    tgt = torch.full((n,), target_class, dtype=torch.long, device=device)

    for _ in range(steps):
        opt.zero_grad()
        logits = model(x_cf)
        ce = F.cross_entropy(logits, tgt)                         # = -log p(target) (drives ascent)
        prox = l2 * ((x_cf - x0) ** 2).mean()
        loss = ce + prox
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs_after = F.softmax(model(x_cf), dim=1)[0]
    delta = (x_cf - x0).detach()
    success = int(probs_after.argmax()) == target_class

    result = {
        "x_cf": x_cf.detach().cpu(),
        "delta": delta.cpu(),
        "probs_before": probs_before.detach().cpu().numpy(),
        "probs_after": probs_after.detach().cpu().numpy(),
        "orig_class": orig_class,
        "target_class": target_class,
        "success": bool(success),
        "fig_path": None,
    }
    if save:
        result["fig_path"] = _save_counterfactual_panel(x0[0], x_cf.detach()[0], delta[0],
                                                         orig_class, target_class,
                                                         probs_before, probs_after,
                                                         out_dir=out_dir, tag=tag)
    return result


def _save_counterfactual_panel(x0, x_cf, delta, orig_class, target_class,
                               probs_before, probs_after, out_dir=None, tag=None):
    import plotting as P
    import matplotlib.pyplot as plt

    before = _denorm(x0)
    after = _denorm(x_cf)
    # |delta| magnitude over channels, scaled to [0,1] for display
    dmag = delta.abs().sum(0)
    dmag = (dmag / (dmag.max() + 1e-8)).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.1))
    p_o = float(probs_before[orig_class]); p_t0 = float(probs_before[target_class])
    p_t1 = float(probs_after[target_class])
    axes[0].imshow(before)
    axes[0].set_title(f"before\n{P.pretty(C.CLASSES[orig_class])} p={p_o:.2f}", fontsize=10)
    axes[1].imshow(after)
    axes[1].set_title(f"after -> {P.pretty(C.CLASSES[target_class])}\np={p_t1:.2f} (was {p_t0:.2f})",
                      fontsize=10)
    im = axes[2].imshow(dmag, cmap="magma")
    axes[2].set_title("|delta| (perturbation)", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Counterfactual (gradient ascent on input)", fontweight="bold", y=1.05)

    out_dir = out_dir or C.FIGDIR
    name = f"counterfactual_{tag}" if tag else \
        f"counterfactual_{C.CLASSES[orig_class]}_to_{C.CLASSES[target_class]}"
    return P.save(fig, os.path.join(out_dir, name))


# ----------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    from models.backbones import build

    C.set_seed()
    print("[exemplar] building RANDOM model on CPU...")
    model = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"],
                  C.STUDENT["dropout"], pretrained=False)

    split = get_split("grouped")
    train_paths = split["train"][:20]
    test_path = split["test"][0]

    # A *freshly random* EfficientNet trunk collapses these inputs to ~0 in eval mode (BN running
    # stats are still the defaults), giving degenerate embeddings and no input gradient. Prime the
    # BatchNorm running statistics with a couple of train-mode passes so this smoke test exercises a
    # model with real gradient flow (a TRAINED student is non-degenerate by construction). This is a
    # smoke-test convenience only; the public API never mutates the caller's model.
    print("[exemplar] priming BatchNorm stats on real images (smoke-test only)...")
    model.train()
    with torch.no_grad():
        warm = torch.stack([_load_tensor(p, torch.device("cpu"))[0] for p in train_paths[:8]])
        for _ in range(3):
            model(warm)
    model.eval()

    print(f"[exemplar] build_index on {len(train_paths)} train paths...")
    idx = build_index(model, paths=train_paths, use_disk_cache=False)
    N, feat = idx["embeddings"].shape
    assert N == len(train_paths), f"index size {N} != {len(train_paths)}"
    assert idx["labels"].shape == (N,), f"bad labels shape {idx['labels'].shape}"
    # F.normalize leaves each row with unit norm, or ~0 for a degenerate (all-zero) embedding
    # (a random untrained trunk can pool to ~0 — the trained model has non-degenerate features).
    norms = np.linalg.norm(idx["embeddings"], axis=1)
    assert np.all((np.abs(norms - 1.0) < 1e-3) | (norms < 1e-3)), \
        f"embeddings rows neither unit-norm nor zero: {norms[:5]}"
    print(f"   embeddings={idx['embeddings'].shape} labels={idx['labels'].shape} "
          f"(unit-norm rows={int((np.abs(norms-1.0)<1e-3).sum())}/{N})")

    print("[exemplar] nearest_exemplars(k=3)...")
    ne = nearest_exemplars(model, test_path, k=3, index=idx)
    assert len(ne["neighbors"]) == 3, f"expected 3 neighbours, got {len(ne['neighbors'])}"
    assert all("sim" in n for n in ne["neighbors"])
    print(f"   query={ne['query_class']} fig={ne['fig_path']}")
    for nb in ne["neighbors"]:
        print(f"     -> {nb['class_name']:24s} sim={nb['sim']:.3f}")

    print("[exemplar] prototype_similarity...")
    ps = prototype_similarity(model, test_path, index=idx)
    assert ps["sims"].shape == (C.NUM_CLASSES,), f"bad proto sims {ps['sims'].shape}"
    print(f"   pred_proto={ps['pred_name']} sims={np.round(ps['sims'], 3)}")

    print("[exemplar] counterfactual (few steps)...")
    # A deep RANDOM trunk has vanishing input gradients in eval mode, so for the smoke test we put
    # the model in train() mode (BN uses batch stats -> real gradient flow) and pass a 2-image batch.
    # A TRAINED student is used in eval() mode in production, where its input gradients are healthy.
    model.train()
    x = torch.stack([_load_tensor(p, torch.device("cpu"))[0] for p in (test_path, train_paths[0])])
    orig = int(F.softmax(model(x), dim=1)[0].argmax())
    target = (orig + 1) % C.NUM_CLASSES
    cf = counterfactual(model, x, target_class=target, steps=15, lr=0.05, l2=0.05, tag="smoke")
    model.eval()
    assert cf["x_cf"].shape == x.shape, f"cf shape {cf['x_cf'].shape} != {tuple(x.shape)}"
    moved = float((cf["x_cf"] - x.cpu()).abs().max())
    assert moved > 1e-6, "counterfactual did not change the input"
    assert cf["probs_after"].shape == (C.NUM_CLASSES,)
    assert cf["delta"].shape == x.shape
    print(f"   orig={C.CLASSES[orig]} target={C.CLASSES[target]} "
          f"p_target {cf['probs_before'][target]:.3f}->{cf['probs_after'][target]:.3f} "
          f"max|delta|={moved:.4f} success={cf['success']} fig={cf['fig_path']}")

    print("OK")
