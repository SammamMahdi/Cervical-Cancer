"""ProtoPNet — case-based interpretable cervical-cytology classifier (U8, axis-3).

Faithful re-implementation of the *Prototypical Part Network* of Chen, Li, Tao,
Barnett, Su & Rudin, **"This Looks Like That: Deep Learning for Interpretable
Image Recognition"** (NeurIPS 2019), specialised to our 5-class NORMAL-first
SIPaKMeD head. ProtoPNet is an *interpretable-by-design* model: instead of a
black-box logit, every prediction is explained as a weighted vote of evidence of
the form *"this patch of the test cell looks like that learned prototypical
patch of class k"* — the prototypes are real training patches, so the reasoning
is inspectable and case-based rather than post-hoc (cf. Grad-CAM).

Architecture (``ProtoPNet``)
----------------------------
::

    backbone trunk  build(backbone).features        -> (N, feat, H, W)
        |                                               (conv feature map z)
    add-on conv     2x [1x1 Conv -> ReLU/Sigmoid]    -> (N, D, H, W)
        |                                               (prototype embedding space)
    prototype layer P prototypes p_j in R^{D,1,1}    -> L2 distance every (h,w)
        |                                               location to every p_j
        |           sim = log((d^2 + 1)/(d^2 + eps))  -> (N, P, H, W) similarities
    global max-pool over (H,W)                        -> (N, P)
        |           (the strongest match of each prototype anywhere in the cell)
    class layer     Linear(P -> 5, bias=False)       -> (N, 5) logits

``forward(x) -> (logits [N,5], min_distances [N,P])``.  The minimum squared
distance of each prototype (equivalently ``max similarity``) is returned because
the **cluster** and **separation** costs are defined on distances, not logits.

Prototype assignment: ``P = n_proto * NUM_CLASSES`` prototypes are partitioned
evenly across the 5 classes (default ``n_proto=10`` -> 50 prototypes, 10 per
class). ``prototype_class_identity[j, k] = 1`` iff prototype ``j`` belongs to
class ``k``. The class-connection layer is initialised to ``+1`` for the
prototype's own class and ``-0.5`` for the others (Chen et al. §3), encouraging
each prototype to be positive evidence for its class and (mild) negative
evidence elsewhere; the off-class weights are driven toward 0 by the L1
sparsity term in the final convex-optimisation stage.

Training schedule (summary — full GPU training is run elsewhere)
----------------------------------------------------------------
ProtoPNet is trained in repeating 3-stage cycles (Chen et al. §3, Algorithm):

  1. **Warm / joint (SGD)** — optimise trunk add-on + prototype vectors (and,
     after a warm-up, the trunk) with
     ``L = CE + lambda_clst * cluster + lambda_sep * separation``
     where ``cluster`` pulls each training image toward its *nearest own-class*
     prototype and ``separation`` pushes it *away from other-class* prototypes.
     Typical ``lambda_clst = 0.8``, ``lambda_sep = -0.08``.
  2. **Prototype push** (``push_prototypes``) — every few epochs, snap each
     prototype vector to the *nearest latent training patch of its own class*,
     so each prototype becomes an actual, visualisable training patch. This is
     the interpretability step and changes the embedding only by projection.
  3. **Last-layer convex optimisation** — with trunk + prototypes frozen, fine-
     tune *only* the class-connection layer with CE + an L1 penalty on the
     off-class weights (``set_last_layer_l1_mask``) to sparsify the explanation.

Accuracy is read off *after a push* (so the reported model is the interpretable
one). For our pipeline this lives behind axis U8 alongside the post-hoc methods.

Public API
----------
- ``ProtoPNet(backbone, n_proto, proto_dim, ...)`` — the model.
- ``model.forward(x) -> (logits, min_distances)``.
- ``model.push_forward(x) -> (conv_features, distances)`` — embeddings + the
  full ``(N,P,H,W)`` distance grid, used by ``push_prototypes``.
- ``cluster_separation_cost(min_distances, labels, model) -> (clst, sep)``.
- ``push_prototypes(model, loader, device=...)`` — the projection step.
- ``explain(model, x, k=3)`` — top-k activated prototypes per example (the
  "this looks like that" evidence), with class identity and similarity.

This module is FUNCTIONAL and importable; it constructs and runs on CPU. Real
GPU training (the multi-stage schedule above) is performed by the training
driver, not here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from models.backbones import build


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class ProtoPNet(nn.Module):
    """ProtoPNet over a configurable conv trunk (Chen et al., NeurIPS 2019).

    Parameters
    ----------
    backbone : str
        Registry key for the conv trunk (see ``models.backbones`` — e.g.
        ``"efficientnet_b0"``). Only ``build(backbone).features`` is used; the
        backbone's own pooling/head are discarded.
    num_classes : int
        Number of output classes (default ``C.NUM_CLASSES`` = 5).
    n_proto : int
        Prototypes **per class**; total ``P = n_proto * num_classes`` (default
        10 -> 50 prototypes for the 5-class head).
    proto_dim : int
        Channel dimension ``D`` of the prototype/embedding space (the add-on
        conv projects the trunk's ``feature_dim`` down to this; default 128).
    prototype_activation : {"log", "linear"}
        Distance->similarity map. ``"log"`` (default, Chen et al.) is
        ``log((d^2 + 1)/(d^2 + eps))``; ``"linear"`` is ``-d^2``.
    pretrained : bool
        Load ImageNet weights into the trunk (default True; set False for the
        CPU smoke to avoid a download).
    eps : float
        Numerical floor in the log activation.
    """

    def __init__(
        self,
        backbone="efficientnet_b0",
        num_classes=C.NUM_CLASSES,
        n_proto=10,
        proto_dim=128,
        prototype_activation="log",
        pretrained=True,
        eps=1e-4,
    ):
        super().__init__()
        if prototype_activation not in ("log", "linear"):
            raise ValueError(f"prototype_activation must be 'log'|'linear', got {prototype_activation!r}")
        self.backbone_name = backbone
        self.num_classes = int(num_classes)
        self.n_proto_per_class = int(n_proto)
        self.num_prototypes = self.n_proto_per_class * self.num_classes
        self.proto_dim = int(proto_dim)
        self.prototype_activation = prototype_activation
        self.eps = float(eps)

        # --- conv trunk (features only) ---
        trunk = build(backbone, num_classes=num_classes, pretrained=pretrained)
        self.features = trunk.features
        feat = trunk.feature_dim

        # --- add-on layers: 1x1 convs projecting feat -> D (Chen et al. use two
        #     1x1 convs; final activation is Sigmoid so embeddings live in (0,1)^D
        #     which bounds distances and stabilises the prototype push). ---
        self.add_on = nn.Sequential(
            nn.Conv2d(feat, self.proto_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proto_dim, self.proto_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # --- prototype layer: P learnable vectors of shape (P, D, 1, 1). ---
        self.prototype_shape = (self.num_prototypes, self.proto_dim, 1, 1)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))
        # ones kernel for the efficient ||z - p||^2 distance via convolution.
        self.register_buffer("_ones", torch.ones(self.prototype_shape))

        # --- prototype -> class identity (P x num_classes one-hot), buffer. ---
        ident = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            ident[j, j // self.n_proto_per_class] = 1.0
        self.register_buffer("prototype_class_identity", ident)

        # --- class-connection layer: Linear(P -> num_classes), no bias. ---
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        self._init_last_layer(pos=1.0, neg=-0.5)

    # ------------------------------------------------------------------ init
    def _init_last_layer(self, pos=1.0, neg=-0.5):
        """Init class weights: +pos for a prototype's own class, neg otherwise."""
        ident = self.prototype_class_identity.t()  # (num_classes, P)
        with torch.no_grad():
            self.last_layer.weight.copy_(pos * ident + neg * (1.0 - ident))

    # ------------------------------------------------------------- embedding
    def conv_features(self, x):
        """Trunk + add-on -> the prototype embedding map ``z`` of shape (N, D, H, W)."""
        return self.add_on(self.features(x))

    def _l2_distances(self, z):
        """Squared L2 distance from every spatial location of ``z`` to every prototype.

        Uses the identity ``||z - p||^2 = ||z||^2 - 2 z.p + ||p||^2`` evaluated by
        convolutions (Chen et al.'s ``_l2_convolution``) -> (N, P, H, W), clamped >= 0.
        """
        z2 = F.conv2d(z ** 2, self._ones)                       # (N,P,H,W) = sum z^2 over D
        p2 = (self.prototype_vectors ** 2).sum(dim=(1, 2, 3))   # (P,)
        zp = F.conv2d(z, self.prototype_vectors)                # (N,P,H,W) = sum z*p
        dist = z2 - 2 * zp + p2.view(1, -1, 1, 1)
        return F.relu(dist)

    def distance_to_similarity(self, dist):
        """Map squared distance -> prototype activation (similarity)."""
        if self.prototype_activation == "log":
            return torch.log((dist + 1.0) / (dist + self.eps))
        return -dist  # linear

    # --------------------------------------------------------------- forward
    def push_forward(self, x):
        """Return (conv_features ``z`` [N,D,H,W], distances [N,P,H,W]).

        The full distance grid is needed by ``push_prototypes`` to find, for
        every prototype, its nearest training patch.
        """
        z = self.conv_features(x)
        dist = self._l2_distances(z)
        return z, dist

    def forward(self, x):
        """ProtoPNet forward.

        Returns
        -------
        logits : (N, num_classes)
        min_distances : (N, P)
            The minimum squared distance of each prototype over all spatial
            locations (equivalently the location of its strongest activation);
            consumed by the cluster/separation costs.
        """
        dist = self._l2_distances(self.conv_features(x))        # (N,P,H,W)
        N, P = dist.shape[0], dist.shape[1]
        # global MAX similarity == global MIN distance over the (H,W) grid.
        min_distances = dist.view(N, P, -1).min(dim=2).values   # (N,P)
        similarities = self.distance_to_similarity(min_distances)  # (N,P)
        logits = self.last_layer(similarities)                  # (N,num_classes)
        return logits, min_distances

    # ----------------------------------------------------------- last-layer L1
    def set_last_layer_l1_mask(self):
        """Return the off-class mask for the L1 sparsity penalty on ``last_layer``.

        ``L1 = (last_layer.weight * mask).norm(1)`` penalises only connections
        from a prototype to classes it does NOT belong to (stage-3 sparsification).
        """
        return (1.0 - self.prototype_class_identity.t())        # (num_classes, P)

    def set_backbone_trainable(self, flag):
        """Freeze/unfreeze the conv trunk (the add-on + prototypes stay trainable);
        mirrors the warm-up vs joint stages of the schedule."""
        for p in self.features.parameters():
            p.requires_grad = bool(flag)
        return self


# --------------------------------------------------------------------------- #
#  Losses (cluster + separation cost — Chen et al. §3)
# --------------------------------------------------------------------------- #
def cluster_separation_cost(min_distances, labels, model):
    """Cluster and separation costs from per-prototype min distances.

    Parameters
    ----------
    min_distances : (N, P) tensor
        ``model.forward``'s second output.
    labels : (N,) int tensor
        Ground-truth class indices.
    model : ProtoPNet
        Provides ``prototype_class_identity`` (P x num_classes).

    Returns
    -------
    cluster_cost : scalar tensor
        Mean over the batch of the distance to the *nearest own-class*
        prototype (minimised -> each image sits near a prototype of its class).
    separation_cost : scalar tensor
        Mean over the batch of the distance to the *nearest other-class*
        prototype (maximised, i.e. used with a NEGATIVE coefficient -> images
        stay far from foreign prototypes).
    """
    ident = model.prototype_class_identity                      # (P, num_classes)
    # (N, P) boolean: is prototype j of the same class as example i?
    same = ident[:, labels].t().bool()                          # (N, P)
    BIG = torch.finfo(min_distances.dtype).max

    own = torch.where(same, min_distances, min_distances.new_full((), BIG))
    cluster_cost = own.min(dim=1).values.mean()

    other = torch.where(~same, min_distances, min_distances.new_full((), BIG))
    separation_cost = other.min(dim=1).values.mean()
    return cluster_cost, separation_cost


# --------------------------------------------------------------------------- #
#  Prototype push (the interpretability projection — Chen et al. §3)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def push_prototypes(model, loader, device=None, class_specific=True):
    """Project (``push``) each prototype onto its nearest TRAIN patch embedding.

    For every prototype ``p_j`` this scans the whole loader and finds the latent
    patch ``z[:, h, w]`` (over all images and spatial locations) with the
    smallest distance to ``p_j``; if ``class_specific`` (default), the search is
    restricted to images of ``p_j``'s own class. ``p_j`` is then replaced by that
    patch embedding, making the prototype an actual, visualisable training patch.

    Returns ``info``: a list of length ``P`` of dicts with ``dist`` (the achieved
    min distance), ``batch_label`` (the source image's label) and ``hw`` (the
    spatial location), so the caller can save the corresponding image crops.

    Run BETWEEN training stages with the trunk in eval mode. Modifies
    ``model.prototype_vectors`` in place.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    P, D = model.num_prototypes, model.proto_dim
    ident = model.prototype_class_identity                      # (P, num_classes)
    proto_of_class = ident.argmax(dim=1)                        # (P,)

    best_dist = np.full(P, np.inf, dtype=np.float64)
    best_vec = np.zeros((P, D), dtype=np.float32)
    best_meta = [dict(dist=float("inf"), batch_label=-1, hw=(-1, -1)) for _ in range(P)]

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = torch.as_tensor(y)
        z, dist = model.push_forward(x)                         # z:(N,D,H,W) dist:(N,P,H,W)
        N, _, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(N, H * W, D).cpu().numpy()   # (N,HW,D)
        dflat = dist.view(N, P, H * W).cpu().numpy()            # (N,P,HW)

        for j in range(P):
            if class_specific:
                mask = (y == int(proto_of_class[j])).numpy()
                if not mask.any():
                    continue
                rows = np.where(mask)[0]
            else:
                rows = np.arange(N)
            sub = dflat[rows, j, :]                             # (n_sel, HW)
            ai = int(np.argmin(sub))
            r, loc = divmod(ai, H * W)
            d = float(sub.reshape(-1)[ai])
            if d < best_dist[j]:
                best_dist[j] = d
                img_idx = int(rows[r])
                best_vec[j] = zf[img_idx, loc]
                best_meta[j] = dict(dist=d, batch_label=int(y[img_idx]), hw=(loc // W, loc % W))

    new_proto = torch.from_numpy(best_vec).view(P, D, 1, 1).to(model.prototype_vectors.device)
    model.prototype_vectors.data.copy_(new_proto)
    return best_meta


# --------------------------------------------------------------------------- #
#  Explanation ("this looks like that")
# --------------------------------------------------------------------------- #
@torch.no_grad()
def explain(model, x, k=3):
    """Top-``k`` activated prototypes per example — the case-based evidence.

    For each input image, return the ``k`` prototypes with the highest
    similarity (smallest distance). Each prototype is tied to a class via
    ``prototype_class_identity``, so the result reads as "this looks like that
    prototype of class c". Combined with the class-connection weight it gives
    the signed contribution of each piece of evidence to the prediction.

    Returns a dict with numpy arrays:
        pred            (N,)        argmax class of the logits
        proto_idx       (N, k)      indices of the top-k prototypes
        proto_class     (N, k)      class identity of those prototypes
        similarity      (N, k)      their activation (similarity)
        min_distance    (N, k)      their squared L2 distance
        contribution    (N, k)      similarity * last_layer weight to pred class
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    logits, min_dist = model(x)                                 # (N,C), (N,P)
    sim = model.distance_to_similarity(min_dist)               # (N,P)
    pred = logits.argmax(dim=1)                                 # (N,)

    topk = sim.topk(k, dim=1)
    proto_idx = topk.indices                                    # (N,k)
    proto_class = model.prototype_class_identity.argmax(dim=1)[proto_idx]  # (N,k)
    md = torch.gather(min_dist, 1, proto_idx)                   # (N,k)
    # signed contribution of each top prototype toward the PREDICTED class.
    w = model.last_layer.weight                                 # (C,P)
    w_pred = w[pred]                                            # (N,P)
    contrib = topk.values * torch.gather(w_pred, 1, proto_idx)  # (N,k)

    return dict(
        pred=pred.cpu().numpy(),
        proto_idx=proto_idx.cpu().numpy(),
        proto_class=proto_class.cpu().numpy(),
        similarity=topk.values.cpu().numpy(),
        min_distance=md.cpu().numpy(),
        contribution=contrib.cpu().numpy(),
    )


# --------------------------------------------------------------------------- #
#  CPU smoke
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(C.SEED)
    model = ProtoPNet("efficientnet_b0", n_proto=10, pretrained=False).eval()
    P = model.num_prototypes
    print(f"ProtoPNet(efficientnet_b0): P={P} (n_proto/class=10 x {model.num_classes}), "
          f"D={model.proto_dim}")

    x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
    logits, min_dist = model(x)
    assert logits.shape == (2, C.NUM_CLASSES), f"bad logits {tuple(logits.shape)}"
    assert min_dist.shape == (2, P), f"bad min_distances {tuple(min_dist.shape)}"
    print(f"forward: logits={tuple(logits.shape)} min_distances={tuple(min_dist.shape)}")

    # one cluster/separation loss computation on random labels.
    labels = torch.tensor([0, 4])
    clst, sep = cluster_separation_cost(min_dist, labels, model)
    print(f"cluster_cost={clst.item():.4f}  separation_cost={sep.item():.4f}")

    # L1 mask shape + explanation.
    mask = model.set_last_layer_l1_mask()
    assert mask.shape == (C.NUM_CLASSES, P)
    ex = explain(model, x, k=3)
    assert ex["proto_idx"].shape == (2, 3)
    print(f"explain: top-3 proto_idx[0]={ex['proto_idx'][0].tolist()} "
          f"proto_class[0]={ex['proto_class'][0].tolist()} pred={ex['pred'].tolist()}")

    # tiny push smoke on a 2-batch fake loader (exercises push_prototypes end-to-end).
    fake = [(torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE), torch.tensor([0, 1])),
            (torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE), torch.tensor([2, 3]))]
    info = push_prototypes(model, fake, device=torch.device("cpu"))
    assert len(info) == P
    finite = sum(np.isfinite(m["dist"]) for m in info)
    print(f"push_prototypes: snapped {finite}/{P} prototypes (others have no own-class patch in fake loader)")
    print("OK")
