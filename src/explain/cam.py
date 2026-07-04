"""Class-Activation-Map FAMILY for the cytology classifier (dependency-free).

Implements six CAM methods behind one uniform interface using only torch forward/backward hooks
(no pytorch-grad-cam, no cv2). The JET overlay is produced from matplotlib's colormap, so the only
runtime dependencies are torch + numpy + matplotlib (already required by the pipeline).

Methods
-------
    gradcam     Grad-CAM (Selvaraju 2017): GAP(grad) . act, ReLU.
    gradcampp   Grad-CAM++ (Chattopadhyay 2018): higher-order pixel weights.
    hirescam    HiRes-CAM (Draelos 2020): element-wise grad . act, channel-sum (faithful gradients).
    layercam    LayerCAM (Jiang 2021): ReLU(grad) . act, channel-sum (spatially sharp).
    eigencam    Eigen-CAM (Muhammad 2020): 1st principal component of the activation map (grad-free).
    scorecam    Score-CAM (Wang 2020): forward-masking importance weights (grad-free, slow).

Public API
----------
    CAM(model, target_layer=None, method='gradcam')         one class, ``method=`` selects the rule
    GradCAM / GradCAMpp / HiResCAM / LayerCAM / EigenCAM / ScoreCAM   thin subclasses
        cam = obj(x, class_idx=None)  -> (H, W) float32 in [0, 1]   (H,W = input spatial size)
    overlay(img01, cam)  -> (H, W, 3) RGB in [0, 1]   (JET heatmap blended over the cell)
    run_cam(model, which=..., methods=..., n_per_class=..., save=True)
        -> {method: saved_png_path}   house-style panels results/figures/cam_<method>.png

Gradient methods need grads -> the model is run in float32 with grad enabled (NO no_grad/autocast).
A trained checkpoint (results/checkpoints/<which>.pt, a raw state_dict) is loaded when present; if it
is absent the caller's model is used as-is, so the module degrades gracefully on an untrained net.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import config as C


# ---------------------------------------------------------------------------
# core: one class, hook-based, method-selectable
# ---------------------------------------------------------------------------
_METHODS = ("gradcam", "gradcampp", "hirescam", "layercam", "eigencam", "scorecam")
_GRAD_FREE = ("eigencam", "scorecam")


class CAM:
    """Class-activation map over ``target_layer`` (default ``model.last_conv_module``).

    Captures the target layer's forward activations (and, for gradient methods, the gradient of the
    target-class logit w.r.t. those activations) via hooks, combines them per ``method`` into a
    coarse spatial map, ReLU-clips, up-samples to the input resolution and min-max normalises to
    [0, 1]. ``__call__`` accepts a single image tensor ``x`` of shape (1, 3, H, W) or (3, H, W).
    """

    def __init__(self, model, target_layer=None, method="gradcam"):
        method = method.lower()
        if method not in _METHODS:
            raise ValueError(f"unknown CAM method {method!r}; choose from {_METHODS}")
        self.model = model
        self.method = method
        self.target_layer = target_layer if target_layer is not None else model.last_conv_module
        self._acts = None
        self._grads = None
        self._fh = self.target_layer.register_forward_hook(self._fwd_hook)
        # full backward hook gives grad w.r.t. the module OUTPUT (the activations we saved).
        self._bh = self.target_layer.register_full_backward_hook(self._bwd_hook)

    # -- hooks --
    def _fwd_hook(self, module, inp, out):
        self._acts = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def remove(self):
        """Detach the forward/backward hooks (call when the CAM object is no longer needed)."""
        try:
            self._fh.remove()
            self._bh.remove()
        except Exception:
            pass

    def __del__(self):
        self.remove()

    # -- main entry --
    def __call__(self, x, class_idx=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise ValueError("CAM expects a single image (batch size 1)")
        dev = next(self.model.parameters()).device
        x = x.to(dev).float()                         # gradient methods require float32
        H, W = x.shape[-2:]

        if self.method in _GRAD_FREE:
            cam = self._grad_free(x, class_idx)
        else:
            cam = self._grad_based(x, class_idx)

        cam = torch.relu(cam)                          # keep evidence FOR the class
        cam = cam.view(1, 1, *cam.shape[-2:])
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy().astype(np.float32)

    # -- gradient-based rules (gradcam / gradcampp / hirescam / layercam) --
    def _grad_based(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                          # forward hook fills self._acts
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        score.backward()                                # backward hook fills self._grads

        acts = self._acts.detach()[0]                   # (Cc, h, w)
        grads = self._grads[0]                          # (Cc, h, w)

        if self.method == "gradcam":
            w = grads.mean(dim=(1, 2), keepdim=True)            # GAP of gradients
            cam = (w * acts).sum(0)
        elif self.method == "hirescam":
            cam = (grads * acts).sum(0)                         # element-wise (faithful)
        elif self.method == "layercam":
            w = torch.relu(grads)                               # positive gradients only
            cam = (w * acts).sum(0)
        elif self.method == "gradcampp":
            g2 = grads ** 2
            g3 = g2 * grads
            denom = 2.0 * g2 + (acts * g3).sum(dim=(1, 2), keepdim=True)
            alpha = g2 / (denom + 1e-8)
            w = (alpha * torch.relu(grads)).sum(dim=(1, 2), keepdim=True)
            cam = (w * acts).sum(0)
        else:  # pragma: no cover
            raise ValueError(self.method)
        return cam

    # -- gradient-free rules (eigencam / scorecam) --
    def _grad_free(self, x, class_idx):
        if self.method == "eigencam":
            with torch.no_grad():
                self.model(x)                            # forward hook fills self._acts
            acts = self._acts.detach()[0]                # (Cc, h, w)
            Cc, h, w = acts.shape
            A = acts.reshape(Cc, h * w)                  # (Cc, hw)
            A = A - A.mean(dim=1, keepdim=True)
            # principal direction over channels -> projection map (1st right singular vector).
            try:
                _, _, Vh = torch.linalg.svd(A, full_matrices=False)
                cam = Vh[0].reshape(h, w)
            except Exception:                            # rare LAPACK fallback
                cam = A.abs().sum(0).reshape(h, w)
            # sign convention: more activation -> larger CAM value.
            if cam.sum() < 0:
                cam = -cam
            return cam

        # score-cam: per-channel forward masking (no gradients).
        with torch.no_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            acts = self._acts.detach()[0]                # (Cc, h, w)
            Cc, h, w = acts.shape
            H, W = x.shape[-2:]
            up = F.interpolate(acts.unsqueeze(0), size=(H, W), mode="bilinear",
                               align_corners=False)[0]   # (Cc, H, W)
            amin = up.amin(dim=(1, 2), keepdim=True)
            amax = up.amax(dim=(1, 2), keepdim=True)
            masks = (up - amin) / (amax - amin + 1e-8)   # (Cc, H, W) in [0,1]
            weights = torch.zeros(Cc, device=x.device)
            base = float(torch.softmax(logits, 1)[0, class_idx])
            bs = 64                                       # batch the Cc masked forwards
            for s in range(0, Cc, bs):
                m = masks[s:s + bs].unsqueeze(1)          # (b, 1, H, W)
                masked = x * m                            # broadcast over the 3 channels
                out = torch.softmax(self.model(masked), 1)[:, class_idx]
                weights[s:s + bs] = torch.relu(out - base)
            cam = (weights.view(Cc, 1, 1) * acts).sum(0)  # weight the low-res acts
            return cam


# thin convenience subclasses -------------------------------------------------
class GradCAM(CAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer, method="gradcam")


class GradCAMpp(CAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer, method="gradcampp")


class HiResCAM(CAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer, method="hirescam")


class LayerCAM(CAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer, method="layercam")


class EigenCAM(CAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer, method="eigencam")


class ScoreCAM(CAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer, method="scorecam")


_BY_NAME = {
    "gradcam": GradCAM, "gradcampp": GradCAMpp, "hirescam": HiResCAM,
    "layercam": LayerCAM, "eigencam": EigenCAM, "scorecam": ScoreCAM,
}


# ---------------------------------------------------------------------------
# overlay (dependency-free JET via matplotlib colormap)
# ---------------------------------------------------------------------------
def overlay(img01, cam, alpha=0.45):
    """Blend a JET heatmap of ``cam`` over the RGB display image ``img01``.

    Args:
        img01: (H, W, 3) float array in [0, 1].
        cam:   (h, w) float array in [0, 1]; resized to img if needed.
        alpha: heatmap opacity (0..1).
    Returns:
        (H, W, 3) float32 RGB in [0, 1].
    """
    import matplotlib

    img01 = np.asarray(img01, dtype=np.float32)
    cam = np.asarray(cam, dtype=np.float32)
    H, W = img01.shape[:2]
    if cam.shape != (H, W):
        t = torch.from_numpy(cam).view(1, 1, *cam.shape)
        cam = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
    cam = np.clip(cam, 0.0, 1.0)
    heat = matplotlib.colormaps["jet"](cam)[..., :3].astype(np.float32)   # (H,W,3) RGB
    out = alpha * heat + (1.0 - alpha) * img01
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# helpers: load model / pick representative test cells / display image
# ---------------------------------------------------------------------------
def _maybe_load_checkpoint(model, which):
    """Load the LATEST checkpoint for tag ``which`` (ckpt root or a run_*/ snapshot subfolder)
    into ``model`` if one exists; bind it as figure provenance. Returns True if loaded."""
    import ckpt as CK
    try:
        meta = CK.load_into(model, which, map_location="cpu")
        print(f"[cam] loaded {which} <- {meta.get('path')}")
        return True
    except Exception as e:
        print(f"[cam] warning: no checkpoint for {which!r} ({e}); using model as-is")
        return False


def _class_of_path(path):
    for ci, cname in enumerate(C.CLASSES):
        if f"im_{cname}" in path:
            return ci
    return -1


def _representative_test_cells(n_per_class=1):
    """Return {class_idx: [paths]} of up to n_per_class TEST cells per SIPaKMeD class (deterministic)."""
    from data.splits import get_split

    test = get_split("grouped")["test"]
    by_cls = {ci: [] for ci in range(C.NUM_CLASSES)}
    for p in sorted(test):
        ci = _class_of_path(p)
        if ci >= 0:
            by_cls[ci].append(p)
    return {ci: by_cls[ci][:n_per_class] for ci in range(C.NUM_CLASSES)}


def _load_display_and_tensor(path, img_size=None):
    """Return (display HxWx3 in [0,1] at img_size, model-ready tensor (1,3,H,W)). A larger img_size
    (e.g. 380) gives the B0 a finer conv grid (12x12 vs 7x7 at 224) -> sharper CAMs, no retraining."""
    from PIL import Image
    from data.dataset import build_transform

    img_size = img_size or C.IMG_SIZE
    pil = Image.open(path).convert("RGB")
    x = build_transform(train=False, img_size=img_size)(pil).unsqueeze(0)   # (1,3,H,W) normalized
    # build a matching display image: denormalize the SAME tensor so overlay aligns with the CAM grid.
    mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    disp = (x[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32)
    return disp, x


def _finer_layer(model):
    """An earlier conv block (~2x the final grid resolution) for sharper LayerCAM/HiResCAM at a fixed
    input size. For torchvision EfficientNet (B0/B4), `features[-4]` is the last 14x14 block at 224px
    (vs the 7x7 final block `features[-1]`); `features[-3]`/`[-2]` are still 7x7. Falls back to the
    final conv layer for other trunks."""
    feats = getattr(model, "features", None)
    try:
        if feats is not None and len(feats) >= 5:
            return feats[-4]
    except TypeError:
        pass
    return model.last_conv_module


# ---------------------------------------------------------------------------
# driver: one house-style panel per method
# ---------------------------------------------------------------------------
def run_cam(model=None, which="student", methods=_METHODS, n_per_class=1, save=True,
            img_size=None, fine_layer=False):
    """Compute each CAM ``method`` for representative TEST cells and save house-style panels.

    For one (or n_per_class) real SIPaKMeD test cell per class, every requested method's CAM is
    overlaid and arranged into a grid (rows = classes, cols = cells). Each panel is saved to
    results/figures/cam_<method>.png (+ .pdf) titled "true -> pred (conf)".

    Args:
        model:       a built Classifier (trained checkpoint auto-loaded if present, else used as-is).
        which:       checkpoint tag for auto-load ('student' / 'teacher').
        methods:     iterable of method names (subset of the six).
        n_per_class: test cells per class (panel columns = n_per_class).
        save:        write PNG/PDF panels when True.
    Returns:
        {method: saved_png_path} (paths only present when save=True and a panel was produced).
    """
    import plotting as P                     # applies house style globally
    import matplotlib.pyplot as plt          # import AFTER plotting per contract

    if model is None:                        # build the right backbone; _maybe_load_checkpoint loads weights
        from models.backbones import build
        cfg = C.TEACHER if which.startswith("teacher") else C.STUDENT
        model = build(cfg["backbone"], C.NUM_CLASSES, cfg["hidden"], cfg["dropout"], pretrained=False).to(C.DEVICE)
    model = model.eval()
    _maybe_load_checkpoint(model, which)     # handles both raw state_dict and {"model_state": ...}
    dev = next(model.parameters()).device

    img_size = img_size or C.IMG_SIZE
    target_layer = _finer_layer(model) if fine_layer else None      # earlier block -> sharper maps
    suffix = (f"_{img_size}px" if img_size != C.IMG_SIZE else "") + ("_fine" if fine_layer else "")

    cells = _representative_test_cells(n_per_class)
    # preload display images + tensors + predictions once (shared across methods).
    rows = []   # list per class: list of (disp, x, true_idx, pred_idx, conf)
    for ci in range(C.NUM_CLASSES):
        row = []
        for path in cells[ci]:
            disp, x = _load_display_and_tensor(path, img_size=img_size)
            x = x.to(dev)
            with torch.no_grad():
                prob = torch.softmax(model(x.float()), 1)[0]
            pred = int(prob.argmax())
            row.append((disp, x, ci, pred, float(prob[pred])))
        rows.append(row)

    out = {}
    for method in methods:
        method = method.lower()
        if method not in _BY_NAME:
            print(f"[cam] skipping unknown method {method!r}")
            continue
        cam_obj = _BY_NAME[method](model, target_layer=target_layer)
        # horizontal layout: one COLUMN per class (5 across), one ROW per cell-per-class.
        ncols = C.NUM_CLASSES
        nrows = max(1, n_per_class)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols + 0.6, 2.6 * nrows + 0.6),
                                 squeeze=False)
        try:
            for ci, row in enumerate(rows):              # ci -> column (class)
                for r in range(nrows):                   # r  -> row (cell index within class)
                    ax = axes[r][ci]
                    ax.set_xticks([]); ax.set_yticks([])
                    if r >= len(row):
                        ax.axis("off")
                        continue
                    disp, x, true_idx, pred_idx, conf = row[r]
                    cam = cam_obj(x, class_idx=pred_idx)
                    ax.imshow(overlay(disp, cam))
                    t = f"{P.pretty(C.CLASSES[true_idx])} $\\rightarrow$ {P.pretty(C.CLASSES[pred_idx])} ({conf:.2f})"
                    color = "#2ECC71" if pred_idx == true_idx else "#D32F2F"
                    ax.set_title(t, fontsize=10, color=color)
        finally:
            cam_obj.remove()                  # always remove hooks, even if a method raises
        res = f" @{img_size}px" if img_size != C.IMG_SIZE else ""
        res_tag = res + ("/fine-layer" if fine_layer else "")
        fig.suptitle(f"{method.upper()} - {which}{res_tag} (true $\\rightarrow$ predicted)",
                     fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        if save:
            out[method] = P.save(fig, os.path.join(C.FIGDIR, f"cam_{method}{suffix}"))
            print(f"[cam] saved {method} -> {out[method]}")
        else:
            plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# smoke test: random student on CPU, Grad-CAM on one real test image
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.backbones import build

    torch.manual_seed(C.SEED)
    m = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"],
              C.STUDENT["dropout"], pretrained=False).eval()    # RANDOM model on CPU

    cells = _representative_test_cells(1)
    path = next(p for ci in range(C.NUM_CLASSES) for p in cells[ci])
    disp, x = _load_display_and_tensor(path)
    H, W = x.shape[-2:]

    cam = GradCAM(m)(x)                                         # predicted class
    assert cam.shape == (H, W), f"cam shape {cam.shape} != input {(H, W)}"
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6, f"cam range [{cam.min()},{cam.max()}]"
    ov = overlay(disp, cam)
    assert ov.shape == (H, W, 3) and ov.min() >= 0 and ov.max() <= 1.0 + 1e-6

    # every method produces a valid [0,1] HxW map on the random CPU model.
    for name in _METHODS:
        cm = _BY_NAME[name](m)(x)
        assert cm.shape == (H, W) and cm.min() >= 0 and cm.max() <= 1.0 + 1e-6, name
        print(f"{name:10s} ok  shape={cm.shape} range=[{cm.min():.3f},{cm.max():.3f}]")
    print("OK")
