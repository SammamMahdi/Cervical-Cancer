"""Pixel-attribution explanations (captum + LIME + SHAP) for the cytology classifier.

Every public function operates on a SINGLE preprocessed input tensor ``x`` of shape (3,H,W) or
(1,3,H,W) (already normalized with C.IMAGENET_MEAN/STD) and an integer ``target`` class, and returns
a numpy attribution map. captum gradient methods need real gradients, so they run in float32 with NO
``no_grad``/autocast.  LIME and SHAP operate on the raw PIL image / a background batch respectively.

Public API
----------
integrated_gradients(model, x, target, steps=50)   -> (3,H,W) signed attribution (numpy)
saliency(model, x, target)                          -> (3,H,W) |grad| attribution (numpy)
guided_backprop(model, x, target)                   -> (3,H,W) attribution (numpy)
deeplift(model, x, target, baseline=None)           -> (3,H,W) attribution (numpy)
occlusion(model, x, target, window=15, stride=8)    -> (3,H,W) attribution (numpy)
lime_explain(model, pil_image, target=None, ...)    -> dict(mask, boundary_img, label) | None
shap_pixel(model, background_batch, x, target=None) -> (3,H,W) attribution (numpy) | None
run_attributions(model, which='student', methods=('ig','saliency','occlusion'), n=5, save=True)
    -> {method: saved_png_path}

Each library call is wrapped in try/except: a missing or misbehaving optional dependency prints a
warning and is skipped rather than crashing the pipeline.

__main__: build a RANDOM model on CPU, run IG + Saliency on one real test image, and assert the
attribution H,W matches the input.
"""
import os
import warnings

import numpy as np
import torch

import config as C


# --------------------------------------------------------------------------- helpers
def _to_batch(x):
    """Accept (3,H,W) or (1,3,H,W); return a float32 4-D tensor (1,3,H,W)."""
    x = x if torch.is_tensor(x) else torch.as_tensor(x)
    x = x.float()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"expected (3,H,W) or (1,3,H,W) tensor, got {tuple(x.shape)}")
    return x


def _device_of(model):
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - model with no params
        return torch.device("cpu")


def _resolve_target(model, x, target):
    """Default target to the model's argmax prediction for x."""
    if target is not None:
        return int(target)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return int(logits.argmax(dim=1).item())


def _np3(attr):
    """captum attribution tensor (1,3,H,W) -> numpy (3,H,W)."""
    return attr.squeeze(0).detach().cpu().numpy()


def _denorm(x):
    """Undo ImageNet normalization on a (3,H,W) tensor -> (H,W,3) numpy in [0,1] for display."""
    x = x.detach().cpu().float()
    mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _heat(attr3):
    """Collapse a (3,H,W) attribution to a normalized (H,W) magnitude heatmap in [0,1]."""
    h = np.abs(np.asarray(attr3)).sum(axis=0)
    rng = float(h.max() - h.min())
    if rng < 1e-12:
        return np.zeros_like(h)
    return (h - h.min()) / rng


# --------------------------------------------------------------------------- captum methods
def integrated_gradients(model, x, target=None, steps=50):
    """Integrated Gradients: returns signed (3,H,W) attribution (numpy). None on failure."""
    try:
        from captum.attr import IntegratedGradients
    except Exception as e:  # pragma: no cover - captum guaranteed present in env
        print(f"[attribution] integrated_gradients skipped: captum import failed ({e})")
        return None
    try:
        model.eval()
        x = _to_batch(x).to(_device_of(model)).requires_grad_(True)
        tgt = _resolve_target(model, x, target)
        ig = IntegratedGradients(model)
        attr = ig.attribute(x, target=tgt, n_steps=steps)
        return _np3(attr)
    except Exception as e:
        print(f"[attribution] integrated_gradients failed: {e}")
        return None


def saliency(model, x, target=None):
    """Vanilla saliency (|d logit / d input|): returns (3,H,W) attribution (numpy). None on failure."""
    try:
        from captum.attr import Saliency
    except Exception as e:  # pragma: no cover
        print(f"[attribution] saliency skipped: captum import failed ({e})")
        return None
    try:
        model.eval()
        x = _to_batch(x).to(_device_of(model)).requires_grad_(True)
        tgt = _resolve_target(model, x, target)
        sal = Saliency(model)
        attr = sal.attribute(x, target=tgt, abs=True)
        return _np3(attr)
    except Exception as e:
        print(f"[attribution] saliency failed: {e}")
        return None


def guided_backprop(model, x, target=None):
    """Guided Backpropagation: returns (3,H,W) attribution (numpy). None on failure."""
    try:
        from captum.attr import GuidedBackprop
    except Exception as e:  # pragma: no cover
        print(f"[attribution] guided_backprop skipped: captum import failed ({e})")
        return None
    try:
        model.eval()
        x = _to_batch(x).to(_device_of(model)).requires_grad_(True)
        tgt = _resolve_target(model, x, target)
        gbp = GuidedBackprop(model)
        attr = gbp.attribute(x, target=tgt)
        return _np3(attr)
    except Exception as e:
        print(f"[attribution] guided_backprop failed: {e}")
        return None


def deeplift(model, x, target=None, baseline=None):
    """DeepLIFT (vs a zero/black baseline by default): (3,H,W) attribution (numpy). None on failure."""
    try:
        from captum.attr import DeepLift
    except Exception as e:  # pragma: no cover
        print(f"[attribution] deeplift skipped: captum import failed ({e})")
        return None
    try:
        model.eval()
        x = _to_batch(x).to(_device_of(model)).requires_grad_(True)
        tgt = _resolve_target(model, x, target)
        base = torch.zeros_like(x) if baseline is None else _to_batch(baseline).to(x.device)
        dl = DeepLift(model)
        attr = dl.attribute(x, baselines=base, target=tgt)
        return _np3(attr)
    except Exception as e:
        print(f"[attribution] deeplift failed: {e}")
        return None


def occlusion(model, x, target=None, window=15, stride=8):
    """Occlusion sensitivity (sliding gray patch): (3,H,W) attribution (numpy). None on failure."""
    try:
        from captum.attr import Occlusion
    except Exception as e:  # pragma: no cover
        print(f"[attribution] occlusion skipped: captum import failed ({e})")
        return None
    try:
        model.eval()
        x = _to_batch(x).to(_device_of(model))
        tgt = _resolve_target(model, x, target)
        occ = Occlusion(model)
        attr = occ.attribute(
            x,
            target=tgt,
            sliding_window_shapes=(3, int(window), int(window)),
            strides=(3, int(stride), int(stride)),
            baselines=0,
        )
        return _np3(attr)
    except Exception as e:
        print(f"[attribution] occlusion failed: {e}")
        return None


# --------------------------------------------------------------------------- LIME
def lime_explain(model, pil_image, target=None, num_samples=1000, top_labels=5,
                 num_features=8, transform=None):
    """LIME image explanation for a raw PIL image.

    Wraps the eval transform + softmax into a ``predict_fn`` over a batch of (H,W,3) uint8 images,
    as required by ``lime.lime_image.LimeImageExplainer``.  Returns a dict with the boolean
    super-pixel ``mask``, a ``boundary_img`` (mark_boundaries overlay, (H,W,3) float in [0,1]) and
    the explained ``label``.  Returns None if lime/skimage is unavailable or the call fails.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except Exception as e:
        print(f"[attribution] lime_explain skipped: lime/skimage import failed ({e})")
        return None
    try:
        from data.dataset import build_transform
        tfm = transform if transform is not None else build_transform(train=False)
        dev = _device_of(model)
        model.eval()

        def predict_fn(images):
            batch = []
            from PIL import Image
            for im in images:
                arr = np.asarray(im)
                pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                batch.append(tfm(pil))
            xb = torch.stack(batch).to(dev).float()
            with torch.no_grad():
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

        rgb = np.asarray(pil_image.convert("RGB"))
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            rgb, predict_fn, top_labels=top_labels, hide_color=0, num_samples=num_samples
        )
        label = explanation.top_labels[0] if target is None else int(target)
        temp, mask = explanation.get_image_and_mask(
            label, positive_only=True, num_features=num_features, hide_rest=False
        )
        boundary = mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask)
        return {"mask": mask, "boundary_img": boundary, "label": int(label)}
    except Exception as e:
        print(f"[attribution] lime_explain failed: {e}")
        return None


# --------------------------------------------------------------------------- SHAP
def shap_pixel(model, background_batch, x, target=None):
    """Pixel SHAP via ``shap.GradientExplainer`` on a background batch.

    ``background_batch`` is a (B,3,H,W) tensor of preprocessed reference images; ``x`` is the single
    input to explain.  Returns a (3,H,W) attribution (numpy) for ``target`` (default argmax class).
    Returns None if shap is unavailable or the call fails.
    """
    try:
        import shap
    except Exception as e:
        print(f"[attribution] shap_pixel skipped: shap import failed ({e})")
        return None
    try:
        model.eval()
        dev = _device_of(model)
        bg = _to_batch(background_batch) if background_batch.dim() == 3 else background_batch.float()
        bg = bg.to(dev)
        x = _to_batch(x).to(dev)
        tgt = _resolve_target(model, x, target)

        explainer = shap.GradientExplainer(model, bg)
        out = explainer.shap_values(x, ranked_outputs=None)
        # shap returns either a list (one (1,3,H,W) array per class) or a single
        # (1,3,H,W,n_classes) array depending on version; normalise to (3,H,W) for tgt.
        if isinstance(out, (list, tuple)):
            arr = np.asarray(out[tgt])          # (1,3,H,W)
            attr = arr[0]
        else:
            arr = np.asarray(out)               # (1,3,H,W,n_classes)
            if arr.ndim == 5:
                attr = arr[0, ..., tgt]
            else:                               # already (1,3,H,W)
                attr = arr[0]
        return np.asarray(attr)
    except Exception as e:
        print(f"[attribution] shap_pixel failed: {e}")
        return None


# --------------------------------------------------------------------------- driver / panels
_METHODS = {
    "ig": ("Integrated Gradients", lambda m, x, t: integrated_gradients(m, x, t)),
    "saliency": ("Saliency", lambda m, x, t: saliency(m, x, t)),
    "guided_backprop": ("Guided Backprop", lambda m, x, t: guided_backprop(m, x, t)),
    "deeplift": ("DeepLIFT", lambda m, x, t: deeplift(m, x, t)),
    "occlusion": ("Occlusion", lambda m, x, t: occlusion(m, x, t)),
}


def _load_model(which):
    """Load the requested checkpoint, degrading to a RANDOM model if absent (contract)."""
    from models.backbones import build
    spec = C.TEACHER if which == "teacher" else C.STUDENT
    import ckpt as CK
    model = build(spec["backbone"], C.NUM_CLASSES, spec["hidden"], spec["dropout"], pretrained=False)
    try:
        meta = CK.load_into(model, which, map_location="cpu", strict=False)   # latest (root or run_*/)
        print(f"[attribution] loaded checkpoint {meta.get('path')}")
    except Exception as e:
        print(f"[attribution] no checkpoint for {which!r} ({e}); using random weights")
    return model.eval()


def _pick_test_images(n):
    """Return up to n (PIL.Image, tensor, label, path) tuples from the grouped test split."""
    from PIL import Image
    from data.splits import get_split
    from data.dataset import build_transform, _class_of
    paths = get_split("grouped")["test"][:max(1, n)]
    tfm = build_transform(train=False)
    out = []
    for p in paths:
        try:
            pil = Image.open(p).convert("RGB")
            out.append((pil, tfm(pil), _class_of(p), p))
        except Exception as e:
            print(f"[attribution] could not load test image {p}: {e}")
    return out


def run_attributions(model=None, which="student", methods=("ig", "saliency", "occlusion"),
                     n=5, save=True):
    """Build house-style overlay panels (original | heatmap | overlay) per method on real test cells.

    Saves results/figures/attr_<method>.png (+ .pdf).  Returns {method: png_path}.  A model may be
    passed in; otherwise the ``which`` checkpoint is loaded (random weights if absent).
    """
    import plotting as P  # applies the house style globally
    import matplotlib.pyplot as plt

    if model is None:
        model = _load_model(which)
    model.eval()
    samples = _pick_test_images(n)
    if not samples:
        print("[attribution] no test images available; nothing to plot")
        return {}

    saved = {}
    for key in methods:
        if key not in _METHODS:
            print(f"[attribution] unknown method {key!r}; skipping")
            continue
        title, fn = _METHODS[key]
        rows = []
        for pil, x, label, _path in samples:
            attr = fn(model, x, label)
            if attr is None:
                continue
            rows.append((_denorm(x), _heat(attr), label))
        if not rows:
            print(f"[attribution] {key}: no attributions produced; skipping panel")
            continue

        nrows = len(rows)
        fig, axes = plt.subplots(nrows, 3, figsize=(9.0, 3.0 * nrows), squeeze=False)
        for r, (img, heat, label) in enumerate(rows):
            cname = C.CLASSES[label] if 0 <= label < len(C.CLASSES) else str(label)
            axes[r][0].imshow(img)
            axes[r][0].set_ylabel(P.pretty(cname), fontsize=11)
            axes[r][1].imshow(heat, cmap="inferno")
            axes[r][2].imshow(img)
            axes[r][2].imshow(heat, cmap="inferno", alpha=0.5)
            for c, ttl in zip(range(3), ("Input", title, "Overlay")):
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
                if r == 0:
                    axes[r][c].set_title(ttl)
        fig.suptitle(f"{title} attributions ({which})", fontweight="bold", y=1.0)
        out_noext = os.path.join(C.FIGDIR, f"attr_{key}")
        if save:
            saved[key] = P.save(fig, out_noext)
            print(f"[attribution] saved {saved[key]}")
        else:
            plt.close(fig)
    return saved


# --------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    C.set_seed()
    from models.backbones import build

    model = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"],
                  C.STUDENT["dropout"], pretrained=False).to("cpu").eval()

    samples = _pick_test_images(1)
    assert samples, "no real test image available for smoke test"
    pil, x, label, path = samples[0]
    H, W = x.shape[-2], x.shape[-1]
    print(f"smoke: image={os.path.basename(path)} input HxW={H}x{W} label={label}")

    ig = integrated_gradients(model, x, target=label, steps=8)
    assert ig is not None, "integrated_gradients returned None"
    assert ig.shape[-2:] == (H, W), f"IG shape {ig.shape} != input HxW {(H, W)}"
    print(f"IG attribution shape={ig.shape} OK")

    sal = saliency(model, x, target=label)
    assert sal is not None, "saliency returned None"
    assert sal.shape[-2:] == (H, W), f"saliency shape {sal.shape} != input HxW {(H, W)}"
    print(f"Saliency attribution shape={sal.shape} OK")

    print("OK")
