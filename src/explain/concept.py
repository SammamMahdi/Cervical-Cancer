"""Axis-3 concept attribution (TCAV) for the cervical-cytology student.

TCAV (Testing with Concept Activation Vectors, Kim et al. 2018) answers a
*directional* question the pixel-level methods (Grad-CAM / IG / counterfactual)
cannot: "How sensitive is the model's prediction of class *k* to a human-named
concept?" -- e.g. *high nuclear:cytoplasm ratio*, *koilocyte perinuclear halo*,
*coarse chromatin*. A concept is defined by a small set of EXAMPLE IMAGES that
exhibit it; TCAV learns a linear "concept activation vector" (CAV) in the
activation space of a chosen layer (here the last conv block, ``features.8``),
then measures the fraction of class examples whose gradient aligns with that
CAV (the TCAV score in [0, 1]). A score significantly above the random-concept
baseline (~0.5) means the concept causally pushes the class logit up.

------------------------------------------------------------------------------
HONEST STATUS -- this is a WORKING SCAFFOLD, not a finished cervical analysis.
------------------------------------------------------------------------------
TCAV requires *curated* concept image folders that we DO NOT ship: building a
defensible "koilocyte halo" or "high N:C ratio" concept set needs a cytologist
to hand-pick crops, plus matched random/control sets. Until those exist this
module proves the pipeline runs end-to-end on SYNTHETIC random images (see the
``synthetic_smoke`` below and ``__main__``). Swap in real folders via
``concept_from_folder`` and you get genuine cervical concept attributions with
no further code changes.

How to supply concept image folders
-----------------------------------
Lay out one folder per concept plus >=1 random/control folder, e.g.::

    concepts/
      high_nc_ratio/      img001.png img002.png ...   (>= ~30-50 crops)
      koilocyte_halo/     ...
      coarse_chromatin/   ...
      random_0/           ...   (arbitrary cells NOT exhibiting the concept)
      random_1/           ...

then::

    from explain.concept import concept_from_folder, tcav_score
    from models.backbones import build
    import config as C
    m = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"],
              C.STUDENT["dropout"], pretrained=False)
    # ... load results/checkpoints/student.pt into m if present ...
    halo   = concept_from_folder("concepts/koilocyte_halo", cid=0, name="halo")
    rand0  = concept_from_folder("concepts/random_0",       cid=1, name="random_0")
    rand1  = concept_from_folder("concepts/random_1",       cid=2, name="random_1")
    # class_idx=3 == "Koilocytotic" (NORMAL-first ordering, see config.CLASSES)
    scores = tcav_score(m, [halo], [rand0, rand1],
                        target_layer="features.8", class_idx=3,
                        inputs=koilocytotic_test_batch)

The returned dict maps each concept name to its TCAV magnitude (sign-count)
score and the layer it was computed at.

Public API
----------
- ``concept_from_folder(folder, cid, name, transform=None, batch_size=...)`` ->
  ``captum.concept.Concept`` backed by the images in ``folder``.
- ``concept_from_images(pil_images, cid, name, transform=None, ...)`` -> Concept
  from in-memory PIL images (used by the synthetic smoke).
- ``tcav_score(model, concept_dirs, random_dirs, target_layer, class_idx,
  inputs, ...)`` -> ``{concept_name: {"tcav": float, "layer": str}}``.
- ``synthetic_smoke(device="cpu")`` -> runs a tiny random-image TCAV to prove
  the pipeline; returns the score dict.

Gradient note: TCAV computes layer gradients, so this MUST run in float32 with
grads enabled (no ``torch.no_grad`` / autocast) -- enforced internally.
"""
from __future__ import annotations

import os
import glob
import tempfile
from typing import List, Optional, Sequence, Dict

import torch
from torch.utils.data import DataLoader, Dataset

import config as C

# captum.concept is optional at import time -> degrade gracefully.
try:
    from captum.concept import TCAV, Concept
    _HAS_CAPTUM_CONCEPT = True
    _CAPTUM_ERR = None
except Exception as e:  # pragma: no cover - only on broken/old captum
    TCAV = Concept = None  # type: ignore
    _HAS_CAPTUM_CONCEPT = False
    _CAPTUM_ERR = e

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Default TCAV target layer: the student's last conv block (Grad-CAM target too).
DEFAULT_LAYER = "features.8"


def captum_concept_available() -> bool:
    """True iff ``captum.concept`` imported successfully."""
    return _HAS_CAPTUM_CONCEPT


def _require_captum():
    if not _HAS_CAPTUM_CONCEPT:
        raise RuntimeError(
            "captum.concept is unavailable -- TCAV cannot run. "
            f"(import error: {_CAPTUM_ERR!r}). Install/upgrade captum: "
            "`pip install captum`."
        )


# ---------------------------------------------------------------------------
# Concept construction
# ---------------------------------------------------------------------------
class _TensorListDataset(Dataset):
    """Yields raw (C,H,W) tensors -- the shape captum.concept expects (no labels)."""

    def __init__(self, tensors: Sequence[torch.Tensor]):
        self.tensors = list(tensors)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        return self.tensors[i]


class _PathDataset(Dataset):
    """Loads each image path -> transformed (C,H,W) tensor (no labels), for captum.concept."""

    def __init__(self, paths: Sequence[str], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


def _default_transform():
    """Eval transform (PIL -> normalized tensor); falls back to a minimal one."""
    try:
        from data.dataset import build_transform
        return build_transform(train=False)
    except Exception:  # pragma: no cover - defensive
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
        ])


def concept_from_images(pil_or_tensor_images, cid: int, name: str,
                        transform=None, batch_size: int = 16) -> "Concept":
    """Build a captum ``Concept`` from in-memory images (PIL.Image or CHW tensors).

    Args:
        pil_or_tensor_images: list of PIL.Image (transformed via ``transform``)
            OR list of already-normalized (C,H,W) float tensors (transform=None).
        cid:  unique integer concept id (must differ across all concepts in a run).
        name: human-readable concept name (keys the output scores).
        transform: PIL->tensor transform; defaults to the eval transform. Ignored
            when inputs are already tensors.
    """
    _require_captum()
    imgs = list(pil_or_tensor_images)
    if len(imgs) and isinstance(imgs[0], torch.Tensor):
        ds: Dataset = _TensorListDataset(imgs)
    else:
        transform = transform or _default_transform()
        ds = _TensorListDataset([transform(im) for im in imgs])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return Concept(id=int(cid), name=str(name), data_iter=loader)


def concept_from_folder(folder: str, cid: int, name: Optional[str] = None,
                        transform=None, batch_size: int = 16) -> "Concept":
    """Build a captum ``Concept`` from every image file under ``folder``.

    Args:
        folder: directory of concept (or random/control) example images.
        cid:    unique integer concept id across the whole TCAV run.
        name:   concept name (defaults to the folder's basename).
        transform: PIL->tensor transform (defaults to eval transform).
    """
    _require_captum()
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
        if os.path.splitext(p)[1].lower() in IMG_EXT
    )
    if not paths:
        raise FileNotFoundError(f"no images ({IMG_EXT}) found under {folder!r}")
    transform = transform or _default_transform()
    ds = _PathDataset(paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return Concept(id=int(cid), name=name or os.path.basename(os.path.normpath(folder)),
                   data_iter=loader)


# ---------------------------------------------------------------------------
# TCAV scoring
# ---------------------------------------------------------------------------
def _resolve_concepts(spec, kind: str, transform, batch_size: int, _cid_start: int):
    """Accept Concept objects OR folder-path strings; return (list[Concept], next_cid)."""
    out: List["Concept"] = []
    cid = _cid_start
    for item in spec:
        if isinstance(item, Concept):
            out.append(item)
        elif isinstance(item, str):
            out.append(concept_from_folder(item, cid=cid, transform=transform,
                                           batch_size=batch_size))
            cid += 1
        else:
            raise TypeError(
                f"{kind} entries must be captum Concept or folder-path str, got {type(item)}"
            )
    return out, cid


def tcav_score(model: torch.nn.Module,
               concept_dirs,
               random_dirs,
               target_layer: str = DEFAULT_LAYER,
               class_idx: int = C.ABNORMAL_IDX[0],
               inputs: Optional[torch.Tensor] = None,
               transform=None,
               batch_size: int = 16,
               device: str = "cpu",
               n_random: Optional[int] = None) -> Dict[str, Dict]:
    """Compute TCAV scores of ``class_idx`` for each concept against random baselines.

    Args:
        model:        any model with the contract API (``.forward(x)->logits``);
                      need NOT be trained -- degrades to a random-init model fine.
        concept_dirs: list of captum ``Concept`` objects OR folder-path strings,
                      one per concept of interest (e.g. ['koilocyte_halo/']).
        random_dirs:  list of captum ``Concept`` objects OR folder-path strings
                      of CONTROL/random images (>=1; >=2 recommended for the
                      magnitude/sign-count statistic).
        target_layer: dotted submodule name of the layer to probe
                      (default ``"features.8"``, the student's last conv block).
        class_idx:    the class whose sensitivity to each concept is measured
                      (index into ``config.CLASSES``; default = first abnormal).
        inputs:       (N,3,H,W) batch of examples of ``class_idx`` whose gradients
                      are scored. If None, a tiny random batch is used (smoke only;
                      for real results pass a batch of that class's test images).
        transform:    PIL->tensor transform used when folder paths are given.
        device:       "cpu" or "cuda".
        n_random:     if set, only the first ``n_random`` random concepts are used.

    Returns:
        ``{concept_name: {"tcav": float, "layer": target_layer,
                          "class_idx": class_idx, "class": class_name}}``
        where "tcav" is the magnitude (sign-count) TCAV score in [0, 1].
        ~0.5 == no effect; ->1.0 == strong positive concept influence.
    """
    _require_captum()
    model = model.to(device).float().eval()

    concepts, cid = _resolve_concepts(concept_dirs, "concept_dirs", transform,
                                      batch_size, _cid_start=0)
    randoms, _ = _resolve_concepts(random_dirs, "random_dirs", transform,
                                   batch_size, _cid_start=cid)
    if n_random is not None:
        randoms = randoms[:n_random]
    if not concepts:
        raise ValueError("concept_dirs is empty")
    if not randoms:
        raise ValueError("random_dirs is empty (need >=1 random/control concept)")

    # Experimental sets: each concept vs the random pool. captum compares the
    # concept's CAV against the random ones to estimate the TCAV score.
    experimental_sets = [[c, *randoms] for c in concepts]

    # Per-run save dir for cached CAVs (temp; cleaned by the OS).
    save_path = os.path.join(tempfile.gettempdir(), "tcav_cav")
    os.makedirs(save_path, exist_ok=True)

    tcav = TCAV(model=model, layers=[target_layer], save_path=save_path)

    if inputs is None:
        # Smoke fallback only -- real calls should pass a batch of `class_idx` images.
        inputs = torch.randn(4, 3, C.IMG_SIZE, C.IMG_SIZE)
    inputs = inputs.to(device).float()

    # TCAV needs gradients through the layer -> force grad on, no autocast.
    with torch.set_grad_enabled(True):
        result = tcav.interpret(inputs=inputs,
                                experimental_sets=experimental_sets,
                                target=int(class_idx))

    # result: {exp_set_key: {layer: {"sign_count": tensor, "magnitude": tensor}}}
    # The concept of interest is index 0 within each experimental set.
    out: Dict[str, Dict] = {}
    class_name = C.CLASSES[class_idx] if 0 <= class_idx < len(C.CLASSES) else str(class_idx)
    for exp_set, concept in zip(experimental_sets, concepts):
        key = "-".join(str(c.id) for c in exp_set)
        layer_res = result[key][target_layer]
        mag = layer_res["magnitude"]  # tensor over concepts in the set
        score = float(mag[0]) if hasattr(mag, "__len__") else float(mag)
        out[concept.name] = {
            "tcav": score,
            "layer": target_layer,
            "class_idx": int(class_idx),
            "class": class_name,
        }
    return out


# ---------------------------------------------------------------------------
# Synthetic smoke -- proves the pipeline runs without any curated concept sets.
# ---------------------------------------------------------------------------
def synthetic_smoke(device: str = "cpu", n_per_set: int = 12,
                    seed: int = C.SEED) -> Dict[str, Dict]:
    """Create two tiny RANDOM-image sets ('concept' vs 'random') on the fly and
    compute a TCAV score, proving the TCAV pipeline runs end-to-end on a
    random-init student model. Returns the score dict.

    This is NOT a meaningful cervical attribution -- both sets are noise, so the
    score should sit near the 0.5 chance line. It exists only to validate wiring.
    """
    if not _HAS_CAPTUM_CONCEPT:
        print("[concept] captum.concept unavailable -> skipping synthetic smoke.")
        print(f"          import error: {_CAPTUM_ERR!r}")
        return {}

    torch.manual_seed(seed)
    from models.backbones import build
    model = build(C.STUDENT["backbone"], C.NUM_CLASSES, C.STUDENT["hidden"],
                  C.STUDENT["dropout"], pretrained=False)

    # Two random concepts (already-normalized tensors -> transform=None).
    def rand_imgs(n):
        return [torch.randn(3, C.IMG_SIZE, C.IMG_SIZE) for _ in range(n)]

    concept = concept_from_images(rand_imgs(n_per_set), cid=0, name="synthetic_concept")
    random0 = concept_from_images(rand_imgs(n_per_set), cid=1, name="random_0")
    random1 = concept_from_images(rand_imgs(n_per_set), cid=2, name="random_1")

    inputs = torch.randn(4, 3, C.IMG_SIZE, C.IMG_SIZE)
    scores = tcav_score(model, [concept], [random0, random1],
                        target_layer=DEFAULT_LAYER, class_idx=C.ABNORMAL_IDX[0],
                        inputs=inputs, device=device)
    return scores


if __name__ == "__main__":
    if not captum_concept_available():
        print("captum.concept is NOT available; TCAV scaffold cannot run.")
        print(f"reason: {_CAPTUM_ERR!r}")
        raise SystemExit(0)
    print(f"Running synthetic TCAV smoke (layer={DEFAULT_LAYER}, "
          f"class={C.CLASSES[C.ABNORMAL_IDX[0]]}) on CPU ...")
    scores = synthetic_smoke(device="cpu")
    for name, info in scores.items():
        print(f"  concept {name!r}: TCAV={info['tcav']:.3f} "
              f"(layer={info['layer']}, class={info['class']})")
    print("OK -- TCAV pipeline runs. (Random images -> score is meaningless; "
          "supply curated concept folders via concept_from_folder for real results.)")
