"""SimCLR self-supervised PRE-training on pooled, LABEL-FREE cervical crops (DATA_PLAN SSL / A9).

Why: the SIPaKMeD supervised set is small (4,049 cells). Several large unlabeled cervical-cytology
sources sit on disk (Herlev, APACC, HiCervix, CRIC). SimCLR contrastive pre-training on the *pooled,
label-free* image union learns a domain-adapted backbone initialisation that train.py can load (via
:func:`load_ssl_init`) instead of the generic ImageNet weights — no labels, no leakage of the sealed
SIPaKMeD test split structure (the SSL pool is image paths only, never the supervised labels/split).

(Module named `selfsup` — NOT `ssl` — so it never shadows Python's stdlib `ssl` on PYTHONPATH.)

Pipeline
--------
``pool_image_paths()``  -> union of crop paths:
    SIPaKMeD cropped cells (data.clustering.load_cells)
  + Herlev   data/raw/herlev/<class>/*.bmp          (skip the '*-d.bmp' segmentation masks)
  + APACC    data/raw/apacc/test/{healthy,unhealthy}/*.png
  + HiCervix data/raw/hicervix/{test,train,val}/<split>/*.{jpg,png}
  + CRIC     data/raw/cric/cric_dataset_complete/*.png   (if present)
  -- de-duplicated, sorted, fully label-free.

``TwoViewDataset`` -> two INDEPENDENT strong-augmented views of each image (the SimCLR view family).
``SimCLR`` -> encoder = build(backbone).features + pool feeding a projection MLP; ``nt_xent`` is the
NT-Xent / InfoNCE loss. ``pretrain`` saves the BACKBONE state_dict to C.CKPTDIR/ssl_<backbone>.pt so
train.py can warm-start; ``load_ssl_init`` transfers those trunk weights into a supervised Classifier.
"""
import os
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

import config as C
from models.backbones import build, count_params

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **_kw):
        return it

IMG_EXT = (".bmp", ".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- #
#  Label-free image pool
# --------------------------------------------------------------------------- #
def _imgs_in(directory, recursive=False):
    """All image files directly in (or, if recursive, under) `directory`; skip Herlev '-d.bmp' masks."""
    if not directory or not os.path.isdir(directory):
        return []
    pat = os.path.join(directory, "**", "*") if recursive else os.path.join(directory, "*")
    out = []
    for p in glob.glob(pat, recursive=recursive):
        low = p.lower()
        if low.endswith("-d.bmp"):                 # Herlev segmentation mask -> skip
            continue
        if low.endswith(IMG_EXT) and os.path.isfile(p):
            out.append(p)
    return out


def pool_image_paths(verbose=False):
    """Union of LABEL-FREE crop paths from every cervical-cytology source present on disk.

    De-duplicated, sorted; missing datasets skipped. NO labels / NO split structure — purely the
    unlabeled image union for contrastive pre-training.
    """
    paths, per_source = [], {}
    try:
        from data.clustering import load_cells
        sip = [c.path for c in load_cells()]
    except Exception:
        sip = []
    per_source["sipakmed"] = len(sip); paths += sip
    herlev = _imgs_in(C.HERLEV_DIR, recursive=True); per_source["herlev"] = len(herlev); paths += herlev
    apacc = _imgs_in(C.APACC_DIR, recursive=True); per_source["apacc"] = len(apacc); paths += apacc
    hicervix = []
    for split in ("test", "train", "val"):
        hicervix += _imgs_in(os.path.join(C.HICERVIX_DIR, split, split))
    per_source["hicervix"] = len(hicervix); paths += hicervix
    cric = _imgs_in(os.path.join(C.DATA_RAW, "cric", "cric_dataset_complete"), recursive=False)
    per_source["cric"] = len(cric); paths += cric
    paths = sorted(set(os.path.abspath(p) for p in paths))
    if verbose:
        print("SSL pool per-source:", per_source, "| total unique:", len(paths))
    return paths


# --------------------------------------------------------------------------- #
#  Two-view augmentation dataset
# --------------------------------------------------------------------------- #
def simclr_transform(img_size=C.IMG_SIZE):
    color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0), antialias=True),
        T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5), T.RandomRotation(180),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


class TwoViewDataset(Dataset):
    """Yield two INDEPENDENT augmented views (v1, v2) of each pooled image — the SimCLR positive pair."""
    def __init__(self, paths=None, transform=None, img_size=C.IMG_SIZE):
        self.paths = list(paths) if paths is not None else pool_image_paths()
        self.transform = transform or simclr_transform(img_size)
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        try:
            return Image.open(self.paths[i]).convert("RGB")
        except Exception:
            return Image.new("RGB", (self.img_size, self.img_size))

    def __getitem__(self, i):
        img = self._load(i)
        return self.transform(img), self.transform(img)


# --------------------------------------------------------------------------- #
#  SimCLR model + NT-Xent loss
# --------------------------------------------------------------------------- #
class SimCLR(nn.Module):
    """Shared encoder (Classifier conv trunk + pool) -> projection MLP head."""
    def __init__(self, backbone="efficientnet_b0", proj_dim=128, hidden=512, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = build(backbone, num_classes=C.NUM_CLASSES, pretrained=pretrained)
        feat = self.encoder.feature_dim
        self.proj_dim = proj_dim
        self.projector = nn.Sequential(
            nn.Linear(feat, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def encode(self, x):
        return self.encoder.features_vec(x)

    def forward(self, x):
        return F.normalize(self.projector(self.encode(x)), dim=1)

    def backbone_state_dict(self):
        return self.encoder.features.state_dict()


def nt_xent(z1, z2, temperature=0.2):
    """NT-Xent / InfoNCE contrastive loss (SimCLR)."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temperature
    diag = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag, float("-inf"))
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
#  Pre-training loop
# --------------------------------------------------------------------------- #
def pretrain(backbone="efficientnet_b0", epochs=100, batch=256, lr=1e-3, weight_decay=1e-4,
             temperature=0.2, proj_dim=128, img_size=C.IMG_SIZE, num_workers=C.NUM_WORKERS,
             paths=None, out=None, device=None, pretrained=True, resume=True, log_every=50):
    """SimCLR pre-train `backbone` on the pooled label-free crops; save the BACKBONE state_dict."""
    device = device or C.DEVICE
    out = out or os.path.join(C.CKPTDIR, f"ssl_{backbone}.pt")
    last_path = out.replace(".pt", "_last.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    paths = paths if paths is not None else pool_image_paths(verbose=True)
    if not paths:
        raise RuntimeError("SSL pool is empty — no images under data/raw/{sipakmed,herlev,apacc,hicervix,cric}")
    ds = TwoViewDataset(paths, transform=simclr_transform(img_size), img_size=img_size)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"))

    model = SimCLR(backbone, proj_dim=proj_dim, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    start_epoch, best_loss = 0, float("inf")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        if ck.get("backbone") == backbone:
            model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
            start_epoch = ck.get("epoch", 0) + 1; best_loss = ck.get("best_loss", float("inf"))
            print(f"resumed from {last_path} @ epoch {start_epoch} (best_loss={best_loss:.4f})")

    total, trainable = count_params(model)
    print(f"SimCLR[{backbone}] params: total={total/1e6:.2f}M | pool={len(ds)} imgs | batch={batch} | device={device}")

    def _save(path, epoch, bl):
        torch.save({"backbone": backbone, "backbone_state": model.backbone_state_dict(),
                    "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                    "epoch": epoch, "best_loss": bl,
                    "config": dict(proj_dim=proj_dim, temperature=temperature, img_size=img_size,
                                   batch=batch, lr=lr, weight_decay=weight_decay, n_pool=len(ds))}, path)

    for epoch in range(start_epoch, epochs):
        model.train(); running, nb = 0.0, 0
        pbar = tqdm(loader, desc=f"SSL {backbone} ep {epoch+1}/{epochs}", leave=False)
        for step, (v1, v2) in enumerate(pbar):
            v1 = v1.to(device, non_blocking=True); v2 = v2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = nt_xent(model(v1), model(v2), temperature=temperature)
            loss.backward(); opt.step()
            running += float(loss.item()); nb += 1
            if log_every and (step % log_every == 0):
                try: pbar.set_postfix(loss=f"{loss.item():.4f}")
                except Exception: pass
        sched.step()
        mean_loss = running / max(1, nb)
        print(f"epoch {epoch+1}/{epochs}  mean NT-Xent = {mean_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")
        _save(last_path, epoch, best_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss; _save(out, epoch, best_loss)

    if not os.path.exists(out):
        _save(out, max(0, epochs - 1), best_loss)
    print(f"done. best_loss={best_loss:.4f} -> {out}")
    return out


# --------------------------------------------------------------------------- #
#  Transfer to a supervised Classifier
# --------------------------------------------------------------------------- #
def load_ssl_init(model, path, strict=True, verbose=True):
    """Load SimCLR-pretrained BACKBONE weights into a supervised Classifier (skip heads)."""
    ck = torch.load(path, map_location="cpu")
    ckpt_bb = ck.get("backbone"); model_bb = getattr(model, "backbone_name", None)
    if strict and ckpt_bb is not None and model_bb is not None and ckpt_bb != model_bb:
        raise ValueError(f"backbone mismatch: checkpoint={ckpt_bb!r} model={model_bb!r}")
    state = ck.get("backbone_state")
    if state is None:
        ms = ck.get("model_state", ck); prefix = "encoder.features."
        state = {k[len(prefix):]: v for k, v in ms.items() if k.startswith(prefix)}
    missing, unexpected = model.features.load_state_dict(state, strict=False)
    if verbose:
        print(f"load_ssl_init[{model_bb}]: loaded {len(state)} trunk tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)}); head left at init.")
    return model


if __name__ == "__main__":
    C.set_seed(C.SEED)
    pool = []
    try:
        pool = pool_image_paths(verbose=True)
    except Exception as e:
        print("pool unavailable:", e)
    n = 8
    if len(pool) >= n:
        ds = TwoViewDataset(pool[:n], img_size=64)
        v1, v2 = zip(*[ds[i] for i in range(n)]); v1 = torch.stack(v1); v2 = torch.stack(v2)
    else:
        v1 = torch.randn(n, 3, 64, 64); v2 = torch.randn(n, 3, 64, 64)
    model = SimCLR("efficientnet_b0", proj_dim=64, hidden=128, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); model.train(); losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = nt_xent(model(v1), model(v2), temperature=0.2)
        assert torch.isfinite(loss); loss.backward(); opt.step(); losses.append(float(loss))
    assert losses[-1] < losses[0], losses
    print(f"NT-Xent {losses[0]:.4f} -> {losses[-1]:.4f}  OK")
