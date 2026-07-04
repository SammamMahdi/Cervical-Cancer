"""CellGAN: class-conditional cell-image synthesis for minority/orphan-class rescue (RQ5 / plan P3).

A compact, single-GPU conditional DCGAN that learns SIPaKMeD cell morphology and synthesises extra
samples for under-represented classes, used as an augmentation source for the RQ5 ablation arm.

Honesty / leakage safety:
  * trains ONLY on the leakage-safe TRAIN split (data/splits.make_split) -- never val/test, so
    synthetic data cannot leak the sealed test;
  * synthetic images are an augmentation, never counted as real data in any metric;
  * losses + sample grids are logged to results/metrics + results/figures (JSON/CSV/PNG) like everything else.

Usage:
    python cellgan.py --smoke                         # 2-epoch sanity check
    python cellgan.py --epochs 120                    # train the conditional GAN
    python cellgan.py --synthesize Koilocytotic 500   # write 500 synthetic Koilocytotic crops
"""
import os, glob, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config as C
import runlog
import plotting as P

GAN_RES = 64
GAN_CKPT = os.path.join(C.CKPTDIR, "cellgan.pt")
SYNTH_DIR = os.path.join(C.RESULTS, "synthetic")
Z_DIM = 128


# --------------------------------------------------------------------------- #
#  Data: SIPaKMeD train-split cells at 64px (train split only)
# --------------------------------------------------------------------------- #
class CellImages(Dataset):
    def __init__(self, res=GAN_RES, split_kind="grouped"):
        from data.splits import get_split
        from data.dataset import _class_of      # path-derived label (case-robust; same as the trainer)
        sp = get_split(split_kind)
        self.items = [(p, _class_of(p)) for p in sp["train"]]
        self.res = res

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        try:
            im = Image.open(path).convert("RGB").resize((self.res, self.res))
            a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0          # -> [-1,1]
        except Exception:
            a = np.zeros((self.res, self.res, 3), np.float32)
        return torch.from_numpy(a).permute(2, 0, 1), y


# --------------------------------------------------------------------------- #
#  Conditional DCGAN (64x64), label-embedding conditioning
# --------------------------------------------------------------------------- #
class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, n_classes=C.NUM_CLASSES, ngf=64):
        super().__init__()
        self.emb = nn.Embedding(n_classes, z_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim * 2, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z, y):
        ze = torch.cat([z, self.emb(y)], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(ze)


class Discriminator(nn.Module):
    def __init__(self, n_classes=C.NUM_CLASSES, ndf=64, res=GAN_RES):
        super().__init__()
        self.emb = nn.Embedding(n_classes, res * res)
        self.res = res
        self.net = nn.Sequential(
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        ymap = self.emb(y).view(-1, 1, self.res, self.res)
        return self.net(torch.cat([x, ymap], dim=1)).view(-1)


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #
def train_cellgan(epochs=120, batch_size=64, lr=2e-4, smoke=False, res=GAN_RES):
    C.set_seed()
    dev = C.DEVICE
    ds = CellImages(res=res)
    if len(ds) == 0:
        raise RuntimeError("no training cells found — build the split first (run.py eda)")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    G, D = Generator().to(dev), Discriminator(res=res).to(dev)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    hist = {"loss_D": [], "loss_G": [], "epoch": []}
    fixed_z = torch.randn(C.NUM_CLASSES * 4, Z_DIM, device=dev)
    fixed_y = torch.arange(C.NUM_CLASSES, device=dev).repeat(4)
    n_epochs = 2 if smoke else epochs

    for ep in range(1, n_epochs + 1):
        ld, lg, nb = 0.0, 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); bs = x.size(0)
            real_lbl = torch.full((bs,), 0.9, device=dev)      # one-sided label smoothing
            fake_lbl = torch.zeros(bs, device=dev)
            # --- D ---
            optD.zero_grad(set_to_none=True)
            z = torch.randn(bs, Z_DIM, device=dev)
            fake = G(z, y)
            lossD = bce(D(x, y), real_lbl) + bce(D(fake.detach(), y), fake_lbl)
            lossD.backward(); optD.step()
            # --- G ---
            optG.zero_grad(set_to_none=True)
            lossG = bce(D(fake, y), torch.ones(bs, device=dev))
            lossG.backward(); optG.step()
            ld += lossD.item(); lg += lossG.item(); nb += 1
            if smoke and nb >= 3:
                break
        hist["loss_D"].append(ld / max(nb, 1)); hist["loss_G"].append(lg / max(nb, 1)); hist["epoch"].append(ep)
        print(f"[cellgan] ep {ep:03d}/{n_epochs}  loss_D={ld/max(nb,1):.3f}  loss_G={lg/max(nb,1):.3f}")
        runlog.dump_json(os.path.join(C.METRICDIR, "cellgan_history.json"), hist)
        if ep % max(1, n_epochs // 6) == 0 or ep == n_epochs:
            _sample_grid(G, fixed_z, fixed_y, ep)

    torch.save({"G": G.state_dict(), "D": D.state_dict(), "res": res, "z_dim": Z_DIM,
                "classes": C.CLASSES}, GAN_CKPT)
    # loss curve (companion CSV via plotting)
    P.training_curves({"loss": hist["loss_D"], "val_loss": hist["loss_G"]},
                      os.path.join(C.FIGDIR, "cellgan_loss"), title="CellGAN loss (D vs G)")
    print(f"[cellgan] saved -> {GAN_CKPT}")
    return hist


def _sample_grid(G, z, y, ep):
    G.eval()
    with torch.no_grad():
        imgs = ((G(z, y).cpu().permute(0, 2, 3, 1).numpy() + 1) / 2).clip(0, 1)
    G.train()
    titles = [C.CLASSES[i][:12] for i in y.cpu().numpy()]
    P.montage(list(imgs), titles, os.path.join(C.FIGDIR, f"cellgan_samples_ep{ep}"),
              suptitle=f"CellGAN samples (epoch {ep})", ncol=C.NUM_CLASSES)


# --------------------------------------------------------------------------- #
#  Synthesis + augmentation hook
# --------------------------------------------------------------------------- #
@torch.no_grad()
def synthesize(class_name, n, out_dir=None, batch=128):
    """Generate n synthetic crops for one class -> PNGs + manifest. Returns the output dir."""
    assert os.path.exists(GAN_CKPT), "train CellGAN first (cellgan.py --epochs ...)"
    ck = torch.load(GAN_CKPT, map_location=C.DEVICE)
    G = Generator().to(C.DEVICE); G.load_state_dict(ck["G"]); G.eval()
    ci = C.CLASSES.index(class_name)
    out_dir = out_dir or os.path.join(SYNTH_DIR, class_name)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    made = 0
    while made < n:
        b = min(batch, n - made)
        z = torch.randn(b, Z_DIM, device=C.DEVICE)
        y = torch.full((b,), ci, device=C.DEVICE, dtype=torch.long)
        imgs = ((G(z, y).cpu().permute(0, 2, 3, 1).numpy() + 1) / 2 * 255).clip(0, 255).astype("uint8")
        for k in range(b):
            p = os.path.join(out_dir, f"{class_name}_synth_{made+k:05d}.png")
            Image.fromarray(imgs[k]).save(p); paths.append(p)
        made += b
    runlog.dump_json(os.path.join(out_dir, "manifest.json"),
                     dict(class_name=class_name, class_idx=ci, n=len(paths),
                          source="cellgan", ckpt=os.path.basename(GAN_CKPT),
                          note="synthetic augmentation; never counted as real data"))
    print(f"[cellgan] wrote {len(paths)} synthetic {class_name} crops -> {out_dir}")
    return out_dir


class SynthClips(Dataset):
    """Labeled dataset over results/synthetic/<class>/*.png for the RQ5 augment arm.
    Labels come from the directory name (never from a real-data path), so synthetic samples are
    clearly separable and never confused with real cells."""
    def __init__(self, transform, root=SYNTH_DIR):
        self.transform = transform; self.items = []
        for cname in (C.CLASSES if os.path.isdir(root) else []):
            cdir = os.path.join(root, cname)
            for p in glob.glob(os.path.join(cdir, "*.png")):
                self.items.append((p, C.CLASSES.index(cname)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y


def augmented_train_dataset(real_train_ds, transform):
    """Concat real train cells with CellGAN synthetic clips (RQ5). Returns a ConcatDataset."""
    from torch.utils.data import ConcatDataset
    synth = SynthClips(transform)
    if len(synth) == 0:
        return real_train_ds
    print(f"[cellgan] augmenting train with {len(synth)} synthetic clips")
    return ConcatDataset([real_train_ds, synth])


def synthesize_minority(target_per_class=None, min_count=None):
    """RQ5 hook: top up under-represented classes to the majority count (or target_per_class)."""
    from collections import Counter
    ds = CellImages()
    counts = Counter(y for _p, y in ds.items)
    target = target_per_class or max(counts.values())
    made = {}
    for ci, cname in enumerate(C.CLASSES):
        need = target - counts.get(ci, 0)
        if min_count is not None:
            need = max(need, min_count)
        if need > 0:
            synthesize(cname, need)
            made[cname] = need
    runlog.dump_json(os.path.join(SYNTH_DIR, "synth_plan.json"),
                     dict(real_counts={C.CLASSES[k]: v for k, v in counts.items()},
                          target=target, synthesized=made))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--synthesize", nargs=2, metavar=("CLASS", "N"), default=None)
    args = ap.parse_args()
    if args.synthesize:
        synthesize(args.synthesize[0], int(args.synthesize[1]))
    else:
        train_cellgan(epochs=args.epochs, smoke=args.smoke)
