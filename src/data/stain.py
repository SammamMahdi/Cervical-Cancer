"""NumPy stain normalization + MixStyle for cytology (A4 / U4 stain-domain robustness).

Implemented directly with NumPy (NO torchstain / staintools dependency):

- ReinhardNormalizer  : LAB mean/std matching (Reinhard et al., 2001).
- MacenkoNormalizer   : OD + SVD stain-vector estimation and concentration
                        renormalization to a fitted reference (Macenko et al., 2009).
- fit_on_paths        : fit a normalizer on a representative TRAIN-only sample.
- MixStyle(nn.Module) : per-instance channel mean/std mixing across a shuffled
                        batch (Zhou et al., ICLR 2021); identity in eval.
- StainTransform      : PIL.Image -> PIL.Image wrapper for the dataset pipeline.

All normalizers take/return uint8 HxWx3 RGB. `fit` must be called before
`transform`; transform falls back to the identity (a copy) if unfitted, so the
pipeline never crashes on an un-fitted normalizer.
"""
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import config as C

__all__ = [
    "ReinhardNormalizer",
    "MacenkoNormalizer",
    "fit_on_paths",
    "MixStyle",
    "StainTransform",
]

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# RGB <-> LAB helpers (numpy, no skimage dependency)                          #
# --------------------------------------------------------------------------- #
# sRGB -> XYZ (D65) and XYZ -> CIELAB. Vectorised over HxWx3 float arrays.
_RGB2XYZ = np.array([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
# D65 reference white.
_WHITE = np.array([0.950456, 1.0, 1.088754], dtype=np.float64)


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def _f(t):
    d = 6.0 / 29.0
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)


def _f_inv(t):
    d = 6.0 / 29.0
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))


def rgb_to_lab(rgb_u8):
    """uint8 HxWx3 RGB -> float64 HxWx3 CIELAB (L in [0,100])."""
    rgb = _srgb_to_linear(rgb_u8.astype(np.float64) / 255.0)
    xyz = rgb @ _RGB2XYZ.T
    xyz = xyz / _WHITE
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """float64 HxWx3 CIELAB -> uint8 HxWx3 RGB."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * _WHITE
    rgb = _linear_to_srgb(xyz @ _XYZ2RGB.T)
    return np.clip(rgb * 255.0, 0, 255).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Reinhard                                                                     #
# --------------------------------------------------------------------------- #
class ReinhardNormalizer:
    """Reinhard et al. (2001): match per-channel LAB mean/std to a reference."""

    def __init__(self):
        self.ref_mean = None  # (3,)
        self.ref_std = None   # (3,)

    @property
    def fitted(self):
        return self.ref_mean is not None

    def fit(self, ref_rgb_u8):
        lab = rgb_to_lab(np.asarray(ref_rgb_u8))
        self.ref_mean = lab.reshape(-1, 3).mean(0)
        self.ref_std = lab.reshape(-1, 3).std(0)
        return self

    def transform(self, img_rgb_u8):
        img = np.asarray(img_rgb_u8)
        if not self.fitted:
            return img.copy()
        lab = rgb_to_lab(img)
        flat = lab.reshape(-1, 3)
        mu, sd = flat.mean(0), flat.std(0)
        # (x - mu)/sd * ref_sd + ref_mean, guarding zero-variance channels.
        scale = self.ref_std / np.where(sd < _EPS, 1.0, sd)
        out = (lab - mu) * scale + self.ref_mean
        return lab_to_rgb(out)


# --------------------------------------------------------------------------- #
# Macenko                                                                      #
# --------------------------------------------------------------------------- #
class MacenkoNormalizer:
    """Macenko et al. (2009): estimate H&E stain vectors in optical-density space
    via SVD, then renormalize stain concentrations to a fitted reference.

    Parameters
    ----------
    Io : background (white) intensity used for OD = -log10(I/Io).
    beta : OD threshold; pixels with OD below this (near-white) are excluded.
    alpha : percentile (and 100-alpha) used to find robust min/max stain angles.
    """

    def __init__(self, Io=240.0, beta=0.15, alpha=1.0):
        self.Io = float(Io)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.ref_stains = None  # (3, 2) stain-vector matrix (columns = H, E)
        self.ref_conc = None    # (2,)   reference 99th-pct concentrations

    @property
    def fitted(self):
        return self.ref_stains is not None

    # -- core estimation ---------------------------------------------------- #
    def _od(self, img):
        """RGB uint8 -> optical density (N, 3), clamped away from 0."""
        rgb = np.asarray(img).reshape(-1, 3).astype(np.float64)
        rgb = np.maximum(rgb, 1.0)  # avoid log(0) for pure-black pixels
        return -np.log10(rgb / self.Io)

    def _stain_matrix(self, img):
        """Estimate the (3, 2) stain matrix [H | E] for one image."""
        od = self._od(img)
        # Keep only sufficiently stained (non-white) pixels.
        mask = (od > self.beta).any(axis=1)
        od_hat = od[mask]
        if od_hat.shape[0] < 10:  # degenerate / nearly-white image
            od_hat = od
        # SVD: plane spanned by the two largest singular vectors.
        _, _, Vt = np.linalg.svd(od_hat - od_hat.mean(0, keepdims=True), full_matrices=False)
        V = Vt[:2].T  # (3, 2) basis of the OD plane
        proj = od_hat @ V
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        v_min = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v_max = V @ np.array([np.cos(max_phi), np.sin(max_phi)])
        # Order so that H (more red/first channel) is column 0.
        if v_min[0] > v_max[0]:
            stains = np.stack([v_min, v_max], axis=1)
        else:
            stains = np.stack([v_max, v_min], axis=1)
        # Force outward (positive) orientation and unit columns.
        stains = stains / (np.linalg.norm(stains, axis=0, keepdims=True) + _EPS)
        stains *= np.sign(stains.sum(0, keepdims=True) + _EPS)
        return stains  # (3, 2)

    def _concentrations(self, img, stains):
        """Solve OD = stains @ C for concentrations C -> (N, 2)."""
        od = self._od(img)
        # Least-squares: C = pinv(stains) @ od^T  -> (2, N)
        conc = np.linalg.lstsq(stains, od.T, rcond=None)[0]
        return np.maximum(conc, 0.0).T  # (N, 2)

    def fit(self, ref_rgb_u8):
        ref = np.asarray(ref_rgb_u8)
        self.ref_stains = self._stain_matrix(ref)
        conc = self._concentrations(ref, self.ref_stains)
        self.ref_conc = np.percentile(conc, 99, axis=0) + _EPS
        return self

    def transform(self, img_rgb_u8):
        img = np.asarray(img_rgb_u8)
        if not self.fitted:
            return img.copy()
        h, w = img.shape[:2]
        try:
            stains = self._stain_matrix(img)
            conc = self._concentrations(img, stains)  # (N, 2)
        except np.linalg.LinAlgError:
            return img.copy()
        max_conc = np.percentile(conc, 99, axis=0) + _EPS
        # Rescale this image's concentrations to the reference dynamic range.
        conc = conc * (self.ref_conc / max_conc)[None, :]
        # Reconstruct OD with the REFERENCE stain matrix, then back to RGB.
        od = conc @ self.ref_stains.T            # (N, 3)
        rgb = self.Io * np.power(10.0, -od)
        rgb = np.clip(rgb, 0, 255).reshape(h, w, 3)
        return rgb.round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Reference fitting from TRAIN paths                                           #
# --------------------------------------------------------------------------- #
def fit_on_paths(normalizer, paths, n=50, seed=C.SEED):
    """Fit `normalizer` on a representative sample of TRAIN images.

    We sample up to `n` paths and build a single reference image by tiling them
    into a montage (a roughly-square grid of fixed 64x64 crops). The montage's
    pooled pixel statistics are a robust stand-in for the "average" train slide,
    which is more stable than picking one arbitrary image while staying cheap and
    fully deterministic. Returns the fitted `normalizer`.
    """
    rng = np.random.default_rng(seed)
    paths = list(paths)
    if not paths:
        raise ValueError("fit_on_paths: empty `paths`")
    sel = paths if len(paths) <= n else [paths[i] for i in rng.choice(len(paths), n, replace=False)]

    tile = 64
    tiles = []
    for p in sel:
        try:
            im = Image.open(p).convert("RGB").resize((tile, tile), Image.BILINEAR)
            tiles.append(np.asarray(im))
        except Exception:
            continue
    if not tiles:
        raise ValueError("fit_on_paths: no readable images in sample")

    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    montage = np.full((rows * tile, cols * tile, 3), 255, dtype=np.uint8)  # white pad
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        montage[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t

    normalizer.fit(montage)
    return normalizer


# --------------------------------------------------------------------------- #
# MixStyle (Zhou et al., ICLR 2021)                                            #
# --------------------------------------------------------------------------- #
class MixStyle(nn.Module):
    """Mix per-instance feature statistics across a shuffled batch.

    During training, with probability `p`, replace each sample's channel-wise
    (mu, sigma) with a Beta(alpha, alpha)-weighted convex combination of its own
    and a shuffled peer's statistics. This synthesizes novel "styles" (here:
    stain appearances) without changing content. Identity in eval mode or for a
    batch size < 2. Intended right after an early conv block.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def extra_repr(self):
        return f"p={self.p}, alpha={self.alpha}"

    def forward(self, x):
        if not self.training or x.size(0) < 2 or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig                       # instance-normalized content

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return x_norm * sig_mix + mu_mix


# --------------------------------------------------------------------------- #
# PIL pipeline wrapper                                                         #
# --------------------------------------------------------------------------- #
class StainTransform:
    """Callable adapter: PIL.Image -> PIL.Image applying a fitted normalizer.

    Drop into a torchvision Compose BEFORE AspectPad/ToTensor, e.g.
        T.Compose([StainTransform(norm), AspectPad(224), T.ToTensor(), ...]).
    If the normalizer is unfitted it acts as the identity.
    """

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB"))
        out = self.normalizer.transform(arr)
        return Image.fromarray(out)


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(C.SEED)
    # Two synthetic "slides" with different colour casts.
    ref = np.clip(rng.normal(180, 25, (96, 96, 3)) + np.array([20, -10, 0]), 0, 255).astype(np.uint8)
    src = np.clip(rng.normal(140, 30, (80, 120, 3)) + np.array([-15, 5, 25]), 0, 255).astype(np.uint8)

    rein = ReinhardNormalizer().fit(ref)
    out_r = rein.transform(src)
    print("Reinhard :", src.shape, "->", out_r.shape, out_r.dtype,
          "| mean", np.round(out_r.reshape(-1, 3).mean(0), 1))

    mac = MacenkoNormalizer().fit(ref)
    out_m = mac.transform(src)
    print("Macenko  :", src.shape, "->", out_m.shape, out_m.dtype,
          "| mean", np.round(out_m.reshape(-1, 3).mean(0), 1))

    # StainTransform on a PIL image.
    pil_out = StainTransform(rein)(Image.fromarray(src))
    print("StainTransform PIL:", pil_out.size, pil_out.mode)

    # MixStyle on a random feature tensor.
    ms = MixStyle(p=1.0, alpha=0.1)
    x = torch.randn(8, 16, 32, 32)
    ms.train(); y_tr = ms(x)
    ms.eval();  y_ev = ms(x)
    print("MixStyle :", tuple(y_tr.shape), "train-changed", bool((y_tr - x).abs().sum() > 0),
          "| eval-identity", torch.allclose(y_ev, x))
    print("OK")
