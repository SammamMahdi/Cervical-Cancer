"""NucleusAnchor — scale-normalize a single-cell crop by its NUCLEUS size (Gate 2).

Motivation (Class_Analysis_Findings, Finding 4): on SIPaKMeD the apparent *cytoplasm* area
correlates +0.989 with the raw crop pixel area, i.e. the model can read class off absolute scale.
The nucleus area, by contrast, is scale-robust (corr -0.045). NucleusAnchor removes the absolute-
scale degree of freedom at the INPUT by rescaling every crop so the nucleus's equivalent diameter
spans a fixed fraction ``target_frac`` of the (square) frame, then center-cropping/padding to a
square. After this, a "big superficial cell" and a "small parabasal cell" present their nuclei at the
SAME pixel size, so any residual scale signal is genuine (cytoplasm-to-nucleus ratio), not a
crop-resolution artifact.

Usage (as a transform placed BEFORE data.dataset.AspectPad):

    from data.anchor import NucleusAnchor
    anchor = NucleusAnchor(target_frac=0.35)
    img = anchor(pil_crop, mask=bool_HxW)          # explicit mask, or
    img = anchor(pil_crop, crop_path="..._.bmp")    # derive mask via explain.nucleus._nucleus_mask

The transform is a no-op (identity) when no nucleus mask is available (fallback='identity') or pads
to a square (fallback='aspectpad') so it composes cleanly with the existing pipeline.
"""
import numpy as np
from PIL import Image


def _equiv_diameter_from_mask(mask):
    """Equivalent diameter (px) of a boolean mask = diameter of a disk with the same area."""
    area = float(np.asarray(mask, bool).sum())
    if area <= 0:
        return 0.0
    return float(np.sqrt(4.0 * area / np.pi))


def _mask_centroid(mask):
    ys, xs = np.nonzero(np.asarray(mask, bool))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _square_pad(img, size, fill=0):
    """Aspect-preserving: resize longest side to ``size`` then center-pad to size x size."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    out = Image.new(img.mode, (size, size), fill)
    out.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return out


class NucleusAnchor:
    """Rescale a crop so its nucleus spans ``target_frac`` of the square frame, then center-crop/pad.

    Parameters
    ----------
    target_frac : float
        Desired nucleus equivalent-diameter as a fraction of the output frame side. The anchoring
        scale is ``s = target_frac * frame_side / nucleus_equiv_diameter`` (in the crop's own pixel
        frame, ``frame_side = max(W, H)`` before squaring).
    out_size : int or None
        Output side length. ``None`` -> use ``max(W, H)`` of the input crop (keeps native sampling).
    fallback : {'aspectpad', 'identity'}
        Behavior when no nucleus mask is available. 'aspectpad' squares the crop; 'identity' returns
        the crop untouched.
    """

    def __init__(self, target_frac=0.35, out_size=None, fallback="aspectpad", fill=0):
        self.target_frac = float(target_frac)
        self.out_size = out_size
        self.fallback = fallback
        self.fill = fill

    # -- helpers ------------------------------------------------------------ #
    def scale_factor(self, nucleus_equiv_diameter, frame_side):
        """Anchoring scale s so the nucleus spans target_frac of the frame. 0/NaN -> 1.0 (identity)."""
        if not np.isfinite(nucleus_equiv_diameter) or nucleus_equiv_diameter <= 0:
            return 1.0
        return float(self.target_frac * frame_side / nucleus_equiv_diameter)

    def _resolve_mask(self, img, mask, crop_path):
        if mask is not None:
            return np.asarray(mask, bool)
        if crop_path is not None:
            from explain.nucleus import _nucleus_mask
            # derive at the native crop resolution (max side) so it aligns with the PIL crop
            w, h = img.size
            m = _nucleus_mask(crop_path, max(w, h))
            if m is None:
                return None
            # _nucleus_mask returns a square (size x size) aspect-padded mask; resample to crop frame
            return m
        return None

    def __call__(self, img, mask=None, crop_path=None):
        w, h = img.size
        frame = max(w, h)
        out_size = self.out_size if self.out_size is not None else frame

        m = self._resolve_mask(img, mask, crop_path)
        if m is None or np.asarray(m, bool).sum() == 0:
            if self.fallback == "identity":
                return img
            return _square_pad(img, out_size, self.fill)

        m = np.asarray(m, bool)
        # If mask came back at a different resolution than the crop, scale equiv-diameter accordingly.
        mh, mw = m.shape
        equiv_d_mask = _equiv_diameter_from_mask(m)
        # express the nucleus diameter in the CROP's own max-side frame
        equiv_d_crop = equiv_d_mask * (frame / max(mw, mh))

        s = self.scale_factor(equiv_d_crop, frame)

        # rescale the crop by s, then center on the nucleus centroid and crop/pad to out_size
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        scaled = img.resize((nw, nh), Image.BILINEAR)

        cen = _mask_centroid(m)
        if cen is None:
            cx, cy = nw / 2.0, nh / 2.0
        else:
            # centroid is in mask pixels -> map into the scaled-crop frame
            cx = cen[0] * (nw / mw)
            cy = cen[1] * (nh / mh)

        out = Image.new(img.mode, (out_size, out_size), self.fill)
        # paste so that (cx, cy) lands at the output center
        left = int(round(out_size / 2.0 - cx))
        top = int(round(out_size / 2.0 - cy))
        out.paste(scaled, (left, top))
        return out
