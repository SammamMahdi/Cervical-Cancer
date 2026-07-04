"""Edge-efficiency profiling (U6) on real hardware.

Measures the cost of a trained model so we can argue the student is deployable on
edge hardware: parameter count, FLOPs, wall-clock latency / throughput, and peak
VRAM. FLOP counting is best-effort across three optional backends (fvcore, ptflops,
thop); if none are installed we return None rather than hard-failing -- the rest of
the profile (params, latency, VRAM) is always available.

Public API:
    count_params(model) -> (total_M, trainable_M)
    flops(model, input_size=(1,3,224,224), device=...) -> float (GFLOPs) | None
    latency(model, input_size, device, warmup=10, iters=50) -> dict
    peak_vram_mb(model, input_size, device) -> float | None
    profile(model, name, input_size=(1,3,224,224), device=C.DEVICE) -> dict  (also appended to JSON)

__main__: profiles a torchvision efficientnet_b0 (num_classes=5) and prints the dict.
"""
from __future__ import annotations

import copy
import json
import os
import time

import torch

import config as C

__all__ = [
    "count_params",
    "flops",
    "latency",
    "peak_vram_mb",
    "profile",
]


def _to_device(model, device):
    """Return an eval-mode model on `device` (does not mutate the caller's object in place)."""
    return model.to(device).eval()


def count_params(model) -> tuple[float, float]:
    """Return (total_params_M, trainable_params_M) in millions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def flops(model, input_size=(1, 3, 224, 224), device=C.DEVICE):
    """Best-effort forward FLOPs in GFLOPs for a single sample (batch dim collapsed to 1).

    Tries fvcore -> ptflops -> thop; returns None if none are installed or all fail.
    Runs on a deep-copied eval model under no_grad so the caller's model/grad state is
    untouched. We normalise everything to GFLOPs for ONE image.
    """
    # Profile a clone so dropout/bn state and the device of the live model are unaffected.
    try:
        m = copy.deepcopy(model)
    except Exception:
        m = model
    m = _to_device(m, device)
    c, h, w = input_size[1], input_size[2], input_size[3]
    dummy = torch.zeros(1, c, h, w, device=device)

    # --- fvcore: returns total multiply-accumulate-ish count (FLOPs ~ 2*MACs by convention,
    #     but fvcore reports MACs as "flops"; we report GFLOPs = fvcore_count/1e9 for consistency
    #     with how fvcore is commonly cited, i.e. one MAC = one "flop"). ---
    with torch.no_grad():
        try:
            from fvcore.nn import FlopCountAnalysis

            fca = FlopCountAnalysis(m, dummy)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            return float(fca.total()) / 1e9
        except Exception:
            pass

        try:
            from ptflops import get_model_complexity_info

            macs, _ = get_model_complexity_info(
                m, (c, h, w),
                as_strings=False, print_per_layer_stat=False, verbose=False,
            )
            return float(macs) / 1e9
        except Exception:
            pass

        try:
            from thop import profile as thop_profile

            macs, _ = thop_profile(m, inputs=(dummy,), verbose=False)
            return float(macs) / 1e9
        except Exception:
            pass

    return None


def latency(model, input_size=(1, 3, 224, 224), device=C.DEVICE, warmup=10, iters=50) -> dict:
    """Measure per-batch latency and image throughput.

    Warms up `warmup` iterations (lets cuDNN autotune / caches settle), then times
    `iters` forward passes, synchronising the CUDA stream around each so we capture
    true device time rather than async launch time.

    Returns {latency_ms (per batch), throughput_img_s}.
    """
    model = _to_device(model, device)
    cuda = (str(device).startswith("cuda") and torch.cuda.is_available())
    batch = input_size[0]
    dummy = torch.zeros(*input_size, device=device)

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(dummy)
        if cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(max(1, iters)):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_s = sum(times) / len(times)
    latency_ms = mean_s * 1e3
    throughput = (batch / mean_s) if mean_s > 0 else float("inf")
    return {"latency_ms": latency_ms, "throughput_img_s": throughput}


def peak_vram_mb(model, input_size=(1, 3, 224, 224), device=C.DEVICE):
    """Peak CUDA memory (MB) for one forward pass; None on CPU.

    Resets the peak-memory counter, runs a single forward, and reads the high-water
    mark. Inference-only (no_grad), so this reflects deployment activation memory.
    """
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return None
    model = _to_device(model, device)
    dummy = torch.zeros(*input_size, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(dummy)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1e6


def profile(model, name, input_size=(1, 3, 224, 224), device=C.DEVICE) -> dict:
    """Full efficiency profile for `model`, appended to C.METRICDIR/efficiency.json.

    Returns dict(name, params_M, flops_G, latency_ms, throughput_img_s, peak_vram_mb).
    The JSON file is a list of such records; we append (keeping any prior entries) so
    repeated profiling of different backbones accumulates a comparison table.
    """
    model = _to_device(model, device)
    total_M, _trainable_M = count_params(model)

    rec = {
        "name": name,
        "params_M": round(total_M, 4),
        "flops_G": None,
        "latency_ms": None,
        "throughput_img_s": None,
        "peak_vram_mb": None,
    }

    g = flops(model, input_size=input_size, device=device)
    rec["flops_G"] = round(g, 4) if g is not None else None

    lat = latency(model, input_size=input_size, device=device)
    rec["latency_ms"] = round(lat["latency_ms"], 4)
    rec["throughput_img_s"] = round(lat["throughput_img_s"], 4)

    vram = peak_vram_mb(model, input_size=input_size, device=device)
    rec["peak_vram_mb"] = round(vram, 4) if vram is not None else None

    _append_json(rec)
    return rec


def _append_json(rec: dict):
    """Append one record to C.METRICDIR/efficiency.json (tolerating a missing/corrupt file)."""
    os.makedirs(C.METRICDIR, exist_ok=True)
    path = os.path.join(C.METRICDIR, "efficiency.json")
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            elif isinstance(loaded, dict):
                data = [loaded]
        except Exception:
            data = []
    data.append(rec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import torchvision

    C.set_seed()
    m = torchvision.models.efficientnet_b0(weights=None)
    # swap classifier head to NUM_CLASSES (mirrors how the student head is built)
    in_f = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_f, C.NUM_CLASSES)

    out = profile(m, name="efficientnet_b0", device=C.DEVICE)
    print(json.dumps(out, indent=2))
