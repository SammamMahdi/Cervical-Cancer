"""Timestamped checkpoint registry + latest-checkpoint resolution (new pipeline).

WHY: every training run should be archivable by timestamp, and every figure / evaluation
must load the MOST RECENT checkpoint for a model tag without hand-editing paths. This module
is the single place that (a) writes timestamped checkpoints + a JSON registry, and (b)
resolves the latest checkpoint for a tag -- searching the checkpoint root AND any timestamped
run subfolder (results/checkpoints/run_<YYYYmmdd_HHMMSS>/), so moved/snapshotted runs still
resolve transparently.

Naming convention
-----------------
    <tag>.pt                       canonical / current (raw state_dict OR dict payload)
    <tag>__<YYYYmmdd_HHMMSS>.pt    timestamped archive (dict payload with metadata)
    results/checkpoints/run_<stamp>/<tag>.pt   a moved/snapshotted training run

Payload (when THIS module writes): {"model_state", "tag", "stamp", "saved_at", "metrics",
...extra}. ``load`` transparently unwraps that payload, a bare state_dict, or the legacy
{"model"/"state_dict": ...} wrappers used elsewhere in the repo.
"""
import os, glob, shutil
from datetime import datetime

import torch

import config as C
import runlog

REGISTRY = os.path.join(C.CKPTDIR, "registry.json")


def now_stamp():
    """Filesystem-safe training-run timestamp, e.g. 20260622_210324."""
    return runlog.now_id()


# --------------------------------------------------------------------------- resolve
def _candidates(tag):
    """Every checkpoint file for ``tag`` in the ckpt root AND any immediate subfolder.

    Matches ``<tag>.pt`` and ``<tag>__<stamp>.pt`` (double-underscore stamp separator), so a
    tag like 'student' never collides with 'student-kd_best' or a different model's files."""
    pats = (f"{tag}.pt", f"{tag}__*.pt")
    roots = [C.CKPTDIR] + [d for d in glob.glob(os.path.join(C.CKPTDIR, "*")) if os.path.isdir(d)]
    out = set()
    for r in roots:
        for p in pats:
            out.update(glob.glob(os.path.join(r, p)))
    return sorted(p for p in out if os.path.isfile(p))


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def latest_path(tag):
    """Path to the most-recent checkpoint for ``tag`` (newest mtime across the ckpt root and
    every run subfolder), or None if none exists. mtime is preserved by ``shutil.move`` so a
    snapshotted run keeps its training time and a fresh run correctly wins as 'latest'."""
    cands = _candidates(tag)
    return max(cands, key=_mtime) if cands else None


def list_runs(tag):
    """All checkpoints for ``tag``, newest first, as dicts {path, stamp, mtime}."""
    rows = []
    for p in _candidates(tag):
        rows.append({"path": p, "mtime": _mtime(p)})
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)


# --------------------------------------------------------------------------- load
def _unwrap(obj):
    """Return (state_dict, meta) from any payload convention used in this repo."""
    if isinstance(obj, dict):
        for k in ("model_state", "model", "state_dict"):
            v = obj.get(k)
            if isinstance(v, dict):
                meta = {kk: obj[kk] for kk in obj if kk != k}
                return v, meta
    return obj, {}


def load(tag_or_path, map_location="cpu"):
    """Resolve ``tag_or_path`` (a tag -> latest, or an explicit existing path) and return
    (state_dict, meta, path). Raises FileNotFoundError when nothing resolves."""
    path = tag_or_path if (tag_or_path and os.path.isfile(tag_or_path)) else latest_path(tag_or_path)
    if not path:
        raise FileNotFoundError(f"no checkpoint for {tag_or_path!r} under {C.CKPTDIR}")
    state, meta = _unwrap(torch.load(path, map_location=map_location))
    meta["path"] = path
    return state, meta, path


def load_into(model, tag_or_path, map_location="cpu", strict=True, bind_provenance=True):
    """Load the latest checkpoint for ``tag_or_path`` INTO ``model``. Binds figure provenance
    (so every figure produced afterwards is stamped with this checkpoint + its training time)
    unless ``bind_provenance=False``. Returns the checkpoint meta dict."""
    state, meta, path = load(tag_or_path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    if bind_provenance:
        try:
            import plotting as P
            P.set_provenance(checkpoint=path, checkpoint_saved_at=meta.get("saved_at"))
        except Exception:
            pass
    return meta


# --------------------------------------------------------------------------- write
def save(tag, state, metrics=None, history=True, stamp=None, **fields):
    """Save a timestamped checkpoint for ``tag``.

    Writes ``<tag>__<stamp>.pt`` (the archive) when ``history`` and refreshes the stable
    ``<tag>.pt`` alias, recording both in registry.json. ``fields`` (optimizer/epoch/scaler
    state, etc.) are stored alongside ``model_state`` for resumable checkpoints. Returns the
    primary path written. Pass ``history=False`` for big rolling 'last' checkpoints to skip the
    per-stamp archive copy (alias-only)."""
    os.makedirs(C.CKPTDIR, exist_ok=True)
    stamp = stamp or now_stamp()
    sd = state.state_dict() if hasattr(state, "state_dict") else state
    payload = {"model_state": sd, "tag": tag, "stamp": stamp,
               "saved_at": datetime.now().isoformat(timespec="seconds"), "metrics": metrics or {}}
    payload.update(fields)
    alias = os.path.join(C.CKPTDIR, f"{tag}.pt")
    primary = os.path.join(C.CKPTDIR, f"{tag}__{stamp}.pt") if history else alias
    torch.save(payload, primary)
    if history:
        try:
            shutil.copyfile(primary, alias)
        except Exception:
            torch.save(payload, alias)
    _register(tag, primary, stamp, payload["saved_at"], metrics or {})
    return primary


def _register(tag, path, stamp, saved_at, metrics):
    reg = runlog.load_json(REGISTRY, default={}) or {}
    reg.setdefault(tag, []).append({"path": os.path.abspath(path), "stamp": stamp,
                                    "saved_at": saved_at, "metrics": metrics or {}})
    runlog.dump_json(REGISTRY, reg)


def register_existing(path, tag=None, stamp=None):
    """Record an already-on-disk checkpoint in the registry (used after a snapshot move)."""
    tag = tag or os.path.splitext(os.path.basename(path))[0].split("__")[0]
    _register(tag, path, stamp or now_stamp(), datetime.fromtimestamp(_mtime(path)).isoformat(timespec="seconds"), {})


def snapshot(dest_subdir, move=True, register=True):
    """Move (or copy) all current ROOT-level ``*.pt`` checkpoints into CKPTDIR/<dest_subdir>/.

    Returns (dest_dir, [moved_paths]). ``latest_path`` searches run subfolders, so eval/figures
    keep resolving these after a move. Existing root archives ``*__*.pt`` are included too."""
    dest = os.path.join(C.CKPTDIR, dest_subdir)
    os.makedirs(dest, exist_ok=True)
    moved = []
    for p in sorted(glob.glob(os.path.join(C.CKPTDIR, "*.pt"))):
        if not os.path.isfile(p):
            continue
        d = os.path.join(dest, os.path.basename(p))
        (shutil.move if move else shutil.copy2)(p, d)
        moved.append(d)
        if register:
            register_existing(d)
    return dest, moved


if __name__ == "__main__":   # quick self-check
    print("CKPTDIR:", C.CKPTDIR)
    for t in ("student", "teacher"):
        print(f"  latest_path({t!r}) -> {latest_path(t)}")
