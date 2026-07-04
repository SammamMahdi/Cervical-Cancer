"""Timestamped, crash-safe JSON run logging + small JSON helpers.

A RunLogger writes results/logs/run_<timestamp>.json, rewritten atomically after every epoch so
nothing is lost if training is interrupted. All metrics live in JSON so every diagram can be
regenerated from disk without re-training.
"""
import os, json
from datetime import datetime


def now_id():
    """Filesystem-safe timestamp id for the current run, e.g. 20260621_143012."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_jsonable(o):
    """Best-effort convert numpy / torch scalars and arrays to plain Python for json.dump."""
    import numpy as np
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    return str(o)


def dump_json(path, obj):
    """Atomic write: temp file then os.replace, so a crash never leaves a half-written file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class RunLogger:
    """Accumulates config + per-epoch metrics + events; flushes to a timestamped JSON each update."""
    def __init__(self, log_dir, run_id, config):
        self.path = os.path.join(log_dir, f"run_{run_id}.json")
        self.data = {
            "run_id": run_id,
            "started": datetime.now().isoformat(timespec="seconds"),
            "ended": None,
            "config": config,
            "stages": {},        # stage -> list of per-epoch records
            "events": [],         # timestamped milestones (resume, stage start/end, ...)
            "final": None,        # summary metrics
        }
        self.flush()

    def event(self, msg):
        self.data["events"].append({"t": datetime.now().isoformat(timespec="seconds"), "msg": msg})
        self.flush()

    def epoch(self, stage, record):
        self.data["stages"].setdefault(stage, []).append(record)
        self.flush()

    def finalize(self, final):
        self.data["final"] = final
        self.data["ended"] = datetime.now().isoformat(timespec="seconds")
        self.flush()

    def flush(self):
        dump_json(self.path, self.data)
