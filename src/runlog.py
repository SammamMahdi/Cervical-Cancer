"""Timestamped, crash-safe JSON run logging + small JSON helpers (new pipeline).

Same contract as the Legacy_src logger: results/logs/run_<timestamp>.json rewritten atomically after
every epoch so an interruption never loses metrics, and every figure can be rebuilt from JSON.
"""
import os, json
from datetime import datetime


def now_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_jsonable(o):
    import numpy as np
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    return str(o)


def dump_json(path, obj):
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
    def __init__(self, log_dir, run_id, config):
        self.path = os.path.join(log_dir, f"run_{run_id}.json")
        self.data = {"run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"),
                     "ended": None, "config": config, "stages": {}, "events": [], "final": None}
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
