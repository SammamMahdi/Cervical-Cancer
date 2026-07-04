"""GATE 1 - 8-leaf exclusion recount (pure CPU measurement, no model).

Enumerates HiCervix test.csv level_1 labels and counts, under BOTH:
  * the existing 5-class E0 crosswalk (LM.hicervix_map -> EXCLUDE when no SIPaKMeD home)
  * the NEW Bethesda-aligned 8-leaf head (LM.hicervix8_candidates -> set of leaves)
how many cells are representable (>=1 candidate) vs still excluded, before vs after.
"""
import os, csv as _csv
from collections import Counter, defaultdict

import config as C
import data.labelmap as LM

csv_path = os.path.join(C.HICERVIX_DIR, "test.csv")

rows = []
with open(csv_path, newline="", encoding="utf-8") as fh:
    for row in _csv.DictReader(fh):
        lvl1 = (row.get("level_1") or "").strip()
        key = lvl1 or (row.get("class_name") or "").strip()
        rows.append(key)

n = len(rows)
native = Counter(rows)

# BEFORE: existing 5-class E0 map. excluded iff e0 is EXCLUDE(None).
before_excl = 0
before_by_cls = defaultdict(lambda: [0, 0])  # cls -> [representable, excluded]
# AFTER: 8-leaf candidate map. excluded iff candidate set empty.
after_excl = 0
after_by_cls = defaultdict(lambda: [0, 0])

for key in rows:
    e0, _e1 = LM.hicervix_map(key)
    if e0 is None:
        before_excl += 1
        before_by_cls[key][1] += 1
    else:
        before_by_cls[key][0] += 1

    cand = LM.hicervix8_candidates(key)
    if not cand:
        after_excl += 1
        after_by_cls[key][1] += 1
    else:
        after_by_cls[key][0] += 1

print(f"n_total={n}")
print(f"native classes: {dict(native.most_common())}")
print()
print(f"BEFORE (5-class E0): excluded={before_excl}  excluded_fraction={before_excl/n:.4f}")
print(f"AFTER  (8-leaf head): excluded={after_excl}  excluded_fraction={after_excl/n:.4f}")
print()
print("per-native-class breakdown [class: n  before(repr/excl)  after(repr/excl)  candidate-leaves]:")
for cls, cnt in native.most_common():
    b = before_by_cls[cls]; a = after_by_cls[cls]
    leaves = sorted(LM.hicervix8_candidates(cls))
    print(f"  {cls:10s} n={cnt:5d}  before[repr={b[0]:5d} excl={b[1]:5d}]  "
          f"after[repr={a[0]:5d} excl={a[1]:5d}]  -> {leaves}")
print()
print(f"BETHESDA8_LEAVES = {LM.BETHESDA8_LEAVES}")
print(f"previously-excluded cells now representable = {before_excl - after_excl}")
