"""POST HOC (2026-09-26): does the stored training-view hole fill drive the camera repairs?

(1) Existing paired rows: `full` vs `full_nofill` (holes inpainted from the frame instead of filled from the stored
training view) on the same tasks and seeds, every file in plus/wraps that has both arms. (2) A dose argument from the
98 in-rollout camera failures: if pasting the stored training view drove repairs, low-coverage units (mostly fill) would
be repaired MORE; report repair rate by coverage.

    uv run --no-project --with numpy python wrapper/scripts/fill_ablation.py
"""
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from plus_common import load_shards  # noqa: E402

D = "C:/Users/vihan/Downloads/local_data/plus"


def two_sided(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


for fam, fn in (("OpenVLA-OFT", "wrap_openvla_oft.jsonl"), ("pi0.5", "wrap_pi05.jsonl")):
    by = {}
    for r in load_shards(f"{D}/wraps/{fn}"):
        if r.get("category") == "camera" and r.get("rep", 0) == 0:
            by.setdefault(r["task_id"], {}).setdefault(r["arm"], r)
    t = [i for i, v in by.items() if "full" in v and "full_nofill" in v]
    if not t:
        continue
    f = np.array([bool(by[i]["full"]["success"]) for i in t])
    n = np.array([bool(by[i]["full_nofill"]["success"]) for i in t])
    raw = [bool(by[i]["raw"]["success"]) for i in t if "raw" in by[i]]
    b, c = int((f & ~n).sum()), int((n & ~f).sum())
    print("%-12s paired tasks %2d | raw %s | full %d | full_nofill %d | fill-only wins %d, nofill-only wins %d, exact McNemar p %.3f"
          % (fam, len(t), sum(raw) if raw else "-", f.sum(), n.sum(), b, c, two_sided(b, c)))
    print("   tasks:", sorted(t))

U = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "coverage_units.jsonl")) if l.strip()]
F = [u for u in U if not u["raw"] and u["coverage"] is not None]
c = np.array([u["coverage"] for u in F])
y = np.array([bool(u["full"]) for u in F])
print("\ncamera raw failures with logged coverage: %d" % len(F))
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
    m = (c >= lo) & (c < hi)
    if m.any():
        print("   coverage %.2f-%.2f (fill share %.0f-%.0f%%): repaired %d of %d (%.0f%%)" % (
            lo, min(hi, 1), 100 * (1 - min(hi, 1)), 100 * (1 - lo), y[m].sum(), m.sum(), 100 * y[m].mean()))
