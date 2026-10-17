"""POST HOC (2026-09-24, reviewer round 2, item P05): does self-measured coverage flag the lost-view bucket in advance?

Join results/local/coverage_units.jsonl (self-measured coverage c of the `full` row, one row per LIBERO-Plus camera task)
to the attribution-ladder bucket of each raw failure (attribution_ladder.bucket, the sets of ladder_counts.py, rep 0).
Positive class: render-only ("not invertible": only a perfect render repairs it). Negative class: repaired by the measured
pose or only by the true pose. Units in 'nothing' / 'upper rungs not run' and units without a logged coverage are left out.
Statistic: AUROC of the score s = 1 - c (lower coverage = predicted lost view), ties counted 1/2.
Intervals (95%, percentile, B = 4000): unit bootstrap (resample units) and task-cluster bootstrap (resample base tasks,
the LIBERO scene+instruction with the camera suffix `_view_...` removed, and keep every unit of a drawn task).

    uv run --no-project --with numpy --with scipy python wrapper/scripts/bucket_coverage_auroc.py
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from attribution_ladder import bucket, units  # noqa: E402
from ladder_counts import SETS  # noqa: E402  (prints the ladder tables on import)

SRC = {"OFT libero_10 (first 30)": "OFT libero_10 first 30", "OFT libero_10 (30 more)": "OFT libero_10 30 more",
       "OFT libero_object": "OFT libero_object", "OFT libero_goal": "OFT libero_goal", "OFT libero_spatial": "OFT libero_spatial",
       "pi0.5 libero_10": "pi0.5 libero_10"}


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return np.nan
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def base(name):
    return re.sub(r"_view_.*$", "", name or "")


def main():
    cov = {}
    for l in open(os.path.join(REPO, "results", "local", "coverage_units.jsonl")):
        if l.strip():
            r = json.loads(l)
            cov[(SRC[r["src"]], r["task_id"])] = r
    rows = []
    for name, files in SETS.items():
        for t, v in units(files).items():
            if "raw" not in v or "full" not in v or v["raw"]["success"]:
                continue
            b = bucket(v)
            if b not in ("recovered by measurement", "not identified", "not invertible"):
                continue
            c = cov.get((name, t), {}).get("coverage")
            if c is None:
                continue
            rows.append(dict(set=name, fam="pi0.5" if name.startswith("pi0.5") else "OFT", task=t, y=int(b == "not invertible"),
                             c=float(c), bucket=b, cl=base(v["raw"].get("name") or v["full"].get("name"))))
    rng = np.random.default_rng(0)
    B = 4000
    for fam in ("OFT", "pi0.5", "pooled"):
        R = [r for r in rows if fam == "pooled" or r["fam"] == fam]
        s = np.array([1 - r["c"] for r in R]); y = np.array([r["y"] for r in R])
        a = auroc(s, y)
        ub = [auroc(s[i], y[i]) for i in (rng.integers(0, len(R), len(R)) for _ in range(B))]
        cls = defaultdict(list)
        for i, r in enumerate(R):
            cls[r["cl"]].append(i)
        keys = list(cls)
        cb = []
        for _ in range(B):
            idx = np.concatenate([cls[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
            cb.append(auroc(s[idx], y[idx]))
        ub, cb = np.array(ub), np.array(cb)
        med = lambda b: np.median([r["c"] for r in R if r["bucket"] == b]) if any(r["bucket"] == b for r in R) else np.nan
        print(f"{fam:7s} n = {len(R)} (render-only {int(y.sum())}, measured {sum(r['bucket'] == 'recovered by measurement' for r in R)}, "
              f"true-pose {sum(r['bucket'] == 'not identified' for r in R)}; {len(keys)} base tasks)  AUROC(1-c) = {a:.3f}  "
              f"unit 95% [{np.nanpercentile(ub, 2.5):.3f}, {np.nanpercentile(ub, 97.5):.3f}]  "
              f"task-cluster 95% [{np.nanpercentile(cb, 2.5):.3f}, {np.nanpercentile(cb, 97.5):.3f}]  "
              f"median c: render-only {med('not invertible'):.2f}, measured {med('recovered by measurement'):.2f}, true-pose {med('not identified'):.2f}")


if __name__ == "__main__":
    main()
