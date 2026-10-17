"""POST HOC (2026-09-24, reviewer round 1): the arithmetic of the attribution ladder (Table 4), per rung.

Rungs a_0..a_3 = raw, full (measured pose), true (cam_true_ref: true pose, training-view fill), oracle (perfect render),
same task, seed, measurement motion and fill across rungs (attribution_ladder.units, rep 0, camera tasks).
S_a(u) in {0,1} is the success of unit u under rung a.
  attr(u)          = min{ j >= 1 : S_{a_j}(u) = 1 } for a raw failure (S_{a_0}(u) = 0); 'nothing' if no rung repairs it
  monotone(u)      : S_{a_j}(u) <= S_{a_k}(u) for all j < k among the rungs run on u
  broken at a_j    : raw successes (S_{a_0} = 1) with S_{a_j} = 0 (only where a_j was run on a raw success)
  first repaired   : raw failures with attr(u) = j
Under monotonicity the attr buckets partition the raw failures and success(a_j) = raw + sum_{i <= j} first_repaired(i)
- broken(a_j); a non-monotone unit makes its bucket a lower bound on the rung that repairs it.
Coverage = reproj_coverage of the full row.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/ladder_counts.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from attribution_ladder import ORDER, bucket, units  # noqa: E402

R = os.path.join(REPO, "results", "drive_vihan")
SETS = {
    "OFT libero_10 first 30": [f"{R}/wrapper_oft/rows.jsonl", f"{R}/rows_oft.jsonl"],
    "OFT libero_10 30 more": [f"{R}/wraps/wrap_openvla_oft.jsonl", f"{R}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl"],
    "OFT libero_object": [f"{R}/wraps_libero_object/wrap_openvla_oft.jsonl", f"{R}/wraps_ladder_libero_object/wrap_openvla_oft.jsonl"],
    "OFT libero_goal": [f"{R}/wraps_libero_goal/wrap_openvla_oft.jsonl", f"{R}/wraps_ladder_libero_goal/wrap_openvla_oft.jsonl"],
    "OFT libero_spatial": [f"{R}/wraps_libero_spatial/wrap_openvla_oft.jsonl", f"{R}/wraps_ladder_libero_spatial/wrap_openvla_oft.jsonl"],
    "pi0.5 libero_10": [f"{R}/wraps/wrap_pi05.jsonl", f"{R}/wraps_ladder/wrap_pi05.jsonl"],
}
LAB = {"recovered by measurement": "full", "not identified": "true", "not invertible": "oracle", "policy": "nothing",
       "upper rungs not run": "not run"}


def summarise(name, u):
    u = {t: v for t, v in u.items() if "raw" in v and "full" in v}
    raw_ok = [t for t, v in u.items() if v["raw"]["success"]]
    fails = [t for t, v in u.items() if not v["raw"]["success"]]
    print(f"== {name}: {len(u)} tasks, raw successes {len(raw_ok)}, raw failures {len(fails)}")
    for rung in ORDER[1:]:
        ran = [t for t in u if rung in u[t]]
        succ = sum(bool(u[t][rung]["success"]) for t in ran)
        ran_ok = [t for t in raw_ok if rung in u[t]]
        broken = sum(not u[t][rung]["success"] for t in ran_ok)
        print(f"  {rung:7s} run on {len(ran):3d} tasks: success {succ}; run on {len(ran_ok)} raw successes, broken {broken}")
    first = {}
    cov = {}
    for t in fails:
        b = LAB[bucket(u[t])]
        first[b] = first.get(b, 0) + 1
        c = u[t]["full"].get("reproj_coverage")
        if c is not None:
            cov.setdefault(b, []).append(float(c))
    print("  raw failures first repaired at:", first)
    print("  coverage median per bucket:", {b: (round(float(np.median(v)), 2), len(v)) for b, v in cov.items()})
    nonmono = []
    for t, v in u.items():
        s = [int(bool(v[k]["success"])) for k in ORDER if k in v]
        if any(s[i] > s[j] for i in range(len(s)) for j in range(i + 1, len(s))):
            nonmono.append((t, s))
    print(f"  non-monotone units: {len(nonmono)} of {len(u)} (of them raw failures: {sum(1 for t, _ in nonmono if t in fails)})",
          nonmono[:8])
    return first


pooled = {}
for name, files in SETS.items():
    f = summarise(name, units(files))
    if name.startswith("OFT"):
        for k, v in f.items():
            pooled[k] = pooled.get(k, 0) + v
print("== pooled OFT raw failures, first repaired at:", pooled, "total", sum(pooled.values()))
