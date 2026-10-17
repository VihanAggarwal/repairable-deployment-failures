"""POST HOC (2026-09-24, round 3, item F11), dagger: the action-level tolerance restated, and an OUTCOME-level check.

(1) K = 1 null. tolerance_probe.probe_unit logs `floor` = d(reference queried again with the same seed, reference). If the
    floors spread, their upper quantile would give a K = 1 threshold from an unperturbed-replicate null. Printed below.
(2) Knees of the rollout-free dose response (results/drive_vihan/tolerance/tolerance_*.jsonl; 20 runs = 10 base tasks x
    orbit 10 / 20 deg per family, coverage >= 0.68): per run the median over the 3 camera axes of d at each dose, then
    the median over runs; knee(theta) = smallest dose whose median exceeds theta, for theta in {0.05, 0.10, 0.20}.
(3) d at 1 / 4 / 16 cm on the unit drawn in the tolerance pictures (figs/renders/tolerance.npz, matched to its run by
    the logged true camera shift).
(4) Outcome tolerance on the in-rollout camera raw failures (results/local/coverage_units.jsonl, raw = False with the
    calibration error logged; every one has cal_applied): repair rate of arm `full` per calibration-error bin
    [0, 1), [1, 2), [2, 5), [5, 20), [20, inf) cm with Wilson 95% intervals, and a monotone non-increasing (isotonic,
    pool-adjacent-violators, unit weights) fit of repaired on error.

    uv run --no-project --with numpy python wrapper/scripts/outcome_tolerance.py
"""
import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL = os.path.join(REPO, "results", "drive_vihan", "tolerance")
FAM = {"openvla_oft": "OFT", "pi05": "pi0.5"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    m, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h


def pava_decreasing(y):
    """Isotonic non-increasing fit (unit weights) of y given in increasing order of x."""
    blocks = [[v, 1] for v in (-np.asarray(y, float))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            v, w = out.pop()
            out[-1][0] += v; out[-1][1] += w
    return -np.concatenate([[s / w] * w for s, w in out])


rows = {}
for p in sorted(glob.glob(os.path.join(TOL, "tolerance_*.jsonl"))):
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            rows.setdefault(FAM.get(r["policy"], r["policy"]), []).append(r)

print("(1) K = 1 null (same frame, same seed, queried twice):")
for f, rs in rows.items():
    fl = np.array([r["floor"] for r in rs])
    print(f"    {f}: {len(rs)} runs on {len({r['task_id'] for r in rs})} tasks, conds {sorted({r['cond'] for r in rs})}, "
          f"coverage >= {min(r['coverage'] for r in rs):.3f}; floor min {fl.min():.4f} median {np.median(fl):.4f} max {fl.max():.4f}")
print("    -> the replicate query is deterministic (every floor is 0): the rows hold no unperturbed-replicate spread, so no K = 1"
      " null threshold can be derived from them.")


def per_run(r, kind, dose):
    return float(np.median([x["distance"] for x in r["doses"] if x["kind"] == kind and x["dose"] == dose]))


print("\n(2) knees: median over runs of the per-run median over axes")
for f, rs in rows.items():
    for kind, grid in (("cm", (0.5, 1, 2, 4, 8, 16)), ("deg", (0.5, 1, 2, 4, 8))):
        med = [float(np.median([per_run(r, kind, d) for r in rs])) for d in grid]
        knees = {th: next((d for d, m in zip(grid, med) if m > th), None) for th in (0.05, 0.10, 0.20)}
        print(f"    {f:5s} {kind:3s} medians {[round(m, 3) for m in med]}  knee at 0.05/0.10/0.20: {knees}")

print("\n(3) the pictured unit (tolerance.npz)")
z = np.load(os.path.join(REPO, "wrapper", "paper", "iclr", "figs", "renders", "tolerance.npz"), allow_pickle=True)
sh = np.asarray(z["shift"], float)
for f, rs in rows.items():
    best = min(rs, key=lambda r: float(np.abs(np.asarray(r["shift_cm_deg"]) - sh).sum()))
    gap = float(np.abs(np.asarray(best["shift_cm_deg"]) - sh).sum())
    print(f"    {f}: matched task {best['task_id']} {best['cond']} (shift {np.round(best['shift_cm_deg'], 2).tolist()} vs npz "
          f"{np.round(sh, 2).tolist()}, |gap| {gap:.3f}); d (median over axes) at 1/4/16 cm: "
          f"{[round(per_run(best, 'cm', d), 3) for d in (1, 4, 16)]}")

print("\n(4) outcome tolerance, in-rollout camera raw failures")
U = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "coverage_units.jsonl")) if l.strip()]
F = [u for u in U if not u["raw"] and u.get("calib_err_cm") is not None]
print(f"    units {len(F)} ({sum(u['policy'] == 'OpenVLA-OFT' for u in F)} OFT, {sum(u['policy'] == 'pi0.5' for u in F)} pi0.5); "
      f"cal_applied on all: {all(u['cal_applied'] for u in F)}; repaired {sum(bool(u['full']) for u in F)}")
e = np.array([u["calib_err_cm"] for u in F]); y = np.array([int(bool(u["full"])) for u in F])
o = np.argsort(e, kind="mergesort")
iso = pava_decreasing(y[o])
edges = [0, 1, 2, 5, 20, np.inf]
for a, b in zip(edges[:-1], edges[1:]):
    m = (e >= a) & (e < b)
    k, n = int(y[m].sum()), int(m.sum())
    lo, hi = wilson(k, n)
    mi = (e[o] >= a) & (e[o] < b)
    print(f"    [{a:g}, {b:g}) cm: {k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]   isotonic fit over the bin {iso[mi].min():.2f} to {iso[mi].max():.2f}")
steps = sorted({round(float(v), 3) for v in iso}, reverse=True)
print(f"    isotonic levels (non-increasing in error): {steps}; breakpoints (cm): "
      f"{[round(float(e[o][i]), 2) for i in range(1, len(iso)) if iso[i] != iso[i - 1]]}")
