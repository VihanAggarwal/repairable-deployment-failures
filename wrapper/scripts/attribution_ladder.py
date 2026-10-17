"""WHERE THE DEFICIT LIVES: a counterfactual ladder on camera tasks (the title claim as a table).

The same task and seed under nested arms, each removing one more cause:
  raw      the frozen policy on the shifted camera
  full     + the wrapper with the pose the robot MEASURED on its own body
  true     + the simulator's TRUE camera pose in the same reprojection      (cam_true_ref / full_truepose): removes measurement error
  oracle   + a perfect render from the training camera                        (oracle / full_oracle): removes the inversion loss
A raw failure is attributed to the FIRST rung that repairs it:
  recovered by measurement | not identified (only the true pose repairs it) | not invertible (only the perfect render does:
  the view is lost) | policy (fails even on a perfect training-view frame: not the camera at all).
The upper rungs are usually run only where `full` failed, so a missing rung above a success is not needed.
Single rollouts are noisy (about 15% flip with nothing applied): units that succeed on a lower rung and fail on a higher one
are COUNTED and printed as that noise rate, never hidden.

    python attribution_ladder.py --set libero_10 rows.jsonl rows_oft.jsonl --set object wraps_libero_object/wrap_openvla_oft.jsonl wraps_ladder_libero_object/wrap_openvla_oft.jsonl
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows  # noqa: E402
from sweep_analysis import auroc, boot_ci  # noqa: E402

RUNG = {"raw": "raw", "full": "full", "cam_true_ref": "true", "full_truepose": "true", "oracle": "oracle", "full_oracle": "oracle"}
ORDER = ["raw", "full", "true", "oracle"]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def units(files):
    u = defaultdict(dict)
    for f in files:
        for r in load_rows(f):
            if r.get("category") != "camera" or r.get("rep", 0) != 0:
                continue
            rung = RUNG.get(r.get("arm", "raw"))                 # baseline files carry no arm: they are the raw policy
            if rung and rung not in u[r["task_id"]]:
                u[r["task_id"]][rung] = r
    return u


def bucket(v):
    ok = lambda k: bool(v[k]["success"])
    if ok("full"):
        return "recovered by measurement"
    if "true" in v and ok("true"):
        return "not identified"
    if "oracle" not in v:
        return "upper rungs not run"
    return "not invertible" if ok("oracle") else "policy"


def report(name, u):
    base = {t: v for t, v in u.items() if "raw" in v and "full" in v}
    print(f"\n===== {name} =====  {len(base)} camera tasks with raw and full")
    if not base:
        return Counter(), []
    for k in ORDER:
        have = [v for v in base.values() if k in v]
        print(f"  {k:<7} success {sum(bool(v[k]['success']) for v in have):>3} / {len(have)}")
    fails = {t: v for t, v in base.items() if not v["raw"]["success"]}
    b = Counter(bucket(v) for v in fails.values())
    print(f"  raw failures {len(fails)}: " + ", ".join(f"{k} {b[k]}" for k in
          ("recovered by measurement", "not identified", "not invertible", "policy", "upper rungs not run") if b[k]))
    for lo, hi in (("full", "true"), ("true", "oracle"), ("full", "oracle")):
        both = [v for v in base.values() if lo in v and hi in v]
        up = sum(bool(v[hi]["success"]) and not v[lo]["success"] for v in both)
        down = sum(bool(v[lo]["success"]) and not v[hi]["success"] for v in both)
        if both:
            print(f"  {hi} vs {lo}: fixes {up}, breaks {down} on {len(both)} paired tasks, exact McNemar p = {mcnemar(up, down):.3f}")
    cov = lambda v: v["full"].get("reproj_coverage")
    for k in ("recovered by measurement", "not identified", "not invertible", "policy"):
        c = [cov(v) for v in fails.values() if bucket(v) == k and cov(v) is not None]
        e = [v["full"]["cam_est_err_cm_deg"][0] for v in fails.values() if bucket(v) == k and v["full"].get("cam_est_err_cm_deg")]
        if c or e:
            print(f"    {k:<26} coverage median {np.median(c):.2f} [{min(c):.2f}, {max(c):.2f}] (n {len(c)})" if c else f"    {k:<26}",
                  f"| calibration error median {np.median(e):.1f} cm (n {len(e)})" if e else "")
        ids = sorted(t for t, v in fails.items() if bucket(v) == k)
        if ids and k != "recovered by measurement":
            print(f"      tasks: {ids}")
    return b, list(fails.values())


def baseline_row(fails):
    """Which single number, known BEFORE the rollout, says whether the wrapper repairs a raw camera failure?"""
    y = [bool(v["full"]["success"]) for v in fails]
    feats = {
        "shift distance (cm), negated": lambda v: -v["full"]["cam_true_shift_cm_deg"][0] if v["full"].get("cam_true_shift_cm_deg") else None,
        "shift rotation (deg), negated": lambda v: -v["full"]["cam_true_shift_cm_deg"][1] if v["full"].get("cam_true_shift_cm_deg") else None,
        "calibration error (cm), negated": lambda v: -v["full"]["cam_est_err_cm_deg"][0] if v["full"].get("cam_est_err_cm_deg") else None,
        "self-measured coverage": lambda v: v["full"].get("reproj_coverage"),
    }
    print(f"\n===== pooled: what predicts repair of a raw camera failure ({len(y)} units, {sum(y)} repaired) =====")
    for k, fn in feats.items():
        x = [(fn(v), yy) for v, yy in zip(fails, y) if fn(v) is not None]
        if len(x) >= 10 and 0 < sum(b for _, b in x) < len(x):
            s, l = [a for a, _ in x], [b for _, b in x]
            lo, hi = boot_ci(s, l)
            print(f"  {k:<34} AUROC {auroc(s, l):.3f} [{lo:.2f}, {hi:.2f}]  (n {len(x)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", action="append", required=True, metavar=("NAME", "FILE"))
    a = ap.parse_args()
    total, every = Counter(), []
    for s in a.set:
        b, f = report(s[0], units(s[1:]))
        total += b
        every += f
    print(f"\n===== pooled attribution of {sum(total.values())} raw camera failures =====")
    for k, n in total.most_common():
        print(f"  {k:<26} {n:>3}  ({100 * n / max(sum(total.values()), 1):.0f}%)")
    baseline_row(every)


if __name__ == "__main__":
    main()
