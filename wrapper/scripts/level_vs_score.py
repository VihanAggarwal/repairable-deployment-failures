"""HARD FOR EVERY POLICY IS NOT UNREPAIRABLE. LIBERO-Plus grades each perturbed task 1 to 5 by how many reference VLAs solve
it (task_classification.json; 5 = none). Does that grade say what the wrapper can repair, and does the robot's own
measurement (training-view coverage of the self-calibrated reprojection) still rank repair INSIDE one grade, where the
benchmark's label is constant? Pure re-analysis of logged paired rows: no simulator, no GPU. POST HOC (say so in the paper).

Join is by task NAME (the ids in the classification file are 1-based per suite; a name join cannot be off by one).

    python level_vs_score.py --levels .../task_classification.json --set oft:libero_10 wraps/wrap_openvla_oft.jsonl --set ...
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows  # noqa: E402
from sweep_analysis import auroc, boot_ci  # noqa: E402


def norm(name):
    return re.sub(r"^[A-Z_]+_SCENE\d+_", "", name).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--set", nargs="+", action="append", required=True, metavar=("FAMILY:SUITE", "FILE"))
    a = ap.parse_args()
    cls = json.load(open(a.levels))
    units, joined, missed = [], 0, 0
    for s in a.set:
        fam, suite = s[0].split(":")
        level = {}
        for e in cls[suite]:
            if e.get("difficulty_level") is not None:
                level[e["name"].lower()] = int(e["difficulty_level"])
                level.setdefault(norm(e["name"]), int(e["difficulty_level"]))
        by = defaultdict(dict)
        for f in s[1:]:
            for r in load_rows(f):
                if r.get("rep", 0) == 0 and r.get("arm") in ("raw", "full") and r["arm"] not in by[r["task_id"]]:
                    by[r["task_id"]][r["arm"]] = r
        for tid, v in by.items():
            if "raw" not in v or "full" not in v:
                continue
            name = v["raw"]["name"]
            lv = level.get(name.lower(), level.get(norm(name)))
            if lv is None:
                missed += 1
                continue
            joined += 1
            units.append(dict(family=fam, suite=suite, task_id=tid, category=v["raw"]["category"], level=lv,
                              raw=bool(v["raw"]["success"]), full=bool(v["full"]["success"]),
                              coverage=v["full"].get("reproj_coverage")))
    print(f"paired tasks joined to a difficulty level: {joined} (no level or no name match: {missed})")

    for cat in ("robot_init", "camera"):
        print(f"\n===== {cat}: raw failures and what the wrapper repairs, by LIBERO-Plus level =====")
        print(" level  tasks  raw fails  repaired   raw succeeds  broken")
        for lv in (1, 2, 3, 4, 5):
            u = [x for x in units if x["category"] == cat and x["level"] == lv]
            f = [x for x in u if not x["raw"]]
            s = [x for x in u if x["raw"]]
            print(f"   {lv}    {len(u):>4}     {len(f):>4}     {sum(x['full'] for x in f):>4}         {len(s):>4}    {sum(not x['full'] for x in s):>4}")

    cam = [x for x in units if x["category"] == "camera" and not x["raw"] and x["coverage"] is not None]
    print(f"\n===== camera raw failures with a logged coverage: {len(cam)} =====")

    def line(label, u, score):
        y = [x["full"] for x in u]
        if len(u) >= 10 and 0 < sum(y) < len(y):
            sc = [score(x) for x in u]
            lo, hi = boot_ci(sc, y)
            print(f"  {label:<46} AUROC {auroc(sc, y):.3f} [{lo:.2f}, {hi:.2f}]  (n {len(u)}, repaired {sum(y)})")
        else:
            print(f"  {label:<46} too few units (n {len(u)}, repaired {sum(y)})")
    line("the benchmark's level (negated), all levels", cam, lambda x: -x["level"])
    line("self-measured coverage, all levels", cam, lambda x: x["coverage"])
    for lo_, hi_ in ((5, 5), (4, 5), (1, 3)):
        line(f"self-measured coverage, WITHIN levels {lo_} to {hi_}", [x for x in cam if lo_ <= x["level"] <= hi_], lambda x: x["coverage"])
    for fam in sorted({x["family"] for x in cam}):
        line(f"self-measured coverage, within level 5, {fam}", [x for x in cam if x["level"] == 5 and x["family"] == fam], lambda x: x["coverage"])
    lv = np.array([x["level"] for x in cam], float)
    cv = np.array([x["coverage"] for x in cam], float)
    if len(cam) > 5:
        from scipy.stats import spearmanr
        r = spearmanr(lv, cv)
        print(f"  Spearman(level, coverage) = {r.correlation:+.2f} (p = {r.pvalue:.3g}): how much of the level is just lost view")
    l5 = [x for x in cam if x["level"] == 5]
    if len(l5) >= 9:
        q = np.quantile([x["coverage"] for x in l5], [1 / 3, 2 / 3])
        for nm, sel in (("low", lambda c: c <= q[0]), ("middle", lambda c: q[0] < c <= q[1]), ("high", lambda c: c > q[1])):
            u = [x for x in l5 if sel(x["coverage"])]
            print(f"  level 5, {nm:<6} coverage tercile: repaired {sum(x['full'] for x in u)} of {len(u)}")


if __name__ == "__main__":
    main()
