"""Paired comparison of every arm in a wrap_run file, per category, on the tasks ALL the listed arms have run.

    python arms_report.py --rows <wrap_*.jsonl> [--arms raw,full,tta,generic,full_homog,full_nofill] [--ref raw]

For each arm: success, and against the reference arm the tasks recovered (ref fails, arm succeeds), the tasks broken
(ref succeeds, arm fails) and the exact two-sided McNemar p value.
"""
import argparse
import json
from collections import defaultdict
from math import comb

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--arms", default="")
ap.add_argument("--ref", default="raw")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.rows) if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[(r["category"], r["arm"])][r["task_id"]] = bool(r["success"])          # the last row of a (task, arm) wins
cats = sorted({c for c, _ in by})
want = [x for x in a.arms.split(",") if x]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


for cat in cats:
    arms = [arm for (c, arm) in by if c == cat and (not want or arm in want)]
    if a.ref not in arms or len(arms) < 2:
        continue
    arms = [a.ref] + sorted(x for x in arms if x != a.ref)
    common = set.intersection(*[set(by[(cat, arm)]) for arm in arms])
    if not common:
        print(f"\n{cat}: the arms {arms} share no task yet")
        continue
    print(f"\n{cat}: {len(common)} tasks run by every arm")
    print(f"  {'arm':<22}{'success':>9}{'recovered':>11}{'broken':>8}{'p vs ' + a.ref:>12}")
    ref = by[(cat, a.ref)]
    for arm in arms:
        v = by[(cat, arm)]
        s = sum(v[t] for t in common) / len(common)
        rec = sum((not ref[t]) and v[t] for t in common)
        brk = sum(ref[t] and (not v[t]) for t in common)
        tail = "" if arm == a.ref else f"{rec:>8}/{sum(not ref[t] for t in common):<3}{brk:>5}/{sum(ref[t] for t in common):<3}{mcnemar(rec, brk):>10.3f}"
        print(f"  {arm:<22}{s:>9.3f}{tail}")
