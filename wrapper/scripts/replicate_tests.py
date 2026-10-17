"""POST HOC (2026-09-24, reviewer round 1): task-level tests for the replicated pi0.5 camera and light runs.

The replication (reps 0, 1, 2: results/drive_vihan/wraps, wraps_rep1, wraps_rep2 /wrap_pi05.jsonl, arms raw / full) was
run after the single-rollout camera test was inconclusive. Episodes of one task share the task, so the episode-level
McNemar treats correlated pairs as independent. Here the unit is the TASK: d_t = sum over reps of
(wrapped success - raw success), in {-3..3}. Tests on the d_t of the tasks with at least one rep:
  sign test     exact two-sided binomial on #(d_t > 0) vs #(d_t < 0), ties dropped
  sign-flip     exact permutation of sum d_t over all 2^m sign assignments of the nonzero d_t, two-sided
  Wilcoxon      signed-rank, exact when scipy allows (zero-differences dropped)
and, for comparison, the episode McNemar over all paired (task, rep) episodes.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/replicate_tests.py
"""
import itertools
import json
import os
from collections import defaultdict
from math import comb

import numpy as np
from scipy.stats import binomtest, wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, "results", "drive_vihan")
FILES = [os.path.join(R, d, "wrap_pi05.jsonl") for d in ("wraps", "wraps_rep1", "wraps_rep2")]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    for cat in ("camera", "light"):
        ep = defaultdict(dict)
        for i, f in enumerate(FILES):
            for l in open(f, encoding="utf-8"):
                if not l.strip():
                    continue
                r = json.loads(l)
                if r["category"] != cat or r["arm"] not in ("raw", "full"):
                    continue
                rep = r.get("rep", i)
                ep[(r["task_id"], rep)].setdefault(r["arm"], int(bool(r["success"])))
        pairs = {k: v for k, v in ep.items() if "raw" in v and "full" in v}
        reps = defaultdict(int)
        for (t, rep) in pairs:
            reps[rep] += 1
        rep_set = sorted(reps)
        tasks_all = sorted({t for t, _ in pairs})
        tasks = [t for t in tasks_all if all((t, r) in pairs for r in rep_set)]         # complete tasks only
        d = np.array([sum(pairs[(t, r)]["full"] - pairs[(t, r)]["raw"] for r in rep_set) for t in tasks])
        up = sum(1 for (t, r), v in pairs.items() if t in tasks and v["full"] > v["raw"])
        dn = sum(1 for (t, r), v in pairs.items() if t in tasks and v["full"] < v["raw"])
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        nz = d[d != 0]
        obs = abs(nz.sum())
        perm = np.mean([abs(np.dot(s, np.abs(nz))) >= obs - 1e-9 for s in itertools.product((-1, 1), repeat=len(nz))]) if len(nz) <= 22 else float("nan")
        try:
            w = wilcoxon(nz).pvalue if len(nz) else float("nan")
        except ValueError:
            w = float("nan")
        # single-rollout (rep 0) episode McNemar for reference
        up0 = sum(1 for (t, r), v in pairs.items() if r == rep_set[0] and v["full"] > v["raw"])
        dn0 = sum(1 for (t, r), v in pairs.items() if r == rep_set[0] and v["full"] < v["raw"])
        print(f"===== pi0.5 {cat}: reps {dict(reps)}, complete tasks {len(tasks)} of {len(tasks_all)}, episodes {len(tasks) * len(rep_set)}")
        print(f"  rep 0 only: repaired {up0}, broken {dn0}, exact McNemar p = {mcnemar(up0, dn0):.3f}")
        print(f"  episodes (all reps): repaired {up}, broken {dn}, exact McNemar p = {mcnemar(up, dn):.3f}")
        print(f"  task-level d_t: {sorted(d.tolist())}; positive {pos}, negative {neg}, zero {len(d) - pos - neg}; mean {d.mean():+.3f}")
        print(f"  sign test p = {binomtest(pos, pos + neg).pvalue:.3f}   exact sign-flip permutation p = {perm:.3f}   Wilcoxon p = {w:.3f}")


if __name__ == "__main__":
    main()
