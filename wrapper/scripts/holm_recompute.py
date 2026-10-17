"""POST HOC (2026-09-24, reviewer round 2, item P09): Holm step-down over the m = 15 secondary tests of Table tab:holm, with
the two pi0.5 3-replicate rows replaced by the task-level exact sign-flip p (replicate_tests.py). McNemar p recomputed
from the repaired / broken counts of the table (exact two-sided); DROID p recomputed from the DROID rows (paired Wilcoxon,
zero-shot vs VLM masks). Holm: sort p ascending, adj_(r) = max_{j <= r} min(1, (m - j + 1) p_(j)) (monotone).

    uv run --no-project --with numpy --with scipy python wrapper/scripts/holm_recompute.py
"""
import itertools
import json
import os
from math import comb

import numpy as np
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, "results", "drive_vihan")


def mc(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def signflip(d):
    nz = np.abs([x for x in d if x])
    obs = abs(sum(d))
    return float(np.mean([abs(np.dot(s, nz)) >= obs - 1e-9 for s in itertools.product((-1, 1), repeat=len(nz))]))


ld = lambda p: {json.loads(l)["file"]: json.loads(l) for l in open(p) if l.strip()}
new, old = ld(os.path.join(R, "droid_selfcal_vlm", "selfcal_results.jsonl")), ld(os.path.join(R, "droid_selfcal", "selfcal_results.jsonl"))
f = [k for k in new if k in old]
droid = wilcoxon([old[k]["pos_err_cm"] for k in f], [new[k]["pos_err_cm"] for k in f]).pvalue
# task-level d_t, as printed by replicate_tests.py
cam_d = [-3, -3, -1] + [0] * 15 + [1] * 6 + [2] * 4 + [3] * 2
light_d = [-3, -3, -2, -1, -1, -1, 0, 0, 0, 1]
T = [("OFT joint, three untuned suites", mc(47, 1)), ("OFT object joint", mc(16, 0)), ("OFT goal joint", mc(16, 0)),
     ("pi0.5 joint", mc(16, 0)), ("OFT libero-10 joint", mc(18, 1)), ("OFT libero-10 camera", mc(20, 3)),
     ("OFT spatial joint", mc(15, 1)), ("OFT object camera", mc(26, 6)), ("motion-free vs raw, pi0.5 sweep (episode)", mc(21, 7)),
     (f"DROID masks (Wilcoxon, {len(f)} paired)", droid), ("pi0.5 light, 3 reps (task sign-flip)", signflip(light_d)),
     ("pi0.5 camera, 3 reps (task sign-flip)", signflip(cam_d)), ("OFT goal camera", mc(12, 5)),
     ("pi0.5 camera, one rollout", mc(7, 2)), ("OFT spatial camera", mc(3, 4))]
# POST HOC (2026-09-24, round 3, item F25): the motion-free vs raw row enters the family at the TASK level, like the two
# pi0.5 3-replicate rows: exact sign-flip over the per-task sums D_i (lazy - raw) of the 170-unit sweep, as printed by
# motionfree_tests.py part 2 (n0 = 7 tasks with D_i != 0, p = 6/128). The episode-level family is printed for reference.
lazy_raw_d = [0, -1, 0, 1, 3, 0, 3, 2, 2, 4]


def holm_print(T, title):
    m = len(T)
    order = sorted(range(m), key=lambda i: T[i][1])
    adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * T[i][1]))
        adj[i] = run
    print(f"{title}: m = {m}")
    for i in order:
        print(f"  {T[i][0]:45s} p = {T[i][1]:.3g}   Holm p = {adj[i]:.3g}")


holm_print(T, "EPISODE-LEVEL motion-free row (reference)")
T2 = [(("motion-free vs raw, pi0.5 sweep (task sign-flip)", signflip(lazy_raw_d)) if n.startswith("motion-free") else (n, p))
      for n, p in T]
holm_print(T2, "TASK-LEVEL motion-free row (F25)")
