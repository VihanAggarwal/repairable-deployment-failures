"""POST HOC (2026-09-24, round 3, item F34): the Prop. 1 rank check on DISTINCT runs of results/local/measurability.jsonl.

The file has 80 rows = 8 tasks x 2 amplitudes (scale 1.0, 0.5) x K in {1, 2, 3, 5, 7}. With K = 1 the single probe is the
rest pose, so the probe amplitude does not enter: the scale-0.5 K = 1 row repeats the scale-1.0 K = 1 row exactly (same
sigma_min and error on all 8 tasks, checked below). Pooling all 80 rows counts those 8 runs twice. Here the scale-0.5
K = 1 rows are dropped, leaving 72 distinct runs.

Spearman rho_s(sigma_min, err_cm) with average ranks for ties; within-task permutation p: errors are permuted among the
runs of the same task (tasks share a camera shift, so runs are exchangeable only within a task), B = 4000, seed 0,
two-sided on |rho_s|, p = (1 + #{|rho_b| >= |rho_obs|}) / (1 + B).

    uv run --no-project --with numpy --with scipy python wrapper/scripts/measurability_dedupe.py
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "measurability.jsonl")) if l.strip()]


def perm_p(rows, B=4000, seed=0):
    s = np.array([r["sigma_min"] for r in rows]); e = np.array([r["err_cm"] for r in rows])
    t = np.array([r["task_id"] for r in rows])
    obs = spearmanr(s, e).statistic
    rng = np.random.default_rng(seed)
    idx = {k: np.where(t == k)[0] for k in np.unique(t)}
    hits = 0
    for _ in range(B):
        ep = e.copy()
        for ii in idx.values():
            ep[ii] = e[rng.permutation(ii)]
        hits += abs(spearmanr(s, ep).statistic) >= abs(obs) - 1e-12
    return float(obs), (1 + hits) / (1 + B)


k1 = {(r["task_id"], r["scale"]): r for r in R if r["n_poses"] == 1}
same = all(abs(k1[(t, 0.5)]["sigma_min"] - k1[(t, 1.0)]["sigma_min"]) < 1e-9 and abs(k1[(t, 0.5)]["err_cm"] - k1[(t, 1.0)]["err_cm"]) < 1e-9
           for t in {r["task_id"] for r in R})
D = [r for r in R if not (r["n_poses"] == 1 and r["scale"] == 0.5)]
print(f"rows {len(R)}; scale-0.5 K=1 rows identical to scale-1.0 K=1 on every task: {same}; distinct runs {len(D)}")
for name, rows in (("all 80 rows (as in the draft)", R), ("72 distinct runs", D),
                   ("full amplitude, 40", [r for r in R if r["scale"] == 1.0]),
                   ("half amplitude, K >= 2, 32", [r for r in D if r["scale"] == 0.5])):
    rho, p = perm_p(rows)
    print(f"  {name:<32} n {len(rows)}  Spearman rho_s = {rho:+.3f}  within-task permutation p = {p:.4f}")
