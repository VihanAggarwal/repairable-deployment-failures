"""POST HOC (2026-09-24, round 3, F29): paired AUROC differences of self-measured coverage against the true shift
magnitude (translation cm, rotation deg), on the in-rollout camera raw failures of results/local/coverage_units.jsonl
(raw = False, coverage logged; the 98 units of Fig. envelope (b)). Positive class: the `full` arm succeeds (repaired).
AUROC = Mann-Whitney with ties 1/2 (sweep_analysis.auroc); shift predictors scored as minus the shift. Paired unit
bootstrap, B = 2000, seed 0, both AUROCs on the same resample, single-class resamples skipped.

    uv run --no-project --with numpy python wrapper/scripts/coverage_paired.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from sweep_analysis import auroc  # noqa: E402

U = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "coverage_units.jsonl")) if l.strip()]
G = [u for u in U if (not u["raw"]) and u["coverage"] is not None]
y = np.array([int(u["full"]) for u in G])
cov = np.array([u["coverage"] for u in G])
tr = -np.array([u["shift_cm"] for u in G])
ro = -np.array([u["shift_deg"] for u in G])
print(f"n {len(G)} (OFT {sum(u['policy'] == 'OpenVLA-OFT' for u in G)}), repaired {int(y.sum())}")
print("AUROC coverage %.3f  -translation %.3f  -rotation %.3f" % (auroc(cov, y), auroc(tr, y), auroc(ro, y)))
rng = np.random.default_rng(0)
dt, dr = [], []
for _ in range(2000):
    i = rng.integers(0, len(G), len(G))
    if len(set(y[i])) < 2:
        continue
    a = auroc(cov[i], y[i])
    dt.append(a - auroc(tr[i], y[i]))
    dr.append(a - auroc(ro[i], y[i]))
print("coverage - translation %+.3f [%+.2f, %+.2f]" % (auroc(cov, y) - auroc(tr, y), *np.percentile(dt, [2.5, 97.5])))
print("coverage - rotation    %+.3f [%+.2f, %+.2f]" % (auroc(cov, y) - auroc(ro, y), *np.percentile(dr, [2.5, 97.5])))
