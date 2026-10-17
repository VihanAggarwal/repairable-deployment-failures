"""E-optimal probe selection (vendor/local_eoptimal.py rows): default 3 poses vs sigma_min-selected 3 vs random 3.
Paired one-sided sign test and Wilcoxon on calibration error; within-2-cm counts; sigma_min at the true camera.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/eoptimal_analysis.py
"""
import json
import os
from math import comb

import numpy as np
from scipy.stats import wilcoxon

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "local_eoptimal.jsonl")
R = [json.loads(l) for l in open(P) if l.strip()]
e = {k: np.array([r[f"err_{k}"][0] for r in R]) for k in ("default", "eopt", "rand")}
s = {k: np.array([r[f"smin_true_{k}"] for r in R]) for k in ("default", "eopt", "rand")}
print(f"{len(R)} camera tasks")
for k in e:
    print(f"  {k:8s} median err {np.median(e[k]):6.2f} cm | within 2 cm {int((e[k] <= 2).sum()):2d} | within 1 cm "
          f"{int((e[k] <= 1).sum()):2d} | median sigma_min(true) {np.median(s[k]):.2f}")
for base in ("default", "rand"):
    d = e[base] - e["eopt"]
    w, l = int((d > 1e-6).sum()), int((d < -1e-6).sum())
    n = w + l
    p = sum(comb(n, i) for i in range(w, n + 1)) / 2 ** n if n else 1.0
    pw = wilcoxon(e[base], e["eopt"], alternative="greater").pvalue if n else 1.0
    print(f"  eopt better than {base}: {w} tasks, worse {l}; sign test one-sided p {p:.4f}; Wilcoxon p {pw:.4f}")
big = np.array([r["true_shift"][0] > 1 for r in R])
print(f"  cameras moved > 1 cm ({big.sum()}): median default {np.median(e['default'][big]):.2f} vs eopt {np.median(e['eopt'][big]):.2f} cm")
