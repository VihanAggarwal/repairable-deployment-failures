"""Sequential E-optimal probe selection (vendor/local_eoptimal_seq.py): default 3 poses vs 3 poses chosen at the robot's
own first estimate, with all 7 as reference. Pre-stated check: sigma_min at the TRUE camera higher for the adaptive set.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/eoptimal_seq_analysis.py
"""
import json
import os
from math import comb

import numpy as np
from scipy.stats import wilcoxon

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "local_eoptimal_seq.jsonl")
R = [json.loads(l) for l in open(P) if l.strip()]


def sign_p(w, l):
    n = w + l
    return sum(comb(n, i) for i in range(w, n + 1)) / 2 ** n if n else 1.0


e = {k: np.array([r[f"err_{k}"][0] for r in R]) for k in ("default", "adaptive", "all7")}
sd, sa = np.array([r["smin_true_default"] for r in R]), np.array([r["smin_true_adaptive"] for r in R])
print(f"{len(R)} camera tasks")
w, l = int((sa > sd + 1e-9).sum()), int((sa < sd - 1e-9).sum())
print(f"PRE-STATED CHECK sigma_min(true): adaptive {np.median(sa):.2f} vs default {np.median(sd):.2f}; higher on {w}, "
      f"lower on {l}; sign p {sign_p(w, l):.4f} -> {'HOLDS' if sign_p(w, l) < 0.05 else 'FAILS'}")
for k in e:
    print(f"  {k:9s} median err {np.median(e[k]):6.2f} cm | within 2 cm {int((e[k] <= 2).sum()):2d} | within 1 cm {int((e[k] <= 1).sum()):2d}")
for base in ("default", "all7"):
    d = e[base] - e["adaptive"]
    w, l = int((d > 1e-6).sum()), int((d < -1e-6).sum())
    pw = wilcoxon(e[base], e["adaptive"], alternative="greater").pvalue if w + l else 1.0
    print(f"  adaptive better than {base}: {w}, worse {l}; sign one-sided p {sign_p(w, l):.4f}; Wilcoxon p {pw:.4f}")
ok = e["default"] <= 2
print(f"  where the first estimate is within 2 cm ({ok.sum()}): median default {np.median(e['default'][ok]):.2f} vs "
      f"adaptive {np.median(e['adaptive'][ok]):.2f} cm")
