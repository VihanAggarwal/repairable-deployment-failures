"""POST HOC (2026-09-25): is repairability a property of the nuisance or of the policy?

On (base task, condition) cells where BOTH policies fail raw, does the development policy's (OFT) wrapped outcome
predict the held-out policy's wrapped outcome (Fisher exact on the 2x2 agreement table), and how does that transfer
predictor compare with the policy-free score? Cluster bootstrap over base tasks, B = 2000, seed 0.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/cross_policy_transfer.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep_conditions as S  # noqa: E402
from plus_common import load_shards  # noqa: E402
from score_baselines import SWEEP  # noqa: E402
from sweep_analysis import auroc, load_scores  # noqa: E402

FAMS = {"OFT": "openvla_oft", "pi05": "pi05", "OpenVLA": "openvla"}
STD = [c["name"] for c in S.CONDITIONS if c["name"] != "nominal"]
_, pred, _ = load_scores(SWEEP)
rng = np.random.default_rng(0)


def cells(pol):
    rows = load_shards(os.path.join(SWEEP, f"episodes_{pol}.jsonl"))
    ep = defaultdict(list)
    for r in rows:
        ep[(r["task_id"], r["cond"], r["arm"])].append(float(r["success"]))
    m = {k: np.mean(v) for k, v in ep.items()}
    out = {}
    for t in sorted({r["task_id"] for r in rows}):
        if not m.get((t, "nominal", "raw"), 0) >= 0.5:
            continue
        for c in STD:
            r, f = m.get((t, c, "raw")), m.get((t, c, "full"))
            if r is None or f is None:
                continue
            p = pred.get((t, c), {})
            out[(t, c)] = dict(task=t, cond=c, fam=S.BY_NAME[c]["family"], raw=int(r >= .5), full=int(f >= .5),
                               score=p.get("recover", np.nan))
    return out


def f(V, k):
    return auroc([u[k] for u in V], [u["y"] for u in V])


def boot(U, a, b, B=2000):
    by = defaultdict(list)
    for u in U:
        by[u["task"]].append(u)
    keys = list(by)
    d = []
    for _ in range(B):
        V = [u for k in rng.choice(len(keys), len(keys)) for u in by[keys[k]]]
        v = f(V, a) - f(V, b)
        if np.isfinite(v):
            d.append(v)
    d = np.array(d)
    return f(U, a) - f(U, b), np.percentile(d, 2.5), np.percentile(d, 97.5), np.mean(d <= 0)


def main():
    I = {k: cells(v) for k, v in FAMS.items()}
    for a, b in (("OFT", "pi05"), ("OFT", "OpenVLA")):
        A, Bd = I[a], I[b]
        sh = [k for k in A if k in Bd and not A[k]["raw"] and not Bd[k]["raw"] and np.isfinite(Bd[k]["score"])]
        U = [dict(task=k[0], fam=Bd[k]["fam"], y=Bd[k]["full"], s=Bd[k]["score"], xa=A[k]["full"]) for k in sh]
        t = [[sum(1 for u in U if u["xa"] == i and u["y"] == j) for j in (0, 1)] for i in (0, 1)]
        agree = t[0][0] + t[1][1]
        print("%s -> %s: shared failed cells %d, outcomes agree on %d (%.0f%%), Fisher p %.2g" % (
            a, b, len(U), agree, 100 * agree / len(U), fisher_exact(t)[1]))
        print("   AUROC score %.3f, transfer %.3f, score - transfer %+.3f [%+.2f, %+.2f] p %.3f" % (
            f(U, "s"), f(U, "xa"), *boot(U, "s", "xa")))


if __name__ == "__main__":
    main()
