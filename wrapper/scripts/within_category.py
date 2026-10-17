"""POST HOC (2026-09-24): two analyses of what the score does that a category router cannot.

1. Stratified AUROC. A router that assigns each unit its category's (or condition's) repair rate, fit on the
   development family, is constant within a category (condition), so its AUROC over pairs drawn from the SAME category
   (condition) is exactly 1/2 by construction. The stratified AUROC of the score over those same pairs measures what the
   score adds on top of any category-level router: ranking which units of one kind are repairable.
     A_strat(s) = sum over strata g of #{(i, j): i repaired, j not, both in g, s_i > s_j} (+ 1/2 ties) / #{such pairs}
2. What ident is for. On camera units, the calibration either succeeds (error under 2 cm) or fails. Does the
   identifiability factor predict calibration failure better than coverage does? If so, the two factors of the score
   separate the two ways a camera correction fails: measurement (ident) and lost view (coverage).
Units and predictors exactly as score_baselines.units_for (24 standard conditions, raw failures on solved base tasks).
Cluster bootstrap over base tasks, B = 2000, seed 0.

    uv run --no-project --with numpy python wrapper/scripts/within_category.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_baselines as SB  # noqa: E402
from plus_common import load_shards  # noqa: E402
from sweep_analysis import auroc, load_scores  # noqa: E402


def strat_auroc(U, key, stratum):
    num = den = 0.0
    g = defaultdict(list)
    for u in U:
        if np.isfinite(u[key]):
            g[u[stratum]].append(u)
    for us in g.values():
        pos = [u[key] for u in us if u["repaired"]]
        neg = [u[key] for u in us if not u["repaired"]]
        for a in pos:
            for b in neg:
                num += 1.0 if a > b else (0.5 if a == b else 0.0)
                den += 1
    return num / den if den else np.nan, int(den)


def cluster_ci(U, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for u in U:
        by[u["task"]].append(u)
    keys = list(by)
    vals = []
    for _ in range(B):
        s = [u for k in rng.choice(len(keys), len(keys)) for u in by[keys[k]]]
        v = fn(s)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    _, pred, _ = load_scores(SB.SWEEP)
    fams = {nice: SB.units_for(pol, pred)[0] for nice, pol in SB.FAMS.items()}
    dev = fams["OpenVLA-OFT"]
    cat = SB.fit_rates(dev, "family")
    cond = SB.fit_rates(dev, "cond")
    for U in fams.values():
        for u in U:
            u["cat_rate"] = cat.get(u["family"], np.nan)
            u["cond_rate"] = cond.get(u["cond"], np.nan)

    print("1. STRATIFIED AUROC (pairs within the same stratum). Any category/condition router is 0.5 here by construction.")
    print("%-34s %-9s %6s %-24s %-24s %-24s" % ("family", "stratum", "pairs", "score", "invert (coverage)", "ident"))
    pooled = []
    for nice, U in fams.items():
        pooled += [dict(u, task=(nice, u["task"])) for u in U]
        for st in ("family", "cond"):
            row = []
            for key in ("score", "invert", "ident"):
                a, n = strat_auroc(U, key, st)
                lo, hi = cluster_ci(U, lambda s, k=key, t=st: strat_auroc(s, k, t)[0])
                row.append("%.3f [%.2f, %.2f]" % (a, lo, hi))
            print("%-34s %-9s %6d %-24s %-24s %-24s" % (nice, "category" if st == "family" else "condition",
                                                        strat_auroc(U, "score", st)[1], *row))
    for st in ("family", "cond"):
        held = [u for u in pooled if u["task"][0] != "OpenVLA-OFT"]
        for name, S_ in (("pooled, both held-out families", held), ("pooled, all three", pooled)):
            a, n = strat_auroc(S_, "score", st)
            lo, hi = cluster_ci(S_, lambda s, t=st: strat_auroc(s, "score", t)[0])
            ai, _ = strat_auroc(S_, "invert", st)
            print("  %-32s %-9s pairs %4d  score %.3f [%.2f, %.2f]  invert %.3f" % (
                name, "category" if st == "family" else "condition", n, a, lo, hi, ai))

    print()
    print("2. WHAT IDENT PREDICTS: calibration failure (error > 2 cm) on camera units, all logged units, all families")
    cal = []
    for nice, pol in SB.FAMS.items():
        seen = set()
        for r in load_shards(os.path.join(SB.SWEEP, f"episodes_{pol}.jsonl")):
            if r.get("family") != "camera" or r.get("arm") != "full" or not r.get("cam_est_err_cm_deg"):
                continue
            k = (r["task_id"], r["cond"])
            p = pred.get(k)
            if p is None or k in seen:
                continue
            seen.add(k)
            cal.append(dict(task=(nice, r["task_id"]), cond=r["cond"], err=float(r["cam_est_err_cm_deg"][0]),
                            ident=p["ident"], coverage=p.get("coverage", p["invert"]), score=p["recover"]))
    y = np.array([c["err"] > 2.0 for c in cal])
    print("   camera units with a logged calibration: %d, calibration failures (> 2 cm): %d" % (len(cal), int(y.sum())))
    for key in ("ident", "coverage", "score"):
        s = -np.array([c[key] for c in cal], float)          # LOW factor predicts FAILURE
        a = auroc(s, y)
        U2 = [dict(c, repaired=int(c["err"] > 2.0), v=-c[key]) for c in cal]
        lo, hi = cluster_ci(U2, lambda ss: auroc([u["v"] for u in ss], [u["repaired"] for u in ss]))
        print("   AUROC, low %-9s predicts calibration failure: %.3f [%.2f, %.2f]" % (key, a, lo, hi))


if __name__ == "__main__":
    main()
