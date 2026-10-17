"""POST HOC (2026-09-24): does the score survive when its identifiability factor is self-measured?

vendor/score_selfmeasured.py recomputes, for every camera unit of the graded sweep, sigma_min of the camera block of J
twice in the same run: around the simulator's TRUE camera (the oracle used by scores_v2) and around the robot's
SELF-CALIBRATED camera. This script swaps each into the score (recover = ident x invert, invert unchanged: it is
already self-measured) and repeats the Table 1 analyses: AUROC per family, within-category AUROC (a category router is
0.5 there), and how well a LOW score predicts calibration failure (> 2 cm) on camera units.

    uv run --no-project --with numpy python wrapper/scripts/score_selfmeasured_analysis.py
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_baselines as SB  # noqa: E402
import within_category as WC  # noqa: E402
from plus_common import load_shards  # noqa: E402
from sweep_analysis import auroc, load_scores  # noqa: E402

SM = os.path.join(HERE, "..", "..", "vendor", "score_selfmeasured.jsonl")


def variant(pred, sm, field):
    out = copy.deepcopy(pred)
    for (tid, cond), r in sm.items():
        p = out.get((tid, cond))
        if p is None:
            continue
        p["ident"] = r[field]
        p["recover"] = r[field] * p["invert"]
    return out


def main():
    _, pred, _ = load_scores(SB.SWEEP)
    sm = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(SM)) if r}
    print("self-measured rows: %d camera units" % len(sm))
    V = {"v2 (logged, true pose)": pred, "true pose (recomputed)": variant(pred, sm, "ident_true"),
         "self-measured pose": variant(pred, sm, "ident_hat")}
    covered = set(sm)
    print("%-26s %-34s %8s %8s %10s" % ("variant", "family", "AUROC", "within", "cam units"))
    for vn, pv in V.items():
        held = []
        for nice, pol in SB.FAMS.items():
            U = SB.units_for(pol, pv)[0]
            cam = [u for u in U if u["family"] == "camera"]
            n_cov = sum((u["task"], u["cond"]) in covered for u in cam)
            W = [u for u in U if np.isfinite(u["score"])]
            a = auroc([u["score"] for u in W], [u["repaired"] for u in W])
            w, _ = WC.strat_auroc(U, "score", "family")
            print("%-26s %-34s %8.3f %8.3f %5d/%-4d" % (vn, nice, a, w, n_cov, len(cam)))
            if nice != "OpenVLA-OFT":
                held += [dict(u, task=(nice, u["task"])) for u in U]
        w, _ = WC.strat_auroc(held, "score", "family")
        lo, hi = WC.cluster_ci(held, lambda s: WC.strat_auroc(s, "score", "family")[0])
        print("%-26s %-34s %8s %8.3f [%.2f, %.2f]" % (vn, "held-out pooled", "", w, lo, hi))

    print()
    print("calibration failure (> 2 cm) on camera units with a self-measured row (error from the same run)")
    y = np.array([r["cal_err_cm"] > 2.0 for r in sm.values()])
    print("   units %d, failures %d" % (len(y), int(y.sum())))
    for key in ("ident_true", "ident_hat", "coverage_hat"):
        print("   low %-13s AUROC %.3f" % (key, auroc(-np.array([r[key] for r in sm.values()]), y)))
    for key in ("ident_true", "ident_hat"):
        s = -np.array([r[key] * r["coverage_hat"] for r in sm.values()])
        print("   low score (%-10s x coverage) AUROC %.3f" % (key, auroc(s, y)))
    # the in-rollout version (Table: calibration logged in the episode rows), score swapped per variant
    for vn, pv in V.items():
        cal, seen = [], set()
        for nice, pol in SB.FAMS.items():
            for r in load_shards(os.path.join(SB.SWEEP, f"episodes_{pol}.jsonl")):
                if r.get("family") != "camera" or r.get("arm") != "full" or not r.get("cam_est_err_cm_deg"):
                    continue
                k = (r["task_id"], r["cond"])
                if k not in pv or (nice, k) in seen:
                    continue
                seen.add((nice, k))
                cal.append((float(r["cam_est_err_cm_deg"][0]) > 2.0, pv[k]["recover"]))
        yy = np.array([c[0] for c in cal])
        print("   in-rollout, %-24s low score AUROC %.3f (n %d, failures %d)" % (
            vn, auroc(-np.array([c[1] for c in cal]), yy), len(cal), int(yy.sum())))


if __name__ == "__main__":
    main()
