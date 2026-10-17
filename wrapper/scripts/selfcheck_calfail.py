"""POST HOC (2026-09-25): can the robot tell, from self-measured quantities only, that its camera measurement failed?

Scoring from the self-calibrated pose (score_selfmeasured_analysis.py) loses the oracle's ability to flag a failed
calibration: ident at a wrong pose looks healthy. The rollouts already log two self-measured quality signals of the
measurement itself: the silhouette self-check F1 (observed silhouette warped into the training view vs the kinematic
prediction) and the calibration's final silhouette residual. On every camera unit with a logged calibration (all
families), how well does each flag calibration failure (> 2 cm), and does gating the fully self-measured score by the
self-check restore within-category ranking?

    uv run --no-project --with numpy python wrapper/scripts/selfcheck_calfail.py
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


def camera_rows():
    """First logged full-arm camera row per (family, task, cond)."""
    out = {}
    for nice, pol in SB.FAMS.items():
        for r in load_shards(os.path.join(SB.SWEEP, f"episodes_{pol}.jsonl")):
            if r.get("family") != "camera" or r.get("arm") != "full" or not r.get("cam_est_err_cm_deg"):
                continue
            out.setdefault((nice, r["task_id"], r["cond"]), r)
    return out


def main():
    _, pred, _ = load_scores(SB.SWEEP)
    sm = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(SM))}
    rows = camera_rows()
    y = np.array([float(r["cam_est_err_cm_deg"][0]) > 2.0 for r in rows.values()])
    print("camera units with a logged calibration: %d, failures (> 2 cm): %d" % (len(y), int(y.sum())))
    sig = {
        "low self-check F1": [-(r.get("selfcheck_f1") if r.get("selfcheck_f1") is not None else 0.0) for r in rows.values()],
        "high silhouette residual": [r.get("cal_final_score") if r.get("cal_final_score") is not None else np.inf
                                     for r in rows.values()],
        "low score, true pose (oracle)": [-pred[(t, c)]["recover"] for (_, t, c) in rows],
        "low score, self-measured": [-(sm[(t, c)]["ident_hat"] * pred[(t, c)]["invert"]) if (t, c) in sm
                                     else -pred[(t, c)]["recover"] for (_, t, c) in rows],
    }
    f1 = np.array([r.get("selfcheck_f1") if r.get("selfcheck_f1") is not None else 0.0 for r in rows.values()])
    sm_score = -np.array(sig["low score, self-measured"])
    sig["low (self-measured score x self-check F1)"] = list(-(sm_score * f1))
    for k, v in sig.items():
        v = np.nan_to_num(np.array(v, float), posinf=1e9)
        print("   %-44s AUROC %.3f" % (k, auroc(v, y)))
    print("   self-check F1 missing on %d units" % sum(r.get("selfcheck_f1") is None for r in rows.values()))

    # within-category ranking with the self-measured score gated by the self-check F1 (camera units only change)
    f1_by = {}
    for (nice, t, c), r in rows.items():
        if r.get("selfcheck_f1") is not None:
            f1_by[(nice, t, c)] = float(r["selfcheck_f1"])
    print()
    print("%-40s %-32s %8s %8s" % ("variant", "family", "AUROC", "within"))
    for gated in (False, True):
        held = []
        for nice, pol in SB.FAMS.items():
            pv = copy.deepcopy(pred)
            for (t, c), r in sm.items():
                if (t, c) in pv:
                    g = f1_by.get((nice, t, c), 1.0) if gated else 1.0
                    pv[(t, c)]["recover"] = r["ident_hat"] * pv[(t, c)]["invert"] * g
            U = SB.units_for(pol, pv)[0]
            W = [u for u in U if np.isfinite(u["score"])]
            a = auroc([u["score"] for u in W], [u["repaired"] for u in W])
            w, _ = WC.strat_auroc(U, "score", "family")
            name = "self-measured x self-check F1" if gated else "self-measured"
            print("%-40s %-32s %8.3f %8.3f" % (name, nice, a, w))
            if nice != "OpenVLA-OFT":
                held += [dict(u, task=(nice, u["task"])) for u in U]
        w, _ = WC.strat_auroc(held, "score", "family")
        lo, hi = WC.cluster_ci(held, lambda s: WC.strat_auroc(s, "score", "family")[0])
        print("%-40s %-32s %8s %8.3f [%.2f, %.2f]" % ("", "held-out pooled", "", w, lo, hi))


if __name__ == "__main__":
    main()
