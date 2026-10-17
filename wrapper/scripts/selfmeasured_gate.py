"""POST HOC (2026-09-25): a fully self-measured score with a measurement gate.

score_hat = ident(at the self-calibrated pose) x coverage(self-measured) x 1[calibration silhouette residual <= tau].
The residual is the calibration's own final silhouette score (logged as cal_final_score): no simulator truth enters.
tau is fit on the DEVELOPMENT family only (OFT: the midpoint between the largest residual of a calibration within 2 cm
and the smallest of one beyond), then frozen and applied to the two held-out families. Non-camera units are unchanged.
Reported: calibration-failure AUROC of the residual per family, and Table 1 AUROC and within-category AUROC with the
oracle score, the ungated self-measured score, and the gated one.

    uv run --no-project --with numpy python wrapper/scripts/selfmeasured_gate.py
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_baselines as SB  # noqa: E402
import selfcheck_calfail as SC  # noqa: E402
import within_category as WC  # noqa: E402
from sweep_analysis import auroc, load_scores  # noqa: E402

DEV = "OpenVLA-OFT"


def main():
    _, pred, _ = load_scores(SB.SWEEP)
    sm = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(SC.SM))}
    rows = SC.camera_rows()
    res = {k: float(r["cal_final_score"]) for k, r in rows.items()}
    err = {k: float(r["cam_est_err_cm_deg"][0]) for k, r in rows.items()}
    dev = [k for k in rows if k[0] == DEV]
    ok = [res[k] for k in dev if err[k] <= 2.0]
    bad = [res[k] for k in dev if err[k] > 2.0]
    tau = 0.5 * (max(ok) + min(bad)) if ok and bad and max(ok) < min(bad) else float(np.median(ok + bad))
    print("tau fit on %s only: %.3f (ok max %.3f, failed min %.3f; %d / %d units)" % (
        DEV, tau, max(ok), min(bad), len(ok), len(bad)))
    for nice in SB.FAMS:
        ks = [k for k in rows if k[0] == nice]
        y = np.array([err[k] > 2.0 for k in ks])
        flag = np.array([res[k] > tau for k in ks])
        print("   %-32s n %3d failures %3d | residual AUROC %.3f | gate flags %d, of them failures %d, failures missed %d" % (
            nice, len(ks), int(y.sum()), auroc([res[k] for k in ks], y), int(flag.sum()), int((flag & y).sum()),
            int((~flag & y).sum())))

    print()
    print("%-26s %-32s %8s %8s" % ("variant", "family", "AUROC", "within"))
    for vn in ("oracle (Table 1)", "self-measured", "self-measured + gate"):
        held = []
        for nice, pol in SB.FAMS.items():
            pv = copy.deepcopy(pred)
            if vn != "oracle (Table 1)":
                for (t, c), r in sm.items():
                    if (t, c) not in pv:
                        continue
                    g = 1.0
                    if vn.endswith("gate") and (nice, t, c) in res:
                        g = float(res[(nice, t, c)] <= tau)
                    pv[(t, c)]["ident"] = r["ident_hat"]
                    pv[(t, c)]["recover"] = r["ident_hat"] * pv[(t, c)]["invert"] * g
            U = SB.units_for(pol, pv)[0]
            W = [u for u in U if np.isfinite(u["score"])]
            a = auroc([u["score"] for u in W], [u["repaired"] for u in W])
            lo, hi = WC.cluster_ci(W, lambda s: auroc([u["score"] for u in s], [u["repaired"] for u in s]))
            w, _ = WC.strat_auroc(U, "score", "family")
            print("%-26s %-32s %8.3f [%.2f, %.2f] %8.3f" % (vn, nice, a, lo, hi, w))
            if nice != DEV:
                held += [dict(u, task=(nice, u["task"])) for u in U]
        w, _ = WC.strat_auroc(held, "score", "family")
        lo, hi = WC.cluster_ci(held, lambda s: WC.strat_auroc(s, "score", "family")[0])
        print("%-26s %-32s %8s %21.3f [%.2f, %.2f]" % ("", "held-out pooled, within", "", w, lo, hi))


if __name__ == "__main__":
    main()
