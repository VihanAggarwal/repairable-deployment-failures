"""POST HOC (2026-09-25): the self-calibration certificate, checked on the logged camera units.

Local linear model: the observed self-observation o = s(eta) + eps, the fitted pose eta_hat with residual
r = s(eta_hat) - o. Then ||s(eta_hat) - s(eta)|| >= sigma_min(J) ||d|| - (L/2) ||d||^2 with d = eta_hat - eta, so inside
the radius where the curvature term is at most half the linear one, ||d|| <= 2 (||r|| + ||eps||) / sigma_min(J).
Prediction checked here: the calibration error scales with sqrt(rho) / sigma_min, rho the logged silhouette residual
(cal_final_score) and sigma_min the self-measured smallest singular value at the estimated pose
(vendor/score_selfmeasured.jsonl, sigma_min_hat). Reported: Spearman rank correlation with the error and AUROC for
calibration failure (> 2 cm) of rho alone, 1/sigma_min alone, and the certificate sqrt(rho)/sigma_min, on every camera
unit that has both a logged rollout calibration and a self-measured sigma_min.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/certificate_check.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selfcheck_calfail as SC  # noqa: E402
from sweep_analysis import auroc  # noqa: E402


def main():
    sm = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(SC.SM))}
    rows = SC.camera_rows()
    U = []
    for (nice, t, c), r in rows.items():
        s = sm.get((t, c))
        if s is None or r.get("cal_final_score") is None or s["sigma_min_hat"] <= 0:
            continue
        U.append(dict(err=float(r["cam_est_err_cm_deg"][0]), rho=float(r["cal_final_score"]), smin=float(s["sigma_min_hat"])))
    e = np.array([u["err"] for u in U])
    y = e > 2.0
    sig = {"rho (residual) alone": np.array([u["rho"] for u in U]),
           "1 / sigma_min alone": np.array([1 / u["smin"] for u in U]),
           "certificate sqrt(rho) / sigma_min": np.array([np.sqrt(u["rho"]) / u["smin"] for u in U])}
    print("units %d, calibration failures %d" % (len(U), int(y.sum())))
    for k, v in sig.items():
        print("   %-36s Spearman with error %.3f | AUROC for failure %.3f" % (k, spearmanr(v, e)[0], auroc(v, y)))
    ok = ~y
    cert = sig["certificate sqrt(rho) / sigma_min"]
    k = np.median(e[ok] / np.maximum(cert[ok], 1e-9))
    print("   successful units: error / certificate median %.2f (the constant a calibrated bound would need)" % k)
    print("   successful units violating error <= 3 x median-constant x certificate: %d of %d" % (
        int((e[ok] > 3 * k * cert[ok]).sum()), int(ok.sum())))


if __name__ == "__main__":
    main()
