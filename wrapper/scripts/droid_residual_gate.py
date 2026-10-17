"""POST HOC (2026-09-25): does the calibration's own residual flag a failed self-calibration on REAL DROID images?

Rows: the final DROID self-calibration run (VLM-chosen masks, results/drive_vihan/droid_selfcal_vlm or the local pull),
one row per (episode, camera). Self-measured quantities at the ESTIMATED pose, no ground truth: `score_est` (the
silhouette score the calibration minimised), `chamfer_est`, `est_render_iou` (rendered robot vs observed mask).
Target: calibration error pos_err_cm above a threshold. AUROC per signal and threshold; cluster bootstrap over DROID
labs is not possible (few labs), so a unit bootstrap (B = 2000, seed 0). The simulator tau does not transfer in scale
(different masks and score normalisation); only the ranking is tested here.

    uv run --no-project --with numpy python wrapper/scripts/droid_residual_gate.py [results.jsonl]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_analysis import auroc  # noqa: E402

P = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/vihan/Downloads/local_data/plus/droid_selfcal_vlm/selfcal_results.jsonl"


def main():
    R = [json.loads(l) for l in open(P) if l.strip()]
    R = [r for r in R if r.get("pos_err_cm") is not None and r.get("score_est") is not None]
    e = np.array([r["pos_err_cm"] for r in R])
    print("DROID rows %d, calibration error median %.1f cm (10th-90th %.1f-%.1f)" % (
        len(R), np.median(e), *np.percentile(e, [10, 90])))
    sig = {"high score_est (residual)": [r["score_est"] for r in R],
           "high chamfer_est": [r.get("chamfer_est", np.nan) for r in R],
           "low est_render_iou": [-r.get("est_render_iou", np.nan) for r in R],
           "high score_est - score_gt (needs GT, reference only)": [r["score_est"] - r["score_gt"] for r in R]}
    rng = np.random.default_rng(0)
    for thr in (5.0, 10.0, 20.0):
        y = e > thr
        print("failure = error > %.0f cm: %d of %d" % (thr, int(y.sum()), len(y)))
        for k, v in sig.items():
            v = np.array(v, float)
            ok = np.isfinite(v)
            a = auroc(v[ok], y[ok])
            bs = []
            idx = np.flatnonzero(ok)
            for _ in range(2000):
                s = rng.choice(idx, len(idx))
                if 0 < y[s].sum() < len(s):
                    bs.append(auroc(v[s], y[s]))
            print("   %-52s AUROC %.3f [%.2f, %.2f]" % (k, a, *np.percentile(bs, [2.5, 97.5])))


if __name__ == "__main__":
    main()
