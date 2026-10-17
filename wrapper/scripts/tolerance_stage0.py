"""THE POLICY'S OWN CALIBRATION TOLERANCE, stage 0 (observational, zero compute, POST HOC).

On logged camera episodes of the graded sweep, how far is the policy's FIRST action chunk under the wrapper from the chunk
it produces on the unperturbed task, and what drives that distance: the calibration error the robot actually made (cm),
or the share of the training view the correction could restore (coverage)? Prediction if calibration is already inside
the policy's tolerance: a flat slope on calibration error, a steep one on coverage.

STOP RULE (written before looking): if the standardized slope on calibration error is clearly nonzero (its task-grouped
bootstrap interval excludes 0 and it is at least half the coverage slope), do NOT spend GPU on the dose-response stage;
rewrite the sentence that says calibration is rarely what fails instead.

    python tolerance_stage0.py --episodes sweep/episodes_openvla_oft.jsonl --episodes .../episodes_pi05.jsonl
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_shards  # noqa: E402


def analyse(path):
    rows = [r for r in load_shards(path) if r.get("rep", 0) == 0]
    if not rows:
        print(f"\n===== {path}: no rows =====")
        return
    nom = {r["task_id"]: np.asarray(r["first_chunk"], float) for r in rows
           if r["cond"] == "nominal" and r["arm"] == "raw" and r.get("first_chunk") is not None}
    raw = {(r["task_id"], r["cond"]): r for r in rows if r["arm"] == "raw"}
    X, y, g, ok, draw = [], [], [], [], []
    for r in rows:
        if (r["arm"] != "full" or r.get("family") != "camera" or r.get("first_chunk") is None or r["task_id"] not in nom
                or not r.get("cam_est_err_cm_deg") or r.get("reproj_coverage") is None or not r.get("cam_applied")):
            continue
        n = nom[r["task_id"]]
        d = float(np.linalg.norm(np.asarray(r["first_chunk"], float) - n) / max(np.linalg.norm(n), 1e-9))
        X.append([r["cam_est_err_cm_deg"][0], 1.0 - r["reproj_coverage"], r["cam_true_shift_cm_deg"][0]])
        y.append(d)
        g.append(r["task_id"])
        ok.append(bool(r["success"]))
        rr = raw.get((r["task_id"], r["cond"]))
        if rr is not None and rr.get("first_chunk") is not None:
            draw.append(float(np.linalg.norm(np.asarray(rr["first_chunk"], float) - n) / max(np.linalg.norm(n), 1e-9)))
    X, y, g = np.asarray(X, float), np.asarray(y, float), np.asarray(g)
    print(f"\n===== {os.path.basename(path)} =====  {len(y)} camera episodes under the wrapper with a logged first chunk, {len(set(g))} base tasks")
    if len(y) < 12:
        print("  too few episodes")
        return
    print(f"  opening-action distance to the unperturbed chunk: corrected median {np.median(y):.3f}" +
          (f", uncorrected median {np.median(draw):.3f}" if draw else ""))
    print(f"  calibration error: median {np.median(X[:, 0]):.2f} cm, 90th percentile {np.quantile(X[:, 0], 0.9):.2f} cm | lost view (1 - coverage): median {np.median(X[:, 1]):.2f}")
    z = lambda v: (v - v.mean(0)) / np.maximum(v.std(0), 1e-9)

    def fit(idx):
        A = np.c_[z(X[idx]), np.ones(len(idx))]
        return np.linalg.lstsq(A, z(y[idx]), rcond=None)[0][:3]
    beta = fit(np.arange(len(y)))
    rng = np.random.default_rng(0)
    tasks = np.unique(g)
    boot = []
    for _ in range(2000):
        pick = rng.choice(tasks, len(tasks))
        idx = np.concatenate([np.flatnonzero(g == t) for t in pick])
        if len(set(idx)) > 6:
            boot.append(fit(idx))
    boot = np.asarray(boot)
    for i, k in enumerate(("calibration error (cm)", "lost view (1 - coverage)", "true camera shift (cm)")):
        lo, hi = np.quantile(boot[:, i], [0.025, 0.975])
        print(f"  standardized slope on {k:<26} {beta[i]:+.2f}  [{lo:+.2f}, {hi:+.2f}]  (task-grouped bootstrap)")
    from scipy.stats import spearmanr
    for i, k in enumerate(("calibration error", "lost view", "true shift")):
        r = spearmanr(X[:, i], y)
        print(f"  Spearman(distance, {k:<17}) = {r.correlation:+.2f}  (p = {r.pvalue:.3g})")
    lo, hi = np.quantile(boot[:, 0], [0.025, 0.975])
    stop = (lo > 0 or hi < 0) and abs(beta[0]) >= 0.5 * abs(beta[1])
    print("  STOP RULE:", "FIRES: calibration error moves the opening action; do not run the dose-response stage" if stop
          else "does not fire: calibration error does not move the opening action on these rows; stage 1 is justified")
    okv = np.asarray(ok)
    if 0 < okv.sum() < len(okv):
        print(f"  distance, successful vs failed episodes: median {np.median(y[okv]):.3f} vs {np.median(y[~okv]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", action="append", required=True)
    a = ap.parse_args()
    for p in a.episodes:
        analyse(p)


if __name__ == "__main__":
    main()
