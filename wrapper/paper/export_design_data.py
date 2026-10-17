"""Export the REAL data behind each proposed figure to CSV for the Claude Design bundle (no simulator, CPU only).

    uv run --no-project --with numpy python wrapper/paper/export_design_data.py <out_dir>
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "wrapper", "scripts"))
import score_baselines as SB  # noqa: E402
import selfcheck_calfail as SC  # noqa: E402
from sweep_analysis import load_scores  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
TAU = 0.547


def write(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", name, len(rows), "rows")


# F3: within-category ranking, one row per (family, base task, condition) raw failure
_, pred, _ = load_scores(SB.SWEEP)
sm = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(SC.SM))}
cam = SC.camera_rows()
U = []
for nice, pol in SB.FAMS.items():
    for u in SB.units_for(pol, pred)[0]:
        k = (u["task"], u["cond"])
        d = u["score"]
        if u["family"] == "camera" and k in sm:
            r = cam.get((nice, *k))
            g = 1.0 if r is None or r.get("cal_final_score") is None else float(r["cal_final_score"] <= TAU)
            d = sm[k]["ident_hat"] * u["invert"] * g
        U.append(dict(policy=nice, held_out=int(nice != "OpenVLA-OFT"), base_task=u["task"], condition=u["cond"],
                      category=u["family"], repaired=u["repaired"], registered_score=u["score"], deployable_score=d,
                      coverage=u["invert"], identifiability=u["ident"]))
write("fig_within_category_units.csv", U, list(U[0]))

# F4a: calibration residual vs error (the certificate), one row per camera rollout with a logged calibration
C = [dict(policy=n, base_task=t, condition=c, calibration_error_cm=float(r["cam_est_err_cm_deg"][0]),
          calibration_error_deg=float(r["cam_est_err_cm_deg"][1]), residual=float(r["cal_final_score"]),
          gate_tau=TAU) for (n, t, c), r in cam.items() if r.get("cal_final_score") is not None]
write("fig_certificate_residual_vs_error.csv", C, list(C[0]))

# F4b: coverage vs repair, 98 raw camera failures in rollouts
V = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "coverage_units.jsonl")) if l.strip()]
write("fig_coverage_units.csv", V, list(V[0]))

# F5: attribution ladder counts (paper Sec. 4.4, RESULTS.md; runs 19 and 24 for the base-rate overlay)
L = [dict(policy="OpenVLA-OFT", rung="repaired at measured pose", failures=61, solved_unperturbed=""),
     dict(policy="OpenVLA-OFT", rung="only the true pose", failures=17, solved_unperturbed=""),
     dict(policy="OpenVLA-OFT", rung="only a perfect render", failures=21, solved_unperturbed="15 of 21"),
     dict(policy="OpenVLA-OFT", rung="nothing", failures=1, solved_unperturbed=""),
     dict(policy="pi0.5", rung="repaired at measured pose", failures=7, solved_unperturbed=""),
     dict(policy="pi0.5", rung="only the true pose", failures=3, solved_unperturbed="2 of 3"),
     dict(policy="pi0.5", rung="only a perfect render", failures=9, solved_unperturbed="9 of 9"),
     dict(policy="pi0.5", rung="nothing", failures=5, solved_unperturbed="5 of 5")]
write("fig_ladder_counts.csv", L, list(L[0]))

# F6: learned sensing calibration, policy-free, per task (sets A and B), refine2 selection rule of pose_sources
R = []
for s, fn in (("A", "local_refine_learned_s3.jsonl"), ("B", "local_refine_learned_s3_setB.jsonl")):
    for r in map(json.loads, open(os.path.join(REPO, "vendor", fn))):
        first = r["refine_loss_final"] <= r["nominal_refine_loss_final"]
        e = r["err_learned_refined"][0] if first else r["err_refined_from_nominal"][0]
        loss = r["refine_loss_final"] if first else r["nominal_refine_loss_final"]
        e2 = e if loss < 0.8 * r["refine_loss_nominal"] else r["true_shift"][0]
        R.append(dict(set=s, task=r["task_id"], true_shift_cm=r["true_shift"][0], mask_iou=r["mask_iou"],
                      learned_search_error_cm=r["err_learned"][0], learned_refined_error_cm=e2, final_loss=loss))
write("fig_learned_calibration.csv", R, list(R[0]))

# Table 2 numbers (paired closed-loop results), from the paper
T = [dict(policy=p, suite=s, channel=c, n=n, raw_success=a, wrapped_success=b, repaired=rep, broken=br)
     for p, s, c, n, a, b, rep, br in [
         ("OpenVLA-OFT", "Spatial", "joint offset", 20, 0.25, 0.95, "15/15", "1/5"),
         ("OpenVLA-OFT", "Object", "joint offset", 20, 0.20, 1.00, "16/16", "0/4"),
         ("OpenVLA-OFT", "Goal", "joint offset", 20, 0.20, 1.00, "16/16", "0/4"),
         ("OpenVLA-OFT", "LIBERO-10", "joint offset", 30, 0.33, 0.90, "18/20", "1/10"),
         ("OpenVLA-OFT", "Spatial", "camera", 60, 0.90, 0.88, "3/6", "4/54"),
         ("OpenVLA-OFT", "Object", "camera", 60, 0.42, 0.75, "26/35", "6/25"),
         ("OpenVLA-OFT", "Goal", "camera", 60, 0.67, 0.78, "12/20", "5/40"),
         ("OpenVLA-OFT", "LIBERO-10", "camera", 60, 0.35, 0.63, "20/39", "3/21"),
         ("pi0.5", "LIBERO-10", "joint offset", 30, 0.37, 0.90, "16/19", "0/11"),
         ("pi0.5", "LIBERO-10", "camera", 30, 0.20, 0.37, "7/24", "2/6")]]
write("table_closed_loop.csv", T, list(T[0]))
