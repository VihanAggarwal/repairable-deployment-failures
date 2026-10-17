"""POST HOC (2026-09-25): summary of vendor/local_refine_learned.py (policy-free, set A, learned masks + table-plane depth).

Per task: error of the learned silhouette calibration, of its EasyHeC-style refinement, of the refinement from the
training pose, and of `refine2` (the two-start rule of pose_sources.estimate: keep the lower final dense loss; applied iff
it moved > 0.5 cm/deg from the training pose and its loss < 0.8 x the loss at the training pose, else the training pose).

    uv run --no-project --with numpy python wrapper/scripts/refine_learned_summary.py [rows.jsonl]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "vendor", "local_refine_learned_s3.jsonl")
R = [json.loads(l) for l in open(P) if l.strip()]


def stats(name, e):
    e = np.asarray(e, float)
    print("%-34s median %6.2f cm | mean %6.2f | <=1 cm %2d | <=2 cm %2d | of %d" % (
        name, np.median(e), e.mean(), (e <= 1).sum(), (e <= 2).sum(), len(e)))


pick = []
for r in R:
    from_sil = r["refine_loss_final"] <= r["nominal_refine_loss_final"]
    e_ref = r["err_learned_refined"][0] if from_sil else r["err_refined_from_nominal"][0]
    loss = r["refine_loss_final"] if from_sil else r["nominal_refine_loss_final"]
    applied = loss < 0.8 * r["refine_loss_nominal"]
    e_true_shift = r["true_shift"][0]
    pick.append(e_ref if applied else e_true_shift)     # not applied: the training pose, off by the true shift
    r["_refine2"] = pick[-1]
print("tasks", len(R), "| mask IoU mean %.3f" % np.mean([r["mask_iou"] for r in R]))
stats("learned silhouette calibration", [r["err_learned"][0] for r in R])
stats("  + refine from that pose", [r["err_learned_refined"][0] for r in R])
stats("  refine from the training pose", [r["err_refined_from_nominal"][0] for r in R])
stats("  refine2 (two starts, lower loss)", pick)
moved = [r for r in R if r["true_shift"][0] > 1.0]
stats("refine2, cameras that moved > 1 cm", [r["_refine2"] for r in moved])
sys.path.insert(0, HERE)
from sweep_analysis import auroc  # noqa: E402
y = np.array([r["_refine2"] > 2.0 for r in R])
loss = np.array([min(r["refine_loss_final"], r["nominal_refine_loss_final"]) for r in R])
print("refine2 failures (> 2 cm): %d of %d; the final dense loss flags them at AUROC %.3f; loss max ok %.4f, min failed %.4f" % (
    int(y.sum()), len(y), auroc(loss, y), loss[~y].max(), loss[y].min()))
for r in sorted(R, key=lambda r: -r["_refine2"])[:6]:
    print("   worst: task %d shift %.1f cm, learned %.2f -> refine2 %.2f cm (IoU %.3f)" % (
        r["task_id"], r["true_shift"][0], r["err_learned"][0], r["_refine2"], r["mask_iou"]))
