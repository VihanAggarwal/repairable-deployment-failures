"""POST HOC (2026-09-24): the fingerprint's probe ablation on all three families, one quantity per row.

Units: exactly those of smc_probe.report's FINGERPRINT block (probe row present, raw and full episodes at rep 0, the base
task solved unperturbed, condition != nominal, the task's nominal probe row present), episodes merged over every session
shard (plus_common.load_shards). Outcome: the wrapped (full) episode succeeds. Distances, all smc.field_distance
(||A - A_ref|| / ||A_ref||) against the SAME task in its training setup:
  7 probes     A_cor (7 x 3) vs the nominal row's A_raw
  3 probes     the first three rows (centre + x pair)
  1 probe      the centre probe only
  opening      no probe: the full episode's first action chunk mean (first_chunk[:3], the wrapper's frame at the episode's
               own start state) vs the nominal raw episode's first_chunk[:3]
AUROC of minus the distance for wrapped success (Mann-Whitney, ties 1/2), 95% percentile bootstrap over units (B = 2000,
seed 0).

    uv run --no-project --with numpy python wrapper/scripts/fingerprint_ablation.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import smc  # noqa: E402
from plus_common import load_shards  # noqa: E402
from sweep_analysis import auroc, boot_ci  # noqa: E402

SWEEP = os.path.join(REPO, "results", "drive_vihan", "sweep")


def main():
    for pol, nice in (("openvla_oft", "OpenVLA-OFT"), ("pi05", "pi0.5"), ("openvla", "OpenVLA (LIBERO-10 fine-tuned)")):
        ep = {}
        for r in load_shards(os.path.join(SWEEP, f"episodes_{pol}.jsonl")):
            if r.get("rep", 0) == 0:
                ep.setdefault((r["task_id"], r["cond"], r["arm"]), r)
        rows = [json.loads(l) for l in open(os.path.join(SWEEP, f"smc_{pol}.jsonl")) if l.strip()]
        U = [r for r in rows if (r["task_id"], r["cond"], "raw") in ep and (r["task_id"], r["cond"], "full") in ep
             and ep.get((r["task_id"], "nominal", "raw"), {}).get("success", False)]
        ref = {r["task_id"]: r["A_raw"] for r in rows if r["cond"] == "nominal"}
        F = [r for r in U if r["task_id"] in ref and r["cond"] != "nominal"]
        ff = np.array([bool(ep[(r["task_id"], r["cond"], "full")]["success"]) for r in F])
        out = {}
        for k, name in ((7, "7 probes"), (3, "3 probes"), (1, "1 probe (centre)")):
            d = np.array([smc.field_distance(np.array(r["A_cor"])[:k], np.array(ref[r["task_id"]])[:k]) for r in F])
            out[name] = (auroc(-d, ff), boot_ci(-d, ff), len(d))
        d, keep = [], []
        for r in F:
            fc = ep[(r["task_id"], r["cond"], "full")].get("first_chunk")
            nc = ep[(r["task_id"], "nominal", "raw")].get("first_chunk")
            if fc is not None and nc is not None:
                d.append(smc.field_distance([np.asarray(fc)[:3]], [np.asarray(nc)[:3]]))
                keep.append(True)
            else:
                keep.append(False)
        keep = np.array(keep)
        d = np.array(d)
        if len(d):
            out["opening action (no probe)"] = (auroc(-d, ff[keep]), boot_ci(-d, ff[keep]), len(d))
        print(f"\n===== {nice}: {len(F)} fingerprint units, wrapped success {int(ff.sum())}")
        for name, (a, ci, n) in out.items():
            print(f"  {name:<27} AUROC {a:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]  (n {n})")


if __name__ == "__main__":
    main()
