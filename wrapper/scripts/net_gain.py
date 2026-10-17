"""POST HOC (2026-09-26): net gain over ALL shifted camera units (raw successes included), not AUROC among failures.

Decision rules on the same units (graded sweep, tasks each policy solves unperturbed, camera conditions):
  never     keep the raw rollout
  always    use the wrapped (full) rollout
  cov       correct iff self-measured coverage >= 0.5 (the registered coverage-gate constant)
  cov+res   correct iff coverage >= 0.5 and the calibration residual <= tau (tau fit on OFT only, as in the paper)
A declined correction is scored with the raw rollout of the same (task, condition, seed); this ignores the effect of
the probe motion, which the gate itself needs, and is stated as such. Net gain = successes minus raw successes.

    uv run --no-project --with numpy python wrapper/scripts/net_gain.py
"""
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_baselines as SB  # noqa: E402
import selfcheck_calfail as SC  # noqa: E402
import sweep_conditions as S  # noqa: E402
from plus_common import load_shards  # noqa: E402
from sweep_analysis import load_scores  # noqa: E402

DEV = "OpenVLA-OFT"


def mcnemar2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def units(pol):
    ep = {}
    for r in load_shards(os.path.join(SB.SWEEP, f"episodes_{pol}.jsonl")):
        ep.setdefault((r["task_id"], r["cond"], r["arm"]), []).append(float(r["success"]))
    ep = {k: np.mean(v) >= 0.5 for k, v in ep.items()}
    tasks = sorted({k[0] for k in ep})
    out = []
    for t in tasks:
        if not ep.get((t, "nominal", "raw")):
            continue
        for c in [c["name"] for c in S.CONDITIONS]:
            if c == "nominal" or S.BY_NAME[c]["family"] != "camera":
                continue
            if (t, c, "raw") in ep and (t, c, "full") in ep:
                out.append((t, c, ep[(t, c, "raw")], ep[(t, c, "full")]))
    return out


def main():
    _, pred, _ = load_scores(SB.SWEEP)
    rows = SC.camera_rows()
    res = {k: float(r["cal_final_score"]) for k, r in rows.items()}
    err = {k: float(r["cam_est_err_cm_deg"][0]) for k, r in rows.items()}
    dev = [k for k in rows if k[0] == DEV]
    ok = [res[k] for k in dev if err[k] <= 2.0]
    bad = [res[k] for k in dev if err[k] > 2.0]
    tau = 0.5 * (max(ok) + min(bad))
    print("tau (OFT only) %.3f" % tau)
    tot = {}
    for nice, pol in SB.FAMS.items():
        U = units(pol)
        raw = np.array([u[2] for u in U])
        full = np.array([u[3] for u in U])
        cov = np.array([pred.get((u[0], u[1]), {}).get("invert", np.nan) for u in U])
        rr = np.array([res.get((nice, u[0], u[1]), np.nan) for u in U])
        dec = {"never": np.zeros(len(U), bool), "always": np.ones(len(U), bool),
               "cov": np.nan_to_num(cov, nan=0) >= 0.5,
               "cov+res": (np.nan_to_num(cov, nan=0) >= 0.5) & ~(np.nan_to_num(rr, nan=-np.inf) > tau),
               "cov+res (logged only)": None}
        print("=== %s: %d camera units, raw successes %d, logged residual on %d" % (nice, len(U), int(raw.sum()), int(np.isfinite(rr).sum())))
        L = np.isfinite(rr)
        for name, d in dec.items():
            if d is None:
                d = dec["cov+res"]
                raw_, full_, d = raw[L], full[L], d[L]
            else:
                raw_, full_ = raw, full
            out = np.where(d, full_, raw_)
            rep, brk = int((out & ~raw_).sum()), int((~out & raw_).sum())
            b = int((out & ~full_).sum()); c = int((~out & full_).sum())
            print("  %-8s corrects %3d | success %3d/%d | repaired %3d broken %3d | net %+3d | vs always: better %d worse %d p %.3f" % (
                name, int(d.sum()), int(out.sum()), len(out), rep, brk, rep - brk, b, c, mcnemar2(b, c)))
            t = tot.setdefault(name, [0, 0, 0, 0])
            t[0] += rep; t[1] += brk; t[2] += b; t[3] += c
    print("=== pooled over families")
    for name, (rep, brk, b, c) in tot.items():
        print("  %-8s repaired %3d broken %3d net %+3d | vs always better %d worse %d p %.3f" % (name, rep, brk, rep - brk, b, c, mcnemar2(b, c)))


if __name__ == "__main__":
    main()
