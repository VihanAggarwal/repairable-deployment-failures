"""POST HOC (2026-09-26): test the unrepairability certificate against the attribution ladder's logged outcomes.

--ids prints the LIBERO-10 camera raw failures (OFT sets A and B, pi0.5 set A) to render with vendor/local_certificate.py.
Otherwise: joins vendor/local_certificate.jsonl to each unit's ladder bucket and reports
  (b) information loss: among units with c_task < 0.5 (fixed before joining outcomes), how many any GEOMETRIC rung
      (measured or true pose) repaired, with an exact one-sided 95% upper bound; AUROC of c_task vs whole-frame true-pose
      coverage for predicting "a geometric rung repairs it";
  (a) non-identifiability: from the logged scores, the identifiability factor on scene / background units and their
      repairs.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/certificate_test.py [--ids]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import attribution_ladder as AL  # noqa: E402
from sweep_analysis import auroc  # noqa: E402

D = "C:/Users/vihan/Downloads/local_data/plus"
SETS = [("OFT set A", [f"{D}/wrapper_oft/rows.jsonl", f"{D}/rows_oft.jsonl"]),
        ("OFT set B", [f"{D}/wraps/wrap_openvla_oft.jsonl", f"{D}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl"]),
        ("pi0.5 set A", [f"{D}/wraps/wrap_pi05.jsonl", f"{D}/wraps_ladder/wrap_pi05.jsonl"])]
GEOM = {"recovered by measurement", "not identified"}          # buckets first repaired by a geometric rung
THR = 0.5


def units():
    U = []
    for name, files in SETS:
        for t, v in AL.units(files).items():
            if "raw" in v and "full" in v and not v["raw"]["success"] and v["raw"].get("category", "camera") == "camera":
                U.append(dict(set=name, task=t, bucket=AL.bucket(v)))
    return U


def main():
    U = units()
    if "--ids" in sys.argv:
        print(",".join(str(t) for t in sorted({u["task"] for u in U})))
        return
    cert = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(REPO, "vendor", "local_certificate.jsonl"))}
    J = [dict(u, **cert[u["task"]]) for u in U if u["task"] in cert and np.isfinite(cert[u["task"]]["c_task"])]
    print("ladder camera failures with a certificate row: %d of %d" % (len(J), len(U)))
    y = np.array([u["bucket"] in GEOM for u in J])
    ct = np.array([u["c_task"] for u in J]); cw = np.array([u["coverage_true"] for u in J])
    certified = ct < THR
    k, n = int(y[certified].sum()), int(certified.sum())
    ub = beta.ppf(0.95, k + 1, n - k) if n else float("nan")
    print("(b) certified unrepairable (c_task < %.1f): %d units; repaired by a geometric rung: %d (one-sided 95%% upper bound %.2f)"
          % (THR, n, k, ub))
    print("    not certified: %d units, geometric repairs %d (%.0f%%)" % ((~certified).sum(), y[~certified].sum(),
                                                                         100 * y[~certified].mean() if (~certified).any() else 0))
    print("    AUROC for 'a geometric rung repairs it': c_task %.3f | whole-frame true-pose coverage %.3f" % (auroc(ct, y), auroc(cw, y)))
    for b in sorted({u["bucket"] for u in J}):
        m = np.array([u["bucket"] == b for u in J])
        print("    bucket %-26s n %2d | c_task median %.2f | certified %d" % (b, m.sum(), np.median(ct[m]), (certified & m).sum()))

    # (a) non-identifiability, from the logged scores (sweep units)
    import score_baselines as SB
    from sweep_analysis import load_scores
    _, pred, _ = load_scores(SB.SWEEP)
    for nice, pol in SB.FAMS.items():
        V = [u for u in SB.units_for(pol, pred)[0] if u["family"] == "background"]
        if V:
            print("(a) %-32s background units %d: ident median %.3f, max %.3f; repaired %d (applied nothing on %d)" % (
                nice, len(V), np.median([u["ident"] for u in V]), max(u["ident"] for u in V),
                sum(u["repaired"] for u in V), sum(bool(u.get("nothing")) for u in V)))


if __name__ == "__main__":
    main()
