"""POST HOC (2026-09-26): the certificate on LEARNED sensing in closed loop (run file 25 rows, OFT set A).

Does the refinement's own final dense loss (self-measured, no truth) flag the rollouts whose calibration failed, and
would a gated wrapper (act only when the loss is low, else hand the policy the raw frame) have avoided the breaks?
The loss threshold is NOT fit on these rollouts: it is the policy-free rule from vendor/local_refine_learned*.jsonl
(the largest final loss of a fit within 2 cm on sets A and B, computed before this analysis).

    uv run --no-project --with numpy python wrapper/scripts/gated_learned.py
"""
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from plus_common import load_shards  # noqa: E402
from sweep_analysis import auroc  # noqa: E402

D = "C:/Users/vihan/Downloads/local_data/plus"


def one_sided(b, c):
    n = b + c
    return sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0


# threshold from the POLICY-FREE runs only
loc = []
for fn in ("local_refine_learned_s3.jsonl", "local_refine_learned_s3_setB.jsonl"):
    for r in map(json.loads, open(os.path.join(REPO, "vendor", fn))):
        first = r["refine_loss_final"] <= r["nominal_refine_loss_final"]
        e = r["err_learned_refined"][0] if first else r["err_refined_from_nominal"][0]
        loss = min(r["refine_loss_final"], r["nominal_refine_loss_final"])
        loc.append((e, loss))
thr = max(l for e, l in loc if e <= 2.0)
print("policy-free threshold (largest loss of a fit within 2 cm, sets A+B): %.4f" % thr)

by = {}
for r in load_shards(f"{D}/wraps_refine2/wrap_openvla_oft.jsonl"):
    if r.get("category") == "camera" and r.get("rep", 0) == 0:
        by.setdefault(r["task_id"], {}).setdefault(r["arm"], r)
U = [(t, v["raw"], v["full_hybrid_refine2"]) for t, v in by.items() if "raw" in v and "full_hybrid_refine2" in v]
keys = sorted({k for _, _, f in U for k in f if "loss" in k})
print("loss fields logged:", keys)
lk = "refine_loss_final" if "refine_loss_final" in keys else keys[0]
err = np.array([f["cam_est_err_cm_deg"][0] for _, _, f in U])
loss = np.array([f.get(lk, np.nan) for _, _, f in U], float)
ok = np.isfinite(loss)
print("rollouts %d; calibration failures (> 2 cm) %d; loss flags them at AUROC %.3f" % (
    ok.sum(), int((err[ok] > 2).sum()), auroc(loss[ok], err[ok] > 2)))
raw = np.array([bool(r["success"]) for _, r, _ in U])
wr = np.array([bool(f["success"]) for _, _, f in U])
gate = ok & (loss <= thr)
gated = np.where(gate, wr, raw)
for name, s in (("ungated", wr), ("gated (raw frame when loss > threshold)", gated)):
    b, c = int((s & ~raw).sum()), int((raw & ~s).sum())
    print("%-40s success %d vs raw %d | repaired %d broken %d | one-sided p %.4f" % (name, s.sum(), raw.sum(), b, c,
                                                                                     one_sided(b, c)))
print("gate acts on %d of %d tasks" % (gate.sum(), len(U)))
