"""POST HOC (2026-09-24, reviewer round 1): the score AUROC with SYMMETRIC exclusion of units where nothing was applied.

Units: the standard-24 prediction units of sweep_analysis (rep 0 episodes, base task solved unperturbed by the raw
policy, raw failed under the condition, `full` row present, score finite). Outcome repaired(u) = full succeeded.
applied(u) = the full row fired at least one channel (state, camera, photometric, denoise; sweep_analysis.applied_nothing).
  registered   AUROC over all units (the registered result)
  one-sided    drop units with repaired(u) AND NOT applied(u)           (the version first written, 2026-09-24)
  symmetric    drop every unit with NOT applied(u), whatever its outcome, i.e. the post hoc outcome
               repaired*(u) = repaired(u) AND applied(u) restricted to units where a channel fired
AUROC = Mann-Whitney with ties 1/2; 95% percentile bootstrap over units, B = 2000, seed 0 (sweep_analysis.boot_ci).

    uv run --no-project --with numpy --with scipy python wrapper/scripts/exclusion_symmetric.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from plus_common import load_shards  # noqa: E402
from sweep_analysis import applied_nothing, auroc, boot_ci, load_scores  # noqa: E402
from sweep_conditions import CONDITIONS  # noqa: E402

D = os.path.join(REPO, "results", "drive_vihan", "sweep")
BY_NAME = {c["name"]: c for c in CONDITIONS}
_, PRED, _ = load_scores(D)


def units(stem):
    ep = {}
    for r in load_shards(os.path.join(D, stem + ".jsonl")):
        if r.get("rep", 0) == 0 and r["cond"] in BY_NAME:
            ep.setdefault((r["task_id"], r["cond"], r["arm"]), r)
    tasks = sorted({k[0] for k in ep})
    U = []
    for t in tasks:
        if not ep.get((t, "nominal", "raw"), {}).get("success"):
            continue
        for c in BY_NAME:
            if c == "nominal":
                continue
            r, f = ep.get((t, c, "raw")), ep.get((t, c, "full"))
            if r is None or f is None or r["success"]:
                continue
            s = PRED.get((t, c), {}).get("recover", np.nan)
            if np.isfinite(s):
                U.append(dict(task=t, cond=c, y=int(bool(f["success"])), s=float(s), applied=not applied_nothing(f)))
    return U


def line(name, U):
    s, y = [u["s"] for u in U], [u["y"] for u in U]
    a, (lo, hi) = auroc(s, y), boot_ci(s, y)
    print(f"  {name:12s} AUROC {a:.3f} [{lo:.2f}, {hi:.2f}]  n {len(U)} (repaired {sum(y)})")
    return a


for stem, fam in (("episodes_openvla_oft", "OpenVLA-OFT"), ("episodes_pi05", "pi0.5"), ("episodes_openvla", "OpenVLA (LIBERO-10 ft)")):
    U = units(stem)
    print(f"== {fam}: {len(U)} units, nothing applied on {sum(not u['applied'] for u in U)} "
          f"({sum((not u['applied']) and u['y'] for u in U)} of them repaired)")
    line("registered", U)
    line("one-sided", [u for u in U if u["applied"] or not u["y"]])
    line("symmetric", [u for u in U if u["applied"]])
