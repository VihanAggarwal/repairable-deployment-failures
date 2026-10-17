"""Tests pre-registered in wrapper/prereg/2026-09-26_learned_refine_replication.md (run files 27 and 28), plus the
pooled secondary line across runs 25, 27 and 28. Every shard of each folder; raw and full from plus/wraps (same seeds).

H9: run 27 (OFT set B) full_hybrid_refine2 > raw (one-sided exact McNemar, 0.05). H10: not worse than simulator `full`
by more than 3 tasks. H11: run 28 (pi0.5 set A) > raw. H12: not worse than `full` by more than 3 tasks.

    uv run --no-project --with numpy python wrapper/scripts/prereg_learned_replication.py [data_dir]
"""
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plus_common import load_shards  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/vihan/Downloads/local_data/plus"
ARM = "full_hybrid_refine2"


def one_sided(b, c):
    n = b + c
    return sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0


def cells(rows):
    out = {}
    for r in rows:
        if r.get("category") == "camera" and r.get("rep", 0) == 0:
            out.setdefault(r["task_id"], {}).setdefault(r["arm"], r)
    return out


def test(name, new_path, base_path, hmain, hnoninf):
    new, base = cells(load_shards(new_path)), cells(load_shards(base_path))
    ids = [t for t in new if ARM in new[t] and "raw" in base.get(t, {})]
    s = lambda t, arm, src: bool(src[t][arm]["success"])
    w = sum(s(t, ARM, new) and not s(t, "raw", base) for t in ids)
    l = sum(s(t, "raw", base) and not s(t, ARM, new) for t in ids)
    p = one_sided(w, l)
    e = np.array([new[t][ARM]["cam_est_err_cm_deg"][0] for t in ids if new[t][ARM].get("cam_est_err_cm_deg")])
    nf = [t for t in ids if "full" in base.get(t, {})]
    sf, sr = sum(s(t, "full", base) for t in nf), sum(s(t, ARM, new) for t in nf)
    print("=== %s: %d tasks with rows" % (name, len(ids)))
    if len(e):
        print("   rollout calibration median %.2f cm, within 2 cm %d of %d" % (np.median(e), (e <= 2).sum(), len(e)))
    print("   %s learned vs raw: raw %d -> learned %d | repaired %d broken %d | one-sided p %.4f -> %s" % (
        hmain, sum(s(t, "raw", base) for t in ids), sum(s(t, ARM, new) for t in ids), w, l, p, "PASS" if p < 0.05 else "FAIL"))
    print("   %s learned vs simulator full: %d vs %d on %d tasks -> %s" % (hnoninf, sr, sf, len(nf), "PASS" if sr >= sf - 3 else "FAIL"))
    return w, l, len(ids)


tot = [0, 0, 0]
for args in (("run 25 (OFT set A, already final)", f"{D}/wraps_refine2/wrap_openvla_oft.jsonl", f"{D}/wraps_refine2/wrap_openvla_oft.jsonl", "H5", "-"),
             ("run 27 (OFT set B)", f"{D}/wraps_refine2_setB/wrap_openvla_oft.jsonl", f"{D}/wraps/wrap_openvla_oft.jsonl", "H9", "H10"),
             ("run 28 (pi0.5 set A)", f"{D}/wraps_refine2_learned_pi05/wrap_pi05.jsonl", f"{D}/wraps/wrap_pi05.jsonl", "H11", "H12")):
    w, l, n = test(*args)
    tot = [tot[0] + w, tot[1] + l, tot[2] + n]
print("=== POOLED (secondary): %d tasks | repaired %d broken %d | one-sided p %.4f" % (tot[2], tot[0], tot[1], one_sided(tot[0], tot[1])))
