"""Tests pre-registered in wrapper/prereg/2026-09-25_refine2.md (run files 25 and 26), on every shard of each folder.

H5: run 25, full_hybrid_refine2 > raw (one-sided exact McNemar, alpha 0.05). H6: wins >= losses against full_hybrid
(plus/wraps). H7: run 26, full_refine2 > raw over all pi0.5 camera tasks (set A raw from plus/wraps). H8: `full` > raw on
the further tasks, and pooled with set A.

    uv run --no-project --with numpy python wrapper/scripts/prereg_refine2.py [data_dir]
"""
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plus_common import load_shards  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/vihan/Downloads/local_data/plus"


def one_sided(b, c):
    """P(X >= b) for X ~ Bin(b + c, 1/2): exact McNemar, one-sided, wins b and losses c."""
    n = b + c
    return sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0


def cells(rows, cat="camera"):
    out = {}
    for r in rows:
        if r.get("category") == cat and r.get("rep", 0) == 0:
            out.setdefault(r["task_id"], {}).setdefault(r["arm"], r)
    return out


def paired(by, a, b, ids=None):
    t = [i for i in (ids if ids is not None else by) if a in by.get(i, {}) and b in by.get(i, {})]
    w = sum(bool(by[i][a]["success"]) and not bool(by[i][b]["success"]) for i in t)
    l = sum(bool(by[i][b]["success"]) and not bool(by[i][a]["success"]) for i in t)
    return len(t), sum(bool(by[i][b]["success"]) for i in t), sum(bool(by[i][a]["success"]) for i in t), w, l


def main():
    print("=== run 25: OFT set A, learned sensing + refine2")
    r25 = cells(load_shards(f"{D}/wraps_refine2/wrap_openvla_oft.jsonl"))
    old = cells(load_shards(f"{D}/wraps/wrap_openvla_oft.jsonl"))
    for t, v in old.items():
        if t in r25 and "full_hybrid" in v:
            r25[t]["full_hybrid"] = v["full_hybrid"]
    e = np.array([v["full_hybrid_refine2"]["cam_est_err_cm_deg"][0] for v in r25.values()
                  if "full_hybrid_refine2" in v and v["full_hybrid_refine2"].get("cam_est_err_cm_deg")])
    print("tasks with refine2 rows %d | rollout calibration error median %.2f cm, within 1 cm %d, within 2 cm %d" % (
        sum("full_hybrid_refine2" in v for v in r25.values()), np.median(e), (e <= 1).sum(), (e <= 2).sum()))
    n, s0, s1, w, l = paired(r25, "full_hybrid_refine2", "raw")
    print("H5 refine2 vs raw: n %d | raw %d -> refine2 %d | repaired %d broken %d | one-sided p %.4f -> %s" % (
        n, s0, s1, w, l, one_sided(w, l), "PASS" if one_sided(w, l) < 0.05 else "FAIL"))
    n, s0, s1, w, l = paired(r25, "full_hybrid_refine2", "full_hybrid")
    print("H6 refine2 vs full_hybrid: n %d | hybrid %d -> refine2 %d | wins %d losses %d -> %s" % (
        n, s0, s1, w, l, "PASS" if w >= l else "FAIL"))

    print("=== run 26: pi0.5 camera, refined pose (simulator sensing)")
    new = cells(load_shards(f"{D}/wraps_refine2_pi05/wrap_pi05.jsonl"))
    base = cells(load_shards(f"{D}/wraps/wrap_pi05.jsonl"))
    further = {t for t, v in new.items() if "raw" in v}
    by = {}
    for t in set(new) | set(base):
        by[t] = dict(base.get(t, {}))
        by[t].update(new.get(t, {}))
    setA = [t for t in by if t not in further and "full_refine2" in by[t]]
    for name, ids in (("set A", setA), ("further", sorted(further)), ("all", setA + sorted(further))):
        n, s0, s1, w, l = paired(by, "full_refine2", "raw", ids)
        tag = " <- H7 (primary)" if name == "all" else ""
        print("refine2 vs raw, %-8s n %2d | raw %2d -> refine2 %2d | repaired %d broken %d | one-sided p %.4f%s" % (
            name, n, s0, s1, w, l, one_sided(w, l), tag))
    for name, ids in (("further", sorted(further)), ("all", setA + sorted(further))):
        n, s0, s1, w, l = paired(by, "full", "raw", ids)
        print("H8 full vs raw,  %-8s n %2d | raw %2d -> full %2d | repaired %d broken %d | one-sided p %.4f" % (
            name, n, s0, s1, w, l, one_sided(w, l)))
    print("further tasks with all three arms: %d of %d started" % (
        sum(all(k in by[t] for k in ("raw", "full", "full_refine2")) for t in further), len(further)))


if __name__ == "__main__":
    main()
