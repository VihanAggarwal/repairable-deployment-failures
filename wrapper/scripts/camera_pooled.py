"""Pooled paired test for the camera claim: does the DEPLOYABLE wrapper beat the unwrapped policy on camera tasks?

Pairs come from two sources, never mixed within a pair:
  subset tasks     baseline rows_oft.jsonl (unwrapped) vs the wrapper run rows.jsonl (arm, default `full`)
  extension tasks  wrap_run rows (wrap_openvla_oft.jsonl): raw vs full on camera tasks outside the subset
Reports rates, Wilson CIs and an exact McNemar test on the pooled discordant pairs, plus the upper-bound arms.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invertibility_report import mcnemar_p  # noqa: E402
from plus_common import load_rows, wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--ext", default="")
    ap.add_argument("--category", default="camera")
    a = ap.parse_args()
    base = {r["task_id"]: bool(r["success"]) for r in load_rows(a.baseline)}
    wrap = {}
    for r in load_rows(a.wrapper):
        if r["category"] == a.category:
            wrap.setdefault(r["arm"], {})[r["task_id"]] = bool(r["success"])
    ext = {}
    for r in (load_rows(a.ext) if a.ext and os.path.exists(a.ext) else []):
        if r["category"] == a.category:
            ext.setdefault(r["arm"], {})[r["task_id"]] = bool(r["success"])

    def report(name, pairs):
        if not pairs:
            return
        n = len(pairs)
        b0, b1 = sum(x for x, _ in pairs), sum(y for _, y in pairs)
        rec = sum(1 for x, y in pairs if not x and y)
        harm = sum(1 for x, y in pairs if x and not y)
        lo0, hi0 = wilson(b0, n)
        lo1, hi1 = wilson(b1, n)
        print(f"{name:<34} n={n:>3}  unwrapped {b0 / n:.2f} [{lo0:.2f},{hi0:.2f}]  arm {b1 / n:.2f} [{lo1:.2f},{hi1:.2f}]  "
              f"recovered {rec} harmed {harm}  McNemar p={mcnemar_p(rec, harm):.4f}")

    print(f"category {a.category}: paired against the unwrapped policy on the same tasks")
    for arm in ("full", "full2", "cam_true_ref", "cam_true", "oracle", "learned", "learned_plane"):
        if arm in wrap:
            report(f"subset: {arm}", [(base[t], ok) for t, ok in wrap[arm].items() if t in base])
    ext_pairs = [(ext["raw"][t], ok) for t, ok in ext.get("full", {}).items() if t in ext.get("raw", {})]
    report("extension: full (raw paired)", ext_pairs)
    sub_full = [(base[t], ok) for t, ok in wrap.get("full", {}).items() if t in base]
    report("POOLED deployable wrapper", sub_full + ext_pairs)


if __name__ == "__main__":
    main()
