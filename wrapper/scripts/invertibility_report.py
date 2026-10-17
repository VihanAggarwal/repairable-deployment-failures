"""Claims 1 and 2: how much of the LIBERO-Plus robustness gap a FROZEN policy recovers from measurements of its own
body, per perturbation class, with paired per-task statistics.

  python invertibility_report.py --family openvla_oft:rows_oft.jsonl:wrapper_oft/rows.jsonl --out report.json

Each --family is NAME:BASELINE_ROWS:WRAPPER_ROWS. Rows pair by task_id. The wrapper arm `full` is the deployable
method; `oracle`, `cam_true` and `learned` are reported next to it when present.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows, wilson  # noqa: E402

TAXONOMY = {
    "robot_init": ("invertible", "proprioception: joint angles vs the training start"),
    "camera": ("invertible", "self-calibration: robot silhouette + depth"),
    "light": ("partly invertible", "photometric gain fitted on the robot's own pixels"),
    "sensor_noise": ("partly invertible", "noise floor estimate + denoising"),
    "layout": ("non-invertible", "new object configuration: a capability test"),
    "background": ("non-invertible", "new scene appearance: a capability test"),
    "language": ("non-invertible", "new instruction: a capability test"),
}
ORDER = ["robot_init", "camera", "light", "sensor_noise", "layout", "background", "language"]


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value from the discordant counts b (base fail -> wrapped success) and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def family_report(name, base_path, wrap_path):
    base = {r["task_id"]: bool(r["success"]) for r in load_rows(base_path) if r.get("arm", "raw") == "raw"}
    arms = defaultdict(dict)
    cats = {}
    for r in load_rows(wrap_path):
        arms[r["arm"]][r["task_id"]] = bool(r["success"])
        cats[r["task_id"]] = r["category"]
    for tid, ok in arms.get("full2", {}).items():          # v2 rows stand in for `full` where no `full` row exists
        arms["full"].setdefault(tid, ok)
    out = {"family": name, "categories": {}, "classes": {}}
    print(f"\n== {name} ==")
    print(f"{'category':<13}{'class':<19}{'n':>3}{'base':>7}{'full':>7}{'delta':>7}{'recovered':>11}{'harmed':>8}"
          f"{'p':>8}   upper bounds")
    agg = defaultdict(lambda: [0, 0, 0, 0])            # class -> [base fails, recovered, base successes, harmed]
    for cat in ORDER:
        ids = [t for t, c in cats.items() if c == cat and t in base and t in arms.get("full", {})]
        if not ids:
            continue
        b0 = [base[t] for t in ids]
        f1 = [arms["full"][t] for t in ids]
        rec = sum(1 for x, y in zip(b0, f1) if not x and y)
        harm = sum(1 for x, y in zip(b0, f1) if x and not y)
        fails, succ = b0.count(False), b0.count(True)
        cls = TAXONOMY[cat][0]
        a = agg[cls]
        a[0] += fails; a[1] += rec; a[2] += succ; a[3] += harm
        ub = {}
        for arm in ("oracle", "cam_true", "learned", "learned_plane", "full_learned", "full_plane"):
            v = [arms[arm][t] for t in ids if t in arms.get(arm, {})]
            if v:
                ub[arm] = round(sum(v) / len(v), 3)
        entry = dict(cls=cls, n=len(ids), base=sum(b0) / len(ids), full=sum(f1) / len(ids),
                     full_ci=wilson(sum(f1), len(ids)), recovered=rec, base_fails=fails, harmed=harm,
                     base_successes=succ, mcnemar_p=mcnemar_p(rec, harm), upper_bounds=ub)
        out["categories"][cat] = entry
        print(f"{cat:<13}{cls:<19}{len(ids):>3}{entry['base']:>7.2f}{entry['full']:>7.2f}"
              f"{entry['full'] - entry['base']:>+7.2f}{f'{rec}/{fails}':>11}{f'{harm}/{succ}':>8}"
              f"{entry['mcnemar_p']:>8.3f}   {ub if ub else ''}")
    print("\nper class: share of baseline failures the wrapper recovers, and share of baseline successes it breaks")
    for cls in ("invertible", "partly invertible", "non-invertible"):
        f, r, s, h = agg[cls]
        if f + s == 0:
            continue
        out["classes"][cls] = dict(recovered=r, base_fails=f, harmed=h, base_successes=s)
        print(f"  {cls:<19} recovered {r}/{f} ({r / max(1, f):.0%})   harmed {h}/{s} ({h / max(1, s):.0%})")
    inv = [c for c in out["categories"] if TAXONOMY[c][0] != "non-invertible"]
    non = [c for c in out["categories"] if TAXONOMY[c][0] == "non-invertible"]
    if inv and non:
        gap_b = sum(1 - out["categories"][c]["base"] for c in inv) / len(inv)
        gap_f = sum(1 - out["categories"][c]["full"] for c in inv) / len(inv)
        out["invertible_gap_closed"] = (gap_b - gap_f) / gap_b if gap_b > 0 else None
        out["non_invertible_success"] = sum(out["categories"][c]["full"] for c in non) / len(non)
        print(f"\ninvertible-class failure rate {gap_b:.2f} -> {gap_f:.2f} "
              f"(gap closed {out['invertible_gap_closed']:.0%}); capability-class success after wrapping "
              f"{out['non_invertible_success']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", required=True, help="NAME:BASELINE_ROWS:WRAPPER_ROWS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reports = []
    for spec in a.family:
        name, base_path, wrap_path = spec.split(":", 2)
        reports.append(family_report(name, base_path, wrap_path))
    json.dump(dict(taxonomy=TAXONOMY, families=reports), open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
