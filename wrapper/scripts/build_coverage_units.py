"""Build ONE unit file for every coverage-based number in the paper (Sec. 4.1, Sec. 4.4, Fig. envelope (b)).

Unit = one LIBERO-Plus camera task under one policy, rep 0, with the raw and the `full` arm on the same task and seed.
Sources (the pooled camera set of Table 2 plus the held-out family):
  OFT libero_10 first 30   results/drive_vihan/wrapper_oft/rows.jsonl (arm full) + results/drive_vihan/rows_oft.jsonl (raw)
  OFT libero_10 30 more    results/drive_vihan/wraps/wrap_openvla_oft.jsonl (raw, full)
  OFT object/goal/spatial  results/drive_vihan/wraps_libero_<suite>/wrap_openvla_oft.jsonl (raw, full)
  pi0.5 libero_10          results/drive_vihan/wraps/wrap_pi05.jsonl (raw, full)
Fields: src, policy, task_id, raw, full (successes), coverage (self-measured training-view coverage of the reprojection,
`reproj_coverage` of the full row, None when that run did not log it), shift_cm, shift_deg (true camera shift),
calib_err_cm, calib_err_deg (self-calibration error against the simulator pose), cal_applied.
Writes results/local/coverage_units.jsonl and prints, over the raw FAILURES with coverage logged, the AUROC of coverage,
-translation, -rotation and -calibration error for predicting repair (Mann-Whitney, ties 1/2; 95% percentile bootstrap,
B = 2000, seed 0, over units), a paired bootstrap of AUROC(coverage) - AUROC(rotation), and a reliability table
(equal-mass bins, Wilson intervals), ECE and Brier score of coverage read as a probability.

    uv run --no-project --with numpy python wrapper/scripts/build_coverage_units.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from sweep_analysis import auroc  # noqa: E402

R = os.path.join(REPO, "results", "drive_vihan")
OUT = os.path.join(REPO, "results", "local", "coverage_units.jsonl")


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def unit(src, pol, t, raw, fr):
    sh = fr.get("cam_true_shift_cm_deg") or [None, None]
    e = fr.get("cam_est_err_cm_deg") or [None, None]
    return dict(src=src, policy=pol, task_id=t, raw=bool(raw), full=bool(fr["success"]), coverage=fr.get("reproj_coverage"),
                shift_cm=sh[0], shift_deg=sh[1], calib_err_cm=e[0], calib_err_deg=e[1], cal_applied=bool(fr.get("cal_applied")))


def build():
    U = []
    raw0 = {r["task_id"]: r["success"] for r in rows(f"{R}/rows_oft.jsonl") if r["category"] == "camera"}
    for r in rows(f"{R}/wrapper_oft/rows.jsonl"):
        if r["category"] == "camera" and r["arm"] == "full" and r["task_id"] in raw0:
            U.append(unit("OFT libero_10 (first 30)", "OpenVLA-OFT", r["task_id"], raw0[r["task_id"]], r))
    for src, pol, p in (("OFT libero_10 (30 more)", "OpenVLA-OFT", f"{R}/wraps/wrap_openvla_oft.jsonl"),
                        ("OFT libero_object", "OpenVLA-OFT", f"{R}/wraps_libero_object/wrap_openvla_oft.jsonl"),
                        ("OFT libero_goal", "OpenVLA-OFT", f"{R}/wraps_libero_goal/wrap_openvla_oft.jsonl"),
                        ("OFT libero_spatial", "OpenVLA-OFT", f"{R}/wraps_libero_spatial/wrap_openvla_oft.jsonl"),
                        ("pi0.5 libero_10", "pi0.5", f"{R}/wraps/wrap_pi05.jsonl")):
        by = defaultdict(dict)
        for r in rows(p):
            if r["category"] == "camera" and r.get("rep", 0) == 0 and r["arm"] in ("raw", "full"):
                by[r["task_id"]].setdefault(r["arm"], r)
        for t, v in sorted(by.items()):
            if "raw" in v and "full" in v:
                U.append(unit(src, pol, t, v["raw"]["success"], v["full"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(u) + "\n" for u in U)
    return U


def boot(x, y, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, int)
    v = [auroc(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    v = [a for a in v if np.isfinite(a)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(U, label):
    F = [u for u in U if not u["raw"] and u["coverage"] is not None]
    y = [int(u["full"]) for u in F]
    print(f"\n===== {label}: {len([u for u in U if not u['raw']])} raw failures, {len(F)} with coverage logged, {sum(y)} repaired")
    feats = {"coverage": lambda u: u["coverage"], "-translation": lambda u: -u["shift_cm"] if u["shift_cm"] is not None else np.nan,
             "-rotation": lambda u: -u["shift_deg"] if u["shift_deg"] is not None else np.nan,
             "-calibration error": lambda u: -u["calib_err_cm"] if u["calib_err_cm"] is not None else np.nan}
    res = {}
    for k, fn in feats.items():
        x = np.array([fn(u) for u in F], float)
        ok = np.isfinite(x)
        a = auroc(x[ok], np.asarray(y)[ok])
        lo, hi = boot(x[ok], np.asarray(y)[ok])
        res[k] = (a, lo, hi, int(ok.sum()))
        print(f"  {k:<20} AUROC {a:.3f} [{lo:.2f}, {hi:.2f}]  (n {int(ok.sum())})")
    # paired: coverage vs rotation on units with both
    G = [u for u in F if u["shift_deg"] is not None]
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        s = [G[i] for i in rng.integers(0, len(G), len(G))]
        yy = [int(u["full"]) for u in s]
        d.append(auroc([u["coverage"] for u in s], yy) - auroc([-u["shift_deg"] for u in s], yy))
    d = np.array([v for v in d if np.isfinite(v)])
    yy = [int(u["full"]) for u in G]
    obs = auroc([u["coverage"] for u in G], yy) - auroc([-u["shift_deg"] for u in G], yy)
    print(f"  coverage - rotation (paired, n {len(G)}): {obs:+.3f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}], share <= 0: {np.mean(d <= 0):.3f}")
    # reliability: coverage read as a probability of repair; equal-mass bins
    c = np.array([u["coverage"] for u in F]); yv = np.array(y)
    order = np.argsort(c)
    for B in (4, 5):
        bins = np.array_split(order, B)
        ece = sum(len(b) / len(c) * abs(yv[b].mean() - c[b].mean()) for b in bins)
        tab = [(round(float(c[b].min()), 2), round(float(c[b].max()), 2), int(yv[b].sum()), len(b), round(float(yv[b].mean()), 2),
                tuple(round(v, 2) for v in wilson(int(yv[b].sum()), len(b)))) for b in bins]
        eb = []
        for _ in range(2000):
            i = rng.integers(0, len(c), len(c))
            cc, yy2 = c[i], yv[i]
            o = np.argsort(cc)
            eb.append(sum(len(b) / len(cc) * abs(yy2[b].mean() - cc[b].mean()) for b in np.array_split(o, B)))
        print(f"  reliability, {B} equal-mass bins (cov min, max, repaired, n, rate, Wilson): {tab}")
        print(f"    ECE {ece:.3f} [{np.percentile(eb, 2.5):.3f}, {np.percentile(eb, 97.5):.3f}]")
    print(f"  Brier {np.mean((c - yv) ** 2):.3f} (base-rate Brier {np.mean((yv.mean() - yv) ** 2):.3f})")
    lo_, hi_ = c < 0.6, c >= 0.6
    print(f"  coverage < 0.6: {yv[lo_].sum()}/{lo_.sum()} repaired ({yv[lo_].mean():.2f}); >= 0.6: {yv[hi_].sum()}/{hi_.sum()} ({yv[hi_].mean():.2f})")
    return res


def main():
    U = build()
    print(f"wrote {OUT}: {len(U)} units;", {s: sum(1 for u in U if u['src'] == s) for s in dict.fromkeys(u['src'] for u in U)})
    print("coverage logged per source (all units / raw failures):",
          {s: (sum(1 for u in U if u['src'] == s and u['coverage'] is not None), sum(1 for u in U if u['src'] == s and not u['raw'] and u['coverage'] is not None)) for s in dict.fromkeys(u['src'] for u in U)})
    report(U, "all (four OFT suites + pi0.5)")
    report([u for u in U if u["policy"] == "OpenVLA-OFT"], "OFT only")


if __name__ == "__main__":
    main()
