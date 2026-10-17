"""POST HOC (2026-09-24, reviewer round 2, items P06/P09): the motion-free check against its components, paired tests.

(1) results/drive_vihan/wraps/wrap_pi05.jsonl, the 40 non-camera pi0.5 tasks (rep 0) run under arms full (measure always),
    full_settle (measure, then settle) and full_lazy (motion-free check first). Per pair of arms (A, B) on the same tasks:
    b = #(A succeeds, B fails), c = #(A fails, B succeeds), exact two-sided McNemar p = min(1, 2 P[Bin(b+c, 1/2) <= min(b, c)]).
    Identity: SR_lazy - SR_full = (SR_lazy - SR_settle) + (SR_settle - SR_full), each SR on the same 40 tasks.
(2) the graded sweep (results/drive_*/sweep/episodes_pi05*.jsonl, every shard merged, rep 0), the units without a camera
    shift that have raw, full and lazy: per base task i, D_i = sum over the task's units of (S_A - S_B) for A in {lazy},
    B in {full, raw}. Exact two-sided sign-flip p = P_{s in {-1,1}^{n0}}(|sum s_i |D_i|| >= |sum D_i|) over the n0 tasks
    with D_i != 0 (all 2^{n0} assignments). Also the sign test (binomial on #D_i > 0 among n0) and Wilcoxon signed-rank.
    Max static residual on these units, from the lazy rows' logged residual.

    uv run --no-project --with numpy --with scipy python wrapper/scripts/motionfree_tests.py
"""
import glob
import itertools
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from plus_common import load_rows, load_shards  # noqa: E402
import sweep_conditions as S  # noqa: E402

R = os.path.join(REPO, "results", "drive_vihan")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def signflip(d):
    nz = np.abs(np.asarray([x for x in d if x != 0], float))
    obs = abs(sum(d))
    return float(np.mean([abs(np.dot(s, nz)) >= obs - 1e-9 for s in itertools.product((-1, 1), repeat=len(nz))])), len(nz)


def part1():
    u = defaultdict(dict)
    for r in load_rows(f"{R}/wraps/wrap_pi05.jsonl"):
        if r.get("rep", 0) == 0 and r.get("category") != "camera":
            u[r["task_id"]].setdefault(r["arm"], int(bool(r["success"])))
    T = [v for v in u.values() if all(a in v for a in ("raw", "full", "full_settle", "full_lazy"))]
    sr = {a: np.mean([v[a] for v in T]) for a in ("raw", "full", "full_settle", "full_lazy")}
    print(f"(1) wrap_pi05 non-camera tasks with all four arms: {len(T)}; SR", {k: round(float(x), 3) for k, x in sr.items()})
    for A, B in (("full_lazy", "full_settle"), ("full_settle", "full"), ("full_lazy", "full"), ("full_lazy", "raw")):
        b = sum(v[A] > v[B] for v in T); c = sum(v[A] < v[B] for v in T)
        print(f"    {A} vs {B}: b = {b} (A wins), c = {c} (B wins), exact McNemar p = {mcnemar(b, c):.4f}; SR diff {sr[A] - sr[B]:+.3f}")
    print(f"    identity: {sr['full_lazy'] - sr['full']:+.3f} = {sr['full_lazy'] - sr['full_settle']:+.3f} + {sr['full_settle'] - sr['full']:+.3f}")


def part2():
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, "results", "drive_*", "sweep", "episodes_pi05.jsonl"))):
        rows += load_shards(p)
    std = {c["name"] for c in S.CONDITIONS}
    cam = {c["name"] for c in S.CONDITIONS if S.BY_NAME[c["name"]]["family"] == "camera"}
    ep = defaultdict(dict)
    for r in rows:
        if r.get("rep", 0) == 0 and r["cond"] in std and r["cond"] not in cam:
            ep[(r["task_id"], r["cond"])].setdefault(r["arm"], r)
    U = {k: v for k, v in ep.items() if all(a in v for a in ("raw", "full", "lazy"))}
    tasks = sorted({k[0] for k in U})
    res = [v["lazy"].get("static_residual") for v in U.values() if v["lazy"].get("static_residual") is not None]
    fams = sorted({S.BY_NAME[k[1]]["family"] for k in U})
    print(f"(2) sweep pi0.5 non-camera units with raw/full/lazy: {len(U)} on {len(tasks)} tasks; families {fams}; "
          f"static residual logged on {len(res)}, max {max(res) if res else float('nan'):.4f}")
    for A, B in (("lazy", "full"), ("lazy", "raw"), ("full", "raw")):
        D = [sum(int(bool(U[k][A]["success"])) - int(bool(U[k][B]["success"])) for k in U if k[0] == t) for t in tasks]
        p, n0 = signflip(D)
        pos, neg = sum(x > 0 for x in D), sum(x < 0 for x in D)
        st = binomtest(pos, pos + neg).pvalue if pos + neg else float("nan")
        try:
            w = wilcoxon([x for x in D if x]).pvalue
        except ValueError:
            w = float("nan")
        eb = sum(U[k][A]["success"] and not U[k][B]["success"] for k in U); ec = sum(U[k][B]["success"] and not U[k][A]["success"] for k in U)
        print(f"    {A} - {B}: D_i = {D}; n0 = {n0}, null size 2^{n0} = {2 ** n0}, min attainable p = {2 / 2 ** n0:.4f}; "
              f"sign-flip p = {p:.5f}; sign test p = {st:.4f}; Wilcoxon p = {w:.4f}; episode b/c = {eb}/{ec}, McNemar p = {mcnemar(eb, ec):.2e}")


if __name__ == "__main__":
    part1()
    part2()
