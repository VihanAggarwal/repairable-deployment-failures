"""POST HOC (2026-09-24): exact McNemar p-values recomputed from the repaired / broken counts of the paired tables, and a
Holm step-down adjustment over the SECONDARY paired tests (the primary registered tests, score AUROC on the two held-out
families and the pooled OFT camera repair, are reported unadjusted). Counts come from make_results_figure.build_pairs
(result rows) and replicate_tests.py; the lazy and DROID p-values are the recorded ones and are passed through.

Exact two-sided McNemar on R repaired and B broken discordant pairs: p = min(1, 2 * sum_{k <= min(R, B)} C(R + B, k) / 2^(R + B)).
Holm: order the m p-values p_(1) <= ... <= p_(m); adjusted p_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)).

    uv run --no-project --with numpy --with matplotlib python wrapper/scripts/holm_table.py
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "paper"))
import make_results_figure as M  # noqa: E402


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    P = dict((lab, s) for lab, s in M.build_pairs() if s)
    keys = list(P)
    oft_cam = [k for k in keys[:8] if "camera" in k]
    R = sum(P[k]["R"] for k in oft_cam); B = sum(P[k]["B"] for k in oft_cam)
    print(f"PRIMARY pooled OFT camera (four suites): repaired {R}/{sum(P[k]['F'] for k in oft_cam)}, broken {B}/{sum(P[k]['S'] for k in oft_cam)}, p = {mcnemar(R, B):.2g}")
    jt = [k for k in keys[:8] if "joint" in k and "dev" not in k]
    Rj = sum(P[k]["R"] for k in jt); Bj = sum(P[k]["B"] for k in jt)
    tests = [(f"OFT joint, three untuned suites ({Rj}/{Bj})", mcnemar(Rj, Bj))]
    for k in keys:
        s = P[k]
        fam = "pi0.5" if k in keys[8:] else "OFT"
        tests.append((f"{fam} {k} ({s['R']}/{s['B']})", mcnemar(s["R"], s["B"])))
    tests += [("pi0.5 camera, 3 reps, episodes (21/8)", mcnemar(21, 8)), ("pi0.5 light, 3 reps, episodes (3/13)", mcnemar(3, 13)),
              ("pi0.5 lazy vs raw, 170-unit sweep (recorded)", 0.013), ("DROID VLM vs zero-shot masks (recorded)", 0.015)]
    adj = holm([p for _, p in tests])
    print(f"\nSECONDARY paired tests, m = {len(tests)}, Holm-adjusted:")
    for (name, p), a in sorted(zip(tests, adj), key=lambda t: t[0][1]):
        print(f"  {name:<52} p = {p:.2g}   Holm {a:.2g}")


if __name__ == "__main__":
    main()
