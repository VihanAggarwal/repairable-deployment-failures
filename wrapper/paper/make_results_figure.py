"""Figure: paired results. Left: raw policy -> wrapper, per suite and nuisance (every pair is the same tasks, same seeds).
Right: the wrapper against baselines on the 30 extra libero_10 camera tasks, with Wilson intervals.

Every number is computed here from the result rows (no hard-coded rates). Arm per source, fixed by one rule (the latest
code version of the method on that source):
  libero_10, first 30 tasks   results/drive_vihan/wrapper_oft/rows.jsonl (run of 2026-09-15), paired on task_id with the
                              raw rows results/drive_vihan/rows_oft.jsonl; camera arm `full`, joint arm `full2`
  libero_10, 30 more tasks    results/drive_vihan/wraps/wrap_openvla_oft.jsonl, arms raw / full, paired on (task_id, rep)
  spatial / object / goal     results/drive_vihan/wraps_libero_<suite>/wrap_openvla_oft.jsonl, arms raw / full
  pi0.5 (held out)            results/drive_vihan/wraps/wrap_pi05.jsonl, arms raw / full, rep 0
libero_10 is the development suite. Repaired R = raw failed and wrapped succeeded, broken B = raw succeeded and wrapped
failed, so SR_wrapped - SR_raw = (R - B) / N on N paired units.

    uv run --no-project --with numpy --with matplotlib python wrapper/paper/make_results_figure.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "results", "drive_vihan")
import sys as _sys
_sys.path.insert(0, HERE)
import palette as P  # noqa: E402
BLUE, GREY, INK = P.FAM["OpenVLA-OFT"], P.MUTED, P.INK     # shared palette (palette.py)
CATS = {"joint offset": "robot_init", "camera": "camera"}


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def paired_wraps(path, cat, arm="full"):
    """{(task, rep): (raw_success, wrapped_success)} from a wraps file holding both arms."""
    u = defaultdict(dict)
    for r in rows(path):
        if r["category"] == cat:
            u[(r["task_id"], r.get("rep", 0))].setdefault(r["arm"], bool(r["success"]))
    return {k: (v["raw"], v[arm]) for k, v in u.items() if "raw" in v and arm in v}


def paired_first30(cat, arm):
    raw = {r["task_id"]: bool(r["success"]) for r in rows(f"{RES}/rows_oft.jsonl") if r["category"] == cat}
    return {(r["task_id"], 0): (raw[r["task_id"]], bool(r["success"])) for r in rows(f"{RES}/wrapper_oft/rows.jsonl")
            if r["category"] == cat and r["arm"] == arm and r["task_id"] in raw}


def summarise(units):
    v = list(units.values())
    n = len(v)
    rep = sum(1 for a, b in v if not a and b)
    brk = sum(1 for a, b in v if a and not b)
    F = sum(1 for a, _ in v if not a)
    return dict(n=n, raw=sum(a for a, _ in v) / n, wr=sum(b for _, b in v) / n, R=rep, F=F, B=brk, S=n - F)


def libero10(cat):
    u = paired_first30(cat, "full" if cat == "camera" else "full2")
    if cat == "camera":            # both 30-task sets
        u.update({("more",) + k: v for k, v in paired_wraps(f"{RES}/wraps/wrap_openvla_oft.jsonl", cat).items()})
    return u


def build_pairs():
    P = [("OpenVLA-OFT", None)]
    for nice, cat in CATS.items():
        P.append((f"libero-10 (dev.) · {nice}", summarise(libero10(cat))))
        for s in ("spatial", "object", "goal"):
            P.append((f"libero-{s} · {nice}", summarise(paired_wraps(f"{RES}/wraps_libero_{s}/wrap_openvla_oft.jsonl", cat))))
    P.append(("$\\pi_{0.5}$ (held-out family)", None))
    for nice, cat in CATS.items():
        P.append((f"libero-10 · {nice}", summarise(paired_wraps(f"{RES}/wraps/wrap_pi05.jsonl", cat))))
    return P


def build_base():
    u = defaultdict(dict)
    for r in rows(f"{RES}/wraps/wrap_openvla_oft.jsonl"):
        if r["category"] == "camera" and r.get("rep", 0) == 0:
            u[r["task_id"]][r["arm"]] = bool(r["success"])
    tasks = [t for t, v in u.items() if "raw" in v]
    out = []
    for arm, lab in (("raw", "raw policy"), ("generic", "generic image fixes"), ("full_homog", "homography (no depth)"),
                     ("tta", "test-time augment."), ("full", "wrapper")):
        k = sum(u[t][arm] for t in tasks if arm in u[t])
        n = sum(1 for t in tasks if arm in u[t])
        out.append((lab, k, n))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, mid - half, mid + half


def main():
    PAIRS, BASE = build_pairs(), build_base()
    for lab, s in PAIRS:
        if s:
            print(f"{lab:40s} n={s['n']:3d} raw {s['raw']:.3f} -> {s['wr']:.3f}  +{s['R']}/{s['F']}  -{s['B']}/{s['S']}")
    print("baselines:", BASE)
    fig = plt.figure(figsize=(5.5, 2.6))
    ax = fig.add_axes([0.255, 0.16, 0.25, 0.74])
    bx = fig.add_axes([0.835, 0.16, 0.15, 0.74])
    y, ticks, labels = 0, [], []
    for lab, s in PAIRS:
        if s is None:
            ax.text(-0.02, y, lab, fontsize=7.5, fontweight="bold", color=INK, va="center", ha="right", transform=ax.get_yaxis_transform())
            y -= 1
            continue
        raw, full = s["raw"], s["wr"]
        ax.plot([raw, full], [y, y], color=BLUE if full >= raw else GREY, lw=1.6, alpha=0.55, solid_capstyle="round", zorder=1)
        ax.scatter([raw], [y], s=22, facecolor="white", edgecolor=GREY, linewidth=1.1, zorder=2)
        ax.scatter([full], [y], s=26, facecolor=BLUE, edgecolor=BLUE, zorder=3)
        ax.text(1.04, y, f"n={s['n']}  +{s['R']}/{s['F']}  $-${s['B']}/{s['S']}", fontsize=7, color="#444444", va="center")
        ticks.append(y); labels.append(lab)
        y -= 1
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_xlim(0, 1.03); ax.set_ylim(y + 0.3, 0.8)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("task success")
    ax.scatter([], [], s=22, facecolor="white", edgecolor=GREY, linewidth=1.1, label="raw policy")
    ax.scatter([], [], s=26, facecolor=BLUE, edgecolor=BLUE, label="same policy, wrapped")
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    for i, (lab, k, n) in enumerate(BASE):
        p, lo, hi = wilson(k, n)
        ours = lab.startswith("wrapper")
        bx.plot([lo, hi], [i, i], color=BLUE if ours else GREY, lw=1.4, alpha=0.6)
        bx.scatter([p], [i], s=26, facecolor=BLUE if ours else "white", edgecolor=BLUE if ours else GREY, linewidth=1.1, zorder=3)
        bx.text(hi + 0.03, i, f"{k}/{n}", fontsize=7, va="center", ha="left", color=INK)
    bx.set_yticks(range(len(BASE)), [b[0] for b in BASE], fontsize=7)
    bx.set_xlim(0, 1.0); bx.set_xticks([0, 0.5, 1.0]); bx.set_ylim(-0.6, len(BASE) - 0.4)
    bx.set_xlabel(f"success, {BASE[0][2]} libero-10\ncamera tasks (95% Wilson)")
    bx.grid(axis="x", color="#e6e6e6", lw=0.6); bx.set_axisbelow(True)
    for a in (ax, bx):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.tick_params(length=0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "iclr", "figs", f"paired_results.{ext}"), dpi=260, bbox_inches="tight")
    print("ok")


if __name__ == "__main__":
    import sys
    if "--force-unused" not in sys.argv:
        sys.exit("make_results_figure.py (paired_results): its outputs are not in the paper (dropped 2026-09-24); pass --force-unused to rebuild them")
    main()
