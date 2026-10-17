"""Paper figures from result files (no simulator). Output: wrapper/paper/iclr/figs/*.pdf and *.png.

    uv run --no-project --with numpy --with matplotlib python wrapper/paper/make_figures.py \
        --organism vendor/organism.jsonl --calib_dir vendor

Palette: two categorical hues checked with the dataviz validator (lightness band, CVD separation, contrast):
blue #0072B2 = raw frame / before, vermillion #D55E00 = corrected frame / after. Theory and medians are drawn in ink,
not in a series colour. One axis per panel, direct labels, recessive grid.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P  # noqa: E402
BLUE, VERM, INK, MUTED = P.FAM["OpenVLA-OFT"], P.FAM["pi0.5"], P.INK, P.MUTED     # shared palette (palette.py)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "pdf.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "iclr", "figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_smc(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "error" not in r and r["archetype"] == "relative"]
    floor = {r["task_id"]: r["smc_raw"]["residual_deg"] for r in rows if r["cond"] == "nominal"}
    pts = [(0.0, floor[t], floor[t], floor[t]) for t in floor]
    for r in rows:
        if r["cond"].startswith("orbit"):
            P, g = np.array(r["P"]), np.array(r["smc_cor"]["goal"])
            b, D = np.mean(np.linalg.norm(P[1:] - P[0], axis=1)), np.linalg.norm(g - P[0])
            th = r["true_shift"][1]
            pts.append((th, r["smc_raw"]["residual_deg"], r["smc_cor"]["residual_deg"],
                        float(np.hypot(floor[r["task_id"]], 0.57 * th * b / D))))
    pts = np.array(sorted(pts))
    xs = np.unique(pts[:, 0])
    mean = lambda col: [pts[pts[:, 0] == x, col].mean() for x in xs]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(xs, mean(3), color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.plot(xs, mean(1), color=BLUE, lw=1.6, zorder=3)
    ax.plot(xs, mean(2), color=VERM, lw=1.6, zorder=3)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=BLUE, edgecolor="white", linewidth=0.6, zorder=4)
    ax.scatter(pts[:, 0], pts[:, 2], s=14, color=VERM, edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(xs[-1] + 1.5, mean(1)[-1], "raw frame", color=INK, ha="left", va="center")
    ax.text(xs[-1] + 1.5, mean(2)[-1], "corrected", color=INK, ha="left", va="center")
    ax.text(xs[-1] + 1.5, mean(3)[-1], "Prop. 2\n(no free\nparameter)", color=MUTED, ha="left", va="center")
    ax.set_xlim(-2, xs[-1] + 16)
    ax.set_xlabel("camera orbit (deg)")
    ax.set_ylabel("self-motion consistency residual (deg)")
    ax.set_xticks(xs)
    ax.set_ylim(0, None)
    save(fig, "smc_dose_response")


def fig_calib(calib_dir):
    colab = {768: 9.5, 800: 11.6, 803: 28.0, 956: 64.8, 958: 63.3, 1060: 2.4}
    def load(tag):
        p = os.path.join(calib_dir, f"local_calib_{tag}.jsonl")
        return {r["task_id"]: r["err_learned"][0] for r in map(json.loads, filter(str.strip, open(p)))}
    variants = [("original", colab), ("+ all mask\nparts", load("fix1")), ("+ 7-pose\nsweep", load("sweep7")),
                ("+ matched\nscoring", load("clean7"))]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    rng = np.random.default_rng(0)
    for i, (name, d) in enumerate(variants):
        v = np.array([d[t] for t in sorted(colab) if t in d])
        ax.scatter(i + 0.12 * rng.standard_normal(len(v)), v, s=16, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=4)
        ax.text(i + 0.32, np.median(v), f"{np.median(v):.0f}", color=INK, va="center", ha="left")
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(1.8, 110)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([n for n, _ in variants])
    ax.set_ylabel("camera pose error, learned mask (cm)")
    ax.set_title("dots: six hard camera tasks, bar: median", loc="left", color=MUTED, fontsize=7)
    ax.set_xlim(-0.5, len(variants) - 0.3)
    save(fig, "learned_calibration")


if __name__ == "__main__":
    import sys
    if "--force-unused" not in sys.argv:
        sys.exit("make_figures.py (smc_dose_response, learned_calibration): its outputs are not in the paper (dropped 2026-09-24); pass --force-unused to rebuild them")
    sys.argv = [a for a in sys.argv if a != "--force-unused"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="vendor/organism.jsonl")
    ap.add_argument("--calib_dir", default="vendor")
    a = ap.parse_args()
    fig_smc(a.organism)
    fig_calib(a.calib_dir)
