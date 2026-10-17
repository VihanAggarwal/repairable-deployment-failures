"""At-a-glance results figure (Fig. 2): (a) raw vs corrected success per suite (Table 2 rows), (b) where the 100 OFT
camera failures go (attribution ladder, Sec. 4.4). Numbers are the paper's table values, not recomputed here.

    uv run --no-project --with matplotlib python wrapper/paper/make_glance_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import palette as P  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iclr", "figs")
P.apply_ml()

# (label, raw, wrapped, family) from Table 2 (tab:suites)
ROWS = [
    ("joint offset, OFT", 0.26, 0.96, "OFT"),
    (r"joint offset, $\pi_{0.5}$", 0.37, 0.90, "pi0.5"),
    ("camera, OFT Spatial", 0.90, 0.88, "OFT"),
    ("camera, OFT Object", 0.42, 0.75, "OFT"),
    ("camera, OFT Goal", 0.67, 0.78, "OFT"),
    ("camera, OFT LIBERO-10", 0.35, 0.63, "OFT"),
    (r"camera, $\pi_{0.5}$ LIBERO-10", 0.20, 0.37, "pi0.5"),
]
# attribution of the 100 OFT camera failures (Fig. 5 / Sec. 4.4)
LADDER = [("repaired by the\nmeasured correction", 61, P.BLUE[3]), ("failed\nmeasurement", 17, P.BLUE[2]),
          ("lost view", 21, P.BLUE[1]), ("policy", 1, P.BLUE[0])]

fig = plt.figure(figsize=(P.TEXTW, 1.75))
fig.set_layout_engine(None)
ax = fig.add_axes([0.235, 0.22, 0.4, 0.68])
for i, (lab, r, w, fam) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    c = P.ML_FAM[fam]
    ax.plot([r, w], [y, y], color=P.BLUE[3], lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter([r], [y], s=16, facecolor="white", edgecolor=c, lw=1.0, marker=P.MARK[fam], zorder=2)
    ax.scatter([w], [y], s=16, color=c, marker=P.MARK[fam], zorder=3)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in ROWS][::-1])
ax.set_xlim(0, 1.02)
ax.set_xlabel("task success (hollow raw, filled corrected)")
ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.grid(axis="x", alpha=0.3, lw=0.5)
ax.grid(axis="y", visible=False)
P.panel_label(ax, "a", dx=-0.58, dy=1.02)

bx = fig.add_axes([0.83, 0.22, 0.14, 0.68])
LAB = ["repaired", "failed measurement", "lost view", "policy"]
for i, ((_, n, col), lab) in enumerate(zip(LADDER, LAB)):
    y = len(LADDER) - 1 - i
    bx.barh(y, n, color=[P.BLUE[1], P.BLUE[2], P.BLUE[3], P.BLUE[0]][i], height=0.6)
    bx.text(n + 2, y, str(n), va="center", ha="left", fontsize=7)
bx.set_yticks(range(len(LADDER)))
bx.set_yticklabels(LAB[::-1])
bx.set_xlim(0, 78)
bx.set_ylim(-0.6, len(LADDER) - 0.4)
bx.set_xlabel("OFT camera failures")
bx.grid(axis="y", visible=False)
bx.grid(axis="x", alpha=0.3, lw=0.5)
P.panel_label(bx, "b", dx=-1.2, dy=1.02)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"glance.{ext}"), dpi=300)
print("saved glance")
