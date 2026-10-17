"""Paper figures that need no simulator renders, in the visual language of the group's Functional Invariance paper:
Times throughout, bold "(a) Title" panel heads, a restrained navy / steel palette with one purple accent, a marker shape per
series (readable in greyscale), light row bands and printed estimates with intervals on forest-style panels.

    uv run --no-project --with numpy --with scipy --with matplotlib python wrapper/paper/make_figs_v2.py

Writes wrapper/paper/iclr/figs_v2/<name>.pdf (+ .png preview) and figs_v2/gallery.html. Reads only the results/ snapshot.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import make_paper_figures as P  # noqa: E402  (data loading for the ladder, the score and the tolerance probe)

P.VEN = os.path.join(REPO, "results", "local")      # the local experiments are snapshotted here, not in vendor/
RES, LOC = P.RES, P.VEN
OUT = os.path.join(HERE, "iclr", "figs_v2")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS, SURF, BAND = "#1a1a1a", "#4b5563", "#9ca3af", "#e5e7eb", "#374151", "#ffffff", "#f1f3f6"
NAVY, STEEL, SKY, PURPLE, WARN, LGRAY = "#1f3a5f", "#3b7ea1", "#8db4d3", "#6a4c93", "#b85450", "#d1d5db"
OFT, PI = NAVY, PURPLE
FAM = {"OpenVLA-OFT": OFT, "pi0.5": PI, "base OpenVLA": STEEL}
MARK = {"OpenVLA-OFT": "o", "pi0.5": "D", "base OpenVLA": "s"}
FAMTEX = {"OpenVLA-OFT": "OpenVLA-OFT", "pi0.5": r"$\pi_{0.5}$", "base OpenVLA": "base OpenVLA"}
POLICY = {"OpenVLA-OFT": "openvla_oft", "pi0.5": "pi05", "base OpenVLA": "openvla"}
BUCKET = [("recovered by measurement", "measured pose", NAVY), ("not identified", "only the true pose", STEEL),
          ("not invertible", "only a perfect render", SKY), ("policy", "nothing", WARN)]
BLUES = LinearSegmentedColormap.from_list("navy", ["#ffffff", "#c9dbea", STEEL, NAVY])
PURPLES = LinearSegmentedColormap.from_list("purple", ["#ffffff", "#d9cfe8", "#9b82c0", PURPLE])
TEXTW = 5.5          # ICLR text width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "xtick.color": AXIS, "ytick.color": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.pad": 2,
    "ytick.major.pad": 2, "axes.labelpad": 3, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42, "legend.frameon": False, "figure.dpi": 110,
    "axes.unicode_minus": True, "lines.solid_capstyle": "round",
})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def heads(fig, items, gap=0.08):
    """Bold '(a) Title' centred over each panel, all on one baseline per row."""
    fig.canvas.draw()
    H = fig.get_size_inches()[1]
    for ax, let, title in items:
        p = ax.get_position()
        s = f"({let}) {title}" if let else title
        fig.text((p.x0 + p.x1) / 2, p.y1 + gap / H, s, fontsize=8.5, fontweight="bold", ha="center", va="baseline")


def bands(ax, ys, height=1.0):
    for k, y in enumerate(ys):
        if k % 2 == 0:
            ax.axhspan(y - height / 2, y + height / 2, color=BAND, zorder=0, linewidth=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.0)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.03, dpi=250)
    plt.close(fig)
    print("wrote", name)


def rbox(ax, x, y, w, h, fc, ec="none", lw=0.7, r=1.0, z=2, ls="-"):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec,
                                        linewidth=lw, linestyle=ls, zorder=z))


def arrow(ax, xy0, xy1, col=AXIS, lw=0.8, ls="-", z=3, conn="arc3"):
    ax.annotate("", xy1, xy0, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                              mutation_scale=7, connectionstyle=conn), zorder=z)


def fmt(v, lo, hi, d=2):
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]".replace("-", "−")


# ================================================================ Figure 2: the method (diagram)
def arm_glyph(ax, x, y, s, col, lw=1.1, ls="-", q=(80, -35, -60), z=5):
    """A three-link arm drawn from its joint angles (deg), base at (x, y), link length s: the robot's body as a line drawing."""
    pts = [(x, y)]
    a = 0.0
    for L, d in zip((1.0, 0.85, 0.5), q):
        a += np.radians(d)
        px, py = pts[-1]
        pts.append((px + s * L * np.cos(a), py + s * L * np.sin(a)))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
    ax.scatter(xs[:-1], ys[:-1], s=6, color=col, zorder=z + 1, linewidths=0)
    ex, ey = pts[-1]
    ax.plot([ex - 0.25 * s, ex + 0.25 * s], [ey, ey], color=col, lw=lw, zorder=z)
    ax.plot([x - 0.35 * s, x + 0.35 * s], [y, y], color=col, lw=lw * 1.4, zorder=z)


def frame(ax, x, y, w, h, ec=AXIS, fc="#fafafa", lw=0.7, ls="-"):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))


def fig_method():
    fig, ax = plt.subplots(figsize=(TEXTW, 1.95))
    fig.subplots_adjust(0, 0, 1, 1)
    X = 140.0
    Y = X * 1.95 / TEXTW
    ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.axis("off")
    rbox(ax, 0.4, 0.4, X - 0.8, Y - 0.8, "none", INK, lw=1.0, r=2.2, z=1)
    cols = [4.5, 30.0, 55.5, 81.0, 106.5]
    iy, ty = 34.0, 20.5           # icon centre height, label baseline
    W = 21.0

    def label(x, key, lines, col=INK):
        ax.text(x + W / 2, ty, key, fontsize=8.5, fontweight="bold", ha="center", va="baseline", color=col)
        for k, t in enumerate(lines):
            ax.text(x + W / 2, ty - 3.9 * (k + 1), t, fontsize=7.4, ha="center", va="baseline", color=INK2)

    # 0 Look: one frame, the arm drawn where the training camera would put it (dashed) over the observed arm
    x = cols[0]
    frame(ax, x + 3.5, iy - 7, 14, 14)
    arm_glyph(ax, x + 9, iy - 5.2, 4.4, INK2, q=(80, -35, -60))
    arm_glyph(ax, x + 11.2, iy - 5.2, 4.4, STEEL, ls=(0, (2, 1.2)), q=(80, -35, -60))
    label(x, "Look", ["one frame at rest:", "did the camera move?"])
    # 1 Measure: three known poses, camera pose fitted to the silhouettes
    x = cols[1]
    frame(ax, x + 3.5, iy - 7, 14, 14)
    for q, c in (((95, -50, -40), SKY), ((80, -35, -60), STEEL), ((65, -20, -80), NAVY)):
        arm_glyph(ax, x + 10.5, iy - 5.2, 4.4, c, q=q)
    label(x, "Measure", ["3 arm poses; fit the", "camera to the silhouette"])
    # 2 Predict: recover = ident x invert, a score bar
    x = cols[2]
    ax.text(x + W / 2, iy + 3.2, r"$\mathrm{ident}\times\mathrm{invert}$", fontsize=8.5, ha="center", va="center")
    ax.add_patch(patches.Rectangle((x + 3.5, iy - 4.5), 14, 2.6, facecolor=BAND, edgecolor=AXIS, linewidth=0.6, zorder=3))
    ax.add_patch(patches.Rectangle((x + 3.5, iy - 4.5), 14 * 0.72, 2.6, facecolor=NAVY, edgecolor="none", zorder=4))
    ax.text(x + W / 2, iy - 8.2, "score in [0, 1]", fontsize=6.8, ha="center", va="center", color=INK2)
    label(x, "Predict", ["measurable and", "undoable? No rollout"])
    # 3 Undo: deployed frame re-rendered into the training frame
    x = cols[3]
    ax.add_patch(patches.Polygon([(x + 3, iy - 5), (x + 11, iy - 7), (x + 12, iy + 3), (x + 4, iy + 5)], closed=True,
                                 facecolor="#fafafa", edgecolor=MUTED, linewidth=0.7, zorder=3))
    frame(ax, x + 9.5, iy - 4, 9.5, 9.5, ec=NAVY)
    arrow(ax, (x + 7.5, iy + 6.5), (x + 13.5, iy + 7.2), col=NAVY, conn="arc3,rad=-0.35")
    label(x, "Undo", ["servo the arm home;", "re-render the view"])
    # 4 Verify: warped silhouette against the kinematic one
    x = cols[4]
    frame(ax, x + 3.5, iy - 7, 14, 14)
    arm_glyph(ax, x + 10.2, iy - 5.2, 4.4, NAVY, q=(80, -35, -60))
    arm_glyph(ax, x + 10.9, iy - 5.2, 4.4, STEEL, ls=(0, (2, 1.2)), q=(80, -35, -60))
    ax.text(x + 16.8, iy + 5.2, "F1", fontsize=7, ha="right", va="center", color=STEEL)
    label(x, "Verify", ["is the silhouette where", "the kinematics put it?"])
    for a_, b_ in zip(cols[:-1], cols[1:]):
        arrow(ax, (a_ + W - 1.5, iy), (b_ + 1.5, iy))
    # frozen policy
    px = 130.5
    arrow(ax, (cols[4] + W - 1.5, iy), (px - 0.5, iy))
    rbox(ax, px, iy - 8, 7.8, 16, "#eef2f7", NAVY, lw=0.9, r=1.0, z=3)
    ax.text(px + 3.9, iy + 2.2, "frozen", fontsize=7.4, ha="center", va="center", rotation=90)
    ax.text(px + 3.9, iy - 4.3, r"$\pi$", fontsize=9, ha="center", va="center")
    # the two control paths
    ax.annotate("", (px + 3.9, iy - 8), (cols[0] + W / 2, 8.2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, mutation_scale=7,
                                                                                  connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                                                                  shrinkA=0, shrinkB=0), zorder=2)
    ax.plot([cols[0] + W / 2, cols[0] + W / 2], [8.2, ty - 10.5], color=MUTED, lw=0.8, zorder=2)
    ax.text(62, 5.9, "camera not moved: no arm motion, raw frame to the policy (post hoc)", fontsize=7, color=INK2, ha="center",
            va="top", style="italic")
    ly = iy + 10.2
    ax.plot([cols[4] + W / 2, cols[4] + W / 2], [iy + 8.5, ly], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.plot([cols[4] + W / 2, cols[1] + W / 2], [ly, ly], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    arrow(ax, (cols[1] + W / 2, ly), (cols[1] + W / 2, iy + 8.5), col=MUTED, ls=(0, (3, 2)))
    ax.text((cols[1] + cols[4]) / 2 + W / 2, ly + 1.1, "F1 < 0.75: measure again with 7 poses", fontsize=7, color=INK2,
            ha="center", va="bottom", style="italic")
    save(fig, "fig2_method_icons")      # superseded by make_image_figures.fig_method_pictures (real renders)


# ================================================================ Figure 3: the score predicts repair
BENCH = [[f"{RES}/wrapper_oft/rows.jsonl", f"{RES}/rows_oft.jsonl"], [f"{RES}/wraps/wrap_openvla_oft.jsonl"],
         [f"{RES}/wraps_libero_object/wrap_openvla_oft.jsonl"], [f"{RES}/wraps_libero_goal/wrap_openvla_oft.jsonl"],
         [f"{RES}/wraps_libero_spatial/wrap_openvla_oft.jsonl"], [f"{RES}/wraps/wrap_pi05.jsonl"]]


def bench_failures():
    """Raw camera failures on the benchmark tasks (four OFT suites and pi0.5) whose wrapped run logged the coverage."""
    F = []
    for files in BENCH:
        for v in P.AL.units(files).values():
            if "raw" in v and "full" in v and v["raw"].get("category") == "camera" and not v["raw"]["success"] \
                    and v["full"].get("reproj_coverage") is not None:
                F.append(v)
    return F


def fig_envelope():
    """Main-text Figure: the policy-free score over camera placements, the score against the measured outcome along the
    two directions the sweep moved the camera, and the camera term against repair on the benchmark tasks."""
    env = [json.loads(l) for l in open(os.path.join(LOC, "envelope.jsonl")) if l.strip()]
    az = list(range(-60, 61, 10))
    sc = [0.8, 1.0, 1.3, 1.6]
    g = np.full((len(sc), len(az)), np.nan)
    for i, s_ in enumerate(sc):
        for j, a_ in enumerate(az):
            v = [r["recover"] for r in env if r.get("az") == a_ and r.get("scale") == s_ and "el" not in r]
            if v:
                g[i, j] = np.mean(v)
    orbit = {a_: np.mean([r["recover"] for r in env if r.get("az") == a_ and r.get("scale") == 1.0 and "el" not in r])
             for a_ in range(0, 61, 10)}
    tilt = defaultdict(list)
    for r in env:
        if "tilt" in r:
            tilt[r["tilt"]].append(r["recover"])
    units = P.score_units("openvla_oft") + P.score_units("pi05")
    by = defaultdict(list)
    for u in units:
        by[u["cond"]].append(u["recovered"])
    F = bench_failures()
    yF = [int(v["full"]["success"]) for v in F]
    cov = [v["full"]["reproj_coverage"] for v in F]

    fig = plt.figure(figsize=(TEXTW, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.05, 1.0], wspace=0.55, left=0.075, right=0.99, bottom=0.21, top=0.85)
    # (a) the policy-free map, with the sweep's camera conditions marked
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(g, origin="lower", cmap=BLUES, vmin=0, vmax=1, aspect="auto", extent=(-65, 65, -0.5, len(sc) - 0.5))
    ax.scatter([0], [sc.index(1.0)], marker="+", s=46, color=WARN, linewidths=1.3, zorder=5)
    ax.set_xticks([-60, -30, 0, 30, 60])
    ax.set_yticks(range(len(sc)), [f"{v:g}x" for v in sc])
    ax.set_xlabel("Camera orbit (deg)")
    ax.set_ylabel("Camera distance")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = ax.inset_axes([1.03, 0, 0.04, 1])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.8, length=1.5)
    # (b) predicted score (lines) against observed repair (points, Wilson intervals) along the two camera moves
    bx = fig.add_subplot(gs[1])
    oa = sorted(orbit)
    bx.plot(oa, [orbit[a_] for a_ in oa], color=STEEL, lw=1.3, zorder=3)
    ta = sorted(tilt)
    bx.plot(ta, [np.mean(tilt[t]) for t in ta], color=WARN, lw=1.3, ls=(0, (4, 2)), zorder=3)
    for series, conds, col, m in (("orbit", [(10, "orbit_az10"), (20, "orbit_az20"), (30, "orbit_az30"), (45, "orbit_az45")], STEEL, "o"),
                                  ("look-away", [(35, "look_away_35"), (55, "look_away_55")], WARN, "s")):
        for x_, c in conds:
            k, n = sum(by[c]), len(by[c])
            lo, hi = P.wilson(k, n)
            bx.plot([x_, x_], [lo, hi], color=col, lw=0.9, zorder=3)
            bx.scatter([x_], [k / n], s=22, marker=m, color=col, edgecolor=SURF, linewidth=0.5, zorder=4)
    bx.text(59, orbit[60] + 0.05, "orbit", color=STEEL, fontsize=7.2, ha="right", va="bottom")
    bx.text(20, 0.14, "look-away", color=WARN, fontsize=7.2, ha="center", va="center")   # clear of every mark (searched)
    bx.set_xlim(0, 60); bx.set_ylim(0, 1.02)
    bx.set_xticks([0, 20, 40, 60]); bx.set_yticks([0, 0.5, 1])
    bx.set_xlabel("Rotation (deg)")
    bx.set_ylabel("Share")
    grid(bx)
    # (c) benchmark: coverage against observed repair
    cx = fig.add_subplot(gs[2])
    edges = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001]
    xs, ys, lo, hi = [], [], [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        gg = [yy for c_, yy in zip(cov, yF) if a_ <= c_ < b_]
        if len(gg) >= 3:
            k, n = sum(gg), len(gg)
            w = P.wilson(k, n)
            xs.append(np.mean([c_ for c_ in cov if a_ <= c_ < b_])); ys.append(k / n); lo.append(k / n - w[0]); hi.append(w[1] - k / n)
    cx.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    cx.errorbar(xs, ys, yerr=[lo, hi], color=NAVY, lw=1.2, marker="o", ms=3.4, mec=SURF, mew=0.6, capsize=2, elinewidth=0.8, zorder=4)
    au, l_, h_ = (P.auroc(cov, yF),) + P.boot_ci(cov, yF)
    cx.text(0.04, 0.97, f"AUROC {au:.2f}\n{len(F)} failures", color=INK2, fontsize=7.2, va="top", linespacing=1.15)
    cx.set_xlim(0, 1); cx.set_ylim(0, 1.02)
    cx.set_xticks([0, 0.5, 1]); cx.set_yticks([0, 0.5, 1])
    cx.set_xlabel("View kept (predicted)")
    cx.set_ylabel("Share repaired")
    grid(cx)
    heads(fig, [(ax, "a", "Score over placements"), (bx, "b", "Predicted and observed"), (cx, "c", "Benchmark tasks")])
    save(fig, "fig_envelope")
    return {"bench": (len(F), round(au, 3), round(l_, 3), round(h_, 3)),
            "conds": {c: (sum(v), len(v)) for c, v in sorted(by.items()) if c.startswith(("orbit", "look", "pull"))}}


def fig_score():
    """Appendix: the score on the graded sweep, three families, and what predicts repair on the benchmark tasks."""
    U = {f: P.score_units(POLICY[f]) for f in FAM}
    F = bench_failures()
    yF = [int(v["full"]["success"]) for v in F]
    cov = [v["full"]["reproj_coverage"] for v in F]

    def neg(key, i):
        return [-(v["full"].get(key) or v["raw"].get(key))[i] for v in F]

    def est(x, y):
        return (P.auroc(x, y),) + P.boot_ci(x, y)
    pooled = sum(U.values(), [])
    rows = [("head", "Graded sweep")]
    for f in FAM:
        rows.append((f"score, {FAMTEX[f]}" + ("" if f == "OpenVLA-OFT" else " (held out)"),
                     est([u["recover"] for u in U[f]], [u["recovered"] for u in U[f]]), FAM[f], MARK[f]))
    rows += [("score, pooled", est([u["recover"] for u in pooled], [u["recovered"] for u in pooled]), INK, "o"),
             ("category label, pooled", est([u["label"] for u in pooled], [u["recovered"] for u in pooled]), MUTED, "^"),
             ("head", "Benchmark camera failures"),
             ("view kept (coverage)", est(cov, yF), NAVY, "o"),
             ("shift rotation", est(neg("cam_true_shift_cm_deg", 1), yF), MUTED, "^"),
             ("calibration error", est(neg("cam_est_err_cm_deg", 0), yF), MUTED, "^"),
             ("shift translation", est(neg("cam_true_shift_cm_deg", 0), yF), MUTED, "^")]

    fig = plt.figure(figsize=(TEXTW, 2.3))
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.37, bottom=0.19, top=0.87)
    gc = fig.add_gridspec(1, 1, left=0.66, right=0.8, bottom=0.19, top=0.87)
    ax = fig.add_subplot(gs[0])
    edges = [0.0, 0.3, 0.55, 0.7, 0.85, 1.01]
    for fam, units in U.items():
        xs, ys, lo, hi = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            gg = [u for u in units if a <= u["recover"] < b]
            if len(gg) >= 3:
                k, n = sum(u["recovered"] for u in gg), len(gg)
                w = P.wilson(k, n)
                xs.append(np.mean([u["recover"] for u in gg])); ys.append(k / n); lo.append(w[0]); hi.append(w[1])
        ax.fill_between(xs, lo, hi, color=FAM[fam], alpha=0.08, linewidth=0)
        ax.plot(xs, ys, color=FAM[fam], lw=1.2, marker=MARK[fam], ms=3.4, mec=SURF, mew=0.6, zorder=4, label=FAMTEX[fam])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.25, fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Failures repaired (share)")
    grid(ax)
    cx = fig.add_subplot(gc[0])
    y, ticks, labs, ys_band = 0.0, [], [], []
    for row in rows:
        if row[0] == "head":
            y -= 0.3 if y else 0
            cx.text(-0.06, y, row[1], transform=cx.get_yaxis_transform(), ha="right", va="center", fontsize=7.2, fontweight="bold")
            y -= 1
            continue
        lab, (v, lo_, hi_), col, m = row
        cx.plot([lo_, hi_], [y, y], color=col, lw=1.1, zorder=3, solid_capstyle="butt")
        for e in (lo_, hi_):
            cx.plot([e, e], [y - 0.16, y + 0.16], color=col, lw=1.1, zorder=3)
        cx.scatter([v], [y], s=18, marker=m, color=col, edgecolor=SURF, linewidth=0.6, zorder=4)
        cx.text(1.06, y, fmt(v, lo_, hi_), transform=cx.get_yaxis_transform(), fontsize=7, va="center", ha="left", color=INK)
        ticks.append(y); labs.append(lab); ys_band.append(y)
        y -= 1
    bands(cx, ys_band)
    cx.axvline(0.5, color=AXIS, lw=0.7, zorder=2)
    cx.set_yticks(ticks, labs)
    cx.tick_params(axis="y", length=0, labelsize=7)
    cx.set_ylim(y + 0.5, 0.6)
    cx.set_xlim(0.2, 1.0); cx.set_xticks([0.25, 0.5, 0.75, 1.0]); cx.set_xticklabels(["", "0.5", "", "1"])
    cx.set_xlabel("AUROC (chance 0.5)")
    cx.spines["left"].set_visible(False)
    grid(cx, "x")
    heads(fig, [(ax, "a", "Score against repair, three families"), (cx, "b", "Predicting repair")])
    save(fig, "fig3_score")
    return {r[0]: tuple(round(t, 3) for t in r[1]) for r in rows if r[0] != "head"}


# ================================================================ Figure 4: paired results
PAIRS = [
    ("OpenVLA-OFT", None),
    ("libero-10, joint offset", 30, 0.333, 0.900, "", ""),
    ("libero-spatial, joint offset", 20, 0.250, 0.950, "15/15", "1/5"),
    ("libero-object, joint offset", 20, 0.200, 1.000, "16/16", "0/4"),
    ("libero-goal, joint offset", 20, 0.200, 1.000, "16/16", "0/4"),
    ("libero-10, camera", 60, 0.350, 0.633, "20/39", "3/21"),
    ("libero-spatial, camera", 60, 0.900, 0.883, "3/6", "4/54"),
    ("libero-object, camera", 60, 0.417, 0.750, "26/35", "6/25"),
    ("libero-goal, camera", 60, 0.667, 0.783, "12/20", "5/40"),
    ("pi0.5", None),
    ("libero-10, joint offset", 30, 0.367, 0.900, "16/19", "0/11"),
    ("libero-10, camera, 3 runs", 90, 0.222, 0.367, "21/70", "8/20"),
]      # Table 2 of the paper, libero-10 camera after the 2026-09-24 audit; pi0.5 camera over three replicated runs (Sec. 4.2)


def fig_paired():
    """Raw against wrapped on the same tasks and seeds. The baselines are Table 3 of the paper and are not repeated here."""
    fig = plt.figure(figsize=(TEXTW, 2.45))
    ax = fig.add_axes([0.3, 0.13, 0.42, 0.72])
    y, fam = 0.0, None
    ticks, labels, ys = [], [], []
    tr = ax.get_yaxis_transform()
    for row in PAIRS:
        if row[1] is None:
            fam = row[0]
            y -= 0.2 if y == 0 else 0.5
            ax.text(-0.02, y, FAMTEX[fam], fontsize=8, fontweight="bold", color=FAM[fam], va="center", ha="right", transform=tr)
            y -= 1
            continue
        lab, n, raw, full, rec, brk = row
        c = FAM[fam]
        ax.plot([raw, full], [y, y], color=c, lw=1.0, alpha=0.55, zorder=2)
        ax.scatter([raw], [y], s=18, facecolor=SURF, edgecolor=MUTED, linewidth=1.0, zorder=3, clip_on=False)
        ax.scatter([full], [y], s=20, marker=MARK[fam], facecolor=c, edgecolor=SURF, linewidth=0.5, zorder=4, clip_on=False)
        ax.text(1.17, y, rec, fontsize=7.4, color=INK, va="center", ha="center", transform=tr)
        ax.text(1.36, y, brk, fontsize=7.4, color=INK, va="center", ha="center", transform=tr)
        ticks.append(y); labels.append(lab); ys.append(y)
        y -= 1
    bands(ax, ys)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", labelsize=7.5, length=0)
    ax.set_ylim(y + 0.5, 0.3)
    ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "", "0.5", "", "1"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("Task success")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.text(1.17, 0.35, "repaired", fontsize=7.6, fontweight="bold", ha="center", va="bottom", transform=tr)
    ax.text(1.36, 0.35, "broken", fontsize=7.6, fontweight="bold", ha="center", va="bottom", transform=tr)
    hs = [plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc=SURF, mec=MUTED, mew=1.0, label="raw policy"),
          plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc=OFT, mec=SURF, label="wrapped, OpenVLA-OFT"),
          plt.Line2D([], [], ls="", marker="D", ms=4.0, mfc=PI, mec=SURF, label=r"wrapped, $\pi_{0.5}$")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.4)
    save(fig, "fig4_paired")


# ================================================================ Figure 5: the attribution ladder
def ladder_counts():
    sets = [("libero-10, first 30", "OpenVLA-OFT", [f"{RES}/wrapper_oft/rows.jsonl", f"{RES}/rows_oft.jsonl"]),
            ("libero-10, 30 more", "OpenVLA-OFT", [f"{RES}/wraps/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl"]),
            ("libero-object", "OpenVLA-OFT", [f"{RES}/wraps_libero_object/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_object/wrap_openvla_oft.jsonl"]),
            ("libero-goal", "OpenVLA-OFT", [f"{RES}/wraps_libero_goal/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_goal/wrap_openvla_oft.jsonl"]),
            ("libero-spatial", "OpenVLA-OFT", [f"{RES}/wraps_libero_spatial/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_spatial/wrap_openvla_oft.jsonl"]),
            ("libero-10", "pi0.5", [f"{RES}/wraps/wrap_pi05.jsonl", f"{RES}/wraps_ladder/wrap_pi05.jsonl"])]
    counts, errs = [], defaultdict(list)
    for name, fam, files in sets:
        u = P.AL.units(files)
        fails = [v for v in u.values() if "raw" in v and "full" in v and not v["raw"]["success"]]
        c = defaultdict(int)
        for v in fails:
            b = P.AL.bucket(v)
            c[b] += 1
            e = v["full"].get("cam_est_err_cm_deg")
            if e and fam == "OpenVLA-OFT" and b in {k for k, _, _ in BUCKET}:
                errs[b].append(e[0])
        counts.append((name, fam, c, len(fails)))
    return counts, errs


def fig_ladder():
    counts, errs = ladder_counts()
    fig = plt.figure(figsize=(TEXTW, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.42, left=0.225, right=0.985, bottom=0.13, top=0.72)
    # the ladder as a row of four keys, drawn in a strip above the panels
    sx = fig.add_axes([0.0, 0.84, 1.0, 0.16]); sx.axis("off"); sx.set_xlim(0, 100); sx.set_ylim(0, 10)
    rbox(sx, 0.3, 0.6, 99.4, 8.8, "none", INK, lw=1.0, r=1.4, z=1)
    sx.text(2.2, 5.0, "raw policy\nfails", fontsize=7.6, ha="left", va="center", linespacing=1.1)
    keys = [("measured pose", "the wrapper"), ("true pose", "simulator's camera"),
            ("perfect render", "training view"), ("nothing", "the policy fails")]
    x0, w, gap = 16.0, 18.5, 4.0
    for k, ((_, _, col), (t, sub)) in enumerate(zip(BUCKET, keys)):
        x = x0 + k * (w + gap)
        arrow(sx, (x - gap + 0.8 if k else 11.2, 5.0), (x - 0.8, 5.0))
        sx.add_patch(patches.Rectangle((x, 2.6), 1.4, 4.8, facecolor=col, edgecolor="none", zorder=3))
        sx.text(x + 2.4, 6.2, t, fontsize=8, fontweight="bold", ha="left", va="center")
        sx.text(x + 2.4, 3.6, sub, fontsize=7.2, ha="left", va="center", color=INK2)
    # (a) stacked bars
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(counts))[::-1]
    for yi, (name, fam, c, n) in zip(y, counts):
        left = 0.0
        for key, _, col in BUCKET:
            k = c.get(key, 0)
            if k:
                w_ = 100.0 * k / n
                ax.barh(yi, w_, left=left, height=0.62, color=col, edgecolor=SURF, linewidth=0.8, zorder=3)
                if w_ >= 6.5:
                    ax.text(left + w_ / 2, yi, str(k), ha="center", va="center", fontsize=7.4, color=SURF if col != SKY else INK, zorder=4)
                left += w_
        ax.text(102, yi, f"{n}", va="center", fontsize=7.4, color=INK2)
    ax.text(102, y[0] + 0.62, "n", va="center", fontsize=7.4, color=INK2, fontstyle="italic")
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _, _ in counts])
    ax.set_ylim(-0.6, len(counts) - 0.4)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Raw camera failures (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.transforms import blended_transform_factory
    fig.canvas.draw()
    r_ = fig.canvas.get_renderer()
    x_lab = min(t.get_window_extent(r_).x0 for t in ax.get_yticklabels())          # left edge of the row labels
    x_br = fig.transFigure.inverted().transform((x_lab - 4.0, 0))[0]                # bracket 4 px (about 3 pt) to its left
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for fam, ys in (("OpenVLA-OFT", y[:5]), ("pi0.5", y[5:])):
        ax.plot([x_br, x_br], [ys.min() - 0.3, ys.max() + 0.3], color=FAM[fam], lw=1.3, transform=tr, clip_on=False)
        ax.text(x_br - 0.01, ys.mean(), FAMTEX[fam], rotation=90, ha="right", va="center", fontsize=7.8, color=FAM[fam], transform=tr)
    # (b) calibration error per bucket
    bx = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    bx.axhspan(0.03, 1.0, color=BAND, zorder=0)
    for i, (key, _, col) in enumerate(BUCKET[:3]):
        e = np.asarray(errs.get(key, []), float)
        bx.scatter(i + rng.uniform(-0.2, 0.2, len(e)), np.maximum(e, 0.05), s=7, color=col, alpha=0.85, linewidths=0, zorder=3)
        m = np.median(e)
        bx.plot([i - 0.23, i + 0.23], [m, m], color=INK, lw=1.2, zorder=4)
        bx.text(i + 0.4, m, f"{m:.1f}", fontsize=7.2, color=INK, va="center", ha="left", zorder=5)
    bx.set_yscale("log"); bx.set_ylim(0.04, 200)
    bx.set_yticks([0.1, 1, 10, 100]); bx.set_yticklabels(["0.1", "1", "10", "100"])
    bx.set_xticks(range(3)); bx.set_xticklabels(["measured\npose", "true\npose", "perfect\nrender"])
    bx.set_xlim(-0.5, 2.8)
    bx.tick_params(axis="x", length=0)
    bx.set_ylabel("Calibration error (cm)")
    heads(fig, [(ax, "a", "First arm that repairs each failure"), (bx, "b", "Calibration error")])
    save(fig, "fig5_ladder")
    return counts


# ================================================================ Figure 6: the harm and the motion-free check
def fig_harm():
    u = defaultdict(dict)
    for r in P.load_rows(f"{RES}/wraps/wrap_pi05.jsonl"):
        if r.get("rep", 0) == 0:
            u[r["task_id"]].setdefault(r["arm"], r)
    S_ = [v for v in u.values() if all(a in v for a in ("raw", "full", "full_settle", "full_lazy"))]
    wins = [v for v in S_ if v["raw"]["success"]]
    arms = [("full", "measure always", LGRAY), ("full_settle", "measure, then settle", MUTED),
            ("full_lazy", "measure only if moved", PI)]
    fig = plt.figure(figsize=(TEXTW, 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.55, left=0.2, right=0.99, bottom=0.21, top=0.84)
    ax = fig.add_subplot(gs[0])
    tot = [sum(not v[a]["success"] for v in wins) for a, _, _ in arms]
    yy = np.arange(len(arms))[::-1]
    for y_, (a, lab, col), t in zip(yy, arms, tot):
        ax.barh(y_, t, height=0.58, color=col, zorder=3)
        ax.text(t + 0.25, y_, f"{t} of {len(wins)}", va="center", fontsize=7.6, color=INK, fontweight="bold" if a == "full_lazy" else "normal")
    ax.set_yticks(yy, [lab for _, lab, _ in arms])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 13.5); ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel(r"Raw $\pi_{0.5}$ successes broken")
    ax.spines["left"].set_visible(False)
    grid(ax, "x")
    sr = [json.loads(l) for l in open(os.path.join(LOC, "static_residual.jsonl")) if l.strip()]
    cam = np.array([r["residual"] for r in sr if r["family"] == "camera"])
    oth = np.array([r["residual"] for r in sr if r["family"] != "camera"])
    bx = fig.add_subplot(gs[1])
    rng = np.random.default_rng(1)
    bx.axvspan(0.012, 1.5, color=BAND, zorder=0)
    bx.scatter(np.maximum(oth, 0.02) * np.exp(rng.uniform(-0.25, 0.25, len(oth))), rng.uniform(0.72, 1.28, len(oth)), s=8, color=MUTED,
               marker="o", alpha=0.85, linewidths=0, zorder=3)
    bx.scatter(cam, rng.uniform(1.72, 2.28, len(cam)), s=8, color=PI, marker="D", alpha=0.85, linewidths=0, zorder=3)
    bx.axvline(1.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    bx.text(1.22, 2.62, "skip", fontsize=7.4, ha="right", va="bottom", color=INK2)
    bx.text(1.95, 2.62, "measure", fontsize=7.4, ha="left", va="bottom", color=INK)
    bx.set_xscale("log"); bx.set_xlim(0.012, 80)
    bx.set_xticks([0.02, 0.1, 1, 10]); bx.set_xticklabels(["0", "0.1", "1", "10"])
    bx.set_yticks([1, 2], [f"not moved ({len(oth)})", f"moved ({len(cam)})"])
    bx.set_ylim(0.35, 2.95)
    bx.set_xlabel("Silhouette residual, arm at rest")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    grid(bx, "x")
    heads(fig, [(ax, "a", "Harm without a camera shift"), (bx, "b", "One-frame check: camera moved?")])
    save(fig, "fig6_harm")
    return tot, len(wins)


# ================================================================ Figure 7: the opening action field diverges
def field_jacobian(Pm, A):
    """Probe layout: centre, x+, x-, y+, y-, z+, z-. Central differences of the commanded translation (first 3 action dims)."""
    Pm, A = np.asarray(Pm, float), np.asarray(A, float)[:, :3]
    M = np.zeros((3, 3))
    for j, (a, b) in enumerate(((1, 2), (3, 4), (5, 6))):
        M[:, j] = (A[a] - A[b]) / (Pm[a, j] - Pm[b, j])
    return M


def fig_field():
    rows = {}
    for fam, pol in (("OpenVLA-OFT", "openvla_oft"), ("pi0.5", "pi05")):
        rows[fam] = [json.loads(l) for l in open(f"{RES}/sweep/smc_{pol}.jsonl") if l.strip()]
    Ms = {fam: np.array([field_jacobian(r["P"], r["A_raw"]) for r in rs]) for fam, rs in rows.items()}
    tr = {fam: np.trace(0.5 * (M + M.transpose(0, 2, 1)), axis1=1, axis2=2) for fam, M in Ms.items()}
    fig = plt.figure(figsize=(TEXTW, 1.9))
    gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.01, right=0.6, bottom=0.2, top=0.74)
    gh = fig.add_gridspec(1, 1, left=0.71, right=0.99, bottom=0.2, top=0.74)
    g = np.linspace(-1, 1, 21)
    X, Yg = np.meshgrid(g, g)
    probes = np.array([[0, 0], [0.6, 0], [-0.6, 0], [0, 0.6], [0, -0.6]])

    def field(ax, M2, col, title):
        U, V = M2[0, 0] * X + M2[0, 1] * Yg, M2[1, 0] * X + M2[1, 1] * Yg
        ax.streamplot(X, Yg, U, V, color=col, linewidth=0.6, density=0.7, arrowsize=0.6, arrowstyle="-|>")
        ax.scatter(probes[:, 0], probes[:, 1], s=9, color=INK, zorder=5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(AXIS); sp.set_linewidth(0.7)
        ax.text(0.5, 1.04, title, transform=ax.transAxes, fontsize=7.6, color=col if col != MUTED else INK2, va="bottom", ha="center")

    ax0 = fig.add_subplot(gs[0])
    field(ax0, -np.eye(2), MUTED, "assumed: a goal")
    ax0.scatter([0], [0], marker="*", s=42, color=INK, zorder=6)
    axes = [ax0]
    for k, fam in enumerate(Ms):
        ax = fig.add_subplot(gs[k + 1])
        field(ax, np.median(Ms[fam], 0)[:2, :2], FAM[fam], "measured, " + FAMTEX[fam])
        axes.append(ax)
    cx = fig.add_subplot(gh[0])
    lo_, lim = -3.6, float(np.max(np.concatenate(list(tr.values())))) + 0.25
    bins = np.linspace(lo_, lim, 22)
    for fam, v in tr.items():
        cx.hist(np.clip(v, lo_, lim), bins=bins, color=FAM[fam], alpha=0.35, zorder=3, histtype="stepfilled", linewidth=0)
        cx.hist(np.clip(v, lo_, lim), bins=bins, color=FAM[fam], zorder=4, histtype="step", linewidth=0.9)
    cx.axvline(0, color=INK, lw=0.8, zorder=5)
    top = cx.get_ylim()[1]
    cx.set_ylim(0, top * 1.3); cx.set_xlim(lo_, lim)
    cx.text(lo_ + 0.12, top * 1.26, "converging", fontsize=7.2, color=INK2, va="top", ha="left")
    cx.text(lim * 0.97, top * 1.26, "diverging", fontsize=7.2, color=INK2, va="top", ha="right")
    for j, (fam, v) in enumerate(tr.items()):
        cx.text(lo_ + 0.12, top * (0.82 - 0.15 * j), f"{'OFT' if j == 0 else FAMTEX[fam]}: {100 * np.mean(v > 0):.0f}% > 0",
                fontsize=7.4, color=FAM[fam], va="top", ha="left")
    cx.set_xlabel("Trace of the symmetric part")
    cx.set_ylabel("Probed units")
    grid(cx)
    fig.canvas.draw()
    p0, p2 = axes[0].get_position(), axes[2].get_position()
    Hh = fig.get_size_inches()[1]
    fig.text((p0.x0 + p2.x1) / 2, p0.y1 + 0.27 / Hh, "(a) Opening action field, seen from above", fontsize=8.5, fontweight="bold",
             ha="center", va="baseline")
    pc = cx.get_position()
    fig.text((pc.x0 + pc.x1) / 2, p0.y1 + 0.27 / Hh, "(b) All 144 probed units", fontsize=8.5, fontweight="bold", ha="center",
             va="baseline")
    save(fig, "fig7_field")
    return {f: (len(v), float(np.mean(v > 0))) for f, v in tr.items()}


# ================================================================ Appendix: tolerance, fidelity, maps
def fig_tolerance():
    rows = defaultdict(list)
    for p in glob.glob(f"{RES}/tolerance/tolerance_*.jsonl"):
        for r in P.load_rows(p):
            rows[{"openvla_oft": "OpenVLA-OFT", "pi05": "pi0.5"}.get(r["policy"], r["policy"])].append(r)
    CM, DEG = (0.5, 1, 2, 4, 8, 16), (0.5, 1, 2, 4, 8)
    fig = plt.figure(figsize=(TEXTW, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.12, left=0.085, right=0.99, bottom=0.18, top=0.74)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    for ax, kind, g_ in ((axes[0], "cm", CM), (axes[1], "deg", DEG)):
        for fam, rs in rows.items():
            med, lo, hi = [], [], []
            for d in g_:
                per = [np.median([x["distance"] for x in r["doses"] if x["kind"] == kind and x["dose"] == d]) for r in rs]
                q = np.quantile(per, [0.25, 0.5, 0.75])
                lo.append(q[0]); med.append(q[1]); hi.append(q[2])
            ax.fill_between(g_, lo, hi, color=FAM[fam], alpha=0.1, linewidth=0)
            ax.plot(g_, med, color=FAM[fam], lw=1.3, marker=MARK[fam], ms=3.4, mec=SURF, mew=0.6, zorder=4)
            ax.axhline(np.median([r["uncorrected"] for r in rs]), color=FAM[fam], lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(0.10, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.set_xscale("log"); ax.set_xticks(g_); ax.set_xticklabels([f"{v:g}" for v in g_]); ax.minorticks_off()
        ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        grid(ax)
    a0 = axes[0]
    a0.set_xlim(0.05, 26)
    a0.plot([0.07, 0.54], [0.035, 0.035], color=STEEL, lw=2.4, solid_capstyle="butt", zorder=5)
    a0.text(0.07, 0.125, "our calibration error", fontsize=7.2, color=STEEL, ha="left", va="bottom")
    for name, e, ha in (("learned sensing", 12.6, "right"), ("DROID", 19.6, "left")):
        a0.axvline(e, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        a0.text(e * (0.9 if ha == "right" else 1.1), 0.99, name, fontsize=7, color=INK2, ha=ha, va="top")
    a0.set_xlabel("Camera pose error added (cm)")
    a0.set_ylabel("Opening-action distance")
    axes[1].set_xlim(0.4, 11)
    axes[1].set_xlabel("Camera pose error added (deg)")
    axes[1].set_yticklabels([])
    hs = [plt.Line2D([], [], color=OFT, lw=1.3, marker="o", ms=3.4, mec=SURF, label="OpenVLA-OFT"),
          plt.Line2D([], [], color=PI, lw=1.3, marker="D", ms=3.2, mec=SURF, label=r"$\pi_{0.5}$"),
          plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="no correction"),
          plt.Line2D([], [], color=INK, lw=0.8, ls=(0, (4, 2)), label="training-time level")]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=4, handlelength=2.2, columnspacing=1.6)
    heads(fig, [(axes[0], "a", "Translation error"), (axes[1], "b", "Rotation error")])
    save(fig, "figA_tolerance")


def fig_fidelity():
    rows = [json.loads(l) for l in open(os.path.join(LOC, "fidelity.jsonl")) if l.strip()]
    f = np.array([r["filled"] for r in rows])
    fig, ax = plt.subplots(figsize=(TEXTW * 0.62, 1.9))
    edges = np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001])
    series = [("err_seen", "pixels the camera saw", NAVY, "-", "o"), ("err_hole_ref", "holes, training-view fill", STEEL, "-", "s"),
              ("err_hole_inpaint", "holes, inpainted", STEEL, (0, (3, 2)), "s"), ("err_raw", "no correction", MUTED, "-", "^")]
    ends = []
    for key, label, col, ls, m in series:
        v = np.array([r[key] for r in rows], float)
        xs, med, lo, hi = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mm = (f >= a) & (f < b) & np.isfinite(v)
            if mm.sum() >= 5:
                q = np.quantile(v[mm], [0.25, 0.5, 0.75])
                xs.append(1 - (a + b) / 2); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        o = np.argsort(xs)
        xs, med, lo, hi = (np.asarray(t)[o] for t in (xs, med, lo, hi))
        ax.fill_between(xs, lo, hi, color=col, alpha=0.08, linewidth=0)
        ax.plot(xs, med, color=col, lw=1.2, ls=ls, marker=m, ms=3, mec=SURF, mew=0.5, zorder=4)
        ends.append([xs[-1], med[-1], label, col])
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 7)
    for x, yv, label, col in ends:
        ax.text(x + 0.02, yv, label, fontsize=7.2, color=col if col != MUTED else INK2, va="center")
    ax.set_xlim(0.03, 0.72); ax.set_xticks([0.1, 0.3, 0.5, 0.7])
    ax.set_xlabel("Share of the training view lost")
    ax.set_ylabel("Error vs. perfect render")
    ax.set_ylim(0, 90)
    grid(ax)
    save(fig, "figA_fidelity")


def fig_maps():
    env = [json.loads(l) for l in open(os.path.join(LOC, "envelope.jsonl")) if l.strip()]
    fid = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(os.path.join(LOC, "fidelity.jsonl")))}
    az = list(range(-60, 61, 10))
    sc = [0.8, 1.0, 1.3, 1.6]
    G = {k: np.full((len(sc), len(az)), np.nan) for k in ("sigma", "coverage", "recover", "err_seen")}
    for i, s_ in enumerate(sc):
        for j, a in enumerate(az):
            rs = [r for r in env if r.get("az") == a and r.get("scale") == s_ and "el" not in r]
            if rs:
                G["recover"][i, j] = np.mean([r["recover"] for r in rs])
                G["coverage"][i, j] = np.mean([r["coverage"] for r in rs])
                G["sigma"][i, j] = np.mean([min(1.0, r["sigma_min"] / r["sigma_min_ref"]) for r in rs])
                fs = [fid[(r["task_id"], r["cond"])]["err_seen"] for r in rs if (r["task_id"], r["cond"]) in fid]
                if fs:
                    G["err_seen"][i, j] = np.mean(fs)
    panels = [("sigma", "Identifiability", BLUES, (0, 1)), ("coverage", "View kept", BLUES, (0, 1)),
              ("recover", "Score", BLUES, (0, 1)), ("err_seen", "Error on seen pixels", PURPLES, (0, 40))]
    fig = plt.figure(figsize=(TEXTW, 1.55))
    gs = fig.add_gridspec(1, 4, wspace=0.5, left=0.07, right=0.95, bottom=0.25, top=0.8)
    hd = []
    for k, (key, title, cmap, (lo, hi)) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        im = ax.imshow(G[key], cmap=cmap, vmin=lo, vmax=hi, aspect="auto", origin="lower", extent=(-65, 65, -0.5, len(sc) - 0.5))
        ax.set_xticks([-60, 0, 60])
        ax.set_yticks(range(len(sc)))
        ax.set_yticklabels([f"{v:g}x" for v in sc] if k == 0 else [])
        ax.set_xlabel("Orbit (deg)")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.scatter([0], [sc.index(1.0)], marker="+", s=30, color=WARN, linewidths=1.1, zorder=5)
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        cb = fig.colorbar(im, cax=cax, ticks=[lo, (lo + hi) / 2, hi])
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.8, length=1.5)
        if k == 0:
            ax.set_ylabel("Distance")
        hd.append((ax, "abcd"[k], title))
    heads(fig, hd)
    save(fig, "figA_maps")


def gallery(notes):
    names = ["fig2_method", "fig_envelope", "fig3_score", "fig4_paired", "fig5_ladder", "fig6_harm", "fig7_field", "figA_tolerance", "figA_fidelity", "figA_maps"]
    body = "".join(f'<figure><img src="{n}.png"><figcaption><b>{n}</b></figcaption></figure>' for n in names
                   if os.path.exists(os.path.join(OUT, n + ".png")))
    html = ("<!doctype html><meta charset='utf-8'><title>Paper figures v2</title><style>body{font:15px/1.5 Georgia,serif;max-width:1000px;"
            "margin:32px auto;padding:0 16px;color:#111;background:#f5f4f0}figure{margin:0 0 36px;background:#fff;padding:18px;"
            "border:1px solid #e1e0d9;border-radius:6px}img{width:100%;height:auto}figcaption{margin-top:8px;color:#55544f}"
            "pre{font-size:12px;color:#55544f;white-space:pre-wrap}</style><h1 style='font-weight:normal'>Paper figures</h1>"
            f"<pre>{notes}</pre>" + body)
    open(os.path.join(OUT, "gallery.html"), "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    notes = []
    for fn in (fig_envelope, fig_score, fig_paired, fig_ladder, fig_harm, fig_field, fig_tolerance, fig_fidelity, fig_maps):
        try:
            r = fn()
            if r is not None:
                notes.append(f"{fn.__name__}: {r}")
        except Exception as e:      # one broken figure must not stop the others
            import traceback
            print("FAILED", fn.__name__, repr(e))
            traceback.print_exc()
    print("\n".join(notes))
    gallery("\n".join(notes))
