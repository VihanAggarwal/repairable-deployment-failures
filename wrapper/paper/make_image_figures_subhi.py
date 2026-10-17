# Subhi's redraw of make_image_figures.py (commit b0503bb), kept verbatim under this name when it conflicted with
# the critique-workflow rewrite of make_image_figures.py; only the default output folder changed (now figs_v2/).
"""Image figures from the local renders (vendor/render_figures.py -> figs/renders/*.npz) plus two maps from the envelope
rows and a method schematic. No simulator needed here.

    uv run --no-project --with numpy --with matplotlib --with opencv-python-headless python wrapper/paper/make_image_figures.py
"""
import glob
import json
import os
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(HERE, "iclr", "figs")
REN = os.path.join(FIGS, "renders")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
                     "mathtext.fontset": "stix",
                     "text.color": INK, "axes.edgecolor": "#c3c2b7", "figure.facecolor": SURF, "savefig.facecolor": SURF,
                     "pdf.fonttype": 42, "legend.frameon": False})
TEXTW = 5.5


def load(name):
    p = os.path.join(REN, name + ".npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def rgb(img):
    """Raw simulator buffers are upside down; every frame and mask is turned 180 degrees for display."""
    a = np.asarray(img)
    a = a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, -1)
    return np.ascontiguousarray(a[::-1, ::-1])


def flipm(mask):
    return np.asarray(mask)[::-1, ::-1]


def clean(mask, frac=0.03):
    """Keep the connected components of the robot mask that are at least `frac` of the largest one (drops stray mount geometry)."""
    m = np.asarray(mask).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m > 0
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= frac * areas.max()]
    return np.isin(lab, keep)


def contour(img, mask, color, width=2):
    out = rgb(img).copy()
    m = cv2.resize(flipm(clean(mask)).astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    col = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    cv2.drawContours(out, cs, -1, col, width)
    return out


def tint(img, mask, color, alpha=0.45):
    out = rgb(img).astype(np.float32)
    m = cv2.resize(flipm(clean(mask)).astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    col = np.array([int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * col
    return out.astype(np.uint8)


def tile(ax, img, title=None, sub=None):
    ax.imshow(rgb(img))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=INK2)


OUTDIR = os.environ.get("FIGS_OUT", os.path.join(HERE, "iclr", "figs_v2"))   # merge 2026-09-24: default moved off figs/ so this redraw cannot overwrite the paper figures      # FIGS_OUT=... writes somewhere else, e.g. a preview folder


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"), bbox_inches="tight", pad_inches=0.0)
    fig.savefig(os.path.join(OUTDIR, name + ".png"), bbox_inches="tight", pad_inches=0.04, dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- picture figures: shared header-bar style
# Every image tile carries a dark bar with a title line and one metric line (the style of the DROID figure); nothing is
# written under an image. Images passed to ptile are already in display orientation (rgb(), contour() and tint() turn
# the raw buffers; turning them again put the old teaser and static_check upside down).
BAR, BAR_SUB, OK_ON_BAR, BAD_ON_BAR = "#1d1d1f", "#b9b8b2", "#6ad8a6", "#ff8a80"
NAVY, STEEL, PURPLE, MUTED2 = "#1f3a5f", "#3b7ea1", "#6a4c93", "#9ca3af"
SIL = BLUE          # the robot's rendered silhouette, blue in every picture figure (as in the DROID figure)
HB = 0.30           # header height, as a fraction of the tile height
TEASER_AUROC = [(r"score, $\pi_{0.5}$", 0.809, (0.69, 0.91), PURPLE, "D"),
                ("score, base OpenVLA", 0.675, (0.52, 0.82), STEEL, "s"),
                (r"label, $\pi_{0.5}$", 0.515, None, MUTED2, "^"),
                ("label, base OpenVLA", 0.472, None, MUTED2, "^")]   # Table 1, the two held-out families


def header(ax, title, sub, sub_color=BAR_SUB, h=HB, fs=(7.4, 6.8)):
    """Dark bar on top of an image axis (axes coordinates: the bar sits above y = 1)."""
    ax.add_patch(patches.Rectangle((0, 1), 1, h, transform=ax.transAxes, facecolor=BAR, edgecolor="none", clip_on=False))
    ax.text(0.04, 1 + h * 0.70, title, transform=ax.transAxes, ha="left", va="center", fontsize=fs[0], color="white")
    ax.text(0.04, 1 + h * 0.28, sub, transform=ax.transAxes, ha="left", va="center", fontsize=fs[1], color=sub_color)


def ptile(ax, img, title, sub, sub_color=BAR_SUB, cmap=None, vmin=None, vmax=None):
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    header(ax, title, sub, sub_color)


def picture_grid(rows, name, width=TEXTW):
    """rows: lists of (img, title, sub[, sub_color[, imshow kwargs]]). Square tiles, one header bar each."""
    nr, nc = len(rows), max(len(r) for r in rows)
    gap = 0.035
    tw = width / (nc + gap * (nc - 1))
    H = nr * tw * (1 + HB) + (nr - 1) * tw * 0.1 + 0.02
    fig = plt.figure(figsize=(width, H))
    gs = fig.add_gridspec(nr, nc, wspace=gap, hspace=(HB + 0.1), left=0, right=1, bottom=0.02 / H, top=1 - tw * HB / H)
    for i, row in enumerate(rows):
        for j, it in enumerate(row):
            img, t, sub = it[:3]
            sc = it[3] if len(it) > 3 and it[3] else BAR_SUB
            kw = it[4] if len(it) > 4 else {}
            ptile(fig.add_subplot(gs[i, j]), img, t, sub, sc, **kw)
    save(fig, name)


# ---------------------------------------------------------------- teaser: measure, undo, verify in pictures
def teaser_layout(rows):
    """rows: dicts with display-ready images (train, deployed, measured, corrected) and the strings for the headers.
    Left (a): one recoverable and one unrecoverable camera shift, four tiles each. Right (b): the headline numbers."""
    fig = plt.figure(figsize=(TEXTW, 2.62))
    gs = fig.add_gridspec(len(rows), 6, width_ratios=[1, 1, 1, 1, 0.1, 1.3], wspace=0.035, hspace=0.4,
                          left=0.03, right=0.99, top=0.875, bottom=0.09)
    for i, r in enumerate(rows):
        ok = r["recoverable"]
        tiles = [(r["train"], "training view", "what the policy saw", BAR_SUB),
                 (r["deployed"], "deployed view", r["shift"], BAR_SUB),
                 (r["measured"], "self-measured", r["cal"], BAR_SUB),
                 (r["corrected"], "given to policy", r["verdict"], OK_ON_BAR if ok else BAD_ON_BAR)]
        for j, (img, t, s, sc) in enumerate(tiles):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            header(ax, t, s, sc, fs=(7.0, 6.6))
            if i == 0 and j == 0:
                p0 = ax.get_position()
                fig.text(p0.x0 - 0.008, p0.y1 + HB * p0.height, "a", fontsize=9, fontweight="bold", ha="right", va="top")
    # (b) the headline numbers, as a forest plot: estimate and 95% interval on the two held-out families
    ax = fig.add_subplot(gs[:, 5])
    n = len(TEASER_AUROC)
    ys = np.arange(n)[::-1].astype(float)
    for k, y in enumerate(ys):
        if k % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f1f3f6", zorder=0, linewidth=0)
    for y, (lab, v, ci, col, m) in zip(ys, TEASER_AUROC):
        ax.plot([0.5, 0.5], [y - 0.5, y + 0.02], color=INK2, lw=0.7, zorder=1)      # chance, kept off the label line
        if ci:
            ax.plot(ci, [y - 0.14, y - 0.14], color=col, lw=1.1, zorder=2, solid_capstyle="butt")
            for e in ci:
                ax.plot([e, e], [y - 0.24, y - 0.04], color=col, lw=1.1, zorder=2)
        ax.scatter([v], [y - 0.14], s=22, marker=m, color=col, edgecolor=SURF, linewidth=0.5, zorder=3)
        ax.text(0.02, y + 0.2, lab, fontsize=6.8, va="center", ha="left", color=INK if ci else INK2)
        ax.text(0.98, y + 0.2, f"{v:.2f}", fontsize=6.8, va="center", ha="right", color=INK, fontweight="bold" if ci else "normal")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.tick_params(axis="x", labelsize=6.8, length=2, pad=1.5)
    ax.set_xlabel("AUROC: predicting repair", fontsize=6.8, labelpad=1.5)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    hb = HB * gs[0, 0].get_position(fig).height / ax.get_position().height
    header(ax, "predicted before any rollout", "two held-out families", h=hb, fs=(7.0, 6.6))
    pb = ax.get_position()
    fig.text(pb.x0 - 0.008, pb.y1 + hb * pb.height, "b", fontsize=9, fontweight="bold", ha="right", va="top")
    return fig


def fig_teaser():
    spec = [("teaser_pull", "pulled back 33 cm"), ("teaser_lookaway", "turned away 35\N{DEGREE SIGN}")]   # Fig. 2 shows the orbit
    rows = []
    for name, shift in spec:
        d = load(name)
        if d is None:
            continue
        e, cov = float(d["cal_err_cm"]), float(d["coverage"])
        rows.append({"train": rgb(d["perfect"]), "deployed": rgb(d["deployed"]),
                     "measured": contour(d["after_sweep"], d["sil_est"], SIL),
                     "corrected": rgb(d["corrected"]), "shift": shift,
                     "cal": f"error {e:.1f} cm" if e < 30 else f"lost ({e:.0f} cm off)",
                     "verdict": f"view kept {cov:.2f}" + (": fixable" if cov > 0.6 else ": lost"),
                     "recoverable": cov > 0.6})
    if rows:
        save(teaser_layout(rows), "teaser")


# ---------------------------------------------------------------- the method: real frames beside the schematic
def fig_method_pictures():
    """Framed like Fig. 1 of the group's Functional Invariance paper: real simulator frames on the left, a divider, and
    the pipeline as a schematic on the right. Every element is placed in inches on a fixed grid."""
    d = load("teaser_orbit")
    if d is None:
        return
    W, H = TEXTW, 2.3
    fig = plt.figure(figsize=(W, H))
    ov = fig.add_axes([0, 0, 1, 1], zorder=10)
    ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis("off"); ov.patch.set_alpha(0)
    ov.add_patch(patches.FancyBboxPatch((0.02, 0.02), W - 0.04, H - 0.04, boxstyle="round,pad=0,rounding_size=0.07",
                                        facecolor="none", edgecolor=INK, linewidth=1.0))
    # ---- left: four real frames, 2 x 2, labels above each
    T, g, lab = 0.86, 0.06, 0.15
    lx, ly = 0.14, 0.14
    frames = [(rgb(d["perfect"]), "training view"), (rgb(d["deployed"]), "deployed, orbit 20\N{DEGREE SIGN}"),
              (contour(d["after_sweep"], d["sil_est"], SIL), "measured silhouette"), (rgb(d["corrected"]), "given to the policy")]
    for k, (img, t) in enumerate(frames):
        c, r_ = k % 2, 1 - k // 2
        x, y = lx + c * (T + g), ly + r_ * (T + lab + 0.04)
        ax = fig.add_axes([x / W, y / H, T / W, T / H])
        ax.imshow(img, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#9ca3af"); sp.set_linewidth(0.5)
        ov.text(x + T / 2, y + T + 0.04, t, fontsize=7, ha="center", va="bottom")
    xdiv = lx + 2 * T + g + 0.13
    ov.plot([xdiv, xdiv], [0.16, H - 0.16], color="#9ca3af", lw=0.6)
    # ---- right: the schematic, two rows of boxes in a snake: Look -> Measure -> Predict / Undo -> Verify -> policy
    bw, bh, gap = 0.9, 0.72, 0.2
    x0 = xdiv + 0.2
    cx = [x0 + k * (bw + gap) for k in range(3)]
    yt, yb = 1.3, 0.36                                 # bottom edges of the top and bottom rows
    IW, IH = 0.34, 0.2                                 # icon slot, centred, same place in every box
    NAV, STL, SKY_ = "#1f3a5f", "#3b7ea1", "#8db4d3"

    def arm(x, y, s_, q, col, ls="-", lw=1.0):
        """Three-link arm from joint angles (deg), base (x, y), link length s_ (in)."""
        pts, a_ = [(x, y)], 0.0
        for L, dq in zip((1.0, 0.85, 0.55), q):
            a_ += np.radians(dq)
            pts.append((pts[-1][0] + s_ * L * np.cos(a_), pts[-1][1] + s_ * L * np.sin(a_)))
        xs_, ys_ = zip(*pts)
        ov.plot(xs_, ys_, color=col, lw=lw, ls=ls, solid_capstyle="round", zorder=12)
        ov.plot([x - 0.035, x + 0.035], [y, y], color=col, lw=lw * 1.3, zorder=12)

    def icon(kind, ix, iy):
        """ix, iy: lower-left corner of the icon slot."""
        mx, my = ix + IW / 2, iy + IH / 2
        if kind == "look":                            # camera; training view ray (dashed) and current ray (solid)
            ov.add_patch(patches.Rectangle((ix + 0.02, my - 0.04), 0.08, 0.08, facecolor=INK2, edgecolor="none", zorder=12))
            ov.add_patch(patches.Polygon([(ix + 0.1, my - 0.025), (ix + 0.14, my - 0.045), (ix + 0.14, my + 0.045),
                                          (ix + 0.1, my + 0.025)], closed=True, facecolor=INK2, edgecolor="none", zorder=12))
            ov.plot([ix + 0.15, ix + IW - 0.01], [my, my + 0.07], color=STL, lw=0.9, ls=(0, (2, 1.5)), zorder=12)
            ov.plot([ix + 0.15, ix + IW - 0.01], [my, my - 0.06], color=NAV, lw=0.9, zorder=12)
        elif kind == "measure":                       # the arm at three known poses
            for q, c in (((105, -60, -45), SKY_), ((80, -35, -60), STL), ((58, -15, -80), NAV)):
                arm(mx - 0.035, iy + 0.005, 0.085, q, c)
        elif kind == "predict":                       # a score bar, filled to the view kept
            ov.add_patch(patches.Rectangle((ix + 0.02, my - 0.035), IW - 0.04, 0.07, facecolor="#f1f3f6", edgecolor=INK2,
                                           linewidth=0.5, zorder=12))
            ov.add_patch(patches.Rectangle((ix + 0.02, my - 0.035), (IW - 0.04) * 0.77, 0.07, facecolor=NAV, edgecolor="none",
                                           zorder=13))
        elif kind == "undo":                          # the deployed frame warped into the training frame
            ov.add_patch(patches.Polygon([(ix + 0.02, iy + 0.03), (ix + 0.11, iy + 0.01), (ix + 0.12, iy + 0.17),
                                          (ix + 0.03, iy + 0.19)], closed=True, facecolor="#fafafa", edgecolor=MUTED2,
                                         linewidth=0.7, zorder=12))
            ov.add_patch(patches.Rectangle((ix + IW - 0.14, iy + 0.02), 0.13, 0.16, facecolor="#fafafa", edgecolor=NAV,
                                           linewidth=0.8, zorder=12))
            ov.annotate("", (ix + IW - 0.16, my), (ix + 0.14, my), arrowprops=dict(arrowstyle="-|>", color=NAV, lw=0.8,
                                                                                    mutation_scale=6, shrinkA=0, shrinkB=0),
                        zorder=13)
        elif kind == "verify":                        # measured silhouette (solid) on the kinematic one (dashed)
            arm(mx - 0.045, iy + 0.005, 0.085, (80, -35, -60), NAV)
            arm(mx - 0.025, iy + 0.005, 0.085, (80, -35, -60), STL, ls=(0, (2, 1.2)))
        elif kind == "policy":                        # a padlock: weights frozen
            ov.add_patch(patches.FancyBboxPatch((mx - 0.06, iy + 0.01), 0.12, 0.1, boxstyle="round,pad=0,rounding_size=0.015",
                                                facecolor=NAV, edgecolor="none", zorder=12))
            ov.add_patch(patches.Arc((mx, iy + 0.11), 0.08, 0.14, theta1=0, theta2=180, color=NAV, lw=1.2, zorder=12))

    def box(x, y, title, math, kind, ec=INK, fc="white", lw=0.8):
        ov.add_patch(patches.FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                            facecolor=fc, edgecolor=ec, linewidth=lw))
        icon(kind, x + (bw - IW) / 2, y + bh - IH - 0.06)
        ov.text(x + bw / 2, y + 0.3, title, fontsize=7.6, fontweight="bold", ha="center", va="center")
        ov.text(x + bw / 2, y + 0.12, math, fontsize=7.2, ha="center", va="center")

    def arr(p0, p1, col=INK2, ls="-", conn="arc3"):
        ov.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, linestyle=ls, mutation_scale=7,
                                                shrinkA=0, shrinkB=0, connectionstyle=conn))
    box(cx[0], yt, "Look", r"$r(q,\eta_0) > \tau\,?$", "look")
    box(cx[1], yt, "Measure", r"$\hat\eta = \mathrm{argmin}_{\eta}\, \Sigma_k\, d_k$", "measure")
    box(cx[2], yt, "Predict", r"$\mathrm{ident}\times\mathrm{invert}$", "predict")
    box(cx[2], yb, "Undo", r"$o' = T_{\hat\eta}(o)$", "undo")
    box(cx[1], yb, "Verify", r"$F_1 \geq 0.75$", "verify")
    box(cx[0], yb, r"Frozen $\pi$", r"$a = \pi(o')$", "policy", ec="#1f3a5f", fc="#eef2f7", lw=1.0)
    mt, mb = yt + bh / 2, yb + bh / 2
    arr((cx[0] + bw, mt), (cx[1], mt))
    arr((cx[1] + bw, mt), (cx[2], mt))
    arr((cx[2] + bw / 2, yt), (cx[2] + bw / 2, yb + bh))
    arr((cx[2], mb), (cx[1] + bw, mb))
    arr((cx[1], mb), (cx[0] + bw, mb))
    arr((cx[0] + bw * 0.3, yt), (cx[0] + bw * 0.3, yb + bh), col=MUTED2)
    ov.text(cx[0] + bw * 0.3 + 0.05, (yt + yb + bh) / 2, "no: raw frame", fontsize=6.8, color=INK2, ha="left", va="center",
            style="italic")
    ov.text(cx[0] + bw + 0.1, mt + 0.06, "yes", fontsize=6.8, color=INK2, ha="center", va="bottom", style="italic")
    arr((cx[1] + bw * 0.5, yb + bh), (cx[1] + bw * 0.5, yt), col=MUTED2, ls=(0, (3, 2)))
    ov.text(cx[1] + bw * 0.5 + 0.05, (yt + yb + bh) / 2, "no: 7 poses", fontsize=6.8, color=INK2, ha="left", va="center",
            style="italic")
    ov.text((cx[0] + cx[2] + bw) / 2, H - 0.16, "one instrument: the robot's encoders, kinematic model and silhouette",
            fontsize=7, color=INK2, ha="center", va="center")
    ov.text((cx[0] + cx[2] + bw) / 2, 0.19, r"$d_k$: silhouette distance at arm pose $q_k$;  $T_{\hat\eta}$: depth re-rendering",
            fontsize=6.8, color=INK2, ha="center", va="center")
    fig._boxes = [(x, y) for x, y in ((cx[0], yt), (cx[1], yt), (cx[2], yt), (cx[2], yb), (cx[1], yb), (cx[0], yb))]
    fig._bsize = (bw, bh)
    save(fig, "fig2_method")


# ---------------------------------------------------------------- the ladder, in pictures
def fig_ladder_pictures():
    d = load("teaser_lookaway")
    if d is None:
        return
    e = float(d["cal_err_cm"])
    picture_grid([[(rgb(d["deployed"]), "raw", "the deployed frame"),
                   (rgb(d["corrected"]), "measured pose", f"off by {e:.0f} cm", BAD_ON_BAR),
                   (rgb(d["truepose"]), "true pose", "no error, same holes"),
                   (rgb(d["perfect"]), "perfect render", "the training view", OK_ON_BAR)]], "ladder_pictures")


# ---------------------------------------------------------------- the motion-free check
def fig_static():
    a, b = load("static_nominal"), load("teaser_orbit")
    if a is None or b is None:
        return
    ra, rb = float(a["static_residual"]), float(b["static_residual"])
    picture_grid([[(contour(a["deployed"], a["static_mask"], SIL), "camera not moved", f"residual {ra:.2f}"),
                   (tint(a["deployed"], a["static_mask"], SIL, 0.4), "lands on the arm", "skip: arm stays still", OK_ON_BAR),
                   (contour(b["deployed"], b["static_mask"], SIL), "camera moved 20\N{DEGREE SIGN}", f"residual {rb:.2f}"),
                   (tint(b["deployed"], b["static_mask"], SIL, 0.4), "misses the arm", "measure with 3 poses", BAD_ON_BAR)]],
                 "static_check")


# ---------------------------------------------------------------- fidelity: where the error sits, in pictures
def fig_fidelity_pictures():
    spec = [("fid_high", "orbit 10\N{DEGREE SIGN}"), ("fid_mid", "orbit 30\N{DEGREE SIGN}"), ("fid_low", "turned away 55\N{DEGREE SIGN}")]
    spec = [(lab, load(n)) for n, lab in spec if load(n) is not None]
    if not spec:
        return
    errmap = matplotlib.colors.LinearSegmentedColormap.from_list("err", ["#ffffff", "#d9cfe8", "#9b82c0", PURPLE])
    rows = []
    for lab, d in spec:
        err = np.abs(rgb(d["truepose"]).astype(np.float32) - rgb(d["perfect"]).astype(np.float32)).mean(-1)
        holes_raw = ~(np.asarray(d["filled_true"]) > 0)          # buffer orientation, for tint (which turns it)
        holes = flipm(holes_raw)                                  # display orientation, matching err
        kept = 1 - holes.mean()
        if holes.any() and not holes.all():
            es = f"seen {err[~holes].mean():.0f}, holes {err[holes].mean():.0f}"
        else:
            es = f"holes {err.mean():.0f}" if holes.all() else f"seen {err.mean():.0f}"
        rows.append([(rgb(d["deployed"]), "deployed view", lab),
                     (tint(d["truepose"], holes_raw, YELLOW, 0.5), "re-rendered", f"view kept {kept:.2f}" if kept > 0 else "no view kept"),
                     (err, "error vs. perfect", es, None, dict(cmap=errmap, vmin=0, vmax=80)),
                     (rgb(d["perfect"]), "perfect render", "the training camera")])
    picture_grid(rows, "fidelity_pictures", width=TEXTW * 0.9)


# ---------------------------------------------------------------- tolerance: what a centimetre looks like
def fig_tolerance_pictures():
    d = load("tolerance")
    if d is None:
        return
    subs = {1: ("inside the tolerance", OK_ON_BAR), 4: ("outside: action changed", BAD_ON_BAR), 16: ("no better than raw", BAD_ON_BAR)}
    picture_grid([[(rgb(d["truepose"]), "true pose", "the training view")] +
                  [(rgb(d[f"dose_{k}"]), f"+{k} cm pose error", subs[k][0], subs[k][1]) for k in (1, 4, 16)]], "tolerance_pictures")


# ---------------------------------------------------------------- maps over camera placements
def fig_maps():
    env = [json.loads(l) for l in open(os.path.join(REPO, "vendor", "envelope.jsonl")) if l.strip()]
    fid = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(os.path.join(REPO, "vendor", "fidelity.jsonl")))}
    az = list(range(-60, 61, 10))
    sc = [0.8, 1.0, 1.3, 1.6]
    grids = {k: np.full((len(sc), len(az)), np.nan) for k in ("recover", "coverage", "err_seen", "sigma")}
    for i, s_ in enumerate(sc):
        for j, a in enumerate(az):
            rs = [r for r in env if r.get("az") == a and r.get("scale") == s_ and "el" not in r]
            if rs:
                grids["recover"][i, j] = np.mean([r["recover"] for r in rs])
                grids["coverage"][i, j] = np.mean([r["coverage"] for r in rs])
                grids["sigma"][i, j] = np.mean([r["sigma_min"] / r["sigma_min_ref"] for r in rs])
                fs = [fid[(r["task_id"], r["cond"])]["err_seen"] for r in rs if (r["task_id"], r["cond"]) in fid]
                if fs:
                    grids["err_seen"][i, j] = np.mean(fs)
    panels = [("sigma", "identifiability", "Blues", (0, 1.25)), ("coverage", "view kept (coverage)", "Blues", (0, 1)),
              ("recover", "recoverability score", "Blues", (0, 1)), ("err_seen", "error on seen pixels", "Oranges", (0, 40))]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 1.55))
    for ax, (k, title, cmap, (lo, hi)) in zip(axes, panels):
        im = ax.imshow(grids[k], cmap=cmap, vmin=lo, vmax=hi, aspect="auto", origin="lower", extent=(-65, 65, -0.5, len(sc) - 0.5))
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(["-60", "-30", "0", "+30", "+60"], fontsize=6.5)
        ax.set_yticks(range(len(sc)))
        ax.set_yticklabels([f"x{v}" for v in sc] if ax is axes[0] else [], fontsize=6.5)
        ax.set_xlabel("camera orbit (deg)", fontsize=7)
        ax.tick_params(length=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.scatter([0], [sc.index(1.0)], marker="+", s=45, color=RED, linewidths=1.1, zorder=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_visible(False)
    axes[0].set_ylabel("camera distance", fontsize=7)
    fig.text(0.5, -0.04, "red cross: the training camera. Means over three base tasks; orbit about the robot, distance relative to training.", ha="center", fontsize=6.5, color=INK2)
    fig.subplots_adjust(wspace=0.35)
    save(fig, "maps")


# ---------------------------------------------------------------- the method, as a schematic
def fig_schematic():
    fig, ax = plt.subplots(figsize=(TEXTW, 2.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2", linewidth=0.9, edgecolor=col, facecolor=SURF, zorder=2))
        ax.add_patch(patches.Rectangle((x, y + h - 10), w, 10, facecolor=col, edgecolor="none", zorder=3))
        ax.text(x + w / 2, y + h - 5, title, ha="center", va="center", fontsize=7.5, color=SURF, fontweight="bold", zorder=4)
        ax.text(x + 2, y + h - 14, body, ha="left", va="top", fontsize=5.8, color=INK, zorder=4, linespacing=1.35)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))

    ax.add_patch(patches.FancyBboxPatch((0.5, 4), 99, 93, boxstyle="round,pad=0,rounding_size=3", linewidth=0.6, edgecolor=GRID, facecolor="#f9f9f7", zorder=1))
    ax.text(2.5, 93, "one instrument, used four times: the robot's own body (encoders, kinematic model, silhouette)", fontsize=6.5, color=INK2, va="top")
    W_, H_, Y_ = 22.5, 66, 14
    xs = [2, 26.5, 51, 75.5]
    box(xs[0], Y_, W_, H_, "Measure", "move the arm through\n   3 known poses\ncamera pose: fit the\n   silhouette\njoint offset: encoders\nlight, noise: body pixels", BLUE)
    box(xs[1], Y_, W_, H_, "Predict", "recover = ident x invert\nident: smallest singular\n   value of the self-\n   observation Jacobian,\n   relative to training\ninvert: view kept after\n   the undo (coverage)", AQUA)
    box(xs[2], Y_, W_, H_, "Undo", "servo the arm to the\n   training pose\nre-render the frame into\n   the training camera\ninvert the light map\ndenoise", YELLOW)
    box(xs[3], Y_, W_, H_, "Verify", "warp the observed\n   silhouette into the\n   training view\ndoes it land where the\n   kinematics say it\n   must (F1)", ORANGE)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(a + W_, b, Y_ + H_ / 2)
    ax.text(50, 8, "the policy stays frozen: no weights change, no gradient, no policy query before the rollout", ha="center", fontsize=6.5, color=INK2)
    save(fig, "schematic")


def gallery():
    order = ["bw_pipeline", "bw_loop", "bw_score", "bw_ladder", "bw_lazy", "schematic", "teaser", "static_check", "ladder_pictures", "ladder", "score", "maps", "fidelity_pictures", "fidelity", "tolerance_pictures", "tolerance", "harm", "envelope", "droid_masks"]
    items = ""
    for n in order:
        for ext in ("png", "jpg"):
            if os.path.exists(os.path.join(FIGS, f"{n}.{ext}")):
                items += f'<figure><img src="{n}.{ext}"><figcaption><b>{n}</b></figcaption></figure>'
                break
    html = ("<!doctype html><meta charset='utf-8'><title>Paper figures</title><style>body{font:15px/1.5 Georgia,serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#0b0b0b;background:#f9f9f7}"
            "figure{margin:0 0 36px;background:#fff;padding:18px;border:1px solid #e1e0d9;border-radius:6px}img{width:100%;height:auto}figcaption{margin-top:8px;color:#52514e}</style>"
            "<h1 style='font-weight:normal'>Paper figures</h1>" + items)
    open(os.path.join(FIGS, "gallery.html"), "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    for fn in (fig_schematic, fig_maps, fig_teaser, fig_method_pictures, fig_ladder_pictures, fig_static, fig_fidelity_pictures,
               fig_tolerance_pictures):
        try:
            fn()
        except Exception as e:
            import traceback
            print("FAILED", fn.__name__, repr(e)); traceback.print_exc()
    gallery()
