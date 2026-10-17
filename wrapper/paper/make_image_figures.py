"""Image figures from the local policy-free renders (vendor/render_figures.py -> figs/renders/*.npz). No simulator here.

    uv run --no-project --with numpy --with matplotlib --with opencv-python-headless python wrapper/paper/make_image_figures.py [names]

Figures (each drawn at its included width, so every label is at its true point size; see WIDTH):
  teaser        Fig. 1: four deployed cameras x (training view | deployed view | measured silhouette (the frame after the
                probe sweep, measured-pose silhouette outlined, palette.SIL_MEASURED) | frame at the measured pose | frame
                at the true pose); holes of each re-rendered frame tinted (palette.HOLES). No under-tile text: every number
                is in the LaTeX caption and in the design bundle's annotations.json. MERGED 2026-09-24 (F49): the schematic
                (Fig. 2) is folded in as the measured-silhouette column; its loop moves into the Fig. 1 caption.
  static_check  the motion-free test: training-camera kinematic silhouette over three deployed frames (unmoved, orbit 10,
                turned away 55), residual in the caption. Its own float (F19).
  fidelity      (a) error vs lost view over the 264 placements (results/local/fidelity.jsonl), ink/grey curves with dash
                styles and direct labels; (b) a 3 x 3 grid, three cameras x (true-pose frame, holes tinted | |error| |
                perfect render), equal tiles, one Greys colour bar (F53).
  tolerance_pictures  the true-pose frame and |difference| to it at +1, +4, +16 cm, one 0 to 80 Greys colour bar (F19).
Dropped 2026-09-24 (files left in place, no longer rebuilt): schematic (merged into teaser, F49), ladder_pictures (its
tiles are Fig. 1 columns 4-5), maps (replaced by one sentence), fidelity_pictures (merged into fidelity), gallery.html.
Design bundle: only with --clean (F21), the run also exports the clean source tiles and layers to BUNDLE, with
annotations.json (per tile: robot-silhouette bbox, hole centroid and a leader anchor, from the npz masks; plus droid_masks).
"""
import json
import os
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import palette as P  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(HERE, "iclr", "figs")
REN = os.path.join(FIGS, "renders")
BUNDLE = os.path.join(os.path.expanduser("~"), "Downloads", "claude_design_bundle", "paper_figures")
P.apply()
INK, INK2, MUTED = P.INK, P.INK2, P.MUTED
TEXTW = P.TEXTW
WIDTH = {"teaser": 0.86, "schematic": 1.0, "static_check": 0.75, "fidelity": 1.0, "tolerance_pictures": 0.8}
CLEAN = "--clean" in sys.argv       # F21: export the design-bundle tiles only when asked (default off)
ANN = {}          # figure -> annotations for Claude Design (numbers read from the npz / rows here)
UP = 3            # bundle tiles are upscaled 3x (256 -> 768 px, bicubic for frames, nearest for masks); native renders are 256 px


def load(name):
    p = os.path.join(REN, name + ".npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- display-orientation helpers
def disp(img):
    """Raw simulator buffers are upside down; every frame is turned 180 degrees for display."""
    a = np.asarray(img)
    a = a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, -1)
    return np.ascontiguousarray(a[::-1, ::-1]).astype(np.uint8)


def dmask(mask, keep_frac=None):
    m = np.ascontiguousarray(np.asarray(mask)[::-1, ::-1] > 0)
    return clean(m, keep_frac) if keep_frac else m


def clean(m, frac=0.03):
    """Keep the connected components of a robot mask that are at least `frac` of the largest (drops stray mount geometry)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    return np.isin(lab, [i + 1 for i, a in enumerate(areas) if a >= frac * areas.max()])


def outline(img, m, color, width=2):
    out = img.copy()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, P.hex_rgb(color), width, cv2.LINE_AA)
    return out


def tint(img, m, color=P.HOLES, alpha=P.HOLES_ALPHA):
    out = img.astype(np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(P.hex_rgb(color), np.float32)
    return out.astype(np.uint8)


def holes_of(d, key):
    return ~dmask(d[key])


def show(ax, img, title=None, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos" if img.ndim == 3 else "nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():          # error maps (white = no error) get a hairline frame so an all-zero tile still reads
        sp.set_visible(cmap is not None)
        sp.set_color(P.AXIS)
        sp.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=7.5, pad=2.5)
    return im


def rowlabel(ax, s):
    ax.text(-0.06, 0.5, s, transform=ax.transAxes, ha="right", va="center", fontsize=7.5, linespacing=1.1)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(os.path.join(FIGS, name + ".png"), bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- design-bundle export
def export(fig, name, img, mask=False):
    if not CLEAN:                    # F21: the bundle is written only with --clean
        return
    d = os.path.join(BUNDLE, fig)
    os.makedirs(d, exist_ok=True)
    if mask:                         # transparent layer: colour where the mask is set
        a = np.zeros(img.shape[:2] + (4,), np.uint8)
        col = P.hex_rgb(mask) if isinstance(mask, str) else (255, 255, 255)
        a[img > 0] = (*col, 255)
        a = cv2.resize(a, (a.shape[1] * UP, a.shape[0] * UP), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(d, name + ".png"), cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA))
    else:
        a = cv2.resize(img, (img.shape[1] * UP, img.shape[0] * UP), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(d, name + ".png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def geom(robot=None, holes=None):
    """F21: annotation anchors for one tile, in BUNDLE pixels (display orientation, origin top-left, x right, y down; the
    256 px render x UP). robot_bbox = [x0, y0, x1, y1] of the robot-silhouette mask; leader_anchor = the silhouette's
    topmost pixel (median x among the top row), where a callout line should attach; hole_centroid = mean (x, y) of the
    hole pixels and hole_share = their share of the tile. A missing or empty mask gives None."""
    out = {}
    if robot is not None and np.asarray(robot).any():
        ys, xs = np.nonzero(robot)
        top = ys.min()
        out.update(robot_bbox=[int(xs.min()) * UP, int(top) * UP, int(xs.max() + 1) * UP, int(ys.max() + 1) * UP],
                   leader_anchor=[round(float(np.median(xs[ys == top])) * UP + UP / 2, 1), round(float(top) * UP + UP / 2, 1)])
    else:
        out.update(robot_bbox=None, leader_anchor=None)
    if holes is not None and np.asarray(holes).any():
        ys, xs = np.nonzero(holes)
        out.update(hole_centroid=[round(float(xs.mean()) * UP + UP / 2, 1), round(float(ys.mean()) * UP + UP / 2, 1)],
                   hole_share=round(float(np.mean(holes)), 3))
    else:
        out.update(hole_centroid=None, hole_share=0.0 if holes is not None else None)
    return out


def outline_layer(m, width=2):
    z = np.zeros(m.shape, np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(z, cs, -1, 1, width)
    return z


# ---------------------------------------------------------------- Fig. 1 teaser
TEASER = [("teaser_orbit", "orbit 20°"), ("teaser_pull", "pulled back\n30%"), ("teaser_az45", "orbit 45°"), ("teaser_lookaway", "turned away\n35°")]


def fig_teaser():
    rows = [(n, lab, load(n)) for n, lab in TEASER if load(n) is not None]
    cols = ["training view", "deployed view", "measured\nsilhouette", "at the\nmeasured pose", "at the\ntrue pose"]
    W = TEXTW * WIDTH["teaser"]
    left, top, gap = 0.62, 0.3, 0.05                    # inches
    tw = (W - left - 4 * gap) / 5
    H = top + 4 * tw + 3 * gap + 0.07 + 0.02          # extra gap between the recoverable and the failing pair
    fig = plt.figure(figsize=(W, H))
    info = []
    for i, (name, lab, d) in enumerate(rows):
        hm, ht = holes_of(d, "filled"), holes_of(d, "filled_true")
        sil = dmask(d["sil_est"], 0.03)
        imgs = [disp(d["perfect"]), disp(d["deployed"]), outline(disp(d["after_sweep"]), sil, P.SIL_MEASURED, 2),
                tint(disp(d["corrected"]), hm), tint(disp(d["truepose"]), ht)]
        y0 = H - top - (i + 1) * tw - i * gap - (0.07 if i >= 2 else 0)
        for j, img in enumerate(imgs):
            ax = fig.add_axes([(left + j * (tw + gap)) / W, y0 / H, tw / W, tw / H])
            show(ax, img, cols[j] if i == 0 else None)
            if j == 0:
                rowlabel(ax, lab)
        e, c, ct = float(d["cal_err_cm"]), float(d["coverage"]), float(1 - ht.mean())
        info.append(dict(row=i + 1, camera=lab.replace("\n", " "), npz=name, cal_err_cm=round(e, 2), cal_err_deg=round(float(d["cal_err_deg"]), 2),
                         coverage_measured=round(c, 3), coverage_true_pose=round(ct, 3), static_residual_px=round(float(d["static_residual"]), 2),
                         camera_shift_cm_deg=[round(float(x), 1) for x in d["shift"]]))
        kin = dmask(d["static_mask"], 0.03)             # the robot as the TRAINING camera sees it (training-view tiles)
        info[-1]["tiles"] = {"1_training_view": geom(kin), "2_deployed_view": geom(None),
                             "3_measured_silhouette_after_sweep": geom(sil),
                             "4_frame_at_measured_pose": geom(kin, hm), "5_frame_at_true_pose": geom(kin, ht)}
        for key, img in zip(("1_training_view", "2_deployed_view", "3_after_probe_sweep", "4_frame_at_measured_pose", "5_frame_at_true_pose"),
                            (disp(d["perfect"]), disp(d["deployed"]), disp(d["after_sweep"]), disp(d["corrected"]), disp(d["truepose"]))):
            export("fig1_teaser", f"row{i + 1}_{name}_{key}", img)
        export("fig1_teaser", f"row{i + 1}_{name}_layer_holes_measured_pose", hm, P.HOLES)
        export("fig1_teaser", f"row{i + 1}_{name}_layer_holes_true_pose", ht, P.HOLES)
        export("fig1_teaser", f"row{i + 1}_{name}_layer_measured_silhouette_on_after_sweep", outline_layer(dmask(d["sil_est"], 0.03)), P.SIL_MEASURED)
    save(fig, "teaser")
    ANN["fig1_teaser"] = dict(figure="Fig. 1 (teaser.pdf)", columns=[c.replace("\n", " ") for c in cols],
                              holes_overlay=f"{P.HOLES} at {P.HOLES_ALPHA:.0%} alpha", measured_silhouette_outline=P.SIL_MEASURED,
                              anchor_note="tiles: bundle px (768 px tiles). Training-view tiles use the kinematic silhouette at the "
                                          "training camera (static_mask); the deployed view has no robot mask in the npz (None).",
                              rows=info)
    print("teaser rows:", [(r["npz"], r["cal_err_cm"], r["coverage_measured"], r["coverage_true_pose"]) for r in info])


# ---------------------------------------------------------------- Fig. 2 the method
def fig_schematic():
    """Measure, predict, undo, check on one deployed camera (orbit 30 deg, fid_mid.npz), four tiles at full width."""
    d = load("fid_mid")
    if d is None:
        return
    hm = holes_of(d, "filled")
    sil = dmask(d["sil_est"], 0.03)
    chk = dmask(d["static_mask"], 0.03)
    tiles = [("1  measure", outline(disp(d["after_sweep"]), sil, P.SIL_MEASURED, 2)),
             ("2  predict", tint(disp(d["corrected"]), hm, P.HOLES, 0.55)),
             ("3  undo", disp(d["corrected"])),
             ("4  check", outline(disp(d["corrected"]), chk, P.SIL_CHECK, 2))]
    W = TEXTW * WIDTH["schematic"]
    gap = 0.2
    tw = min((W - 3 * gap) / 4, 1.42 - 0.17)
    H = tw + 0.17
    x0 = (W - 4 * tw - 3 * gap) / 2
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k, (t, img) in enumerate(tiles):
        ax = fig.add_axes([(x0 + k * (tw + gap)) / W, 0.005, tw / W, tw / H])
        show(ax, img, t)
        ax.title.set_fontsize(8)
        axes.append(ax)
    for a, b in zip(axes[:-1], axes[1:]):
        p0, p1 = a.get_position(), b.get_position()
        y = (p0.y0 + p0.y1) / 2
        fig.add_artist(patches.FancyArrowPatch((p0.x1 + 0.004, y), (p1.x0 - 0.004, y), transform=fig.transFigure, arrowstyle="-|>",
                                               mutation_scale=8, color=INK2, lw=0.9))
    save(fig, "schematic")
    c = float(1 - hm.mean())
    for key, img in (("0_deployed_view", disp(d["deployed"])), ("1_after_probe_sweep", disp(d["after_sweep"])), ("3_undone_frame", disp(d["corrected"])),
                     ("reference_training_view", disp(d["perfect"]))):
        export("fig2_method", key, img)
    export("fig2_method", "layer_1_measured_silhouette_outline", outline_layer(sil), P.SIL_MEASURED)
    export("fig2_method", "layer_1_measured_silhouette_fill", sil, P.SIL_MEASURED)
    export("fig2_method", "layer_2_holes", hm, P.HOLES)
    export("fig2_method", "layer_4_check_silhouette_outline", outline_layer(chk), P.SIL_CHECK)
    ANN["fig2_method"] = dict(figure="Fig. 2 (schematic.pdf)", camera="orbit 30 deg (fid_mid.npz), not a Fig. 1 camera",
                              steps={"1 measure": f"3 probe poses, calibration error {float(d['cal_err_cm']):.1f} cm / {float(d['cal_err_deg']):.2f} deg",
                                     "2 predict": f"coverage c = {c:.2f} (holes = training-view pixels the camera cannot see); recover = ident x invert",
                                     "3 undo": "re-rendered into the training camera",
                                     "4 check": "kinematic silhouette at the training camera lands on the arm (F1)"},
                              static_residual_px=round(float(d["static_residual"]), 2))
    print(f"schematic: fid_mid error {float(d['cal_err_cm']):.3f} cm, coverage key {float(d['coverage']):.3f}, 1 - holes {c:.3f}")


# ---------------------------------------------------------------- the motion-free check
def fig_static():
    rows = [("static_nominal", "camera unmoved"), ("fid_high", "orbit 10°"), ("fid_low", "turned away 55°")]
    rows = [(n, t, load(n)) for n, t in rows if load(n) is not None]
    W = TEXTW * WIDTH["static_check"]
    gap = 0.06
    tw = (W - 2 * gap) / 3
    fig = plt.figure(figsize=(W, tw + 0.16))
    res = []
    for k, (n, t, d) in enumerate(rows):
        m = dmask(d["static_mask"], 0.03)
        ax = fig.add_axes([k * (tw + gap) / W, 0.0, tw / W, tw / (tw + 0.16)])
        show(ax, outline(disp(d["deployed"]), m, P.SIL_CHECK, 2), t)
        res.append((t, round(float(d["static_residual"]), 2)))
        export("static_check", f"{k + 1}_{n}_deployed_view", disp(d["deployed"]))
        export("static_check", f"{k + 1}_{n}_layer_predicted_silhouette_outline", outline_layer(m), P.SIL_CHECK)
    save(fig, "static_check")
    ANN["static_check"] = dict(figure="static_check.pdf (appendix)", overlay=f"training-camera kinematic silhouette, {P.SIL_CHECK} outline",
                               residual_px=dict(res), threshold_px=1.5)
    print("static residuals:", res)


# ---------------------------------------------------------------- fidelity: plot + pictures (MERGED)
def fid_plot(ax):
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "fidelity.jsonl")) if l.strip()]
    f = np.array([r["filled"] for r in rows])
    edges = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001])
    # F48: no hue here (hues are reserved for the policy families): ink and the grey ramp, told apart by dash and marker
    series = [("err_seen", "pixels the\ncamera saw", INK, "-", "o"), ("err_hole_inpaint", "holes,\ninpainted", P.GREY3[2], (0, (1.2, 1.2)), "^"),
              ("err_hole_ref", "holes, training-\nview fill", P.GREY3[1], (0, (4, 1.5)), "s"), ("err_raw", "no\ncorrection", MUTED, (0, (6, 2, 1, 2)), "D")]
    nbin, out = [], {}
    for key, label, col, ls, mk in series:
        v = np.array([r[key] for r in rows], float)
        xs, med, lo, hi, ns = [], [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (f >= a) & (f < b) & np.isfinite(v)
            if m.sum() >= 5:
                q = np.quantile(v[m], [0.25, 0.5, 0.75])
                xs.append(1 - (a + b) / 2); lo.append(q[0]); med.append(q[1]); hi.append(q[2]); ns.append(int(m.sum()))
        if key == "err_seen":
            nbin = list(zip(xs, ns))
        out[key] = (xs, med)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(xs, med, color=col, linewidth=1.3, linestyle=ls, marker=mk, markersize=3, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        j = int(np.argmax(xs))
        ax.text(xs[j] + 0.02, med[j], label, fontsize=7, color=INK if col == INK else INK2, va="center", ha="left", linespacing=1.0)
    for x, n in nbin:
        ax.text(x, 1.01, f"{n}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, ha="center", va="bottom")
    ax.text(-0.02, 1.01, "n =", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlim(0.0, 0.98)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("share of the training view lost (1 - c)")
    ax.set_ylabel("mean |pixel error| vs perfect render (0-255)")
    ax.set_ylim(0, 90)
    ax.grid(True, axis="y", color=P.GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    print("fidelity bins (1 - coverage, n):", [(round(x, 2), n) for x, n in nbin], "placements", len(rows), "left out (>70% lost):", int((f < 0.3).sum()),
          "median hole err (training fill) per bin:", [round(v, 1) for v in out["err_hole_ref"][1]])
    return len(rows), int((f < 0.3).sum())


def fig_fidelity():
    """(a) the plot; (b) F53: a 3 x 3 grid, every row the same three equal tiles: true-pose frame with holes tinted |
    |error| of that frame against the perfect training-view render (Greys, 0 to 80 of 0 to 255, clipped) | the perfect
    render of the same row. No deployed column."""
    rows = [("fid_high", "orbit 10°"), ("fid_mid", "orbit 30°"), ("fid_low", "turned away\n55°")]
    rows = [(n, lab, load(n)) for n, lab in rows if load(n) is not None]
    W = TEXTW * WIDTH["fidelity"]
    tw, gap = 0.76, 0.05
    top, bot = 0.3, 0.42
    H = 3 * tw + 2 * gap + top + bot
    fig = plt.figure(figsize=(W, H))
    axp = fig.add_axes([0.62 / W, bot / H, 1.55 / W, (3 * tw + 2 * gap - 0.02) / H])
    n_pl, n_out = fid_plot(axp)
    axp.text(-0.3, 1.1, "a", transform=axp.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    gx = W - 3 * tw - 2 * gap                          # right-aligned picture grid (inches)
    cols = ["true-pose frame\n(holes tinted)", "|error| vs the\nperfect render", "perfect render\n(training camera)"]
    info, im = [], None
    for i, (name, lab, d) in enumerate(rows):
        ht = holes_of(d, "filled_true")
        tp = disp(d["truepose"]).astype(np.float32)
        err = np.abs(tp - disp(d["perfect"]).astype(np.float32)).mean(-1)
        y0 = H - top - (i + 1) * tw - i * gap
        for j, img in enumerate((tint(disp(d["truepose"]), ht), err, disp(d["perfect"]))):
            ax = fig.add_axes([(gx + j * (tw + gap)) / W, y0 / H, tw / W, tw / H])
            if j == 1:
                im = show(ax, img, cols[j] if i == 0 else None, cmap=P.ERR, vmin=0, vmax=80)
            else:
                show(ax, img, cols[j] if i == 0 else None)
            if j == 0:
                rowlabel(ax, lab)
            if i == 0 and j == 0:
                ax.text(-0.75, 1.02, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
        seen = ~ht
        kin = dmask(d["static_mask"], 0.03)
        info.append(dict(row=i + 1, npz=name, camera=lab.replace("\n", " "), coverage_true_pose=round(float(seen.mean()), 3),
                         coverage_self_measured=round(float(d["coverage"]), 3), cal_err_cm=round(float(d["cal_err_cm"]), 1),
                         err_seen=round(float(err[seen].mean()), 1) if seen.any() else None, err_holes=round(float(err[ht].mean()), 1) if ht.any() else None,
                         tiles={"1_true_pose_frame": geom(kin, ht), "2_abs_error": geom(None, ht), "3_perfect_render": geom(kin)}))
        export("fidelity", f"row{i + 1}_{name}_1_true_pose_frame", disp(d["truepose"]))
        export("fidelity", f"row{i + 1}_{name}_layer_holes_true_pose", ht, P.HOLES)
        e8 = (np.clip(err / 80.0, 0, 1) * 255).astype(np.uint8)
        export("fidelity", f"row{i + 1}_{name}_2_abs_error_greys", cv2.cvtColor(255 - e8, cv2.COLOR_GRAY2RGB))
        export("fidelity", f"row{i + 1}_{name}_3_perfect_render", disp(d["perfect"]))
    cax = fig.add_axes([(gx + (tw + gap)) / W, 0.2 / H, tw / W, 0.05 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 40, 80], extend="max")
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    cb.set_label("|error| (0 to 255)", fontsize=7, labelpad=1)
    save(fig, "fidelity")
    ANN["fidelity"] = dict(figure="fidelity.pdf (plot + 3 x 3 pictures)", placements=n_pl, left_out_over_70pct_lost=n_out, rows=info,
                           colour_bar="Greys, 0 to 80 of 0 to 255, clipped above 80",
                           note="errors are at the true pose, as in panel (a); the self-measured coverage of row 3 is at a pose 108.9 cm off")
    print("fidelity pictures:", [{k: v for k, v in r.items() if k != "tiles"} for r in info])


# ---------------------------------------------------------------- tolerance: what a centimetre looks like
def fig_tolerance_pictures():
    d = load("tolerance")
    if d is None:
        return
    # F19: one row, the true-pose frame and |difference| to it at +1, +4, +16 cm (the dosed frames themselves look nearly
    # identical to the true-pose frame, so they are not drawn), one 0 to 80 Greys colour bar; no blank tile.
    W = TEXTW * WIDTH["tolerance_pictures"]
    gap, cb_h = 0.05, 0.3
    tw = (W - 3 * gap) / 4
    H = tw + 0.2 + cb_h
    fig = plt.figure(figsize=(W, H))
    base = disp(d["truepose"]).astype(np.float32)
    ax = fig.add_axes([0.0, cb_h / H, tw / W, tw / H])
    show(ax, disp(d["truepose"]), "true-pose frame")
    export("tolerance_pictures", "1_true_pose", disp(d["truepose"]))
    diffs, im = {"true pose": 0.0}, None
    for j, k in enumerate((1, 4, 16)):
        img = disp(d[f"dose_{k}"])
        e = np.abs(img.astype(np.float32) - base).mean(-1)
        ax2 = fig.add_axes([((j + 1) * (tw + gap)) / W, cb_h / H, tw / W, tw / H])
        im = show(ax2, e, f"|difference|, +{k} cm", cmap=P.ERR, vmin=0, vmax=80)
        diffs[f"+ {k} cm"] = round(float(e.mean()), 1)
        export("tolerance_pictures", f"{j + 2}_plus_{k}_cm", img)
    cax = fig.add_axes([(tw + gap) / W, 0.2 / H, (3 * tw + 2 * gap) / W, 0.05 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60, 80], extend="max")
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    cb.set_label("|difference| to the true-pose frame (0 to 255, shown 0 to 80)", fontsize=7, labelpad=1)
    save(fig, "tolerance_pictures")
    ANN["tolerance_pictures"] = dict(figure="tolerance_pictures.pdf (appendix)", mean_abs_difference_to_true_pose_0_255=diffs,
                                     camera_shift_cm_deg=[round(float(x), 1) for x in d["shift"]])
    print("tolerance picture mean |diff|:", diffs)


# ---------------------------------------------------------------- new 3D renders (run file 20, render_paper_views.py)
# These npz are 512 px and ALREADY in display orientation (key `orientation` = "display"): never disp() them. Each loader
# only exports clean tiles, transparent robot layers and pixel anchors for Claude Design; it returns quietly while the npz
# is missing, so nothing in the paper depends on it until the renders exist. Colours follow the bundle PROMPT.md.
TRAIN_C, DEPLOY_C = "#14B4C8", "#E67814"      # aqua = training setup, orange = what is there at deployment
UP3D = 2                                      # 512 -> 1024 px, the size of the bundle's other 3D tiles


def _load3d(name):
    d = load(name)
    if d is None:
        print(f"{name}.npz not rendered yet (run file 20): skipped")
        return None
    assert str(d.get("orientation", "")) == "display", f"{name}: expected display-orientation frames"
    return d


def _export3d(fig, name, img, color=None):
    dd = os.path.join(BUNDLE, fig)
    os.makedirs(dd, exist_ok=True)
    a = np.asarray(img)
    if color is not None:                         # mask -> transparent layer in that colour
        lay = np.zeros(a.shape[:2] + (4,), np.uint8)
        lay[a > 0] = (*P.hex_rgb(color), 255)
        lay = cv2.resize(lay, (a.shape[1] * UP3D, a.shape[0] * UP3D), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dd, name + ".png"), cv2.cvtColor(lay, cv2.COLOR_RGBA2BGRA))
    else:
        a = cv2.resize(a[..., :3].astype(np.uint8), (a.shape[1] * UP3D, a.shape[0] * UP3D), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dd, name + ".png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))


def _px(uv):
    """Anchor(s) in bundle pixels (1024 px tiles), rounded to 0.1 px."""
    return np.round(np.asarray(uv, float) * UP3D, 1).tolist()


def fig_scene_frusta():
    d = _load3d("scene_frusta")
    if d is None:
        return
    f = "scene_frusta"
    _export3d(f, "01_overview_walls_hidden", d["overview"])
    _export3d(f, "02_layer_robot_in_overview", d["overview_robot"], DEPLOY_C)
    _export3d(f, "03_training_view", d["training_view"])
    cams = [str(c) for c in d["cameras"]]
    for i, c in enumerate(cams[1:]):
        _export3d(f, f"{i + 4:02d}_deployed_view_{c}", d[f"view_{c}"])
    ANN[f] = dict(figure="appendix: where the cameras are (god's-eye view, task 1167)", tile_px=int(d["res"]) * UP3D,
                  anchor_check=round(float(d["anchor_check"]), 3),
                  cameras={c: dict(role="training" if c == "training" else "deployed (Fig. 1)",
                                   centre=_px(d[f"cam_{c}_px"][0]), look_at=_px(d[f"cam_{c}_px"][1]),
                                   frustum_corners_tl_tr_br_bl=_px(d[f"cam_{c}_px"][2:6]),
                                   shift_vs_training_cm_deg=(None if c == "training" else
                                                             [round(float(x), 1) for x in d[f"shift_{c}"][:2]]))
                           for c in cams},
                  colours=dict(training=TRAIN_C, deployed=DEPLOY_C), note="numbers are from the npz; frusta drawn at 0.30 m depth")
    print("scene_frusta: anchor check", ANN[f]["anchor_check"])


def fig_joint_offset_ghost():
    new = os.path.exists(os.path.join(REN, "joint_offset_ghost.npz"))
    d = _load3d("joint_offset_ghost" if new else "joint_offset")     # render runs before 2026-09-24 wrote joint_offset.npz
    if d is None:
        return
    f = "joint_offset_ghost"
    for view in ("training", "overview"):
        base = d[f"offset_{view}_view"] if view == "training" else d["offset_overview"]
        _export3d(f, f"{view}_01_arm_at_offset_start", base)
        _export3d(f, f"{view}_02_arm_at_training_start", d[f"nominal_{view}_view"] if view == "training" else d["nominal_overview"])
        _export3d(f, f"{view}_03_layer_offset_arm", d[f"offset_{view}_robot"], DEPLOY_C)
        _export3d(f, f"{view}_04_layer_training_start_arm", d[f"nominal_{view}_robot"], TRAIN_C)
    dq = np.asarray(d["joint_dev_rad"], float)
    ANN[f] = dict(figure="appendix, next to the Table 2 discussion: offset start vs training start (sweep condition "
                         + str(d["condition"]) + ", task 1167)", tile_px=int(d["res"]) * UP3D,
                  anchor_check=round(float(d["anchor_check"]), 3), max_joint_offset_rad=round(float(dq.max()), 3),
                  joint_offset_rad=[round(float(x), 3) for x in dq],
                  eef_px={v: dict(offset=_px(d[f"offset_eef_px_{v}"]), training_start=_px(d[f"nominal_eef_px_{v}"])) for v in ("training", "overview")},
                  eef_displacement_cm=round(float(np.linalg.norm(d["offset_eef_world"] - d["nominal_eef_world"]) * 100), 1),
                  colours=dict(training_start=TRAIN_C, offset_start=DEPLOY_C))
    print("joint_offset_ghost: max |dq|", ANN[f]["max_joint_offset_rad"], "rad, eef moved", ANN[f]["eef_displacement_cm"], "cm")


def fig_probe_ghosts(smc_rows=os.path.join(REPO, "results", "drive_vihan", "sweep", "smc_openvla_oft.jsonl"), units=("nominal", "orbit_az20")):
    d = _load3d("probe_ghosts")
    if d is None:
        return
    sys.path.insert(0, os.path.join(REPO, "wrapper", "scripts"))
    from render_paper_views import project        # the same projection the renders' anchors use
    f = "probe_ghosts"
    for k in range(len(d["P"])):
        _export3d(f, f"training_{k + 1}_probe", d["training_views"][k])
        _export3d(f, f"training_{k + 1}_layer_robot", d["training_robot"][k], TRAIN_C)
        _export3d(f, f"overview_{k + 1}_probe", d["overviews"][k])
        _export3d(f, f"overview_{k + 1}_layer_robot", d["overview_robot"][k], TRAIN_C)
    logged = {}
    if os.path.exists(smc_rows):                  # per-probe action vectors, numbers from the logged fingerprint rows
        for r in map(json.loads, open(smc_rows)):
            if r["task_id"] == int(d["task_id"]) and r["cond"] in units:
                P_log = np.asarray(r["P"], float)
                arrows = {}
                for key in ("A_raw", "A_cor"):
                    A = np.asarray(r[key], float)
                    tip = P_log + 0.06 * A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)   # 6 cm arrows
                    res = int(d["res"])
                    base_px = project(P_log, d["camera_C"], d["camera_R"], float(d["fovy"]), res)[0]
                    tip_px = project(tip, d["camera_C"], d["camera_R"], float(d["fovy"]), res)[0]
                    arrows[key] = dict(vectors=np.round(A, 4).tolist(), base_px_training=_px(base_px), tip_px_training=_px(tip_px))
                    # the arrows as their OWN transparent layer (bundle resolution, no text), never drawn on a tile
                    lay = np.zeros((res * UP3D, res * UP3D, 4), np.uint8)
                    col = (*P.hex_rgb(DEPLOY_C if key == "A_raw" else TRAIN_C), 255)
                    for b, t in zip(np.asarray(base_px, float) * UP3D, np.asarray(tip_px, float) * UP3D):
                        cv2.arrowedLine(lay, tuple(int(round(x)) for x in b), tuple(int(round(x)) for x in t), col, 4,
                                        cv2.LINE_AA, tipLength=0.25)
                    os.makedirs(os.path.join(BUNDLE, f), exist_ok=True)
                    cv2.imwrite(os.path.join(BUNDLE, f, f"training_layer_arrows_{r['cond']}_{key}.png"), cv2.cvtColor(lay, cv2.COLOR_RGBA2BGRA))
                logged[r["cond"]] = dict(pick=r.get("pick"), max_probe_position_gap_cm=round(float(np.abs(P_log - d["P"]).max() * 100), 2),
                                         **arrows)
    ANN[f] = dict(figure="appendix Sec. 4.5: the 7 fingerprint probe poses from the training camera (task "
                         + str(int(d["task_id"])) + ")", tile_px=int(d["res"]) * UP3D, anchor_check=round(float(d["anchor_check"]), 3),
                  probe_offsets_m=np.round(d["probe_offsets"], 3).tolist(),
                  eef_px_training=_px(d["eef_px_training"]), eef_px_overview=_px(d["eef_px_overview"]),
                  logged_actions=logged or "no logged rows found", arrow_note="arrows: unit direction of the logged action, drawn 6 cm long; A_raw on the raw frame, A_cor on the corrected frame",
                  colours=dict(robot=TRAIN_C))
    print("probe_ghosts:", len(d["P"]), "probes; logged units", sorted(logged))


# ---------------------------------------------------------------- the design bundle's text files
def write_bundle_docs():
    os.makedirs(BUNDLE, exist_ok=True)
    old = {}
    p = os.path.join(BUNDLE, "annotations.json")
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
    old.update(ANN)
    json.dump(old, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for n in ("teaser", "schematic", "static_check", "fidelity", "tolerance_pictures", "envelope", "harm", "tolerance", "ladder", "envelope_tilt", "droid_masks"):
        for ext in ("png",):
            s = os.path.join(FIGS, f"{n}.{ext}")
            if os.path.exists(s):
                os.makedirs(os.path.join(BUNDLE, "_current_figures"), exist_ok=True)
                shutil.copy(s, os.path.join(BUNDLE, "_current_figures", f"{n}.{ext}"))
    print("bundle:", BUNDLE)


if __name__ == "__main__":
    pick = {a for a in sys.argv[1:] if not a.startswith("--")}      # flags (--clean) are not figure names
    for fn in (fig_teaser, fig_schematic, fig_static, fig_fidelity, fig_tolerance_pictures,
               fig_scene_frusta, fig_joint_offset_ghost, fig_probe_ghosts):
        if fn is fig_schematic and "schematic" not in pick:        # DROPPED (merged into the teaser, F49): only on request
            continue
        if pick and fn.__name__[4:] not in pick:
            continue
        try:
            fn()
        except Exception as e:
            import traceback
            print("FAILED", fn.__name__, repr(e)); traceback.print_exc()
    write_bundle_docs()
