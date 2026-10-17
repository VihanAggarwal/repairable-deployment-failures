"""The paper's result figures, regenerated from the logged rows (results/ snapshot + vendor/ local experiments).

    uv run --no-project --with numpy --with scipy --with matplotlib python wrapper/paper/make_paper_figures.py

Writes wrapper/paper/iclr/figs/{ladder,tolerance,harm}.pdf (+ .png previews); score only on request ('score');
fidelity moved to make_image_figures.py (merged with its pictures); gallery.html no longer written.
One claim per figure; ML-conference style from palette.py (apply_ml, 2026-09-26): one blue ramp for all data (OFT darkest
circle, pi0.5 mid square; ladder rungs and harm arms light to dark, hatched for nothing), grey only for reference lines.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "results", "drive_vihan")
VEN = os.path.join(REPO, "vendor")
FIGS = os.path.join(HERE, "iclr", "figs")
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "wrapper", "scripts"))
import attribution_ladder as AL  # noqa: E402
import sweep_conditions as S  # noqa: E402
from plus_common import load_rows, load_shards, wilson  # noqa: E402
from sweep_analysis import auroc, boot_ci, load_scores  # noqa: E402

sys.path.insert(0, HERE)
import palette as P  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURF = P.INK, P.INK2, P.REF, P.GRID, P.AXIS, P.SURF
FAM = P.ML_FAM                                # ML-conference style (palette.py): one blue ramp, grey for references only
BUCKET = [("recovered by measurement", "measured pose", P.ML_RUNG["recovered by measurement"]),
          ("not identified", "only the true pose", P.ML_RUNG["not identified"]),
          ("not invertible", "only a perfect render", P.ML_RUNG["not invertible"]), ("policy", "nothing", P.ML_RUNG["policy"])]
P.apply_ml()
TEXTW = P.TEXTW


def grid(ax, axis="y"):
    ax.grid(False)
    ax.grid(True, axis=axis, color="#b0b0b0", linewidth=0.4, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def panel_letter(ax, s, dx=-0.12, dy=1.02):
    P.panel_label(ax, s, dx=dx, dy=dy)


def save(fig, name):
    """Saved at exactly the figure size (explicit axes placement), so text prints at its set size at the included width."""
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Figure: the attribution ladder
def fig_ladder():
    """(a) every raw camera failure attributed to the first rung that repairs it, per source (attribution_ladder.bucket, rep 0);
    (b) the calibration error of the measured pose per bucket (both families). Rung colours: palette.RUNG; 'nothing' hatched ink."""
    sets = [("LIBERO-10 set A", "OpenVLA-OFT", [f"{RES}/wrapper_oft/rows.jsonl", f"{RES}/rows_oft.jsonl"]),
            ("LIBERO-10 set B", "OpenVLA-OFT", [f"{RES}/wraps/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_10/wrap_openvla_oft.jsonl"]),
            ("LIBERO-Object", "OpenVLA-OFT", [f"{RES}/wraps_libero_object/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_object/wrap_openvla_oft.jsonl"]),
            ("LIBERO-Goal", "OpenVLA-OFT", [f"{RES}/wraps_libero_goal/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_goal/wrap_openvla_oft.jsonl"]),
            ("LIBERO-Spatial", "OpenVLA-OFT", [f"{RES}/wraps_libero_spatial/wrap_openvla_oft.jsonl", f"{RES}/wraps_ladder_libero_spatial/wrap_openvla_oft.jsonl"]),
            ("LIBERO-10 set A", "pi0.5", [f"{RES}/wraps/wrap_pi05.jsonl", f"{RES}/wraps_ladder/wrap_pi05.jsonl"])]
    counts, errs, pooled = [], defaultdict(list), defaultdict(int)
    for name, fam, files in sets:
        u = AL.units(files)
        fails = [v for v in u.values() if "raw" in v and "full" in v and not v["raw"]["success"]]
        c = defaultdict(int)
        for v in fails:
            b = AL.bucket(v)
            c[b] += 1
            e = v["full"].get("cam_est_err_cm_deg")
            if e and b in dict((k, 0) for k, _, _ in BUCKET):
                errs[b].append((e[0], fam))
        counts.append((name, fam, c, len(fails)))
        if fam == "OpenVLA-OFT":
            for k, n in c.items():
                pooled[k] += n
    fig = plt.figure(figsize=(TEXTW, 2.6), layout="none")
    ax = fig.add_axes([0.2, 0.155, 0.32, 0.66])
    y = np.arange(len(counts))[::-1].astype(float)
    y[-1] -= 0.55                                      # a gap before the held-out family's row
    for yi, (name, fam, c, n) in zip(y, counts):
        left = 0.0
        for key, label, col in BUCKET:
            k = c.get(key, 0)
            if k:
                w = 100.0 * k / n
                if key == "policy":
                    ax.barh(yi, w, left=left, height=0.64, facecolor=SURF, edgecolor=col, hatch="/////", linewidth=0.8, zorder=3)
                else:
                    ax.barh(yi, w, left=left, height=0.64, color=col, edgecolor=SURF, linewidth=1.0, zorder=3)
                if w >= 6:
                    ax.text(left + w / 2, yi, str(k), ha="center", va="center", fontsize=7, zorder=4,
                            color=SURF if col == P.BLUE[1] else INK,
                            bbox=dict(boxstyle="square,pad=0.12", fc=SURF, ec="none") if key == "policy" else None)
                left += w
        rest = c.get("upper rungs not run", 0)
        if rest:
            ax.barh(yi, 100.0 * rest / n, left=left, height=0.64, facecolor="none", edgecolor=MUTED, linewidth=0.5, zorder=3)
        ax.text(101.5, yi, f"n = {n}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _, _ in counts], fontsize=7)
    ax.text(-0.575, (y[0] + y[4]) / 2, "OFT", transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=7)
    ax.text(-0.575, y[5], r"$\pi_{0.5}$", transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=7)
    ax.plot([-0.54, -0.54], [y[4] - 0.3, y[0] + 0.3], transform=ax.get_yaxis_transform(), color=INK2, lw=0.6, clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("raw camera failures, by first repairing rung (%)")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [patches.Patch(facecolor=SURF if k == "policy" else col, edgecolor=col if k == "policy" else SURF,
                             hatch="/////" if k == "policy" else None, linewidth=0.6, label=label) for k, label, col in BUCKET]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.46, 1.0), ncol=2, columnspacing=1.0, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.15)
    panel_letter(ax, "a", dx=-0.6, dy=1.1)
    print("ladder pooled OFT:", dict(pooled), "per set:", [(nm, dict(c), n) for nm, _, c, n in counts])
    # (b) the independent check: calibration error per bucket
    ax2 = fig.add_axes([0.665, 0.155, 0.325, 0.66])
    rng = np.random.default_rng(0)
    nfam = {}
    for i, (key, label, col) in enumerate(BUCKET):
        ef = errs.get(key, [])
        if not ef:
            continue
        e = np.asarray([v for v, _ in ef], float)
        for fam in ("OpenVLA-OFT", "pi0.5"):          # marker shape per family (F18): OFT circle, pi0.5 square
            v = np.asarray([x for x, f in ef if f == fam], float)
            nfam[(key, fam)] = len(v)
            if not len(v):
                continue
            x = i + (-0.1 if fam == "OpenVLA-OFT" else 0.14) + rng.uniform(-0.1, 0.1, len(v))
            face = SURF if key == "policy" else col
            ax2.scatter(x, np.maximum(v, 0.05), s=12, marker=P.MARK[fam], facecolor=face, edgecolor=P.BLUE[0], linewidths=0.5, zorder=3)
        if len(e) > 1:                                  # no median bar for a single unit
            ax2.plot([i - 0.3, i + 0.3], [np.median(e)] * 2, color=INK, linewidth=1.3, zorder=4)
            ax2.text(i, 150, f"{np.median(e):.1f}", ha="center", fontsize=7, color=INK)
        ax2.text(i, 0.034, f"{nfam[(key, 'OpenVLA-OFT')]}/{nfam[(key, 'pi0.5')]}", ha="center", va="top", fontsize=7, color=INK2)
    ax2.text(-0.62, 0.034, "n:", ha="right", va="top", fontsize=7, color=INK2)
    hs = [plt.Line2D([], [], ls="", marker=P.MARK[f], ms=3.8, mfc=SURF, mec=P.BLUE[0], mew=0.6, label=P.FAMNAME[f])
          for f in ("OpenVLA-OFT", "pi0.5")]
    hs.insert(1, plt.Line2D([], [], color=INK, lw=1.3, label="median"))
    hs.append(plt.Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 3)), label="tolerance, about 1 cm"))
    ax2.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.04, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8,
               handlelength=1.4, labelspacing=0.3, borderaxespad=0.15)
    print("ladder (b) n per family per strip:", nfam)
    ax2.set_yscale("log")
    ax2.set_ylim(0.013, 260)
    ax2.set_yticks([0.1, 1, 10, 100])
    ax2.set_yticklabels(["0.1", "1", "10", "100"])
    ax2.minorticks_off()
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["measured\npose", "true\npose", "perfect\nrender", "nothing"], fontsize=7, linespacing=1.0)
    ax2.tick_params(axis="x", length=0)
    ax2.set_xlabel("first repairing rung")
    ax2.set_ylabel("calibration error (cm, log)")
    ax2.axhline(1.0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax2.set_xlim(-0.5, 3.5)
    grid(ax2, "y")
    panel_letter(ax2, "b", dx=-0.27, dy=1.1)
    save(fig, "ladder")


# ---------------------------------------------------------------- Figure: the score predicts repair
STD = {c["name"] for c in S.CONDITIONS}


def score_units(pol):
    rows, pred, _ = load_scores(f"{RES}/sweep")
    ep = {}
    for r in load_shards(f"{RES}/sweep/episodes_{pol}.jsonl"):
        if r.get("rep", 0) == 0 and r["cond"] in STD:
            ep.setdefault((r["task_id"], r["cond"], r["arm"]), r)
    tasks = sorted({k[0] for k in ep})
    ok = {t: ep.get((t, "nominal", "raw"), {}).get("success") for t in tasks}
    units = []
    for t in tasks:
        if not ok.get(t):
            continue
        for c in sorted(STD):
            if c == "nominal" or (t, c, "raw") not in ep or (t, c, "full") not in ep or ep[(t, c, "raw")]["success"]:
                continue
            p = pred.get((t, c), {})
            units.append(dict(task=t, cond=c, family=S.BY_NAME[c]["family"], recovered=int(bool(ep[(t, c, "full")]["success"])),
                              recover=p.get("recover", np.nan), label=float(S.BY_NAME[c]["family"] not in ("background",))))
    return [u for u in units if np.isfinite(u["recover"])]


def fig_score():
    U = {"OpenVLA-OFT": score_units("openvla_oft"), "pi0.5": score_units("pi05")}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXTW, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.15], wspace=0.38))
    edges = [0.0, 0.3, 0.55, 0.7, 0.85, 1.01]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    for fam, units in U.items():
        xs, ys, lo, hi = [], [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            g = [u["recovered"] for u in units if a <= u["recover"] < b]
            if len(g) >= 3:
                k, n = sum(g), len(g)
                w = wilson(k, n)
                xs.append(np.mean([u["recover"] for u in units if a <= u["recover"] < b])); ys.append(k / n); lo.append(w[0]); hi.append(w[1])
        ax.fill_between(xs, lo, hi, color=FAM[fam], alpha=0.12, linewidth=0)
        ax.plot(xs, ys, color=FAM[fam], linewidth=1.6, marker="o", markersize=4, markeredgecolor=SURF, zorder=4)
        y = [u["recovered"] for u in units]
        au = auroc([u["recover"] for u in units], y)
        lab = auroc([u["label"] for u in units], y)
        ax.text(0.03, 0.97 - 0.15 * (fam == "pi0.5"), f"{fam}: AUROC {au:.2f}, n = {len(units)}\n(category label {lab:.2f})", transform=ax.transAxes, fontsize=7, color=FAM[fam], ha="left", va="top")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("predicted recoverability (policy-free)")
    ax.set_ylabel("share of raw failures the wrapper repaired")
    grid(ax)
    panel_letter(ax, "a")
    # (b) per condition: predicted score against observed repair rate, both families
    for fam, units in U.items():
        by = defaultdict(list)
        for u in units:
            by[u["cond"]].append(u)
        for c, g in by.items():
            x, yv = np.mean([u["recover"] for u in g]), np.mean([u["recovered"] for u in g])
            ax2.scatter(x, yv, s=6 + 2.2 * len(g), color=FAM[fam], alpha=0.8, linewidths=0.6, edgecolors=SURF, zorder=3)
            # named conditions only (OFT dots), each with a leader line to a label placed in empty space
            off = {"look_away_55": (0.2, 0.27), "colored_room": (0.12, 0.36), "orbit_az30": (0.44, 0.9), "offset0.35": (0.8, 0.93),
                   "orbit_az45": (0.76, 0.4)}
            nice = {"look_away_55": "look away 55°", "colored_room": "coloured room", "orbit_az30": "orbit 30°", "offset0.35": "offset 0.35",
                    "orbit_az45": "orbit 45°"}
            if c in off and fam == "OpenVLA-OFT":
                ax2.annotate(nice[c], (x, yv), xytext=off[c], textcoords="data", fontsize=7, color=INK2, ha="center", va="center",
                             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("predicted recoverability, condition mean")
    ax2.set_ylabel("repair rate on that condition")
    grid(ax2)
    hs = [plt.Line2D([], [], marker="o", linestyle="", color=FAM[f], markersize=4, label=f) for f in FAM]
    ax2.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 0.76), handletextpad=0.1, fontsize=7, labelspacing=0.2, borderaxespad=0.1)
    ax2.text(0.02, 0.6, "dot area: raw failures\non the condition", transform=ax2.transAxes, fontsize=7, color=INK2, va="top", ha="left")
    panel_letter(ax2, "b")
    save(fig, "score")


# ---------------------------------------------------------------- Figure: the policy's calibration tolerance
MEAS = os.path.join(REPO, "results", "local", "measurability.jsonl")
OFT_CAMERA_FILES = [f"{RES}/wrapper_oft/rows.jsonl", f"{RES}/wraps/wrap_openvla_oft.jsonl", f"{RES}/wraps_libero_object/wrap_openvla_oft.jsonl",
                    f"{RES}/wraps_libero_goal/wrap_openvla_oft.jsonl", f"{RES}/wraps_libero_spatial/wrap_openvla_oft.jsonl"]


def selfcal_marks():
    """Every calibration-error mark on the tolerance axis, read from the rows (cm, median).
    Policy-free: results/local/measurability.jsonl, full excursion (scale 1.0), 8 libero_10 camera tasks, per pose count.
    In-rollout: arm `full` on camera tasks where the calibration was applied, all four OFT suites (the pooled camera set).
    Learned sensing: arm `full_hybrid` (learned masks + learned depth), 30 libero_10 camera tasks.
    DROID: the VLM-mask self-calibration on 53 real DROID cameras."""
    m = [r for r in load_rows(MEAS) if r["scale"] == 1.0]
    pose = {k: float(np.median([r["err_cm"] for r in m if r["n_poses"] == k])) for k in (1, 3, 7)}
    # F23 (2026-09-24): wraps/wrap_openvla_oft.jsonl repeats 6 set-A camera tasks (768, 800, 803, 956, 958, 1060) already read
    # from wrapper_oft/rows.jsonl; keep only its set-B task ids so no unit is counted twice (240 evaluated units, 219 logged).
    cam_full = lambda r: r.get("arm") == "full" and r.get("category") == "camera"
    set_a = {r["task_id"] for r in load_rows(OFT_CAMERA_FILES[0]) if cam_full(r)}
    units = [r for p in OFT_CAMERA_FILES for r in load_rows(p)
             if cam_full(r) and not (p == OFT_CAMERA_FILES[1] and r["task_id"] in set_a)]
    inroll = [r["cam_est_err_cm_deg"][0] for r in units if r.get("cal_applied") and r.get("cam_est_err_cm_deg")]
    learned = [r["cam_est_err_cm_deg"][0] for r in load_rows(f"{RES}/wraps/wrap_openvla_oft.jsonl")
               if r.get("arm") == "full_hybrid" and r.get("cam_est_err_cm_deg")]
    droid = [r["pos_err_cm"] for r in load_rows(f"{RES}/droid_selfcal_vlm/selfcal_results.jsonl")]
    out = dict(pose1=pose[1], pose3=pose[3], pose7=pose[7], n_meas_tasks=len({r["task_id"] for r in m}),
               inroll=float(np.median(inroll)), n_inroll=len(inroll), inroll_le2=float(np.mean(np.asarray(inroll) <= 2)), n_inroll_units=len(units),
               learned=float(np.median(learned)), n_learned=len(learned), droid=float(np.median(droid)), n_droid=len(droid))
    print("tolerance marks:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def fig_tolerance():
    """Rollout-free tolerance: fingerprint distance d of the opening action at true pose + added error vs at the true pose,
    median over the rows of results/drive_vihan/tolerance/tolerance_*.jsonl per family (IQR band); dotted: no correction."""
    rows = defaultdict(list)
    for p in sorted(glob.glob(f"{RES}/tolerance/tolerance_*.jsonl")):
        for r in load_rows(p):
            rows[{"openvla_oft": "OpenVLA-OFT", "pi05": "pi0.5"}.get(r["policy"], r["policy"])].append(r)
    ncount = {f: (len(rs), len({r["task_id"] for r in rs})) for f, rs in rows.items()}
    assert len(set(ncount.values())) == 1, ncount
    n_runs, n_tasks = next(iter(ncount.values()))
    mk = selfcal_marks()
    CM, DEG = (0.5, 1, 2, 4, 8, 16), (0.5, 1, 2, 4, 8)
    fig = plt.figure(figsize=(TEXTW * 0.85, 1.88), layout="none")     # printed at 0.85 linewidth: saved at that size
    axes = [fig.add_axes([0.105, 0.215, 0.5, 0.715]), fig.add_axes([0.72, 0.215, 0.215, 0.715])]
    ends = {}
    for ax, kind, grid_ in ((axes[0], "cm", CM), (axes[1], "deg", DEG)):
        for fam, rs in rows.items():
            med, lo, hi = [], [], []
            for d in grid_:
                per = [np.median([x["distance"] for x in r["doses"] if x["kind"] == kind and x["dose"] == d]) for r in rs]
                q = np.quantile(per, [0.25, 0.5, 0.75])
                lo.append(q[0]); med.append(q[1]); hi.append(q[2])
            ax.fill_between(grid_, lo, hi, color=FAM[fam], alpha=0.15, linewidth=0)
            ax.plot(grid_, med, color=FAM[fam], linewidth=1.3, marker=P.MARK[fam], markersize=3.0, markeredgecolor=SURF, markeredgewidth=0.4, zorder=4)
            ends[(kind, fam)] = (grid_[-1], med[-1])
            ax.axhline(np.median([r["uncorrected"] for r in rs]), color=FAM[fam], linewidth=1.0, linestyle=(0, (1, 1.6)), zorder=1)
        ax.axhline(0.10, color=MUTED, linewidth=0.7, linestyle=(0, (4, 2)), zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(grid_)
        ax.set_xticklabels([f"{g:g}" for g in grid_])
        ax.minorticks_off()
        ax.set_ylim(0, 0.76)
        ax.set_yticks([0, 0.1, 0.2, 0.4, 0.6])
        grid(ax)
    # family names at the right end of each curve, in both panels, pushed apart so they never overprint (F16)
    for ax, kind in ((axes[0], "cm"), (axes[1], "deg")):
        pts = sorted(((ends[(kind, f)][1], f) for f in rows), reverse=True)
        ys = [p[0] for p in pts]
        if ys[0] - ys[1] < 0.07:
            mid = (ys[0] + ys[1]) / 2
            ys = [mid + 0.04, mid - 0.04]
        dotted = [np.median([r["uncorrected"] for r in rs]) for rs in rows.values()]
        ys = [min(dotted) - 0.07 if any(abs(y - d) < 0.04 for d in dotted) else y for y in ys]   # never on a dotted line
        for (yv, fam), yl in zip(pts, ys):
            ax.annotate(P.FAMNAME[fam], (ends[(kind, fam)][0], yv), xytext=(ends[(kind, fam)][0] * (1.33 if kind == "cm" else 1.18), yl),
                        fontsize=7, color=FAM[fam], va="center", ha="left", annotation_clip=False)
    a0 = axes[0]
    a0.set_xlim(0.06, 48)
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.1, 0.2, 0.4, 0.6, 0.8])
    a0.axvspan(mk["pose7"], mk["pose1"], color=MUTED, alpha=0.2, linewidth=0, zorder=0)
    a0.text(np.sqrt(mk["pose7"] * mk["pose1"]), 0.56, f"self-cal.,\npolicy-free\n{mk['pose7']:.2f} to\n{mk['pose1']:.2f} cm",
            fontsize=7, color=INK2, ha="center", va="top", linespacing=1.05)
    a0.text(0.065, 0.115, "$d$ = 0.10 ($K$ = 7)", fontsize=7, color=INK2, va="bottom", ha="left")
    unc = max(np.median([r["uncorrected"] for r in rs]) for rs in rows.values())
    a0.text(0.065, unc + 0.012, "no correction (dotted)", fontsize=7, color=INK2, va="bottom")
    # the three reference errors, stacked as a staircase: each vertical line stops at its own label, a short horizontal
    # leader joins it to the label (right-aligned just left of the line), so no leader crosses another line
    for name, x, yl in ((f"in rollouts {mk['inroll']:.1f} cm", mk["inroll"], 0.8),
                        (f"learned sensing {mk['learned']:.1f} cm", mk["learned"], 0.88),
                        (f"DROID {mk['droid']:.1f} cm", mk["droid"], 0.96)):
        a0.plot([x, x], [0, yl], color=MUTED, linewidth=0.7, zorder=1)
        a0.plot([x / 1.12, x], [yl, yl], color=MUTED, linewidth=0.7, zorder=1)
        a0.text(x / 1.15, yl, name, fontsize=7, color=INK2, ha="right", va="center")
    a0.set_xlabel("added position error (cm, log)")
    a0.set_ylabel("fingerprint distance\n$d$ (unitless)")
    axes[1].set_xlim(0.4, 14)
    axes[1].set_xlabel("added rotation error (deg, log)")
    panel_letter(a0, "a", dx=-0.19, dy=1.0)
    panel_letter(axes[1], "b", dx=-0.42, dy=1.0)
    save(fig, "tolerance")
    print(f"tolerance: {ncount}")
    return mk


# ---------------------------------------------------------------- Figure: the harm and its removal
LOCAL = os.path.join(REPO, "results", "local")


def fig_harm():
    """(a) pi0.5, broken rate = raw successes the variant breaks / raw successes, per category and variant, Wilson 95%;
    summary row: the 170-unit graded sweep (arms full and lazy, rep 0). (b) the motion-free signal: the static residual
    (results/local/static_residual.jsonl, simulator masks, one rest pose; 60 unshifted = 10 nominal + 50 lighting units,
    70 camera shifts, policy-free) and, hollow, the pose-averaged silhouette score with LEARNED masks (arm full_learned in
    results/drive_vihan/wraps/wrap_openvla_oft.jsonl, field cal_nominal_score: robot_init rows unshifted, camera rows
    shifted), a different statistic on the same axis. Arms: grey ramp + marker shape (palette rule F48); direct labels."""
    u = defaultdict(dict)
    for r in load_rows(f"{RES}/wraps/wrap_pi05.jsonl"):
        if r.get("rep", 0) == 0:
            u[r["task_id"]].setdefault(r["arm"], r)
    S_ = [v for v in u.values() if all(a in v for a in ("raw", "full", "full_settle", "full_lazy"))]
    cats = ["background", "layout", "light", "sensor_noise"]
    arms = [("full", "always", P.ML_ARM["full"], "o"), ("full_settle", "settle", P.ML_ARM["full_settle"], "D"),
            ("full_lazy", "check", P.ML_ARM["full_lazy"], "s")]
    sw = defaultdict(dict)
    for r in load_shards(f"{RES}/sweep/episodes_pi05.jsonl"):
        if r.get("rep", 0) == 0:
            sw[(r["task_id"], r["cond"])].setdefault(r["arm"], r)
    SW = [v for v in sw.values() if all(a in v for a in ("raw", "full", "lazy"))]
    sw_s = [v for v in SW if v["raw"]["success"]]
    fig = plt.figure(figsize=(TEXTW, 2.15), layout="none")
    ax = fig.add_axes([0.185, 0.17, 0.33, 0.675])
    ax2 = fig.add_axes([0.655, 0.17, 0.335, 0.675])
    groups = [(c.replace("_", " "), [v for v in S_ if v["raw"]["category"] == c and v["raw"]["success"]]) for c in cats]
    rs_all = [v for v in S_ if v["raw"]["success"]]
    groups.append((f"all four ({len(S_)} tasks)", rs_all))
    ylab, yv, ypos = [], 0.0, []

    def mark(arm_col, mk, b, n, yy):
        lo, hi = wilson(b, n)
        ax.plot([lo, hi], [yy, yy], color=P.BLUE[1], linewidth=0.9, zorder=2, solid_capstyle="butt")
        ax.scatter([b / n], [yy], color=arm_col, marker=mk, s=22, zorder=3, edgecolors=P.BLUE[0], linewidths=0.6)
        return hi

    sr_all = {a: float(np.mean([bool(v[a]["success"]) for v in S_])) for a in ("raw", "full", "full_settle", "full_lazy")}
    for gi, (name, rs) in enumerate(groups):
        ypos.append(yv)
        for j, (arm, label, col, mk) in enumerate(arms):
            b = sum(not v[arm]["success"] for v in rs)
            yy = yv - (j - 1) * 0.27
            hi = mark(col, mk, b, len(rs), yy)
        ylab.append(f"{name}\n({len(rs)} raw succ.)")
        yv -= 1.0
    yv -= 0.3
    ypos.append(yv)
    for arm, (label, col, mk) in (("full", arms[0][1:]), ("lazy", arms[2][1:])):
        b = sum(not v[arm]["success"] for v in sw_s)
        yy = yv + (0.2 if arm == "full" else -0.2)
        hi = mark(col, mk, b, len(sw_s), yy)
        ax.text(hi + 0.03, yy, f"{label}  {b}/{len(sw_s)}", fontsize=7, va="center", color=INK)
    ylab.append(f"graded sweep\n({len(SW)} units, {len(sw_s)} succ.)")
    ax.axhline(ypos[-2] - 0.62, color="#d0d0d0", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "", "0.5", "", "1"])
    ax.set_xlabel("share of raw successes broken (Wilson 95%)")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    tot = {arm: sum(not v[arm]["success"] for v in rs_all) for arm, *_ in arms}
    # the arms' success rates over the 40 tasks sit in the key (they were direct labels on the pooled row, too tight at print size)
    srs = {a: (f"{sr_all[a]:.2f}" if round(sr_all[a], 2) == sr_all[a] else f"{sr_all[a]:.3f}") for a, *_ in arms}
    hs = [plt.Line2D([], [], ls="", marker=mk, ms=4.2, mfc=col, mec=P.BLUE[0], mew=0.6, label=f"{label} (SR {srs[arm]})")
          for arm, label, col, mk in arms]
    hs = [hs[0], hs[2], hs[1]]                     # legend fills by column: rows read always, settle / check
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.36, 1.0), ncol=2, handletextpad=0.1, columnspacing=1.0,
              labelspacing=0.3, borderaxespad=0.1)
    P.panel_label(ax, "a", dx=-0.53, dy=1.1)
    # (b) the motion-free signal: simulator masks (filled) and learned masks (hollow, a different statistic)
    sr = load_rows(os.path.join(LOCAL, "static_residual.jsonl"))
    cam = np.array([r["residual"] for r in sr if r["family"] == "camera"])
    oth = np.array([r["residual"] for r in sr if r["family"] != "camera"])
    fam_oth = sorted({r["family"] for r in sr if r["family"] != "camera"})
    lr = [r for r in load_rows(f"{RES}/wraps/wrap_openvla_oft.jsonl") if r.get("arm") == "full_learned" and r.get("cal_nominal_score") is not None]
    l_un = np.array([r["cal_nominal_score"] for r in lr if r["category"] != "camera"])
    l_sh = np.array([r["cal_nominal_score"] for r in lr if r["category"] == "camera"])
    rng = np.random.default_rng(1)
    X = (0.0, 1.0, 2.45, 3.45)
    for x, v, filled, col in ((X[0], oth, True, P.BLUE[2]), (X[1], cam, True, P.BLUE[0]), (X[2], l_un, False, P.BLUE[2]), (X[3], l_sh, False, P.BLUE[0])):
        jit = x + rng.uniform(-0.2, 0.2, len(v))
        if filled:
            ax2.scatter(jit, v, s=9, color=col, alpha=0.9, linewidths=0, zorder=3)
        else:
            ax2.scatter(jit, v, s=11, facecolor=SURF, edgecolor=col, linewidths=0.7, zorder=3)
    ax2.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ymax = max(cam.max(), l_sh.max()) * 1.6
    ax2.set_ylim(-0.12, ymax)
    ax2.set_yticks([0, 1, 10])
    ax2.set_yticklabels(["0", "1", "10"])
    ax2.axhline(1.5, color=MUTED, linewidth=0.7, linestyle=(0, (4, 2)), zorder=2)
    ax2.text(-0.4, 1.62, r"$\tau$ = 1.5", fontsize=7, va="bottom", ha="left", color=INK2)
    for (xa, xb), top, bot in (((X[0], X[1]), oth.max(), cam.min()), ((X[2], X[3]), l_un.max(), l_sh.min())):
        xm = (xa + xb) / 2
        ax2.annotate("", xy=(xm, bot), xytext=(xm, max(top, 0.0)), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
        ax2.text(xm + 0.08, 0.5, f"gap {top:.2f}\nto {bot:.2f}", fontsize=7, color=INK2, va="center", ha="left", linespacing=1.0)
    ax2.set_xticks(X)
    ax2.set_xticklabels([f"unshifted\nn = {len(oth)}", f"shifted\nn = {len(cam)}", f"unshifted\nn = {len(l_un)}", f"shifted\nn = {len(l_sh)}"],
                        fontsize=7, linespacing=1.05)
    ax2.tick_params(axis="x", length=0, pad=2)
    ax2.text((X[0] + X[1]) / 2, 1.02, "simulator masks\nstatic residual", transform=ax2.get_xaxis_transform(), fontsize=7, ha="center", va="bottom")
    ax2.text((X[2] + X[3]) / 2, 1.02, "learned masks\nscore $S$ (hollow)", transform=ax2.get_xaxis_transform(), fontsize=7, ha="center", va="bottom")
    ax2.axvline(1.72, color="#d0d0d0", linewidth=0.6)
    ax2.set_xlim(-0.45, 3.95)
    ax2.set_ylabel("silhouette residual (px;\nsymlog, linear below 1)")
    grid(ax2, "y")
    P.panel_label(ax2, "b", dx=-0.3, dy=1.02)
    save(fig, "harm")
    print(f"harm: {len(S_)} tasks, {len(rs_all)} raw successes, broken {tot}, SR {sr_all}; sweep {len(SW)} units, {len(sw_s)} raw successes; "
          f"residual unshifted max {oth.max():.3f} (n {len(oth)}, families {fam_oth}), shifted min {cam.min():.3f} (n {len(cam)}); "
          f"learned unshifted n {len(l_un)} max {l_un.max():.3f}, shifted n {len(l_sh)} min {l_sh.min():.3f}")


# ---------------------------------------------------------------- Figure: where the corrected frame is wrong
# MOVED 2026-09-24 to make_image_figures.fig_fidelity, merged with its pictures into one figure (fidelity.pdf).


def gallery():
    names = ["ladder", "tolerance", "harm", "fidelity", "envelope", "droid_masks", "paired_results"]
    cap = {"ladder": "Where the deficit lives: every raw camera failure attributed to the first arm that repairs it; right, the calibration error the wrapper made, per bucket.",
           "score": "The recoverability score predicts repair on both families (development and held out); the benchmark's category label is at chance.",
           "tolerance": "Rollout-free calibration tolerance of two frozen policies, with the robot's own calibration and the real-image result on the same axis.",
           "harm": "The measurement's arm motion harms pi0.5; a motion-free look at the robot's silhouette decides when to measure and removes most of it.",
           "fidelity": "Coverage predicts repair because it tracks damage to the pixels the camera did see, not hole area.",
           "envelope": "Recoverability envelope over camera placements (existing figure).",
           "droid_masks": "DROID: zero-shot masks against VLM-chosen masks (existing figure).",
           "paired_results": "Paired results (existing figure)."}
    items = "".join(f'<figure><img src="{n}.png"><figcaption><b>{n}</b>. {cap[n]}</figcaption></figure>'
                    for n in names if os.path.exists(os.path.join(FIGS, n + ".png")) or os.path.exists(os.path.join(FIGS, n + ".jpg")))
    items = items.replace('src="droid_masks.png"', 'src="droid_masks.jpg"') if not os.path.exists(os.path.join(FIGS, "droid_masks.png")) else items
    html = ("<!doctype html><meta charset='utf-8'><title>Paper figures</title><style>body{font:15px/1.5 Georgia,serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#0b0b0b;background:#f9f9f7}"
            "figure{margin:0 0 40px;background:#fff;padding:18px;border:1px solid #e1e0d9;border-radius:6px}img{width:100%;height:auto}figcaption{margin-top:10px;color:#52514e}</style>"
            "<h1 style='font-weight:normal'>Paper figures, rendered from the logged rows</h1>" + items)
    open(os.path.join(FIGS, "gallery.html"), "w", encoding="utf-8").write(html)
    print("wrote gallery.html")


if __name__ == "__main__":
    ALL = (fig_ladder, fig_tolerance, fig_harm)   # fig_score DROPPED (Table 1 carries the AUROCs); fig_fidelity MOVED to make_image_figures.py (merged with its pictures), 2026-09-24
    if "score" in sys.argv[1:]:
        ALL = ALL + (fig_score,)
    pick = set(sys.argv[1:])       # optional: names of the figures to build, e.g. `tolerance harm`
    for fn in ALL:
        if pick and fn.__name__[4:] not in pick:
            continue
        try:
            fn()
        except Exception as e:      # one broken figure must not stop the others
            import traceback
            print("FAILED", fn.__name__, repr(e))
            traceback.print_exc()
    pass   # gallery.html no longer written (P44)
