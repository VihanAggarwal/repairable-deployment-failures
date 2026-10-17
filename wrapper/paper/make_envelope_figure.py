"""Figure: the recoverability envelope (main text) and the tilt curve (appendix).

  envelope.pdf       (a) the policy-free score recover = ident x invert over camera placements around the training view
                         (mean over the base tasks), with the graded sweep's measured outcomes overlaid;
                     (b) reliability of the camera term (self-measured coverage) on the LIBERO-Plus camera raw failures.
  envelope_tilt.pdf  the same score as the camera rotates away from the scene (look-away tilt and elevation), with the
                     sweep's look-away outcomes.

Inputs, all in the repo:
  results/local/envelope.jsonl             policy-free scores per placement (vendor/envelope.py, local, no policy)
  results/drive_vihan/sweep/episodes_*.jsonl  graded sweep episodes (every session shard merged), rep 0
  results/local/coverage_units.jsonl       one row per LIBERO-Plus camera task (wrapper/scripts/build_coverage_units.py)

    uv run --no-project --with numpy --with matplotlib python wrapper/paper/make_envelope_figure.py
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "wrapper", "scripts"))
from plus_common import load_shards  # noqa: E402
from sweep_analysis import auroc  # noqa: E402

sys.path.insert(0, HERE)
import palette as P  # noqa: E402
P.apply_ml()                                                  # ML-conference style: one blue ramp, grey for references
INK, INK2, MUTED, SURF = P.INK, P.INK2, P.REF, P.SURF        # one palette for every figure script (palette.py)
FAMC = P.ML_FAM
FIGS = os.path.join(HERE, "iclr", "figs")
rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "envelope.jsonl")) if l.strip()]
tasks = sorted({r["task_id"] for r in rows})

# ---- the map: mean over base tasks
az_vals = sorted({r["az"] for r in rows if "az" in r and "el" not in r})
sc_vals = [0.8, 1.0, 1.3, 1.6]
grid = np.full((len(sc_vals), len(az_vals)), np.nan)
for i, sc in enumerate(sc_vals):
    for j, az in enumerate(az_vals):
        v = [r["recover"] for r in rows if r.get("az") == az and r.get("scale") == sc and "el" not in r]
        if v:
            grid[i, j] = float(np.mean(v))
tilt, elev = defaultdict(list), defaultdict(list)
for r in rows:
    if "tilt" in r:
        tilt[r["tilt"]].append(r["recover"])
    if "el" in r and r.get("az", 0) == 0:
        elev[abs(r["el"])].append(r["recover"])

# ---- measured outcomes at the sweep's camera conditions: repaired / raw failures, per policy (rep 0)
# F27 (2026-09-24): only the 5 STANDARD non-look-away camera placements (sweep_conditions.CONDITIONS; the extended
# orbit 15 / 25 / 38 were never part of the registered sweep and are not pooled), and only base tasks the policy solves
# unperturbed (raw nominal success), the same unit definition as the score's prediction units.
COND = {"orbit_az10": (10, 1.0), "orbit_az20": (20, 1.0), "orbit_az30": (30, 1.0), "orbit_az45": (45, 1.0), "pullback_1.3": (0, 1.3)}
TILT = {"look_away_15": 15, "look_away_25": 25, "look_away_35": 35, "look_away_55": 55}
SWEEP = os.path.join(REPO, "results", "drive_vihan", "sweep")
policies = {"OFT": "episodes_openvla_oft.jsonl", "pi0.5": "episodes_pi05.jsonl"}
out, unsolved_too = {}, {}
for pol, fn in policies.items():
    by = defaultdict(dict)
    for r in load_shards(os.path.join(SWEEP, fn)):
        if r.get("rep", 0) == 0:
            by[(r["task_id"], r["cond"])].setdefault(r["arm"], bool(r["success"]))
    solved = {tid for (tid, cond), arms in by.items() if cond == "nominal" and arms.get("raw")}
    for (tid, cond), arms in by.items():
        if cond in COND and "raw" in arms and "full" in arms and not arms["raw"]:     # logged before the solved filter
            d0 = unsolved_too.setdefault(pol, [0, 0, set()])
            d0[0] += 1; d0[1] += int(arms["full"]); d0[2].add(cond)
        if cond in COND and tid not in solved:
            continue
        if "raw" in arms and "full" in arms and (cond in COND or cond in TILT):
            d = out.setdefault((pol, cond), [0, 0, 0])          # raw failures, repaired, units
            d[2] += 1
            if not arms["raw"]:
                d[0] += 1
                d[1] += int(arms["full"])
marks = {"OFT": "o", "pi0.5": "s"}

# ---- reliability units
U = [json.loads(l) for l in open(os.path.join(REPO, "results", "local", "coverage_units.jsonl")) if l.strip()]
F = [u for u in U if not u["raw"] and u["coverage"] is not None]
c = np.array([u["coverage"] for u in F]); y = np.array([int(u["full"]) for u in F])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    m, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h


def boot_auc(x, yy, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auroc(x[i], yy[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def edges_of(v):
    """Cell edges at the midpoints of consecutive azimuths; the outer cells are as wide as their inner neighbour."""
    v = np.asarray(v, float)
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])


def fig_envelope():
    fig = plt.figure(figsize=(P.TEXTW, 1.88), layout="none")          # explicit spacing, saved at exactly the printed size
    ax = fig.add_axes([0.085, 0.255, 0.42, 0.62])
    cax = fig.add_axes([0.515, 0.255, 0.012, 0.62])
    ax3 = fig.add_axes([0.72, 0.255, 0.245, 0.62])
    ax.grid(False)
    ax.set_facecolor(SURF)                                                # placements not rendered: white (hatched below)
    xe = edges_of(az_vals)
    ye = np.arange(len(sc_vals) + 1) - 0.5
    for i in range(len(sc_vals)):                                        # unrendered cells: light hatch, not a map value
        for j in range(len(az_vals)):
            if np.isnan(grid[i, j]):
                ax.add_patch(plt.Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], 1, facecolor=SURF, edgecolor="#d0d0d0", hatch="////", linewidth=0.3, zorder=1))
    # the Blues ramp truncated to its light half (0.03 to 0.55), so the dark family markers keep their contrast on every cell
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("blues_light", plt.get_cmap("Blues")(np.linspace(0.03, 0.55, 256)))
    im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid), cmap=blues, vmin=0, vmax=1, shading="flat", edgecolors=SURF, linewidth=0.3, zorder=2)
    ax.set_xticks([-120, -90, -60, -30, 0, 30, 60, 90, 120])
    ax.set_xlim(xe[0], xe[-1])
    ax.set_yticks(range(len(sc_vals)), [f"{s:g}x" for s in sc_vals])
    ax.set_xlabel("camera azimuth from the training view (deg)")
    ax.set_ylabel("camera distance (x training)")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)
    tot = {"OFT": [0, 0, 0], "pi0.5": [0, 0, 0]}                            # repaired, raw failures, placements
    for (pol, cond), (nf, nr, n) in sorted(out.items()):
        if cond in COND and nf:
            az, sc = COND[cond]
            yv = sc_vals.index(sc) + (0.3 if pol == "OFT" else -0.3)          # OFT upper third, pi0.5 lower third of the cell
            filled = nr >= nf / 2
            ax.scatter([az], [yv], marker=marks[pol], s=34, facecolor=SURF, edgecolor=SURF, linewidth=2.0, zorder=3)   # white halo
            ax.scatter([az], [yv], marker=marks[pol], s=15, facecolor=FAMC[pol] if filled else SURF, edgecolor=FAMC[pol], linewidth=1.1, zorder=4)
            tot[pol][0] += nr; tot[pol][1] += nf; tot[pol][2] += 1
    # key above the map (frameless), the totals in the entries
    hs = [plt.Line2D([], [], ls="", marker=marks[pol], ms=4, mfc=FAMC[pol], mec=FAMC[pol],
                     label=(f"OFT: {tot[pol][0]} of {tot[pol][1]} raw failures repaired" if pol == "OFT" else
                            f"$\\pi_{{0.5}}$: {tot[pol][0]} of {tot[pol][1]}")) for pol in marks]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("predicted recover (0-1)", fontsize=7, labelpad=2)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    P.panel_label(ax, "a", dx=-0.1, dy=1.01)
    # (b) reliability, 5 equal-mass bins; k/n in one row under the axis
    order = np.argsort(c)
    ax3.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    xs, ys, lo, hi, kn = [], [], [], [], []
    for b in np.array_split(order, 5):
        k, n = int(y[b].sum()), len(b)
        w = wilson(k, n)
        xs.append(c[b].mean()); ys.append(k / n); lo.append(k / n - w[0]); hi.append(w[1] - k / n); kn.append(f"{k}/{n}")
    ax3.errorbar(xs, ys, yerr=[lo, hi], color=P.BLUE[0], lw=1.2, marker="o", ms=3.2, capsize=1.6, elinewidth=0.8, ecolor=P.BLUE[1], zorder=3)
    a = auroc(c, y)
    l_, h_ = boot_auc(c, y)
    fam_n = {p: sum(u["policy"] == p for u in F) for p in ("OpenVLA-OFT", "pi0.5")}
    pm = np.array([u["policy"] == "pi0.5" for u in F])
    a_pi = auroc(c[pm], y[pm])
    a_oft = auroc(c[~pm], y[~pm])
    ax3.text(0.02, 1.0, f"AUROC {a:.2f} [{l_:.2f}, {h_:.2f}]\nn = {len(F)} ({fam_n['OpenVLA-OFT']} OFT, {fam_n['pi0.5']} $\\pi_{{0.5}}$)",
             transform=ax3.transAxes, fontsize=7, va="top")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.3)          # headroom above y = 1 holds the AUROC text, clear of the data
    ax3.set_yticks([0, 0.5, 1]); ax3.set_xticks([0, 0.5, 1])
    ax3.set_xlabel("self-measured coverage c (0-1)\n" + "repaired/n: " + ", ".join(kn), fontsize=7, linespacing=1.3, ha="right")
    ax3.xaxis.set_label_coords(1.0, -0.2)                  # right-aligned so the long count line stays on the page
    ax3.set_ylabel("share of raw failures\nrepaired (0-1)")
    P.panel_label(ax3, "b", dx=-0.37, dy=1.01)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"envelope.{ext}"), dpi=300)
    plt.close(fig)
    print(f"envelope: {len(rows)} placement rows, tasks {tasks}; sweep overlay, solved base tasks only (repaired, raw failures, "
          f"placements) {tot}; before the solved filter (raw failures, repaired, placements) "
          f"{ {k: (v[0], v[1], len(v[2])) for k, v in unsolved_too.items()} }; reliability n {len(F)} {fam_n}, AUROC {a:.3f} "
          f"[{l_:.2f}, {h_:.2f}], OFT alone {a_oft:.3f}, pi0.5 alone {a_pi:.3f}, bins {kn}")


def fig_tilt():
    fig, ax = plt.subplots(figsize=(0.65 * P.TEXTW, 2.2))
    ts = sorted(tilt)
    pt = [np.mean(tilt[t]) for t in ts]
    ax.plot(ts, pt, color=P.BLUE[0], lw=1.3, marker="o", ms=3.0)
    ax.text(46, 0.36, "predicted,\nlook-away tilt", fontsize=7, color=P.BLUE[0], va="bottom", ha="left")
    if elev:
        es = sorted(elev)
        pe = [np.mean(elev[e]) for e in es]
        ax.plot(es, pe, color=P.BLUE[2], lw=1.2, ls=(0, (3, 2)), marker="^", ms=3.2)
        ax.text(es[-1] + 1.2, pe[-1], "predicted,\nelevation", fontsize=7, color=P.BLUE[1], va="center", ha="left")
    for (pol, cond), (nf, nr, n) in sorted(out.items()):
        if cond in TILT and nf:
            t = TILT[cond] + (-1.2 if pol == "OFT" else 1.2)
            ax.scatter([t], [nr / nf], marker=marks[pol], s=22, facecolor="white", edgecolor=FAMC[pol], linewidth=1.2, zorder=3)
            above = pol == "OFT"
            if above:                                   # OFT: label to the left of its marker (above it sits the curve)
                ax.text(t - 1.8, nr / nf, f"{nr}/{nf}", fontsize=7, va="center", ha="right", color=FAMC[pol])
            else:
                ax.text(t, nr / nf - 0.045, f"{nr}/{nf}", fontsize=7, va="top", ha="center", color=FAMC[pol])
    hs = [plt.Line2D([], [], ls="", marker=marks[p], ms=4.5, mfc="white", mec=FAMC[p], mew=1.2,
                     label=("OFT (label left)" if p == "OFT" else r"$\pi_{0.5}$ (label below)")) for p in marks]
    ax.legend(handles=hs, loc="upper right", title="repaired / raw failures", title_fontsize=7, handletextpad=0.3,
              labelspacing=0.3, borderaxespad=0.1)
    ax.set_ylim(-0.1, 1)
    ax.set_xlim(5, 72)
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.set_xlabel("rotation away from the scene (deg)")
    ax.set_ylabel("score recover (lines) or\nrepaired share (markers), 0-1")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"envelope_tilt.{ext}"), dpi=300)
    plt.close(fig)
    print("tilt:", {k: round(float(np.mean(v)), 3) for k, v in sorted(tilt.items())}, "overlay", {k: v for k, v in out.items() if k[1] in TILT})


if __name__ == "__main__":
    fig_envelope()
    fig_tilt()
