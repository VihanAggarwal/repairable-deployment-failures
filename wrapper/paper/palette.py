"""One palette for every paper figure (Okabe-Ito, colour-blind safe), imported by every make_*.py figure script.

  families     OFT (OpenVLA-OFT) #0072B2 (blue), pi0.5 #D55E00 (vermillion), OpenVLA fine-tuned on LIBERO-10 #009E73
               (green); marker shape per family MARK: OFT circle, pi0.5 square, OpenVLA triangle
  RULE (2026-09-24, F48): blue, vermillion and green are RESERVED for the three policy families and mean nothing else in
               any figure. Every ordinal category that is not a family (ladder rungs, harm arms, fidelity curves) is drawn
               on the grey ramp GREY3 = #d9d9d9 / #969696 / #525252 (light to dark, in the order below) plus ink, told
               apart by marker shape or dash style as well as lightness, with hatched white for 'nothing'.
  arms / rungs GREY3 in this order, hatched white with ink edge for 'nothing'
               ladder rungs: measured pose, only the true pose, only a perfect render, nothing
               harm arms:    measure always (circle), measure then settle (diamond), motion-free check first (square)
  overlays     measured silhouette #E69F00 outline, check silhouette #56B4E9 outline, holes #F0E442 at 40% alpha;
               DROID zero-shot mask #CC79A7, VLM mask #E69F00; the 3D bundle's semantic colours (make_image_figures
               TRAIN_C / DEPLOY_C) are training #14B4C8 and deployment #E67814
  maps         sequential Blues for quantities on [0, 1]; Greys for pixel error
  ink          text #0b0b0b, secondary text #52514e, muted marks #898781, grid #e1e0d9, axes #9a998f
Print rules: text >= 7 pt at the included width, data lines >= 1.0 pt, reference lines >= 0.6 pt, direct labels.
"""
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#9a998f", "#ffffff"
FAM = {"OpenVLA-OFT": "#0072B2", "pi0.5": "#D55E00", "OpenVLA": "#009E73"}
FAM["OFT"] = FAM["OpenVLA-OFT"]
MARK = {"OpenVLA-OFT": "o", "OFT": "o", "pi0.5": "s", "OpenVLA": "^"}
FAMNAME = {"OpenVLA-OFT": "OFT", "OFT": "OFT", "pi0.5": r"$\pi_{0.5}$", "OpenVLA": "OpenVLA"}   # display strings
GREY3 = ("#d9d9d9", "#969696", "#525252")     # light, mid, dark (F48)
ARM3 = GREY3                                   # alias kept: scripts that drew arms / rungs in hues now draw greys
RUNG = {"recovered by measurement": ARM3[0], "not identified": ARM3[1], "not invertible": ARM3[2], "policy": INK}
ARM = {"full": ARM3[0], "full_settle": ARM3[1], "full_lazy": ARM3[2]}
SIL_MEASURED, SIL_CHECK, HOLES, HOLES_ALPHA = "#E69F00", "#56B4E9", "#F0E442", 0.40
DROID_ZEROSHOT, DROID_VLM = "#CC79A7", "#E69F00"     # VLM masks were green (#009E73), the OpenVLA family colour (F48)
SEQ, ERR = "Blues", "Greys"
FONT_MIN = 7.0
TEXTW = 5.5          # ICLR text width, inches

RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "legend.frameon": False, "figure.dpi": 110, "hatch.linewidth": 0.6, "mathtext.fontset": "stix",
}


def apply():
    plt.rcParams.update(RC)


# ---- ML-conference style for the five data charts (envelope, envelope_tilt, ladder, harm, tolerance), 2026-09-26.
# ONE colour family (a sequential blue ramp) for all data, told apart by shade plus marker / line style; neutral grey
# only for reference lines, chance lines and reference bands. Sans-serif, 7 to 8 pt at the printed width.
BLUE = ("#08306b", "#2171b5", "#6baed6", "#c6dbef")        # darkest to lightest
REF = "#8c8c8c"                                              # reference / chance lines and bands only
ML_FAM = {"OpenVLA-OFT": BLUE[0], "OFT": BLUE[0], "pi0.5": "#4292c6"}   # OFT darkest (circle), pi0.5 mid (square)
ML_RUNG = {"recovered by measurement": BLUE[3], "not identified": BLUE[2], "not invertible": BLUE[1], "policy": BLUE[0]}
ML_ARM = {"full": BLUE[3], "full_settle": BLUE[2], "full_lazy": BLUE[0]}
ML_RC = dict(RC)
ML_RC.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.titlesize": 7.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#b0b0b0", "grid.linewidth": 0.4,
    "grid.alpha": 0.3, "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.2, "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})


def apply_ml():
    plt.rcParams.update(ML_RC)


def panel_label(ax, s, dx=-0.12, dy=1.02):
    """Bold '(a)' at the top-left, outside the axes (axes fraction; dx negative moves it left of the spine)."""
    ax.text(dx, dy, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")


def hex_rgb(c):
    return tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
