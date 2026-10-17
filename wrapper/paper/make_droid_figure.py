"""Figure: real DROID frames. Per camera (one row): the zero-shot mask the first attempt used, the mask chosen by the
VLM over SAM regions, and the robot re-rendered at OUR estimated camera pose over the frame (what the robot can check by
itself), with the calibration error against DROID's reference before and after.

Two steps:
  render   (needs MuJoCo + the local DROID npz files; vendor/.venv_mj) writes the tile sheet figs/droid_masks_src.png
           vendor/.venv_mj/Scripts/python.exe wrapper/paper/make_droid_figure.py render
  compose  (no simulator) crops the in-image label bars off the sheet and lays it out with column titles and per-row
           numbers read from the result rows, at 7 pt or larger; writes figs/droid_masks.png (PNG, lossless)
           uv run --no-project --with numpy --with matplotlib --with opencv-python-headless python wrapper/paper/make_droid_figure.py compose
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIGS = os.path.join(HERE, "iclr", "figs")
sys.path.insert(0, HERE)
import palette as P  # noqa: E402  (one palette for every figure script)
CAMS = ["AUTOLab_0d4edc83_2023-10-27-20h-16m-47s_22008760.npz", "IPRL_2fe9715d_2023-08-24-22h-14m-47s_25916956.npz",
        "TRI_52ca9b6a_2023-11-02-15h-23m-15s_25947356.npz", "AUTOLab_84bd5053_2023-07-14-16h-55m-45s_24400334.npz"]
W_, H_, GAP, BAR = 640, 360, 8, 58          # tile size, gap and the height of the label bar `render` draws
ld = lambda p: {json.loads(l)["file"]: json.loads(l) for l in open(p) if l.strip()}


def render():
    import importlib.util
    import cv2
    spec = importlib.util.spec_from_file_location("ds", os.path.join(REPO, "wrapper", "scripts", "droid_selfcal.py"))
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    L = os.path.join(os.path.expanduser("~"), "Downloads", "local_data", "plus")
    new, old = ld(os.path.join(L, "droid_selfcal_vlm", "selfcal_results.jsonl")), ld(os.path.join(L, "droid_selfcal", "selfcal_results.jsonl"))
    rob = ds.Franka(os.path.join(REPO, "vendor", "mujoco_menagerie"), W=W_, H=H_, hand=True)
    INK, RED, GREEN, BLUE = (245, 245, 245), (225, 80, 60), (60, 190, 110), (70, 140, 240)

    def tint(img, mask, col, a=0.45):
        out = img.astype(np.float32)
        m = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out[m] = (1 - a) * out[m] + a * np.array(col, np.float32)
        out = out.astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 2, cv2.LINE_AA)
        return out

    def label(img, text, sub=None):
        cv2.rectangle(img, (0, 0), (img.shape[1], 34 if sub is None else BAR), (20, 20, 20), -1)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, INK, 1, cv2.LINE_AA)
        if sub:
            cv2.putText(img, sub, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    rows = []
    for name in CAMS:
        d = np.load(os.path.join(L, "droid_selfcal_vlm", name), allow_pickle=True)
        t = int(np.argmax(d["mask"].reshape(len(d["mask"]), -1).sum(1)))             # the frame with the most robot in view
        img = cv2.resize(d["img"][t], (W_, H_), interpolation=cv2.INTER_CUBIC)
        fx, _, fy, _ = d["intr"]
        fy = fy if (np.isfinite(fy) and fy > 0) else (fx if (np.isfinite(fx) and fx > 0) else 525.0)
        wh_y = float(d["intr_wh"][1]) if (np.isfinite(d["intr_wh"][1]) and d["intr_wh"][1] > 0) else 720.0
        fyp = float(fy) * H_ / wh_y
        r = new[name]
        ren = rob.mask(d["q"][t], np.asarray(r["est_pos"]), ds.euler_xyz_to_mat(np.asarray(r["est_euler"])), fyp)
        a = label(tint(img, d["mask_owl"][t], RED), "zero-shot mask (text box + SAM)", f"calibration error {old[name]['pos_err_cm']:.0f} cm")
        b = label(tint(img, d["mask"][t], GREEN), "regions chosen by the VLM + own motion", f"calibration error {r['pos_err_cm']:.0f} cm")
        c = img.copy()
        cs, _ = cv2.findContours(ren.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c, cs, -1, BLUE, 2, cv2.LINE_AA)
        c = label(c, "robot re-rendered at the estimated camera", f"self-check IoU {r['est_render_iou']:.2f}")
        gap = np.full((H_, GAP, 3), 255, np.uint8)
        rows.append(np.concatenate([a, gap, b, gap, c], 1))
    sheet = np.concatenate([np.concatenate([x, np.full((GAP, x.shape[1], 3), 255, np.uint8)], 0) for x in rows], 0)[:-GAP]
    cv2.imwrite(os.path.join(FIGS, "droid_masks_src.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(sheet.shape)


GEOM = {}      # F21: per-camera anchors for the Claude Design bundle (frame pixels, origin top-left, x right, y down)


def _geom(m):
    """bbox [x0, y0, x1, y1], centroid and leader anchor (topmost pixel, median x of that row) of a boolean mask."""
    if not np.asarray(m).any():
        return None
    ys, xs = np.nonzero(m)
    top = ys.min()
    return dict(bbox=[int(xs.min()), int(top), int(xs.max() + 1), int(ys.max() + 1)],
                centroid=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                leader_anchor=[round(float(np.median(xs[ys == top])), 1), int(top)])


def recolor_tiles(tiles):
    """Rebuild the mask tiles in the shared palette (zero-shot palette.DROID_ZEROSHOT, VLM palette.DROID_VLM) from the local
    DROID npz files (the same frame `render` picked), and recolour the re-rendered silhouette of column 3 to palette.SIL_CHECK
    by differencing the sheet tile against the plain frame. No simulator. Falls back to the sheet's tiles when the local
    files are missing. Also exports clean frames and mask/outline layers to the Claude Design bundle."""
    import cv2
    L = os.path.join(os.path.expanduser("~"), "Downloads", "local_data", "plus", "droid_selfcal_vlm")
    out_dir = os.path.join(os.path.expanduser("~"), "Downloads", "claude_design_bundle", "paper_figures", "droid")
    if not all(os.path.exists(os.path.join(L, n)) for n in CAMS):
        print("DROID npz not found locally: keeping the sheet's colours")
        return tiles
    os.makedirs(out_dir, exist_ok=True)

    def tint(img, m, col, a=0.45):
        out = img.astype(np.float32)
        out[m] = (1 - a) * out[m] + a * np.array(col, np.float32)
        out = out.astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, 2, cv2.LINE_AA)
        return out

    def layer(m, col):
        a = np.zeros(m.shape + (4,), np.uint8)
        a[m] = (*col, 255)
        return cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA)

    new = []
    for r, name in enumerate(CAMS):
        d = np.load(os.path.join(L, name), allow_pickle=True)
        t = int(np.argmax(d["mask"].reshape(len(d["mask"]), -1).sum(1)))
        img = cv2.resize(d["img"][t], (W_, H_), interpolation=cv2.INTER_CUBIC)
        rs = lambda m: cv2.resize(m.astype(np.uint8), (W_, H_), interpolation=cv2.INTER_NEAREST) > 0
        m_owl, m_vlm = rs(d["mask_owl"][t]), rs(d["mask"][t])
        a = tint(img, m_owl, P.hex_rgb(P.DROID_ZEROSHOT))
        b = tint(img, m_vlm, P.hex_rgb(P.DROID_VLM))
        c3 = tiles[r][2]
        base = img[BAR:]                                       # the sheet's tiles lost the label bar at the top
        line = (np.abs(c3.astype(np.int16) - base.astype(np.int16)).sum(-1) > 60)
        line[:3], line[-3:], line[:, :3], line[:, -3:] = False, False, False, False     # PNG/resize seams at the tile edge
        c = base.copy()
        c[line] = P.hex_rgb(P.SIL_CHECK)
        new.append([a[BAR:], b[BAR:], c])
        stem = f"{r + 1}_{name.split('_')[0]}_{name.split('_')[-1][:-4]}"
        cv2.imwrite(os.path.join(out_dir, stem + "_frame.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, stem + "_layer_zero_shot_mask.png"), layer(m_owl, P.hex_rgb(P.DROID_ZEROSHOT)))
        cv2.imwrite(os.path.join(out_dir, stem + "_layer_vlm_mask.png"), layer(m_vlm, P.hex_rgb(P.DROID_VLM)))
        full = np.zeros((H_, W_), bool)
        full[BAR:] = line
        cv2.imwrite(os.path.join(out_dir, stem + "_layer_rerendered_robot_outline.png"), layer(full, P.hex_rgb(P.SIL_CHECK)))
        GEOM[name] = dict(stem=stem, tile_px=[W_, H_], vlm_mask=_geom(m_vlm), zero_shot_mask=_geom(m_owl), rerendered_outline=_geom(full))
    print("DROID tiles recoloured to the shared palette; clean frames + layers exported to", out_dir)
    return new


def compose():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P.apply()
    R = os.path.join(REPO, "results", "drive_vihan")
    new, old = ld(os.path.join(R, "droid_selfcal_vlm", "selfcal_results.jsonl")), ld(os.path.join(R, "droid_selfcal", "selfcal_results.jsonl"))
    sheet = cv2.cvtColor(cv2.imread(os.path.join(FIGS, "droid_masks_src.png")), cv2.COLOR_BGR2RGB)
    tiles = [[sheet[r * (H_ + GAP) + BAR:r * (H_ + GAP) + H_, c * (W_ + GAP):c * (W_ + GAP) + W_] for c in range(3)] for r in range(len(CAMS))]
    tiles = recolor_tiles(tiles)
    all_new = np.array([v["pos_err_cm"] for v in new.values()])
    paired = [f for f in new if f in old]
    print(f"DROID: {len(new)} cameras (VLM masks), median {np.median(all_new):.1f} cm; zero-shot on the same {len(paired)}: "
          f"median {np.median([old[f]['pos_err_cm'] for f in paired]):.1f} cm")
    print("rank of the shown cameras by VLM-mask error (1 = best):", [int((all_new < new[f]["pos_err_cm"]).sum()) + 1 for f in CAMS])
    print("shown rows:", [(f.split('_')[0], f.split('_')[-1][:-4], round(old[f]['pos_err_cm'], 1), round(new[f]['pos_err_cm'], 1),
                           round(new[f]['est_render_iou'], 2)) for f in CAMS])
    W = 5.5
    th = (W * 0.795 / 3) * tiles[0][0].shape[0] / tiles[0][0].shape[1]          # tile height at the included width
    fig, axes = plt.subplots(len(CAMS), 3, figsize=(W, th * len(CAMS) * 1.04 + 0.34))
    cols = ["zero-shot mask\n(text box + SAM)", "VLM-chosen regions\n+ own motion", "robot re-rendered at\nthe estimated camera"]
    for r, name in enumerate(CAMS):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(tiles[r][c], interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, pad=3)
        lab = (f"{name.split('_')[0]} {name.split('_')[-1][:-4]}\nerror {old[name]['pos_err_cm']:.0f} → "
               f"{new[name]['pos_err_cm']:.0f} cm\nself-check IoU {new[name]['est_render_iou']:.2f}")
        axes[r, 0].text(-0.04, 0.5, lab, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7, linespacing=1.15)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, left=0.2, right=0.995, top=1 - 0.34 / (th * len(CAMS) * 1.04 + 0.34), bottom=0.005)
    fig.savefig(os.path.join(FIGS, "droid_masks.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIGS, "droid_masks.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.02)     # vector text, raster tiles
    plt.close(fig)
    print("wrote droid_masks.png and droid_masks.pdf")
    if GEOM:                                            # F21: droid entries of the bundle's annotations.json
        p = os.path.join(os.path.expanduser("~"), "Downloads", "claude_design_bundle", "paper_figures", "annotations.json")
        ann = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        ann["droid_masks"] = dict(
            figure="droid_masks.pdf (appendix)", colours=dict(zero_shot=P.DROID_ZEROSHOT, vlm=P.DROID_VLM, rerendered=P.SIL_CHECK),
            n_cameras=len(new), median_err_cm_zero_shot=round(float(np.median([old[f]["pos_err_cm"] for f in paired])), 1),
            median_err_cm_vlm=round(float(np.median(all_new)), 1),
            rows=[dict(row=r + 1, camera=f"{n.split('_')[0]} {n.split('_')[-1][:-4]}", err_cm_zero_shot=round(old[n]["pos_err_cm"], 1),
                       err_cm_vlm=round(new[n]["pos_err_cm"], 1), selfcheck_iou=round(new[n]["est_render_iou"], 2), **GEOM.get(n, {}))
                  for r, n in enumerate(CAMS)],
            anchor_note="pixels of the exported *_frame.png (640 x 360; the re-rendered outline layer is offset by the 58 px label bar)")
        json.dump(ann, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("annotations.json: droid_masks entries written")


if __name__ == "__main__":
    {"render": render, "compose": compose}[(sys.argv[1:] or ["compose"])[0]]()
