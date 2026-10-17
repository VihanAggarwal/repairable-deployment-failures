"""Real-image pose sources for DROID self-calibration (task W22). Scored exactly like droid_selfcal.py: camera pose in
the robot base frame against DROID's post-hoc calibrated extrinsics (cam2base_extrinsic_superset), per camera file.

Sources (one row per camera per source in posesrc_results.jsonl):
  ctrnetx     CtRNet-X keypoints (pretrained, darthandvader/CtRNet-X) + PnP over all K frames of the camera. The CLIP
              visible-part filter of the original pipeline is replaced by its keypoint-confidence filter alone
              (threshold 0.08, the repo's value); disclosed deviation.
  sam2        render-and-segment: render the arm at the current pose, prompt SAM 2 with the render's box plus
              points sampled inside / outside it, refine the pose against the SAM 2 masks (local CEM on soft IoU),
              repeat for ROUNDS rounds. Started from the zero-shot W6 v6 estimate (`--init w6`) or from CtRNet-X
              (`--init ctrnetx`, the hybrid).
Mask choice is SAM 2's own predicted score among its multimask outputs (not IoU with the render), so the loop cannot
simply copy the render; a mask is kept only if it overlaps the render at IoU >= 0.1 (otherwise the frame is skipped).

  run     --source ctrnetx|sam2 [--init w6|ctrnetx] over every *.npz in --cams (droid_selfcal export + segment files)
  report  per source: median position / rotation error, within 5 cm / 5 deg and 10 cm / 10 deg, on all cameras and on
          references DROID flags reliable (quality < 20), paired against the W6 rows on the same cameras.
"""
import argparse
import glob
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from droid_selfcal import (Franka, euler_xyz_to_mat, mat_to_euler_xyz, pose_err, soft_iou, iou)  # noqa: E402

ROUNDS = 3
CONF_THR = 0.08
W, H = 320, 180                     # DROID RLDS exterior frames; also the render size here


# ------------------------------------------------------------------ shared
def load_cam(p, median_fy=530.0):
    """Frames, joints, reference pose (C, R in base frame), fy in pixels at W x H, intrinsics at W x H."""
    d = np.load(p, allow_pickle=True)
    fx, cx, fy, cy = [float(v) for v in d["intr"]]
    wh = d["intr_wh"].astype(float)
    if not (np.isfinite(fx) and fx > 0):
        fx = fy = median_fy
        cx, cy = 640.0, 360.0
        wh = np.array([1280.0, 720.0])
    if not np.isfinite(wh).all() or wh.min() <= 0:
        wh = np.array([1280.0, 720.0])
    s = W / wh[0]
    K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = euler_xyz_to_mat(d["ext"][3:6]), d["ext"][:3]
    return d, K, T


def reference_pose(rob, d, K, T):
    """DROID stores cam2base; pick the convention whose render matches the old masks better (same check as W6)."""
    variants = {"cam2base": T, "base2cam": np.linalg.inv(T)}
    obs = d["mask"] if "mask" in d.files else None
    if obs is None:
        return T[:3, 3], T[:3, :3], "cam2base"
    sc = {k: np.mean([iou(rob.mask(d["q"][i], V[:3, 3], V[:3, :3], K[1, 1]), obs[i]) for i in range(len(obs))])
          for k, V in variants.items()}
    k = max(sc, key=sc.get)
    return variants[k][:3, 3], variants[k][:3, :3], k


def cem(score, C0, R0, sd0, iters, rng, pop=32, elite=6):
    mu, sd = np.zeros(6), np.asarray(sd0, float)
    best = (score(C0, R0), C0, R0)
    for _ in range(iters):
        X = mu + sd * rng.standard_normal((pop, 6))
        X[0] = mu
        vals = []
        for x in X:
            C, R = C0 + x[:3], euler_xyz_to_mat(x[3:]) @ R0
            v = score(C, R)
            vals.append(v)
            if v < best[0]:
                best = (v, C, R)
        el = X[np.argsort(vals)[:elite]]
        mu, sd = el.mean(0), el.std(0) + 1e-3
    return best


# ------------------------------------------------------------------ CtRNet-X
def ctr_to_cam_pose(cTr):
    """CtRNet-X cTr = (axis-angle, translation) of the robot base in the CAMERA frame -> camera pose in the base frame."""
    import cv2
    R_cb, _ = cv2.Rodrigues(np.asarray(cTr[:3], float).reshape(3, 1))
    t_cb = np.asarray(cTr[3:6], float)
    R = R_cb.T
    C = -R_cb.T @ t_cb
    return C, R


def cam_pose_to_ctr(C, R):
    import cv2
    R_cb = R.T
    t_cb = -R.T @ C
    rv, _ = cv2.Rodrigues(R_cb)
    return np.concatenate([rv.ravel(), t_cb])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class PandaKeypoints:
    """Drop-in for CtRNet-X's PandaArm.get_3d_keypoints without roboticstoolbox (the repo pins 1.0.1, which has no
    wheel for current Python). Joint origins are read from the repo's own panda.urdf; links[k] there is the frame of
    panda_link<k>, i.e. after joint k. Keypoint offsets are copied from models/robot_arm.py."""
    BASE = np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.0, 0.07, 0.0], [0.0, 0.0, 0.1], [-0.135, 0.0, 0.0], [0.0, -0.07, 0.0]])
    L6 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, -0.125]])
    L7 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, -0.07]])

    def __init__(self, urdf_file):
        import xml.etree.ElementTree as ET
        self.joints = []
        for j in ET.parse(urdf_file).getroot().iter("joint"):
            name = j.get("name")
            if name in [f"panda_joint{k}" for k in range(1, 8)]:
                o = j.find("origin")
                xyz = np.array([float(v) for v in o.get("xyz").split()])
                rpy = np.array([float(v) for v in o.get("rpy").split()])
                self.joints.append((int(name[-1]), xyz, _rpy(*rpy)))
        self.joints.sort(key=lambda t: t[0])
        assert len(self.joints) == 7

    def link_frames(self, q):
        T, out = np.eye(4), {0: np.eye(4)}
        for (k, xyz, Ro), qk in zip(self.joints, q):
            O = np.eye(4); O[:3, :3], O[:3, 3] = Ro, xyz
            Z = np.eye(4); Z[:3, :3] = _rpy(0, 0, qk)
            T = T @ O @ Z
            out[k] = T
        return out

    def get_3d_keypoints(self, joint_angle):
        q = np.asarray(joint_angle, float)
        assert q.shape[0] == 7
        F = self.link_frames(q)
        tr = lambda T, P: (T[:3, :3] @ P.T).T + T[:3, 3]
        return np.concatenate([tr(F[0], self.BASE), tr(F[6], self.L6), tr(F[7], self.L7)], axis=0)


class CtRNetX:
    def __init__(self, repo, weights, use_gpu):
        # the repo's mesh renderer imports pytorch3d at module load; we never render with it, so stub the module
        for name in ("pytorch3d", "pytorch3d.io", "pytorch3d.structures", "pytorch3d.renderer"):
            if name not in sys.modules:
                m = types.ModuleType(name)

                def _ga(attr):
                    if attr.startswith("__"):          # module internals (inspect reads __file__): really absent
                        raise AttributeError(attr)
                    return type(attr, (), {"__init__": lambda self, *a_, **k: None})   # placeholder class
                m.__getattr__ = _ga
                sys.modules[name] = m
        sys.path.insert(0, repo)
        import torch
        # the repo calls torch.load(path) bare: torch >= 2.6 defaults to weights_only=True (rejects this checkpoint)
        # and a GPU-saved checkpoint needs map_location on a CPU machine
        _load = torch.load
        torch.load = lambda f, *a_, **k: _load(f, *a_, **{"weights_only": False,
                                                          "map_location": None if use_gpu else "cpu", **k})
        import models.robot_arm as _ra
        _ra.PandaArm = PandaKeypoints                    # CtRNet.__init__ imports PandaArm from here at call time
        from models.CtRNet import CtRNet
        self.torch = torch
        self.args = argparse.Namespace(
            use_gpu=use_gpu, trained_on_multi_gpus=True, keypoint_seg_model_path=weights,
            urdf_file=os.path.join(repo, "urdfs", "Panda", "panda.urdf"), robot_name="Panda", n_kp=12,
            height=H, width=W, scale=1.0, fx=1.0, fy=1.0, px=0.0, py=0.0)
        self.CtRNet = CtRNet

    def estimate(self, d, K):
        import torchvision.transforms as T
        torch = self.torch
        a = self.args
        a.fx, a.fy, a.px, a.py = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        model = self.CtRNet(a)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        p2, p3 = [], []
        for img, q in zip(d["img"], d["q"]):
            x = tf(img.copy())
            if a.use_gpu:
                x = x.cuda()
            pts2, pts3, conf = model.inference_keypoints_dark(x, np.asarray(q, float)[:7])
            for j in range(pts2.shape[1]):
                if float(conf[0][j]) > CONF_THR:
                    p2.append(pts2[0][j])
                    p3.append(pts3[j])
        if len(p2) < 6:
            return None, len(p2)
        # RANSAC PnP instead of the repo's least-squares BPnP: without the CLIP visible-part filter, keypoints of
        # out-of-view links pass the confidence threshold and wreck a least-squares fit (pilot 2026-10-08)
        import cv2
        P2 = torch.stack(p2).detach().cpu().numpy().astype(np.float64).reshape(-1, 2)
        P3 = torch.stack(p3).detach().cpu().numpy().astype(np.float64).reshape(-1, 3)
        ok, rv, tv, inl = cv2.solvePnPRansac(P3, P2, K.astype(np.float64), None, reprojectionError=4.0,
                                             iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 6:
            return None, len(p2)
        idx = inl.ravel()
        ok, rv, tv = cv2.solvePnP(P3[idx], P2[idx], K.astype(np.float64), None, rv, tv, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        C, R = ctr_to_cam_pose(np.concatenate([rv.ravel(), tv.ravel()]))
        self.last_inliers = int(len(idx))
        return (C, R), len(p2)


# ------------------------------------------------------------------ SAM 2 render-and-segment
def farthest_points(mask, k, rng):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2))
    pts = np.stack([xs, ys], 1).astype(float)
    chosen = [pts[rng.integers(len(pts))]]
    d = np.linalg.norm(pts - chosen[0], axis=1)
    for _ in range(min(k, len(pts)) - 1):
        i = int(np.argmax(d))
        chosen.append(pts[i])
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(chosen)


def prompts_from_render(ren, rng, n_pos=5, n_neg=4):
    """Box around the render (10% pad), positives inside the eroded render, negatives in a ring 10 to 25 px outside."""
    from scipy.ndimage import binary_dilation, binary_erosion
    ys, xs = np.nonzero(ren)
    if len(xs) < 10:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = 0.1 * (x1 - x0 + 1), 0.1 * (y1 - y0 + 1)
    box = np.array([max(0, x0 - px), max(0, y0 - py), min(W - 1, x1 + px), min(H - 1, y1 + py)])
    m = ren.astype(bool)
    inner = binary_erosion(m, iterations=3)
    if inner.sum() < 5:
        inner = m
    ring = binary_dilation(m, iterations=25) & ~binary_dilation(m, iterations=10)
    pos, neg = farthest_points(inner, n_pos, rng), farthest_points(ring, n_neg, rng)
    pts = np.concatenate([pos, neg]) if len(neg) else pos
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return box, pts, lab


class SAM2:
    def __init__(self, model_id, device):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.pred = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    def segment(self, img, box, pts, lab):
        self.pred.set_image(img)
        masks, scores, _ = self.pred.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
        return masks.astype(np.uint8), np.asarray(scores, float)


def sam2_loop(rob, seg, d, K, C, R, rng, rounds=ROUNDS, log=None):
    fy = K[1, 1]
    hist = []
    for r in range(rounds):
        masks, used = [], []
        for i, (img, q) in enumerate(zip(d["img"], d["q"])):
            ren = rob.mask(q, C, R, fy)
            pr = prompts_from_render(ren, rng)
            if pr is None:
                continue
            ms, sc = seg.segment(img, *pr)
            ok = [j for j in range(len(ms)) if iou(ms[j], ren) >= 0.1]
            if not ok:
                continue
            j = max(ok, key=lambda j: sc[j])          # SAM 2's own confidence, not agreement with the render
            masks.append(ms[j]); used.append(i)
        if len(used) < 2:
            break
        score = lambda C_, R_: 1.0 - float(np.mean([soft_iou(masks[k], rob.mask(d["q"][i], C_, R_, fy)) for k, i in enumerate(used)]))
        sd = np.array([0.08, 0.08, 0.08, 0.25, 0.25, 0.25]) * (0.6 ** r)
        _, Cn, Rn = cem(score, C, R, sd, iters=8, rng=rng)
        step = pose_err(Cn, Rn, C, R)
        mask_iou = float(np.mean([iou(masks[k], rob.mask(d["q"][i], Cn, Rn, fy)) for k, i in enumerate(used)]))
        C, R = Cn, Rn
        hist.append(dict(round=r + 1, frames=len(used), step_cm=step[0], step_deg=step[1], render_mask_iou=mask_iou))
        if log:
            log(hist[-1])
        if step[0] < 0.5 and step[1] < 0.5:
            break
    return C, R, hist


# ------------------------------------------------------------------ run / report
def w6_init(w6_file):
    rows = {}
    if w6_file and os.path.exists(w6_file):
        for l in open(w6_file):
            if l.strip():
                r = json.loads(l)
                rows[r["file"]] = r
    return rows


def run(a):
    rng = np.random.default_rng(a.seed)
    rob = Franka(a.menagerie, W=W, H=H, hand=True)
    out = os.path.join(a.out_dir, "posesrc_results.jsonl")
    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.source if a.source == "ctrnetx" else f"sam2_from_{a.init}"
    done = set()
    if os.path.exists(out):
        done = {(r["file"], r["source"]) for r in map(json.loads, filter(str.strip, open(out)))}
    w6 = w6_init(a.w6)
    ctr_rows = {r["file"]: r for r in map(json.loads, filter(str.strip, open(out))) if r["source"] == "ctrnetx"} \
        if os.path.exists(out) else {}
    ctr = CtRNetX(a.ctrnetx_repo, a.ctrnetx_weights, a.gpu) if a.source == "ctrnetx" else None
    seg = SAM2(a.sam2_model, "cuda" if a.gpu else "cpu") if a.source == "sam2" else None
    files = sorted(glob.glob(os.path.join(a.cams, "*.npz")))[: a.limit or None]
    with open(out, "a") as f:
        for p in files:
            name = os.path.basename(p)
            if (name, tag) in done:
                continue
            d, K, T = load_cam(p)
            Cg, Rg, conv = reference_pose(rob, d, K, T)
            row = dict(file=name, source=tag, convention=conv, quality=float(d["quality"]))
            if a.source == "ctrnetx":
                est, npts = ctr.estimate(d, K)
                row["keypoints"] = npts
                if est is None:
                    row["failed"] = True
                    f.write(json.dumps(row) + "\n"); f.flush()
                    print(f"{name}: ctrnetx FAILED ({npts} keypoints)", flush=True)
                    continue
                Ce, Re = est
            else:
                if a.init == "w6":
                    r6 = w6.get(name)
                    if r6 is None:
                        print(f"{name}: no W6 row, skipped", flush=True)
                        continue
                    C0, R0 = np.array(r6["est_pos"]), euler_xyz_to_mat(np.array(r6["est_euler"]))
                else:
                    rc = ctr_rows.get(name)
                    if rc is None or rc.get("failed"):
                        print(f"{name}: no CtRNet-X pose, skipped", flush=True)
                        continue
                    C0, R0 = np.array(rc["est_pos"]), euler_xyz_to_mat(np.array(rc["est_euler"]))
                row["init_err_cm_deg"] = list(pose_err(C0, R0, Cg, Rg))
                Ce, Re, hist = sam2_loop(rob, seg, d, K, C0, R0, rng)
                row["rounds"] = hist
            e = pose_err(Ce, Re, Cg, Rg)
            obs = d["mask"] if "mask" in d.files else None
            row.update(pos_err_cm=e[0], rot_err_deg=e[1], est_pos=Ce.tolist(), est_euler=mat_to_euler_xyz(Re).tolist(),
                       old_mask_iou_est=float(np.mean([iou(rob.mask(q, Ce, Re, K[1, 1]), m) for q, m in zip(d["q"], obs)])) if obs is not None else None,
                       old_mask_iou_ref=float(np.mean([iou(rob.mask(q, Cg, Rg, K[1, 1]), m) for q, m in zip(d["q"], obs)])) if obs is not None else None)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{name}: {tag} error {e[0]:.1f} cm / {e[1]:.1f} deg" +
                  (f" (start {row['init_err_cm_deg'][0]:.1f} cm)" if "init_err_cm_deg" in row else ""), flush=True)


def summarize(pe, re_):
    pe, re_ = np.asarray(pe), np.asarray(re_)
    return (f"n {len(pe)} | median {np.median(pe):.1f} cm / {np.median(re_):.1f} deg | within 5 cm & 5 deg "
            f"{np.mean((pe <= 5) & (re_ <= 5)):.2f} | within 10 cm & 10 deg {np.mean((pe <= 10) & (re_ <= 10)):.2f}")


def report(a):
    from scipy.stats import wilcoxon
    rows = [json.loads(l) for l in open(os.path.join(a.out_dir, "posesrc_results.jsonl")) if l.strip()]
    base = {}
    for name, path in (("w6_v6_zero_shot", a.w6), ("w6b_vlm_masks", a.w6_vlm)):
        if path and os.path.exists(path):
            base[name] = {r["file"]: r for r in map(json.loads, filter(str.strip, open(path)))}
    by = {}
    for r in rows:
        by.setdefault(r["source"], {})[r["file"]] = r
    by.update(base)
    for src, rs in by.items():
        ok = [r for r in rs.values() if not r.get("failed")]
        print(f"\n== {src}: {len(ok)} cameras ({len(rs) - len(ok)} failed)")
        if not ok:
            continue
        print("  all:      " + summarize([r["pos_err_cm"] for r in ok], [r["rot_err_deg"] for r in ok]))
        rel = [r for r in ok if np.isfinite(r.get("quality", np.nan)) and r["quality"] < 20]
        if rel:
            print("  reliable: " + summarize([r["pos_err_cm"] for r in rel], [r["rot_err_deg"] for r in rel]))
    for src in by:
        if src in base:
            continue
        for bname, b in base.items():
            common = [k for k, r in by[src].items() if not r.get("failed") and k in b
                      and np.isfinite(b[k].get("quality", np.nan)) and b[k]["quality"] < 20]
            if len(common) >= 5:
                x = np.array([by[src][k]["pos_err_cm"] for k in common]); y = np.array([b[k]["pos_err_cm"] for k in common])
                p = wilcoxon(x, y, alternative="less").pvalue
                print(f"  {src} vs {bname}, reliable, paired position error: median {np.median(x):.1f} vs "
                      f"{np.median(y):.1f} cm, better on {int((x < y).sum())} of {len(common)}, one-sided Wilcoxon p {p:.3g}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--source", choices=["ctrnetx", "sam2"], required=True)
    r.add_argument("--init", choices=["w6", "ctrnetx"], default="w6")
    r.add_argument("--cams", required=True)
    r.add_argument("--out_dir", required=True)
    r.add_argument("--w6", default=None, help="W6 v6 selfcal_results.jsonl (start poses for --init w6)")
    r.add_argument("--menagerie", default="/content/mujoco_menagerie")
    r.add_argument("--ctrnetx_repo", default="/content/CtRNet-X")
    r.add_argument("--ctrnetx_weights", default="/content/ctrnetx_weights/net_best.pth")
    r.add_argument("--sam2_model", default="facebook/sam2.1-hiera-large")
    r.add_argument("--gpu", type=int, default=1)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    q = sp.add_parser("report")
    q.add_argument("--out_dir", required=True)
    q.add_argument("--w6", default=None)
    q.add_argument("--w6_vlm", default=None)
    a = ap.parse_args()
    {"run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
