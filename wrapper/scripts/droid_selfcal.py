"""Real-image self-calibration on DROID: estimate an exterior camera's pose from the robot's own body in REAL frames,
and score it against DROID's post-hoc calibrated extrinsics (KarlP/droid, cam2base_extrinsic_superset.json).

  export     droid_100 RLDS (Colab system python, TF + tfds): episodes that have calibrated extrinsics; per camera, K
             frames spread over the episode with joint angles, the calibrated cam->base pose and the intrinsics.
  segment    text-prompted robot masks on the real frames: OWLv2 box ("a robot arm") -> SAM mask (transformers).
  calibrate  render the Franka (mujoco_menagerie panda_nohand; DROID's Robotiq gripper is not modelled, so the score is
             one-sided: rendered arm boundary -> observed boundary, plus rendered pixels outside the observed mask),
             grid over viewpoints around the workspace + CEM over the 6-DoF pose using all K frames (the demo's own arm
             motion is the sweep). First checks the extrinsics convention on the calibrated poses (render IoU).
  report     position / rotation error against the calibrated extrinsics, success rates at 5 cm / 5 deg and 10 cm / 10 deg,
             and error vs the dataset's own calibration quality metric.
The calibrated extrinsics are themselves estimates (IoU-scored), so errors are measured against a reference, not truth.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HF_REPO = "KarlP/droid"
CALIB_FILES = ("cam2base_extrinsic_superset", "camera_serials", "intrinsics", "episode_id_to_path")
OUTSIDE_W = 3.0
TARGET = np.array([0.45, 0.0, 0.15])          # workspace point the exterior cameras look at, base frame (search prior)


# ------------------------------------------------------------------ geometry
def euler_xyz_to_mat(e):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", e).as_matrix()


def mat_to_euler_xyz(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_euler("xyz")


def pose_err(C1, R1, C2, R2):
    from scipy.spatial.transform import Rotation
    return float(np.linalg.norm(C1 - C2) * 100), float(np.rad2deg(Rotation.from_matrix(R1 @ R2.T).magnitude()))


def look_at(C, target, up=np.array([0.0, 0.0, 1.0])):
    """OpenCV camera (x right, y down, z forward) at C looking at target: columns of R are camera axes in base."""
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def pan_tilt(R_cv, dx, dy, f):
    """Rotate an OpenCV camera (columns of R = camera axes in base) about its own y then x axis so that the scene moves
    by (dx, dy) pixels in the image (u right, v down). Position unchanged: orientation is decoupled from the grid."""
    th, ph = -np.arctan2(dx, f), np.arctan2(dy, f)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    return R_cv @ Ry @ Rx


def best_shift(obs_masks, ren_masks, sigma=2.0):
    """The image translation (dx, dy) that best aligns the rendered masks with the observed ones, shared across the
    frames (one camera rotation moves every frame alike): peak of the summed FFT cross-correlation of blurred masks."""
    import cv2
    acc = None
    for o, r in zip(obs_masks, ren_masks):
        A = np.fft.rfft2(cv2.GaussianBlur(o.astype(np.float32), (0, 0), sigma))
        B = np.fft.rfft2(cv2.GaussianBlur(r.astype(np.float32), (0, 0), sigma))
        c = np.fft.irfft2(A * np.conj(B), s=o.shape)
        acc = c if acc is None else acc + c
    H, W = acc.shape
    iy, ix = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return (ix if ix <= W // 2 else ix - W), (iy if iy <= H // 2 else iy - H)


def cv_to_mj(R_cv):
    """MuJoCo cameras look along -z with +y up: flip the OpenCV y and z axes."""
    return R_cv @ np.diag([1.0, -1.0, -1.0])


def mat_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


# ------------------------------------------------------------------ export (TF / tfds)
def rel_path_key(file_path):
    m = re.search(r"([^/]+/(?:success|failure)/\d{4}-\d{2}-\d{2}/[^/]+)/trajectory", file_path)
    return m.group(1) if m else None


def export(a):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from huggingface_hub import hf_hub_download
    tf.config.set_visible_devices([], "GPU")
    cal = {k: json.load(open(hf_hub_download(HF_REPO, f"{k}.json"))) for k in CALIB_FILES}
    # several episode ids can share one path in DROID's mapping; only unambiguous paths are used, never guessed
    by_path = {}
    for k, v in cal["episode_id_to_path"].items():
        by_path.setdefault(v.strip("/"), []).append(k)
    path_to_id = {p: ks[0] for p, ks in by_path.items() if len(ks) == 1}
    print(f"episode_id_to_path: {len(cal['episode_id_to_path'])} ids, {len(by_path)} paths, "
          f"{len(path_to_id)} unambiguous", flush=True)
    # fallback on (lab, timestamp folder) only where that pair is unique; ambiguous pairs are dropped, never guessed
    pairs = {}
    for k, v in cal["episode_id_to_path"].items():
        parts = v.strip("/").split("/")
        if len(parts) == 4:
            pairs.setdefault((parts[0], parts[3]), []).append(k)
    tail_to_id = {p: ks[0] for p, ks in pairs.items() if len(ks) == 1}
    hits = tf.io.gfile.glob(a.rlds.rstrip("/") + "/*/dataset_info.json") or tf.io.gfile.glob(a.rlds.rstrip("/") + "/dataset_info.json")
    assert hits, f"no RLDS dataset under {a.rlds}"
    ds = tfds.builder_from_directory(os.path.dirname(hits[0])).as_dataset(split="train")
    os.makedirs(a.out_dir, exist_ok=True)
    seen = matched = saved = 0
    for ep in ds:
        seen += 1
        fp = ep["episode_metadata"]["file_path"].numpy().decode()
        rel = rel_path_key(fp)
        if seen <= 3:
            print("example file_path:", fp, "->", rel, flush=True)
        eid = path_to_id.get(rel) if rel else None
        if eid is None and rel:
            parts = rel.split("/")
            eid = tail_to_id.get((parts[0], parts[3]))
        if eid is None or eid not in cal["cam2base_extrinsic_superset"] or eid not in cal["camera_serials"]:
            continue
        matched += 1
        steps = list(ep["steps"].as_numpy_iterator())
        idx = np.linspace(0, len(steps) - 1, a.frames).round().astype(int)
        for cam_key, img_key in (("ext1_cam_serial", "exterior_image_1_left"), ("ext2_cam_serial", "exterior_image_2_left")):
            serial = cal["camera_serials"][eid][cam_key]
            ext = cal["cam2base_extrinsic_superset"][eid].get(serial)
            intr = cal["intrinsics"].get(eid, {}).get(serial)
            if ext is None or intr is None:
                continue
            # the live file stores {cameraMatrix: [fx, cx, fy, cy], width, height}; the README documents a bare 4-list
            if isinstance(intr, dict):
                cm, wh = intr["cameraMatrix"], [intr["width"], intr["height"]]
            else:
                cm, wh = intr, [2 * intr[1], 2 * intr[3]]                 # principal point ~ image centre
            np.savez_compressed(
                os.path.join(a.out_dir, f"{eid.replace('+', '_')}_{serial}.npz"),
                img=np.stack([steps[i]["observation"][img_key] for i in idx]),
                q=np.stack([steps[i]["observation"]["joint_position"] for i in idx]).astype(np.float64),
                ext=np.asarray(ext, float), intr=np.asarray(cm, float),
                intr_wh=np.array(wh, float), eid=np.array(eid), serial=np.array(serial),
                quality=np.array(float(cal["cam2base_extrinsic_superset"][eid].get(f"{serial}_quality_metric", np.nan))))
            saved += 1
        if saved >= a.max_cameras:
            break
    print(f"episodes read {seen}, with calibrated extrinsics {matched}, cameras saved {saved} -> {a.out_dir}")


# ------------------------------------------------------------------ segment (transformers)
def segment(a):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owl_p = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    sam_p = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
    queries = [["a robot arm", "a black robotic arm", "an industrial robot"]]
    for p in sorted(glob.glob(os.path.join(a.out_dir, "*.npz"))):
        d = dict(np.load(p, allow_pickle=True))
        if "mask" in d:
            continue
        masks = []
        for img in d["img"]:
            big = Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.BICUBIC)
            with torch.no_grad():
                inp = owl_p(text=queries, images=big, return_tensors="pt").to(dev)
                out = owl(**inp)
                # OWLv2 pads the image to a square; box scaling by the padded side is the same in every transformers
                # version, unlike target_sizes (older releases squash the y axis on non-square frames)
                # transformers renamed the OWLv2 post-processor; both return boxes normalised to the padded square
                pp = getattr(owl_p, "post_process_object_detection", None) or owl_p.post_process_grounded_object_detection
                res = pp(out, threshold=0.05, target_sizes=None)[0]
            if len(res["scores"]) == 0:
                masks.append(np.zeros(img.shape[:2], np.uint8))
                continue
            side = float(max(big.size))
            box = (np.asarray(res["boxes"][int(torch.argmax(res["scores"]))].tolist()) * side)
            box = np.clip(box, 0, [big.size[0], big.size[1], big.size[0], big.size[1]]).tolist()
            with torch.no_grad():
                si = sam_p(big, input_boxes=[[box]], return_tensors="pt").to(dev)
                so = sam(**si)
                m = sam_p.image_processor.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu(),
                                                            si["reshaped_input_sizes"].cpu())[0][0]
            best = m[int(torch.argmax(so.iou_scores[0, 0]))].numpy().astype(np.uint8)
            m_full = np.asarray(Image.fromarray(best * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 127
            masks.append(clean_mask(m_full) > 0)                # largest component, holes filled: no speckle edges
        d["mask"] = np.stack(masks).astype(np.uint8)
        np.savez_compressed(p, **d)
        print(f"{os.path.basename(p)}: robot pixel fraction {d['mask'].mean():.3f}", flush=True)


# ------------------------------------------------------------------ calibrate (mujoco)
class Franka:
    def __init__(self, menagerie, W=160, H=90, hand=True):
        """hand=True renders menagerie's panda.xml (Franka hand, fingers open) as a bulk proxy for DROID's Robotiq
        gripper: the observed silhouette always contains a gripper, and a hand-less render is systematically smaller,
        which biases the size term toward a farther camera (2026-09-18: estimates farther than the reference on 44 of
        53 cameras, 72% of the error along the viewing ray)."""
        import mujoco
        src = os.path.join(menagerie, "franka_emika_panda", "panda.xml" if hand else "panda_nohand.xml")
        xml = open(src).read()
        xml = xml.replace("<worldbody>", '<worldbody>\n    <camera name="cal" pos="1 0 1" xyaxes="0 -1 0 0 0 1" fovy="60"/>', 1)
        path = os.path.join(os.path.dirname(src), "_panda_cal.xml" if hand else "_panda_nohand_cal.xml")
        open(path, "w").write(xml)
        self.mj = mujoco
        self.hand = hand
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.cid = self.model.camera("cal").id
        self.W, self.H = W, H
        self.r = mujoco.Renderer(self.model, height=H, width=W)
        self.r.enable_segmentation_rendering()

    def mask(self, q, C, R_cv, fy_px):
        m = self.model
        m.cam_pos[self.cid] = C
        m.cam_quat[self.cid] = mat_to_quat_wxyz(cv_to_mj(R_cv))
        m.cam_fovy[self.cid] = float(np.rad2deg(2 * np.arctan(0.5 * self.H / fy_px)))
        self.data.qpos[:7] = q
        if self.hand and self.model.nq >= 9:
            self.data.qpos[7:9] = 0.04                     # fingers open
        self.mj.mj_forward(m, self.data)
        self.r.update_scene(self.data, camera=self.cid)
        seg = self.r.render()
        return (seg[..., 0] >= 0).astype(np.uint8)


def clean_mask(mask, min_px=30, keep_frac=0.0):
    """Largest connected component with holes filled (uint8 0/1); tiny masks returned as is. keep_frac > 0 keeps every
    component at least that share of the largest: an arm cut by the frame border, or a gripper seen apart from the arm,
    is several components (the same lesson as the LIBERO learned masks; largest-only cost 0.15 recall on a DROID camera)."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < min_px:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        area = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + (np.where(area >= keep_frac * area.max())[0] if keep_frac > 0 else np.array([int(np.argmax(area))]))
        m = np.isin(lab, keep).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((1 - m).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    for k in range(1, n):
        if k not in border:
            m[lab == k] = 1
    return m


def motion_masks(imgs, sam_masks, thr=28.0, grow=25):
    """Complete the segmenter's masks with the robot's own motion: across the stored frames the room is static and
    the arm is not, so pixels that differ from the per-pixel median background belong to the arm (links, gripper,
    cables, whatever colour), restricted to the neighbourhood of the segmenter's masks so people and moved objects
    elsewhere in the frame do not enter. Union with the segmenter mask (which covers the static base), then cleaned.
    Self-measurement: the robot knows it moved. Returns uint8 0/1 masks at the frame resolution."""
    import cv2
    imgs = np.asarray(imgs).astype(np.float32)
    sam = np.asarray(sam_masks) > 0
    # background = per-pixel median over the frames in which the segmenter saw no robot there (a plain median turns
    # into the arm wherever the arm parks for half the frames, and then the arm's absence reads as motion)
    free = np.where(sam[..., None], np.nan, imgs)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(free, axis=0)
    bg = np.where(np.isnan(bg), np.median(imgs, axis=0), bg)
    region = cv2.dilate(np.any(np.asarray(sam_masks) > 0, axis=0).astype(np.uint8), np.ones((grow, grow), np.uint8))
    out = []
    for im, sm in zip(imgs, sam_masks):
        diff = np.abs(im - bg).max(-1)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        mv = (diff > thr).astype(np.uint8) & region
        out.append(clean_mask((mv | (np.asarray(sm) > 0).astype(np.uint8))))
    return out


def soft_iou(a_, b_, sigma=2.0):
    """IoU of Gaussian-blurred masks: indifferent to boundary detail, so a ragged real mask and a clean render can
    still agree; the coarse search and the CEM both optimise 1 - soft_iou."""
    import cv2
    A = cv2.GaussianBlur(a_.astype(np.float32), (0, 0), sigma)
    B = cv2.GaussianBlur(b_.astype(np.float32), (0, 0), sigma)
    u = np.maximum(A, B).sum()
    return float(np.minimum(A, B).sum() / u) if u > 0 else 0.0


def one_sided(obs_mask, ren_mask):
    import cv2
    if ren_mask.sum() < 5:
        return 1e3
    edge = lambda m: m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8)))
    eo, er = edge(obs_mask), edge(ren_mask)
    if eo.sum() == 0:
        return 1e3
    dt = cv2.distanceTransform((1 - eo).astype(np.uint8), cv2.DIST_L2, 3)
    outside = float((ren_mask & (1 - cv2.dilate(obs_mask, np.ones((5, 5), np.uint8)))).sum() / ren_mask.sum())
    # size term, as in the simulator calibration (calibrate_v2 area_w): without it the edge chamfer is minimised by a
    # render that explains part of the boundary and never sticks out, i.e. the estimate drifts to a smaller, wrong
    # silhouette (seen on DROID 2026-09-17: estimate score below the reference with lower IoU)
    size = abs(float(np.log((ren_mask.sum() + 1.0) / (obs_mask.sum() + 1.0))))
    # OUTSIDE_W was 20: with an incomplete real-image mask the TRUE pose renders the missing part of the arm and is
    # punished for it (DROID v2: reference scores 130 to 311 at IoU 0.46), so this is one clue among three, not a veto
    return float(dt[er > 0].mean()) + OUTSIDE_W * outside + 4.0 * size


def iou(a_, b_):
    u = (a_ | b_).sum()
    return float((a_ & b_).sum() / u) if u else 0.0


def calibrate(a):
    import cv2
    rob = Franka(a.menagerie, hand=bool(a.hand))
    out_path = os.path.join(a.out_dir, "selfcal_results.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["file"] for l in open(out_path) if l.strip()}
    files = [p for p in sorted(glob.glob(os.path.join(a.out_dir, "*.npz"))) if os.path.basename(p) not in done]
    valid = [float(np.load(p, allow_pickle=True)["intr"][2]) for p in sorted(glob.glob(os.path.join(a.out_dir, "*.npz")))]
    valid = [v for v in valid if np.isfinite(v) and v > 0]
    median_fy = float(np.median(valid)) if valid else 530.0
    print(f"intrinsics: {len(valid)} cameras with data, median fy {median_fy:.1f} px at 1280x720 (used where DROID has none)", flush=True)
    rng = np.random.default_rng(0)
    convention = a.convention
    with open(out_path, "a") as f:
        for p in files:
            d = np.load(p, allow_pickle=True)
            if "mask" not in d:
                continue
            full_masks = motion_masks(d["img"], d["mask"]) if a.motion else [clean_mask(m, keep_frac=a.keep_frac) for m in d["mask"]]
            obs = [cv2.resize(m, (rob.W, rob.H), interpolation=cv2.INTER_NEAREST).astype(np.uint8) for m in full_masks]
            sam_only = [cv2.resize(clean_mask(m), (rob.W, rob.H), interpolation=cv2.INTER_NEAREST) for m in d["mask"]]
            keep = [i for i, m in enumerate(obs) if m.mean() > 0.01]
            if len(keep) < 2:
                continue
            fx, cx, fy, cy = d["intr"]
            imputed = False
            if not (np.isfinite(fx) and fx > 0):           # DROID ships all-zero intrinsics for some cameras
                fx = fy = median_fy; imputed = True        # same camera model everywhere: the median of the valid ones
            elif not np.isfinite(fy) or fy <= 0:
                fy = fx
            wh_y = float(d["intr_wh"][1])
            if not np.isfinite(wh_y) or wh_y <= 0:          # no intrinsics record: DROID calibrates at 1280 x 720
                wh_y = 720.0
            fy_px = fy * rob.H / wh_y
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = euler_xyz_to_mat(d["ext"][3:6]), d["ext"][:3]
            variants = {"cam2base": T, "base2cam": np.linalg.inv(T)}
            gt_iou = {k: float(np.mean([iou(rob.mask(d["q"][i], V[:3, 3], V[:3, :3], fy_px), obs[i]) for i in keep]))
                      for k, V in variants.items()}
            if convention == "auto":
                convention = max(gt_iou, key=gt_iou.get)
                print(f"extrinsics convention check on the first camera: render IoU {gt_iou} -> using {convention}", flush=True)
            Tg = variants[convention]
            Cg, Rg = Tg[:3, 3], Tg[:3, :3]
            chamfer = lambda C, R: float(np.mean([one_sided(obs[i], rob.mask(d["q"][i], C, R, fy_px)) for i in keep]))
            score = lambda C, R: 1.0 - float(np.mean([soft_iou(obs[i], rob.mask(d["q"][i], C, R, fy_px)) for i in keep]))
            # mask completeness: share of the reference render that the mask covers (1 = nothing of the arm is missed)
            recall_sam = float(np.mean([(rob.mask(d["q"][i], Cg, Rg, fy_px) & sam_only[i]).sum() / max(rob.mask(d["q"][i], Cg, Rg, fy_px).sum(), 1) for i in keep]))
            prec_full = float(np.mean([(rob.mask(d["q"][i], Cg, Rg, fy_px) & obs[i]).sum() / max(obs[i].sum(), 1) for i in keep]))
            recall_full = float(np.mean([(rob.mask(d["q"][i], Cg, Rg, fy_px) & obs[i]).sum() / max(rob.mask(d["q"][i], Cg, Rg, fy_px).sum(), 1) for i in keep]))

            def refine(Cb, Rb, iters=10):
                mu, sd = np.zeros(6), np.array([0.10, 0.10, 0.10, 0.35, 0.35, 0.35])
                bestl = (score(Cb, Rb), Cb, Rb)
                for _ in range(iters):
                    X = mu + sd * rng.standard_normal((32, 6))
                    X[0] = mu
                    vals = []
                    for x in X:
                        C, R = Cb + x[:3], euler_xyz_to_mat(x[3:]) @ Rb
                        v = score(C, R)
                        vals.append(v)
                        if v < bestl[0]:
                            bestl = (v, C, R)
                    el_ = X[np.argsort(vals)[:6]]
                    mu, sd = el_.mean(0), el_.std(0) + 1e-3
                return bestl
            cands = []
            # DROID's exterior cameras sit 0.44 to 0.86 m from the workspace point (median 0.63) at 2 to 45 deg
            # elevation (53 reference poses, 2026-09-18); the old grid started at 0.7 m and 15 deg, outside 36 of 53
            for rad in (0.4, 0.55, 0.7, 0.9, 1.15):
                for az in range(0, 360, 20):
                    for el in (0, 10, 20, 30, 45):
                        e, zr = np.deg2rad(el), np.deg2rad(az)
                        C = TARGET + rad * np.array([np.cos(e) * np.cos(zr), np.cos(e) * np.sin(zr), np.sin(e)])
                        R = look_at(C, TARGET)
                        # real mounts point up to 50+ deg away from the look-at prior, so the robot sits off-centre and
                        # a centred render scores ~0 even at the right POSITION (v5: from-reference refinement held
                        # within 5 cm while the search landed 50 cm / 85 deg away). Align the render to the observed
                        # masks by an image shift, turn that shift into a pan/tilt, score the re-aimed camera.
                        rens = [rob.mask(d["q"][i], C, R, fy_px) for i in keep]
                        if sum(int(m.sum()) for m in rens) > 0:
                            dx, dy = best_shift([obs[i] for i in keep], rens)
                            R = pan_tilt(R, dx, dy, fy_px)
                        cands.append((score(C, R), C, R))
            cands.sort(key=lambda x: x[0])
            best = cands[0]
            seeds = []                      # distinct basins: skip a candidate within 15 cm and 20 deg of a chosen seed
            for cnd in cands:
                if all(pose_err(cnd[1], cnd[2], s_[1], s_[2])[0] > 15 or pose_err(cnd[1], cnd[2], s_[1], s_[2])[1] > 20
                       for s_ in seeds):
                    seeds.append(cnd)
                if len(seeds) == 5:
                    break
            for s0, Cb, Rb in seeds:
                r_ = refine(Cb, Rb, iters=8)
                if r_[0] < best[0]:
                    best = r_
            _, Ce, Re = best
            err = pose_err(Ce, Re, Cg, Rg)
            # diagnostic: refine FROM the reference. If it stays put and scores no worse than the estimate, the
            # estimate sits in another basin (search); if it drifts away, the objective itself prefers a wrong pose (mask)
            ref_ref = refine(Cg, Rg, iters=6)
            ref_err = pose_err(ref_ref[1], ref_ref[2], Cg, Rg)
            z_view = Re[:, 2]
            dvec = Ce - Cg
            along = float(abs(dvec @ z_view)) * 100
            lateral = float(np.linalg.norm(dvec - (dvec @ z_view) * z_view)) * 100
            row = dict(file=os.path.basename(p), eid=str(d["eid"]), serial=str(d["serial"]), convention=convention,
                       intr_imputed=imputed,
                       quality=float(d["quality"]), frames=len(keep), gt_render_iou=gt_iou[convention],
                       est_render_iou=float(np.mean([iou(rob.mask(d["q"][i], Ce, Re, fy_px), obs[i]) for i in keep])),
                       score_gt=score(Cg, Rg), score_est=best[0], chamfer_gt=chamfer(Cg, Rg), chamfer_est=chamfer(Ce, Re),
                       search="v6", pos_err_cm=err[0], rot_err_deg=err[1], err_along_cm=along, err_lateral_cm=lateral,
                       mask_recall_sam=recall_sam, mask_recall_full=recall_full, mask_precision_full=prec_full,
                       ref_refined_err_cm=ref_err[0], ref_refined_err_deg=ref_err[1], score_ref_refined=ref_ref[0],
                       est_pos=Ce.tolist(), est_euler=mat_to_euler_xyz(Re).tolist(), gt_pos=Cg.tolist())
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{row['file']}: error {err[0]:.1f} cm / {err[1]:.1f} deg (along {along:.0f}, lateral {lateral:.0f}) | "
                  f"IoU gt {row['gt_render_iou']:.2f} est {row['est_render_iou']:.2f} | mask recall sam {recall_sam:.2f} "
                  f"full {recall_full:.2f} precision {prec_full:.2f} | from-reference refine stays within {ref_err[0]:.1f} cm", flush=True)


def report(a):
    rows = [json.loads(l) for l in open(os.path.join(a.out_dir, "selfcal_results.jsonl")) if l.strip()]
    pe, re_ = np.array([r["pos_err_cm"] for r in rows]), np.array([r["rot_err_deg"] for r in rows])
    print(f"cameras {len(rows)} | median error {np.median(pe):.1f} cm / {np.median(re_):.1f} deg")
    for c, g in ((5, 5), (10, 10), (20, 15)):
        print(f"  within {c} cm and {g} deg: {np.mean((pe <= c) & (re_ <= g)):.2f}")
    gi = np.array([r["gt_render_iou"] for r in rows])
    print(f"render IoU with the reference extrinsics (convention sanity): median {np.median(gi):.2f}")
    good = gi >= np.median(gi)
    print(f"  on cameras where the reference itself renders well (IoU >= median): median error "
          f"{np.median(pe[good]):.1f} cm / {np.median(re_[good]):.1f} deg")
    worse = np.mean([r["score_est"] <= r["score_gt"] for r in rows])
    print(f"estimate explains the observed silhouette at least as well as the reference in {worse:.0%} of cameras")
    if "err_along_cm" in rows[0]:
        al = np.array([r["err_along_cm"] for r in rows]); la = np.array([r["err_lateral_cm"] for r in rows])
        rr = np.array([r["ref_refined_err_cm"] for r in rows])
        rs = np.array([r["mask_recall_sam"] for r in rows]); rf = np.array([r["mask_recall_full"] for r in rows])
        print(f"error split: along the viewing ray {np.median(al):.0f} cm, lateral {np.median(la):.0f} cm (medians)")
        print(f"mask completeness (share of the reference render covered): segmenter {np.median(rs):.2f} -> with motion {np.median(rf):.2f}")
        print(f"refining from the reference stays within 5 cm on {np.mean(rr <= 5):.0%} of cameras (objective consistent with "
              f"the reference there); drifts beyond 20 cm on {np.mean(rr > 20):.0%} (objective prefers another pose: mask or reference)")
    # DROID publishes a per-camera calibration quality metric; values in the tens of thousands mark a reference that
    # DROID itself flags as unreliable. Report the trustworthy references separately.
    q = np.array([r["quality"] for r in rows])
    imp = np.array([bool(r.get("intr_imputed", False)) for r in rows])
    ok = np.isfinite(q) & (q < 20.0) & ~imp
    if imp.any():
        print(f"intrinsics imputed on {int(imp.sum())} cameras (DROID has none): median error {np.median(pe[imp]):.1f} cm, "
              f"reported separately")
    if ok.any():
        print(f"reference flagged reliable by DROID (quality < 20): {int(ok.sum())} cameras | median error "
              f"{np.median(pe[ok]):.1f} cm / {np.median(re_[ok]):.1f} deg | within 20 cm and 15 deg: "
              f"{np.mean((pe[ok] <= 20) & (re_[ok] <= 15)):.2f} | reference IoU median {np.median(gi[ok]):.2f}")
        bad = ~ok
        if bad.any():
            print(f"reference flagged unreliable ({int(bad.sum())} cameras): median error {np.median(pe[bad]):.1f} cm; "
                  f"errors against these are not evidence either way")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "segment", "calibrate", "report"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rlds", default="gs://gresearch/robotics/droid_100")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max_cameras", type=int, default=60)
    ap.add_argument("--motion", type=int, default=1, help="complete the segmenter masks with the robot's own motion (v5)")
    ap.add_argument("--keep_frac", type=float, default=0.0,
                    help="with --motion 0: keep every mask component at least this share of the largest (0 = largest only, v6)")
    ap.add_argument("--hand", type=int, default=1, help="render the Franka hand as a gripper proxy (v5)")
    ap.add_argument("--menagerie", default="/content/mujoco_menagerie")
    ap.add_argument("--convention", default="auto", choices=["auto", "cam2base", "base2cam"])
    a = ap.parse_args()
    dict(export=export, segment=segment, calibrate=calibrate, report=report)[a.mode](a)


if __name__ == "__main__":
    main()
