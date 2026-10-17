"""Deployment-time canonicalization for a FROZEN policy in LIBERO / LIBERO-Plus.

Two channels, both using only what a real deployment knows (the robot's nominal start configuration, the
training camera's nominal pose and intrinsics, the robot's own geometry, and the RGB-D image):

  state channel   measure how far the arm's joints are from the nominal start; if off, drive the end effector
                  back to the nominal start pose with ordinary controller actions (no teleporting).
  camera channel  a short arm sweep, then render-and-compare on the robot's own silhouette to estimate the
                  actual camera pose; every step, reproject the RGB-D image into the nominal camera and give
                  the frozen policy that canonical view.

Sim-only stand-ins, disclosed: the observed robot mask comes from the renderer's segmentation (a perfect arm
segmenter), and depth from the renderer (an RGB-D camera). Upper-bound arms use ground truth explicitly.
"""
import os
import re

import cv2
import numpy as np

ROBOT_PREFIXES = ("robot0", "gripper0", "mount0")
TABLE_Z = 0.8                      # workspace plane used only to pick the orbit pivot on the nominal optical axis
SWEEP = [np.zeros(3), np.array([0.0, 0.12, -0.05]), np.array([0.08, -0.12, 0.0])]
# learned sensing has no metric depth, so the silhouettes alone must fix the camera's distance: more of the robot's own
# KNOWN motion is the metric baseline. Measured against the true camera on six hard tasks (2026-09-19, local): median
# calibration error 21.1 cm with 3 poses, 7.9 cm with 7, 7.1 cm with 7 and like-with-like mask scoring.
SWEEP_LEARNED = SWEEP + [np.array([-0.06, 0.0, 0.10]), np.array([0.10, 0.10, 0.08]), np.array([0.0, -0.15, -0.06]),
                         np.array([-0.05, 0.15, 0.06])]


def scene_of(name):
    """Scene prefix of a LIBERO task name, e.g. KITCHEN_SCENE3. Each scene has its own agentview camera."""
    m = re.match(r"([A-Z]+(?:_[A-Z]+)*_SCENE\d+)_", name)
    if not m:                   # libero_spatial / libero_object / libero_goal: one scene per suite, no prefix in the name;
        return "SUITE_SCENE"    # read_nominals still REQUIRES the unperturbed tasks of a scene to agree on the camera
    return m.group(1)


def seed_flip(suite, make_env, nominals, cam="agentview"):
    """Detect the raw-render -> observation flip ONCE on a clean, unperturbed frame (a default-view language task) and
    cache it. Sensor-noise tasks corrupt the image so heavily (mean abs error ~76 gray levels even downsampled) that
    detection on their frames cannot work."""
    if cam in _FLIP_CACHE:
        return
    from plus_common import DEFAULT_VIEW, categorize
    for i in range(suite.n_tasks):
        n = suite.get_task(i).name
        v = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", n)
        s = re.search(r"_initstate_(\d+)", n)
        if categorize(n) == "language" and not (v and v.groups() != DEFAULT_VIEW) and not (s and s.group(1) != "0") \
                and scene_of(n) in nominals:
            break
    else:
        return
    env = make_env(i)
    try:
        obs = env.reset()
        Canonicalizer(env, nominals[scene_of(n)], cam=cam)._obs_convention(obs)
        print(f"image orientation for {cam}: {_FLIP_CACHE.get(cam)} (detected on clean task {i})", flush=True)
    finally:
        env.close()


def read_nominals(suite, make_env, cache=None, per_scene=2):
    """Nominal agentview camera and robot start PER SCENE. LIBERO-10 spans several rooms with different cameras,
    so a single global nominal is wrong for most tasks. Read from unperturbed tasks of each scene (language and
    sensor-noise variants with the default view and robot start) and require them to agree."""
    import json
    import os
    from plus_common import DEFAULT_VIEW, categorize
    if cache and os.path.exists(cache) and os.path.exists(cache + ".ref.npz"):
        out = json.load(open(cache))
        refs = np.load(cache + ".ref.npz")
        for s in out:
            if s in refs.files:
                out[s]["ref_image"] = refs[s]
        seed_flip(suite, make_env, out)
        if any("robot_shading_d" not in d or "table_z" not in d for d in out.values()):   # older cache: upgrade in place
            _add_shading_reference(suite, make_env, out, cache)
        return out
    src = {}
    for i in range(suite.n_tasks):
        n = suite.get_task(i).name
        v = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", n)
        s = re.search(r"_initstate_(\d+)", n)
        if categorize(n) != "language" or (v and v.groups() != DEFAULT_VIEW) or (s and s.group(1) != "0"):
            continue
        ids = src.setdefault(scene_of(n), [])
        if len(ids) < per_scene:
            ids.append(i)
    out = {}
    for scene, ids in sorted(src.items()):
        got = []
        for i in ids:
            env = make_env(i)
            obs = env.reset()
            m = env.env.sim.model
            cid = m.camera_name2id("agentview")
            nom = dict(cam_pos=[float(x) for x in m.cam_pos[cid]], cam_quat_wxyz=[float(x) for x in m.cam_quat[cid]],
                       q_nom=[float(x) for x in env.env.robots[0].init_qpos])
            # photometric and noise references: the robot's own pixels and the image noise floor in the training setup
            mask = Canonicalizer(env, nom).obs_robot_mask(obs) > 0
            px = obs["agentview_image"][mask].reshape(-1, 3).astype(np.float64)
            nom.update(robot_rgb_mean=px.mean(0).tolist(), robot_rgb_std=px.std(0).tolist(),
                       noise={k: noise_level(obs[k]) for k in IMG_KEYS if k in obs})
            # training-view reference of the scene (a nominal-camera frame of an unperturbed task): fills reprojection
            # holes with what the training camera sees there, instead of inventing pixels by inpainting. The robot is
            # inpainted out of it (dilated mask) so a hole can never receive a ghost arm at the reset pose.
            ref = np.ascontiguousarray(obs["agentview_image"]).copy()
            robot = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
            nom["ref_image"] = cv2.inpaint(ref, robot, 5, cv2.INPAINT_TELEA)
            got.append(nom)
            env.close()
        spread = max(float(np.abs(np.subtract(g["cam_pos"], got[0]["cam_pos"])).max()) for g in got)
        assert spread < 1e-4, f"{scene}: unperturbed tasks {ids} disagree on the camera by {spread:.4f} m"
        out[scene] = got[0]
        print(f"nominal {scene}: cam_pos {np.round(got[0]['cam_pos'], 4).tolist()} "
              f"q_nom {np.round(got[0]['q_nom'], 4).tolist()} (tasks {ids})", flush=True)
    if cache:
        json.dump({s: {k: v for k, v in d.items() if k != "ref_image"} for s, d in out.items()}, open(cache, "w"), indent=1)
        np.savez_compressed(cache + ".ref.npz", **{s: d["ref_image"] for s, d in out.items()})
    seed_flip(suite, make_env, out)
    _add_shading_reference(suite, make_env, out, cache)
    return out


def _add_shading_reference(suite, make_env, out, cache):
    """The body's shading statistic in the training setup, per scene (one unperturbed source task each), written into
    the cache. Cheap (one env per scene), so an older cache is upgraded in place rather than rebuilt."""
    import json
    from plus_common import DEFAULT_VIEW, categorize
    src = {}
    for i in range(suite.n_tasks):
        n = suite.get_task(i).name
        v = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", n)
        s = re.search(r"_initstate_(\d+)", n)
        if categorize(n) != "language" or (v and v.groups() != DEFAULT_VIEW) or (s and s.group(1) != "0"):
            continue
        src.setdefault(scene_of(n), i)
    for scene, d in out.items():
        if ("robot_shading_d" in d and "table_z" in d) or scene not in src:
            continue
        env = make_env(src[scene])
        try:
            obs = env.reset()
            can = Canonicalizer(env, d)
            if "robot_shading_d" not in d:
                d["robot_shading_d"] = can.photometric_consistency(obs)
            if "table_z" not in d:
                d["table_z"] = can.training_plane_z()
        finally:
            env.close()
        print(f"training references {scene}: shading {d['robot_shading_d']:.3f}, table z {d['table_z']:.3f} m", flush=True)
    if cache:
        json.dump({s: {k: v for k, v in d.items() if k != "ref_image"} for s, d in out.items()}, open(cache, "w"), indent=1)


# ---------------------------------------------------------------- rotation helpers (pure numpy / cv2)
def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2.0
        q = np.zeros(4)
        q[1 + i] = 0.25 * s
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return quat_wxyz_to_mat([w, x, y, z])


def rotvec_to_mat(r):
    return cv2.Rodrigues(np.asarray(r, np.float64).reshape(3, 1))[0]


def mat_to_rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)


def rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def orbit(C0, R0, pivot, az_deg, el_deg, scale):
    """Camera pose after orbiting about `pivot`: azimuth about world z, elevation about the horizontal axis
    perpendicular to the viewing direction, then distance scaling. R columns are camera axes in world."""
    Rz = rot_z(az_deg)
    C = pivot + Rz @ (C0 - pivot)
    R = Rz @ R0
    v = C - pivot
    h = np.array([-v[1], v[0], 0.0])
    if np.linalg.norm(h) < 1e-9:
        h = R[:, 0].copy()
    h = h / np.linalg.norm(h)
    Rel = rotvec_to_mat(h * np.deg2rad(el_deg))
    C = pivot + Rel @ (C - pivot)
    R = Rel @ R
    C = pivot + scale * (C - pivot)
    return C, R


def optical_pivot(C0, R0, plane_z=TABLE_Z):
    d = -R0[:, 2]                                  # MuJoCo cameras look along -z
    t = (plane_z - C0[2]) / d[2] if abs(d[2]) > 1e-6 else 1.0
    return C0 + max(t, 0.1) * d


def edge(mask):
    m = mask.astype(np.uint8)
    return m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8)))


def edge_dt(mask):
    """Distance from every pixel to the mask boundary."""
    return cv2.distanceTransform((1 - edge(mask)).astype(np.uint8), cv2.DIST_L2, 3)


def chamfer(obs_mask, ren_mask, dt_obs=None, trim=0.0):
    """Symmetric mean boundary distance in pixels between two binary masks. Boundary-based, so a shift of d pixels
    scores on the order of d rather than being diluted by the overlapping interior. trim > 0 drops that share of the
    largest boundary distances on each side: a learned mask that is wrong on one link then cannot drag the pose."""
    eo, er = edge(obs_mask), edge(ren_mask)
    if eo.sum() == 0 or er.sum() == 0:
        return 1e3
    if dt_obs is None:
        dt_obs = edge_dt(obs_mask)
    dt_ren = edge_dt(ren_mask)
    a, b = dt_ren[eo > 0], dt_obs[er > 0]
    if trim > 0:
        a = np.sort(a)[:max(1, int(round(len(a) * (1 - trim))))]
        b = np.sort(b)[:max(1, int(round(len(b) * (1 - trim))))]
    return 0.5 * (float(a.mean()) + float(b.mean()))


# the photometric fit on the robot's own body is trusted only when enough unclipped body pixels remain and the
# implied gain is sane; otherwise the channel abstains (Canonicalizer.photometric_fit)
PHOTO_MIN_PX, PHOTO_CLIP_LO, PHOTO_CLIP_HI, PHOTO_GAIN_MAX = 200, 3.0, 252.0, 2.5
# the photometric channel inverts a global gain/offset; when the body's shading PATTERN has changed (directional or
# grazing light) that model is wrong and the channel abstains. Ratio of the affine-invariant body statistic to its
# value in the training setup (photometric_consistency vs nominal["robot_shading_d"]): grazing scored 0.60 to 0.62 on
# the OFT sweep, every affine lighting change 0.68 or above.
PHOTO_SHADING_MIN = 0.65
IMG_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
_FLIP_CACHE = {}                   # camera name -> confident raw-to-observation flip (fixed per camera)


def noise_level(img):
    """Robust per-pixel noise sigma in gray levels: scaled MAD of the residual after a 3x3 median filter."""
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    r = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def denoise(img, level):
    h = float(max(3.0, 1.2 * level))
    return cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(img), None, h, h, 3, 7)


def reproject(rgb, z, f, Ct, Rt, Cn, Rn, splat=2, inpaint=True, return_mask=False):
    """Warp an RGB-D image from camera (Ct, Rt) into camera (Cn, Rn). Raw OpenGL buffer convention: row 0 is the
    image bottom. z is metric depth along the optical axis. Nearest point wins; holes are inpainted."""
    H, W = z.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = (cc + 0.5 - W / 2) * z / f
    y = (rr + 0.5 - H / 2) * z / f
    pc = np.stack([x, y, -z], -1).reshape(-1, 3)
    pw = Ct + pc @ Rt.T
    pn = (pw - Cn) @ Rn
    zn = -pn[:, 2]
    ok = zn > 1e-3
    col = np.round(f * pn[:, 0] / np.where(ok, zn, 1) + W / 2 - 0.5).astype(int)
    row = np.round(f * pn[:, 1] / np.where(ok, zn, 1) + H / 2 - 0.5).astype(int)
    colors = rgb.reshape(-1, 3)
    out = np.zeros_like(rgb)
    filled = np.zeros((H, W), bool)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(-zn[idx])]                 # far first, near last: near overwrites
    for dr in range(splat):
        for dc in range(splat):
            r, c = row[idx] + dr, col[idx] + dc
            m = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            out[r[m], c[m]] = colors[idx[m]]
            filled[r[m], c[m]] = True
    if inpaint and (~filled).any():
        out = cv2.inpaint(out, (~filled).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    if return_mask:
        return out, float(filled.mean()), filled
    return out, float(filled.mean())


def plane_depth(H, W, f, C, R, plane_z, far=10.0):
    """Depth along the optical axis of the ray through every pixel (raw buffer convention, row 0 = bottom) to the
    horizontal plane z = plane_z in world coordinates, for a camera at C with axes R (MuJoCo: looks along -z).
    Rays that never meet the plane (pointing up, or the plane behind the camera) get `far`."""
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = np.stack([(cc + 0.5 - W / 2) / f, (rr + 0.5 - H / 2) / f, -np.ones((H, W))], -1)     # camera-frame ray, z = -1
    dw = d @ R.T                                                                                # world-frame ray
    dz = dw[..., 2]
    t = np.where(np.abs(dz) > 1e-6, (plane_z - C[2]) / np.where(np.abs(dz) > 1e-6, dz, 1.0), -1.0)
    return np.where(t > 1e-3, t, far)                                                           # depth = t since d_z = -1


def points_depth(pts, f, C, R, H, W, splat=2, grow=3, see_through=0.04):
    """Depth along the optical axis of a WORLD point cloud z-buffered into camera (C, R); raw buffer convention, same
    projection as `reproject`. Pixels no point reaches are inf. Nearest point wins; `grow` passes of a 3x3 minimum
    over EMPTY pixels close the gaps a cloud sampled from another viewpoint leaves."""
    pc = (np.asarray(pts, np.float64) - C) @ R
    zn = -pc[:, 2]
    idx = np.where(zn > 1e-3)[0]
    idx = idx[np.argsort(-zn[idx])]                 # far first, near last: near overwrites
    col = np.round(f * pc[idx, 0] / zn[idx] + W / 2 - 0.5).astype(int)
    row = np.round(f * pc[idx, 1] / zn[idx] + H / 2 - 0.5).astype(int)
    z = np.full((H, W), np.inf)
    for dr in range(splat):
        for dc in range(splat):
            r, c = row + dr, col + dc
            m = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            z[r[m], c[m]] = zn[idx][m]
    for _ in range(grow):
        mn = cv2.erode(np.where(np.isfinite(z), z, 1e9).astype(np.float32), np.ones((3, 3), np.uint8))
        z = np.where(np.isfinite(z), z, np.where(mn < 1e8, mn, np.inf))
    # a point cloud is see-through: far points show between the samples of a near surface. A pixel lying well behind
    # the nearest depth in its 5x5 neighbourhood takes that depth (costs a 2 px fattening of near silhouettes)
    mn = cv2.erode(np.where(np.isfinite(z), z, 1e9).astype(np.float32), np.ones((5, 5), np.uint8))
    return np.where(np.isfinite(z) & (z > mn + see_through), mn, z)


def clean_mask(mask, min_px=30, keep_frac=0.15):
    """Speckle-free binary mask with its holes filled (uint8 0/1): every connected component at least `keep_frac` of
    the largest one is kept. Keeping ONLY the largest (before 2026-09-19) threw away half the robot whenever the image
    border cut the arm in two, which is the normal case under a large camera shift (task 768: mask IoU 0.50 on that
    frame, 31 cm calibration error). Tiny masks are returned as is."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() < min_px:
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        area = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(area >= max(min_px, keep_frac * area.max()))[0]
        m = np.isin(lab, keep).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    for k in range(1, n):
        if k not in border:                    # a background component not touching the border is a hole
            m[lab == k] = 1
    return m


def backproject(z, mask, f):
    """Camera-frame 3D points (MuJoCo convention: looks along -z, raw buffer row 0 = bottom) of masked pixels."""
    H, W = z.shape
    rr, cc = np.nonzero(mask)
    zz = z[rr, cc]
    return np.stack([(cc + 0.5 - W / 2) * zz / f, (rr + 0.5 - H / 2) * zz / f, -zz], -1)


def kabsch(P, Q):
    """R, C minimising sum ||R p + C - q||^2 over paired rows of P and Q."""
    mp, mq = P.mean(0), Q.mean(0)
    Hm = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(Hm)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def _nearest(model, query):
    try:
        from scipy.spatial import cKDTree
        d, i = cKDTree(model).query(query)
        return d, i
    except Exception:
        d = np.empty(len(query)); i = np.empty(len(query), int)
        for s in range(0, len(query), 512):
            dd = ((query[s:s + 512, None] - model[None]) ** 2).sum(-1)
            i[s:s + 512] = dd.argmin(1)
            d[s:s + 512] = np.sqrt(dd[np.arange(len(dd)), i[s:s + 512]])
        return d, i


def icp(P_cam, Q_world, R0, C0, iters=40, keep=0.8):
    """Rigid camera pose (R, C) aligning camera-frame points to a world-frame model; trimmed point-to-point ICP."""
    R, C = R0.copy(), C0.copy()
    for _ in range(iters):
        Pw = P_cam @ R.T + C
        d, idx = _nearest(Q_world, Pw)
        k = d <= np.quantile(d, keep)
        R_new, C_new = kabsch(P_cam[k], Q_world[idx[k]])
        done = np.linalg.norm(C_new - C) < 1e-5 and np.linalg.norm(mat_to_rotvec(R_new @ R.T)) < 1e-5
        R, C = R_new, C_new
        if done:
            break
    d, _ = _nearest(Q_world, P_cam @ R.T + C)
    return R, C, float(np.mean(np.sort(d)[: max(1, int(keep * len(d)))]))




# ---------------------------------------------------------------- simulator-facing wrapper
class Canonicalizer:
    def __init__(self, env, nominal, cam="agentview", res=256, mask_res=96, point_res=128, sensing=None):
        """env: LIBERO OffScreenRenderEnv. nominal: dict(cam_pos, cam_quat_wxyz, q_nom) read from an
        unperturbed task (the training setup)."""
        import mujoco
        self.mj = mujoco
        self.env, self.sim, self.cam, self.res, self.mres, self.pres = env, env.env.sim, cam, res, mask_res, point_res
        self.sensing = sensing          # None = renderer stand-ins; LearnedSensing = trained segmenter + monocular depth
        ref = nominal.get("ref_image")
        self.ref_image = None if ref is None else np.asarray(ref, np.uint8)
        m = self.sim.model
        self.cid = m.camera_name2id(cam)
        assert int(m.cam_bodyid[self.cid]) == 0, "camera must be attached to the world body"
        self.robot = env.env.robots[0]
        self.arm_idx = list(self.robot._ref_joint_pos_indexes)
        self.grip_idx = list(getattr(self.robot, "_ref_gripper_joint_pos_indexes", []))
        self.robot_geoms = np.array([g for g in range(m.ngeom)
                                     if self._body_name(int(m.geom_bodyid[g])).startswith(ROBOT_PREFIXES)])
        assert len(self.robot_geoms) > 0, "no robot geoms found"
        self.Cn = np.asarray(nominal["cam_pos"], float)
        self.Rn = quat_wxyz_to_mat(nominal["cam_quat_wxyz"])
        self.q_nom = np.asarray(nominal["q_nom"], float)
        self.table_z = nominal.get("table_z")           # training-setup table plane (world z), for plane-depth sensing
        self.static_pts = nominal.get("static_pts")     # training-setup static geometry (world points), hybrid depth
        self.gripper_geoms = np.array([g for g in range(m.ngeom)
                                       if self._body_name(int(m.geom_bodyid[g])).startswith("gripper0")])
        self._grip_max = 0.0
        self.mask_diag = []                             # learned mask vs renderer mask at the true camera (logging only)
        self.fovy = float(m.cam_fovy[self.cid])
        self.flip = None
        self.C_est = self.R_est = None

    def _body_name(self, i):
        m = self.sim.model
        try:
            return m.body_id2name(i) or ""
        except Exception:
            return self.mj.mj_id2name(m._model, self.mj.mjtObj.mjOBJ_BODY, i) or ""

    # -- state save / restore (bit-exact, camera included)
    def _save(self):
        m = self.sim.model
        return (np.array(self.sim.get_state().flatten()), self.sim.data.qacc_warmstart.copy(),
                m.cam_pos[self.cid].copy(), m.cam_quat[self.cid].copy())

    def _restore(self, s):
        m = self.sim.model
        m.cam_pos[self.cid][:] = s[2]
        m.cam_quat[self.cid][:] = s[3]
        self.sim.set_state_from_flattened(s[0])
        self.sim.data.qacc_warmstart[:] = s[1]
        self.sim.forward()

    def true_camera(self):
        m = self.sim.model
        return m.cam_pos[self.cid].copy(), quat_wxyz_to_mat(m.cam_quat[self.cid])

    def _set_camera(self, C, R):
        m = self.sim.model
        m.cam_pos[self.cid][:] = C
        m.cam_quat[self.cid][:] = mat_to_quat_wxyz(R)

    def _focal(self, H):
        return 0.5 * H / np.tan(self.fovy * np.pi / 360)

    def grip_opening(self):
        """Finger opening from proprioception (sum of absolute finger joint positions)."""
        return float(np.abs(self.sim.data.qpos[self.grip_idx]).sum()) if self.grip_idx else 0.0

    def grip_closed(self, frac=0.75):
        """Closed = the fingers are well inside the widest opening seen this episode. Proprioception only."""
        o = self.grip_opening()
        self._grip_max = max(self._grip_max, o)
        return self._grip_max > 0 and o < frac * self._grip_max

    def _gripper_mask(self, res):
        seg = self.sim.render(width=res, height=res, camera_name=self.cam, segmentation=True)
        return ((seg[..., 0] == int(self.mj.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], self.gripper_geoms)).astype(np.uint8)

    def _static_mask(self, res):
        """Pixels of geometry that cannot move: not the robot, and not a body whose root carries a free joint."""
        m = getattr(self.sim.model, "_model", self.sim.model)      # raw mujoco model: every field is there
        seg = self.sim.render(width=res, height=res, camera_name=self.cam, segmentation=True)
        gid = seg[..., 1].astype(int)
        isgeom = (seg[..., 0] == int(self.mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0) & (gid < m.ngeom)
        free = int(self.mj.mjtJoint.mjJNT_FREE)
        fixed = np.ones(m.ngeom, bool)
        for g in range(m.ngeom):
            root = int(m.body_rootid[int(m.geom_bodyid[g])])
            if int(m.body_jntnum[root]) > 0 and int(m.jnt_type[int(m.body_jntadr[root])]) == free:
                fixed[g] = False
        fixed[self.robot_geoms] = False
        out = np.zeros(gid.shape, np.uint8)
        out[isgeom] = fixed[gid[isgeom]]
        return out

    def training_static_cloud(self, res=384, voxel=0.006):
        """World points of the training setup's STATIC geometry (table, fixtures, walls, floor) scanned once from
        the nominal camera and a ring of viewpoints around it, robot and movable objects removed, 6 mm voxels. Training-setup
        information, like the reference image and the table plane; never computed on a deployment frame."""
        pivot = optical_pivot(self.Cn, self.Rn, self.table_z if self.table_z is not None else TABLE_Z)
        views = [(self.Cn, self.Rn)] + [orbit(self.Cn, self.Rn, pivot, az, el, 1.0)
                                        for az in (-60, -30, 30, 60) for el in (-15, 0, 15)]
        out = []
        s = self._save()
        try:
            for C, R in views:              # a scan: a single view misses the faces a shifted camera will look at
                self._set_camera(C, R)
                self.sim.forward()
                z, _ = self._depth_and_mask(res)
                st = self._static_mask(res)
                pts = backproject(z, (st > 0) & (z < 5.0), self._focal(res))
                out.append(C + pts @ R.T)
        finally:
            self._restore(s)
        pts = np.concatenate(out)
        _, keep = np.unique(np.round(pts / voxel).astype(np.int64), axis=0, return_index=True)
        return pts[keep].astype(np.float32)

    def _robot_mask(self, res):
        seg = self.sim.render(width=res, height=res, camera_name=self.cam, segmentation=True)
        return ((seg[..., 0] == int(self.mj.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], self.robot_geoms)).astype(np.uint8)

    def _obs_convention(self, obs):
        """Which flip maps the raw render onto the observation image. Compared on 32x32 area-downsampled images so
        heavy sensor noise averages out (full-resolution errors reach ~77 gray levels on LIBERO-Plus noise tasks).
        The flip is fixed per camera, so a confident detection is cached and reused if a later frame is ambiguous."""
        if self.cam in _FLIP_CACHE:         # fixed per camera: never re-detect on a (possibly corrupted) frame
            self.flip = _FLIP_CACHE[self.cam]
            return
        raw = self.sim.render(width=self.res, height=self.res, camera_name=self.cam)
        img = obs[f"{self.cam}_image"]
        small = lambda a: cv2.resize(np.ascontiguousarray(a).astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
        ref = small(img)
        cands = {"id": raw, "ud": raw[::-1], "lr": raw[:, ::-1], "rot": raw[::-1, ::-1]}
        errs = {k: float(np.abs(small(v) - ref).mean()) for k, v in cands.items()}
        best = min(errs, key=errs.get)
        second = sorted(errs.values())[1]
        if errs[best] < 40.0 and second - errs[best] > 2.0:
            self.flip = best
            _FLIP_CACHE[self.cam] = best
        elif self.cam in _FLIP_CACHE:
            self.flip = _FLIP_CACHE[self.cam]
        else:
            raise AssertionError(f"cannot match raw render to observation image: {errs}")

    def _to_obs(self, raw):
        return {"id": raw, "ud": raw[::-1], "lr": raw[:, ::-1], "rot": raw[::-1, ::-1]}[self.flip].copy()

    # -- state channel
    def joint_dev(self):
        return float(np.abs(self.sim.data.qpos[self.arm_idx] - self.q_nom).max())

    def nominal_eef(self):
        s = self._save()
        self.sim.data.qpos[self.arm_idx] = self.q_nom
        self.sim.forward()
        ob = self.env.env._get_observations(force_update=True)
        p, q = np.array(ob["robot0_eef_pos"]), np.array(ob["robot0_eef_quat"])
        self._restore(s)
        self.env.env._get_observations(force_update=True)
        return p, q

    def servo(self, obs, p_t, q_t_xyzw, max_steps=150, tol_p=0.004, tol_r=0.03):
        Rt = quat_xyzw_to_mat(q_t_xyzw)
        for i in range(max_steps):
            dp = p_t - np.asarray(obs["robot0_eef_pos"])
            dr = mat_to_rotvec(Rt @ quat_xyzw_to_mat(obs["robot0_eef_quat"]).T)
            if np.linalg.norm(dp) < tol_p and np.linalg.norm(dr) < tol_r:
                return obs, i
            a = np.zeros(7)
            a[:3] = np.clip(dp / 0.05, -1, 1)
            a[3:6] = np.clip(dr / 0.5, -1, 1)
            a[6] = -1.0
            obs, _, _, _ = self.env.step(a.tolist())
        return obs, max_steps

    def settle(self, obs, p_t, q_t_xyzw, max_steps=120, tol_p=0.0008, tol_r=0.006, rest=0.01):
        """Finish a measurement the way an episode starts: AT the target and AT REST. `servo` stops inside 4 mm with the
        arm still moving (measured 2026-09-19: 2.6 mm off and 0.19 rad/s at hand-over, against 0.00 to 0.03 rad/s for an
        untouched start), so a `full` episode began from a different physical state than its `raw` twin even when the
        wrapper then applied nothing. Keeps servoing (never a zero action: under it the arm sags 6 mm) until the pose is
        inside 0.8 mm / 0.006 rad AND every arm joint is slower than `rest` rad/s. Measured: 0.16 mm, 0.005 rad/s, 19 steps.
        Returns (obs, dict(steps, err_mm, speed))."""
        Rt = quat_xyzw_to_mat(q_t_xyzw)
        i = 0
        for i in range(max_steps):
            dp = p_t - np.asarray(obs["robot0_eef_pos"])
            dr = mat_to_rotvec(Rt @ quat_xyzw_to_mat(obs["robot0_eef_quat"]).T)
            speed = float(np.abs(self.sim.data.qvel[self.arm_idx]).max())
            if np.linalg.norm(dp) < tol_p and np.linalg.norm(dr) < tol_r and speed < rest:
                break
            a = np.zeros(7)
            a[:3] = np.clip(dp / 0.05, -1, 1)
            a[3:6] = np.clip(dr / 0.5, -1, 1)
            a[6] = -1.0
            obs, _, _, _ = self.env.step(a.tolist())
        return obs, dict(steps=i, err_mm=float(1e3 * np.linalg.norm(p_t - np.asarray(obs["robot0_eef_pos"]))),
                         speed=float(np.abs(self.sim.data.qvel[self.arm_idx]).max()))

    # -- camera channel
    def sweep(self, obs, poses=None):
        """poses: optional list of end-effector offsets; default is the method's sweep (3 poses, 7 under learned masks).
        The `verify` arm passes the 7-pose list when the self-check rejects the first calibration."""
        p0, q0 = self.nominal_eef()
        recs = []
        for off in (poses if poses is not None else (SWEEP if self.sensing is None else SWEEP_LEARNED)):
            obs, _ = self.servo(obs, p0 + off, q0)
            img = np.ascontiguousarray(obs[f"{self.cam}_image"]).copy()
            if self.sensing is None:
                mask, pts = self._robot_mask(self.mres), self._robot_points_camera(self.pres)
            else:                           # learned: segment the observed frame; metric points come after the silhouette fit
                if self.flip is None:
                    self._obs_convention(obs)
                mask = cv2.resize(self._to_obs(self.sensing.mask_obs(img)), (self.mres, self.mres),
                                  interpolation=cv2.INTER_NEAREST)
                pts = None
                true = self._robot_mask(self.mres)           # LOGGING ONLY: never reaches the method
                u = float(((mask > 0) | (true > 0)).sum())
                self.mask_diag.append(dict(iou=float(((mask > 0) & (true > 0)).sum() / u) if u else 0.0,
                                           area_ratio=float((mask > 0).sum() / max(float((true > 0).sum()), 1.0))))
            recs.append((self.sim.data.qpos[self.arm_idx].copy(),
                         self.sim.data.qpos[self.grip_idx].copy() if self.grip_idx else None,
                         mask, pts, img))
        obs, _ = self.servo(obs, p0, q0)
        return obs, recs

    def static_residual(self, obs, area_w=4.0):
        """MOTION-FREE check: at the pose the arm is in NOW, does the robot's silhouette sit where the TRAINING camera says
        it must? Same silhouette score as the calibration (`cal_nominal_score`), from one record and no arm motion. Small:
        the camera has not moved, so the measurement sweep would be pure cost (it moves the arm, and a sampling policy then
        starts its episode from a different state). Logged rows, 2026-09-21: every one of 240 camera-shifted tasks reads
        above 3.0, every one of 140 others below 0.02 (sweep conditions: above 2.26 against below 0.86)."""
        if self.sensing is None:
            mask, pts = self._robot_mask(self.mres), self._robot_points_camera(self.pres)
        else:
            img = np.ascontiguousarray(obs[f"{self.cam}_image"]).copy()
            if self.flip is None:
                self._obs_convention(obs)
            mask = cv2.resize(self._to_obs(self.sensing.mask_obs(img)), (self.mres, self.mres), interpolation=cv2.INTER_NEAREST)
            pts = None
        rec = (self.sim.data.qpos[self.arm_idx].copy(), self.sim.data.qpos[self.grip_idx].copy() if self.grip_idx else None,
               mask, pts, None)
        s = self._save()
        try:
            return float(self._score([rec], [(edge_dt(mask), mask)], self.Cn, self.Rn, area_w, 0.0, clean=self.sensing is not None))
        finally:
            self._restore(s)

    def _score(self, recs, dts, C, R, area_w=0.0, trim=0.0, res=None, clean=False):
        """Mean boundary chamfer (pixels) over the sweep poses. area_w adds |log(rendered area / observed area)|:
        pixel chamfer alone shrinks as a candidate camera backs away and the robot gets small, so without the size
        term the search drifts to far cameras."""
        self._set_camera(C, R)
        total = 0.0
        for rec, dt in zip(recs, dts):
            qa, qg = rec[0], rec[1]
            self.sim.data.qpos[self.arm_idx] = qa
            if qg is not None:
                self.sim.data.qpos[self.grip_idx] = qg
            self.sim.forward()
            ren = self._robot_mask(res or self.mres)
            if clean:                       # compare like with like: the observed (learned) mask went through clean_mask,
                ren = clean_mask(ren)       # so stray robot geometry far from the arm must not count in the render either
            total += chamfer(dt[1], ren, dt_obs=dt[0], trim=trim)
            if area_w:
                total += area_w * abs(np.log((ren.sum() + 1.0) / (dt[1].sum() + 1.0)))
        return total / len(recs)

    def calibrate(self, recs, grid_az=range(-60, 61, 10), grid_el=range(-20, 21, 10),
                  grid_scale=(0.8, 1.0, 1.25, 1.5, 1.75, 2.0), cem_iters=8, pop=24, elite=6, seed=0):
        dts = [(edge_dt(rec[2]), rec[2]) for rec in recs]
        s = self._save()
        try:
            nominal_score = self._score(recs, dts, self.Cn, self.Rn)
            pivot = optical_pivot(self.Cn, self.Rn)
            grid = []
            for az in grid_az:
                for el in grid_el:
                    for sc in grid_scale:
                        C, R = orbit(self.Cn, self.Rn, pivot, az, el, sc)
                        grid.append((self._score(recs, dts, C, R), C, R))
            grid.sort(key=lambda g: g[0])
            best_score, Cb, Rb = grid[0]
            rng = np.random.default_rng(seed)
            mu, sd = np.zeros(6), np.array([0.06, 0.06, 0.06, 0.08, 0.08, 0.08])
            for _ in range(cem_iters):
                cand = mu + sd * rng.standard_normal((pop, 6))
                cand[0] = mu
                sc_ = []
                for c in cand:
                    C, R = Cb + c[:3], rotvec_to_mat(c[3:]) @ Rb
                    sc_.append(self._score(recs, dts, C, R))
                sc_ = np.array(sc_)
                el_ = cand[np.argsort(sc_)[:elite]]
                if sc_.min() < best_score:
                    best_score = float(sc_.min())
                mu, sd = el_.mean(0), el_.std(0) + 1e-4
            self.C_est, self.R_est = Cb + mu[:3], rotvec_to_mat(mu[3:]) @ Rb
            final = self._score(recs, dts, self.C_est, self.R_est)
        finally:
            self._restore(s)
        return dict(nominal_score=nominal_score, grid_best=grid[0][0], final_score=final)

    def calibrate_v2(self, recs, max_cm=110.0, max_deg=80.0, area_w=4.0, accept=0.8, seed=0):
        """Camera pose from the sweep. Silhouette grid + CEM with the size term and a plausibility bound, keeping the
        best candidate seen (not the CEM mean); then depth ICP refined from the best silhouette pose and from nominal.
        Every candidate is judged by the same silhouette score, and the nominal pose is kept unless beaten by the
        factor `accept`. The orbit pivot sits on the nominal optical axis at the robot's measured median depth."""
        dts = [(edge_dt(rec[2]), rec[2]) for rec in recs]
        learned = any(rec[3] is None for rec in recs)
        trim = 0.0      # measured 2026-09-18: trimming changes no synthetic fit (blob, shadow, missing link); kept at 0
        s = self._save()
        out = dict(method="nominal", icp_residual_cm=None)
        try:
            if learned:                     # pivot depth from the robot's known geometry rendered at the nominal camera
                self._set_camera(self.Cn, self.Rn)
                self.sim.data.qpos[self.arm_idx] = recs[0][0]
                self.sim.forward()
                z0, mk0 = self._depth_and_mask(self.pres)
                pivot_depth = float(np.median(z0[mk0 > 0])) if mk0.any() else 1.0
                P = None
            else:
                P = np.concatenate([rec[3] for rec in recs])
                pivot_depth = float(np.median(-P[:, 2]))
            pivot = self.Cn + pivot_depth * (-self.Rn[:, 2])
            def score(C, R):
                e = pose_error(C, R, self.Cn, self.Rn)
                if e[0] > max_cm or e[1] > max_deg:
                    return np.inf
                return self._score(recs, dts, C, R, area_w, trim, clean=learned)

            nominal = score(self.Cn, self.Rn)
            best = (nominal, self.Cn, self.Rn, "nominal")
            cands = []
            # measured LIBERO-Plus camera shifts reach 99 cm and 70 deg; calibration was exact up to 35 cm / 31 deg and
            # failed beyond, where the old 35 cm / 35 deg bound pinned it. Coarser steps keep the grid size (539 poses).
            for az in range(-75, 76, 15):
                for el in range(-45, 46, 15):
                    for sc in (0.7, 0.85, 1.0, 1.2, 1.45, 1.75, 2.1):
                        C, R = orbit(self.Cn, self.Rn, pivot, az, el, sc)
                        v = score(C, R)
                        if np.isfinite(v):
                            cands.append((v, C, R))
            cands.sort(key=lambda x: x[0])
            rng = np.random.default_rng(seed)
            for v0, Cb, Rb in cands[:3]:
                if v0 < best[0]:
                    best = (v0, Cb, Rb, "silhouette")
                mu, sd = np.zeros(6), np.array([0.04, 0.04, 0.04, 0.05, 0.05, 0.05])
                for _ in range(6):
                    X = mu + sd * rng.standard_normal((20, 6))
                    X[0] = mu
                    vals = []
                    for x in X:
                        C, R = Cb + x[:3], rotvec_to_mat(x[3:]) @ Rb
                        v = score(C, R)
                        vals.append(v)
                        if v < best[0]:
                            best = (v, C, R, "silhouette")
                    elite = X[np.argsort(np.array(vals))[:5]]
                    mu, sd = elite.mean(0), elite.std(0) + 1e-4
            if learned:                     # metric robot points: learned mask + monocular depth scaled at the silhouette pose
                P = self.sensing.robot_points(self, recs, best[1], best[2])
            Q = self._robot_model_world(recs, self.pres, pivot)
            for _, C0, R0, _ in [best, (nominal, self.Cn, self.Rn, "nominal")]:
                R, C, resid = icp(P, Q, R0, C0)
                v = score(C, R)
                if v < best[0]:
                    best = (v, C, R, "icp")
                    out["icp_residual_cm"] = resid * 100
            if learned and best[3] != "nominal" and getattr(self.sensing, "fine_cal", False):
                # fine stage at twice the mask resolution (the 96 px score leaves about 3 px of slack at 256): masks
                # re-segmented from the stored sweep frames, trimmed chamfer, CEM from 3 cm / 3 deg down
                F = 2 * self.mres
                fm = [cv2.resize(self._to_obs(self.sensing.mask_obs(rec[4])), (F, F), interpolation=cv2.INTER_NEAREST)
                      for rec in recs]
                fdts = [(edge_dt(k), k) for k in fm]
                fine = lambda C, R: self._score(recs, fdts, C, R, area_w, trim, res=F, clean=True)
                fb = (fine(best[1], best[2]), best[1], best[2])
                out["fine_score_before"] = float(fb[0])
                mu, sd = np.zeros(6), np.array([0.03, 0.03, 0.03, 0.05, 0.05, 0.05])
                for _ in range(5):
                    X = mu + sd * rng.standard_normal((24, 6))
                    X[0] = mu
                    vals = []
                    for x in X:
                        C, R = best[1] + x[:3], rotvec_to_mat(x[3:]) @ best[2]
                        v = fine(C, R)
                        vals.append(v)
                        if v < fb[0]:
                            fb = (v, C, R)
                    elite = X[np.argsort(np.array(vals))[:6]]
                    mu, sd = elite.mean(0), elite.std(0) + 1e-4
                out["fine_score_after"] = float(fb[0])
                v = score(fb[1], fb[2])
                if v <= best[0] + 0.3:              # never trade the coarse agreement away
                    best = (min(v, best[0]), fb[1], fb[2], best[3] + "+fine")
        finally:
            self._restore(s)
        # relative AND absolute margin: near a perfect nominal match (score ~0) a ratio alone fires on noise
        applied = best[3] != "nominal" and best[0] < accept * nominal and nominal - best[0] > 0.5
        self.C_est, self.R_est = (best[1], best[2]) if applied else (self.Cn.copy(), self.Rn.copy())
        shift = pose_error(self.C_est, self.R_est, self.Cn, self.Rn)
        out.update(nominal_score=float(nominal), final_score=float(best[0]), applied=bool(applied),
                   method=best[3] if applied else "nominal", est_shift_cm=shift[0], est_shift_deg=shift[1])
        return out

    # -- depth-based calibration (ICP of the robot's own surface)
    def _depth_and_mask(self, res):
        rgb, depth = self.sim.render(width=res, height=res, camera_name=self.cam, depth=True)
        m = self.sim.model
        extent = float(m.stat.extent)
        near, far = float(m.vis.map.znear) * extent, float(m.vis.map.zfar) * extent
        z = near / (1.0 - depth * (1.0 - near / far))
        return z, self._robot_mask(res)

    def training_plane_z(self, res=128):
        """World height of the dominant horizontal surface seen by the nominal camera in the training setup: the
        table. Renderer depth at the nominal camera, non-robot pixels, histogram mode over 1 cm bins. Training-setup
        information, like the reference image; never computed on a deployment frame."""
        s = self._save()
        try:
            self._set_camera(self.Cn, self.Rn)
            self.sim.forward()
            z, mk = self._depth_and_mask(res)
        finally:
            self._restore(s)
        pts = backproject(z, (mk == 0) & (z < 5.0), self._focal(res))
        zw = (self.Cn + pts @ self.Rn.T)[:, 2]
        hist, edges = np.histogram(zw, bins=np.arange(zw.min(), zw.max() + 0.01, 0.01))
        return float(edges[int(np.argmax(hist))] + 0.005)

    def _robot_points_camera(self, res):
        """Observed robot surface in the (unknown) actual camera's frame, from the RGB-D image and arm segmentation."""
        z, mk = self._depth_and_mask(res)
        pts = backproject(z, mk, self._focal(res))
        if len(pts) > 3000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
        return pts

    def _robot_model_world(self, recs, res, pivot=None):
        """The robot's surface in world coordinates at the recorded joint angles, rendered from known virtual
        viewpoints around the nominal camera (robot geometry plus known joints only)."""
        s = self._save()
        pts = []
        try:
            pivot = optical_pivot(self.Cn, self.Rn) if pivot is None else pivot
            views = [(self.Cn, self.Rn)] + [orbit(self.Cn, self.Rn, pivot, az, el, 1.0)
                                            for az, el in ((-40, 0), (40, 0), (0, 20))]
            for rec in recs:
                self.sim.data.qpos[self.arm_idx] = rec[0]
                if rec[1] is not None:
                    self.sim.data.qpos[self.grip_idx] = rec[1]
                for C, R in views:
                    self._set_camera(C, R)
                    self.sim.forward()
                    z, mk = self._depth_and_mask(res)
                    pc = backproject(z, mk, self._focal(res))
                    pts.append(C + pc @ R.T)
        finally:
            self._restore(s)
        Q = np.concatenate(pts)
        if len(Q) > 12000:
            Q = Q[np.random.default_rng(1).choice(len(Q), 12000, replace=False)]
        return Q

    def obs_robot_mask(self, obs):
        """Robot mask in the observation image's orientation and resolution (segmentation stand-in)."""
        if self.flip is None:
            self._obs_convention(obs)
        if self.sensing is not None:
            return self.sensing.mask_obs(obs[f"{self.cam}_image"])
        return self._to_obs(self._robot_mask(obs[f"{self.cam}_image"].shape[0]))

    # -- photometric channel
    def photometric_fit(self, obs, nominal):
        """Per-channel gain and offset mapping the robot's current appearance onto its appearance in the training
        setup. The robot's own body is the reference, so table, background and layout changes do not enter the
        estimate. Call with the arm at its nominal pose.

        Clipped pixels carry no information about a linear map, so they are dropped from the fit. If too few pixels
        survive, or the implied gain leaves a sane range, the channel abstains (photo_ok False) rather than apply a
        map the robot's own measurements do not support: a saturated or strongly directional lighting change is not
        identifiable from the body alone. The reported shift and gain still use every robot pixel, so the
        identifiability score sees the full magnitude of the change either way."""
        mask = self.obs_robot_mask(obs) > 0
        allpx = obs[f"{self.cam}_image"][mask].reshape(-1, 3).astype(np.float32)
        rmu = np.asarray(nominal["robot_rgb_mean"], np.float32)
        rsd = np.asarray(nominal["robot_rgb_std"], np.float32)
        keep = ((allpx > PHOTO_CLIP_LO) & (allpx < PHOTO_CLIP_HI)).all(1)
        n_keep, n_all = int(keep.sum()), max(int(len(allpx)), 1)
        ok = n_keep >= PHOTO_MIN_PX
        px = allpx[keep] if ok else allpx
        mu, sd = px.mean(0), px.std(0) + 1e-3
        gain = rsd / sd
        ok = bool(ok and np.all(gain > 1.0 / PHOTO_GAIN_MAX) and np.all(gain < PHOTO_GAIN_MAX))
        d0 = nominal.get("robot_shading_d")
        shading = float("nan")
        if d0 is not None and np.isfinite(d0) and d0 > 0:
            d = self.photometric_consistency(obs)
            shading = float(min(d, d0) / max(d, d0)) if np.isfinite(d) and max(d, d0) > 0 else float("nan")
            ok = bool(ok and (np.isnan(shading) or shading >= PHOTO_SHADING_MIN))
        self.photo = (gain, rmu - gain * mu) if ok else None
        return dict(photo_ok=ok, photo_mean_shift=float(np.abs(allpx.mean(0) - rmu).max()),
                    photo_gain=[float(g) for g in gain], photo_px=n_keep,
                    photo_clipped=float(1.0 - n_keep / n_all), photo_shading=shading)

    def apply_photo(self, img):
        if self.photo is None:                 # abstained: leave the image as measured
            return img
        gain, off = self.photo
        return np.clip(img.astype(np.float32) * gain + off, 0, 255).astype(np.uint8)

    def photometric_consistency(self, obs, min_px=50):
        """Affine-invariant split-half statistic of the robot's own body: log ratio of per-channel spread between the
        upper and lower halves of the body plus their mean difference in units of the whole-body spread. A global
        gain/offset change (the only lighting change the photometric channel can invert) leaves it unchanged; a
        directional or grazing change alters the body's shading pattern and moves it. Compared against the same
        statistic at the unperturbed state, it says whether the affine inversion model is valid, from the body alone.
        Returns nan when the body is too small to split."""
        mask = self.obs_robot_mask(obs) > 0
        img = obs[f"{self.cam}_image"].astype(np.float32)
        rows = np.where(mask.any(1))[0]
        if len(rows) < 4:
            return float("nan")
        mid = int((rows.min() + rows.max()) // 2)
        top = img[:mid][mask[:mid]].reshape(-1, 3)
        bot = img[mid:][mask[mid:]].reshape(-1, 3)
        if len(top) < min_px or len(bot) < min_px:
            return float("nan")
        sd = img[mask].reshape(-1, 3).std(0) + 1e-3
        d = np.abs(np.log((top.std(0) + 1e-3) / (bot.std(0) + 1e-3))).sum() + np.abs((top.mean(0) - bot.mean(0)) / sd).sum()
        return float(d)

    def canonical_image(self, obs, C=None, R=None, return_mask=False, fill="ref"):
        """The OBSERVED image reprojected into the nominal camera, using the renderer's depth at the actual camera
        (RGB-D stand-in) and camera pose (C, R) (default: the self-calibrated estimate)."""
        if self.flip is None:
            self._obs_convention(obs)
        C = self.C_est if C is None else C
        R = self.R_est if R is None else R
        img = obs[f"{self.cam}_image"]
        H = img.shape[0]
        if self.sensing is not None:
            z = self.sensing.metric_depth(self, img, C, R)
        else:
            _, depth = self.sim.render(width=H, height=H, camera_name=self.cam, depth=True)
            m = self.sim.model
            extent = float(m.stat.extent)
            near, far = float(m.vis.map.znear) * extent, float(m.vis.map.zfar) * extent
            z = near / (1.0 - depth * (1.0 - near / far))
        # the flips are involutions, so _to_obs also maps the observation back to the raw buffer orientation
        ref = self.ref_image if (fill == "ref" and self.ref_image is not None and self.ref_image.shape == img.shape) else None
        out, cover, filled = reproject(self._to_obs(img), z, self._focal(H), C, R, self.Cn, self.Rn,
                                       inpaint=ref is None, return_mask=True)
        if ref is not None:                  # holes take the training camera's view of the scene, not invented pixels
            out[~filled] = self._to_obs(ref)[~filled]
        if return_mask:
            return self._to_obs(out), cover, self._to_obs(filled)
        return self._to_obs(out), cover

    def self_check(self, C=None, R=None, res=128):
        """VERIFY THE UNDO with the instrument that measured it (policy-free, no reference episode, no ground truth).
        Warp the robot's OBSERVED silhouette into the training camera with the same pose and depth the image correction
        uses, and compare it with where the robot's own kinematics say the silhouette must be in the training view. A
        correction that landed puts the arm where the policy expects it; one that did not, does not.
        Returns dict(precision, recall, f1): precision = warped robot pixels that land on the predicted silhouette,
        recall = predicted silhouette (inside the area the warp could fill) that is covered. Simulator sensing only."""
        if self.sensing is not None:
            return None
        C = self.C_est if C is None else C
        R = self.R_est if R is None else R
        m_obs = self._robot_mask(res)
        _, depth = self.sim.render(width=res, height=res, camera_name=self.cam, depth=True)
        m = self.sim.model
        extent = float(m.stat.extent)
        near, far = float(m.vis.map.znear) * extent, float(m.vis.map.zfar) * extent
        z = near / (1.0 - depth * (1.0 - near / far))
        rgb = np.repeat((m_obs * 255).astype(np.uint8)[..., None], 3, axis=-1)
        out, _, filled = reproject(rgb, z, self._focal(res), C, R, self.Cn, self.Rn, inpaint=False, return_mask=True)
        m_can = (out[..., 0] > 127) & filled
        s = self._save()
        try:
            self._set_camera(self.Cn, self.Rn)
            self.sim.forward()
            m_nom = self._robot_mask(res) > 0
        finally:
            self._restore(s)
        inter = float((m_can & m_nom).sum())
        prec = inter / max(float(m_can.sum()), 1.0)
        rec = inter / max(float((m_nom & filled).sum()), 1.0)
        return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9))

    def oracle_nominal_image(self, obs):
        """UPPER BOUND ONLY: the true render from the nominal camera."""
        if self.flip is None:
            self._obs_convention(obs)
        s = self._save()
        try:
            self._set_camera(self.Cn, self.Rn)
            self.sim.forward()
            raw = self.sim.render(width=self.res, height=self.res, camera_name=self.cam)
        finally:
            self._restore(s)
        return self._to_obs(raw)


HELD_HALO_M = 0.08                  # a held object stays within this distance of the gripper geometry


def hybrid_depth(z_plane, z_robot, mask_robot, mask_grip, static_pts, f, C, R, closed, img=None, ref=None,
                 Cn=None, Rn=None, same=30):
    """Non-robot depth for the 'hybrid' mode (see LearnedSensing). Pure function of training-setup references, the
    robot's rendered geometry at the calibrated pose, the observed frame and one proprioceptive bit. img / ref (raw
    buffer orientation) are optional: a halo pixel whose colour matches the training reference at the place the TABLE
    homography sends it is table, and stays on the plane; only what the reference cannot explain rides with the hand."""
    z = z_plane.copy()
    H, W = z.shape
    if static_pts is not None and len(static_pts):
        zs = points_depth(static_pts, f, C, R, H, W)
        z = np.where(np.isfinite(zs), zs, z)
    if closed and mask_grip is not None and mask_grip.sum() > 5:
        z_g = float(np.median(z_robot[mask_grip > 0]))
        r = int(np.clip(round(HELD_HALO_M * f / max(z_g, 0.05)), 2, H // 4))
        halo = cv2.dilate(mask_grip.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        sel = (halo > 0) & (mask_robot == 0) & (z > z_g + 0.02)      # only what lies BEHIND the hand is pulled forward
        if img is not None and ref is not None and sel.any():
            rr, cc = np.nonzero(sel)
            zz = z_plane[rr, cc]
            pc = np.stack([(cc + 0.5 - W / 2) * zz / f, (rr + 0.5 - H / 2) * zz / f, -zz], -1)
            pn = ((C + pc @ R.T) - Cn) @ Rn
            zn = np.maximum(-pn[:, 2], 1e-6)
            c2 = np.round(f * pn[:, 0] / zn + W / 2 - 0.5).astype(int)
            r2 = np.round(f * pn[:, 1] / zn + H / 2 - 0.5).astype(int)
            inside = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            a = cv2.GaussianBlur(img, (5, 5), 0).astype(np.int16)
            b = cv2.GaussianBlur(ref, (5, 5), 0).astype(np.int16)
            is_table = np.zeros(len(rr), bool)
            is_table[inside] = np.abs(a[rr[inside], cc[inside]] - b[r2[inside], c2[inside]]).max(-1) < same
            sel[rr[is_table], cc[is_table]] = False
        z[sel] = z_g
    return z


def read_static_clouds(suite, make_env, nominals, cache=None):
    """Attach `static_pts` (Canonicalizer.training_static_cloud) to every scene's nominal. Stored NEXT TO the nominal
    cache (`<cache>.cloud.npz`), never inside it, so runs that share the cache are not disturbed."""
    import os
    from plus_common import DEFAULT_VIEW, categorize
    path = (cache + ".cloud.npz") if cache else None
    have = dict(np.load(path)) if path and os.path.exists(path) else {}
    src = {}
    for i in range(suite.n_tasks):
        n = suite.get_task(i).name
        v = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", n)
        s = re.search(r"_initstate_(\d+)", n)
        if categorize(n) != "language" or (v and v.groups() != DEFAULT_VIEW) or (s and s.group(1) != "0"):
            continue
        src.setdefault(scene_of(n), i)
    new = False
    for scene, d in nominals.items():
        if scene not in have and scene in src:
            env = make_env(src[scene])
            try:
                env.reset()
                have[scene] = Canonicalizer(env, d).training_static_cloud()
            finally:
                env.close()
            new = True
            zs = have[scene][:, 2]
            print(f"static cloud {scene}: {len(zs)} points, height {zs.min():.2f} to {zs.max():.2f} m "
                  f"(table {d.get('table_z', float('nan')):.2f})", flush=True)
        d["static_pts"] = have.get(scene)
    if path and new:
        np.savez_compressed(path, **have)
    return nominals


def fit_inverse_depth(disp, z_ref, sel, z_min=0.05, z_max=10.0):
    """Map relative disparity to metric depth, 1/z = a * disp + b, fitted on the selected pixels against reference
    metric depth. Falls back to a pure scale if fewer than 50 pixels are available."""
    inv_ref = 1.0 / np.maximum(z_ref[sel], 1e-3)
    if sel.sum() >= 50:
        A = np.stack([disp[sel], np.ones(int(sel.sum()))], 1)
        a, b = np.linalg.lstsq(A, inv_ref, rcond=None)[0]
    else:
        a, b = (float(np.median(inv_ref) / max(np.median(disp[sel]), 1e-6)) if sel.any() else 1.0), 0.0
    return 1.0 / np.clip(a * disp + b, 1.0 / z_max, 1.0 / z_min)


METRIC_DEPTH_MODELS = ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "Intel/zoedepth-nyu-kitti")


class MetricDepth:
    """Monocular metric depth in metres for an HxWx3 uint8 image. Loaded in this process when its transformers can read
    the model; otherwise (the OFT venv pins transformers 4.40, which cannot) served by depth_server.py in the system
    python, one persistent process, frames passed as .npy files in /tmp."""

    def __init__(self, device="cuda"):
        self.proc = self.server = None
        last = None
        for name in METRIC_DEPTH_MODELS:
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                import torch
                self.proc = AutoImageProcessor.from_pretrained(name)
                self.model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
                self.name, self.where, self.torch, self.device = name, "in process", torch, device
                return
            except Exception as e:
                last = e
        import subprocess
        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_server.py")
        py = os.environ.get("METRIC_DEPTH_PYTHON", "/usr/bin/python3")
        self.server = subprocess.Popen([py, server], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        line = self.server.stdout.readline().strip()
        if not line.startswith("ready"):
            raise RuntimeError(f"no metric depth model: in process {last!r}; server said {line!r}")
        self.name, self.where = line.split(" ", 1)[1], f"server {py}"

    def __call__(self, img):
        img = np.ascontiguousarray(np.asarray(img)[..., :3]).astype(np.uint8)
        if self.server is None:
            import torch.nn.functional as F
            with self.torch.no_grad():
                inputs = self.proc(images=img, return_tensors="pt").to(self.device)
                pred = self.model(**inputs).predicted_depth
                pred = F.interpolate(pred[:, None].float(), size=img.shape[:2], mode="bicubic", align_corners=False)[0, 0]
            return pred.cpu().numpy().astype(np.float64)
        src, dst = "/tmp/_md_in.npy", "/tmp/_md_out.npy"
        np.save(src, img)
        self.server.stdin.write(src + " " + dst + chr(10))
        ans = self.server.stdout.readline().strip()
        assert ans == "ok", f"depth server: {ans}"
        return np.load(dst).astype(np.float64)


class LearnedSensing:
    """Replaces both simulator stand-ins. Robot mask: a segmenter trained on held-out tasks (train_seg.py). Depth: a
    pretrained monocular network (Depth Anything, relative disparity) mapped to metric depth on the robot's OWN pixels
    against its known geometry rendered at the estimated camera pose and current joint angles. No renderer depth or
    segmentation of the observed scene is used."""

    def __init__(self, seg_ckpt, depth_model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda", depth="mono"):
        """depth: 'mono' = Depth Anything scaled on the robot's own pixels; 'plane' = no depth network: the table plane
        of the training setup (nominal cache) intersected with the calibrated camera's rays, the robot's own pixels
        taking the depth of its known geometry. 'hybrid' = plane, plus (a) the training setup's static geometry
        (fixtures, walls, floor: one depth capture of the training scene, z-buffered into the calibrated camera) and
        (b) when proprioception says the gripper is closed, the pixels around the gripper take the gripper's depth, so a
        held object travels with the hand instead of being torn off it by the table homography (2026-09-18: 12 to 55 px
        of held-object displacement under plane depth at the exact pose). All use only the learned segmenter."""
        import torch
        from train_seg import load as load_seg
        self.torch, self.device, self.depth_mode = torch, device, depth
        self.seg = load_seg(seg_ckpt, device)
        self.fine_cal = depth == "hybrid"      # 192 px refinement of the silhouette pose; off for the older arms so
        if depth == "metric":                  # POST HOC (2026-09-24): a monocular METRIC depth network, used as is
            self.depth, self.depth_name = MetricDepth(device), "metric"
            print(f"learned sensing: segmenter {seg_ckpt}, metric depth {self.depth.name} ({self.depth.where})", flush=True)
            return
        if depth in ("plane", "hybrid"):       # their rows stay comparable (cam_est_err_cm_deg is logged for both)
            self.depth, self.depth_name = None, depth
            print(f"learned sensing: segmenter {seg_ckpt}, depth from the training-setup table plane"
                  + (" + static geometry + held-object rule" if depth == "hybrid" else ""), flush=True)
            return
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        last = None
        for name in (depth_model, "LiheYoung/depth-anything-small-hf"):
            try:
                self.proc = AutoImageProcessor.from_pretrained(name)
                self.depth = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
                self.depth_name = name
                break
            except Exception as e:          # older transformers cannot read the V2 config: fall back to V1
                last = e
        else:
            raise RuntimeError(f"no monocular depth model could be loaded: {last}")
        print(f"learned sensing: segmenter {seg_ckpt}, depth {self.depth_name}", flush=True)

    def mask_obs(self, img):
        """Learned robot mask, cleaned for silhouette matching: the boundary chamfer that drives calibration chases
        speckle and pinholes, so keep the largest connected component and fill its holes (learned-mask calibration was
        5 to 30 cm off on a third of the camera tasks before this, 2026-09-18)."""
        from train_seg import predict
        return clean_mask(predict(self.seg, img, self.device))

    def disparity_obs(self, img):
        import torch.nn.functional as F
        with self.torch.no_grad():
            inputs = self.proc(images=np.ascontiguousarray(img), return_tensors="pt").to(self.device)
            pred = self.depth(**inputs).predicted_depth
            pred = F.interpolate(pred[:, None].float(), size=img.shape[:2], mode="bicubic", align_corners=False)[0, 0]
        return pred.cpu().numpy().astype(np.float64)

    def metric_depth(self, can, img_obs, C, R):
        H = img_obs.shape[0]
        s = can._save()
        try:
            can._set_camera(C, R)
            can.sim.forward()
            z_r, mask_r = can._depth_and_mask(H)          # the robot's OWN geometry at the calibrated pose
            grip = can._gripper_mask(H) if self.depth_mode == "hybrid" else None
        finally:
            can._restore(s)
        if self.depth_mode in ("plane", "hybrid"):
            assert can.table_z is not None, "plane depth needs table_z in the nominal cache (read_nominals upgrades it)"
            f = can._focal(H)
            z = plane_depth(H, img_obs.shape[1], f, C, R, can.table_z)
            if self.depth_mode == "hybrid":
                ref = can.ref_image if (can.ref_image is not None and can.ref_image.shape == img_obs.shape) else None
                z = hybrid_depth(z, z_r, mask_r, grip, can.static_pts, f, C, R, can.grip_closed(),
                                 img=np.ascontiguousarray(can._to_obs(img_obs)),
                                 ref=None if ref is None else np.ascontiguousarray(can._to_obs(ref)), Cn=can.Cn, Rn=can.Rn)
            return np.where(mask_r > 0, z_r, z)
        if self.depth_mode == "metric":
            # metres straight from the network, robot pixels included: no fit to the robot's rendered geometry, so the
            # calibration's point cloud is measured, not assumed at the candidate pose
            return np.clip(can._to_obs(self.depth(img_obs)), 0.05, 10.0)
        disp = can._to_obs(self.disparity_obs(img_obs))
        mask_l = can._to_obs(self.mask_obs(img_obs)) > 0
        sel = mask_l & (mask_r > 0)
        if sel.sum() < 50:
            sel = mask_r > 0
        return fit_inverse_depth(disp, z_r, sel)

    def robot_points(self, can, recs, C, R):
        out = []
        for rec in recs:
            can.sim.data.qpos[can.arm_idx] = rec[0]
            if rec[1] is not None:
                can.sim.data.qpos[can.grip_idx] = rec[1]
            can.sim.forward()
            img = rec[4]
            z = self.metric_depth(can, img, C, R)
            pts = backproject(z, can._to_obs(self.mask_obs(img)) > 0, can._focal(img.shape[0]))
            if len(pts) > 3000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
            out.append(pts)
        return np.concatenate(out)


def pose_error(C1, R1, C2, R2):
    return float(np.linalg.norm(C1 - C2) * 100), float(np.rad2deg(np.linalg.norm(mat_to_rotvec(R1 @ R2.T))))
