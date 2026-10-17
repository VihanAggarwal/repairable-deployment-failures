"""POST HOC (TASKS.md P47, added 2026-09-24; not a registered test, reported with a dagger): the POSE-SOURCE ABLATION.

The same reprojection (sweep_run.prepare_full, `full`) driven by three camera-pose sources on the same sweep, the same
state channel, the same hand-over and the same seed, so the ONLY difference between the arms is where the pose comes from:
  full           (i)   the paper's silhouette search (wrapper_libero.Canonicalizer.calibrate_v2), unchanged
  full_refine    (ii)  EasyHeC-style refinement (Chen et al., "EasyHeC", RA-L 2023): gradient descent on a DENSE mask loss
                       from the SAME initial pose (the silhouette estimate (i)), on the SAME sweep masks. EasyHeC
                       back-propagates through a differentiable rasteriser; MuJoCo's renderer is not differentiable, so the
                       gradient here is a central finite difference of a soft-silhouette loss (both masks blurred, the
                       role a soft rasteriser's blur plays), with Adam, coarse to fine in the blur. The loss, the
                       initialisation and the optimiser follow EasyHeC; the gradient estimator does not.
  full_fiducial  (iii) a fiducial pose, the AnyCamVLA pose source: known 3D corners of four square markers lying on the
                       training table, projected into the deployed camera with the simulator's TRUE pose, perturbed by
                       Gaussian corner-localisation noise (sigma = FID_SIGMA_PX px at the 256 px observation), then
                       Perspective-n-Point (OpenCV) with the known intrinsics. Corners are never occluded ("oracle
                       corners"), which favours this source; a marker counts only when all four corners are in the image.
Nothing here changes a registered rule or an existing arm: `full` rows are produced exactly as before, and the two new
arms only replace (C_est, R_est) after calibrate_v2 has run. Every row logs `cam_est_err_cm_deg` of the pose it USED and,
for (ii) and (iii), `cam_silhouette_err_cm_deg` of the silhouette pose it started from or replaced.

Pure numpy / OpenCV; `refine` needs only the Canonicalizer methods _save/_restore/_set_camera/_robot_mask and its
arm_idx/grip_idx/sim/mres, so it is tested locally against a mock renderer (_wrapper_tests.py).
"""
import cv2
import numpy as np

FID_SIGMA_PX = 0.5                     # corner localisation noise of a square fiducial detector at 256 px (typical 0.1-0.5)
FID_SIDE = 0.06                        # marker side, metres
FID_OFFSETS = [(0.15, 0.15), (0.15, -0.15), (-0.15, 0.15), (-0.15, -0.15)]   # marker centres around the table pivot, m
REFINE_SIGMAS = (4.0, 2.0, 1.0)        # blur (px at the mask resolution) per stage, coarse to fine
REFINE_ITERS = 30                      # Adam steps per stage
REFINE_LR = (0.01, 0.01)               # Adam step: metres, radians (first stage; halved each stage)
REFINE_H = (0.004, 0.004)              # finite-difference step: metres, radians
ACCEPT = 0.8                           # refined pose used iff its loss < ACCEPT x the loss at the training pose


def _rotvec_to_mat(r):
    return cv2.Rodrigues(np.asarray(r, np.float64).reshape(3, 1))[0]


def _pose_error(C1, R1, C2, R2):
    rv = cv2.Rodrigues(np.asarray(R1 @ R2.T, np.float64))[0].ravel()
    return float(np.linalg.norm(np.asarray(C1) - np.asarray(C2)) * 100), float(np.rad2deg(np.linalg.norm(rv)))


# ---------------------------------------------------------------- (ii) EasyHeC-style dense refinement
def _soft(mask, sigma):
    m = (np.asarray(mask) > 0).astype(np.float32)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(m, (k, k), sigma)


def dense_loss(can, recs, C, R, sigma, soft_obs=None):
    """L(C, R) = (1 / V) sum_v mean_pixels (G_sigma * M_render_v(C, R) - G_sigma * M_obs_v)^2 over the V sweep poses,
    both masks at the sweep's mask resolution, G_sigma a Gaussian blur of sigma pixels. The caller restores the state."""
    soft_obs = soft_obs or [_soft(rec[2], sigma) for rec in recs]
    can._set_camera(C, R)
    total = 0.0
    for rec, so in zip(recs, soft_obs):
        can.sim.data.qpos[can.arm_idx] = rec[0]
        if rec[1] is not None:
            can.sim.data.qpos[can.grip_idx] = rec[1]
        can.sim.forward()
        ren = cv2.resize(can._robot_mask(can.mres).astype(np.uint8), (so.shape[1], so.shape[0]), interpolation=cv2.INTER_NEAREST)
        total += float(np.mean((_soft(ren, sigma) - so) ** 2))
    return total / len(recs)


def refine(can, recs, C0, R0, sigmas=REFINE_SIGMAS, iters=REFINE_ITERS, lr=REFINE_LR, h=REFINE_H):
    """Adam on x in R^6 (C = C0 + x[:3], R = exp([x[3:]]) R0) with a central-difference gradient of dense_loss; the best
    pose seen at the finest blur is returned. 2 x 6 + 1 renders of every sweep pose per step."""
    s = can._save()
    try:
        x = np.zeros(6)
        pose = lambda x: (C0 + x[:3], _rotvec_to_mat(x[3:]) @ R0)
        hh = np.array([h[0]] * 3 + [h[1]] * 3)
        for k, sg in enumerate(sigmas):
            so = [_soft(rec[2], sg) for rec in recs]
            f = lambda x: dense_loss(can, recs, *pose(x), sg, so)
            step = np.array([lr[0]] * 3 + [lr[1]] * 3) / 2 ** k
            m, v = np.zeros(6), np.zeros(6)
            best = (f(x), x.copy())
            for t in range(1, iters + 1):
                g = np.array([(f(x + hh[i] * e) - f(x - hh[i] * e)) / (2 * hh[i]) for i, e in enumerate(np.eye(6))])
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                x = x - step * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-12)
                fx = f(x)
                if fx < best[0]:
                    best = (fx, x.copy())
            x = best[1]
        C, R = pose(x)
        so = [_soft(rec[2], sigmas[-1]) for rec in recs]
        out = dict(refine_loss_init=dense_loss(can, recs, C0, R0, sigmas[-1], so),
                   refine_loss_final=dense_loss(can, recs, C, R, sigmas[-1], so),
                   refine_loss_nominal=dense_loss(can, recs, can.Cn, can.Rn, sigmas[-1], so))
    finally:
        can._restore(s)
    return C, R, out


# ---------------------------------------------------------------- (iii) fiducial (oracle-corner) pose, AnyCamVLA's source
def marker_corners(pivot, table_z, offsets=FID_OFFSETS, side=FID_SIDE):
    """World corners (4 per marker, (M, 4, 3)) of square markers lying flat on the table around `pivot`."""
    c = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    return np.array([[[pivot[0] + dx + a, pivot[1] + dy + b, table_z] for a, b in c] for dx, dy in offsets], float)


def _cv_from_mj(R):
    """MuJoCo camera axes (columns x right, y up, z back) -> OpenCV rotation world->camera (x right, y down, z forward)."""
    return np.diag([1.0, -1.0, -1.0]) @ R.T


def project_cv(X, C, R, f, H, W):
    """World points -> OpenCV pixels (u right, v down from the top-left pixel centre at 0) and depth, for a MuJoCo camera.
    The same pinhole as wrapper_libero.reproject (principal point at the image centre, square pixels)."""
    pc = (np.asarray(X, float).reshape(-1, 3) - C) @ _cv_from_mj(R).T
    uv = f * pc[:, :2] / pc[:, 2:3] + np.array([W / 2 - 0.5, H / 2 - 0.5])
    return uv, pc[:, 2]


def fiducial_pose(Ct, Rt, corners, f, H, W, sigma=FID_SIGMA_PX, rng=None):
    """PnP from the markers whose four corners all project inside the image and in front of the TRUE camera. Returns
    (C, R, info); (None, None, info) when no marker is fully visible."""
    rng = rng or np.random.default_rng(0)
    obj, img = [], []
    for M in corners:
        uv, z = project_cv(M, Ct, Rt, f, H, W)
        if (z > 1e-3).all() and (uv >= 0).all() and (uv[:, 0] <= W - 1).all() and (uv[:, 1] <= H - 1).all():
            obj.append(M)
            img.append(uv + sigma * rng.standard_normal(uv.shape))
    info = dict(fid_markers_visible=len(obj), fid_sigma_px=sigma)
    if not obj:
        return None, None, info
    K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]], float)
    P, Q = np.concatenate(obj).astype(np.float64), np.concatenate(img).astype(np.float64)
    flag = cv2.SOLVEPNP_IPPE if len(obj) == 1 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(P, Q, K, None, flags=flag)
    if not ok:
        return None, None, info
    if len(obj) > 1:
        rvec, tvec = cv2.solvePnPRefineLM(P, Q, K, None, rvec, tvec)
    Rcv = cv2.Rodrigues(rvec)[0]
    R = Rcv.T @ np.diag([1.0, -1.0, -1.0])                  # inverse of _cv_from_mj
    C = -Rcv.T @ tvec.ravel()
    uv, _ = project_cv(P, C, R, f, H, W)
    info["fid_reproj_px"] = float(np.sqrt(np.mean(np.sum((uv - Q) ** 2, 1))))
    return C, R, info


# ---------------------------------------------------------------- entry point used by sweep_run.prepare_full
def estimate(source, can, recs, obs_hw, table_z_default):
    """source 'refine' | 'fiducial'. Returns (C, R, applied, info). `can.C_est, can.R_est` hold the silhouette pose (i)."""
    import wrapper_libero as W
    Ct, Rt = can.true_camera()
    C0, R0 = can.C_est.copy(), can.R_est.copy()
    info = dict(pose_source=source, cam_silhouette_err_cm_deg=list(W.pose_error(C0, R0, Ct, Rt)))
    if source == "refine":
        C, R, o = refine(can, recs, C0, R0)
        info.update(o)
        moved = max(_pose_error(C, R, can.Cn, can.Rn)) > 0.5          # cm or deg away from the training pose
        applied = bool(moved and o["refine_loss_final"] < ACCEPT * o["refine_loss_nominal"])
    elif source == "refine2":
        # POST HOC 2026-09-25: two starts, the silhouette pose and the training pose; keep the refined pose whose OWN
        # dense loss is lower (self-measured, no simulator truth). Guards against a wrong basin of the silhouette search.
        C1, R1, o1 = refine(can, recs, C0, R0)
        C2, R2, o2 = refine(can, recs, can.Cn.copy(), can.Rn.copy())
        first = o1["refine_loss_final"] <= o2["refine_loss_final"]
        C, R, o = (C1, R1, o1) if first else (C2, R2, o2)
        info.update(o, refine_start="silhouette" if first else "training",
                    refine_loss_from_silhouette=o1["refine_loss_final"], refine_loss_from_training=o2["refine_loss_final"])
        moved = max(_pose_error(C, R, can.Cn, can.Rn)) > 0.5
        applied = bool(moved and o["refine_loss_final"] < ACCEPT * o["refine_loss_nominal"])
    elif source == "fiducial":
        H, Wd = obs_hw
        tz = can.table_z if getattr(can, "table_z", None) is not None else table_z_default
        corners = marker_corners(W.optical_pivot(can.Cn, can.Rn, tz), tz)
        C, R, o = fiducial_pose(Ct, Rt, corners, can._focal(H), H, Wd)
        info.update(o)
        applied = C is not None
    else:
        raise ValueError(source)
    if not applied:
        C, R = can.Cn.copy(), can.Rn.copy()
    return C, R, applied, info
