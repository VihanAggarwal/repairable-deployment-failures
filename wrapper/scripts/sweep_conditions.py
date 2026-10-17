"""Graded perturbation grid for the within-type identifiability test.

Every condition is applied programmatically to an UNPERTURBED LIBERO-10 scene (a LIBERO-Plus language variant with the
default view and robot start, run with the original instruction), so strength is controlled per type:
  camera      orbit azimuth 10/20/30/45 deg, pull-back x1.3, and look-away tilts 35/55 deg (arm leaves the frame)
  light       directional-light azimuth 90/180 deg, grazing elevation, diffuse gain 0.4/1.8, warm tint
  noise       Gaussian sensor noise sigma 5/10/20/40 on both cameras
  joint       start-pose offsets with max |dq| 0.10/0.20/0.35 rad
  background  table recolour red/dark (no coupling to the robot), and a "coloured room" (table recolour + matching
              light tint, which does reach the robot's appearance)
`override` marks conditions where the category label and the self-identifiability score are expected to disagree:
look-away camera shifts are labelled invertible but the arm is not visible; the coloured room is labelled background
(capability) but the colour cast is visible on the robot. MuJoCo has no global illumination, so colour bleed from a
coloured room is emulated by tinting the lights; this is disclosed as a simulation stand-in.
"""
import re

import numpy as np

INVERTIBLE_LABEL = {"nominal": 1, "camera": 1, "light": 1, "noise": 1, "joint": 1, "background": 0}


def C(family, name, level, override=False, **params):
    return dict(family=family, name=name, level=float(level), override=bool(override), params=params)


CONDITIONS = [
    C("nominal", "nominal", 0),
    C("camera", "orbit_az10", 10, az=10), C("camera", "orbit_az20", 20, az=20),
    C("camera", "orbit_az30", 30, az=30), C("camera", "orbit_az45", 45, az=45),
    C("camera", "pullback_1.3", 1.3, scale=1.3),
    C("camera", "look_away_35", 35, override=True, tilt=35), C("camera", "look_away_55", 55, override=True, tilt=55),
    C("light", "dir_az90", 90, dir_az=90), C("light", "dir_az180", 180, dir_az=180),
    C("light", "grazing", 1, grazing=True),
    C("light", "gain_0.4", 0.4, gain=0.4), C("light", "gain_1.8", 1.8, gain=1.8),
    C("light", "tint_warm", 1, tint=[1.35, 0.85, 0.6]),
    C("noise", "sigma5", 5, sigma=5), C("noise", "sigma10", 10, sigma=10),
    C("noise", "sigma20", 20, sigma=20), C("noise", "sigma40", 40, sigma=40),
    C("joint", "offset0.10", 0.10, offset=0.10), C("joint", "offset0.20", 0.20, offset=0.20),
    C("joint", "offset0.35", 0.35, offset=0.35),
    C("background", "table_red", 1, table_rgb=[0.8, 0.1, 0.1]),
    C("background", "table_dark", 1, table_rgb=[0.08, 0.08, 0.08]),
    C("background", "colored_room", 1, override=True, table_rgb=[0.8, 0.1, 0.1], tint=[1.3, 0.8, 0.8]),
]
# boundary conditions (--extended): levels between the base grid's steps, where a predictor is actually tested.
# Added 2026-09-18 after the OFT development analysis; every family runs them, none is tuned on them.
EXTENDED = [
    C("camera", "orbit_az15", 15, az=15), C("camera", "orbit_az25", 25, az=25), C("camera", "orbit_az38", 38, az=38),
    C("camera", "look_away_15", 15, override=True, tilt=15), C("camera", "look_away_25", 25, override=True, tilt=25),
    C("light", "gain_1.4", 1.4, gain=1.4),
    C("noise", "sigma30", 30, sigma=30),
    C("joint", "offset0.50", 0.50, offset=0.50),
]
BY_NAME = {c["name"]: c for c in CONDITIONS + EXTENDED}
DEFAULT_VIEW = ("0", "0", "100", "0", "0")


def base_tasks(suite, n):
    """n unperturbed base tasks spread over scenes: a language variant with the default view and robot start,
    paired with the ORIGINAL LIBERO-10 instruction recovered from the task name."""
    from plus_common import categorize
    from wrapper_libero import scene_of
    first = {}
    for i in range(suite.n_tasks):
        nm = suite.get_task(i).name
        if categorize(nm) != "language":
            continue
        v = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)", nm)
        s = re.search(r"_initstate_(\d+)", nm)
        if (v and v.groups() != DEFAULT_VIEW) or (s and s.group(1) != "0"):
            continue
        base = re.sub(r"_(view_\d+_\d+_\d+_\d+_\d+|initstate_\d+|language_\d+)", "", nm)
        first.setdefault(base, i)
    by_scene = {}
    for b in sorted(first):
        by_scene.setdefault(scene_of(b + "_"), []).append(b)
    order, k = [], 0
    while len(order) < len(first):
        for sc in sorted(by_scene):
            if k < len(by_scene[sc]):
                order.append(by_scene[sc][k])
        k += 1
    out = []
    for b in order[:n]:
        instr = re.sub(r"^[A-Z]+(?:_[A-Z]+)*_SCENE\d+_", "", b).replace("_", " ")
        out.append(dict(task_id=first[b], base=b, scene=scene_of(b + "_"), instruction=instr))
    return out


def center_depth(sim, cam, res=64):
    _, depth = sim.render(width=res, height=res, camera_name=cam, depth=True)
    m = sim.model
    extent = float(m.stat.extent)
    near, far = float(m.vis.map.znear) * extent, float(m.vis.map.zfar) * extent
    z = near / (1.0 - depth * (1.0 - near / far))
    c = res // 2
    return float(np.median(z[c - 4:c + 4, c - 4:c + 4]))


def _table_geoms(m):
    out = []
    for g in range(m.ngeom):
        gname = (m.geom_id2name(g) or "").lower()
        try:
            bname = (m.body_id2name(int(m.geom_bodyid[g])) or "").lower()
        except Exception:
            bname = ""
        if "table" in gname or "table" in bname:
            out.append(g)
    return out


def apply_scene(env, cond, cam="agentview", scale=1.0):
    """Model-level perturbation (camera, lights, table colour), `scale` in [0, 1+] multiplies the strength (used by
    the finite-difference score). Returns a dict describing what was changed."""
    import wrapper_libero as W
    sim = env.env.sim
    m = sim.model
    p = cond["params"]
    info = {}
    cid = m.camera_name2id(cam)
    C0, R0 = m.cam_pos[cid].copy(), W.quat_wxyz_to_mat(m.cam_quat[cid])
    if "tilt" in p:
        R = R0 @ W.rotvec_to_mat(np.array([np.deg2rad(p["tilt"] * scale), 0.0, 0.0]))
        m.cam_quat[cid][:] = W.mat_to_quat_wxyz(R)
        info["cam_shift_cm_deg"] = W.pose_error(C0, R, C0, R0)
    elif "az" in p or "scale" in p:
        pivot = C0 + center_depth(sim, cam) * (-R0[:, 2])
        C, R = W.orbit(C0, R0, pivot, p.get("az", 0) * scale, 0.0, 1.0 + (p.get("scale", 1.0) - 1.0) * scale)
        m.cam_pos[cid][:] = C
        m.cam_quat[cid][:] = W.mat_to_quat_wxyz(R)
        info["cam_shift_cm_deg"] = W.pose_error(C, R, C0, R0)
    if m.nlight > 0:
        if "dir_az" in p:
            m.light_dir[:] = (W.rot_z(p["dir_az"] * scale) @ m.light_dir.T).T
        if p.get("grazing"):
            d = m.light_dir.copy()
            horiz = np.where(np.linalg.norm(d[:, :2], axis=1, keepdims=True) > 1e-3, d[:, :2], np.array([[1.0, 0.0]]))
            horiz = horiz / np.linalg.norm(horiz, axis=1, keepdims=True)
            target = np.concatenate([horiz * 0.99, np.full((len(d), 1), -0.12)], 1)
            new = (1 - scale) * d + scale * target
            m.light_dir[:] = new / np.linalg.norm(new, axis=1, keepdims=True)
        if "gain" in p:
            m.light_diffuse[:] = m.light_diffuse * (1.0 + (p["gain"] - 1.0) * scale)
        if "tint" in p:
            m.light_diffuse[:] = m.light_diffuse * (1.0 + (np.asarray(p["tint"]) - 1.0) * scale)
    if "table_rgb" in p:
        for g in _table_geoms(m):
            tgt = np.asarray(p["table_rgb"])
            m.geom_rgba[g, :3] = m.geom_rgba[g, :3] + (tgt - m.geom_rgba[g, :3]) * min(scale, 1.0)
            mat = int(m.geom_matid[g])
            if mat >= 0:
                m.mat_rgba[mat, :3] = m.mat_rgba[mat, :3] + (tgt - m.mat_rgba[mat, :3]) * min(scale, 1.0)
        info["table_geoms"] = len(_table_geoms(m))
    sim.forward()
    return info


def joint_offset_vector(cond, seed, n=7):
    v = np.random.default_rng(seed).normal(size=n)
    return v / np.abs(v).max() * cond["params"]["offset"]


def apply_joint(env, cond, seed):
    robot = env.env.robots[0]
    idx = list(robot._ref_joint_pos_indexes)
    sim = env.env.sim
    sim.data.qpos[idx] = sim.data.qpos[idx] + joint_offset_vector(cond, seed, len(idx))
    sim.data.qvel[:] = 0.0
    sim.forward()
    try:
        robot.controller.update(force=True)
        robot.controller.reset_goal()
    except Exception:
        pass


def add_noise(obs, sigma, rng):
    if not sigma:
        return obs
    for k in ("agentview_image", "robot0_eye_in_hand_image"):
        if k in obs:
            obs[k] = np.clip(obs[k].astype(np.float32) + rng.normal(0.0, sigma, obs[k].shape), 0, 255).astype(np.uint8)
    return obs
