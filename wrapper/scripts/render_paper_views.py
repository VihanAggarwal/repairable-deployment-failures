"""POLICY-FREE renders for the paper's new 3D figures (Colab, MuJoCo offscreen under EGL, 512 x 512, nothing drawn on
any frame). No policy is loaded and nothing is scored; every number stored is read from the simulator.

    python render_paper_views.py --nominals <nominals_v3.json> --out_dir <dir> [--jobs scene_frusta,joint_offset_ghost,probe_ghosts]
    python render_paper_views.py --smoke --nominals <json> --out_dir <dir>      # one job at 256 px, prints the anchor check

Jobs (one .npz each in --out_dir; copy them to wrapper/paper/iclr/figs/renders/, make_image_figures.py reads them):
  scene_frusta   god's-eye view of one LIBERO-10 scene (task 1167, walls hidden), the TRAINING camera and the four Fig. 1
                 deployed cameras (orbit 20, pulled back 30%, orbit 45, turned away 35) as frusta: 3D centres, look-at
                 points and frustum corners in world metres AND projected into the overview image; plus each camera's own
                 clean 512 px view of the unperturbed scene.
  joint_offset_ghost  one joint-offset condition (sweep condition offset0.20, task 1167): the arm at the OFFSET start and at the
                 TRAINING start (q_nom), from the training camera and from the overview, with robot masks for ghosting.
  probe_ghosts   the 7 fingerprint probe poses (smc.PROBES, the cross the probe servo drives through) from the training
                 camera and from the overview, with robot masks and the end-effector position of each probe (world and
                 pixels). The per-probe ACTION vectors are never drawn on a frame: make_image_figures.py joins them from
                 the logged fingerprint rows (smc_<policy>.jsonl) and writes them as a SEPARATE transparent arrow layer
                 (one PNG per logged unit and frame, anchored at eef_px_training), so the tiles stay clean.

Frames are stored in DISPLAY orientation (row 0 = top; mujoco.Renderer output, NOT the upside-down robosuite buffer), key
`orientation` = "display", so the figure script must not turn them. Every anchor is a display pixel (u right, v down).
`anchor_check` = fraction of robot-geom centres whose projection lands on the rendered robot mask (dilated 3 px): below
0.9 means the projection convention is wrong and the anchors must not be used.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TASK = 1167                                                     # the Fig. 1 orbit task; a scene every figure already uses
FIG1_CAMERAS = ["orbit_az20", "pullback_1.3", "orbit_az45", "look_away_35"]
JOINT_COND = "offset0.20"
OVERVIEW = dict(az=55.0, el=28.0, scale=2.5)                    # same god's-eye pose as the design bundle's overviews
FRUSTUM_DEPTH = 0.30                                            # metres along the optical axis for the drawn frustum
HIDE = ("wall", "ceiling")                                      # geoms hidden in the god's-eye view (name contains)


# ---------------------------------------------------------------- pure geometry (tested locally, no simulator)
def project(X, C, R, fovy_deg, H, W=None):
    """World points X (N,3) -> display pixels (N,2) for a MuJoCo camera at C with axes R (columns: x right, y up, z back).
    Returns (uv, in_front). Square pixels; f = 0.5 H / tan(fovy / 2), fovy is the vertical field of view."""
    W = H if W is None else W
    X = np.atleast_2d(np.asarray(X, float))
    pc = (X - np.asarray(C, float)) @ np.asarray(R, float)      # = R^T (X - C) for each row
    z = -pc[:, 2]
    f = 0.5 * H / np.tan(np.deg2rad(fovy_deg) / 2)
    zs = np.where(z > 1e-9, z, np.nan)
    uv = np.stack([W / 2 + f * pc[:, 0] / zs, H / 2 - f * pc[:, 1] / zs], 1)
    return uv, z > 1e-9


def frustum(C, R, fovy_deg, depth=FRUSTUM_DEPTH):
    """World corners (4,3) of the image rectangle at `depth` in front of the camera: top-left, top-right, bottom-right,
    bottom-left as seen in the camera's own display image."""
    t = np.tan(np.deg2rad(fovy_deg) / 2) * depth
    pc = np.array([[-t, t, -depth], [t, t, -depth], [t, -t, -depth], [-t, -t, -depth]])
    return np.asarray(C, float) + pc @ np.asarray(R, float).T


# ---------------------------------------------------------------- simulator side (Colab only)
class Shot:
    """A 512 px renderer of its own (robosuite's offscreen buffer is too small for 512), sharing the env's model/data."""

    def __init__(self, env, res):
        import mujoco
        self.mj, self.res = mujoco, res
        self.sim = env.env.sim
        self.m, self.d = self.sim.model._model, self.sim.data._data
        self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), res)
        self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), res)
        self.r = mujoco.Renderer(self.m, res, res)
        self.opt = mujoco.MjvOption()
        self.hidden = []

    def hide_walls(self):
        mj = self.mj
        for g in range(self.m.ngeom):
            name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_GEOM, g) or ""
            if any(h in name.lower() for h in HIDE):
                self.hidden.append((g, int(self.m.geom_group[g])))
                self.m.geom_group[g] = 5
        self.opt.geomgroup[5] = 0
        return [self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, g) for g, _ in self.hidden]

    def show_walls(self):
        for g, grp in self.hidden:
            self.m.geom_group[g] = grp
        self.hidden = []
        self.opt.geomgroup[5] = 1

    def _scene(self, cam):
        try:
            self.r.update_scene(self.d, camera=cam, scene_option=self.opt)
        except TypeError:                   # an older mujoco without scene_option: walls stay visible, said once
            if not getattr(self, "_warned", False):
                print("note: this mujoco's Renderer has no scene_option; walls are NOT hidden in the overview", flush=True)
                self._warned = True
            self.r.update_scene(self.d, camera=cam)

    def rgb(self, cam="agentview"):
        self.mj.mj_forward(self.m, self.d)
        self._scene(cam)
        return self.r.render().copy()

    def robot_mask(self, geoms, cam="agentview"):
        self.mj.mj_forward(self.m, self.d)
        self.r.enable_segmentation_rendering()
        try:
            self._scene(cam)
            seg = self.r.render().copy()
        finally:
            self.r.disable_segmentation_rendering()
        return (seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], geoms)   # Renderer: (id, type)


def anchor_check(shot, can, C, R, mask):
    import cv2
    pts = shot.d.geom_xpos[can.robot_geoms]
    uv, ok = project(pts, C, R, can.fovy, shot.res)
    m = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    hit = [m[int(v), int(u)] for (u, v), k in zip(uv, ok) if k and 0 <= u < shot.res and 0 <= v < shot.res]
    return float(np.mean(hit)) if hit else 0.0


def overview_pose(can):
    import wrapper_libero as W
    pivot = W.optical_pivot(can.Cn, can.Rn, can.table_z if can.table_z is not None else W.TABLE_Z)
    C, R = W.orbit(can.Cn, can.Rn, pivot, OVERVIEW["az"], OVERVIEW["el"], OVERVIEW["scale"])
    return C, R, pivot


def at_camera(shot, can, C, R, fn):
    s = can._save()
    try:
        can._set_camera(C, R)
        return fn()
    finally:
        can._restore(s)


def eef(env):
    return np.asarray(env.env._get_observations(force_update=True)["robot0_eef_pos"], float)


def job_scene_frusta(ctx):
    import wrapper_libero as W
    env, t = ctx["env"](), ctx["task"]
    out = {}
    try:
        ctx["R"].start(env, ctx["suite"], t, ctx["S"].BY_NAME["nominal"], seed=t["task_id"])
        can = W.Canonicalizer(env, ctx["nominal"])
        shot = Shot(env, ctx["res"])
        Cov, Rov, pivot = overview_pose(can)
        out["hidden_geoms"] = np.array(shot.hide_walls(), dtype=object)
        out["overview"] = at_camera(shot, can, Cov, Rov, shot.rgb)
        out["overview_robot"] = at_camera(shot, can, Cov, Rov, lambda: shot.robot_mask(can.robot_geoms))
        out["anchor_check"] = at_camera(shot, can, Cov, Rov, lambda: anchor_check(shot, can, Cov, Rov, out["overview_robot"]))
        shot.show_walls()
        out["training_view"] = at_camera(shot, can, can.Cn, can.Rn, shot.rgb)
        cams = {"training": (can.Cn, can.Rn)}
        env.close()
        for c in FIG1_CAMERAS:                      # each deployed camera from a fresh env (apply_scene edits the model)
            env = ctx["env"]()
            ctx["R"].start(env, ctx["suite"], t, ctx["S"].BY_NAME[c], seed=t["task_id"])
            c2 = W.Canonicalizer(env, ctx["nominal"])
            cams[c] = c2.true_camera()
            out[f"view_{c}"] = Shot(env, ctx["res"]).rgb()
            out[f"shift_{c}"] = np.array(W.pose_error(*cams[c], can.Cn, can.Rn), float)       # (cm, deg) vs training
            env.close()
        env = None
        for k, (C, R) in cams.items():
            corners = frustum(C, R, can.fovy)
            look = W.optical_pivot(C, R, can.table_z if can.table_z is not None else W.TABLE_Z)
            uv, _ = project(np.vstack([C[None], look[None], corners]), Cov, Rov, can.fovy, ctx["res"])
            out[f"cam_{k}_world"] = np.vstack([C[None], look[None], corners])                 # centre, look-at, 4 corners
            out[f"cam_{k}_px"] = uv
            out[f"cam_{k}_R"] = np.asarray(R, float)
        out.update(overview_C=Cov, overview_R=Rov, overview_pivot=pivot, fovy=can.fovy, cameras=np.array(list(cams), dtype=object))
    finally:
        if env is not None:
            env.close()
    return out


def job_joint_offset(ctx):
    import wrapper_libero as W
    env, t = ctx["env"](), ctx["task"]
    out = {}
    try:
        cond = ctx["S"].BY_NAME[JOINT_COND]
        ctx["R"].start(env, ctx["suite"], t, cond, seed=t["task_id"])
        can = W.Canonicalizer(env, ctx["nominal"])
        shot = Shot(env, ctx["res"])
        Cov, Rov, _ = overview_pose(can)

        def grab(tag):
            shot.show_walls()                               # training-camera tiles keep the walls, as the policy sees them
            out[f"{tag}_training_view"] = shot.rgb()
            out[f"{tag}_training_robot"] = shot.robot_mask(can.robot_geoms)
            shot.hide_walls()
            out[f"{tag}_overview"] = at_camera(shot, can, Cov, Rov, shot.rgb)
            out[f"{tag}_overview_robot"] = at_camera(shot, can, Cov, Rov, lambda: shot.robot_mask(can.robot_geoms))
            p = eef(env)
            out[f"{tag}_eef_world"] = p
            out[f"{tag}_eef_px_training"] = project(p, can.Cn, can.Rn, can.fovy, ctx["res"])[0][0]
            out[f"{tag}_eef_px_overview"] = project(p, Cov, Rov, can.fovy, ctx["res"])[0][0]
            out[f"{tag}_q"] = shot.d.qpos[can.arm_idx].copy()

        grab("offset")
        out["anchor_check"] = anchor_check(shot, can, can.Cn, can.Rn, out["offset_training_robot"])
        shot.d.qpos[can.arm_idx] = can.q_nom                # what nominal_start does: the arm PLACED at the training start
        shot.d.qvel[list(can.robot._ref_joint_vel_indexes)] = 0.0
        grab("nominal")
        out["joint_dev_rad"] = np.abs(out["offset_q"] - out["nominal_q"])
        out.update(condition=JOINT_COND, overview_C=Cov, overview_R=Rov, fovy=can.fovy)
    finally:
        env.close()
    return out


def job_probe_ghosts(ctx):
    import smc
    import wrapper_libero as W
    env, t = ctx["env"](), ctx["task"]
    out = {}
    try:
        obs, _ = ctx["R"].start(env, ctx["suite"], t, ctx["S"].BY_NAME["nominal"], seed=t["task_id"])
        can = W.Canonicalizer(env, ctx["nominal"])
        shot = Shot(env, ctx["res"])
        Cov, Rov, _ = overview_pose(can)
        p0, q0 = can.nominal_eef()
        imgs, masks, oimgs, omasks, P, uv_t, uv_o = [], [], [], [], [], [], []
        for k, off in enumerate(smc.PROBES):
            obs, _ = can.servo(obs, p0 + off, q0)           # the probe servo, exactly as smc_probe.probe_unit drives it
            shot.show_walls()
            imgs.append(shot.rgb()); masks.append(shot.robot_mask(can.robot_geoms))
            shot.hide_walls()
            oimgs.append(at_camera(shot, can, Cov, Rov, shot.rgb))
            omasks.append(at_camera(shot, can, Cov, Rov, lambda: shot.robot_mask(can.robot_geoms)))
            p = np.asarray(obs["robot0_eef_pos"], float)
            P.append(p)
            uv_t.append(project(p, can.Cn, can.Rn, can.fovy, ctx["res"])[0][0])
            uv_o.append(project(p, Cov, Rov, can.fovy, ctx["res"])[0][0])
        out["anchor_check"] = anchor_check(shot, can, can.Cn, can.Rn, masks[-1])
        out.update(training_views=np.stack(imgs), training_robot=np.stack(masks), overviews=np.stack(oimgs),
                   overview_robot=np.stack(omasks), P=np.stack(P), probe_offsets=np.stack(smc.PROBES), p0=p0,
                   eef_px_training=np.stack(uv_t), eef_px_overview=np.stack(uv_o), overview_C=Cov, overview_R=Rov,
                   camera_C=can.Cn, camera_R=can.Rn, fovy=can.fovy)
    finally:
        env.close()
    return out


JOBS = {"scene_frusta": job_scene_frusta, "joint_offset_ghost": job_joint_offset, "probe_ghosts": job_probe_ghosts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominals", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--jobs", default=",".join(JOBS))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--smoke", action="store_true", help="probe_ghosts only, 256 px, printed check, nothing kept")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import sweep_conditions as S
    import sweep_run as R
    import wrapper_libero as W
    from calib_check import make_env
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    task = {x["task_id"]: x for x in S.base_tasks(suite, 10)}[TASK]
    ctx = dict(suite=suite, S=S, R=R, task=task, nominal=nominals[task["scene"]], res=256 if a.smoke else a.res,
               env=lambda: make_env(suite, TASK))
    os.makedirs(a.out_dir, exist_ok=True)
    for name in (["probe_ghosts"] if a.smoke else a.jobs.split(",")):
        path = os.path.join(a.out_dir, name + ("_smoke" if a.smoke else "") + ".npz")
        if os.path.exists(path) and not a.redo and not a.smoke:
            print(f"{name}: exists, skipped ({path})", flush=True)
            continue
        d = JOBS[name](ctx)
        d.update(task_id=TASK, orientation="display", res=ctx["res"])
        chk = float(d["anchor_check"])
        np.savez_compressed(path, **{k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in d.items()})
        print(f"{name}: wrote {path} | anchor check {chk:.2f} (need >= 0.90) | "
              + json.dumps({k: np.round(np.asarray(v, float), 2).tolist() for k, v in d.items() if k.startswith("shift_")}), flush=True)
        if chk < 0.9:
            print(f"WARNING {name}: anchors do not land on the robot; do not use them, send this line to the agent", flush=True)


if __name__ == "__main__":
    main()
