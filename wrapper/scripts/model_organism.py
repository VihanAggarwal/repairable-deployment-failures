"""A transparent, image-driven reaching policy (a "model organism") for testing self-motion consistency end to end in
the real simulator, WITHOUT a learned policy and with ground truth for everything.

The organism sees only what a camera policy sees: where the target object and its own gripper appear in the agentview
image (pixel centroids). It converts pixels to 3D by assuming the TRAINING camera, which is what a frozen visuomotor
policy implicitly does, and commands a saturated displacement toward the target. Two archetypes:
  relative   gripper AND target located in the image: a moved camera distorts the perceived displacement differently
             at every end-effector position, so the action rays stop meeting (SMC should detect it)
  absolute   target located in the image, gripper from proprioception: a moved camera shifts the perceived target, the
             rays still meet, at the wrong point (SMC is blind by construction: the stated boundary of the method)
The wrapper's camera channel (self-calibration + reprojection into the training view) is applied to the label image
exactly as it is to RGB, so "corrected" means what it means for a real policy.

    MUJOCO_GL=glfw python model_organism.py --nominals <json> --out <jsonl> [--n_tasks 3] [--archetype relative]

Per unit: SMC residual on the raw and the corrected frame at smc.PROBES, then a reaching rollout under each, with
success = end-effector within 3 cm of the point 10 cm above the true target.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smc  # noqa: E402

REACH_UP, TOL, STEPS = 0.10, 0.03, 160


class Organism:
    def __init__(self, env, can, W, archetype):
        self.env, self.can, self.W, self.arch = env, can, W, archetype
        sim = can.sim
        m = sim.model
        name = env.env.obj_of_interest[0]
        self.t_bodies = [b for b in range(m.nbody) if can._body_name(b).startswith(name)]
        assert self.t_bodies, f"target body {name} not found"
        self.t_geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.t_bodies])
        self.correct = False

    def target_true(self):
        return np.mean([self.can.sim.data.body_xpos[b] for b in self.t_bodies], 0)

    def _labels(self, res=256):
        """Raw-buffer label image at the TRUE camera: 1 target, 2 gripper; plus depth."""
        can = self.can
        seg = can.sim.render(width=res, height=res, camera_name=can.cam, segmentation=True)
        isg = seg[..., 0] == int(can.mj.mjtObj.mjOBJ_GEOM)
        lab = np.zeros(seg.shape[:2], np.uint8)
        lab[isg & np.isin(seg[..., 1], self.t_geoms)] = 1
        lab[isg & np.isin(seg[..., 1], can.gripper_geoms)] = 2
        z, _ = can._depth_and_mask(res)
        return lab, z

    def perceive(self, obs):
        """Perceived (target, gripper) world points under the ASSUMED training camera, from pixel centroids."""
        can, W = self.can, self.W
        lab, z = self._labels()
        H = lab.shape[0]
        f = can._focal(H)
        if self.correct:                          # the wrapper's camera channel, applied to the label image
            rgb = np.stack([lab * 100] * 3, -1).astype(np.uint8)
            out, _ = W.reproject(rgb, z, f, can.C_est, can.R_est, can.Cn, can.Rn, inpaint=False)
            lab = (np.round(out[..., 0] / 100.0)).astype(np.uint8)
        eef = np.asarray(obs["robot0_eef_pos"], float)
        h = {1: float(self.target_true()[2]), 2: float(eef[2])}           # heights the policy is assumed to know
        pts = {}
        for k in (1, 2):
            rr, cc = np.nonzero(lab == k)
            if len(rr) < 4:
                return None
            d = np.array([(cc.mean() + 0.5 - H / 2) / f, (rr.mean() + 0.5 - H / 2) / f, -1.0])
            dw = can.Rn @ d
            if abs(dw[2]) < 1e-6:
                return None
            pts[k] = can.Cn + (h[k] - can.Cn[2]) / dw[2] * dw
        return pts[1], pts[2], eef

    def act(self, obs):
        p = self.perceive(obs)
        if p is None:
            return np.zeros(7)
        t, g, eef = p
        v = (t + np.array([0, 0, REACH_UP])) - ((g + np.array([0, 0, 0.0])) if self.arch == "relative" else eef)
        if self.arch == "relative":
            v[2] = (t[2] + REACH_UP) - eef[2]     # height is proprioceptive in both archetypes; the image gives x and y
        n = np.linalg.norm(v)
        a = np.zeros(7)
        a[:3] = 0.9 * v / n if n > 0.04 else v / 0.04 * 0.9
        a[6] = -1.0
        return a


def unit(env, suite, task, cond, nominal, archetype, R, S, W):
    seed = task["task_id"]
    obs, info = R.start(env, suite, task, cond, seed)
    can = W.Canonicalizer(env, nominal)
    Ct, Rt = can.true_camera()
    obs, recs = can.sweep(obs)
    cal = can.calibrate_v2(recs)
    C_est, R_est = can.C_est.copy(), can.R_est.copy()
    org = Organism(env, can, W, archetype)
    p0, q0 = can.nominal_eef()
    P, A = [], {False: [], True: []}
    for k, off in enumerate(smc.PROBES):
        if k:
            obs, _ = can.servo(obs, p0 + off, q0)
        P.append(np.asarray(obs["robot0_eef_pos"], float).tolist())
        for corr in (False, True):
            org.correct = corr
            A[corr].append(org.act(obs)[:3].tolist())
    out = dict(true_shift=list(W.pose_error(Ct, Rt, can.Cn, can.Rn)), est_err=list(W.pose_error(C_est, R_est, Ct, Rt)),
               cal_applied=bool(cal["applied"]), P=P, A_raw=A[False], A_cor=A[True],
               cam_est_pose=[*C_est.tolist(), *W.mat_to_rotvec(R_est).tolist()],
               cam_nom_pose=[*can.Cn.tolist(), *W.mat_to_rotvec(can.Rn).tolist()],
               smc_raw=smc.triangulate(P, A[False]), smc_cor=smc.triangulate(P, A[True]))
    out["pick"] = smc.compare(out["smc_raw"], out["smc_cor"])
    for corr in (False, True):                    # reaching rollouts from a fresh, identical start
        obs, _ = R.start(env, suite, task, cond, seed)
        can2 = W.Canonicalizer(env, nominal)
        can2.C_est, can2.R_est = C_est, R_est
        o2 = Organism(env, can2, W, archetype)
        o2.correct = corr
        goal = o2.target_true() + np.array([0, 0, REACH_UP])
        best = np.inf
        for _ in range(STEPS):
            obs, _, _, _ = env.step(o2.act(obs).tolist())
            best = min(best, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - goal)))
            if best < TOL:
                break
        out["success_cor" if corr else "success_raw"] = bool(best < TOL)
        out["reach_cm_cor" if corr else "reach_cm_raw"] = best * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--archetype", default="relative", choices=["relative", "absolute"])
    ap.add_argument("--families", default="nominal,camera")
    a = ap.parse_args()
    import sweep_conditions as S
    import sweep_run as R
    import wrapper_libero as W
    from calib_check import make_env
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    tasks = S.base_tasks(suite, a.n_tasks)
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    conds = [c for c in S.CONDITIONS + S.EXTENDED if c["family"] in a.families.split(",")]
    done = set()
    if os.path.exists(a.out):
        done = {(r["task_id"], r["cond"], r["archetype"]) for r in map(json.loads, filter(str.strip, open(a.out)))}
    todo = [(c, t) for c in conds for t in tasks if (t["task_id"], c["name"], a.archetype) not in done]
    print(f"organism ({a.archetype}): {len(todo)} units", flush=True)
    with open(a.out, "a") as f:
        for n, (cond, t) in enumerate(todo):
            t0 = time.time()
            env = make_env(suite, t["task_id"])
            try:
                row = unit(env, suite, t, cond, nominals[t["scene"]], a.archetype, R, S, W)
            except Exception as e:                       # a target out of view or an unreachable probe: record, move on
                row = dict(error=repr(e)[:200])
            finally:
                env.close()
            row = dict(task_id=t["task_id"], cond=cond["name"], family=cond["family"], archetype=a.archetype, **row)
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            if "error" in row:
                print(f"{t['task_id']} {cond['name']}: {row['error']}", flush=True)
                continue
            print(f"{t['task_id']:5d} {cond['name']:<14} shift {row['true_shift'][0]:5.1f}cm/{row['true_shift'][1]:4.1f}deg "
                  f"cal err {row['est_err'][0]:4.1f}cm | SMC raw {row['smc_raw']['residual_deg']:5.1f} cor "
                  f"{row['smc_cor']['residual_deg']:5.1f} deg -> {row['pick']:<3} | reach raw {row['reach_cm_raw']:5.1f}cm "
                  f"{'Y' if row['success_raw'] else '.'}  cor {row['reach_cm_cor']:5.1f}cm {'Y' if row['success_cor'] else '.'} "
                  f"| {time.time() - t0:.0f}s ({n + 1}/{len(todo)})", flush=True)


if __name__ == "__main__":
    main()
