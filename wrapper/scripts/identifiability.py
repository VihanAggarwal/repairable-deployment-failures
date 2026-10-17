"""Self-identifiability of LIBERO-Plus perturbations: can the robot measure a perturbation from its own body?
No policy is run.

  local  per scene: finite-difference sensitivity of the robot's SELF-observation (silhouette, body colour,
         proprioception, image noise floor) to each perturbation family's parameters, and the singular values of
         the camera block with 1 vs 3 sweep poses (how well-conditioned self-calibration is).
  tasks  per task in the evaluation subset: the self-measured shift on every channel and an identifiability index
         (>1 means at least one channel sees the perturbation above its noise floor). With --wrapper and --baseline,
         tests whether identifiability predicts which baseline failures the wrapper recovers.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrapper_libero as W  # noqa: E402
from plus_common import load_rows  # noqa: E402

FLOOR = dict(silhouette_px=0.5, body_colour=2.0, proprio_rad=0.005, noise=1.0)   # per-channel noise floors
TASK_THR = dict(joint=0.05, silhouette=1.5, photo_shift=15.0, photo_gain=0.25)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ local sensitivity
def self_obs(can, recs, res=96, arm_offset=None):
    import cv2
    masks, cols, blur = [], [], []
    for rec in recs:
        q = rec[0] if arm_offset is None else rec[0] + arm_offset
        can.sim.data.qpos[can.arm_idx] = q
        if rec[1] is not None:
            can.sim.data.qpos[can.grip_idx] = rec[1]
        can.sim.forward()
        mk = can._robot_mask(res)
        rgb = can.sim.render(width=res, height=res, camera_name=can.cam)
        px = rgb[mk > 0].reshape(-1, 3).astype(np.float64)
        masks.append(mk)
        cols.append(px.mean(0) if len(px) else np.zeros(3))
        blur.append(cv2.GaussianBlur(mk.astype(np.float32), (0, 0), 2.0).ravel())
    joints = np.array([rec[0] if arm_offset is None else rec[0] + arm_offset for rec in recs])
    return masks, np.array(cols), joints, np.concatenate(blur)


def compare(base, pert):
    m0, c0, j0, b0 = base
    m1, c1, j1, b1 = pert
    sil = float(np.mean([W.chamfer(a, b) if (a.any() and b.any()) else 0.0 for a, b in zip(m0, m1)]))
    return dict(silhouette_px=sil, body_colour=float(np.abs(c1 - c0).max()),
                proprio_rad=float(np.abs(j1 - j0).max())), b1 - b0


def local_scene(env, obs, nominal):
    can = W.Canonicalizer(env, nominal)
    obs, recs = can.sweep(obs)
    m, sim = can.sim.model, can.sim
    s = can._save()
    light0, rgba0 = m.light_diffuse.copy(), m.geom_rgba.copy()
    C0, R0 = can.true_camera()

    def undo():
        m.light_diffuse[:] = light0
        m.geom_rgba[:] = rgba0
        can._restore(s)

    base = self_obs(can, recs)
    fam = defaultdict(list)          # family -> list of (param, unit, channels)
    cam_cols = {1: [], 3: []}

    def record(family, param, unit, pert, jac=False):
        ch, dvec = compare(base, pert)
        fam[family].append(dict(param=param, unit=unit, **ch))
        if jac:
            per = len(dvec) // len(recs)
            cam_cols[3].append(dvec)
            cam_cols[1].append(dvec[:per])

    for i, ax in enumerate("xyz"):
        d = np.zeros(3); d[i] = 0.01
        can._set_camera(C0 + d, R0)
        record("camera", f"pos_{ax}", "1 cm", self_obs(can, recs), jac=True)
        undo()
    for i, ax in enumerate("xyz"):
        r = np.zeros(3); r[i] = np.deg2rad(1.0)
        can._set_camera(C0, W.rotvec_to_mat(r) @ R0)
        record("camera", f"rot_{ax}", "1 deg", self_obs(can, recs), jac=True)
        undo()
    if m.nlight > 0:
        m.light_diffuse[:] = np.clip(light0 * 1.1, 0, None)
        record("light", "diffuse", "+10%", self_obs(can, recs))
        undo()
    for k in range(len(can.arm_idx)):
        off = np.zeros(len(can.arm_idx)); off[k] = 0.02
        record("robot_init", f"joint_{k}", "0.02 rad", self_obs(can, recs, arm_offset=off))
        undo()
    robot_q = set(int(sim.model.jnt_qposadr[j]) for j in range(sim.model.njnt)
                  if can._body_name(int(sim.model.jnt_bodyid[j])).startswith(W.ROBOT_PREFIXES))
    free = [int(sim.model.jnt_qposadr[j]) for j in range(sim.model.njnt)
            if int(sim.model.jnt_type[j]) == 0 and int(sim.model.jnt_qposadr[j]) not in robot_q]
    for adr in free[:4]:
        for ax in range(2):
            sim.data.qpos[adr + ax] += 0.02
            sim.forward()
            record("layout", f"object_qpos{adr}_{'xy'[ax]}", "2 cm", self_obs(can, recs))
            undo()
    table = [g for g in range(m.ngeom) if "table" in (m.geom_id2name(g) or "").lower()]
    if table:
        m.geom_rgba[table, :3] = np.clip(rgba0[table, :3] + 0.1, 0, 1)
        record("background", "table_rgb", "+0.1", self_obs(can, recs))
        undo()
    img = obs["agentview_image"]
    lvl0 = W.noise_level(img)
    noisy = np.clip(img + np.random.default_rng(0).normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    fam["sensor_noise"].append(dict(param="gaussian", unit="sigma 5", noise=W.noise_level(noisy) - lvl0))
    undo()

    sv = {}
    for n, cols in cam_cols.items():
        if cols:
            svals = np.linalg.svd(np.stack(cols, 1), compute_uv=False)
            sv[f"{n}_pose"] = dict(singular_values=svals.tolist(), condition=float(svals[0] / max(svals[-1], 1e-9)))
    return dict(families=fam, camera_svd=sv)


def identifiable(entry):
    return any(entry.get(k, 0.0) > f for k, f in FLOOR.items())


# ------------------------------------------------------------------ per-task self-measured shift
def task_signals(env, obs, nominal):
    can = W.Canonicalizer(env, nominal)
    jd = can.joint_dev()
    if jd > TASK_THR["joint"]:
        p, q = can.nominal_eef()
        obs, _ = can.servo(obs, p, q, max_steps=200)
    obs, recs = can.sweep(obs)
    dts = [(W.edge_dt(rec[2]), rec[2]) for rec in recs]
    s = can._save()
    try:
        sil = can._score(recs, dts, can.Cn, can.Rn)
    finally:
        can._restore(s)
    ph = can.photometric_fit(obs, nominal)
    nz = W.noise_level(obs["agentview_image"])
    ref = nominal["noise"]["agentview_image"]
    gain_dev = max(abs(g - 1.0) for g in ph["photo_gain"])
    parts = dict(joint=jd / TASK_THR["joint"], silhouette=sil / TASK_THR["silhouette"],
                 photometric=max(ph["photo_mean_shift"] / TASK_THR["photo_shift"], gain_dev / TASK_THR["photo_gain"]),
                 noise=nz / (2.0 * ref + 1.0))
    return dict(joint_dev=jd, silhouette_px=sil, photo_mean_shift=ph["photo_mean_shift"], photo_gain_dev=gain_dev,
                noise_level=nz, index_parts=parts, index=max(parts.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["local", "tasks"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--nominals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_cat", type=int, default=30)
    ap.add_argument("--wrapper", default="")
    ap.add_argument("--baseline", default="")
    a = ap.parse_args()

    from libero.libero import benchmark
    from calib_check import make_env, start
    from plus_common import categorize
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    sub = json.load(open(a.subset))
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.mode == "local":
        res = {}
        for i in range(suite.n_tasks):
            name = suite.get_task(i).name
            scene = W.scene_of(name)
            if scene in res or categorize(name) != "language":
                continue
            env = make_env(suite, i)
            try:
                res[scene] = local_scene(env, start(env, suite, i), nominals[scene])
            finally:
                env.close()
            print(f"scene {scene} done", flush=True)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print(f"\n{'family':<13}{'identifiable':>13}  median per-unit sensitivity across scenes and parameters")
        for f in ("robot_init", "camera", "light", "sensor_noise", "layout", "background"):
            ent = [e for r in res.values() for e in r["families"].get(f, [])]
            if not ent:
                continue
            frac = np.mean([identifiable(e) for e in ent])
            med = {k: round(float(np.median([e.get(k, 0.0) for e in ent])), 3) for k in FLOOR}
            print(f"{f:<13}{frac:>13.2f}  {med}")
        print(f"{'language':<13}{0.0:>13.2f}  no physical parameter: zero coupling to the self-observation by construction")
        for scene, r in res.items():
            sv = r["camera_svd"]
            if sv:
                print(f"camera self-calibration conditioning {scene}: 1 pose {sv['1_pose']['condition']:.1f}, "
                      f"3 poses {sv['3_pose']['condition']:.1f}")
        return

    done = {}
    if os.path.exists(a.out):
        done = {r["task_id"]: r for r in load_rows(a.out)}
    tasks = [t for c in ("robot_init", "camera", "light", "sensor_noise", "layout", "background", "language")
             for t in [x for x in sub["tasks"] if x["category"] == c][:a.per_cat]]
    with open(a.out, "a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            env = make_env(suite, t["task_id"])
            try:
                sig = task_signals(env, start(env, suite, t["task_id"]), nominals[W.scene_of(t["name"])])
            finally:
                env.close()
            row = dict(task_id=t["task_id"], category=t["category"], **sig)
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            print(f"{t['task_id']:5d} {t['category']:<13} index {sig['index']:6.2f} "
                  f"{ {k: round(v, 2) for k, v in sig['index_parts'].items()} }", flush=True)
    rows = load_rows(a.out)
    print(f"\n{'category':<13}{'n':>4}{'identifiable':>13}{'median index':>14}")
    for c in ("robot_init", "camera", "light", "sensor_noise", "layout", "background", "language"):
        v = [r for r in rows if r["category"] == c]
        if v:
            print(f"{c:<13}{len(v):>4}{np.mean([r['index'] > 1 for r in v]):>13.2f}"
                  f"{np.median([r['index'] for r in v]):>14.2f}")
    if a.wrapper and a.baseline:
        base = {r["task_id"]: bool(r["success"]) for r in load_rows(a.baseline)}
        full = {}
        for r in load_rows(a.wrapper):
            if r["arm"] in ("full", "full2"):
                full.setdefault(r["task_id"], bool(r["success"]))
        idx = {r["task_id"]: r["index"] for r in rows}
        cats, ident, recov = [], [], []
        for c in ("robot_init", "camera", "light", "sensor_noise", "layout", "background", "language"):
            fails = [t for t, r in ((r["task_id"], r) for r in rows if r["category"] == c)
                     if t in base and t in full and not base[t]]
            if not fails:
                continue
            cats.append(c)
            ident.append(np.mean([idx[t] > 1 for t in fails]))
            recov.append(np.mean([full[t] for t in fails]))
            print(f"  {c:<13} baseline failures {len(fails):>3} | identifiable {ident[-1]:.2f} | recovered {recov[-1]:.2f}")
        pairs = [(idx[t], full[t]) for t in idx if t in base and t in full and not base[t]]
        rec_i = [i for i, ok in pairs if ok]
        non_i = [i for i, ok in pairs if not ok]
        print(f"\nprediction test: Spearman(identifiable fraction, recovery rate) across {len(cats)} categories = "
              f"{spearman(ident, recov):.2f}; median index of recovered failures {np.median(rec_i) if rec_i else float('nan'):.2f} "
              f"vs unrecovered {np.median(non_i) if non_i else float('nan'):.2f}")


if __name__ == "__main__":
    main()
