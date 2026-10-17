"""Calibration-only check (no policy): camera pose error of the old and new calibration on LIBERO-Plus tasks.

Needs only the LIBERO-Plus install, so it can run beside a policy run. For every task: true camera shift, the pose
each method would feed the reprojection (nominal when its decision rule does not fire), and that pose's error.
  old  calibrate()     silhouette grid + CEM mean, fires when nominal > 1.5 px and final < 0.7 x nominal
  new  calibrate_v2()  size-aware silhouette + best tracking + depth ICP, nominal unless beaten by 0.8x
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# LIBERO stores init states as numpy pickles and loads them with a bare torch.load; torch >= 2.6 defaults to
# weights_only=True and refuses them. Every script that starts an env imports this module, so patch once here.
import torch  # noqa: E402
_torch_load = torch.load


def _load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _load_trusted
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrapper_libero as W  # noqa: E402

JOINT_THR, CAM_THR = 0.05, 1.5


def make_env(suite, tid, res=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    env.seed(0)
    return env


def start(env, suite, tid):
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(tid)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_robot", type=int, default=6)
    ap.add_argument("--n_camera", type=int, default=10)
    a = ap.parse_args()

    from libero.libero import benchmark
    sub = json.load(open(a.subset))
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i),
                               cache=os.path.join(os.path.dirname(a.out) or ".", "nominals_v3.json"))
    tasks = ([t for t in sub["tasks"] if t["category"] == "robot_init"][:a.n_robot]
             + [t for t in sub["tasks"] if t["category"] == "camera"][:a.n_camera])
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task_id"] for l in open(a.out) if l.strip()}
    with open(a.out, "a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            env = make_env(suite, t["task_id"])
            obs = start(env, suite, t["task_id"])
            can = W.Canonicalizer(env, nominals[W.scene_of(t["name"])])
            Ct, Rt = can.true_camera()
            shift = W.pose_error(Ct, Rt, can.Cn, can.Rn)
            if can.joint_dev() > JOINT_THR:
                p, q = can.nominal_eef()
                obs, _ = can.servo(obs, p, q, max_steps=200)
            obs, recs = can.sweep(obs)

            t0 = time.time()
            c1 = can.calibrate(recs)
            fire1 = c1["nominal_score"] > CAM_THR and c1["final_score"] < 0.7 * c1["nominal_score"]
            raw1 = W.pose_error(can.C_est, can.R_est, Ct, Rt)
            used1 = raw1 if fire1 else shift
            s1 = time.time() - t0

            t0 = time.time()
            c2 = can.calibrate_v2(recs)
            used2 = W.pose_error(can.C_est, can.R_est, Ct, Rt)
            s2 = time.time() - t0

            row = dict(task_id=t["task_id"], category=t["category"], true_shift=shift,
                       old=dict(raw_err=raw1, fired=bool(fire1), used_err=used1, sec=s1, **c1),
                       new=dict(used_err=used2, sec=s2, **c2))
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            print(f"{t['task_id']:5d} {t['category']:<10} shift {shift[0]:5.1f}cm/{shift[1]:4.1f}deg | "
                  f"OLD estimate {raw1[0]:5.1f}cm/{raw1[1]:4.1f}deg fired={fire1!s:<5} used {used1[0]:5.1f}cm/{used1[1]:4.1f}deg "
                  f"({s1:.0f}s) | NEW {c2['method']:<10} used {used2[0]:5.1f}cm/{used2[1]:4.1f}deg "
                  f"score {c2['nominal_score']:.2f}->{c2['final_score']:.2f} ({s2:.0f}s)", flush=True)
            env.close()
            del env, can
            gc.collect()

    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    print(f"\n{'category':<11}{'n':>3}  {'true shift':>12}  {'OLD used err':>14}  {'NEW used err':>14}  "
          f"{'OLD fired':>9}  {'NEW fired':>9}   (median cm / deg)")
    for cat in ("robot_init", "camera"):
        v = [r for r in rows if r["category"] == cat]
        if not v:
            continue
        med = lambda key: (np.median([k[0] for k in key]), np.median([k[1] for k in key]))
        ts, o, n = med([r["true_shift"] for r in v]), med([r["old"]["used_err"] for r in v]), med([r["new"]["used_err"] for r in v])
        print(f"{cat:<11}{len(v):>3}  {ts[0]:5.1f}/{ts[1]:5.1f}   {o[0]:6.1f}/{o[1]:5.1f}   {n[0]:6.1f}/{n[1]:5.1f}   "
              f"{np.mean([r['old']['fired'] for r in v]):9.2f}  {np.mean([r['new']['applied'] for r in v]):9.2f}")
    print("go bar for the camera channel: NEW used error on camera tasks well below the true shift "
          "(target <= 3 cm / 3 deg), and robot_init (nominal camera) NEW fired ~0")


if __name__ == "__main__":
    main()
