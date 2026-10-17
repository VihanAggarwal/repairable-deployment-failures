"""Self-labelled training pairs for the learned canonicalizer (claim 3).

Held-out LIBERO-Plus tasks only (camera-perturbed and language tasks that are NOT in the evaluation subset). Per task:
self-calibrate the camera with calibrate_v2 (the same measurement the wrapper uses), move the arm to scripted random
end-effector targets, and store (observed image, observed image reprojected into the nominal camera, validity mask,
the TRUE nominal-camera render). The self-labelled reprojection needs no privileged information. The oracle render is
a training-time-only target for `train_canon.py --target oracle`: the network trained on it still takes only the
observed RGB frame at deployment, so nothing privileged crosses into the wrapper. The true pose is saved only for
reporting.

With --seg_ckpt (and --depth plane|mono) the reprojection and its hole mask come from LEARNED sensing (trained
segmenter, table-plane or monocular depth), so a canonicalizer trained on these shards learns to repair the artefacts
of the deployable sensing rather than of the renderer's. Segmenter labels (`r`) always come from the renderer.

Resumable: one .npz per task. Run several workers in parallel with --worker i --workers k.
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrapper_libero as W  # noqa: E402
from calib_check import make_env, start  # noqa: E402
from plus_common import categorize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--nominals", required=True, help="nominals_v3.json cache (built on first use)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_camera", type=int, default=120)
    ap.add_argument("--n_language", type=int, default=30)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seg_ckpt", default="", help="learned sensing for the reprojection (train_seg.py output)")
    ap.add_argument("--depth", default="plane", choices=["plane", "mono"])
    a = ap.parse_args()

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    sub = json.load(open(a.subset))
    held_in = {t["task_id"] for t in sub["tasks"]}
    small = a.subset.replace("subset_libero10.json", "subset_libero10_small.json")
    if os.path.exists(small):
        held_in |= {t["task_id"] for t in json.load(open(small))["tasks"]}
    rng = np.random.default_rng(0)
    pool = {"camera": [], "language": []}
    for i in range(suite.n_tasks):
        c = categorize(suite.get_task(i).name)
        if c in pool and i not in held_in:
            pool[c].append(i)
    ids = sorted(rng.permutation(pool["camera"])[:a.n_camera].tolist()
                 + rng.permutation(pool["language"])[:a.n_language].tolist())
    ids = ids[a.worker::a.workers]
    os.makedirs(a.out_dir, exist_ok=True)
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    sensing = W.LearnedSensing(a.seg_ckpt, depth=a.depth) if a.seg_ckpt else None
    print(f"worker {a.worker}/{a.workers}: {len(ids)} held-out tasks (none in the evaluation subsets); sensing "
          f"{'learned, ' + a.depth + ' depth' if sensing else 'renderer'}", flush=True)

    for tid in ids:
        path = os.path.join(a.out_dir, f"task_{tid:05d}.npz")
        if os.path.exists(path):
            continue
        name = suite.get_task(tid).name
        env = make_env(suite, tid)
        try:
            obs = start(env, suite, tid)
            can = W.Canonicalizer(env, nominals[W.scene_of(name)], sensing=sensing)
            if can.joint_dev() > 0.05:
                p, q = can.nominal_eef()
                obs, _ = can.servo(obs, p, q, max_steps=200)
            obs, recs = can.sweep(obs)
            cal = can.calibrate_v2(recs)
            Ct, Rt = can.true_camera()
            p0, q0 = can.nominal_eef()
            r = np.random.default_rng(tid)
            X, Y, M, RM, O = [], [], [], [], []
            for _ in range(a.frames):
                obs, _ = can.servo(obs, p0 + r.uniform([-0.12, -0.15, -0.08], [0.12, 0.15, 0.08]), q0, max_steps=40)
                tgt, _, filled = can.canonical_image(obs, return_mask=True)
                X.append(obs["agentview_image"].copy())
                Y.append(tgt)
                M.append(filled)
                RM.append(can._to_obs(can._robot_mask(obs["agentview_image"].shape[0])))   # labels: renderer, always
                O.append(can.oracle_nominal_image(obs))         # privileged target, training time only
            meta = dict(task_id=tid, name=name, category=categorize(name), scene=W.scene_of(name),
                        sensing=("learned:" + a.depth) if sensing else "renderer",
                        calib=dict(cal), est_err_cm_deg=W.pose_error(can.C_est, can.R_est, Ct, Rt),
                        true_shift_cm_deg=W.pose_error(Ct, Rt, can.Cn, can.Rn))
            np.savez_compressed(path + ".tmp.npz", x=np.stack(X), y=np.stack(Y), m=np.stack(M), r=np.stack(RM), o=np.stack(O),
                                meta=json.dumps(meta, default=float))
            os.replace(path + ".tmp.npz", path)
            print(f"{tid:5d} {meta['category']:<9} shift {meta['true_shift_cm_deg'][0]:5.1f}cm | calib {cal['method']:<10}"
                  f" err {meta['est_err_cm_deg'][0]:4.1f}cm/{meta['est_err_cm_deg'][1]:4.1f}deg | "
                  f"mask {np.mean(M):.2f}", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
