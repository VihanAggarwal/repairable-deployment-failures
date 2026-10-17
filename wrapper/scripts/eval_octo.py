"""Octo (community LIBERO fine-tune, akuramshin/octo-libero-100-finetuned) on the LIBERO-Plus subset.

The checkpoint ships no model card, so `calib` tries the two image orientations and two gripper
conventions on a few tasks and saves the combination that succeeds most; `run` uses it."""
import argparse
import gc
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows, summarize  # noqa: E402

MAX_STEPS, WAIT, CKPT = 520, 10, "akuramshin/octo-libero-100-finetuned"


def load_model(step):
    from huggingface_hub import snapshot_download
    from octo.model.octo_model import OctoModel
    path = snapshot_download(CKPT)
    try:
        model = OctoModel.load_pretrained(path, step=step)
    except TypeError:
        model = OctoModel.load_pretrained(path)
    stats = model.dataset_statistics
    if "action" not in stats:
        keys = [k for k in stats if "libero_10" in k] or sorted(stats)
        print("dataset_statistics keys:", sorted(stats), "-> using", keys[0])
        stats = stats[keys[0]]
    return model, stats["action"]


def to_env_gripper(g, mode):
    if mode == "sign":          # already in env convention: +1 close, -1 open
        return 1.0 if g > 0 else -1.0
    return 1.0 if g < 0.5 else -1.0   # "unit": [0,1] with 1 = open (OpenVLA RLDS convention)


def episode(env, init_state, instruction, model, act_stats, flip, gmode, seed):
    import jax
    from PIL import Image
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(WAIT):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    task = model.create_tasks(texts=[instruction])
    rng = jax.random.PRNGKey(seed)
    steps = 0
    while steps < MAX_STEPS:
        img = obs["agentview_image"]
        img = img[::-1, ::-1] if flip else img
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((256, 256)))
        o = {"image_primary": img[None, None], "timestep_pad_mask": np.ones((1, 1), dtype=bool)}
        rng, key = jax.random.split(rng)
        acts = np.asarray(model.sample_actions(o, task, unnormalization_statistics=act_stats, rng=key))[0]
        for a in acts:
            a = np.asarray(a, dtype=np.float64).copy()
            a[6] = to_env_gripper(float(a[6]), gmode)
            obs, _, done, _ = env.step(a.tolist())
            steps += 1
            if done:
                return True
            if steps >= MAX_STEPS:
                break
    return False


def make_env(task, res=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res, camera_widths=res)
    env.seed(0)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "run"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_file", required=True)
    ap.add_argument("--step", type=int, default=30000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    sub = json.load(open(a.subset))
    model, act_stats = load_model(a.step)

    if a.mode == "calib":
        pref = [t for t in sub["tasks"] if t["category"] in ("other", "light", "language")] or sub["tasks"]
        probe = pref[:3]
        res = {}
        for flip in (True, False):
            for gmode in ("unit", "sign"):
                wins = 0
                for t in probe:
                    task = suite.get_task(t["task_id"])
                    env = make_env(task)
                    wins += episode(env, suite.get_task_init_states(t["task_id"])[0], task.language, model,
                                    act_stats, flip, gmode, seed=0)
                    env.close(); del env; gc.collect()
                res[f"{flip}|{gmode}"] = wins
                print(f"flip={flip} gripper={gmode}: {wins}/{len(probe)}", flush=True)
        best = max(res, key=res.get)
        flip, gmode = best.split("|")
        json.dump({"flip": flip == "True", "gmode": gmode, "results": res, "valid": bool(res[best] > 0)},
                  open(a.calib_file, "w"))
        print("chosen:", best, "" if res[best] > 0 else "  WARNING: no combination succeeded; paste this output before running")
        return

    cal = json.load(open(a.calib_file))
    if not cal.get("valid", True):   # max() over four zeros is not a convention; refuse rather than run on a guess
        sys.exit("calibration never won a probe task: fix the setup before running (see RESULTS.md 2026-09-16)")
    tasks = sub["tasks"][:a.limit] if a.limit else sub["tasks"]
    done = {r["task_id"] for r in load_rows(a.out)}
    with open(a.out, "a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            task = suite.get_task(t["task_id"])
            env = make_env(task)
            ok = episode(env, suite.get_task_init_states(t["task_id"])[0], task.language, model, act_stats,
                         cal["flip"], cal["gmode"], seed=t["task_id"])
            env.close(); del env; gc.collect()
            f.write(json.dumps(dict(family="octo", **t, success=bool(ok))) + "\n")
            f.flush()
            print(f"{t['task_id']:5d} {t['category']:<12} success={ok}", flush=True)
    summarize(load_rows(a.out), "Octo")


if __name__ == "__main__":
    main()
