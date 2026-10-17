"""Base OpenVLA (discrete action tokens) on the LIBERO-Plus subset, mirroring the official openvla
run_libero_eval loop. Run from the openvla repo root."""
import argparse
import gc
import inspect
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import append_row, load_rows, summarize  # noqa: E402

MAX_STEPS = 520  # libero_10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="openvla/openvla-7b-finetuned-libero-10")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (get_action, get_image_resize_size, get_model, invert_gripper_action,
                                               normalize_gripper_action, set_seed_everywhere)

    tasks = json.load(open(a.subset))["tasks"]
    tasks = tasks[:a.limit] if a.limit else tasks
    cfg = GenerateConfig(pretrained_checkpoint=a.ckpt, task_suite_name="libero_10", center_crop=True)
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    cfg.unnorm_key = cfg.task_suite_name
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, f"unnorm key missing; have {list(model.norm_stats)}"
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    img_takes_resize = len(inspect.signature(get_libero_image).parameters) >= 2
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    done = {r["task_id"] for r in load_rows(a.out)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if True:        # rows go through append_row (open, write, CLOSE): Drive uploads a file only when it is closed
        for t in tasks:
            if t["task_id"] in done:
                continue
            task = suite.get_task(t["task_id"])
            init = suite.get_task_init_states(t["task_id"])
            env, desc = get_libero_env(task, cfg.model_family, resolution=256)
            env.reset()
            obs = env.set_init_state(init[0])
            step, success = 0, False
            while step < MAX_STEPS + cfg.num_steps_wait:
                if step < cfg.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    step += 1
                    continue
                img = get_libero_image(obs, resize_size) if img_takes_resize else get_libero_image(obs)
                observation = {"full_image": img,
                               "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                        obs["robot0_gripper_qpos"]))}
                action = get_action(cfg, model, observation, desc, processor=processor)
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _, done_, _ = env.step(action.tolist())
                if done_:
                    success = True
                    break
                step += 1
            env.close()
            del env
            gc.collect()
            append_row(a.out, dict(family="openvla", **t, success=success))
            print(f"{t['task_id']:5d} {t['category']:<12} success={success}", flush=True)
    summarize(load_rows(a.out), "OpenVLA (base)")


if __name__ == "__main__":
    main()
