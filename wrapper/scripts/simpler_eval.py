"""SIMPLER (real-to-sim, WidowX visual-matching envs): Octo and OpenVLA, clean and under camera and lighting
perturbations applied to the camera image. Loop mirrors simple_inference_visual_matching_prepackaged_envs.py."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows, wilson  # noqa: E402

TASKS = ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
CONDS = ["clean", "cam3", "cam6", "dim", "bright"]


def perturb(img, cond):
    import cv2
    if cond == "clean":
        return img
    if cond.startswith("cam"):              # camera bump: rotation about the optical centre, as a homography
        deg = float(cond[3:])
        h, w = img.shape[:2]
        f = float(w)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        ay, ax = np.deg2rad(deg), np.deg2rad(deg / 2)
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        H = K @ Ry @ Rx @ np.linalg.inv(K)
        return cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    gain, gamma = {"dim": (0.55, 1.2), "bright": (1.45, 0.85)}[cond]
    x = np.clip((img.astype(np.float32) / 255.0) ** gamma * gain, 0, 1)
    return (x * 255).astype(np.uint8)


def make_model(policy):
    if policy.startswith("octo"):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
        from simpler_env.policies.octo.octo_model import OctoInference
        return OctoInference(model_type=policy, policy_setup="widowx_bridge", init_rng=0)
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference
    return OpenVLAInference(saved_model_path="openvla/openvla-7b", policy_setup="widowx_bridge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="octo-small | octo-base | openvla")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()

    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    model = make_model(a.policy)
    done = {(r["task"], r["cond"], r["ep"]) for r in load_rows(a.out)}
    with open(a.out, "a") as f:
        for task in a.tasks.split(","):
            env = simpler_env.make(task)
            for cond in a.conds.split(","):
                for ep in range(a.n):
                    if (task, cond, ep) in done:
                        continue
                    obs, _ = env.reset()
                    instruction = env.get_language_instruction()
                    is_final = env.is_final_subtask()
                    model.reset(instruction)
                    image = perturb(get_image_from_maniskill2_obs_dict(env, obs), cond)
                    terminated, success, truncated = False, False, False
                    while not (terminated or truncated):
                        _, action = model.step(image, instruction)
                        terminated = bool(action["terminate_episode"][0] > 0)
                        if terminated and not is_final:
                            terminated = False
                            env.advance_to_next_subtask()
                        obs, _, success, truncated, _ = env.step(
                            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
                        new = env.get_language_instruction()
                        if new != instruction:
                            instruction = new
                        is_final = env.is_final_subtask()
                        image = perturb(get_image_from_maniskill2_obs_dict(env, obs), cond)
                    f.write(json.dumps(dict(policy=a.policy, task=task, cond=cond, ep=ep, success=bool(success))) + "\n")
                    f.flush()
                    print(f"{task} {cond} ep{ep} success={bool(success)}", flush=True)
            env.close()
    rows = load_rows(a.out)
    print(f"\n{a.policy}")
    for cond in CONDS:
        v = [r["success"] for r in rows if r["cond"] == cond]
        if v:
            lo, hi = wilson(sum(v), len(v))
            print(f"  {cond:<7} {sum(v)}/{len(v)} = {sum(v)/len(v):.3f}  [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
