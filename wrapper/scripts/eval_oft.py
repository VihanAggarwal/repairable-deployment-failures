"""OpenVLA-OFT on the LIBERO-Plus subset, using OFT's own run_episode. Run from the openvla-oft repo root."""
import argparse
import gc
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="moojink/openvla-7b-oft-finetuned-libero-10")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, run_episode
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

    tasks = json.load(open(a.subset))["tasks"]
    tasks = tasks[:a.limit] if a.limit else tasks
    cfg = GenerateConfig(pretrained_checkpoint=a.ckpt, task_suite_name="libero_10", center_crop=True)
    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    done = {r["task_id"] for r in load_rows(a.out)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            task = suite.get_task(t["task_id"])
            init = suite.get_task_init_states(t["task_id"])
            env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            success, _ = run_episode(cfg, env, desc, model, resize_size, processor, action_head,
                                     proprio_projector, noisy_action_projector, init[0])
            env.close()
            del env
            gc.collect()
            row = dict(family="openvla_oft", **t, success=bool(success))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{t['task_id']:5d} {t['category']:<12} success={bool(success)}", flush=True)
    summarize([r for r in load_rows(a.out)], "OpenVLA-OFT")


if __name__ == "__main__":
    main()
