"""OpenVLA-OFT on LIBERO-Plus with the self-measured canonicalization wrapper. Run from the openvla-oft repo root.

  validate  a few camera tasks: self-calibration error vs ground truth, reprojection quality vs the true nominal
            render, side-by-side PNGs.
  run       arms, in PRIORITY order (so the most informative episodes finish first if compute runs out):
              full      deployable wrapper; every channel diagnoses from measurements and fires only when needed:
                        state (joint deviation), camera (calibrate_v2), photometric (robot's own pixels),
                        denoise (noise floor). Run on every category; on capability categories it is a no-harm control.
              state     state channel only (robot_init)
              cam_true  true camera pose + reprojection (upper bound, camera)
              oracle    true nominal-camera render (upper bound, camera)
              learned   learned canonicalizer (camera; needs --canon_ckpt). Checkpoints trained with --input rgb take the
                        raw frame only (no depth, no sweep); --input reproj checkpoints take the wrapper's own
                        self-calibrated reprojection + hole mask + frame (same sensing as `full`)
"""
import argparse
import gc
import json
import os
import sys
from collections import deque

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows, wilson  # noqa: E402
import wrapper_libero as W  # noqa: E402

PRIORITY = [("camera", "full"), ("camera", "cam_true_ref"), ("robot_init", "full"), ("camera", "learned"),
            ("light", "full"), ("sensor_noise", "full"), ("layout", "full"), ("background", "full"),
            ("language", "full"), ("camera", "oracle"), ("camera", "cam_true"), ("robot_init", "state")]
PER_CAT = {"camera": 30, "robot_init": 30, "light": 10, "sensor_noise": 10, "layout": 10, "background": 10, "language": 10}
NOMINAL_VERSION = 4                # 4 = wider calibration search + training-view hole fill; rows >= 2 are kept
OLD_Q_NOM = [0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, 0.7854]
JOINT_THR = 0.05                   # rad
NOISE_MULT, NOISE_ADD = 2.0, 1.0   # denoise when the noise floor exceeds 2x the training floor + 1 gray level
PHOTO_SHIFT, PHOTO_GAIN = 15.0, 1.25


def setup(ckpt):
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
    cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name="libero_10", center_crop=True)
    set_seed_everywhere(cfg.seed)
    parts = initialize_model(cfg)
    return cfg, parts, get_image_resize_size(cfg)


def photo_stale(r):
    """A row whose photometric map was applied before the shading gate existed (no photo_shading field)."""
    return bool(r.get("photo_applied")) and "photo_shading" not in r


def drop_stale(path, nominals, canon_tag="", learned_label="learned"):
    """Re-queue rows from older wrapper versions. A `state` row stays valid when its scene's robot start equals the
    old global one (the state channel never used the camera). A `learned` row is tied to the checkpoint that produced
    it (canon_tag), so a retrained canonicalizer re-queues them. The old file is backed up first."""
    rows = load_rows(path)
    keep = [r for r in rows if (r.get("nominal_version", 0) >= 2          # v2 = per-scene nominal, still valid
            or (r["arm"] == "state" and np.allclose(nominals[W.scene_of(r["name"])]["q_nom"], OLD_Q_NOM, atol=1e-3)))
            and (r["arm"] != learned_label or r.get("canon", "") == canon_tag)
            and not photo_stale(r)]
    if len(keep) < len(rows):
        with open(path + f".before_v{NOMINAL_VERSION}_backup", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in keep)
        print(f"re-queued {len(rows) - len(keep)} rows from older wrapper versions, kept {len(keep)}", flush=True)


def episode(cfg, parts, resize_size, env, desc, init_state, arm, nominal, canon=None, sensing=None):
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS, prepare_observation, process_action
    from experiments.robot.robot_utils import get_action
    model, action_head, proprio_projector, noisy_action_projector, processor = parts
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    can = W.Canonicalizer(env, nominal, sensing=sensing if arm == "learned" else None)   # learned sensing: learned arm only
    Ct, Rt = can.true_camera()
    info = dict(arm=arm, joint_dev=can.joint_dev(), cam_true_shift_cm_deg=W.pose_error(Ct, Rt, can.Cn, can.Rn),
                sensing=("learned:" + sensing.depth_mode) if (sensing is not None and arm == "learned") else "renderer")
    use_cam, photo, denoise_keys = None, False, []
    if arm == "state" or (arm == "full" and info["joint_dev"] > JOINT_THR):
        p, q = can.nominal_eef()
        obs, info["home_steps"] = can.servo(obs, p, q, max_steps=200)
        info["state_applied"] = True
    if arm == "full":
        obs, recs = can.sweep(obs)
        cal = can.calibrate_v2(recs)
        info.update({f"cal_{k}": v for k, v in cal.items()})
        info["cam_est_err_cm_deg"] = W.pose_error(can.C_est, can.R_est, Ct, Rt)
        if cal["applied"]:
            use_cam = "est"
        for k in W.IMG_KEYS:
            if k in obs and k in nominal["noise"]:
                lvl = W.noise_level(obs[k])
                info[f"noise_{k}"] = lvl
                if lvl > NOISE_MULT * nominal["noise"][k] + NOISE_ADD:
                    denoise_keys.append(k)
        probe = dict(obs)
        for k in denoise_keys:
            probe[k] = W.denoise(obs[k], info[f"noise_{k}"])
        ph = can.photometric_fit(probe, nominal)
        info.update(ph)
        photo = ph["photo_ok"] and (ph["photo_mean_shift"] > PHOTO_SHIFT
                               or max(max(g, 1 / max(g, 1e-6)) for g in ph["photo_gain"]) > PHOTO_GAIN)
    elif arm in ("cam_true", "cam_true_ref", "oracle", "learned"):
        use_cam = {"cam_true": "true", "cam_true_ref": "true_ref", "oracle": "oracle", "learned": "learned"}[arm]
        if arm == "learned" and getattr(canon, "in_ch", 3) == 7:   # measured geometry in: same self-calibration as full
            obs, recs = can.sweep(obs)
            cal = can.calibrate_v2(recs)
            info.update({f"cal_{k}": v for k, v in cal.items()})
            info["cam_est_err_cm_deg"] = W.pose_error(can.C_est, can.R_est, Ct, Rt)
    info.update(cam_applied=use_cam, photo_applied=bool(photo), denoise_applied=denoise_keys)
    queue = deque(maxlen=cfg.num_open_loop_steps)
    success, covers = False, []
    for t in range(TASK_MAX_STEPS["libero_10"]):
        for k in denoise_keys:
            obs[k] = W.denoise(obs[k], info[f"noise_{k}"])
        if photo:
            for k in W.IMG_KEYS:
                if k in obs:
                    obs[k] = can.apply_photo(obs[k])
        if use_cam == "est":
            obs["agentview_image"], cv = can.canonical_image(obs)
            covers.append(cv)
        elif use_cam == "true":                          # legacy upper bound: true pose, holes inpainted
            obs["agentview_image"], cv = can.canonical_image(obs, Ct, Rt, fill="inpaint")
            covers.append(cv)
        elif use_cam == "true_ref":                      # true pose, holes from the training-view reference
            obs["agentview_image"], cv = can.canonical_image(obs, Ct, Rt, fill="ref")
            covers.append(cv)
        elif use_cam == "oracle":
            obs["agentview_image"] = can.oracle_nominal_image(obs)
        elif use_cam == "learned":
            import canon_net
            if getattr(canon, "in_ch", 3) == 7:
                rep, cv, filled = can.canonical_image(obs, return_mask=True)
                covers.append(cv)
                inp = np.concatenate([rep, (filled.astype(np.uint8) * 255)[..., None], obs["agentview_image"]], -1)
                obs["agentview_image"] = canon_net.apply(canon, inp)
            else:
                obs["agentview_image"] = canon_net.apply(canon, obs["agentview_image"])
        observation, _ = prepare_observation(obs, resize_size)
        if not queue:
            queue.extend(get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                                    proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                                    use_film=cfg.use_film))
        action = process_action(queue.popleft(), cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
    if covers:
        info["reproj_coverage"] = float(np.mean(covers))
    return success, info


def validate(cfg, suite, sub, nominals, get_libero_env, out_dir, n):
    import cv2
    tasks = [t for t in sub["tasks"] if t["category"] == "camera"][:n]
    psnr = lambda a, b: float(10 * np.log10(255.0 ** 2 / max(np.mean((a.astype(float) - b.astype(float)) ** 2), 1e-9)))
    for t in tasks:
        env, _ = get_libero_env(suite.get_task(t["task_id"]), cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(t["task_id"])[0])
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        can = W.Canonicalizer(env, nominals[W.scene_of(t["name"])])
        Ct, Rt = can.true_camera()
        obs, recs = can.sweep(obs)
        cal = can.calibrate_v2(recs)
        est_err = W.pose_error(can.C_est, can.R_est, Ct, Rt)
        shift = W.pose_error(Ct, Rt, can.Cn, can.Rn)
        oracle = can.oracle_nominal_image(obs)
        est, _ = can.canonical_image(obs)
        tru, cov_t = can.canonical_image(obs, Ct, Rt)
        print(f"task {t['task_id']}: true shift {shift[0]:.1f} cm / {shift[1]:.1f} deg | calibrate_v2 {cal['method']} error "
              f"{est_err[0]:.1f} cm / {est_err[1]:.1f} deg | PSNR vs true nominal render: raw "
              f"{psnr(obs['agentview_image'], oracle):.1f}, reproj(true pose) {psnr(tru, oracle):.1f}, "
              f"reproj(estimate) {psnr(est, oracle):.1f} dB | coverage {cov_t:.2f}", flush=True)
        panel = np.concatenate([obs["agentview_image"], est, tru, oracle], 1)
        cv2.imwrite(os.path.join(out_dir, f"validate_{t['task_id']}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        env.close(); del env; gc.collect()
    print("panels: observed | reprojected from estimate | reprojected from true pose | true nominal render")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "run"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--ckpt", default="moojink/openvla-7b-oft-finetuned-libero-10")
    ap.add_argument("--canon_ckpt", default="")
    ap.add_argument("--seg_ckpt", default="", help="learned sensing for the `learned` arm's reprojection (else renderer)")
    ap.add_argument("--depth", default="plane", choices=["plane", "mono"])
    ap.add_argument("--skip_arms", default="", help="comma-separated arms to skip, e.g. cam_true")
    ap.add_argument("--rerun_full2", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env
    sub = json.load(open(a.subset))
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    cfg, parts, resize_size = setup(a.ckpt)
    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    nominals = W.read_nominals(
        suite, lambda i: get_libero_env(suite.get_task(i), cfg.model_family, resolution=cfg.env_img_res)[0],
        cache=os.path.join(out_dir, "nominals_v3.json"))
    missing = sorted({W.scene_of(t["name"]) for t in sub["tasks"]} - set(nominals))
    assert not missing, f"no unperturbed source task for scenes {missing}"
    if a.mode == "validate":
        validate(cfg, suite, sub, nominals, get_libero_env, out_dir, a.n)
        return

    canon, canon_tag = None, ""
    if a.canon_ckpt and os.path.exists(a.canon_ckpt):
        import canon_net
        canon = canon_net.load(a.canon_ckpt)
        import torch
        ck = torch.load(a.canon_ckpt, map_location="cpu")
        canon_tag = f"{os.path.basename(a.canon_ckpt)}:{ck.get('input', 'rgb')}:{ck.get('target', 'reproj')}:{ck.get('epoch', 0)}"
        if a.seg_ckpt:
            canon_tag += f":learned-{a.depth}"
        print(f"learned canonicalizer loaded from {a.canon_ckpt} ({canon_tag})", flush=True)
    sensing = W.LearnedSensing(a.seg_ckpt, depth=a.depth) if (a.seg_ckpt and canon is not None) else None
    skip = set(filter(None, a.skip_arms.split(",")))
    if canon is None:
        skip.add("learned")
        print("no --canon_ckpt found: the learned arm is skipped", flush=True)

    learned_label = f"learned_{a.depth}" if sensing is not None else "learned"   # learned sensing: its own row label
    drop_stale(a.out, nominals, canon_tag, learned_label)
    done = {(r["task_id"], r["arm"]) for r in load_rows(a.out)}
    # v2 `full2` rows (per-scene camera + calibrate_v2, no photometric/denoise channels) count as `full` on robot_init
    # and camera, where those two channels are not expected to fire; pass --rerun_full2 to re-run them instead
    # robot_init full2 rows stand in for `full` (its state channel is unchanged). Camera and light are re-run: the camera
    # channel changed (wider calibration, training-view hole fill) and light now has the photometric channel.
    cat_of = {t["task_id"]: t["category"] for t in sub["tasks"]}
    if not a.rerun_full2:
        done |= {(tid, "full") for tid, arm in done if arm == "full2" and cat_of.get(tid) == "robot_init"}
    by_cat = {c: [t for t in sub["tasks"] if t["category"] == c][:k] for c, k in PER_CAT.items()}
    with open(a.out, "a") as f:
        for cat, arm in PRIORITY:
            if arm in skip:
                continue
            for t in by_cat[cat]:
                label = learned_label if arm == "learned" else arm
                if (t["task_id"], label) in done:
                    continue
                scene = W.scene_of(t["name"])
                env, desc = get_libero_env(suite.get_task(t["task_id"]), cfg.model_family, resolution=cfg.env_img_res)
                ok, info = episode(cfg, parts, resize_size, env, desc, suite.get_task_init_states(t["task_id"])[0], arm,
                                   nominals[scene], canon, sensing)
                env.close(); del env; gc.collect()
                info["arm"] = label
                row = dict(family="openvla_oft", **t, success=bool(ok), scene=scene, nominal_version=NOMINAL_VERSION,
                           canon=canon_tag if arm == "learned" else "", **info)
                f.write(json.dumps(row, default=float) + "\n")
                f.flush()
                print(f"{t['task_id']:5d} {cat:<12} {arm:<8} success={ok!s:<5} state={info.get('state_applied', False)!s:<5} "
                      f"cam={info['cam_applied']} photo={info['photo_applied']} denoise={len(info['denoise_applied'])} "
                      f"cam_err={np.round(info.get('cam_est_err_cm_deg', ()), 1).tolist()}", flush=True)

    rows = load_rows(a.out)
    base = {r["task_id"]: r["success"] for r in load_rows(a.baseline)} if a.baseline else {}
    print(f"\n{'category':<13}{'arm':<10}{'n':>4}{'success':>9}{'95% CI':>15}{'baseline':>10}")
    for cat, arm in PRIORITY + [("robot_init", "full2"), ("camera", "full2"), ("light", "full2")]:
        v = [r for r in rows if r["category"] == cat and r["arm"] == arm]
        if not v:
            continue
        k = sum(r["success"] for r in v)
        lo, hi = wilson(k, len(v))
        b = [base[r["task_id"]] for r in v if r["task_id"] in base]
        bs = f"{sum(b) / len(b):.3f}" if b else "-"
        print(f"{cat:<13}{arm:<10}{len(v):>4}{k / len(v):>9.3f}   [{lo:.2f}, {hi:.2f}]{bs:>10}")
    print("\ndiagnosis (full arm): fraction of episodes where each channel fired")
    for cat in PER_CAT:
        v = [r for r in rows if r["category"] == cat and r["arm"] == "full"]
        if v:
            print(f"  {cat:<13} state {np.mean([bool(r.get('state_applied')) for r in v]):.2f} | camera "
                  f"{np.mean([r.get('cam_applied') == 'est' for r in v]):.2f} | photometric "
                  f"{np.mean([bool(r.get('photo_applied')) for r in v]):.2f} | denoise "
                  f"{np.mean([bool(r.get('denoise_applied')) for r in v]):.2f}")


if __name__ == "__main__":
    main()
