"""Graded sweep: does self-identifiability predict which perturbations a frozen policy recovers from?

  score     policy-free. For every (base task, condition): apply the perturbation, run the arm sweep, and compute the
            self-identifiability of the condition's own parameters: the smallest singular value of the finite-
            difference Jacobian of the robot's self-observation (silhouette, body colour, joint angles, noise floor)
            with respect to those parameters, at the perturbed state, relative to the same block at the unperturbed
            state (stored as sigma_min_ref; scores_v2.jsonl). Also records the invertibility components (training-view
            coverage from the self-calibrated pose, body shading consistency, denoiser efficacy) and the composite
            recoverability score (sweep_analysis.recoverability).
  episodes  one policy family. For every (condition, base task): a RAW episode (perturbed, no wrapper) and a FULL
            episode (perturbed + self-measured wrapper). The raw episode also records a policy-internal probe: the
            change of the policy's first action chunk against the unperturbed first chunk.
Both are resumable (jsonl, one row per unit of work).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_conditions as S  # noqa: E402
import wrapper_libero as W  # noqa: E402
from calib_check import make_env  # noqa: E402
from plus_common import append_row, load_rows, load_shards  # noqa: E402

from plus_common import MAX_STEPS as _HORIZON, SUITE  # noqa: E402
MAX_STEPS, WAIT = _HORIZON[SUITE], 10
JOINT_THR, NOISE_MULT, NOISE_ADD, PHOTO_SHIFT, PHOTO_GAIN = 0.05, 2.0, 1.0, 15.0, 1.25


def start(env, suite, task, cond, seed):
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task["task_id"])[0])
    for _ in range(WAIT):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    info = S.apply_scene(env, cond)
    if cond["family"] == "joint":
        S.apply_joint(env, cond, seed)
    for _ in range(3):                              # re-render with the perturbed model and let contacts settle
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs, info


# ------------------------------------------------------------------ policy-free self-identifiability
def obs_vector(can, recs, sigma, arm_offset=None):
    import identifiability as I
    _, cols, joints, blur = I.self_obs(can, recs, arm_offset=arm_offset)
    img = can.sim.render(width=128, height=128, camera_name=can.cam)
    img = S.add_noise({"agentview_image": img}, sigma, np.random.default_rng(0))["agentview_image"]
    return np.concatenate([blur, cols.ravel() / 255.0 * 10.0, joints.ravel() * 10.0, [W.noise_level(img) / 255.0 * 10.0]])


def block_deltas(can, cond):
    """(label, apply) pairs: one small step of each parameter of the condition's own family, applied on top of the
    current (perturbed) state. Undo is handled by the caller's save/restore of model arrays."""
    m = can.sim.model
    fam, p = cond["family"], cond["params"]
    C0, R0 = can.true_camera()
    out = []
    if fam in ("camera", "nominal"):
        for i in range(3):
            d = np.zeros(3); d[i] = 0.01
            out.append((f"cam_pos{i}", lambda d=d: can._set_camera(C0 + d, R0)))
        for i in range(3):
            r = np.zeros(3); r[i] = np.deg2rad(1.0)
            out.append((f"cam_rot{i}", lambda r=r: can._set_camera(C0, W.rotvec_to_mat(r) @ R0)))
    elif fam == "light":
        if "dir_az" in p or p.get("grazing"):
            out.append(("light_dir_az", lambda: m.light_dir.__setitem__(slice(None), (W.rot_z(5.0) @ m.light_dir.T).T)))

            def tilt():
                d = m.light_dir.copy()
                d[:, 2] = d[:, 2] * np.cos(np.deg2rad(5.0)) - 0.087
                m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
            out.append(("light_dir_el", tilt))
        else:
            for c in range(3):
                g = np.ones(3); g[c] = 1.1
                out.append((f"light_gain{c}", lambda g=g: m.light_diffuse.__setitem__(slice(None), m.light_diffuse * g)))
    elif fam == "background":
        if "tint" in p:                                   # coloured room: one scalar drives table colour AND light tint
            step = dict(family="background", params=dict(table_rgb=p["table_rgb"], tint=p["tint"]))
            out.append(("room_colour", lambda: S.apply_scene(can.env, step, scale=0.1)))
        else:
            for c in range(3):
                def recolour(c=c):
                    for g in S._table_geoms(m):
                        m.geom_rgba[g, c] = min(1.0, m.geom_rgba[g, c] + 0.1)
                        if int(m.geom_matid[g]) >= 0:
                            m.mat_rgba[int(m.geom_matid[g]), c] = min(1.0, m.mat_rgba[int(m.geom_matid[g]), c] + 0.1)
                out.append((f"table_rgb{c}", recolour))
    return out


def score_instance(env, obs, cond, nominal, seed):
    can = W.Canonicalizer(env, nominal)
    joint_dev = can.joint_dev()
    obs, recs = can.sweep(obs)
    m = can.sim.model
    sigma = cond["params"].get("sigma", 0.0)
    snap = (can._save(), m.light_dir.copy(), m.light_diffuse.copy(), m.geom_rgba.copy(), m.mat_rgba.copy())

    def undo():
        m.light_dir[:], m.light_diffuse[:], m.geom_rgba[:], m.mat_rgba[:] = snap[1], snap[2], snap[3], snap[4]
        can._restore(snap[0])

    base = obs_vector(can, recs, sigma)
    cols, labels = [], []
    if cond["family"] == "joint":
        for k in range(len(can.arm_idx)):
            off = np.zeros(len(can.arm_idx)); off[k] = 0.02
            cols.append(obs_vector(can, recs, sigma, arm_offset=off) - base); labels.append(f"joint{k}")
            undo()
    elif cond["family"] == "noise":
        cols.append(obs_vector(can, recs, sigma + 2.0) - base); labels.append("sigma")
    else:
        for lab, apply in block_deltas(can, cond):
            apply()
            can.sim.forward()
            cols.append(obs_vector(can, recs, sigma) - base); labels.append(lab)
            undo()
    J = np.stack(cols, 1)
    sv = np.linalg.svd(J, compute_uv=False)
    area = float(np.mean([rec[2].mean() for rec in recs]))
    # invertibility components, all policy-free and self-measured (see sweep_analysis.recoverability):
    #   coverage   fraction of the training view the reprojection from the self-calibrated pose can reconstruct
    #   photo_d    affine-invariant body shading statistic (valid affine inversion <=> unchanged vs the reference)
    #   noise_inv  share of the added noise the denoiser removes, judged by the noise floor of the robot's own pixels
    can.calibrate_v2(recs)
    _, coverage = can.canonical_image(obs)
    photo_d = can.photometric_consistency(obs)
    noisy = S.add_noise({"agentview_image": obs["agentview_image"]}, sigma, np.random.default_rng(seed))["agentview_image"]
    lvl, floor = W.noise_level(noisy), float(nominal["noise"]["agentview_image"])
    if lvl <= NOISE_MULT * floor + NOISE_ADD:
        noise_inv = 1.0
    else:
        resid = max(W.noise_level(W.denoise(noisy, lvl)) - floor, 0.0)
        noise_inv = float(np.clip(1.0 - resid / max(lvl - floor, 1e-6), 0.0, 1.0))
    return dict(sigma_min=float(sv[-1]), sigma_max=float(sv[0]), n_params=len(labels), params=labels,
                robot_area=area, joint_dev=joint_dev, coverage=float(coverage), photo_d=photo_d, noise_inv=noise_inv)


def block_key(cond):
    """Which finite-difference block a condition is scored on (its own family's parameters)."""
    fam, p = cond["family"], cond["params"]
    if fam == "light":
        return (fam, "dir" if ("dir_az" in p or p.get("grazing")) else "gain")
    if fam == "background":
        return (fam, "tint" if "tint" in p else "table")
    return (fam,)


def run_score(a, suite, tasks, nominals):
    """scores_v2: every condition is scored on its OWN block and normalised by the SAME block at the unperturbed
    state of the same base task (sigma_min_ref). The v1 file normalised every family by the camera block, which
    made light, noise and joint scores incomparable to camera scores. Also stores the invertibility components."""
    from sweep_analysis import recoverability
    path = os.path.join(a.out_dir, "scores_v2.jsonl")
    rows = load_rows(path)
    done = {(r["task_id"], r["cond"]) for r in rows}
    refs = {(r["task_id"], tuple(r["block"])): r["ref"] for r in rows if r.get("ref")}
    with open(path, "a") as f:
        for cond in S.CONDITIONS + (S.EXTENDED if a.extended else []):
            for t in tasks:
                if (t["task_id"], cond["name"]) in done:
                    continue
                key = (t["task_id"], block_key(cond))
                if key not in refs:                                  # same block, unperturbed state, once per task
                    cond_ref = dict(cond, params={k: v for k, v in cond["params"].items() if k != "sigma"})
                    env = make_env(suite, t["task_id"])
                    try:
                        obs, _ = start(env, suite, t, S.BY_NAME["nominal"], seed=t["task_id"])
                        refs[key] = score_instance(env, obs, cond_ref, nominals[t["scene"]], seed=t["task_id"])
                    finally:
                        env.close()
                env = make_env(suite, t["task_id"])
                try:
                    obs, info = start(env, suite, t, cond, seed=t["task_id"])
                    sc = score_instance(env, obs, cond, nominals[t["scene"]], seed=t["task_id"])
                finally:
                    env.close()
                row = dict(task_id=t["task_id"], base=t["base"], cond=cond["name"], family=cond["family"],
                           level=cond["level"], override=cond["override"], block=list(key[1]),
                           sigma_min_ref=refs[key]["sigma_min"], photo_d_ref=refs[key]["photo_d"],
                           ref=refs[key], **sc, **info)
                row.update(recoverability(row))
                f.write(json.dumps(row, default=float) + "\n")
                f.flush()
                print(f"score {t['task_id']:5d} {cond['name']:<14} ident {row['ident']:.2f} invert {row['invert']:.2f} "
                      f"recover {row['recover']:.2f} | sigma_min {sc['sigma_min']:.4f} / ref {refs[key]['sigma_min']:.4f} "
                      f"coverage {sc['coverage']:.2f} area {sc['robot_area']:.3f}", flush=True)


# ------------------------------------------------------------------ policy episodes
# ---- the do-no-harm gate (pre-registered 2026-09-18, RESULTS.md) -------------------------------------------------
# A correction is an estimate. When the policy is already robust to the nuisance, the corrected frame can be FURTHER
# from what the policy handles than the raw one (pi0.5: light 0.90 -> 0.50 under the wrapper). The gate asks the frozen
# policy itself, without labels: is it less self-consistent on the corrected frame than on the raw frame? Self-
# consistency = how far its action moves under benign jitters of the input (2 px shifts, 4 percent brightness). A
# channel is VETOED only on positive evidence: the action deviation is larger on the corrected frame for at least
# GATE_MIN_WORSE of the GATE jitters, paired. Ties and mixed evidence leave the wrapper's own decision standing.
# No threshold is tuned on any policy; the state servo is never gated (no harm in either family).
GATE_JITTERS = (("shift", 2, 0), ("shift", -2, 0), ("shift", 0, 2), ("shift", 0, -2), ("gain", 1.04), ("gain", 0.96))
GATE_MIN_WORSE = 5


def _jitter(o, j):
    import cv2
    out = dict(o)
    for k in W.IMG_KEYS:
        if k not in out:
            continue
        img = np.asarray(out[k])
        if j[0] == "shift":
            M = np.float32([[1, 0, j[1]], [0, 1, j[2]]])
            out[k] = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        else:
            out[k] = np.clip(img.astype(np.float32) * j[1], 0, 255).astype(np.uint8)
    return out


def gate_profile(act, o):
    """The policy's action on o and its deviations under the jitters. act: observation -> action vector."""
    a0 = np.asarray(act(o), float)
    return a0, np.array([float(np.linalg.norm(np.asarray(act(_jitter(o, j)), float) - a0)) for j in GATE_JITTERS])


def gate_decide(prof_raw, prof_cor):
    (a_raw, d_raw), (a_cor, d_cor) = prof_raw, prof_cor
    worse = int(np.sum(d_cor > d_raw))
    veto = bool(worse >= GATE_MIN_WORSE and d_cor.mean() > d_raw.mean())
    return dict(veto=veto, worse=worse, s_raw=float(d_raw.mean()), s_cor=float(d_cor.mean()),
                action_change=float(np.linalg.norm(a_cor - a_raw)))


def generic_correction(o):
    """BASELINE `generic`: the corrections anyone would reach for without measuring anything: gray-world white balance,
    CLAHE on the lightness channel, a light edge-preserving denoise. Applied to every frame, no abstention."""
    import cv2
    for k in W.IMG_KEYS:
        if k not in o:
            continue
        img = np.asarray(o[k]).astype(np.float32)
        img *= img.mean() / np.maximum(img.reshape(-1, 3).mean(0), 1.0)              # gray world
        lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        o[k] = cv2.bilateralFilter(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 5, 25, 3)     # edge-preserving, ~2 ms a frame
    return o


LAZY_RESIDUAL = 1.5      # POST HOC (2026-09-21), chosen from logged rows, NOT pre-registered: the silhouette residual at the
#                          training camera is under 0.86 on every unshifted unit and over 2.26 on every camera-shifted one
SELFCHECK_MIN = 0.75     # frozen 2026-09-19 from policy-free data (self-check F1 0.82 at 5 cm of true error, 0.63 at 10 cm)


def prepare_full(env, obs, nominal, noise, sensing=None, pose=None, gate=None, cam_mode="full", settle=False,
                 verify=False, lazy=False):
    """The deployable wrapper (same rules as eval_oft_wrapper `full`); returns the per-step transform.
    pose="true" is a DIAGNOSTIC upper bound: the reprojection uses the simulator's camera pose instead of the
    calibrated one (rows are labelled *_truepose and never reported as deployable)."""
    can = W.Canonicalizer(env, nominal, sensing=sensing)
    info = dict(joint_dev=can.joint_dev())
    start_pose = ((np.asarray(obs["robot0_eef_pos"]).copy(), np.asarray(obs["robot0_eef_quat"]).copy())
                  if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs else None)
    if hasattr(can, "nominal_eef") and "robot0_eef_pos" in obs:      # logging only: how far the untouched start is from the training pose
        info["start_off_mm"] = float(1e3 * np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - can.nominal_eef()[0]))
    if info["joint_dev"] > JOINT_THR:
        p, q = can.nominal_eef()
        obs, info["home_steps"] = can.servo(obs, p, q, max_steps=200)
        info["state_applied"] = True
    skipped = False
    if lazy:
        # MEASURE ONLY WHEN THE BODY SAYS THE CAMERA MOVED: one motion-free look at the robot's own silhouette. When it
        # sits where the training camera puts it, the sweep is skipped and the episode starts from its untouched state.
        info["static_residual"] = can.static_residual(obs)
        skipped = info["static_residual"] <= LAZY_RESIDUAL
        info["lazy_skipped"] = bool(skipped)
    if skipped:
        can.C_est, can.R_est = can.Cn.copy(), can.Rn.copy()
        cal = dict(method="not measured: the silhouette sits where the training camera puts it", applied=False)
    else:
        obs, recs = can.sweep(obs)
        cal = can.calibrate_v2(recs)
    info.update({f"cal_{k}": v for k, v in cal.items()})
    arm_speed = lambda: float(np.abs(can.sim.data.qvel[can.arm_idx]).max())
    if hasattr(can, "arm_idx"):
        info["handover_speed"] = arm_speed()                     # logged for every arm: how still the arm is handed over
    if verify and not skipped and hasattr(can, "self_check"):
        # MEASURE UNTIL VERIFIED: the self-check (policy-free) rejects the calibration -> measure again with the richer
        # 7-pose sweep and keep whichever calibration the robot's own silhouette agrees with more.
        sc0 = can.self_check()
        if sc0 is not None and sc0["f1"] < SELFCHECK_MIN:
            keep = (can.C_est.copy(), can.R_est.copy(), cal, sc0["f1"])
            obs, recs2 = can.sweep(obs, poses=W.SWEEP_LEARNED)
            cal2 = can.calibrate_v2(recs2)
            sc1 = can.self_check()
            info.update(verify_retried=True, verify_f1_before=sc0["f1"], verify_f1_after=sc1["f1"])
            if sc1["f1"] > keep[3]:
                cal = cal2
                info.update({f"cal_{k}": v for k, v in cal.items()})
            else:
                can.C_est, can.R_est = keep[0], keep[1]
    if (settle or verify) and not (skipped and not info.get("state_applied")):   # nothing moved the arm: nothing to settle
        # target: the training pose when the state channel moved the arm there, otherwise the pose the episode STARTED
        # from (on some tasks that is 17 to 25 mm from the training pose: `full` silently moved the arm by that much)
        tgt = can.nominal_eef() if (info.get("state_applied") or start_pose is None) else start_pose
        obs, st = can.settle(obs, *tgt)
        info.update(settle_steps=st["steps"], settle_err_mm=st["err_mm"], handover_speed=st["speed"],
                    settle_target="training" if info.get("state_applied") or start_pose is None else "start")
    use_cam = bool(cal["applied"])
    try:                                                         # logging only, and never allowed to stop a run
        sc = can.self_check() if hasattr(can, "self_check") else None
        if sc:
            info.update(selfcheck_f1=sc["f1"], selfcheck_precision=sc["precision"], selfcheck_recall=sc["recall"])
    except Exception as e:
        info["selfcheck_error"] = str(e)[:120]
    Ct, Rt = can.true_camera()                                   # logging only, except under pose="true"
    info["cam_est_err_cm_deg"] = list(W.pose_error(can.C_est, can.R_est, Ct, Rt))
    info["cam_true_shift_cm_deg"] = list(W.pose_error(Ct, Rt, can.Cn, can.Rn))
    info["cam_est_pose"] = [*can.C_est.tolist(), *W.mat_to_rotvec(can.R_est).tolist()]      # position + rotation vector
    info["cam_nom_pose"] = [*can.Cn.tolist(), *W.mat_to_rotvec(can.Rn).tolist()]
    if can.mask_diag:
        info["mask_iou"] = float(np.mean([d["iou"] for d in can.mask_diag]))
        info["mask_area_ratio"] = float(np.mean([d["area_ratio"] for d in can.mask_diag]))
    if pose == "true":
        can.C_est, can.R_est, use_cam = Ct, Rt, True
    elif pose in ("refine", "refine2", "fiducial"):
        # POST HOC pose-source ablation (TASKS.md P47, pose_sources.py): same sweep, state channel and hand-over as
        # `full`; only the pose that drives the reprojection changes. cam_est_err_cm_deg is the error of the pose USED.
        assert not skipped, "the pose-source arms always measure"
        import pose_sources as PS
        img_hw = np.asarray(obs["agentview_image"]).shape[:2]
        C, R, use_cam, pinfo = PS.estimate(pose, can, recs, img_hw, W.TABLE_Z)
        can.C_est, can.R_est = C, R
        info.update(pinfo, cam_pose_applied=bool(use_cam))
        info["cam_est_err_cm_deg"] = list(W.pose_error(C, R, Ct, Rt))
        info["cam_est_pose"] = [*C.tolist(), *W.mat_to_rotvec(R).tolist()]
    cov = []
    obs = noise(obs)
    den = []
    for k in W.IMG_KEYS:
        if k in obs and k in nominal["noise"]:
            lvl = W.noise_level(obs[k])
            info[f"noise_{k}"] = lvl
            if lvl > NOISE_MULT * nominal["noise"][k] + NOISE_ADD:
                den.append(k)
    probe = dict(obs)
    for k in den:
        probe[k] = W.denoise(obs[k], info[f"noise_{k}"])
    ph = can.photometric_fit(probe, nominal)
    photo = ph["photo_ok"] and (ph["photo_mean_shift"] > PHOTO_SHIFT
                           or max(max(g, 1 / max(g, 1e-6)) for g in ph["photo_gain"]) > PHOTO_GAIN)
    if gate is not None:                   # gate = act(observation) -> action; channels are judged in pipeline order
        info.update(wanted_cam=use_cam, wanted_photo=bool(photo), wanted_denoise=list(den))
        copy = lambda o: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.items()}

        def ch_den(o):
            for k in den:
                o[k] = W.denoise(o[k], info[f"noise_{k}"])
            return o

        def ch_photo(o):
            for k in W.IMG_KEYS:
                if k in o:
                    o[k] = can.apply_photo(o[k])
            return o

        def ch_cam(o):
            o["agentview_image"], _ = can.canonical_image(o)
            return o

        base, prof, vetoes = copy(obs), None, []
        for name, want, fn in (("denoise", bool(den), ch_den), ("photo", bool(photo), ch_photo), ("cam", use_cam, ch_cam)):
            if not want:
                continue
            if prof is None:
                prof = gate_profile(gate, base)
            cand = fn(copy(base))
            prof_c = gate_profile(gate, cand)
            dec = gate_decide(prof, prof_c)
            info[f"gate_{name}"] = dec
            if dec["veto"]:
                vetoes.append(name)
            else:
                base, prof = cand, prof_c
        if "denoise" in vetoes:
            den = []
        if "photo" in vetoes:
            photo = False
        if "cam" in vetoes:
            use_cam = False
        info["gate_vetoes"] = vetoes
    info.update(ph, cam_applied=use_cam, photo_applied=bool(photo), denoise_applied=den)

    def transform(o):
        for k in den:
            o[k] = W.denoise(o[k], info[f"noise_{k}"])
        if photo:
            for k in W.IMG_KEYS:
                if k in o:
                    o[k] = can.apply_photo(o[k])
        if cam_mode == "oracle":
            # UPPER BOUND, never a method: the simulator's own render from the training camera replaces the frame. Same
            # measurement sweep, state channel and hand-over as `full`, so the ONLY difference is a perfect camera undo.
            # Top rung of the attribution ladder (attribution_ladder.py): what still fails here is not the camera.
            o["agentview_image"] = can.oracle_nominal_image(o)
        elif use_cam and cam_mode == "homog":
            # BASELINE: the measured pose with NO depth: one plane-induced homography (the training setup's table
            # plane) for every pixel, the robot included; holes inpainted. What a calibrated camera alone buys.
            img = o["agentview_image"]
            Hh = img.shape[0]
            z = W.plane_depth(Hh, img.shape[1], can._focal(Hh), can.C_est, can.R_est, can.table_z)
            out, c = W.reproject(can._to_obs(img), z, can._focal(Hh), can.C_est, can.R_est, can.Cn, can.Rn, inpaint=True)
            o["agentview_image"] = can._to_obs(out)
            cov.append(c)
        elif use_cam:
            o["agentview_image"], c = can.canonical_image(o, fill="ref" if cam_mode == "full" else "inpaint")
            cov.append(c)
        if cov:
            info["reproj_coverage"] = float(np.mean(cov))
        return o
    return obs, transform, info


def run_episode(adapter, env, suite, task, cond, arm, nominal, sensing=None, seed_offset=0, pose=None,
                gate_skip=False):
    """arm: raw | full | gated (full + the do-no-harm gate) | baselines: tta (the raw policy, actions averaged over
    benign jitters of the frame), generic (gray world + CLAHE + denoise on every frame), homog (full with a depth-free
    homography as the camera channel), nofill (full with inpainted holes instead of the training-view reference),
    settle (full, but the measurement ends AT REST at the pose the episode started from, or at the training pose
    when the state channel moved the arm), verify (settle + measure again with 7 poses when the self-check rejects the
    calibration), full_tta (full, with the policy averaged over benign jitters of the corrected frame).
    gate_skip: a gated episode in which the gate vetoes nothing is identical to the full episode, so it returns
    (None, 0, None, info) and the caller copies the full row."""
    assert not (arm in ("tta", "full_tta") and adapter.name in ("pi05", "pi0", "smolvla", "lerobot")), "tta needs a chunk-returning adapter"
    seed = task["task_id"] + seed_offset
    obs, info = start(env, suite, task, cond, seed)
    rng = np.random.default_rng(seed)
    sigma = cond["params"].get("sigma", 0.0)
    noise = lambda o: S.add_noise(o, sigma, rng)
    adapter.new_episode(task["instruction"], seed)
    first_chunk = None
    transform = None
    if arm in ("homog", "nofill", "settle", "verify", "oracle", "lazy"):
        obs, transform, winfo = prepare_full(env, obs, nominal, noise, sensing=sensing, pose=pose,
                                             cam_mode=arm if arm in ("homog", "nofill", "oracle") else "full",
                                             settle=arm in ("settle", "lazy"), verify=arm == "verify", lazy=arm == "lazy")
        info.update(winfo)
    elif arm == "generic":
        obs = noise(obs)
        transform = generic_correction
        winfo = {}
    elif arm == "nominal_view":
        # DIAGNOSTIC base rate for the camera category: nothing moves and nothing is measured; every frame the policy sees
        # is the simulator's own render from the TRAINING camera (the ladder's top-rung render without the measurement
        # sweep, the state channel or the hand-over). The camera shift changes only the image, so this is the unperturbed
        # task at the same layout, start state and seed: the success any camera correction can at best return to.
        can_nv = W.Canonicalizer(env, nominal)
        obs = noise(obs)

        def transform(o):
            o["agentview_image"] = can_nv.oracle_nominal_image(o)
            return o
        winfo = {}
        info["nominal_view"] = True
    elif arm == "nominal_start":
        # DIAGNOSTIC base rate for the robot-init category: the arm is PLACED at the training start q_nom (no servo, no
        # measurement, no correction of anything), then the raw policy runs. Only the joint perturbation is removed, so
        # this is the unperturbed task at the same object layout and seed: what the servo arm can at best return to.
        robot = env.env.robots[0]
        qi, vi = list(robot._ref_joint_pos_indexes), list(robot._ref_joint_vel_indexes)
        env.env.sim.data.qpos[qi] = np.asarray(nominal["q_nom"], float)
        env.env.sim.data.qvel[vi] = 0.0
        env.env.sim.forward()
        for _ in range(3):                          # re-render and let contacts settle, as `start` does
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        obs = noise(obs)
        info["nominal_start"] = True
        info["joint_dev_after"] = float(np.abs(env.env.sim.data.qpos[qi] - np.asarray(nominal["q_nom"], float)).max())
    elif arm in ("full", "gated", "full_tta"):
        gate = None
        if arm == "gated":
            def gate(o):                        # same instruction, same seed for every probe: only the frame differs
                adapter.new_episode(task["instruction"], seed)
                return np.mean([np.asarray(q, float)[:6] for q in adapter.chunk(dict(o))], 0)
        obs, transform, winfo = prepare_full(env, obs, nominal, noise, sensing=sensing, pose=pose, gate=gate)
        info.update(winfo)
        if arm == "gated":
            adapter.new_episode(task["instruction"], seed)          # the probes must leave no trace in the episode
            if gate_skip and not info.get("gate_vetoes"):
                info["gate_same_as_full"] = True
                return None, 0, None, info
    else:
        obs = noise(obs)
    queue, success, steps = [], False, 0
    while steps < MAX_STEPS:
        o = dict(obs)
        if transform is not None:
            o = transform(o)
        if not queue and arm in ("tta", "full_tta"):   # tta: the raw frame; full_tta: the CORRECTED frame (o is post-transform)
            chunks = [list(adapter.chunk(dict(o)))] + [list(adapter.chunk(_jitter(dict(o), j))) for j in GATE_JITTERS[:4]]
            n = min(len(c) for c in chunks)
            queue = [np.mean([np.asarray(c[i], float) for c in chunks], 0) for i in range(n)]
            for q in queue:
                q[6] = np.sign(chunks[0][0][6]) if abs(q[6]) < 1e-6 else np.sign(q[6])       # the gripper command is binary
        elif not queue:
            queue = list(adapter.chunk(o))
        if first_chunk is None:
            first_chunk = np.mean([q[:6] for q in queue], 0).tolist()
        obs, _, done, _ = env.step(queue.pop(0).tolist())
        obs = noise(obs)
        steps += 1
        if done:
            success = True
            break
    if transform is not None:
        info.update(winfo)                  # per-step diagnostics (reprojection coverage) accumulate during the episode
    return success, steps, first_chunk, info


def run_episodes(a, suite, tasks, nominals):
    from policy_adapters import make_adapter
    adapter = make_adapter(a.family, a.octo_calib)
    base = os.path.join(a.out_dir, f"episodes_{adapter.name}.jsonl")
    import socket
    shard = socket.gethostname()[-6:] if a.shard == "auto" else a.shard          # same suffix as this runtime's mirrored logs
    path = base[:-len(".jsonl")] + f".{shard}.jsonl" if shard else base          # this session appends ONLY to its own shard
    print(f"results shard: {os.path.basename(path)} (rows of every other session count as done)", flush=True)
    rows = load_rows(path)
    is_stale = lambda r: r.get("photo_applied") and "photo_shading" not in r and not r.get("recovered")
    stale = [r for r in rows if is_stale(r)]
    if stale:                                             # photometric map applied before the shading gate: redo
        keep = [r for r in rows if not is_stale(r)]
        with open(path + ".before_shading_gate", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in keep)
        rows = keep
        print(f"re-queued {len(stale)} episodes whose photometric map predates the shading gate", flush=True)
    rows = [r for r in load_shards(base) if not is_stale(r)]          # what EVERY session has finished counts as done
    done = {(r["task_id"], r["cond"], r["arm"], r.get("rep", 0)) for r in rows}
    full_rows = {(r["task_id"], r["cond"], r.get("rep", 0)): r for r in rows if r["arm"] == "full" and "steps" in r}
    nominal_chunk = {r["task_id"]: r["first_chunk"] for r in rows
                     if r["cond"] == "nominal" and r["arm"] == "raw" and r.get("first_chunk") is not None}
    arms = a.arms.split(",")
    conds = S.CONDITIONS + (S.EXTENDED if a.extended else [])
    todo = [(c, t, arm, rep) for c in conds if not a.conditions or c["name"] in a.conditions.split(",")
            for t in tasks for arm in arms for rep in range(a.repeats)
            if (t["task_id"], c["name"], arm, rep) not in done]
    print(f"{adapter.name}: {len(todo)} episodes to run", flush=True)
    t0 = time.time()
    for n, (cond, t, arm, rep) in enumerate(todo):
        env = make_env(suite, t["task_id"])
        try:
            fr = full_rows.get((t["task_id"], cond["name"], rep)) if arm == "gated" else None
            ok, steps, chunk, info = run_episode(adapter, env, suite, t, cond, arm, nominals[t["scene"]],
                                                 seed_offset=1000 * rep, gate_skip=fr is not None)
            if ok is None:                                  # the gate vetoed nothing: same episode as `full`
                ok, steps, chunk = fr["success"], fr["steps"], fr.get("first_chunk")
                info["copied_from_full"] = True
        finally:
            env.close()
        if cond["name"] == "nominal" and arm == "raw":
            nominal_chunk[t["task_id"]] = chunk
        probe = None
        if arm == "raw" and chunk is not None and t["task_id"] in nominal_chunk:
            probe = float(np.linalg.norm(np.subtract(chunk, nominal_chunk[t["task_id"]])))
        row = dict(policy=adapter.name, task_id=t["task_id"], base=t["base"], cond=cond["name"],
                   family=cond["family"], level=cond["level"], override=cond["override"], arm=arm, rep=rep,
                   success=bool(ok), steps=steps, first_chunk=chunk, probe=probe, **info)
        append_row(path, row)
        if arm == "full":
            full_rows[(t["task_id"], cond["name"], rep)] = row
        el = (time.time() - t0) / 60
        print(f"{adapter.name} {t['task_id']:5d} {cond['name']:<14} {arm:<5} success={ok!s:<5} veto={info.get('gate_vetoes', '-')} "
              f"cam={info.get('cam_applied')} photo={info.get('photo_applied')} "
              f"denoise={len(info.get('denoise_applied', []))} | {n + 1}/{len(todo)} "
              f"eta {el / (n + 1) * (len(todo) - n - 1):.0f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "episodes", "list"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nominals", required=True)
    ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--family", default="oft", choices=["oft", "openvla", "octo", "pi05", "pi0", "smolvla"])
    ap.add_argument("--octo_calib", default="")
    ap.add_argument("--arms", default="raw,full")
    ap.add_argument("--conditions", default="", help="comma-separated condition names (default: all)")
    ap.add_argument("--extended", action="store_true", help="also run the boundary conditions (sweep_conditions.EXTENDED)")
    ap.add_argument("--shard", default="auto", help="per-session results file episodes_<policy>.<shard>.jsonl (readers merge the "
                    "shards); auto = this runtime's hostname suffix, empty = the one shared file (never with two sessions)")
    ap.add_argument("--task_ids", default="", help="comma-separated base task ids: keep only these of the first --n_tasks "
                    "base tasks (a post hoc extension picks its tasks by a rule written in TASKS.md before it runs)")
    ap.add_argument("--repeats", type=int, default=1, help="episodes per (condition, task, arm) with different seeds")
    a = ap.parse_args()

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[SUITE]()
    tasks = S.base_tasks(suite, a.n_tasks)
    if a.task_ids:
        keep = {int(x) for x in a.task_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t["task_id"] in keep]
        assert tasks, f"--task_ids {a.task_ids}: none of them is among the first {a.n_tasks} base tasks"
    os.makedirs(a.out_dir, exist_ok=True)
    for t in tasks:
        print(f"base task {t['task_id']} [{t['scene']}]: {t['instruction']}", flush=True)
    if a.mode == "list":
        print(f"{len(S.CONDITIONS)} conditions:", [c["name"] for c in S.CONDITIONS])
        return
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    (run_score if a.mode == "score" else run_episodes)(a, suite, tasks, nominals)


if __name__ == "__main__":
    main()
