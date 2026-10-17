"""The self-measured wrapper on ANY policy family, over the shared LIBERO-Plus subset (claim 1 across families).

For each family adapter (oft, openvla, octo, pi05, pi0) and each subset task, arms:
  raw           the perturbed task through the same rollout loop, no wrapper (paired baseline)
  full          deployable wrapper, simulator stand-ins for robot segmentation and depth
  full_learned  deployable wrapper with LEARNED sensing: a trained robot segmenter + monocular depth
                (Depth Anything) scaled on the robot's own pixels (needs --seg_ckpt)
  full_plane    deployable wrapper with the learned segmenter and NO depth network: the training setup's table
                plane meets the calibrated camera's rays; the robot's pixels take its known geometry (needs --seg_ckpt)
  full_gated    full plus the do-no-harm gate (sweep_run.gate_decide): a channel is vetoed when the frozen policy is
                less self-consistent on the corrected frame than on the raw one. Run AFTER `full` in the same --arms list:
                an episode in which nothing is vetoed is identical to `full` and is copied, not re-run.
  BASELINES (OFT / OpenVLA; pi0.5 keeps its own action queue, so `tta` is not defined for it):
  tta           the raw policy, its action chunk averaged over the raw frame and four 2 px shifts of it
  generic       gray-world white balance + CLAHE + denoise on every frame, nothing measured, no abstention
  full_homog    full, but the camera channel is a depth-free plane homography with inpainting (calibration alone)
  full_settle   full, but the measurement sweep ends with the arm at the target and AT REST (plain `full` hands over
                inside 4 mm and still moving, which alone can flip a chaotic policy's outcome)
  full_verify   full_settle, plus: when the robot's own self-check rejects the calibration (F1 < 0.75), measure again
                with the 7-pose sweep and keep the calibration the silhouette agrees with more
  full_tta      full, with the policy's actions averaged over benign jitters of the CORRECTED frame (wrapper + tta)
  full_nofill   full, but reprojection holes are inpainted instead of filled from the training-view reference
  full_hybrid   full_plane plus the training setup's static geometry and the held-object rule (needs --seg_ckpt)
  nominal_view  DIAGNOSTIC base rate for camera tasks: every frame is the simulator's render from the TRAINING camera, the
                arm is untouched and nothing is measured; the unperturbed task at the same layout and seed
  full_metric   POST HOC: full with LEARNED sensing: the trained segmenter plus a monocular METRIC depth network
                (Depth Anything V2 metric, used as is, no fit to the robot's geometry); needs --seg_ckpt. Tests whether
                learned sensing with metric depth brings self-calibration inside the policy's ~1 cm tolerance
  nominal_start DIAGNOSTIC base rate for robot_init: the arm is placed at the training start (no servo, no correction),
                then the raw policy runs; the success an ideal joint reset can reach on the same layout and seed
  full_refine   POST HOC (TASKS.md P47, pose-source ablation): full, but the silhouette pose is refined EasyHeC-style
                (dense soft-mask loss, Adam from the same initial pose, finite-difference gradients; pose_sources.py)
  full_fiducial POST HOC (P47): full, but the pose comes from PnP on four table fiducials with oracle (never occluded)
                corners and 0.5 px corner noise, the AnyCamVLA pose source (pose_sources.py)
  *_truepose    DIAGNOSTIC upper bounds (full_truepose, full_plane_truepose, full_hybrid_truepose): same sensing, but
                the reprojection uses the simulator's camera pose. Separates depth error from calibration error.
Rows are written in the invertibility_report format (arm, task_id, category, success), so
  invertibility_report.py --family NAME:wrap_NAME.jsonl:wrap_NAME.jsonl
uses the paired raw rows as the baseline.
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
import sweep_run as R  # noqa: E402
import wrapper_libero as W  # noqa: E402
from calib_check import make_env  # noqa: E402
from plus_common import append_row, load_rows  # noqa: E402

PER_CAT = {"camera": 30, "robot_init": 30, "light": 10, "sensor_noise": 10, "layout": 10, "background": 10, "language": 10}
POSE_SOURCE = {"full_refine": "refine", "full_fiducial": "fiducial"}      # POST HOC P47 (pose_sources.py)
PRIORITY = ["camera", "robot_init", "light", "sensor_noise", "layout", "background", "language"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["oft", "openvla", "octo", "pi05", "pi0", "smolvla"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nominals", required=True)
    ap.add_argument("--arms", default="raw,full")
    ap.add_argument("--per_cat_scale", type=float, default=1.0, help="multiply PER_CAT (e.g. 0.34 for a cheap pass)")
    ap.add_argument("--categories", default=",".join(PRIORITY))
    ap.add_argument("--octo_calib", default="")
    ap.add_argument("--seg_ckpt", default="")
    ap.add_argument("--rep", type=int, default=0,
                    help="replicate index: episode seeds become task_id + 1000 * rep (start state, noise and the policy's "
                         "sampler all change). Give every replicate its OWN --out_dir; rep 0 is the main table")
    ap.add_argument("--tasks", default="", help="comma-separated task ids: run only these (diagnostics)")
    ap.add_argument("--only_failed_in", default="",
                    help="a finished wrap file: run ONLY the tasks that fail there under BOTH raw and full (the units the "
                         "attribution ladder needs upper rungs for); arms already present in that file are not repeated. "
                         "Use with an --out_dir of its own")
    ap.add_argument("--redo", action="store_true",
                    help="with --tasks: drop the existing rows of these tasks for the requested arms that carry no "
                         "cam_est_err_cm_deg (made before the diagnostics were logged), then run them again")
    ap.add_argument("--extra_from_suite", type=int, default=0,
                    help="add N more tasks per category from the full suite, outside every evaluation subset (seed 1)")
    ap.add_argument("--shard", default="auto",
                    help="per-session results shard (wrap_<family>.<shard>.jsonl) so several runtimes can share one "
                         "--out_dir; auto = hostname suffix, empty = the one shared file (never with two sessions)")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the task list from the end: give the SECOND runtime on a shared --out_dir this flag")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="start this fraction of the way into the task list (a THIRD runtime: --rotate 0.5)")
    a = ap.parse_args()

    from libero.libero import benchmark
    from policy_adapters import make_adapter
    suite = benchmark.get_benchmark_dict()[R.SUITE]()
    print(f"suite {R.SUITE}, horizon {R.MAX_STEPS} steps", flush=True)
    from plus_common import load_or_build
    sub = load_or_build(a.subset)                      # built for the suite in LIBERO_SUITE when the file does not exist
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    adapter = make_adapter(a.family, a.octo_calib)
    sensings, by_mode = {}, {}
    for arm in a.arms.split(","):
        mode = {"full_learned": "mono", "full_plane": "plane", "full_hybrid": "hybrid", "full_metric": "metric"}.get(
            arm.replace("_truepose", "").replace("_refine2", ""))
        if mode:
            assert a.seg_ckpt and os.path.exists(a.seg_ckpt), f"{arm} needs --seg_ckpt (train_seg.py output)"
            if mode not in by_mode:
                by_mode[mode] = W.LearnedSensing(a.seg_ckpt, depth=mode)
            sensings[arm] = by_mode[mode]
    if "hybrid" in by_mode:
        W.read_static_clouds(suite, lambda i: make_env(suite, i), nominals, cache=a.nominals)
    only = {int(x) for x in a.tasks.split(",") if x.strip()}
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, f"wrap_{adapter.name}.jsonl")
    import socket
    from plus_common import load_shards
    shard = socket.gethostname()[-6:] if a.shard == "auto" else a.shard
    path = base[:-len(".jsonl")] + f".{shard}.jsonl" if shard else base      # this session appends ONLY to its own shard
    claims = os.path.join(a.out_dir, "claims")
    os.makedirs(claims, exist_ok=True)
    print(f"results shard: {os.path.basename(path)} (rows and claims of every other session count as taken)", flush=True)
    rows = load_shards(base)
    stale = [r for r in rows if r.get("photo_applied") and "photo_shading" not in r]
    if stale:                                             # photometric map applied before the shading gate: redo
        with open(path + ".before_shading_gate", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        rows = [r for r in rows if not (r.get("photo_applied") and "photo_shading" not in r)]
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"re-queued {len(stale)} episodes whose photometric map predates the shading gate", flush=True)
    if a.redo and only:
        old = [r for r in rows if r["task_id"] in only and r["arm"] in a.arms.split(",") and r["arm"] != "raw"
               and "cam_est_err_cm_deg" not in r]
        if old:
            with open(path + ".before_redo", "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in old)
            rows = [r for r in rows if r not in old]
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"redo: moved {len(old)} rows without calibration diagnostics to {os.path.basename(path)}.before_redo", flush=True)
    done = {(r["task_id"], r["arm"]) for r in rows}
    if a.only_failed_in:
        prev = [r for r in load_rows(a.only_failed_in) if r.get("rep", 0) == a.rep]
        assert prev, f"--only_failed_in: no rows in {a.only_failed_in}"
        ok = lambda arm: {r["task_id"] for r in prev if r["arm"] == arm and r["success"]}
        ran = lambda arm: {r["task_id"] for r in prev if r["arm"] == arm}
        only = ((ran("raw") - ok("raw")) & (ran("full") - ok("full"))) & (only or ran("raw"))
        done |= {(r["task_id"], r["arm"]) for r in prev}
        print(f"--only_failed_in: {len(only)} tasks fail under both raw and full in {os.path.basename(a.only_failed_in)}", flush=True)
        if not only:
            return
    full_rows = {r["task_id"]: r for r in rows if r["arm"] == "full"}
    nominal_cond = S.BY_NAME["nominal"]
    extra = {}
    if a.extra_from_suite:
        from plus_common import categorize
        taken = {t["task_id"] for t in sub["tasks"]}
        small = a.subset.replace("subset_libero10.json", "subset_libero10_small.json")
        if os.path.exists(small):
            taken |= {t["task_id"] for t in json.load(open(small))["tasks"]}
        rng = np.random.default_rng(1)
        for cat in a.categories.split(","):
            pool = [i for i in range(suite.n_tasks) if i not in taken and categorize(suite.get_task(i).name) == cat]
            extra[cat] = [dict(task_id=int(i), name=suite.get_task(int(i)).name, language=suite.get_task(int(i)).language,
                               category=cat) for i in sorted(rng.permutation(pool)[:a.extra_from_suite].tolist())]
    todo = []
    for cat in a.categories.split(","):
        k = int(round(PER_CAT[cat] * a.per_cat_scale)) if a.per_cat_scale > 0 else 0
        for t in [x for x in sub["tasks"] if x["category"] == cat][:k] + extra.get(cat, []):
            for arm in a.arms.split(","):
                if (t["task_id"], arm) not in done and (not only or t["task_id"] in only):
                    todo.append((t, arm))
    if a.reverse:
        todo = todo[::-1]
    if a.rotate:                        # a THIRD runtime on the same folder starts part-way in (e.g. 0.5 = the middle)
        k = int(len(todo) * a.rotate) % max(len(todo), 1)
        todo = todo[k:] + todo[:k]
    print(f"{adapter.name}: {len(todo)} episodes to run", flush=True)
    t0 = time.time()

    def taken(t, arm):
        """Done in ANY session's shard, or claimed by another session (claims/<task>_<arm>_<rep>.<shard>). Drive syncs
        with a lag, so two sessions can rarely both run one episode: harmless (separate shards, readers keep one row)."""
        key = f"{t['task_id']}_{arm}_{a.rep}"
        if any(r["task_id"] == t["task_id"] and r["arm"] == arm and r.get("rep", 0) == a.rep for r in load_shards(base)):
            return True
        now = time.time()                    # a claim older than 30 min is a dead session's (one episode takes minutes)
        return any(f.startswith(key + ".") and not f.endswith("." + (shard or "main"))
                   and now - os.path.getmtime(os.path.join(claims, f)) < 1800 for f in os.listdir(claims))

    for n, (t, arm) in enumerate(todo):
        if taken(t, arm):
            print(f"{adapter.name} {t['task_id']:5d} {arm:<12} taken by another session, skipped", flush=True)
            continue
        open(os.path.join(claims, f"{t['task_id']}_{arm}_{a.rep}.{shard or 'main'}"), "w").close()
        task = dict(task_id=t["task_id"], instruction=t["language"], scene=W.scene_of(t["name"]))
        env = make_env(suite, t["task_id"])
        try:
            fr = full_rows.get(t["task_id"]) if arm == "full_gated" else None
            ok, steps, chunk, info = R.run_episode(adapter, env, suite, task, nominal_cond,
                                                   {"full_gated": "gated", "full_lazy": "lazy", "full_oracle": "oracle", "full_homog": "homog", "full_nofill": "nofill", "full_settle": "settle", "full_verify": "verify", "full_tta": "full_tta",
                                                    "tta": "tta", "generic": "generic", "nominal_start": "nominal_start", "nominal_view": "nominal_view"}.get(
                                                       arm, "full" if arm.startswith("full") else "raw"),
                                                   nominals[task["scene"]],
                                                   sensing=sensings.get(arm),
                                                   pose="true" if arm.endswith("_truepose") else
                                                   ("refine2" if arm.endswith("_refine2") else POSE_SOURCE.get(arm)),
                                                   gate_skip=fr is not None, seed_offset=1000 * a.rep)
            if ok is None:                                  # the gate vetoed nothing: same episode as `full`
                ok, steps = fr["success"], fr["steps"]
                info["copied_from_full"] = True
        finally:
            env.close()
        row = dict(family=adapter.name, task_id=t["task_id"], name=t["name"], category=t["category"], arm=arm,
                   rep=a.rep, success=bool(ok), steps=steps, **{k: v for k, v in info.items() if k != "first_chunk"})
        append_row(path, row)
        if arm == "full":
            full_rows[t["task_id"]] = row
        el = (time.time() - t0) / 60
        print(f"{adapter.name} {t['task_id']:5d} {t['category']:<12} {arm:<12} success={ok!s:<5} "
              f"state={info.get('state_applied', False)!s:<5} cam={info.get('cam_applied')} "
              f"photo={info.get('photo_applied')} veto={info.get('gate_vetoes', '-')} | {n + 1}/{len(todo)} eta {el / (n + 1) * (len(todo) - n - 1):.0f} min",
              flush=True)


if __name__ == "__main__":
    main()
