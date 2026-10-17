"""Self-motion consistency probe: NO rollouts. For every (base task, condition) of the graded sweep, set the scene up
exactly as sweep_run does, let the wrapper prepare its corrections, move the end-effector through the probe cross
(smc.PROBES), and at each probe position ask the frozen policy for its action on the RAW frame and on the CORRECTED
frame. Rows store positions and actions, so every statistic can be recomputed offline.

    python smc_probe.py run    --family oft|openvla|pi05 --out_dir <sweep dir> --nominals <json> [--n_tasks 3] [--extended]
    python smc_probe.py report --out_dir <sweep dir>          # joins with episodes_<policy>.jsonl (raw / full outcomes)

About 1 to 1.5 minutes per unit: a few hundred simulator steps and 14 policy forward passes.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smc  # noqa: E402


def probe_unit(adapter, env, suite, task, cond, nominal, R, S, W):
    seed = task["task_id"]
    obs, info = R.start(env, suite, task, cond, seed)
    rng = np.random.default_rng(seed)
    sigma = cond["params"].get("sigma", 0.0)
    noise = lambda o: S.add_noise(o, sigma, rng)
    adapter.new_episode(task["instruction"], seed)
    obs, transform, winfo = R.prepare_full(env, obs, nominal, noise)         # obs is already noised here
    can = W.Canonicalizer(env, nominal)
    p0, q0 = can.nominal_eef()
    copy = lambda o: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.items()}

    def act(o):
        adapter.new_episode(task["instruction"], seed)                       # same seed: only the frame differs
        return np.mean([np.asarray(q, float)[:3] for q in adapter.chunk(dict(o))], 0)

    P, A_raw, A_cor = [], [], []
    for k, off in enumerate(smc.PROBES):
        if k:
            obs, _ = can.servo(obs, p0 + off, q0)
            obs = noise(obs)
        P.append(np.asarray(obs["robot0_eef_pos"], float).tolist())
        A_raw.append(act(copy(obs)).tolist())
        A_cor.append(act(transform(copy(obs))).tolist())
    s_raw, s_cor = smc.triangulate(P, A_raw), smc.triangulate(P, A_cor)
    keep = ("cam_applied", "photo_applied", "denoise_applied", "state_applied", "cam_est_err_cm_deg", "cam_true_shift_cm_deg",
            "cam_est_pose", "cam_nom_pose")
    return dict(P=P, A_raw=A_raw, A_cor=A_cor, smc_raw=s_raw, smc_cor=s_cor, pick=smc.compare(s_raw, s_cor),
                **{k: winfo[k] for k in keep if k in winfo})


def run(a):
    import sweep_conditions as S
    import sweep_run as R
    import wrapper_libero as W
    from calib_check import make_env
    from libero.libero import benchmark
    from plus_common import append_row, load_rows
    from policy_adapters import make_adapter
    suite = benchmark.get_benchmark_dict()[R.SUITE]()
    tasks = S.base_tasks(suite, a.n_tasks)
    nominals = W.read_nominals(suite, lambda i: make_env(suite, i), cache=a.nominals)
    adapter = make_adapter(a.family, a.octo_calib)
    path = os.path.join(a.out_dir, f"smc_{adapter.name}.jsonl")
    done = {(r["task_id"], r["cond"]) for r in load_rows(path)}
    conds = S.CONDITIONS + (S.EXTENDED if a.extended else [])
    todo = [(c, t) for c in conds for t in tasks if (t["task_id"], c["name"]) not in done]
    print(f"{adapter.name}: {len(todo)} units to probe", flush=True)
    t0 = time.time()
    if True:        # rows go through append_row (open, write, CLOSE): Drive uploads a file only when it is closed
        for n, (cond, t) in enumerate(todo):
            env = make_env(suite, t["task_id"])
            try:
                row = probe_unit(adapter, env, suite, t, cond, nominals[t["scene"]], R, S, W)
            finally:
                env.close()
            row = dict(policy=adapter.name, task_id=t["task_id"], cond=cond["name"], family=cond["family"], **row)
            append_row(path, row)
            el = (time.time() - t0) / 60
            print(f"{adapter.name} {t['task_id']:5d} {cond['name']:<14} residual raw {row['smc_raw']['residual_deg']:5.1f} "
                  f"cor {row['smc_cor']['residual_deg']:5.1f} deg -> pick {row['pick']:<3} | {n + 1}/{len(todo)} "
                  f"eta {el / (n + 1) * (len(todo) - n - 1):.0f} min", flush=True)


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if label.all() or (~label).all() or len(label) < 4:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):                                                # average ranks over ties
        m = score == v
        ranks[m] = ranks[m].mean()
    return float((ranks[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))


def report(a):
    for path in sorted(glob.glob(os.path.join(a.out_dir, "smc_*.jsonl"))):
        pol = os.path.basename(path)[4:-6]
        ep_path = os.path.join(a.out_dir, f"episodes_{pol}.jsonl")
        if not os.path.exists(ep_path):
            print(f"{pol}: no episodes file next to the probe rows, skipping")
            continue
        ep = {}
        for l in open(ep_path):
            if l.strip():
                r = json.loads(l)
                if r.get("rep", 0) == 0:
                    ep[(r["task_id"], r["cond"], r["arm"])] = bool(r["success"])
        rows = [json.loads(l) for l in open(path) if l.strip()]
        U = [r for r in rows if (r["task_id"], r["cond"], "raw") in ep and (r["task_id"], r["cond"], "full") in ep
             and ep.get((r["task_id"], "nominal", "raw"), False)]
        raw = np.array([ep[(r["task_id"], r["cond"], "raw")] for r in U])
        full = np.array([ep[(r["task_id"], r["cond"], "full")] for r in U])
        pick = np.array([r["pick"] for r in U])
        chosen = np.where(pick == "raw", raw, full)                           # 'cor' and 'tie' keep the wrapper's frame
        print(f"\n===== {pol} =====  {len(U)} units on base tasks the policy solves unperturbed")
        print(f"success: always raw {raw.mean():.3f} | always full {full.mean():.3f} | SMC-selected {chosen.mean():.3f} "
              f"| oracle {np.maximum(raw, full).mean():.3f}")
        print(f"broken by the wrapper (raw ok, full fails): {int((raw & ~full).sum())}, of which SMC picks raw: "
              f"{int((raw & ~full & (pick == 'raw')).sum())}")
        print(f"recovered by the wrapper (raw fails, full ok): {int((~raw & full).sum())}, of which SMC keeps the wrapper: "
              f"{int((~raw & full & (pick != 'raw')).sum())}")
        r_raw = np.array([r["smc_raw"]["residual_deg"] for r in U])
        r_cor = np.array([r["smc_cor"]["residual_deg"] for r in U])
        print(f"AUROC, low residual on the RAW frame predicts raw success: {auroc(-r_raw, raw):.3f}")
        print(f"AUROC, low residual on the CORRECTED frame predicts full success: {auroc(-r_cor, full):.3f}")
        print(f"median residual deg: raw frame, raw succeeds {np.nanmedian(r_raw[raw]):.1f} / fails "
              f"{np.nanmedian(r_raw[~raw]) if (~raw).any() else float('nan'):.1f}")
        print(f"picks: {dict(zip(*np.unique(pick, return_counts=True)))}")
        # action-field fingerprint: distance of the probe actions to the SAME task's actions in the training setup
        ref = {r["task_id"]: r["A_raw"] for r in rows if r["cond"] == "nominal"}
        F = [r for r in U if r["task_id"] in ref and r["cond"] != "nominal"]      # the reference units are not test units
        if F:
            fr = np.array([ep[(r["task_id"], r["cond"], "raw")] for r in F])
            ff = np.array([ep[(r["task_id"], r["cond"], "full")] for r in F])
            d_r = np.array([smc.field_distance(r["A_raw"], ref[r["task_id"]]) for r in F])
            d_c = np.array([smc.field_distance(r["A_cor"], ref[r["task_id"]]) for r in F])
            img = np.array([bool(r.get("cam_applied") or r.get("photo_applied") or r.get("denoise_applied")) for r in F])
            keep_raw = np.array([bool(i) and smc.decide_fingerprint(a_, b_)[0] == "raw" for i, a_, b_ in zip(img, d_r, d_c)])
            sel = np.where(keep_raw, fr, ff)
            mean = lambda v, m: float(v[m].mean()) if m.any() else float("nan")
            print(f"\nFINGERPRINT ({len(F)} units, image channel fired on {int(img.sum())})")
            print(f"  AUROC, small raw-frame distance predicts raw success: {auroc(-d_r, fr):.3f} | small corrected-frame "
                  f"distance predicts full success: {auroc(-d_c, ff):.3f}")
            for k, name in ((1, "centre probe only"), (3, "centre + x pair"), (7, "all 7 probes")):
                dk = np.array([smc.field_distance(np.array(r["A_cor"])[:k], np.array(ref[r["task_id"]])[:k]) for r in F])
                print(f"    {name:<18} corrected-frame AUROC {auroc(-dk, ff):.3f}")
            print(f"  success: always raw {fr.mean():.3f} | always full {ff.mean():.3f} | fingerprint-selected {sel.mean():.3f} "
                  f"| oracle {np.maximum(fr, ff).mean():.3f}")
            print(f"  where an image channel fired: raw {mean(fr, img):.3f} | full {mean(ff, img):.3f} | selected {mean(sel, img):.3f}")
            print(f"  broken by an image channel (raw ok, full fails): {int((fr & ~ff & img).sum())}, kept raw: "
                  f"{int((fr & ~ff & keep_raw).sum())} | recovered by an image channel: {int((~fr & ff & img).sum())}, "
                  f"kept corrected: {int((~fr & ff & img & ~keep_raw).sum())}")
            calm = img & (d_r <= smc.FP_UNAFFECTED)
            print(f"  left alone (raw field within {smc.FP_UNAFFECTED} of the fingerprint): {int(calm.sum())} units, raw success there {mean(fr, calm):.3f}")
            for lo, hi in ((0, 0.15), (0.15, smc.FP_LOST), (smc.FP_LOST, 9)):
                m = (d_c > lo) & (d_c <= hi)
                print(f"  corrected distance in ({lo}, {hi}]: full success {mean(ff, m):.2f} (n={int(m.sum())})")
        # minimal intervention (smc.decide_v2): floor = the same task's residual in the training setup; the MEASURED
        # camera predicts how a translated-goal policy's sub-goal must move (needs cam_est_pose in the rows)
        import wrapper_libero as W
        floor = {r["task_id"]: r["smc_raw"]["residual_deg"] for r in rows if r["cond"] == "nominal"}
        v2, why = [], {}
        for r in U:
            pred = None
            if r.get("cam_applied") and r.get("cam_est_pose") and r["smc_cor"]["goal"] is not None:
                e, n = np.array(r["cam_est_pose"]), np.array(r["cam_nom_pose"])
                pred = smc.predicted_raw_goal(r["smc_cor"]["goal"], e[:3], W.rotvec_to_mat(e[3:]), n[:3], W.rotvec_to_mat(n[3:]))
            c, reason = smc.decide_v2(r["smc_raw"], r["smc_cor"], floor.get(r["task_id"], float("nan")), pred)
            v2.append(c)
            why[reason[:1]] = why.get(reason[:1], 0) + 1
        v2 = np.array(v2)
        chosen2 = np.where(v2 == "raw", raw, full)
        print(f"minimal intervention (decide_v2): success {chosen2.mean():.3f} | broken units kept raw "
              f"{int((raw & ~full & (v2 == 'raw')).sum())}/{int((raw & ~full).sum())} | recovered units kept corrected "
              f"{int((~raw & full & (v2 == 'cor')).sum())}/{int((~raw & full).sum())} | cases {why}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nominals", default="")
    ap.add_argument("--family", default="oft", choices=["oft", "openvla", "octo", "pi05", "pi0", "smolvla"])
    ap.add_argument("--octo_calib", default="")
    ap.add_argument("--n_tasks", type=int, default=3)
    ap.add_argument("--extended", action="store_true")
    a = ap.parse_args()
    (run if a.mode == "run" else report)(a)
