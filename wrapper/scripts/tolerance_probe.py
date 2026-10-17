"""THE POLICY'S OWN CALIBRATION TOLERANCE, stage 1: a rollout-free dose-response (POST HOC, exploratory; touches no method code).

How wrong may the camera pose be before a frozen policy's OPENING ACTION leaves what it does under a perfect calibration?
For a base task under a camera shift the correction can undo (orbit 10 and 20 deg, coverage above 0.6), at the rest pose:
  reference  the first action chunk on the frame corrected with the TRUE camera pose
  dose       the same frame corrected with the true pose PLUS a known residual: 0.5 to 16 cm along each camera axis, 0.5 to
             8 deg about each camera axis; same instruction, same sampler seed, so only the residual differs
  distance   smc.field_distance(chunk, reference), the quantity whose frozen threshold FP_UNAFFECTED = 0.10 means
             "the policy behaves as in training"
  floor      the reference queried again (same seed): what the distance reads when nothing changed
The knee (smallest dose whose median distance exceeds 0.10) is the policy's tolerance in centimetres and degrees. The
robot's own calibration error (0.07 to 0.54 cm from 7 to 1 poses, 2.8 cm under learned masks, 19.6 cm on DROID) is then read
off the same axis. No episode is rolled out. Single-pose distance is a weaker outcome proxy than the 7-probe fingerprint.

    python tolerance_probe.py run --family oft --n_tasks 10 --out_dir .../tolerance --nominals ...
    python tolerance_probe.py report --out_dir .../tolerance
"""
import argparse
import glob
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONDS = ("orbit_az10", "orbit_az20")
CM = (0.5, 1, 2, 4, 8, 16)
DEG = (0.5, 1, 2, 4, 8)
MARKS_CM = (("7-pose self-calibration", 0.07), ("3-pose (the method)", 0.28), ("1 pose", 0.54), ("learned sensing, 30 tasks", 12.6), ("DROID real images", 19.6))


def probe_unit(adapter, env, suite, task, cond, nominal, R, W, smc):
    seed = task["task_id"]
    obs, _ = R.start(env, suite, task, cond, seed)
    can = W.Canonicalizer(env, nominal)
    Ct, Rt = can.true_camera()
    copy = lambda o: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.items()}

    def act(C, Rm):
        o = copy(obs)
        o["agentview_image"], cov = can.canonical_image(o, C=C, R=Rm)
        adapter.new_episode(task["instruction"], seed)                       # same seed: only the frame differs
        return np.mean([np.asarray(q, float)[:3] for q in adapter.chunk(o)], 0), float(cov)

    ref, cov = act(Ct, Rt)
    again, _ = act(Ct, Rt)
    adapter.new_episode(task["instruction"], seed)
    raw = np.mean([np.asarray(q, float)[:3] for q in adapter.chunk(copy(obs))], 0)
    doses = []
    for axis in range(3):
        for d in CM:
            a, _ = act(Ct + Rt[:, axis] * d / 100.0, Rt)
            doses.append(dict(kind="cm", axis=axis, dose=d, distance=smc.field_distance([a], [ref])))
        for d in DEG:
            r = np.zeros(3)
            r[axis] = np.deg2rad(d)
            a, _ = act(Ct, Rt @ W.rotvec_to_mat(r))
            doses.append(dict(kind="deg", axis=axis, dose=d, distance=smc.field_distance([a], [ref])))
    return dict(coverage=cov, floor=smc.field_distance([again], [ref]), uncorrected=smc.field_distance([raw], [ref]),
                shift_cm_deg=list(W.pose_error(Ct, Rt, can.Cn, can.Rn)), doses=doses)


def run(a):
    import smc
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
    os.makedirs(a.out_dir, exist_ok=True)
    done = {(r["task_id"], r["cond"]) for p in glob.glob(os.path.join(a.out_dir, f"tolerance_{adapter.name}*.jsonl")) for r in load_rows(p)}
    path = os.path.join(a.out_dir, f"tolerance_{adapter.name}.{socket.gethostname()[-6:]}.jsonl")     # one file per runtime
    todo = [(S.BY_NAME[c], t) for c in CONDS for t in tasks if (t["task_id"], c) not in done]
    print(f"{adapter.name}: {len(todo)} units to probe, {3 * (len(CM) + len(DEG)) + 3} policy queries each", flush=True)
    t0 = time.time()
    for n, (cond, t) in enumerate(todo):
        env = make_env(suite, t["task_id"])
        try:
            row = probe_unit(adapter, env, suite, t, cond, nominals[t["scene"]], R, W, smc)
        finally:
            env.close()
        append_row(path, dict(policy=adapter.name, task_id=t["task_id"], cond=cond["name"], **row))
        d = {(x["kind"], x["dose"]): [] for x in row["doses"]}
        for x in row["doses"]:
            d[(x["kind"], x["dose"])].append(x["distance"])
        el = (time.time() - t0) / 60
        print(f"{adapter.name} {t['task_id']:5d} {cond['name']:<11} floor {row['floor']:.3f} uncorrected {row['uncorrected']:.2f} | "
              f"1 cm {np.median(d[('cm', 1)]):.3f}  4 cm {np.median(d[('cm', 4)]):.3f}  16 cm {np.median(d[('cm', 16)]):.3f} | "
              f"1 deg {np.median(d[('deg', 1)]):.3f}  8 deg {np.median(d[('deg', 8)]):.3f} | {n + 1}/{len(todo)} "
              f"eta {el / (n + 1) * (len(todo) - n - 1):.0f} min", flush=True)


def report(a):
    import smc
    from plus_common import load_rows
    by = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "tolerance_*.jsonl"))):
        for r in load_rows(p):
            by.setdefault(r["policy"], {})[(r["task_id"], r["cond"])] = r
    for pol, rows in by.items():
        rows = list(rows.values())
        print(f"\n===== {pol} =====  {len(rows)} units | noise floor (same frame, same seed) median {np.median([r['floor'] for r in rows]):.3f} | "
              f"uncorrected frame median {np.median([r['uncorrected'] for r in rows]):.2f} | coverage median {np.median([r['coverage'] for r in rows]):.2f}")
        for kind, grid, unit in (("cm", CM, "cm"), ("deg", DEG, "deg")):
            knee = None
            print(f"  injected residual ({unit})   median distance   [25%, 75%]   share of units over {smc.FP_UNAFFECTED}")
            for d in grid:
                per_unit = [np.median([x["distance"] for x in r["doses"] if x["kind"] == kind and x["dose"] == d]) for r in rows]
                q = np.quantile(per_unit, [0.25, 0.5, 0.75])
                over = float(np.mean(np.asarray(per_unit) > smc.FP_UNAFFECTED))
                print(f"    {d:>5}                      {q[1]:.3f}          [{q[0]:.3f}, {q[2]:.3f}]        {over:.2f}")
                if knee is None and q[1] > smc.FP_UNAFFECTED:
                    knee = d
            print(f"  TOLERANCE ({unit}): " + (f"the median distance first exceeds {smc.FP_UNAFFECTED} at {knee} {unit}" if knee
                  else f"never exceeds {smc.FP_UNAFFECTED} up to {grid[-1]} {unit}"))
            if kind == "cm":
                for name, e in MARKS_CM:
                    side = "no knee in range" if knee is None else ("INSIDE the tolerance" if e < knee else "OUTSIDE the tolerance")
                    print(f"      {name:<26} {e:>5} cm  -> {side}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nominals", default="")
    ap.add_argument("--family", default="oft", choices=["oft", "openvla", "octo", "pi05", "pi0", "smolvla"])
    ap.add_argument("--octo_calib", default="")
    ap.add_argument("--n_tasks", type=int, default=10)
    a = ap.parse_args()
    (run if a.mode == "run" else report)(a)
