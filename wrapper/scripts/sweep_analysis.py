"""Does a policy-free, self-measured score predict which perturbations the wrapper recovers, better than the category
label and a policy-internal probe?

Unit of analysis: (policy, base task, condition) where the policy succeeds on the unperturbed task and FAILS on the
perturbed one (raw). Outcome: the self-measured wrapper recovers it (full succeeds). Predictors:
  recover  RECOVERABILITY = ident x invert (pre-registered 2026-09-17 on OFT, to be tested on held-out families):
           ident   sigma_min(condition block, perturbed) / sigma_min(SAME block, unperturbed), clipped to [0, 1]:
                   can the robot still measure the shift?
           invert  can the wrapper's inversion reconstruct the training observation from what it measured?
                   camera: training-view coverage of the reprojection from the self-calibrated pose;
                   light: body shading consistency vs the unperturbed state (a global gain/offset is the only
                   lighting change the photometric channel inverts); noise: share of the added noise the denoiser
                   removes; joint: 1 (servoing restores the encoders exactly); scene content: 0 (no inversion exists).
  ident    the identifiability factor alone
  score    v1 relative identifiability (normalised by the CAMERA block for every family; kept for the record, it is
           the predictor that failed on OFT with AUROC 0.48)
  label    category label: camera, light, noise, joint -> invertible (1); background -> not (0)
  probe    policy-internal: minus the change of the first action chunk vs the unperturbed first chunk
Reported: AUROC per policy and pooled with bootstrap CIs; within-type Spearman across levels; the override
conditions; identifiable-but-unrecovered cases; and the paired-comparison noise floor (episodes whose outcome changed
although the wrapper applied nothing).
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_rows, load_shards  # noqa: E402
from sweep_conditions import BY_NAME, INVERTIBLE_LABEL  # noqa: E402


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(allv):                                   # average ranks for ties
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        v = auroc(s[i], y[i])
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def recoverability(row):
    """ident x invert from a scores_v2 row (see the module docstring). Every input is policy-free and self-measured."""
    ref = row.get("sigma_min_ref")
    if ref is None:
        ident = float("nan")
    elif ref <= 0:                                     # the body never responds to this block: unmeasurable, not unknown
        ident = 0.0
    else:
        ident = float(np.clip(row["sigma_min"] / ref, 0.0, 1.0))
    fam = row["family"]
    if fam in ("camera", "nominal"):
        invert = float(row.get("coverage", np.nan))
    elif fam == "joint":
        invert = 1.0
    elif fam == "noise":
        invert = float(row.get("noise_inv", np.nan))
    elif fam == "light":
        d, d0 = row.get("photo_d"), row.get("photo_d_ref")
        invert = float(min(d, d0) / max(d, d0)) if d and d0 and not (np.isnan(d) or np.isnan(d0)) and max(d, d0) > 0 else float("nan")
    else:                                              # scene content: the wrapper has no inversion for it
        invert = 0.0
    return dict(ident=ident, invert=invert, recover=ident * invert)


def load_scores(out_dir):
    """scores_v2 (own-block reference + invertibility components) when present, else the v1 file. Returns
    (rows by (task, cond), predictors by (task, cond), version)."""
    v2 = os.path.join(out_dir, "scores_v2.jsonl")
    if os.path.exists(v2) and load_rows(v2):
        rows = {(r["task_id"], r["cond"]): r for r in load_rows(v2)}
        scale = {tid: r["sigma_min"] for (tid, c), r in rows.items() if c == "nominal" and r["sigma_min"] > 0}
        pred = {}
        for k, r in rows.items():
            rc = recoverability(r)
            pred[k] = dict(score=r["sigma_min"] / scale[k[0]] if k[0] in scale else np.nan, **rc)
        return rows, pred, "v2"
    rows = {(r["task_id"], r["cond"]): r for r in load_rows(os.path.join(out_dir, "scores.jsonl"))}
    scale = {tid: r["sigma_min"] for (tid, c), r in rows.items() if c == "nominal" and r["sigma_min"] > 0}
    pred = {k: dict(score=r["sigma_min"] / scale[k[0]], ident=np.nan, invert=np.nan, recover=np.nan)
            for k, r in rows.items() if k[0] in scale}
    return rows, pred, "v1"


def applied_nothing(row):
    """True when the full arm left the observation untouched (no channel fired)."""
    return (not row.get("state_applied") and not row.get("cam_applied") and not row.get("photo_applied")
            and not row.get("denoise_applied"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--exclude", default="octo", help="policies whose episodes are not evidence (comma separated)")
    a = ap.parse_args()
    scores, pred, version = load_scores(a.out_dir)
    print(f"scores: {version} ({len(scores)} rows)" + ("" if version == "v2" else
          "  WARNING: v1 normalises every family by the camera block; re-run the score cell for scores_v2"))
    rel = {k: p["score"] for k, p in pred.items()}
    report = {"policies": {}}
    pooled = defaultdict(list)
    for path in sorted({os.path.join(a.out_dir, os.path.basename(p).split(".")[0] + ".jsonl")
                        for p in glob.glob(os.path.join(a.out_dir, "episodes_*.jsonl"))}):
        rows = load_shards(path)                          # the policy's main file plus its per-session shards
        if not rows:
            continue
        pol = rows[0]["policy"]
        if pol in a.exclude.split(","):      # a policy that cannot do the clean task has no recovery to measure
            print(f"skipping {pol}: excluded")
            continue
        ep = {}
        for r in rows:                                    # repeats (rep) average into one row per (task, cond, arm)
            k = (r["task_id"], r["cond"], r["arm"])
            if k in ep:
                n = ep[k]["_n"]
                ep[k]["success"] = (ep[k]["success"] * n + float(r["success"])) / (n + 1)
                ep[k]["_n"] = n + 1
            else:
                ep[k] = dict(r, success=float(r["success"]), _n=1)
        for r in ep.values():
            r["success"] = r["success"] >= 0.5 if r["_n"] == 1 else r["success"]
        tasks = sorted({r["task_id"] for r in rows})
        nominal_ok = {t: ep.get((t, "nominal", "raw"), {}).get("success") for t in tasks}
        print(f"\n===== {pol} =====  unperturbed success on base tasks: "
              f"{sum(bool(v) for v in nominal_ok.values())}/{len(tasks)}")
        print(f"{'condition':<15}{'label':>6}{'recover':>8}{'ident':>7}{'invert':>7}{'v1':>7}{'raw':>6}{'full':>6}  n")
        nm = lambda v: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
        for cname, cond in BY_NAME.items():
            raw = [ep[(t, cname, 'raw')]["success"] for t in tasks if (t, cname, "raw") in ep]
            full = [ep[(t, cname, 'full')]["success"] for t in tasks if (t, cname, "full") in ep]
            pp = [pred.get((t, cname), {}) for t in tasks]
            if raw or full:
                print(f"{cname:<15}{INVERTIBLE_LABEL[cond['family']]:>6}"
                      f"{nm([p.get('recover', np.nan) for p in pp]):>8.3f}{nm([p.get('ident', np.nan) for p in pp]):>7.2f}"
                      f"{nm([p.get('invert', np.nan) for p in pp]):>7.2f}{nm([p.get('score', np.nan) for p in pp]):>7.3f}"
                      f"{np.mean(raw) if raw else float('nan'):>6.2f}{np.mean(full) if full else float('nan'):>6.2f}  {len(raw)}")
        flips = [(t, c) for (t, c, arm), r in ep.items() if arm == "full" and applied_nothing(r)
                 and (t, c, "raw") in ep and ep[(t, c, "raw")]["success"] != r["success"]]
        idle = [(t, c) for (t, c, arm), r in ep.items() if arm == "full" and applied_nothing(r) and (t, c, "raw") in ep]
        print(f"noise floor: {len(flips)} of {len(idle)} episodes changed outcome with NOTHING applied"
              + (f" -> {flips}" if flips else ""))
        units = []
        for t in tasks:
            if not nominal_ok.get(t):
                continue
            for cname, cond in BY_NAME.items():
                if cname == "nominal":
                    continue
                r, fu = ep.get((t, cname, "raw")), ep.get((t, cname, "full"))
                if r is None or fu is None or float(r["success"]) >= 0.5:
                    continue
                pp = pred.get((t, cname), {})
                units.append(dict(task=t, cond=cname, family=cond["family"], override=cond["override"],
                                  recovered=int(float(fu["success"]) >= 0.5), score=rel.get((t, cname), np.nan),
                                  recover=pp.get("recover", np.nan), ident=pp.get("ident", np.nan),
                                  invert=pp.get("invert", np.nan), label=INVERTIBLE_LABEL[cond["family"]],
                                  probe=-r["probe"] if r.get("probe") is not None else np.nan))
        y = [u["recovered"] for u in units]
        res = {"n_units": len(units), "recovered": int(sum(y))}
        print(f"\nprediction units (unperturbed success, perturbed raw failure): {len(units)}, recovered {sum(y)}")
        for name in ("recover", "ident", "score", "label", "probe"):
            x = [u[name] for u in units]
            if np.all(np.isnan(np.asarray(x, float))):
                continue
            au, ci = auroc(x, y), boot_ci(x, y)
            res[name] = dict(auroc=au, ci=ci)
            pooled[name] += list(zip(x, y))
            print(f"  AUROC {name:<8} {au:.3f}  [{ci[0]:.2f}, {ci[1]:.2f}]")
        use = "recover" if version == "v2" else "score"
        print(f"\nwithin-type: Spearman({use}, full success - raw success) across levels and tasks")
        for fam in ("camera", "light", "noise", "joint", "background"):
            xs, ds = [], []
            for t in tasks:
                for cname, cond in BY_NAME.items():
                    if cond["family"] != fam or (t, cname, "raw") not in ep or (t, cname, "full") not in ep:
                        continue
                    xs.append(pred.get((t, cname), {}).get(use, np.nan))
                    ds.append(int(ep[(t, cname, "full")]["success"]) - int(ep[(t, cname, "raw")]["success"]))
            ok = ~np.isnan(np.asarray(xs, float))
            rho = spearman(np.asarray(xs)[ok], np.asarray(ds)[ok]) if ok.sum() else float("nan")
            # the Spearman is dominated by episodes that never failed (diff 0); the units-only AUROC within the
            # family is the number a reviewer asks for ("does it predict within a perturbation type?")
            fu = [u for u in units if u["family"] == fam and not np.isnan(u[use])]
            au_f = auroc([u[use] for u in fu], [u["recovered"] for u in fu]) if fu else float("nan")
            res.setdefault("within_type", {})[fam] = dict(rho=rho, auroc=au_f, units=len(fu),
                                                         recovered=int(sum(u["recovered"] for u in fu)))
            print(f"  {fam:<11} rho {rho:+.2f}  (n {int(ok.sum())})   units {len(fu):>2}, recovered "
                  f"{int(sum(u['recovered'] for u in fu)):>2}, AUROC {au_f:.3f}")
        print("\noverride conditions: label call vs score vs outcome")
        for u in [u for u in units if u["override"]]:
            print(f"  task {u['task']} {u['cond']:<14} label {u['label']} recover {u['recover']:.3f} "
                  f"(ident {u['ident']:.2f} invert {u['invert']:.2f}) recovered {u['recovered']}")
        hi = [u for u in units if not np.isnan(u[use]) and u[use] > 0.5 and not u["recovered"]]
        print(f"\npredicted recoverable ({use} > 0.5) but NOT recovered: {len(hi)} -> "
              f"{sorted({u['cond'] for u in hi})}")
        lo = [u for u in units if not np.isnan(u[use]) and u[use] <= 0.5 and u["recovered"]]
        print(f"predicted NOT recoverable ({use} <= 0.5) but recovered: {len(lo)} -> {sorted({u['cond'] for u in lo})}")
        res["units"] = units
        report["policies"][pol] = res
    if pooled:
        print("\n===== pooled over policies =====")
        for name, xy in pooled.items():
            x, yy = [p[0] for p in xy], [p[1] for p in xy]
            au, ci = auroc(x, yy), boot_ci(x, yy)
            report[f"pooled_{name}"] = dict(auroc=au, ci=ci, n=len(xy))
            print(f"  AUROC {name:<8} {au:.3f}  [{ci[0]:.2f}, {ci[1]:.2f}]  (n {len(xy)})")
    json.dump(report, open(os.path.join(a.out_dir, "sweep_report.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
