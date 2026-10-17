"""POST HOC (2026-09-24, reviewer round 1): the recoverability score against category-based and shift-size predictors,
on the SAME prediction units as sweep_analysis.py, restricted to the 24 standard conditions (S.CONDITIONS).

Unit u = (policy, base task, condition) with the policy solving the unperturbed task and FAILING the perturbed one (raw);
repaired(u) = 1 when the self-measured wrapper (arm `full`) succeeds. Repeats average into one row per (task, cond, arm)
exactly as in sweep_analysis.main.

Predictors (all known before the rollout):
  score        recover = ident x invert (the registered score)
  ident        identifiability factor alone;  invert: invertibility factor alone
  label        INVERTIBLE_LABEL (camera/light/noise/joint = 1, background = 0), the benchmark's routing label
  cat_rate     per-family repair rate FIT ON OFT (the development family) and applied to a held-out family
  cond_rate    per-condition repair rate fit on OFT (finer than the family; condition names are shared)
  -transl, -rot  minus the true camera shift (cm, deg), camera units only
AUROC = Mann-Whitney with ties counted 1/2, NaN excluded. CIs: percentile bootstrap, B = 2000, seed 0, resampling units,
and resampling base tasks (cluster bootstrap). Paired difference AUROC(score) - AUROC(baseline): both on the same
resample, units with either predictor NaN dropped first.
Also the phantom-repair sensitivity: repaired*(u) = repaired(u) AND some channel applied. One-sided: drop repaired
units with nothing applied; symmetric: drop every unit whose full row applied nothing, whatever the outcome.

    uv run --no-project --with numpy python wrapper/scripts/score_baselines.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import sweep_conditions as S  # noqa: E402
from plus_common import load_shards  # noqa: E402
from sweep_analysis import applied_nothing, auroc, load_scores  # noqa: E402

SWEEP = os.path.join(REPO, "results", "drive_vihan", "sweep")
FAMS = {"OpenVLA-OFT": "openvla_oft", "pi0.5": "pi05", "OpenVLA (LIBERO-10 fine-tuned)": "openvla"}
STD = [c["name"] for c in S.CONDITIONS]


def units_for(pol, pred):
    rows = load_shards(os.path.join(SWEEP, f"episodes_{pol}.jsonl"))
    ep = {}
    for r in rows:
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
    ok = {t: ep.get((t, "nominal", "raw"), {}).get("success") for t in tasks}
    U = []
    for t in tasks:
        if not ok.get(t):
            continue
        for c in STD:
            if c == "nominal":
                continue
            r, fu = ep.get((t, c, "raw")), ep.get((t, c, "full"))
            if r is None or fu is None or float(r["success"]) >= 0.5:
                continue
            p = pred.get((t, c), {})
            fam = S.BY_NAME[c]["family"]
            sh = fu.get("cam_true_shift_cm_deg") or fu.get("cam_shift_cm_deg") or r.get("cam_shift_cm_deg")
            U.append(dict(task=t, cond=c, family=fam, repaired=int(float(fu["success"]) >= 0.5), nothing=applied_nothing(fu),
                          score=p.get("recover", np.nan), ident=p.get("ident", np.nan), invert=p.get("invert", np.nan),
                          label=float(S.INVERTIBLE_LABEL[fam]),
                          ntransl=-sh[0] if (fam == "camera" and sh) else np.nan, nrot=-sh[1] if (fam == "camera" and sh) else np.nan))
    return U, sum(bool(v) for v in ok.values()), len(tasks)


def fit_rates(U, key):
    g = defaultdict(list)
    for u in U:
        g[u[key]].append(u["repaired"])
    return {k: float(np.mean(v)) for k, v in g.items()}


def boot(U, fn, B=2000, seed=0, cluster=False):
    rng = np.random.default_rng(seed)
    if cluster:
        by = defaultdict(list)
        for u in U:
            by[u["task"]].append(u)
        keys = list(by)
    vals = []
    for _ in range(B):
        if cluster:
            s = [u for k in rng.choice(len(keys), len(keys)) for u in by[keys[k]]]
        else:
            s = [U[i] for i in rng.integers(0, len(U), len(U))]
        v = fn(s)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def au(U, k):
    return auroc([u[k] for u in U], [u["repaired"] for u in U])


def p_le0(U, a, b, B=2000, seed=0, cluster=False):
    """Bootstrap share of resamples with AUROC(a) - AUROC(b) <= 0 (one-sided, a one-sided p for 'a beats b')."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for u in U:
        by[u["task"]].append(u)
    keys = list(by)
    d = []
    for _ in range(B):
        s = [u for k in rng.choice(len(keys), len(keys)) for u in by[keys[k]]] if cluster else [U[i] for i in rng.integers(0, len(U), len(U))]
        v = au(s, a) - au(s, b)
        if np.isfinite(v):
            d.append(v)
    return float(np.mean(np.asarray(d) <= 0))


def main():
    _, pred, _ = load_scores(SWEEP)
    fam_units = {}
    for nice, pol in FAMS.items():
        U, solved, nt = units_for(pol, pred)
        fam_units[nice] = U
        print(f"\n===== {nice}: base tasks {nt} ({solved} solved unperturbed), {len(U)} units, {sum(u['repaired'] for u in U)} repaired")
    oft = fam_units["OpenVLA-OFT"]
    cat, cond = fit_rates(oft, "family"), fit_rates(oft, "cond")
    print("\nOFT per-family repair rates (fit):", {k: round(v, 3) for k, v in cat.items()})
    for nice, U in fam_units.items():
        for u in U:
            u["cat_rate"] = cat.get(u["family"], np.nan)
            u["cond_rate"] = cond.get(u["cond"], np.nan)
        print(f"\n----- {nice} (n = {len(U)}, repaired {sum(u['repaired'] for u in U)})")
        for k in ("score", "ident", "invert", "label", "cat_rate", "cond_rate", "ntransl", "nrot"):
            V = [u for u in U if np.isfinite(u[k])]
            if len(V) < 5 or len({u['repaired'] for u in V}) < 2:
                continue
            a = au(V, k)
            ci_u = boot(V, lambda s: au(s, k))
            ci_t = boot(V, lambda s: au(s, k), cluster=True)
            line = f"  {k:<10} AUROC {a:.3f}  unit [{ci_u[0]:.2f}, {ci_u[1]:.2f}]  task-cluster [{ci_t[0]:.2f}, {ci_t[1]:.2f}]  (n {len(V)}, tasks {len({u['task'] for u in V})})"
            if k != "score":
                W = [u for u in V if np.isfinite(u["score"])]
                d = au(W, "score") - au(W, k)
                dci = boot(W, lambda s: au(s, "score") - au(s, k))
                dct = boot(W, lambda s: au(s, "score") - au(s, k), cluster=True)
                line += f"\n             score - {k}: {d:+.3f} unit [{dci[0]:+.2f}, {dci[1]:+.2f}] p(<=0) {p_le0(W, 'score', k):.3f}; task [{dct[0]:+.2f}, {dct[1]:+.2f}] p(<=0) {p_le0(W, 'score', k, cluster=True):.3f}"
            print(line)
        # label as a binary predictor: AUROC = (1 + TPR - FPR) / 2
        pos = [u for u in U if u["repaired"]]
        neg = [u for u in U if not u["repaired"]]
        tpr, fpr = np.mean([u["label"] for u in pos]), np.mean([u["label"] for u in neg])
        print(f"  label: TPR {tpr:.3f} FPR {fpr:.3f} -> (1 + TPR - FPR) / 2 = {(1 + tpr - fpr) / 2:.3f}")
        # phantom repairs
        ph = [u for u in U if u["repaired"] and u["nothing"]]
        nothing = [u for u in U if u["nothing"]]
        one = [u for u in U if not (u["repaired"] and u["nothing"])]
        sym = [u for u in U if not u["nothing"]]
        print(f"  phantom repairs (repaired, nothing applied): {len(ph)} of {sum(u['repaired'] for u in U)}; units with nothing applied {len(nothing)}")
        for nm_, V in (("one-sided", one), ("symmetric", sym)):
            V = [u for u in V if np.isfinite(u["score"])]
            print(f"  {nm_:<9} AUROC {au(V, 'score'):.3f} [{boot(V, lambda s: au(s, 'score'))[0]:.2f}, {boot(V, lambda s: au(s, 'score'))[1]:.2f}]  (n {len(V)}, repaired {sum(u['repaired'] for u in V)})")
        by = defaultdict(list)
        for u in U:
            by[u["family"]].append(u)
        print("  per family:", {f: (len(v), sum(u['repaired'] for u in v), round(au(v, 'score'), 3) if len({u['repaired'] for u in v}) > 1 else None) for f, v in by.items()})


def _h(a, b):
    return float(a > b) + 0.5 * float(a == b)


def pair_split():
    """POST HOC (2026-09-24, round 2, pair lemma). With U1 = units with ident = 1 and U2 = units with ident < 1,
    AUROC(recover) - AUROC(invert) = (1/PN) sum over positive-negative pairs with at least one unit in U2 of
    [h(recover_i, recover_j) - h(invert_i, invert_j)], h = 1[>] + 1/2 1[=]; pairs inside U1 contribute exactly 0.
    Prints |U2| per family and splits the U2 pair contributions by whether a look-away unit is in the pair.

        uv run --no-project --with numpy python wrapper/scripts/score_baselines.py --pairs
    """
    _, pred, _ = load_scores(SWEEP)
    for nice, pol in FAMS.items():
        U, _, _ = units_for(pol, pred)
        V = [u for u in U if np.isfinite(u["score"]) and np.isfinite(u["invert"]) and np.isfinite(u["ident"])]
        pos = [u for u in V if u["repaired"]]
        neg = [u for u in V if not u["repaired"]]
        P, N = len(pos), len(neg)
        u2 = [u for u in V if u["ident"] < 1]
        tot = {"all": 0.0, "look": 0.0, "other": 0.0, "u1u1": 0.0}
        cnt = {"look": 0, "other": 0, "u1u1": 0}
        by_type = {}
        for i in pos:
            for j in neg:
                c = _h(i["score"], j["score"]) - _h(i["invert"], j["invert"])
                tot["all"] += c
                if i["ident"] >= 1 and j["ident"] >= 1:
                    k = "u1u1"
                elif i["cond"].startswith("look_away") or j["cond"].startswith("look_away"):
                    k = "look"
                else:
                    k = "other"
                tot[k] += c
                cnt[k] += 1
                typ = ("pos in U2, neg in U1" if i["ident"] < 1 and j["ident"] >= 1 else
                       "pos in U1, neg in U2" if i["ident"] >= 1 and j["ident"] < 1 else
                       "both in U2" if i["ident"] < 1 else None)
                if typ:
                    by_type[typ] = by_type.get(typ, 0.0) + c
        la = [u for u in u2 if u["cond"].startswith("look_away")]
        print(f"{nice} by pair type (sum / PN; the first can only be <= 0, the second only >= 0):",
              {k: round(v / (P * N), 4) for k, v in by_type.items()})
        print(f"{nice}: n {len(V)} (P {P}, N {N}); |U2| {len(u2)} (look-away {len(la)}, repaired in U2 {sum(u['repaired'] for u in u2)})")
        print(f"  AUROC recover {au(V, 'score'):.4f}  invert {au(V, 'invert'):.4f}  diff {au(V, 'score') - au(V, 'invert'):+.4f}")
        print(f"  pair sum / PN: all {tot['all'] / (P * N):+.4f}  [U1xU1 {tot['u1u1'] / (P * N):+.4f} over {cnt['u1u1']} pairs]"
              f"  look-away {tot['look'] / (P * N):+.4f} over {cnt['look']} pairs  other {tot['other'] / (P * N):+.4f} over {cnt['other']} pairs")
        byc = defaultdict(list)
        for u in u2:
            byc[u["family"]].append(u)
        print("  U2 by family:", {f: (len(v), sum(u['repaired'] for u in v)) for f, v in byc.items()})
        print("  U2 ident range:", (round(min(u['ident'] for u in u2), 3), round(max(u['ident'] for u in u2), 3)) if u2 else None)


def _fast_au(s, y):
    """Mann-Whitney AUROC with ties 1/2 (same value as sweep_analysis.auroc), vectorised for the bootstrap."""
    from scipy.stats import rankdata
    s, y = np.asarray(s, float), np.asarray(y, int)
    P, N = int(y.sum()), int(len(y) - y.sum())
    if P == 0 or N == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


FID_COND = {"orbit_az10": "az+10_x1.0", "orbit_az20": "az+20_x1.0", "orbit_az30": "az+30_x1.0", "pullback_1.3": "az+0_x1.3",
            "look_away_35": "tilt35", "look_away_55": "tilt55"}


def extensions(B=2000, seed=0):
    """POST HOC (2026-09-24, round 3, item F02), dagger. Three more baselines beside the score, same prediction units:
      cov_true   true-pose coverage (share of the training view the camera sees at the TRUE pose; upper bound on the
                 self-measured coverage). Logged only policy-free in results/local/fidelity.jsonl (`filled`) for the three
                 envelope base tasks (1167, 1208, 1442) at placements az +10/+20/+30 x1.0, az 0 x1.3, tilt 35/55: so only
                 the camera units on those tasks and conditions (orbit 45 is not on the envelope grid) can be scored.
      img_true   image statistic: minus the mean |pixel error| (0-255) of the true-pose re-rendering against a perfect
                 training-view render (`err_total_ref`, same file, same restriction). NOT available for the frame the
                 wrapper actually fed the policy: no SSIM / pixel-error field is logged in any rollout row.
      minii      min(ident, invert), beside the product (all units).
      nrot       minus the true rotation magnitude of the camera shift (deg), camera units (as in main()).
    AUROC with unit bootstrap (B resamples of units); paired difference AUROC(score) - AUROC(x) on the same resample,
    units with either predictor NaN dropped first; p(<= 0) = share of resamples with difference <= 0."""
    import json as _json
    fid = {(r["task_id"], r["cond"]): r for r in map(_json.loads, open(os.path.join(REPO, "results", "local", "fidelity.jsonl"))) if r}
    _, pred, _ = load_scores(SWEEP)
    rng = np.random.default_rng(seed)
    for nice, pol in FAMS.items():
        U, _, _ = units_for(pol, pred)
        for u in U:
            f = fid.get((u["task"], FID_COND.get(u["cond"], "")))
            u["cov_true"] = f["filled"] if f else np.nan
            u["img_true"] = -f["err_total_ref"] if f else np.nan
            u["minii"] = min(u["ident"], u["invert"]) if np.isfinite(u["ident"]) and np.isfinite(u["invert"]) else np.nan
        print(f"\n===== {nice}: {len(U)} units, {sum(u['repaired'] for u in U)} repaired")
        for k in ("minii", "nrot", "cov_true", "img_true"):
            V = [u for u in U if np.isfinite(u[k]) and np.isfinite(u["score"])]
            y = np.array([u["repaired"] for u in V])
            if len(V) < 5 or len(set(y)) < 2:
                print(f"  {k:<9} n {len(V)} (repaired {int(y.sum()) if len(V) else 0}): too few units or one class, not computed")
                continue
            a, sc = np.array([u[k] for u in V]), np.array([u["score"] for u in V])
            A, S0 = _fast_au(a, y), _fast_au(sc, y)
            bs, bd = [], []
            for _ in range(B):
                i = rng.integers(0, len(V), len(V))
                x, s_ = _fast_au(a[i], y[i]), _fast_au(sc[i], y[i])
                if np.isfinite(x) and np.isfinite(s_):
                    bs.append(x); bd.append(s_ - x)
            ci, dci = np.percentile(bs, [2.5, 97.5]), np.percentile(bd, [2.5, 97.5])
            print(f"  {k:<9} AUROC {A:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]   score on the same units {S0:.3f}   "
                  f"score - {k} {S0 - A:+.3f} [{dci[0]:+.2f}, {dci[1]:+.2f}] p(<=0) {np.mean(np.asarray(bd) <= 0):.3f}   "
                  f"(n {len(V)}, repaired {int(y.sum())}, tasks {len({u['task'] for u in V})})")


if __name__ == "__main__":
    if "--pairs" in sys.argv:
        pair_split()
    elif "--ext" in sys.argv:
        extensions()
    else:
        main()
