"""REGISTERED (frozen 2026-09-25, before any row): tests of wrapper/prereg/2026-09-25_openvla_extension.md on run file 21.

Reads the extension's rows (plus/sweep_openvla_ext, pulled to EXT below) and the frozen policy-free scores
(scores_v2.jsonl of the registered sweep; vendor/score_selfmeasured.jsonl). No constant here is fit on the new rows.

  D = ident(self-calibrated pose) x invert x 1[cal_final_score <= 0.547] on camera units; = registered score elsewhere.
  H1: within-category AUROC of D on the new units > 0.5, permutation test shuffling outcomes within category
      (10,000 permutations, seed 0), one-sided alpha 0.05.
  H2: the same pooled with the 60 registered OpenVLA units.
  H3: the residual flags calibration failure (> 2 cm) on the new camera units with AUROC >= 0.90.
  H4: AUROC(D) >= AUROC(registered score) - 0.05 on the new units (point estimate).
  Underpowered (reported as such): fewer than 2 new base tasks enter, or a class of a test has under 5 units.
Baselines beside each: category label, category rate fit on OFT, condition rate fit on OFT with category fallback.

    uv run --no-project --with numpy python wrapper/scripts/prereg_openvla_ext.py [EXT_DIR]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_baselines as SB  # noqa: E402
import within_category as WC  # noqa: E402
from plus_common import load_shards  # noqa: E402
from sweep_analysis import auroc, load_scores  # noqa: E402

EXT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/vihan/Downloads/local_data/plus/sweep_openvla_ext"
SM = os.path.join(HERE, "..", "..", "vendor", "score_selfmeasured.jsonl")
TAU = 0.547
NEW_TASKS = {1101, 1146, 1255, 1297, 1325, 1371}


def perm_within(U, key, B=10000, seed=0):
    obs, n = WC.strat_auroc(U, key, "family")
    rng = np.random.default_rng(seed)
    by = {}
    for u in U:
        by.setdefault(u["family"], []).append(u)
    ge = 0
    for _ in range(B):
        V = []
        for us in by.values():
            y = rng.permutation([u["repaired"] for u in us])
            V += [dict(u, repaired=int(v)) for u, v in zip(us, y)]
        ge += WC.strat_auroc(V, key, "family")[0] >= obs
    return obs, n, (ge + 1) / (B + 1)


def main():
    _, pred, _ = load_scores(SB.SWEEP)
    sm = {(r["task_id"], r["cond"]): r for r in map(json.loads, open(SM))}
    oft = SB.units_for("openvla_oft", pred)[0]
    cat, cond = SB.fit_rates(oft, "family"), SB.fit_rates(oft, "cond")
    reg = SB.units_for("openvla", pred)[0]

    def full_camera_rows(d):
        out = {}
        for r in load_shards(os.path.join(d, "episodes_openvla.jsonl")):
            if r.get("arm") == "full" and r.get("family") == "camera":
                out.setdefault((r["task_id"], r["cond"]), r)
        return out

    rows_reg = full_camera_rows(SB.SWEEP)
    SB.SWEEP = EXT
    new, n_ok, n_tasks = SB.units_for("openvla", pred)
    new = [u for u in new if u["task"] in NEW_TASKS]
    n_ok = len({u["task"] for u in new}) if new else 0       # solved new tasks that contribute units
    rows = full_camera_rows(EXT)
    for U, R in ((reg, rows_reg), (new, rows)):
        for u in U:
            k = (u["task"], u["cond"])
            u["cat_rate"] = cat.get(u["family"], np.nan)
            u["cond_rate"] = cond.get(u["cond"], u["cat_rate"])
            u["D"] = u["score"]
            if u["family"] == "camera" and k in sm:
                fr = R.get(k)
                g = float(fr is None or fr.get("cal_final_score") is None or fr["cal_final_score"] <= TAU)
                u["D"] = sm[k]["ident_hat"] * u["invert"] * g
    print("new base tasks with rows: %d, solved unperturbed: %d, new units %d (repaired %d)" % (
        n_tasks, n_ok, len(new), sum(u["repaired"] for u in new)))
    under = n_ok < 2 or min(sum(u["repaired"] for u in new), sum(1 - u["repaired"] for u in new)) < 5
    if under:
        print("UNDERPOWERED by the registered rule; results below are descriptive")
    obs, npairs, p = perm_within(new, "D")
    print("H1 within-category AUROC(D) new units %.3f (pairs %d), permutation p %.4f -> %s" % (
        obs, npairs, p, "PASS" if p < 0.05 and not under else ("underpowered" if under else "FAIL")))
    for k in ("score", "invert", "cond_rate"):
        print("     beside: within-category %-9s %.3f" % (k, WC.strat_auroc(new, k, "family")[0]))
    pooled = [dict(u, task=("reg", u["task"])) for u in reg] + [dict(u, task=("new", u["task"])) for u in new]
    obs2, n2, p2 = perm_within(pooled, "D")
    print("H2 pooled within-category AUROC(D) %.3f (pairs %d), p %.4f -> %s" % (obs2, n2, p2, "PASS" if p2 < 0.05 else "FAIL"))
    cams = [(float(r["cam_est_err_cm_deg"][0]) > 2.0, float(r["cal_final_score"])) for k, r in rows.items()
            if k[0] in NEW_TASKS and r.get("cam_est_err_cm_deg") and r.get("cal_final_score") is not None]
    if cams and 0 < sum(c[0] for c in cams) < len(cams):
        a3 = auroc([c[1] for c in cams], [c[0] for c in cams])
        small = min(sum(c[0] for c in cams), len(cams) - sum(c[0] for c in cams)) < 5
        print("H3 residual flags calibration failure: AUROC %.3f (n %d, failures %d) -> %s" % (
            a3, len(cams), sum(c[0] for c in cams), "underpowered" if small else ("PASS" if a3 >= 0.90 else "FAIL")))
    else:
        print("H3 undetermined: one class only on the new camera units (n %d)" % len(cams))
    W = [u for u in new if np.isfinite(u["D"]) and np.isfinite(u["score"])]
    aD = auroc([u["D"] for u in W], [u["repaired"] for u in W])
    aR = auroc([u["score"] for u in W], [u["repaired"] for u in W])
    v4 = "undetermined" if not np.isfinite(aD) else ("PASS" if aD >= aR - 0.05 else "FAIL")
    print("H4 AUROC(D) %.3f vs registered %.3f -> %s" % (aD, aR, v4))
    for k in ("label", "cat_rate", "cond_rate", "invert"):
        V = [u for u in W if np.isfinite(u[k])]
        print("     beside: AUROC %-9s %.3f" % (k, auroc([u[k] for u in V], [u["repaired"] for u in V])))


if __name__ == "__main__":
    main()
