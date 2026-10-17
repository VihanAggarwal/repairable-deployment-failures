"""POST HOC (2026-09-24, reviewer round 1): separation of the motion-free static residual (silhouette score, px units of
the residual as logged) between unshifted and camera-shifted units, per source.

  local      results/local/static_residual.jsonl  (policy-free, one rest pose; the 60 / 70 split of Prop. 3 and Fig. harm b)
  sweep      static_residual on the pi0.5 graded-sweep rows (results/drive_vihan/sweep/episodes_pi05*.jsonl, full_lazy)
  plus       static_residual on the pi0.5 LIBERO-Plus rows (results/drive_vihan/wraps/wrap_pi05.jsonl, full_lazy)
For each: max over unshifted units, min over shifted units, and the margins to the post hoc threshold tau = 1.5.

    uv run --no-project --with numpy python wrapper/scripts/residual_separation.py
"""
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import sweep_conditions as S  # noqa: E402
from plus_common import load_rows, load_shards  # noqa: E402

TAU = 1.5
FAM = {c["name"]: c["family"] for c in S.CONDITIONS}


def show(name, groups):
    un, sh = groups.get(False, []), groups.get(True, [])
    if not un or not sh:
        print(f"{name}: unshifted {len(un)}, shifted {len(sh)} (incomplete)")
        return
    print(f"{name}: unshifted n {len(un)} max {max(un):.3f} (margin to tau {TAU - max(un):.2f}); "
          f"shifted n {len(sh)} min {min(sh):.3f} (margin {min(sh) - TAU:.2f}); overlap {max(un) >= min(sh)}")


g = defaultdict(list)
for r in load_rows(os.path.join(REPO, "results", "local", "static_residual.jsonl")):
    g[r["family"] == "camera"].append(float(r["residual"]))
show("local (policy-free)", g)

ep = [r for r in load_shards(os.path.join(REPO, "results", "drive_vihan", "sweep", "episodes_pi05.jsonl"))
      if r.get("static_residual") is not None]
print("  sweep rows with a residual, by arm:", dict(Counter(r["arm"] for r in ep)))
g = defaultdict(list)
for r in ep:
    g[FAM.get(r["cond"]) == "camera"].append(float(r["static_residual"]))
show("sweep (pi0.5 rows)", g)

pl = [r for r in load_rows(os.path.join(REPO, "results", "drive_vihan", "wraps", "wrap_pi05.jsonl")) if r.get("static_residual") is not None]
print("  LIBERO-Plus rows with a residual, by (category, arm):", dict(Counter((r["category"], r["arm"]) for r in pl)))
g = defaultdict(list)
for r in pl:
    g[r["category"] == "camera"].append(float(r["static_residual"]))
show("LIBERO-Plus (pi0.5 rows)", g)

print("  sweep conds (family):", dict(Counter(FAM.get(r["cond"], "?" + r["cond"]) for r in ep)))
print(f"  sweep max residual {max(r['static_residual'] for r in ep):.3f}; LIBERO-Plus non-camera max {max(r['static_residual'] for r in pl):.3f}")
