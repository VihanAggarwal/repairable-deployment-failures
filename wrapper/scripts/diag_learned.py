"""Learned-sensing diagnostics from wrap_run rows: per task and arm, success, TRUE calibration error, learned-mask
quality and reprojection coverage; then the 2x2 {calibrated pose, true pose} x {depth mode} success table.

    python diag_learned.py --rows <wraps>/wrap_openvla_oft.jsonl [--tasks 768,800,...]

Reading guide (2026-09-18 root-cause analysis):
  cam_est_err above ~5 cm on a lost task      -> calibration under the learned mask is the cause there
  mask_area_ratio away from 1.00 (uniformly)  -> the segmenter's boundary bias moves the camera along its axis
  *_truepose arm succeeds, deployable fails    -> calibration; both fail but sim-depth truepose succeeds -> depth
"""
import argparse
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--tasks", default="")
ap.add_argument("--category", default="camera")
a = ap.parse_args()
only = {int(x) for x in a.tasks.split(",") if x.strip()}
rows = [json.loads(l) for l in open(a.rows) if l.strip()]
rows = [r for r in rows if r.get("category") == a.category and r["arm"].startswith("full") and (not only or r["task_id"] in only)]
last = {}
for r in rows:
    last[(r["task_id"], r["arm"])] = r
arms = sorted({k[1] for k in last})
print(f"{'task':>5} {'arm':<22} ok   est err cm/deg   true shift cm/deg  mask IoU  area  coverage  cal method")
for (tid, arm), r in sorted(last.items()):
    e, s = r.get("cam_est_err_cm_deg"), r.get("cam_true_shift_cm_deg")
    f = lambda v, n=2: "    -" if v is None else f"{v:5.{n}f}"
    print(f"{tid:>5} {arm:<22} {'Y' if r['success'] else '.'}   "
          f"{f(e and e[0], 1)} /{f(e and e[1], 1)}     {f(s and s[0], 1)} /{f(s and s[1], 1)}     {f(r.get('mask_iou'))} {f(r.get('mask_area_ratio'))}"
          f"    {f(r.get('reproj_coverage'))}  {r.get('cal_method', '')}")
print()
by = defaultdict(list)
for (tid, arm), r in last.items():
    by[arm].append(r)
common = set.intersection(*[{r["task_id"] for r in v} for v in by.values()]) if by else set()
print(f"success per arm (all rows | the {len(common)} tasks every arm has run)")
for arm in arms:
    v = by[arm]
    c = [r for r in v if r["task_id"] in common]
    errs = sorted(r["cam_est_err_cm_deg"][0] for r in v if r.get("cam_est_err_cm_deg"))
    med = f", median est err {errs[len(errs) // 2]:.1f} cm, above 5 cm on {sum(x > 5 for x in errs)}/{len(errs)}" if errs else ""
    print(f"  {arm:<22} {sum(r['success'] for r in v):>2}/{len(v):<3} | {sum(r['success'] for r in c):>2}/{len(c):<3}{med}")
