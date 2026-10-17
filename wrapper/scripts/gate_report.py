"""The do-no-harm gate against the ungated wrapper, from wrap_run rows (arms raw / full / full_gated) or sweep episodes
(arms raw / full / gated). Pre-registered 2026-09-18 (RESULTS.md):

  H1 (harm)       among units the raw policy solves, the gated wrapper breaks at most HALF as many as the ungated one
                  (testable only if the ungated wrapper breaks at least 3)
  H2 (retention)  among units the raw policy fails, the gated wrapper recovers at least 80 percent of what the ungated
                  one recovers

    python gate_report.py --rows <file.jsonl> [--rows <file2.jsonl> ...]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plus_common import load_shards  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rows", action="append", required=True)
a = ap.parse_args()

for path in a.rows:
    rows = load_shards(path)                              # the file plus its per-session shards (plus_common)
    units = defaultdict(dict)
    for r in rows:
        arm = {"full_gated": "gated"}.get(r["arm"], r["arm"])
        if arm in ("raw", "full", "gated"):
            units[(r["task_id"], r.get("cond", ""), r.get("rep", 0))][arm] = r
    full3 = {k: v for k, v in units.items() if {"raw", "full", "gated"} <= set(v)}
    print(f"\n===== {path} =====  {len(full3)} units with raw, full and gated ({len(units)} units in the file)")
    if not full3:
        continue
    ok = lambda v, arm: bool(v[arm]["success"])
    fail_raw = [v for v in full3.values() if not ok(v, "raw")]
    succ_raw = [v for v in full3.values() if ok(v, "raw")]
    rec_f, rec_g = sum(ok(v, "full") for v in fail_raw), sum(ok(v, "gated") for v in fail_raw)
    brk_f, brk_g = sum(not ok(v, "full") for v in succ_raw), sum(not ok(v, "gated") for v in succ_raw)
    print(f"raw fails     {len(fail_raw):>3}: recovered by full {rec_f}, by gated {rec_g}")
    print(f"raw succeeds  {len(succ_raw):>3}: broken by full {brk_f}, by gated {brk_g}")
    h1 = "untestable (the ungated wrapper breaks fewer than 3)" if brk_f < 3 else ("HOLDS" if brk_g <= brk_f / 2 else "FAILS")
    h2 = "untestable (nothing recovered by full)" if rec_f == 0 else ("HOLDS" if rec_g >= 0.8 * rec_f else "FAILS")
    print(f"H1 harm: {h1} | H2 retention: {h2}")
    vetoed = [(k, v) for k, v in full3.items() if v["gated"].get("gate_vetoes")]
    by_ch = defaultdict(int)
    for _, v in vetoed:
        for c in v["gated"]["gate_vetoes"]:
            by_ch[c] += 1
    wanted = sum(bool(v["gated"].get("wanted_cam") or v["gated"].get("wanted_photo") or v["gated"].get("wanted_denoise"))
                 for v in full3.values())
    print(f"gate: a channel fired in {wanted} units, vetoed in {len(vetoed)} {dict(by_ch)}")
    print(f"{'task':>5} {'cond':<14} raw full gated  vetoed        s_raw -> s_cor (worse of 6)")
    for (tid, cond, rep), v in sorted(vetoed):
        g = v["gated"]
        det = "  ".join(f"{c}: {g['gate_' + c]['s_raw']:.3f} -> {g['gate_' + c]['s_cor']:.3f} ({g['gate_' + c]['worse']})"
                        for c in g["gate_vetoes"])
        print(f"{tid:>5} {cond:<14} {'Y' if ok(v, 'raw') else '.':>3} {'Y' if ok(v, 'full') else '.':>4} "
              f"{'Y' if ok(v, 'gated') else '.':>5}  {','.join(g['gate_vetoes']):<12}  {det}")
    kept_bad = [(k, v) for k, v in full3.items() if ok(v, "raw") and not ok(v, "gated") and not v["gated"].get("gate_vetoes")]
    if kept_bad:
        print(f"harm the gate did NOT catch ({len(kept_bad)}):", [f"{k[0]}:{k[1]}" for k, _ in kept_bad])
