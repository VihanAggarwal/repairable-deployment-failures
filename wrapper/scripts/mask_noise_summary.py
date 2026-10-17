"""POST HOC (2026-09-26): calibration error against mask quality (vendor/local_mask_noise.jsonl, policy-free, set A).

    uv run --no-project --with numpy python wrapper/scripts/mask_noise_summary.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "..", "..", "vendor", "local_mask_noise.jsonl")) if l.strip()]
print("rows %d, tasks %d" % (len(R), len({r["task_id"] for r in R})))
print("level | mask IoU median | refine2 error median cm | within 1 cm | within 2 cm | search error median cm")
for lv in sorted({r["level"] for r in R}):
    S = [r for r in R if r["level"] == lv]
    e = np.array([r["err_refine2"][0] for r in S]); s = np.array([r["err_search"][0] for r in S])
    print("  L%d  | %.3f | %6.2f | %2d/%d | %2d/%d | %6.2f" % (lv, np.median([r["mask_iou"] for r in S]), np.median(e),
                                                            (e <= 1).sum(), len(S), (e <= 2).sum(), len(S), np.median(s)))
