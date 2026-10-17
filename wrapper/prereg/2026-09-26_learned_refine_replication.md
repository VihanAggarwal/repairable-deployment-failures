# Pre-registration, 2026-09-26: run files 27 and 28 (LEARNED sensing + refined pose, replication)

Written and pushed BEFORE any row of `plus/wraps_refine2_setB` or `plus/wraps_refine2_learned_pi05` exists. The arm is
exactly run 25's `full_hybrid_refine2` (learned segmenter seg_net.pt, table-plane depth, pose_sources "refine2"; code
frozen at the commit that adds this file). Disclosed: interim rows of run 25 (14 of 30 set A tasks, OFT) had been seen
when this was written: rollout calibration median 0.70 cm, success 9 of 14 against raw 9 of 14. Policy-free set B
calibration (vendor/local_refine_learned_s3_setB.jsonl) had also been seen.

## Run 27: OFT, set B (30 camera tasks; raw and full rows exist in plus/wraps, same seeds)
- H9 (primary): success of full_hybrid_refine2 > raw, one-sided exact McNemar on discordant tasks, alpha 0.05.
- H10: full_hybrid_refine2 not worse than simulator-sensing `full` by more than 3 tasks (point estimate).

## Run 28: pi0.5, set A (30 camera tasks; raw and full exist)
- H11 (primary): success of full_hybrid_refine2 > raw, one-sided exact McNemar, alpha 0.05.
- H12: full_hybrid_refine2 not worse than simulator-sensing `full` by more than 3 tasks (point estimate).

Pooled across runs 25, 27 and 28 (90 tasks, two policies): the same McNemar test is reported as a secondary line.
Every test is reported whatever its outcome.
