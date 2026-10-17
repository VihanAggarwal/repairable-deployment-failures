# Pre-registration, 2026-09-25: run file 21 (W17), OpenVLA fine-tuned on LIBERO-10, 6 further base tasks

Written and pushed BEFORE any row of `plus/sweep_openvla_ext` exists (checked on Drive 2026-09-25 00:24: folder not
found). Scores are policy-free and already frozen: `scores_v2.jsonl` (registered score, all 10 base tasks) and
`vendor/score_selfmeasured.jsonl` (self-measured camera ident, 70 camera units, all 10 base tasks, computed 2026-09-24).
Nothing below is tuned on the new rows.

## Units
Exactly W17's rule: base tasks 1101, 1146, 1255, 1297, 1325, 1371; a task enters iff its unperturbed raw episode
(condition nominal, seed = task id) succeeds; units = (base task, condition) over the 24 standard conditions whose raw
episode fails; positive = the `full` arm succeeds. Analysis code: `score_baselines.units_for`, `within_category.py`,
`selfmeasured_gate.py`, pointed at the new shard folder; no change to any of them except the input path.

## Frozen predictors
- R: registered score, `recover` from scores_v2.
- D: deployable score = ident at the self-calibrated pose (score_selfmeasured.jsonl) x invert x 1[cal_final_score <= 0.547]
  on camera units; = R on all other units. tau = 0.547 was fit on OFT only on 2026-09-25.
- G: the gate alone, cal_final_score > 0.547, as a predictor of calibration failure (cam_est_err > 2 cm).
- Baselines: category label; category rate fit on OFT; condition rate fit on OFT with category fallback.

## Tests (one-sided, alpha 0.05, stated before the data)
- H1 (primary): within-category AUROC of D on the new units > 0.5, permutation test that shuffles outcomes within
  category (10,000 permutations, seed 0).
- H2: within-category AUROC of D pooled over the registered 60 OpenVLA units and the new units > 0.5, same test.
- H3: G flags calibration failure on the new camera units with AUROC >= 0.90.
- H4 (secondary): AUROC of D >= AUROC of R - 0.05 on the new units (non-inferiority, point estimate).
Reported whatever the outcome, with the condition-rate baseline beside each. If fewer than 2 new base tasks enter or
either class of a test has under 5 units, that test is reported as underpowered, not as a pass or fail.
