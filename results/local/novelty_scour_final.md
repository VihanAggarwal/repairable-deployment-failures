TOP 5 (ranked by novelty x reviewer impact x probability of a positive result / cost)

Housekeeping facts that apply to every item:
- The main.tex grep finds no oracle-render or true-pose arm: "cam_true" and "true pose" match nothing, and the three "oracle" hits are unrelated. Idea 1 therefore adds new material to the paper.
- The draft has no Soatto citation.
- File 13 is running. Any GPU work must use its own --out_dir, because of the two-runtime overwrite rule.
- The local box fits one MuJoCo process at a time.

---

1. WHERE THE DEFICIT LIVES: THE COUNTERFACTUAL LADDER

This merges the theory-lens "coverage ceiling" with the reanalysis-lens "counterfactual ladder". It also takes the one baseline row that survived from the "horizon" idea.

**Abstract sentence.** Handing the reprojection the TRUE camera pose recovers 0.50 to 0.70 of a frozen VLA's tasks, where a perfect training-view render recovers 0.90. Swapping the true pose for the self-measured one changes nothing beyond rollout noise. None of the 17 raw camera failures survives a perfect render. The 8 failures that both the measured and the true pose leave unrepaired are lost field of view, which the robot had already scored as not invertible before the rollout.

**Exact experiment.**
- (a) Zero compute. Write wrapper/scripts/attribution_ladder.py on plus_common.load_rows and invertibility_report.mcnemar_p.
  - Inputs: C:/Users/vihan/Downloads/local_data/plus/wrapper_oft/rows.jsonl and rows_oft.jsonl.
  - Arms on the 30 OFT libero_10 camera tasks: raw 13/30, full 19, cam_true_ref 21, oracle 27.
  - Per-task first succeeding rung. The 17 raw failures split into 7 recovered by measurement, 2 not identified (tasks 784 and 1065), 8 not invertible, and 0 policy.
  - Report exact McNemar per rung (oracle vs cam_true is 13 vs 1, p = 0.002) and a task-cluster bootstrap.
  - Report the identifiability gap as "zero within the 15 percent flip floor". Never call it a positive gap, because full beats cam_true 5 vs 1.
  - Print the 3 of 17 non-nested units as a measured noise rate.
  - Same script, baseline row on the 68 raw failures with a logged shift. AUROC for recovery: shift distance 0.49, shift rotation 0.61, calibration error 0.63, self-measured coverage 0.77. Put it beside the category-label baseline in Table 1.
- (b) Local CPU, 2 to 3 hours, one process. Write vendor/fidelity.py by cloning vendor/envelope.py (same 264 placements, frozen score untouched).
  - Call can.oracle_nominal_image and W.reproject(..., return_mask=True) once with can.true_camera() and once with the calibrated pose.
  - Log seen-pixel error and hole-pixel error SEPARATELY against the oracle render.
  - The question is which one tracks coverage and which one reference fill removes. Coverage predicts the reference-filled arm at AUROC 0.993, so coverage is a proxy for total reprojection damage, not hole mass.
  - State the MMSE inequality as a one-line Remark citing Soatto 2009 / 1110.2053 for the invertible versus non-invertible nuisance split. Do not make it Proposition 1. Keep "the invert terms are designed, not learned" and add "validated by the ladder".
- (c) GPU, up to 2.5 A100-hours, own --out_dir.
  - Run ONLY the cam_true_ref and oracle arms on the wrapped-failure camera units of object, goal and spatial (roughly 35 to 50 episodes per arm).
  - If the pi0.5 runner takes the two arms without a method change, also run them on the 17 unrecovered pi0.5 camera failures. That says whether the held-out family (camera gain p = 0.18) loses to inversion or to itself.
- Caveats to print: the oracle rows are NOMINAL_VERSION 2 while full and cam_true_ref are v4. Argue that the oracle render does not depend on calibration, or refresh those rows for about 1 A100-hour. The closure check (27/30 against unperturbed success) is a consistency check, not proof.
- Figure: left, restoration error vs (1 - coverage) over 264 placements, split into seen and hole pixels. Right, a stacked bar per checkpoint: recovered / not identified / not invertible / policy.

**Cost.** 1.5 person-days, 0 A100-hours required, up to 2.5 optional.

**Closest prior art.**
- AnyCamVLA (2603.05868): original-view reference only, and its whole method is the true-pose rung.
- GS-VLA (2608.19066): frames view canonicalization as disocclusion.
- Soatto 2009: owns the invertible versus non-invertible nuisance taxonomy.
- "VLA Models Are More Generalizable Than You Think" (2512.02902): shows perception rather than policy, by adapting the encoder.
- Differentiating sentence: "View-correction wrappers report an original-view reference and failure detectors predict that a policy fails; none intervenes with nested arms (measured pose, true pose, perfect training-view render) on the same task and seed of a frozen policy to say why a corrected policy still fails."

**Probability of a positive result.**
- Table and ladder: 0.9. The numbers have already been reproduced twice from the logged rows.
- Fidelity split giving a clean seen-versus-hole story: 0.6.
- GPU extension reproducing "policy bucket near empty" on other suites and pi0.5: 0.6.

**If null.**
- If the other suites or pi0.5 show a non-empty policy bucket, that is still the attribution nobody has reported. Write it as "on the held-out family x of 17 residual failures are the policy's own", which honestly explains p = 0.18.
- If fidelity.py shows coverage tracks seen-pixel damage and not holes, say so. That corrects the paper's reading of its own invert term and explains why inpainting gave no gain (0.63 vs 0.67).

---

2. HARD FOR EVERY POLICY IS NOT UNREPAIRABLE: THE SCORE INSIDE LIBERO-PLUS LEVEL 5

**Abstract sentence.** LIBERO-Plus grades tasks by how many of four reference VLAs solve them. On level-5 tasks that none solves, a servo to the training pose repairs 38 of 40 robot-initial-state failures. Within level-5 camera tasks, where the benchmark's grade is constant, the policy-free coverage term still separates repairable from unrepairable failures (AUROC 0.75, 95% [0.59, 0.88], n = 47).

**Exact experiment.** Zero GPU. Write wrapper/scripts/level_vs_score.py.
- Join the wrap files to vendor/LIBERO-plus/libero/libero/benchmark/task_classification.json BY NAME. The files are wraps/wrap_openvla_oft.jsonl, wraps_libero_object|goal|spatial, wrap_pi05.jsonl and plus_subhi. An id join is off by one. Drop the 121 entries with a null level.
- Reuse auroc and boot_ci from sweep_analysis.py.
- De-duplicate pi0.5 tasks that appear in both people's files. Use rep 0 only, or cluster-bootstrap by task.
- Report within-level AUROCs (L5, L4 to 5, L1 to 4) and recovery by level for camera and robot init.
- State honestly that ACROSS levels the grade carries repair information on camera (flipped about 0.68). It proxies coverage (rho -0.38), and the score adds to it within level.
- One clause: levels are policy-derived and OFT is one of the four definers, so no claim is made about raw failure.
- One figure panel: recovery given failure by level, camera vs robot init, with a coverage-tercile split inside L5.
- Replace the 45 degree orbit vs 35 degree tilt anecdote in the intro with the L5 sentence. Label it post hoc.
- Optional: backfill coverage on about 50 camera failures locally (under 1 CPU-hour), only if the box is free.

**Cost.** 0.5 person-day, 0 A100-hours.

**Closest prior art.**
- LIBERO-Plus (2510.13626): success by level per dimension; the levels are defined by four models.
- RoboART (2502.06575): predicts degradation, not repairability.
- Restoring the Right Stream (Entropy 2026): routes by category, never by level.
- Differentiating sentence: "Nobody has tested a benchmark's hardness annotation against a paired repair target, because nobody else has paired repair outcomes; where the grade is constant by construction, a quantity the robot measures before acting still ranks what is fixable."

**Probability of a positive result.** 0.8. The peek on 453 pairs already gives 0.745 within L5, and the lower CI bound is 0.59. The 38/40 robot-init number is already a count.

**If null.** If the within-L5 CI crosses 0.5 after de-duplication, keep only the robot-init sentence (L5 failures repaired 38 of 40, L3 and above 74 of 76). Narrow the camera claim to "the level does not replace the score" and put the partial AUROC in the appendix.

---

3. THE POLICY'S OWN CALIBRATION TOLERANCE

This is observational first and a rollout-free dose-response second. The validated centimetre ruler from the "source field" idea supplies the y-axis unit only.

**Abstract sentence.** Injecting a known pose residual on top of the true camera pose and reading only the first action chunk gives each frozen VLA a calibration tolerance in centimetres and degrees, with no rollouts. Body-as-target calibration falls below that knee from three poses on. That is why identifiability error tracks calibration (rho -0.63 to -0.82) yet does not decide outcome (AUROC 0.52), and why the invert factor carries the variance.

**Exact experiment.**
- Stage 0, zero GPU, half a day. Do this before spending anything.
  - On logged full-arm sweep rows, regress the relative first-chunk distance to the task's nominal raw chunk on three covariates: cam_est_err_cm_deg, coverage and cam_true_shift. Use a task-grouped bootstrap. The first_chunk field is written at sweep_run.py line 458 and the nominal chunk at line 492.
  - Prediction: a flat slope on achieved calibration error and a steep slope on coverage.
  - If pose='true' rows exist for the same units, add the paired full vs true-pose comparison.
  - Also fit M on the nominal probe rows of smc_<policy>.jsonl and run leave-one-probe-out against the known 10 cm offsets. If median ruler error is under 2 cm, use it as the cm unit of the Stage 1 y-axis. Do not claim provable invertibility.
- STOP RULE. If Stage 0 shows a clear slope on calibration error, do not run Stage 1. Rewrite the "calibration is rarely what fails" sentence instead.
- Stage 1, about 1 A100-hour per family, capped at 2 total. Write a new runner wrapper/scripts/tolerance_probe.py modelled on smc_probe.py.
  - Leave prepare_full untouched. Use its own out_dir and a hostname-suffixed file.
  - Copy the pose == 'true' branch (sweep_run.py lines 317 to 318) and add doses: translation 0, 0.5, 1, 2, 4, 8, 16 cm per camera axis, and rotation 0, 0.5, 1, 2, 4, 8 degrees per axis.
  - About 30 tasks, one in-envelope condition (coverage above 0.6), rest pose only, fixed sampler seed, 3 seeds on pi0.5.
  - Knee = the smallest dose where the median field distance crosses the frozen FP_UNAFFECTED = 0.10.
  - State that single-pose distance is a weaker outcome proxy than the seven-probe fingerprint (0.817 vs 0.947 on OFT).
- Drop the 120 anchoring rollouts.
- Figure: distance vs injected residual (log cm) per family with an IQR band, a line at 0.10, and vertical marks at the 1, 3 and 7 pose errors (0.54 to 0.07 cm), the learned-mask error (2.8 cm) and DROID (19.6 cm).

**Cost.** 1.5 person-days, 0 A100-hours for Stage 0, at most 2 for Stage 1.

**Closest prior art.**
- CamVLA (2607.05396, already cited as fixedtofree): App. C injects 0 to 45 degree rotation noise into the hand-eye matrix and reports success vs dose with rollouts.
- AxisGuide (2606.06761).
- Differentiating sentence: "CamVLA reports success against injected rotation noise using rollouts for a policy that consumes extrinsics; we measure, without rollouts and in both centimetres and degrees, the residual at which a frozen VLA's opening action leaves its training-time fingerprint, and place the robot's own calibration error on that axis."

**Probability of a positive result.**
- Stage 0 showing flat-on-error and steep-on-coverage: 0.6.
- Stage 1 knee above the 3-pose error on both families: 0.55. OFT is the risk.

**If null.**
- A knee below the achieved error on one family is still a finding: "OFT needs calibration better than x cm; the measurement delivers y". It forces one paragraph rewrite and makes the "not identified" bucket of idea 1 the explanation (tasks 784 and 1065 at 5.3 and 12.8 cm).
- If Stage 0 is simply noisy, skip Stage 1 and keep one appendix sentence. No GPU is spent.
- DROID landing far right of the knee is to be written as the quantified gap real-image calibration must close.

---

4. THE ERROR LIVES WHERE THE SCORE SAYS IT CANNOT SEE: DIRECTIONAL IDENT ON 53 REAL DROID CAMERAS

**Abstract sentence.** On 53 DROID cameras from 5 labs, a policy-free Jacobian of the robot's own silhouette, computed without ground truth, predicts the DIRECTION in which the self-measurement fails (an orbit about the arm, lateral error vs rotation error rho +0.87). Inside the subspace the score calls measurable the error is X cm against 19.6 cm raw, and the predicted workspace error ranks the realized one where scalar sigma_min did not (rho -0.21 n.s.).

**Exact experiment.** Local CPU only, under 15 minutes of MuJoCo. It must not overlap fidelity.py.
- Extend vendor/droid_identifiability.py (Jacobian built at lines 49 to 55) to keep Vt and the 6x6 Gram in cm and deg. Compute it both at the reference and AT THE ESTIMATE. The at-estimate version is the ground-truth-free one the claim needs.
- Build the realized 6-vector error from est_pos, est_euler and ext, as translation in cm plus the rotvec of R_est R_gt^T in deg.
- Report:
  - |cos| with v_min in the whitened space;
  - the fraction of squared error in the bottom-2 singular subspace;
  - the error projected on the top-4 subspace.
- Mandatory controls:
  - a GENERIC-ORBIT baseline, one hand-written orbit direction about the arm centroid, compared with the per-camera v_min;
  - a random-direction null in the whitened space;
  - the other-camera permutation null.
- The undo-relevant metric goes on points the fit did NOT see: a table-plane grid 0.3 to 0.7 m in front of the base, in cm and pixels.
- Payoff test:
  - predicted workspace error = sqrt(trace(P (J^T J)^-1 P^T));
  - Spearman against realized error on the 43 reliable-reference cameras;
  - the 26 same-episode camera pairs (currently 18 of 26, p = 0.076) as the within-unit control.
- Non-local check, salvaged from the auditor idea: inject equal-size offsets along v_max and v_min at 3 magnitudes on the 43 cameras. Show the silhouette score responds later along v_min.
- Keep the raw 19.6 cm row in Table tab:droid. Label it post hoc. Headline on the 43 reliable references, with all 53 in the appendix. Frozen scalar score untouched.
- One sentence in Sec. sec:droid: no calibration motion was commanded, so the frames are configurations the demo happened to visit. Cite Maye et al. IJRR 2016 and EasyHeC.

**Cost.** 1 person-day, 0 A100-hours.

**Closest prior art.**
- Muthukkumar (2510.21785): differentiable-rendering Cramer-Rao bound, never checked against realized error on real robots.
- Nahvi and Hollerbach 1996 and Sun and Hollerbach 2008: sigma_min observability index.
- CtRNet-X, CalibAll, Posed DROID: scalar errors only.
- DROID App. G-A: silhouette audit.
- Differentiating sentence: "Identifiability analyses predict a covariance that is never checked against realized error on in-the-wild robot data; we check it on 53 cameras from 5 labs and report the error restricted to the subspace the robot could measure."

**Probability of a positive result.**
- Alignment beating the random null: 0.75.
- Per-camera v_min beating the generic orbit: 0.4.
- Workspace-error Spearman significant at n = 43: 0.5.

**If null.**
- If per-camera v_min does not beat the generic orbit, present it as the lever-arm identity made quantitative. Report the measurable-subspace and workspace numbers anyway. They are what the undo step depends on, and they reframe 19.6 cm.
- If mask bias sets the direction, say so in limitations. Either way, cite Nahvi and Hollerbach, Sun and Hollerbach, and Muthukkumar for ident.

---

5. A GEOMETRIC CEILING CENSUS OF LIBERO-PLUS CAMERA TASKS

This is salvaged from the annotation idea. It is the lowest priority and runs only if the local box is idle overnight.

**Abstract sentence.** Computed from the true camera pose with no policy and no estimator, X percent of LIBERO-Plus camera tasks retain less than 0.6 of the training view after an exact reprojection. That bounds what any geometric single-view re-rendering wrapper, ours or AnyCamVLA's, can restore on that column without generative fill. The fraction rises from a percent at level 1 to b percent at level 5.

**Exact experiment.** Zero GPU, overnight. Write vendor/coverage_census.py from vendor/envelope.py.
- Stratified sample of about 400 camera tasks (suite x level), with resumable per-suite shards.
- One env per process, because robosuite leaks environments.
- Log true-pose reprojection coverage only: one render plus a z-buffered warp.
- Output: JSON keyed by task NAME under a neutral name. Do not call it LIBERO-Plus-R, because LIBERO-RECOVER (2609.05178) collides.
- Report one sentence plus one panel beside the idea 1 figure, and rank agreement with the full frozen score on whatever overlap exists with the 264 envelope rows.
- Dropped: IFM/PFM, the re-scored leaderboard, and the light, noise and robot annotations.

**Cost.** 0.5 person-day of attention, plus 6 to 10 hours of background CPU that blocks all other local simulator work.

**Closest prior art.**
- AnyCamVLA and GS-VLA: name coverage loss as the failure mode, with no census.
- LIBERO-Plus: owns the levels.
- Differentiating sentence: "No per-task true-pose coverage census of the benchmark's camera column exists; it turns idea 1's 30-task ceiling into a statement about the whole column."

**Probability of a positive result.** 0.7 that it finishes cleanly and shows a monotone rise with level. The level-coverage rho of -0.38 already points that way.

**If null.** If it is flat across levels, drop the panel and keep one sentence giving the overall fraction. If the box is busy, skip it entirely. Ideas 1 and 2 stand without it.

---

KILLED, WITH ZERO-COST SALVAGE ONLY

- **Cramer-Rao law.** The CRB is for a silhouette-only model and does not bound calibrate_v2, which also uses depth ICP. The slope is confounded with pose count. Muthukkumar 2025 and Nahvi-Hollerbach own it. Keep two sentences and the citations in Sec. 3.2.
- **Free actions and gauges.** The joint-0 gauge does not exist in this problem, because the encoders read the offset exactly. A rank table would also expose that background gets ident 0.92 from a within-family ratio, and that is a frozen constant. Keep one sentence plus one limitation line.
- **Slope and intercept calibration.** The intercept LR test gives p = 0.11 and the pi0.5 slope is 2.4 with SE 2.7. Keep only a one-sentence noise-ceiling simulation and an appendix Brier/ECE.
- **Recoverability horizon.** Neither the raw nor the wrapped curve crosses 0.5, so there is no h50. Its one useful row is in idea 1.
- **Null arm.** OFT has zero applied-nothing pairs. The pi0.5 null gain of 2/8 would erase the 7/24 camera result. Fold one before-and-after-settle sentence into Sec. 4.4 when the settle rows land, plus a closed-form minimum detectable effect line. Cite STEP, Vincent, Block and LIBERO-CTRL (2609.15940).
- **Divergence-rate margin.** Only 7 events. A per-task scalar cannot explain within-task placement variance. Cite 2609.15940 and 2507.09061.
- **DROID auditor.** Pre-empted by DROID App. G-A and CtRNet-X. Its injection contrast lives in idea 4.
- **Frame-subset selection.** Pre-empted by Maye 2016, Schneider 2017 and EasyHeC. Its 26-pair control and one sentence live in idea 4.

---

WHICH ONE FIRST, AND WHAT COMBINES

Do idea 1 first, starting with the zero-compute table today. It is the only idea whose main result is already in hand (13/19/21/27, zero failures in the policy bucket, 8 of 8 non-invertible units at true-pose coverage 0.35 to 0.54). It needs no GPU to be publishable. It turns the title into a table, which is the largest gain in central-claim strength available, and the draft does not use these arms at all. Its only real risk is that a reviewer calls 17 units an anecdote. The up to 2.5 A100-hour extension to object, goal, spatial and pi0.5 addresses that, and it can share a Colab session with file 13 if it writes to its own out_dir.

Sequencing on the single local MuJoCo slot:
1. fidelity.py from idea 1 (2 to 3 hours).
2. The idea 4 DROID run (under 15 minutes).
3. The idea 5 census overnight, only if nothing else is queued.

Ideas 2 and 3 Stage 0 are pure log re-analysis and can run alongside anything.

Section structure:
- Ideas 1, 2 and 5 become ONE new subsection in Sec. 4.1/4.2, "Where the deficit lives":
  - the ladder table;
  - the restoration-error figure, with the census panel beside it;
  - the within-level-5 paragraph;
  - the shift-distance, rotation and calibration-error baseline row in Table 1.
- Idea 3 extends Sec. 4.3, "What the policy can tell us", as the policy-side explanation of why the identification rung of the ladder is nearly empty. It should cross-reference the two "not identified" units.
- Idea 4 stays inside Sec. 4.5 (DROID) as the real-image validation of ident.

The one line through all five: the ladder shows the deficit is inversion, not measurement. The tolerance curve explains that from inside the policy. Level 5 and the census show it holds benchmark-wide. DROID shows the measurable part is also measurable on real robots.

Total: about 5 person-days if everything is done, and at most 4.5 A100-hours. If time runs short, do ideas 1 and 2 plus idea 3 Stage 0. That is 2.5 person-days and zero required GPU.

Relevant paths:
- C:/Users/vihan/robot-deployment-papers/wrapper/paper/iclr/main.tex
- C:/Users/vihan/Downloads/local_data/plus/wrapper_oft/rows.jsonl
- C:/Users/vihan/robot-deployment-papers/vendor/envelope.py
- C:/Users/vihan/robot-deployment-papers/vendor/droid_identifiability.py
- C:/Users/vihan/robot-deployment-papers/vendor/LIBERO-plus/libero/libero/benchmark/task_classification.json
- C:/Users/vihan/robot-deployment-papers/wrapper/scripts/smc_probe.py
- C:/Users/vihan/robot-deployment-papers/wrapper/scripts/sweep_run.py
- C:/Users/vihan/Downloads/local_data/plus/droid_selfcal_vlm/selfcal_results.jsonl