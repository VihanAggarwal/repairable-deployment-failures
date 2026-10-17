# Abstract, draft 2 (2026-09-18, submitted to OpenReview as the placeholder; refine until 2026-09-25)

Title: What a Robot Can Measure, It Can Undo: Separating Invertible Nuisances from Policy Failures at Deployment

Manipulation policies that succeed in the lab lose much of their success when the camera moves, the arm starts somewhere new, the light changes or the sensor becomes noisy. We ask how much of that loss belongs to the policy and how much to the observation, and answer with a wrapper that changes no weights. The wrapper measures the deployment shift with the robot's own body as the only reference: joint encoders against the nominal start, the silhouette of its own arm against the observed arm, and the colour statistics of its own links. It then inverts what it measured, by servoing to the nominal start, reprojecting the image into the training camera and correcting the photometric map, and hands the result to the frozen policy. On LIBERO-Plus with OpenVLA-OFT, success under shifted initial robot states rises from 0.33 to 0.90 and under shifted cameras from 0.35 to 0.65 (60 paired tasks, 20 recovered against 2 broken, exact McNemar p = 0.0001). The same wrapper, unchanged, takes pi0.5 from 0.37 to 0.90 under shifted initial states with no success broken. We then give a policy-free recoverability score, the product of whether the robot can measure a shift and whether inverting it can reconstruct the training observation, computed before any policy runs. In a pre-registered test on a held-out policy family, all seven unrecovered perturbations were camera conditions, six of them the two conditions the score ranked lowest, and the score predicts recovery (AUROC 0.83) where the perturbation category (0.46) and a policy-internal probe (0.62) do not. We also report the cost. A correction is an estimate, and on a policy that already tolerates a nuisance it can hurt: photometric correction lowers pi0.5 from 0.90 to 0.50 under light changes it handled on its own. We treat this as a finding rather than an artefact, and pre-register a label-free gate that asks the frozen policy whether it is less self-consistent on the corrected frame than on the raw one. A large part of what benchmarks report as robustness failure is an invertible nuisance; we provide a taxonomy, a test that says which is which, and an account of when not to act.

Status of every number: see `Downloads/ICLR2027_wrapper_openreview_abstract.txt` (bottom) and RESULTS.md. The AUROC is provisional; the gate sentence claims only the pre-registration.

---

# Abstract, draft 1 (2026-09-16)

Working title: what a robot can measure about its own deployment, it can undo.

## Draft

Manipulation policies that succeed in the lab lose much of their success rate when the camera is moved, the arm
starts somewhere new, the light changes, or the sensor becomes noisy. We ask how much of that loss is a property of
the policy and how much is a property of the observation, and we answer it with a wrapper that changes no weights.
The wrapper measures the deployment shift using the robot's own body as the only reference: joint encoders against
the nominal start, the rendered silhouette and depth of its own arm against the observed arm, and the colour
statistics of its own links. It then inverts what it measured, servoing to the nominal start, reprojecting the
image into the training camera, and correcting the photometric map, before handing the observation to the frozen
policy. On LIBERO-Plus with OpenVLA-OFT, robot-initial-state success rises from 0.33 to 0.90 and camera success
from 0.35 to 0.65 (n = 60 paired tasks, 20 recovered against 2 broken, exact McNemar p = 0.0001), while the
categories the robot cannot measure are left untouched at 0.97. Across classes the wrapper recovers 70% of
invertible failures and breaks 9% of successes. [FAMILIES] [SCORE] [LEARNED] [REAL] We argue that a large part of
what benchmarks report as policy robustness failure is an invertible nuisance, and give a taxonomy and a
policy-free test that says which is which before the policy runs.

## Locked numbers (will not change)

- OFT baseline, 210 tasks: overall 0.76; robot_init 0.33, camera 0.43, sensor_noise 0.73, background 0.90,
  language 0.97, layout 0.97, light 0.97.
- robot_init: 0.33 -> 0.90 (n = 30, 18/20 failures recovered, 1/10 successes broken).
- camera pooled: 0.35 [0.24, 0.48] -> 0.65 [0.52, 0.76], n = 60, 20 recovered / 2 broken, exact McNemar p = 0.0001.
- Upper bounds on the camera subset: true pose with training-view fill 0.70, perfect training-view render 0.90.
- Per class: invertible 26/37 failures recovered (70%), 2/23 successes broken (9%); invertible failure rate
  0.34 -> 0.17; capability-class success after wrapping 0.97.

## Slots and what fills each

- [FAMILIES] "The same wrapper, unchanged, does X on pi0.5 and Y on OpenVLA base." Needs W5 (running) and W9.
- [SCORE] "A policy-free recoverability score, the product of whether the robot can measure the shift and whether
  its inversion can reconstruct the training observation, predicts which perturbations are recovered (AUROC X on
  held-out policy families) where measurability alone (Y) and the perturbation-category label (0.50) do not."
  Development value on OFT: 0.913 [0.73, 1.00] (2026-09-17, not evidence). X and Y come from pi0.5 and OpenVLA base.
  This is the novelty sentence, and it now says WHY recovery fails: measurable is not reconstructible.
- [LEARNED] "A canonicalizer distilled from self-labelled pairs closes X of the gap to the oracle view." Needs W7.
- [REAL] "The same self-calibration recovers camera pose to X cm on real DROID images." Needs W6.

## Honest weaknesses to keep in the paper

- The partly-invertible class is unsupported: light and noise baselines are already 0.90 to 0.97, so there is
  almost nothing to recover. Report it as no-harm, not as recovery.
- The denoise channel never fires on real LIBERO-Plus noise tasks and fires falsely on half the background tasks.
- The photometric channel abstains under saturated or strongly directional light rather than apply a map the body
  cannot identify. Report the abstention rate.
- Until W7 lands, the camera results use the simulator's depth and the ground-truth robot mask.
