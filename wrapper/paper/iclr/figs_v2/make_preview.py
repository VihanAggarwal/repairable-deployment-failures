"""Preview of Vihan's 2026-09-24 main.tex with the redrawn figures swapped in, one for one. His text, tables, order and
cuts are kept; only figure files and the captions that describe them change."""
import sys

p = sys.argv[1]
s = open(p).read()


def swap(label, path, width, caption):
    global s
    i = s.index("\\label{" + label + "}")
    a = s.rindex("\\begin{figure}", 0, i)
    b = s.index("\\end{figure}", i) + len("\\end{figure}")
    head = s[a:s.index("\n", a)]                     # keep his float placement ([t] or [h])
    # natural size (scale 1.0): every figure is drawn at its print width, so all figure text prints at the same size
    s = s[:a] + (f"{head}\n\\centering\n\\includegraphics{{{path}}}\n\\caption{{{caption}}}\n"
                 f"\\label{{{label}}}\n\\end{{figure}}") + s[b:]


swap("fig:teaser", "figs_v2/teaser.pdf", "\\linewidth",
     "The wrapper changes no weights. (a) Two camera shifts. The robot measures the camera from its own body (blue: its "
     "silhouette rendered at the measured pose) and re-renders the frame into the training view for the frozen policy. How "
     "much of the view survives is known before any rollout: the pull-back can be undone, the look-away cannot. (b) The "
     "policy-free score predicts which failures are repaired on two held-out policy families, with 95\\% intervals; the "
     "perturbation category is at chance (Table~\\ref{tab:score}).")
swap("fig:schematic", "figs_v2/fig2_method.pdf", "\\linewidth",
     "The wrapper around a frozen policy. Left: one task in simulation (camera orbited 20 degrees): the training view, the "
     "deployed view, the robot's silhouette rendered at the camera pose it measured from its own body (blue), and the frame "
     "re-rendered into the training camera. Right: the loop. Look was added after the main experiments "
     "(Sec.~\\ref{sec:cost}); if the camera has not moved, the arm stays still and the policy gets the raw frame.")
swap("fig:envelope", "figs_v2/fig_envelope.pdf", "\\linewidth",
     "The score before any policy runs, and what happened when one did. (a) The score over 264 camera placements around the "
     "training camera (cross), three base tasks. (b) Along an orbit and a look-away: the score (lines) and the share of raw "
     "failures the wrapper repaired on the sweep (points, OFT and \\pihalf{} pooled, 95\\% Wilson intervals). (c) Every "
     "LIBERO-Plus camera task that logged the camera term (four OFT suites and \\pihalf): share repaired against the view the "
     "robot measured it would keep; dashed: perfect calibration.")
swap("fig:paired", "figs_v2/fig4_paired.pdf", "\\linewidth",
     "Task success of the raw policy (hollow) and the same policy wrapped (filled) on the same tasks and seeds; right: raw "
     "failures repaired and raw successes broken. The \\pihalf{} camera row pools three replicated runs "
     "(Sec.~\\ref{sec:undo}). Baselines are in Table~\\ref{tab:baselines}.")
swap("fig:harm", "figs_v2/fig6_harm.pdf", "\\linewidth",
     "(a) Raw successes of \\pihalf{} broken by the wrapper on 40 tasks where it corrects nothing: measuring always, "
     "measuring then settling, and measuring only when the silhouette says the camera moved. (b) That signal: the silhouette "
     "residual at the training camera from one rest pose, unshifted against camera-shifted units.")
swap("fig:tolerance", "figs_v2/figA_tolerance.pdf", "\\linewidth",
     "How exact the calibration must be: the policy's opening-action distance to its true-pose action as a known pose error "
     "is added in translation (a) or rotation (b), two frozen families, no rollouts; median and interquartile range over "
     "tasks. Dashed: the frozen fingerprint threshold; dotted: no correction. Marked on the same axis: the wrapper's own "
     "calibration (bar, 0.07 to 0.54 cm), learned sensing (12.6 cm) and real DROID images (19.6 cm).")
swap("fig:score", "figs_v2/fig3_score.pdf", "\\linewidth",
     "(a) Observed repair rate against the policy-free score on the graded sweep, binned with Wilson bands, for the "
     "development and both held-out families. (b) AUROC with 95\\% bootstrap intervals: the score against the category label "
     "on the sweep, and, among benchmark camera failures, coverage against the size of the shift and the calibration error.")
swap("fig:ladder", "figs_v2/fig5_ladder.pdf", "\\linewidth",
     "Top: the four nested arms, run on the same task and seed. (a) Every raw camera failure attributed to the first arm "
     "that repairs it; pooled over OFT, 1 of 100 belongs to the policy. (b) The calibration error per bucket (bars: "
     "medians; shaded: below the policy's measured tolerance of about 1 cm): only-true-pose units are gross calibration failures, only-render units were calibrated to about 2 cm and "
     "lost the view.")
swap("fig:fidelity", "figs_v2/figA_fidelity.pdf", "0.62\\linewidth",
     "Error against a perfect training-view render on the 264 placements, split into seen pixels and holes, as a function of "
     "the view lost (Prop.~\\ref{prop:cov}); median and interquartile range. With a training-view fill the holes are cheap; the "
     "error that grows with lost view sits on the seen pixels.")
swap("fig:static", "figs_v2/static_check.pdf", "\\linewidth",
     "The motion-free test of Prop.~\\ref{prop:static}: the silhouette the training camera predicts for the current joint "
     "configuration (blue) over the deployed frame. When the camera has not moved it lands on the arm and the sweep is "
     "skipped; when it has, the miss is visible before the arm moves.")
swap("fig:ladderpics", "figs_v2/ladder_pictures.pdf", "\\linewidth",
     "The ladder on one task (camera turned away 35 degrees): the measured pose fails, the true pose still leaves the view "
     "the camera lost, only the training camera's own view is complete.")
swap("fig:maps", "figs_v2/figA_maps.pdf", "\\linewidth",
     "Policy-free maps over camera placements around the training camera (cross), means over three base tasks: "
     "identifiability, view kept, the score, and the seen-pixel error after a true-pose re-rendering.")
swap("fig:fidpics", "figs_v2/fidelity_pictures.pdf", "0.9\\linewidth",
     "Where the corrected frame is wrong, in pictures: holes (yellow) are filled cheaply; the error sits on seen surfaces and "
     "grows with the orbit.")
swap("fig:tolpics", "figs_v2/tolerance_pictures.pdf", "\\linewidth",
     "The frame handed to the policy when the true camera pose is perturbed by 1, 4 and 16 cm.")
open(p, "w").write(s)
print("preview ok")
