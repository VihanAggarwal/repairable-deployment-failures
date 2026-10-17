"""Self-motion consistency (SMC): a label-free, physically grounded test of whether a frozen visuomotor policy is
perceiving the scene correctly.

A goal-directed policy that commands end-effector displacements defines an action field over end-effector positions.
If the policy perceives the scene correctly, the rays `p_k + t * u_k` (p_k: end-effector positions the robot visits
and KNOWS from its encoders, u_k: the direction of the policy's commanded translation there) must meet at one point,
the policy's current sub-goal. If the observation is wrong in a way the policy cannot absorb (a moved camera, a bad
correction), the perceived displacement is a distorted function of the true one and the rays stop meeting. The
robot's own proprioception is the metric ground truth; no label, no simulator state and no model of the policy is used.

    residual_deg   RMS angle between each commanded direction and the direction to the triangulated sub-goal
    forward        share of probes whose sub-goal lies in FRONT of them along their own ray
    min_eig        smallest eigenvalue of the normal matrix / K: near 0 when the rays are parallel (no information)

Uses: (1) a policy-side, label-free harm test for any input correction (lower residual = the policy agrees with
itself more), (2) selection among candidate corrections, including "none", (3) calibration of nuisance parameters
with the frozen policy as the sensor.
"""
import numpy as np

# probe offsets from the nominal end-effector position (metres): a centred cross, 10 cm arms, 8 cm vertically
PROBES = [np.zeros(3), np.array([0.10, 0, 0]), np.array([-0.10, 0, 0]), np.array([0, 0.10, 0]), np.array([0, -0.10, 0]),
          np.array([0, 0, 0.08]), np.array([0, 0, -0.06])]
MIN_ACTION = 1e-3


def triangulate(P, A):
    """P (K,3) end-effector positions, A (K,3) commanded translations. Returns dict(goal, residual_deg, forward,
    min_eig, n). Probes with a vanishing command carry no direction and are dropped."""
    P, A = np.asarray(P, float), np.asarray(A, float)
    n = np.linalg.norm(A, axis=1)
    keep = n > MIN_ACTION
    P, U = P[keep], A[keep] / n[keep, None]
    K = len(P)
    if K < 3:
        return dict(goal=None, residual_deg=float("nan"), forward=float("nan"), min_eig=0.0, n=int(K))
    M = np.eye(3)[None] - U[:, :, None] * U[:, None, :]                  # projectors orthogonal to each ray
    N = M.sum(0)
    ev = np.linalg.eigvalsh(N)
    g = np.linalg.lstsq(N, (M @ P[:, :, None]).sum(0)[:, 0], rcond=None)[0]
    d = g[None] - P
    dist = np.linalg.norm(d, axis=1)
    cosang = np.clip((d * U).sum(1) / np.maximum(dist, 1e-9), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    return dict(goal=g.tolist(), residual_deg=float(np.sqrt(np.mean(ang ** 2))), forward=float(np.mean(cosang > 0)),
                min_eig=float(ev[0] / K), n=int(K))


def compare(smc_raw, smc_cor, margin_deg=0.0):
    """Which frame does the policy agree with itself on? Returns 'raw', 'cor' or 'tie'. A variant whose rays are
    parallel (min_eig ~ 0) or point away from their own sub-goal is uninformative and never wins on its own."""
    def usable(s):
        return s["n"] >= 3 and np.isfinite(s["residual_deg"]) and s["min_eig"] > 1e-3 and s["forward"] >= 0.5
    ur, uc = usable(smc_raw), usable(smc_cor)
    if not ur and not uc:
        return "tie"
    if ur != uc:
        return "raw" if ur else "cor"
    diff = smc_raw["residual_deg"] - smc_cor["residual_deg"]
    return "tie" if abs(diff) <= margin_deg else ("cor" if diff > 0 else "raw")


# ---- beyond the blind spot: the measured nuisance PREDICTS how the policy's perceived sub-goal should move ------------
# Proposition 1: a policy that takes its own position from proprioception perceives a TRANSLATED goal under a moved
# camera, and its rays still meet. The wrapper, however, has measured the camera (policy-free). Geometry then predicts
# where such a policy must see a point x that really sits at height h: project x into the ACTUAL camera, back-project
# that pixel through the ASSUMED training camera onto the plane z = h. If the sub-goal triangulated on the raw frame
# sits where this map sends the sub-goal triangulated on the corrected frame, the policy was misperceiving exactly as
# geometry says and the correction is doing its job. If the sub-goal does not move at all, the policy is invariant to
# this nuisance and the correction is unnecessary: do nothing.
GOAL_MOVE_CM = 2.0            # below this the two sub-goals are the same point (triangulation noise on the organism: ~1 cm)


def predicted_raw_goal(g_cor, C_act, R_act, C_train, R_train):
    """Where a policy that assumes the training camera sees the world point g_cor when the actual camera is
    (C_act, R_act). MuJoCo camera convention (looks along -z); columns of R are camera axes in the world."""
    g = np.asarray(g_cor, float)
    pc = (g - C_act) @ R_act                              # point in the actual camera's frame
    if pc[2] > -1e-6:
        return None
    ray = np.array([pc[0] / -pc[2], pc[1] / -pc[2], -1.0])  # the pixel's ray, read as if it came from the training camera
    dw = R_train @ ray
    if abs(dw[2]) < 1e-9:
        return None
    t = (g[2] - C_train[2]) / dw[2]
    return None if t <= 0 else C_train + t * dw


def decide_v2(smc_raw, smc_cor, floor_deg, pred_raw_goal=None, margin_deg=1.0):
    """Minimal intervention. Returns (choice, reason), choice in {'raw', 'cor'}.
      A  raw residual above the policy's nominal floor and the correction lowers it      -> cor  (distortion removed)
      D  the correction raises the residual above the floor                               -> raw  (correction harmful)
      B  raw at the floor and the sub-goal does not move under the correction             -> raw  (policy invariant)
      C  raw at the floor, sub-goal moves, and it moves the way the MEASURED shift says    -> cor  (translated goal)
         raw at the floor, sub-goal moves some other way                                   -> raw  (unexplained: do nothing)
    floor_deg: the policy's residual in the training setup (the 'nominal' probe of the same task)."""
    rr, rc = smc_raw["residual_deg"], smc_cor["residual_deg"]
    if not (np.isfinite(rr) and np.isfinite(rc)) or smc_raw["goal"] is None or smc_cor["goal"] is None:
        return "cor", "uninformative: the wrapper's own decision stands"
    hi = floor_deg + margin_deg
    if rc > hi and rc > rr:
        return "raw", "D: the correction makes the policy less self-consistent"
    if rr > hi and rc < rr:
        return "cor", "A: the correction removes a distortion the raw frame shows"
    move = float(np.linalg.norm(np.array(smc_cor["goal"]) - np.array(smc_raw["goal"]))) * 100
    if move < GOAL_MOVE_CM:
        return "raw", f"B: the perceived sub-goal does not move ({move:.1f} cm): the policy is invariant here"
    if pred_raw_goal is not None:
        miss = float(np.linalg.norm(np.array(smc_raw["goal"]) - np.asarray(pred_raw_goal))) * 100
        if miss < 0.5 * move:
            return "cor", f"C: the sub-goal moves {move:.1f} cm, as the measured shift predicts (within {miss:.1f} cm)"
        return "raw", f"the sub-goal moves {move:.1f} cm but not as the measured shift predicts ({miss:.1f} cm off)"
    return "raw", f"the sub-goal moves {move:.1f} cm and no measured shift explains it"


# ---- action-field fingerprint (2026-09-19, after the OFT probe) ------------------------------------------------------
# The concurrency premise above is FALSE for OpenVLA-OFT: its opening action field DIVERGES (the commanded speed grows
# along the reach, as in the accelerating phase of its demonstrations), the triangulated point sits a few cm BEHIND the
# gripper in every task, and the residual is ~132 deg even at the training camera. What the probe data do carry is a
# FINGERPRINT: the actions of the policy over the probe cross, recorded once in the reference configuration of a task
# (when the system is commissioned and known to work), against the same actions under a candidate frame at deployment.
# No assumption about the shape of the field is needed. On OFT (development family, 72 units, stored rows): AUROC of a
# small corrected-frame distance for success of the wrapped policy 0.947, of a small raw-frame distance for raw success
# 0.786; with ONE probe 0.817 / 0.685, with three 0.874 / 0.739: sampling the FIELD through self-motion is what makes
# the fingerprint discriminative. Corrected distance <= 0.15: success 0.96 (n = 54); above 0.25: 0.00 (n = 6).
FP_UNAFFECTED = 0.10          # raw field within this of the fingerprint: the policy is unaffected, do nothing
FP_LOST = 0.25                # neither field within this: expect failure whatever is done


def field_distance(A, A_ref):
    """Relative distance between two sets of probe actions (K x 3, same probe order): ||A - A_ref|| / ||A_ref||."""
    A, A_ref = np.asarray(A, float), np.asarray(A_ref, float)
    return float(np.linalg.norm(A - A_ref) / max(np.linalg.norm(A_ref), 1e-9))


def decide_fingerprint(d_raw, d_cor):
    """(choice, reason) for the IMAGE channels of the wrapper; the state servo is never gated.
    1. raw field within FP_UNAFFECTED of the fingerprint  -> raw: the policy behaves as in training, leave it alone
    2. otherwise the frame whose field is closer          -> cor if the correction brings the policy back, else raw
    The reason flags d > FP_LOST on the chosen frame: the policy does not behave as in training under either frame."""
    if d_raw <= FP_UNAFFECTED:
        return "raw", f"unaffected: raw field {d_raw:.3f} from the fingerprint"
    choice, d = ("cor", d_cor) if d_cor < d_raw else ("raw", d_raw)
    lost = "; EXPECT FAILURE, neither frame looks like training" if d > FP_LOST else ""
    return choice, f"{choice} field closer ({d_raw:.3f} raw, {d_cor:.3f} corrected){lost}"
