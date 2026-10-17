"""Self-motion consistency: synthetic validation (no simulator, no policy weights).

    uv run --no-project --with numpy python wrapper/_smc_tests.py

Fake policy: commands a (saturating, noisy) displacement toward a goal, but PERCEIVES the end-effector-to-goal vector
through a camera it believes is the training camera. If the real camera is rotated by E relative to that belief and
nothing corrects the image, the perceived vector is E (g - p). A correction with residual error E_c leaves E_c (g - p).
  S1  correct perception: rays meet, residual ~ 0 (up to action noise)
  S2  residual grows monotonically with the uncorrected camera rotation
  S3  compare() prefers the better-corrected frame, and prefers RAW when the correction is worse than doing nothing
  S4  selection among candidate corrections recovers the right one (policy-as-sensor calibration)
  S5  a policy that ignores position (parallel rays) is flagged uninformative, not consistent
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import smc  # noqa: E402

rng = np.random.default_rng(0)
p0 = np.array([0.0, 0.0, 1.10])
goal = np.array([0.18, -0.22, 0.95])
P = np.stack([p0 + o for o in smc.PROBES])


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def policy(P, E, noise=0.03, sat=0.9):
    v = (goal[None] - P) @ E.T
    a = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9) * sat        # saturated reach: direction only
    return a + noise * rng.standard_normal(a.shape)


axis = [0.3, 0.2, 1.0]
s = smc.triangulate(P, policy(P, np.eye(3)))
print(f"S1 correct perception: residual {s['residual_deg']:.1f} deg, forward {s['forward']:.2f}, "
      f"sub-goal error {np.linalg.norm(np.array(s['goal']) - goal) * 100:.1f} cm")
assert s["residual_deg"] < 5 and s["forward"] == 1.0 and np.linalg.norm(np.array(s["goal"]) - goal) < 0.05

res = []
for deg in (0, 10, 20, 30, 45, 60):
    r = np.mean([smc.triangulate(P, policy(P, rot(axis, deg)))["residual_deg"] for _ in range(20)])
    res.append(r)
print("S2 residual vs uncorrected camera rotation (0,10,20,30,45,60 deg):", np.round(res, 1))
assert all(b > a for a, b in zip(res, res[1:])) and res[3] > 2 * res[0]      # 30 deg is twice the noise floor; 10 deg is NOT separable

wins = {"good": 0, "harm": 0}
for _ in range(200):
    raw = smc.triangulate(P, policy(P, rot(axis, 35)))
    good = smc.triangulate(P, policy(P, rot(axis, 4)))               # a correction that works
    wins["good"] += smc.compare(raw, good) == "cor"
    ok_raw = smc.triangulate(P, policy(P, rot(axis, 3)))             # the policy was coping on its own
    bad = smc.triangulate(P, policy(P, rot(axis, 25), noise=0.08))   # the correction made it worse
    wins["harm"] += smc.compare(ok_raw, bad) == "raw"
print(f"S3 picks the working correction {wins['good'] / 2:.0f}% of the time; refuses the harmful one {wins['harm'] / 2:.0f}%")
assert wins["good"] >= 190 and wins["harm"] >= 190

true_err = 38.0
cands = np.arange(0, 61, 4.0)                                          # candidate corrections: rotate back by c degrees
hits = []
for _ in range(100):
    r = [smc.triangulate(P, policy(P, rot(axis, true_err - c)))["residual_deg"] for c in cands]
    hits.append(cands[int(np.argmin(r))])
print(f"S4 policy-as-sensor calibration: true rotation {true_err} deg, selected {np.mean(hits):.1f} +/- {np.std(hits):.1f} deg")
assert abs(np.mean(hits) - true_err) < 4 and np.std(hits) < 8

flat = smc.triangulate(P, np.tile([0.5, -0.6, -0.3], (len(P), 1)) + 0.01 * rng.standard_normal((len(P), 3)))
print(f"S5 position-blind policy: min_eig {flat['min_eig']:.4f} (uninformative), compare vs a good frame -> "
      f"{smc.compare(flat, smc.triangulate(P, policy(P, np.eye(3))))}")
assert flat["min_eig"] < 1e-3 and smc.compare(flat, smc.triangulate(P, policy(P, np.eye(3)))) == "cor"
print("all SMC tests pass")

# ---- decide_v2: minimal intervention, with the measured camera shift covering the translated-goal blind spot ----------
import wrapper_libero as W  # noqa: E402

Cn = np.array([0.6586, 0.0, 1.6103])
Rn = W.quat_wxyz_to_mat([0.6380, 0.3048, 0.3048, 0.6380])
Ca, Ra = W.orbit(Cn, Rn, W.optical_pivot(Cn, Rn, 0.90), 30, 0, 1.0)             # the ACTUAL camera: a 30 deg orbit
g_true = np.array([0.10, -0.15, 0.95])
P2 = np.stack([np.array([-0.05, 0.05, 1.15]) + o for o in smc.PROBES])


def see(x, C_act, R_act):                                                        # what an absolute-archetype policy perceives
    return smc.predicted_raw_goal(x, C_act, R_act, Cn, Rn)


def rays(goal, noise=0.01):
    a = goal[None] - P2
    return a / np.linalg.norm(a, axis=1, keepdims=True) * 0.9 + noise * rng.standard_normal(a.shape)


floor = smc.triangulate(P2, rays(g_true))["residual_deg"]
g_seen = see(g_true, Ca, Ra)
print(f"decide_v2: floor {floor:.2f} deg; an absolute-archetype policy sees the goal {np.linalg.norm(g_seen - g_true) * 100:.1f} cm off under the raw frame")
tally = {k: 0 for k in "ABCDU"}
for _ in range(100):
    # C: translated goal, rays meet on both frames; the MEASURED camera (2 cm, 1 deg off the truth) predicts the move
    Cm, Rm = Ca + 0.02 * rng.standard_normal(3) / np.sqrt(3), W.rotvec_to_mat(np.radians(1.0) * np.array([0, 0, 1.0])) @ Ra
    raw, cor = smc.triangulate(P2, rays(g_seen)), smc.triangulate(P2, rays(g_true))
    tally["C"] += smc.decide_v2(raw, cor, floor, see(np.array(cor["goal"]), Cm, Rm))[0] == "cor"
    # B: the policy is invariant to the nuisance: same goal on both frames -> do nothing
    tally["B"] += smc.decide_v2(smc.triangulate(P2, rays(g_true)), smc.triangulate(P2, rays(g_true)), floor,
                                see(g_true, Cm, Rm))[0] == "raw"
    # D: the correction damages perception (noisy, rotated rays) -> do nothing
    bad = (g_true[None] - P2) @ rot(axis, 25).T
    bad = bad / np.linalg.norm(bad, axis=1, keepdims=True) * 0.9 + 0.06 * rng.standard_normal(bad.shape)
    tally["D"] += smc.decide_v2(smc.triangulate(P2, rays(g_true)), smc.triangulate(P2, bad), floor, None)[0] == "raw"
    # A: relative archetype, distortion on the raw frame removed by the correction
    dist = (g_true[None] - P2) @ rot(axis, 35).T
    dist = dist / np.linalg.norm(dist, axis=1, keepdims=True) * 0.9 + 0.01 * rng.standard_normal(dist.shape)
    tally["A"] += smc.decide_v2(smc.triangulate(P2, dist), smc.triangulate(P2, rays(g_true)), floor, None)[0] == "cor"
    # U: the correction moves the goal somewhere the measured shift does not explain -> do nothing
    # (a move that happens to point along the geometric displacement, here +y, is half explained and is NOT
    # separable: miss 6.8 cm against a 6 cm threshold. The rule is blind to a harm that mimics the measured shift.)
    wrong = g_true + np.array([0.0, -0.12, 0.0])
    tally["U"] += smc.decide_v2(smc.triangulate(P2, rays(g_true)), smc.triangulate(P2, rays(wrong)), floor,
                                see(wrong, Cm, Rm))[0] == "raw"
print("decide_v2 correct decisions out of 100:", tally)
assert all(v >= 95 for v in tally.values()), tally
print("decide_v2 tests pass")


# ---- action-field fingerprint: a behaviour-cloned style policy whose opening field DIVERGES (what OpenVLA-OFT does) ----
def bc_policy(P, E, noise=0.01):
    """Opening field of a behaviour-cloned policy, as measured on OpenVLA-OFT: it DIVERGES from a point a few cm behind
    the gripper along the perceived direction of the reach, a = k (p - c), c = centre - 6 cm * unit(E (g - centre)).
    Noise 0.01 reproduces OFT's natural field variability (relative distance about 0.04)."""
    ctr = P.mean(0)
    d = E @ (goal - ctr)
    c = ctr - 0.06 * d / np.linalg.norm(d)
    return 5.0 * (P - c) + noise * rng.standard_normal(P.shape)


ref_field = bc_policy(P, np.eye(3), noise=0.0)
nom = smc.triangulate(P, bc_policy(P, np.eye(3)))
print(f"fingerprint: OFT-like diverging policy at the TRAINING camera: concurrency residual {nom['residual_deg']:.1f} deg, "
      f"{nom['forward']:.2f} of the rays point at the triangulated point (the geometric premise is simply false for it)")
assert nom["residual_deg"] > 90 and nom["forward"] < 0.5
d_shift = [np.mean([smc.field_distance(bc_policy(P, rot(axis, a)), ref_field) for _ in range(20)]) for a in (0, 10, 20, 30, 45)]
print("fingerprint distance vs uncorrected camera rotation (0,10,20,30,45 deg):", np.round(d_shift, 3))
assert all(b > a for a, b in zip(d_shift, d_shift[1:])) and d_shift[0] < smc.FP_UNAFFECTED < d_shift[2]
ok = {"correct": 0, "leave alone": 0, "refuse harm": 0}
for _ in range(200):
    f = lambda deg, n=0.01: smc.field_distance(bc_policy(P, rot(axis, deg), n), ref_field)
    ok["correct"] += smc.decide_fingerprint(f(35), f(3))[0] == "cor"            # broken raw, working correction
    ok["leave alone"] += smc.decide_fingerprint(f(2), f(6))[0] == "raw"         # policy unaffected: do nothing
    ok["refuse harm"] += smc.decide_fingerprint(f(15), f(30, 0.06))[0] == "raw" # the correction makes it worse
print("fingerprint decisions right out of 200:", ok)
assert all(v >= 190 for v in ok.values()), ok
print("fingerprint tests pass")
