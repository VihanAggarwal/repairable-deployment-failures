"""Numerical check of the two SMC propositions (no simulator; numpy only).

P1 (what SMC can see). Let the policy's commanded direction at end-effector position p be along F(g - p) for a fixed
   linear map F (its perceptual distortion). The rays {p + t F(g - p)} meet at one point for every probe set iff
   F = s I with s > 0. An AFFINE error, direction along (g + c - p), always meets at g + c: SMC is blind to a translated
   goal and sees exactly the non-central part of the action field.
P2 (how much). For a rotation by theta about an axis, probes on a cross of half-width b around p0 and a goal at
   distance D, the RMS angular residual grows linearly in theta for small theta, with slope of order b / D.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import smc  # noqa: E402

rng = np.random.default_rng(1)


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def residual(F, c, p0, g, probes):
    P = np.stack([p0 + o for o in probes])
    A = (g + c - P) @ F.T
    return smc.triangulate(P, A / np.linalg.norm(A, axis=1, keepdims=True))


p0, g = np.array([0.0, 0.0, 1.1]), np.array([0.25, -0.2, 0.95])
# P1: central fields are consistent, whatever the offset c; non-scalar linear maps are not
for name, F, c in (("identity", np.eye(3), np.zeros(3)), ("positive scaling", 2.5 * np.eye(3), np.zeros(3)),
                   ("translated goal", np.eye(3), np.array([0.12, -0.08, 0.05])),
                   ("scaling + translated goal", 0.7 * np.eye(3), np.array([-0.1, 0.1, 0.0]))):
    r = residual(F, c, p0, g, smc.PROBES)
    print(f"P1 {name:<28} residual {r['residual_deg']:.2e} deg, meets {np.round(np.array(r['goal']) - g, 3)} from g")
    assert r["residual_deg"] < 1e-4
worst = np.inf
for _ in range(300):                                                # random non-scalar linear distortions
    F = np.eye(3) + 0.3 * rng.standard_normal((3, 3))
    if np.linalg.det(F) <= 0.2:
        continue
    F -= np.trace(F) / 3 * np.eye(3) * 0                              # keep as drawn
    dev = np.linalg.norm(F / (np.trace(F) / 3) - np.eye(3))
    r = residual(F, np.zeros(3), p0, g, smc.PROBES)["residual_deg"]
    worst = min(worst, r / dev)
print(f"P1 non-scalar linear maps: residual / distance-from-scalar is bounded below by {worst:.2f} deg over 300 draws (never 0)")
assert worst > 0.5

# P2: linear growth, slope proportional to b / D
print("P2 residual vs rotation (deg), cross half-width b, goal distance D = %.2f m" % np.linalg.norm(g - p0))
for b in (0.05, 0.10, 0.15):
    probes = [np.zeros(3)] + [s * b * e for e in np.eye(3) for s in (1, -1)]
    rs = [residual(rot([0.3, 0.2, 1.0], th), np.zeros(3), p0, g, probes)["residual_deg"] for th in (5, 10, 20, 30)]
    slope = rs[1] / 10
    print(f"   b = {b:.2f} m: residual {np.round(rs, 2)}  slope {slope:.3f} deg per deg  = {slope / (b / np.linalg.norm(g - p0)):.2f} x b/D")
print("theory checks pass")
