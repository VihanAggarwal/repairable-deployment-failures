"""The do-no-harm gate: synthetic tests (no simulator, no GPU, no policy weights).

    uv run --no-project --with numpy --with scipy --with opencv-python-headless python wrapper/_gate_tests.py

A fake policy stands in for the frozen one: a smooth readout of the image that turns erratic (large action change under
tiny input jitter) on frames carrying a marker for "this correction pushed the frame off-distribution".
  G1  identical frames: no veto (ties leave the wrapper's decision standing)
  G2  the correction makes the policy erratic: veto
  G3  the correction makes the policy steadier: no veto
  G4  prepare_full end to end: the harmful channel is vetoed and NOT applied per step, the helpful one is kept,
      every decision is logged, and the wrapper's own wishes are recorded separately from what was applied
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
sys.modules["calib_check"] = types.SimpleNamespace(make_env=None)          # needs torch + LIBERO; not under test
import sweep_run as R  # noqa: E402
import wrapper_libero as W  # noqa: E402

rng = np.random.default_rng(0)
frame = rng.integers(60, 200, (64, 64, 3)).astype(np.uint8)
frame = np.ascontiguousarray(np.repeat(np.repeat(frame[::8, ::8], 8, 0), 8, 1))      # blocky, so 2 px shifts matter a little


def policy(o, erratic_gain=40.0):
    """Smooth in the image; erratic where the top-left pixel carries the marker 255."""
    img = np.asarray(o["agentview_image"], float)
    feats = np.array([img[:, :32].mean(), img[:, 32:].mean(), img[:32].mean(), img[32:].mean(), img.std(), img[8:24, 8:24].mean()])
    a = (feats - 128.0) / 128.0
    if img[0, 0, 0] == 255 or img[0, 1, 0] == 255 or img[1, 0, 0] == 255 or img[0, 2, 0] == 255 or img[2, 0, 0] == 255:
        a = a + erratic_gain * np.sin(img.sum() * 1e-3 + np.arange(6))                # tiny input change, large output change
    return a


obs = {"agentview_image": frame}
p_raw = R.gate_profile(policy, obs)

# G1
d = R.gate_decide(p_raw, R.gate_profile(policy, {"agentview_image": frame.copy()}))
assert not d["veto"] and d["action_change"] == 0.0, d
# G2
bad = frame.copy()
bad[:3, :3] = 255
d = R.gate_decide(p_raw, R.gate_profile(policy, {"agentview_image": bad}))
assert d["veto"] and d["worse"] >= R.GATE_MIN_WORSE and d["s_cor"] > d["s_raw"], d
# G3: start from the erratic frame, the correction removes the marker
d = R.gate_decide(R.gate_profile(policy, {"agentview_image": bad}), p_raw)
assert not d["veto"] and d["s_cor"] < d["s_raw"], d
print("G1 to G3 pass:", {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()})


# G4: prepare_full with a fake Canonicalizer. photo = the harmful channel (writes the marker), cam = a benign shift
class FakeCan:
    def __init__(self, env, nominal, sensing=None):
        self.Cn, self.Rn = np.zeros(3), np.eye(3)
        self.C_est, self.R_est, self.mask_diag = np.zeros(3), np.eye(3), []

    def joint_dev(self):
        return 0.0

    def sweep(self, obs):
        return obs, []

    def calibrate_v2(self, recs):
        return dict(applied=True, method="silhouette")

    def true_camera(self):
        return np.zeros(3), np.eye(3)

    def photometric_fit(self, probe, nominal):
        return dict(photo_ok=True, photo_mean_shift=99.0, photo_gain=[1.0, 1.0, 1.0])

    def apply_photo(self, img):
        out = img.copy()
        out[:3, :3] = 255
        return out

    def canonical_image(self, o, fill="ref"):
        return np.roll(o["agentview_image"], 1, axis=1), 0.9


W.Canonicalizer = FakeCan
nominal = dict(noise={})
o2, transform, info = R.prepare_full(None, dict(obs), nominal, lambda o: o, gate=policy)
assert info["gate_vetoes"] == ["photo"], info["gate_vetoes"]
assert info["wanted_photo"] and not info["photo_applied"], "the wish is logged, the vetoed channel is not applied"
assert info["wanted_cam"] and info["cam_applied"] and not info["gate_cam"]["veto"]
step = transform({"agentview_image": frame.copy()})
assert step["agentview_image"][0, 0, 0] != 255 or frame[0, 0, 0] == 255, "vetoed photometric map must not run per step"
assert (step["agentview_image"] == np.roll(frame, 1, axis=1)).all(), "the kept camera channel still runs"
# without the gate the same wrapper applies both (the harm case the gate exists for)
_, t_full, info_full = R.prepare_full(None, dict(obs), nominal, lambda o: o, gate=None)
assert info_full["photo_applied"] and "gate_vetoes" not in info_full
assert (t_full({"agentview_image": frame.copy()})["agentview_image"][:3, 1:4] == 255).all()
print("G4 pass: photo vetoed (", {k: round(v, 2) if isinstance(v, float) else v for k, v in info["gate_photo"].items()},
      "), cam kept; ungated wrapper applies both")
print("all gate tests pass")
