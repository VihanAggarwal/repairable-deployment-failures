"""Simulator-free checks for the graded-sweep analysis: the v1 normalisation bug is reproduced, the v2 own-block
normalisation and the recoverability score are exercised end to end, and the body shading statistic is affine
invariant. Run: uv run --no-project --with numpy --with scipy --with opencv-python-headless python wrapper/_sweep_tests.py"""
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import sweep_analysis as A  # noqa: E402
import wrapper_libero as W  # noqa: E402
from sweep_conditions import BY_NAME  # noqa: E402

TASKS = [1167, 1208, 1442]


def write_rows(path, rows):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- 1. the v1 bug: every family divided by the camera block
d = tempfile.mkdtemp()
v1 = []
for t in TASKS:
    for name, c in BY_NAME.items():
        sig = {"camera": 0.05, "nominal": 0.05, "light": 0.00005, "noise": 0.002, "joint": 0.015, "background": 0.0}[c["family"]]
        v1.append(dict(task_id=t, base=t, cond=name, family=c["family"], level=c["level"], override=c["override"], sigma_min=sig))
write_rows(os.path.join(d, "scores.jsonl"), v1)
rows, pred, version = A.load_scores(d)
assert version == "v1"
# a light condition whose own block is as sensitive as at nominal still scores 0.001 under v1: that is the bug
assert abs(pred[(1167, "gain_1.8")]["score"] - 0.001) < 1e-9, pred[(1167, "gain_1.8")]
assert np.isnan(pred[(1167, "gain_1.8")]["recover"])

# ---------------------------------------------------------------- 2. v2: own-block reference, recoverability composition
v2 = []
for t in TASKS:
    for name, c in BY_NAME.items():
        fam = c["family"]
        sig_ref = {"camera": 0.05, "nominal": 0.05, "light": 0.00005, "noise": 0.002, "joint": 0.015, "background": 0.0}[fam]
        sig = sig_ref * (0.25 if name == "look_away_55" else 1.0)          # arm half out of view: measurability drops
        cov = {"orbit_az45": 0.62, "look_away_35": 0.30, "look_away_55": 0.15, "pullback_1.3": 0.9}.get(name, 1.0 if fam != "camera" else 0.85)
        photo_d, photo_ref = (2.0, 0.8) if name == "grazing" else (0.8, 0.8)
        noise_inv = {"sigma5": 0.95, "sigma10": 0.9, "sigma20": 0.7, "sigma40": 0.2}.get(name, 1.0)
        v2.append(dict(task_id=t, base=t, cond=name, family=fam, level=c["level"], override=c["override"],
                       sigma_min=sig, sigma_min_ref=sig_ref, coverage=cov, photo_d=photo_d, photo_d_ref=photo_ref,
                       noise_inv=noise_inv))
write_rows(os.path.join(d, "scores_v2.jsonl"), v2)
rows, pred, version = A.load_scores(d)
assert version == "v2"
p = pred[(1167, "gain_1.8")]
assert abs(p["ident"] - 1.0) < 1e-9 and abs(p["invert"] - 1.0) < 1e-9 and abs(p["recover"] - 1.0) < 1e-9, p
p = pred[(1167, "look_away_35")]
assert abs(p["ident"] - 1.0) < 1e-9 and abs(p["recover"] - 0.30) < 1e-9, p      # measurable, not reconstructible
p = pred[(1167, "look_away_55")]
assert abs(p["ident"] - 0.25) < 1e-9 and abs(p["recover"] - 0.25 * 0.15) < 1e-9, p
p = pred[(1167, "grazing")]
assert abs(p["invert"] - 0.4) < 1e-9, p                                          # shading pattern changed: affine model invalid
assert pred[(1167, "offset0.20")]["invert"] == 1.0 and pred[(1167, "table_red")]["invert"] == 0.0
assert pred[(1167, "table_red")]["ident"] == 0.0 and pred[(1167, "table_red")]["recover"] == 0.0   # 0/0: unmeasurable, not nan
assert abs(pred[(1167, "sigma40")]["recover"] - 0.2) < 1e-9
r = A.recoverability(dict(family="camera", sigma_min=0.1, sigma_min_ref=0.05, coverage=0.5))
assert r["ident"] == 1.0 and r["recover"] == 0.5                                   # ident is clipped at 1

# ---------------------------------------------------------------- 3. end to end: recoverability separates outcomes v1 could not
# outcomes mirror the OFT sweep: orbits recover, look-aways never, sigma40 never, joint offsets always
eps = []
for t in TASKS:
    for name, c in BY_NAME.items():
        raw = name in ("nominal", "dir_az90", "gain_0.4", "gain_1.8", "tint_warm", "sigma20", "table_red", "table_dark", "colored_room")
        full = name not in ("look_away_35", "look_away_55", "sigma40", "grazing")
        base = dict(policy="fake", task_id=t, base=t, cond=name, family=c["family"], level=c["level"],
                    override=c["override"], first_chunk=[0.0] * 6, probe=0.1)
        eps.append(dict(base, arm="raw", success=bool(raw), steps=10))
        eps.append(dict(base, arm="full", success=bool(full), steps=10, cam_applied=c["family"] == "camera",
                        photo_applied=c["family"] == "light", denoise_applied=["agentview_image"] if c["family"] == "noise" else [],
                        state_applied=c["family"] == "joint"))
write_rows(os.path.join(d, "episodes_fake.jsonl"), eps)
r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "sweep_analysis.py"), "--out_dir", d],
                   capture_output=True, text=True)
print(r.stdout[-1800:], r.stderr[-800:])
assert r.returncode == 0, r.stderr
rep = json.load(open(os.path.join(d, "sweep_report.json")))
au = rep["policies"]["fake"]
assert au["recover"]["auroc"] > 0.9 > au["score"]["auroc"], (au["recover"], au["score"])
assert "noise floor: 0 of" in r.stdout

# ---------------------------------------------------------------- 4. v1 file only: analysis still runs, warns, no recover column
os.remove(os.path.join(d, "scores_v2.jsonl"))
r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "sweep_analysis.py"), "--out_dir", d],
                   capture_output=True, text=True)
assert r.returncode == 0 and "WARNING: v1" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
assert "recover" not in json.load(open(os.path.join(d, "sweep_report.json")))["policies"]["fake"]

# ---------------------------------------------------------------- 5. body shading statistic: affine invariant, moves under a gradient
class Fake(W.Canonicalizer):
    def __init__(self):
        self.cam, self.flip = "agentview", 0
        self.mask = np.zeros((64, 64), bool); self.mask[16:48, 20:44] = True

    def obs_robot_mask(self, obs):
        return self.mask


can = Fake()
rng = np.random.default_rng(0)
img = np.clip(rng.normal(120, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
d0 = can.photometric_consistency({"agentview_image": img})
d1 = can.photometric_consistency({"agentview_image": np.clip(img.astype(np.float32) * 1.6 + 10, 0, 255).astype(np.uint8)})
grad = np.clip(img.astype(np.float32) * np.linspace(0.3, 1.5, 64)[:, None, None], 0, 255).astype(np.uint8)
d2 = can.photometric_consistency({"agentview_image": grad})
assert abs(d1 - d0) < 0.15 * d0 + 0.05, (d0, d1)                                 # global gain/offset: unchanged
assert d2 > 3 * d0, (d0, d2)                                                      # vertical shading gradient: moves
can.mask = np.zeros((64, 64), bool); can.mask[30:33, 30:33] = True                # body too small to split
assert np.isnan(can.photometric_consistency({"agentview_image": img}))

print("ALL SWEEP TESTS PASS")
