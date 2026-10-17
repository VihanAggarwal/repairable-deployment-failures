"""Simulator-free checks for the claim 1-3 pieces: photometric/noise helpers, reprojection mask, the learned
canonicalizer (identity at init, trains on self-labelled shards), and the invertibility report."""
import json, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import wrapper_libero as W
import torch
from canon_net import CanonNet
import invertibility_report as IR

rng = np.random.default_rng(0)
# noise floor: a smooth image is near 0, sigma-10 noise reads near 10, denoising lowers it
yy, xx = np.mgrid[0:128, 0:128]
clean = np.stack([xx * 1.5, yy * 1.5, np.full_like(xx, 100)], -1).astype(np.uint8)
noisy = np.clip(clean + rng.normal(0, 10, clean.shape), 0, 255).astype(np.uint8)
l0, l1 = W.noise_level(clean), W.noise_level(noisy)
l2 = W.noise_level(W.denoise(noisy, l1))
print(f"noise level clean {l0:.2f} noisy {l1:.2f} denoised {l2:.2f}")
assert l0 < 1.5 and l1 > 5 and l2 < 0.5 * l1

# reprojection mask: identity reprojection observes every pixel
C0 = np.array([0.6586, 0.0, 1.6104]); R0 = W.quat_wxyz_to_mat([0.638, 0.3048, 0.3048, 0.638])
z = np.full((64, 64), 1.2)
_, cov, filled = W.reproject(clean[:64, :64], z, 60.0, C0, R0, C0, R0, splat=1, inpaint=False, return_mask=True)
assert filled.shape == (64, 64) and filled.all() and cov == 1.0

# canonicalizer network: exact identity at init, output in range, gradients flow
net = CanonNet().eval()
x = torch.rand(2, 3, 64, 64)
with torch.no_grad():
    y, alpha = net(x)
assert y.shape == x.shape and float(alpha.min()) > 0.95
assert float((y - x).abs().mean()) < 0.03, float((y - x).abs().mean())
net.train()
y, _ = net(x)
y.mean().backward()
assert all(p.grad is not None for p in net.parameters() if p.requires_grad)

# training script end to end on tiny synthetic shards: target = input shifted right by 6 pixels
d = tempfile.mkdtemp()
for tid in range(6):
    X = rng.integers(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    X = np.stack([np.stack([np.clip(np.mgrid[0:64, 0:64][1] * 4 + tid * 10 + k, 0, 255)] * 3, -1) for k in range(3)]).astype(np.uint8)
    Y = np.roll(X, 6, axis=2)
    M = np.ones((3, 64, 64), bool); M[:, :, :6] = False
    meta = dict(task_id=tid, category="camera" if tid % 2 else "language")
    np.savez_compressed(os.path.join(d, f"task_{tid:05d}.npz"), x=X, y=Y, m=M, meta=json.dumps(meta))
ck = os.path.join(d, "canon.pt")
r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "train_canon.py"), "--data_dir", d, "--out", ck,
                    "--epochs", "3", "--bs", "4", "--feat_w", "0", "--val_frac", "0.34"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-1500:])
assert r.returncode == 0 and os.path.exists(ck)
import canon_net
loaded = canon_net.load(ck, device="cpu")
out = canon_net.apply(loaded, X[0], device="cpu")
assert out.shape == X[0].shape and out.dtype == np.uint8

# oracle target: shards without an oracle render are refused; shards with one train against it (every pixel scored)
r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "train_canon.py"), "--data_dir", d, "--out", ck + "o",
                    "--epochs", "1", "--bs", "4", "--feat_w", "0", "--target", "oracle"], capture_output=True, text=True)
assert r.returncode != 0 and "no oracle render" in (r.stdout + r.stderr), (r.stdout[-300:], r.stderr[-300:])
d2 = tempfile.mkdtemp()
for tid in range(6):
    X = np.stack([np.stack([np.clip(np.mgrid[0:64, 0:64][1] * 4 + tid * 10 + k, 0, 255)] * 3, -1) for k in range(3)]).astype(np.uint8)
    Y = np.roll(X, 6, axis=2); O = np.roll(X, 3, axis=1)
    M = np.ones((3, 64, 64), bool); M[:, :, :6] = False
    meta = dict(task_id=tid, category="camera" if tid % 2 else "language")
    np.savez_compressed(os.path.join(d2, f"task_{tid:05d}.npz"), x=X, y=Y, m=M, o=O, meta=json.dumps(meta))
r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "train_canon.py"), "--data_dir", d2, "--out", ck + "o",
                    "--epochs", "2", "--bs", "4", "--feat_w", "0", "--val_frac", "0.34", "--target", "oracle"],
                   capture_output=True, text=True)
assert r.returncode == 0 and "target oracle" in r.stdout and "mean observed fraction 1.00" in r.stdout, r.stdout[-400:]
assert torch.load(ck + "o", map_location="cpu")["target"] == "oracle"

# measured-geometry input: 7 channels (reprojection + hole mask + frame), checkpoint records in_ch, apply takes 7ch
r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "train_canon.py"), "--data_dir", d2, "--out", ck + "r",
                    "--epochs", "2", "--bs", "4", "--feat_w", "0", "--val_frac", "0.34", "--target", "oracle",
                    "--input", "reproj"], capture_output=True, text=True)
assert r.returncode == 0 and "input reproj (7 ch)" in r.stdout, r.stdout[-400:] + r.stderr[-400:]
ckr = torch.load(ck + "r", map_location="cpu")
assert ckr["net_args"]["in_ch"] == 7 and ckr["input"] == "reproj"
net7 = canon_net.load(ck + "r", device="cpu")
inp = np.concatenate([X[0], np.full(X[0].shape[:2] + (1,), 255, np.uint8), X[0]], -1)
out = canon_net.apply(net7, inp, device="cpu")
assert out.shape == X[0].shape and out.dtype == np.uint8

# invertibility report on synthetic rows
base = [dict(task_id=i, category=c, success=s) for i, (c, s) in enumerate(
    [("camera", False)] * 6 + [("camera", True)] * 4 + [("layout", True)] * 5 + [("layout", False)] * 5)]
wrap = [dict(task_id=i, category=b["category"], arm="full",
             success=(True if (b["category"] == "camera" and i < 5) else b["success"])) for i, b in enumerate(base)]
wrap += [dict(task_id=i, category="camera", arm="oracle", success=True) for i in range(10)]
bp, wp = os.path.join(d, "b.jsonl"), os.path.join(d, "w.jsonl")
open(bp, "w").writelines(json.dumps(x) + "\n" for x in base)
open(wp, "w").writelines(json.dumps(x) + "\n" for x in wrap)
rep = IR.family_report("toy", bp, wp)
cam = rep["categories"]["camera"]
assert cam["recovered"] == 5 and cam["harmed"] == 0 and abs(cam["full"] - 0.9) < 1e-9
assert rep["classes"]["non-invertible"]["harmed"] == 0 and abs(IR.mcnemar_p(5, 0) - 0.0625) < 1e-9
print("ALL CLAIM TESTS PASS")
