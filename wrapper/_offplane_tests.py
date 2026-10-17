"""Off-plane depth and learned-mask calibration: synthetic regression tests (no simulator, no GPU).

    uv run --no-project --with numpy --with scipy --with opencv-python-headless python wrapper/_offplane_tests.py

A box-world ray caster stands in for the renderer (raw buffer row 0 = bottom, camera looks along -z), and the REAL
Canonicalizer.canonical_image / LearnedSensing.metric_depth / plane_depth / hybrid_depth / points_depth / reproject
run on top of it through a stub simulator. What is tested is geometry at the EXACT camera pose:

  T1  the defect: under `plane` depth a held object is torn off the gripper and a tall fixture is displaced
  T2  `hybrid` depth keeps the held object on the gripper (held-object rule, proprioception only)
  T3  `hybrid` depth puts a tall static fixture where simulator depth puts it (training-setup static cloud)
  T4  `hybrid` changes nothing where `plane` was already right (table pixels, robot pixels)
  T5  which learned-mask errors move the silhouette fit: not blobs or missing links, but a uniform boundary bias
  T6  agreement of shift MAGNITUDES is only a lower bound on the pose error (why "calibration exact" was unproven)
"""
import os
import sys
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import wrapper_libero as W  # noqa: E402

TABLE_Z, FOVY = 0.90, 45.0
NEAR, FAR, EXTENT = 0.01, 50.0, 4.0
ROBOT, GRIP, HELD, REST, TALL = (200, 30, 30), (230, 120, 20), (20, 200, 40), (30, 60, 220), (160, 30, 200)
KIND = {ROBOT: "robot", GRIP: "grip", HELD: "movable", REST: "movable", TALL: "static"}


def boxes(held_lifted=True):
    held = ((-0.05, -0.03, 1.02), (0.01, 0.03, 1.12)) if held_lifted else ((0.20, -0.20, 0.90), (0.26, -0.14, 1.00))
    return [((-0.30, -0.04, 0.90), (-0.22, 0.04, 1.40), ROBOT),
            ((-0.22, -0.045, 1.12), (0.01, 0.045, 1.18), GRIP),
            ((-0.05, -0.045, 1.05), (0.01, -0.03, 1.12), GRIP),       # fingers either side of the held object (in y,
            ((-0.05, 0.03, 1.05), (0.01, 0.045, 1.12), GRIP),         # so the camera on +x still sees the object)
            (held[0], held[1], HELD),
            ((0.10, 0.15, 0.90), (0.16, 0.21, 1.00), REST),
            ((-0.10, -0.40, 0.90), (0.10, -0.25, 1.15), TALL)]


def raycast(C, R, H, Wd, scene):
    """Raw-buffer rgb, axis depth, and a per-pixel kind map ('table', 'robot', 'grip', 'movable', 'static', '')."""
    rr, cc = np.meshgrid(np.arange(H), np.arange(Wd), indexing="ij")
    ty = np.tan(np.deg2rad(FOVY) / 2)
    d = np.stack([((cc + 0.5) / Wd * 2 - 1) * ty * Wd / H, ((rr + 0.5) / H * 2 - 1) * ty, -np.ones((H, Wd))], -1) @ R.T
    t_best = np.full((H, Wd), np.inf)
    rgb = np.zeros((H, Wd, 3), np.uint8) + 90
    kind = np.full((H, Wd), "", dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = (TABLE_Z - C[2]) / d[..., 2]
    hit = (tp > 1e-6) & np.isfinite(tp)
    P = C + np.where(hit, tp, 0)[..., None] * d
    chk = ((np.floor(P[..., 0] / 0.05) + np.floor(P[..., 1] / 0.05)) % 2).astype(int)
    tex = np.stack([120 + 60 * chk, 100 + 50 * np.sin(7 * P[..., 0]), 80 + 60 * np.cos(5 * P[..., 1])], -1)
    rgb[hit] = np.clip(tex[hit], 0, 255).astype(np.uint8)
    t_best[hit] = tp[hit]
    kind[hit] = "table"
    for lo, hi, col in scene:
        lo, hi = np.array(lo), np.array(hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - C) / d, (hi - C) / d
        tmin, tmax = np.nanmax(np.minimum(t1, t2), -1), np.nanmin(np.maximum(t1, t2), -1)
        h = (tmax >= tmin) & (tmin > 1e-6) & (tmin < t_best)
        t_best[h], rgb[h], kind[h] = tmin[h], col, KIND[col]
    Pw = C + np.where(np.isfinite(t_best), t_best, FAR)[..., None] * d
    z = np.where(np.isfinite(t_best), -((Pw - C) @ R)[..., 2], FAR)
    return rgb, z, kind


class FakeSim:
    def __init__(self, can):
        self.can = can
        self.model = types.SimpleNamespace(stat=types.SimpleNamespace(extent=EXTENT),
                                           vis=types.SimpleNamespace(map=types.SimpleNamespace(znear=NEAR, zfar=FAR)))

    def forward(self):
        pass

    def render(self, width, height, camera_name=None, depth=False, segmentation=False):
        rgb, z, _ = raycast(self.can.cur[0], self.can.cur[1], height, width, self.can.scene)
        near, far = NEAR * EXTENT, FAR * EXTENT
        return (rgb, ((1.0 - near / z) / (1.0 - near / far)).astype(np.float32)) if depth else rgb


class StubCan(W.Canonicalizer):
    def __init__(self, scene, Ct, Rt, Cn, Rn, sensing=None, ref=None, closed=False, static_pts=None):
        self.scene, self.cur, self.closed = scene, (Ct.copy(), Rt.copy()), closed
        self.sim, self.cam, self.fovy, self.flip = FakeSim(self), "agentview", FOVY, "rot"
        self.Cn, self.Rn, self.sensing, self.ref_image, self.table_z = Cn, Rn, sensing, ref, TABLE_Z
        self.static_pts, self.C_est, self.R_est = static_pts, None, None

    def _save(self):
        return (self.cur[0].copy(), self.cur[1].copy())

    def _restore(self, s):
        self.cur = s

    def _set_camera(self, C, R):
        self.cur = (np.asarray(C, float).copy(), np.asarray(R, float).copy())

    def _kind(self, res, kinds):
        return np.isin(raycast(self.cur[0], self.cur[1], res, res, self.scene)[2], kinds).astype(np.uint8)

    def _robot_mask(self, res):
        return self._kind(res, ["robot", "grip"])

    def _gripper_mask(self, res):
        return self._kind(res, ["grip"])

    def _static_mask(self, res):
        return self._kind(res, ["static", "table"])

    def grip_closed(self, frac=0.75):
        return self.closed


def sensing(mode):
    s = object.__new__(W.LearnedSensing)
    s.depth_mode = mode
    return s


def cmask(img, col):
    return (img == np.array(col, np.uint8)).all(-1)


def centroid(m):
    r, c = np.nonzero(m)
    return np.array([r.mean(), c.mean()]) if len(r) else np.array([np.nan, np.nan])


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


Cn = np.array([0.6586, 0.0, 1.6103])
Rn = W.quat_wxyz_to_mat([0.6380, 0.3048, 0.3048, 0.6380])
pivot = W.optical_pivot(Cn, Rn, TABLE_Z)
SHIFTS = ((10, 0, 1.0), (20, 0, 1.0), (30, 0, 1.2), (-30, 10, 1.2), (45, -15, 1.45))
scene, train_scene = boxes(True), boxes(False)            # deployment: object held and lifted; training: it rests
ref_can = StubCan(train_scene, Cn, Rn, Cn, Rn)
cloud = ref_can.training_static_cloud()
ref = ref_can._to_obs(raycast(Cn, Rn, 256, 256, [])[0])
print(f"training static cloud: {len(cloud)} points, height {cloud[:, 2].min():.2f} to {cloud[:, 2].max():.2f} m")
assert cloud[:, 2].max() > 1.10 and len(cloud) > 10000

rows = []
for az, el, sc in SHIFTS:
    C, R = W.orbit(Cn, Rn, pivot, az, el, sc)
    shift = W.pose_error(C, R, Cn, Rn)
    obs = {"agentview_image": ref_can._to_obs(raycast(C, R, 256, 256, scene)[0])}
    gt = ref_can._to_obs(raycast(Cn, Rn, 256, 256, scene)[0])
    res = {}
    for mode in ("sim", "plane", "hybrid"):
        can = StubCan(scene, C, R, Cn, Rn, sensing=None if mode == "sim" else sensing(mode), ref=ref, closed=True,
                      static_pts=cloud)
        img, _ = can.canonical_image(obs, C, R)
        rel_gt = centroid(cmask(gt, HELD)) - centroid(cmask(gt, GRIP))
        rel = centroid(cmask(img, HELD)) - centroid(cmask(img, GRIP))
        res[mode] = dict(held=float(np.linalg.norm(rel - rel_gt)), held_iou=iou(cmask(img, HELD), cmask(gt, HELD)),
                         tall=float(np.linalg.norm(centroid(cmask(img, TALL)) - centroid(cmask(gt, TALL)))),
                         robot_iou=iou(cmask(img, ROBOT) | cmask(img, GRIP), cmask(gt, ROBOT) | cmask(gt, GRIP)),
                         table=float((np.abs(img.astype(int) - gt.astype(int)).max(-1) <= 8)[np.array(raycast(Cn, Rn, 256, 256, scene)[2] == "table")[::-1, ::-1]].mean()))
    rows.append((shift, res))
    print(f"shift {shift[0]:5.1f} cm / {shift[1]:4.1f} deg | held-object offset error px: sim {res['sim']['held']:5.1f} "
          f"plane {res['plane']['held']:5.1f} hybrid {res['hybrid']['held']:5.1f} | held IoU: sim {res['sim']['held_iou']:.2f} "
          f"plane {res['plane']['held_iou']:.2f} hybrid {res['hybrid']['held_iou']:.2f} | tall fixture px: sim "
          f"{res['sim']['tall']:4.1f} plane {res['plane']['tall']:5.1f} hybrid {res['hybrid']['tall']:4.1f} | table px kept: "
          f"plane {res['plane']['table']:.3f} hybrid {res['hybrid']['table']:.3f}")

# T1 the defect is visible to this test
assert max(r["plane"]["held"] for _, r in rows) > 20, "plane depth should tear the held object off the gripper"
assert max(r["plane"]["tall"] for _, r in rows) > 8, "plane depth should displace a tall fixture"
# T2 held object travels with the hand
for shift, r in rows:
    assert r["hybrid"]["held"] <= max(4.0, r["sim"]["held"] + 3.0), (shift, r["hybrid"]["held"])
    assert r["hybrid"]["held"] < 0.35 * max(r["plane"]["held"], 1e-6) or r["plane"]["held"] < 6, (shift, r)
# T3 tall static fixture lands where simulator depth puts it
for shift, r in rows:               # within the range where calibration is exact; beyond it, better than plane
    if shift[0] < 65:
        assert r["hybrid"]["tall"] <= r["sim"]["tall"] + 3.0, (shift, r["hybrid"]["tall"], r["sim"]["tall"])
    else:
        assert r["hybrid"]["tall"] < r["plane"]["tall"], (shift, r["hybrid"]["tall"], r["plane"]["tall"])
# T4 nothing that was right is broken
for shift, r in rows:
    assert r["hybrid"]["robot_iou"] >= r["plane"]["robot_iou"] - 0.02, (shift, r)
    assert r["hybrid"]["table"] >= r["plane"]["table"] - 0.07, (shift, r)      # near silhouettes fatten by 2 px
print("T1 to T4 pass: hybrid depth keeps the held object on the gripper and the fixture in place, table and robot unchanged")

# gripper OPEN: hybrid must not lift anything (no held-object rule), only the static cloud differs from plane
C, R = W.orbit(Cn, Rn, pivot, 20, 0, 1.0)
obs = {"agentview_image": ref_can._to_obs(raycast(C, R, 256, 256, scene)[0])}
a, _ = StubCan(scene, C, R, Cn, Rn, sensing=sensing("hybrid"), ref=ref, closed=False, static_pts=None).canonical_image(obs, C, R)
b, _ = StubCan(scene, C, R, Cn, Rn, sensing=sensing("plane"), ref=ref).canonical_image(obs, C, R)
assert (a == b).all(), "hybrid with an open gripper and no static cloud must equal plane"
print("open gripper, no cloud: hybrid == plane, bit for bit")

# T5 what a learned mask does to the silhouette fit (scale = camera distance). Measured facts, kept as a regression:
# a hallucinated blob or a missing link barely moves the fit, a UNIFORM 1 px dilation at 96 px moves it by about 6 %
def arm(sc):
    m = np.zeros((192, 192), np.uint8)
    for p0, p1 in (((50, 60), (70, 160)), ((70, 60), (130, 80)), ((120, 80), (135, 110))):
        q0, q1 = ((np.array(p0) - 96) * sc + 96).round().astype(int), ((np.array(p1) - 96) * sc + 96).round().astype(int)
        cv2.rectangle(m, tuple(int(v) for v in q0), tuple(int(v) for v in q1), 1, -1)
    return cv2.resize(m, (96, 96), interpolation=cv2.INTER_NEAREST)


true = arm(1.0)
blob, miss = true.copy(), true.copy()
cv2.rectangle(blob, (65, 28), (80, 44), 1, -1)
miss[40:55, :] = 0
fit = {}
for name, om in (("clean", true), ("blob", blob), ("missing link", miss), ("dilated 1 px", cv2.dilate(true, np.ones((3, 3), np.uint8)))):
    dt = W.edge_dt(om)
    vals = {round(float(sc), 2): W.chamfer(om, arm(sc), dt_obs=dt) + 4.0 * abs(np.log((arm(sc).sum() + 1) / (om.sum() + 1)))
            for sc in np.arange(0.7, 1.41, 0.02)}
    fit[name] = min(vals, key=vals.get)
print("T5 best scale under mask errors:", fit)
assert fit["clean"] == 1.0 and abs(fit["blob"] - 1.0) <= 0.02 and abs(fit["missing link"] - 1.0) <= 0.05
assert fit["dilated 1 px"] >= 1.04, "a uniform boundary bias is what moves the camera; watch mask_area_ratio in the rows"
assert W.chamfer(true, true) == W.chamfer(true, true, trim=0.1) == 0.0

# T6 equal shift magnitudes do not imply equal poses
Ca, Ra = W.orbit(Cn, Rn, pivot, 25, 0, 1.0)
Cb, Rb = W.orbit(Cn, Rn, pivot, -25, 0, 1.0)
sa, sb = W.pose_error(Ca, Ra, Cn, Rn), W.pose_error(Cb, Rb, Cn, Rn)
mutual = W.pose_error(Ca, Ra, Cb, Rb)
print(f"T6 mirrored cameras: shift magnitudes {sa[0]:.1f} vs {sb[0]:.1f} cm agree, mutual pose error {mutual[0]:.1f} cm / {mutual[1]:.1f} deg")
assert abs(sa[0] - sb[0]) < 0.5 and mutual[0] > 30
print("all off-plane tests pass")
