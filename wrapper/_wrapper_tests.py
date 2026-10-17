"""Pure-math checks for wrapper_libero (no simulator)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import wrapper_libero as W

rng = np.random.default_rng(0)
for _ in range(50):
    R = W.rotvec_to_mat(rng.normal(size=3))
    assert np.allclose(W.quat_wxyz_to_mat(W.mat_to_quat_wxyz(R)), R, atol=1e-6)
    r = rng.normal(size=3) * 0.5
    assert np.allclose(W.mat_to_rotvec(W.rotvec_to_mat(r)), r, atol=1e-6)

# nominal LIBERO-Plus agentview (wxyz) and an orbit that must be identity at zero
C0 = np.array([0.6586131746834771, 0.0, 1.6103500240372423])
R0 = W.quat_wxyz_to_mat([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])
piv = W.optical_pivot(C0, R0)
C, R = W.orbit(C0, R0, piv, 0, 0, 1.0)
assert np.allclose(C, C0) and np.allclose(R, R0)
C, R = W.orbit(C0, R0, piv, 30, 10, 1.5)
assert abs(np.linalg.norm(C - piv) - 1.5 * np.linalg.norm(C0 - piv)) < 1e-9     # distance scaled
d_old, d_new = -R0[:, 2], -R[:, 2]
to_p_old, to_p_new = (piv - C0) / np.linalg.norm(piv - C0), (piv - C) / np.linalg.norm(piv - C)
assert np.dot(d_old, to_p_old) > 0.999 and np.dot(d_new, to_p_new) > 0.999     # still looking at the pivot
assert abs(W.pose_error(C, R, C0, R0)[1] - np.rad2deg(np.linalg.norm(W.mat_to_rotvec(R @ R0.T)))) < 1e-9


def render_plane(C, R, f, H, W_, plane_z=0.8):
    """Ray-cast a textured plane z=plane_z from camera (C, R); raw buffer convention (row 0 = bottom)."""
    rr, cc = np.meshgrid(np.arange(H), np.arange(W_), indexing="ij")
    dirs_c = np.stack([(cc + 0.5 - W_ / 2) / f, (rr + 0.5 - H / 2) / f, -np.ones_like(rr, float)], -1)
    dirs_w = dirs_c @ R.T
    t = (plane_z - C[2]) / dirs_w[..., 2]
    P = C + t[..., None] * dirs_w
    z = t                                            # depth along the optical axis (dirs_c has z = -1)
    # smooth texture that encodes world x, y (1 gray level = 5 mm), so the error measures geometry, not aliasing
    tex = np.stack([np.clip(128 + P[..., 0] * 200, 0, 255), np.clip(128 + P[..., 1] * 200, 0, 255),
                    np.full(P.shape[:2], 128.0)], -1)
    return tex.astype(np.uint8), z


H = W_ = 128
f = 0.5 * H / np.tan(45 * np.pi / 360)
img0, z0 = render_plane(C0, R0, f, H, W_)
same, cov = W.reproject(img0, z0, f, C0, R0, C0, R0, splat=1, inpaint=False)
assert cov > 0.999 and np.abs(same.astype(int) - img0).mean() < 1e-9, "identity reprojection must be exact"
C1, R1 = W.orbit(C0, R0, piv, 8, 4, 1.1)
img1, z1 = render_plane(C1, R1, f, H, W_)
back, cov = W.reproject(img1, z1, f, C1, R1, C0, R0)
inside = (img0[..., 0] > 0) & (img0[..., 0] < 255) & (img0[..., 1] > 0) & (img0[..., 1] < 255)
err = np.abs(back.astype(int) - img0.astype(int))[inside].mean()
naive = np.abs(img1.astype(int) - img0.astype(int))[inside].mean()
print(f"plane reprojection: mean abs error {err:.2f} vs unwarped {naive:.2f} gray levels (coverage {cov:.2f})")
assert err < 0.2 * naive and err < 3.0, "reprojection must undo a known camera move"

m = np.zeros((96, 96), np.uint8); m[30:60, 40:70] = 1
assert W.chamfer(m, m) == 0.0
sh = np.roll(m, 3, axis=1)
assert 1.0 < W.chamfer(m, sh) < 3.5
# ICP recovers a camera pose from camera-frame points of a known world-frame surface
g = np.random.default_rng(3)
Q = np.concatenate([g.uniform([-0.3, -0.3, 0.8], [0.3, 0.3, 0.8], (3000, 3)),
                    g.uniform([0.0, -0.05, 0.8], [0.1, 0.05, 1.3], (3000, 3)),
                    g.uniform([-0.2, 0.1, 0.8], [-0.1, 0.2, 1.0], (2000, 3))])
Ct, Rt = W.orbit(C0, R0, piv, 15, 8, 1.1)
P = (Q[g.choice(len(Q), 2500, replace=False)] - Ct) @ Rt                  # world -> camera frame
# ICP is a local refiner (measured: it converges from seeds near the truth, stalls from far ones), so it is tested
# as one: exact from the truth, and converging from a seed ~20 cm / 9 deg away at the right distance
Re, Ce, res = W.icp(P, Q, Rt, Ct)
assert W.pose_error(Ce, Re, Ct, Rt)[0] < 1e-6 and res < 1e-9, "ICP must be a fixed point at the true pose"
Cs, Rs = W.orbit(C0, R0, piv, 20, 0, 1.1)
Re, Ce, res = W.icp(P, Q, Rs, Cs)
e_cm, e_deg = W.pose_error(Ce, Re, Ct, Rt)
print(f"ICP local refine: seed {W.pose_error(Cs, Rs, Ct, Rt)[0]:.1f} cm off, recovered to {e_cm:.2f} cm / {e_deg:.2f} deg")
assert e_cm < 1.0 and e_deg < 1.0, "ICP must refine a nearby seed to the true pose"
z = np.full((8, 8), 2.0); mk = np.zeros((8, 8), bool); mk[4, 4] = True
pt = W.backproject(z, mk, 10.0)[0]
assert np.allclose(pt, [0.5 * 2 / 10, 0.5 * 2 / 10, -2.0])
assert W.scene_of("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_10") == "KITCHEN_SCENE3"
assert W.scene_of("LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_3") == "LIVING_ROOM_SCENE2"
assert W.scene_of("STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy") == "STUDY_SCENE1"
print("ALL WRAPPER TESTS PASS")

# plane depth: MuJoCo cameras look along their own -z, so the identity camera looks straight down. 1 m above the
# table it sees the plane at depth 1 m at every pixel; tilted 45 deg the centre ray meets it at sqrt(2) m; a camera
# flipped to look up never meets it and every pixel gets `far`
import wrapper_libero as W
R_down = np.eye(3)
z = W.plane_depth(8, 8, 4.0, np.array([0.0, 0.0, 1.8]), R_down, 0.8)
assert np.allclose(z, 1.0, atol=1e-6), z
R_tilt = W.rotvec_to_mat(np.array([np.deg2rad(45.0), 0.0, 0.0])) @ R_down
z = W.plane_depth(9, 9, 4.5, np.array([0.0, 0.0, 1.8]), R_tilt, 0.8)
assert abs(z[4, 4] - np.sqrt(2.0)) < 0.05, z[4, 4]
z = W.plane_depth(4, 4, 2.0, np.array([0.0, 0.0, 1.8]), np.diag([1.0, -1.0, -1.0]), 0.8)   # above the table, looking up
assert np.all(z == 10.0), z
print("plane depth ok")

# mask cleanup: speckle removed, holes filled, main component kept
m = np.zeros((40, 40), np.uint8); m[10:30, 10:30] = 1; m[18:22, 18:22] = 0; m[2, 2] = 1; m[35:37, 3:5] = 1
c = W.clean_mask(m)
assert c[20, 20] == 1 and c[2, 2] == 0 and c[35, 3] == 0 and c[10:30, 10:30].all() and c.sum() == 400, c.sum()
print("mask cleanup ok")

# NumPy 2 alias patch: the real LIBERO-plus line is `np.float_`, and a pattern written for the bare `np.float`
# leaves it untouched, which cost two runs. Concrete dtypes must survive.
import robosuite_np2 as NP2
assert NP2.patch_aliases("maparray = np.empty((n, n), dtype=np.float_)").endswith("dtype=np.float64)")
assert NP2.patch_aliases("x = np.float)") == "x = np.float64)"
for keep in ["np.float64", "np.float32", "np.floating", "np.int8", "np.bool_", "np.str_", "np.object_"]:
    assert NP2.patch_aliases(keep) == keep, keep
assert NP2.remaining("dtype=np.float_") == ["np.float_"] and NP2.remaining("dtype=np.float64") == []
print("numpy 2 alias patch ok")

# POST HOC pose-source ablation (P47): fiducial PnP recovers a camera from noise-free corners and stays within about a
# centimetre with 0.5 px corner noise; the EasyHeC-style dense refinement pulls a 3.6 cm / 3 deg perturbed pose back
# toward truth on a mock renderer (a projected point cloud as the robot silhouette, same pinhole as reproject).
import cv2
import pose_sources as PSRC
Ct = C0 + np.array([0.12, -0.08, 0.05]); Rt = W.rotvec_to_mat(np.array([0.05, -0.1, 0.08])) @ R0
fpx = 0.5 * 256 / np.tan(np.deg2rad(45.0) / 2)
corners = PSRC.marker_corners(W.optical_pivot(C0, R0), 0.8)
Cp, Rp, inf = PSRC.fiducial_pose(Ct, Rt, corners, fpx, 256, 256, sigma=0.0)
assert inf["fid_markers_visible"] >= 2 and max(W.pose_error(Cp, Rp, Ct, Rt)) < 1e-3, (inf, W.pose_error(Cp, Rp, Ct, Rt))
Cp, Rp, inf = PSRC.fiducial_pose(Ct, Rt, corners, fpx, 256, 256, sigma=0.5)
assert W.pose_error(Cp, Rp, Ct, Rt)[0] < 2.0 and W.pose_error(Cp, Rp, Ct, Rt)[1] < 1.0, W.pose_error(Cp, Rp, Ct, Rt)
uv, z = PSRC.project_cv(corners.reshape(-1, 3), Ct, Rt, fpx, 256, 256)
pc = (corners.reshape(-1, 3) - Ct) @ Rt
row_raw = fpx * pc[:, 1] / -pc[:, 2] + 128 - 0.5                  # the raw-buffer row that reproject uses
assert np.allclose(uv[:, 1], 255 - row_raw), "OpenCV v must be the raw buffer row flipped"
print("fiducial pose ok: 0.5 px noise ->", np.round(W.pose_error(Cp, Rp, Ct, Rt), 2))


class _MockCan:
    """Just enough of Canonicalizer for pose_sources.refine: a rigid point cloud rendered as a 96 px mask."""
    def __init__(self, pts, Cn, Rn):
        self.pts, self.Cn, self.Rn, self.mres = pts, Cn, Rn, 96
        self.arm_idx, self.grip_idx = [0], []
        self.C, self.R = Cn, Rn
        self.sim = type("S", (), {"data": type("D", (), {"qpos": np.zeros(1)})(), "forward": lambda self: None})()

    def _save(self):
        return (self.C.copy(), self.R.copy())

    def _restore(self, s):
        self.C, self.R = s

    def _set_camera(self, C, R):
        self.C, self.R = np.asarray(C, float), np.asarray(R, float)

    def _robot_mask(self, res):
        f = 0.5 * res / np.tan(np.deg2rad(45.0) / 2)
        uv, z = PSRC.project_cv(self.pts, self.C, self.R, f, res, res)
        m = np.zeros((res, res), np.uint8)
        k = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < res - 0.5) & (uv[:, 1] >= 0) & (uv[:, 1] < res - 0.5)
        m[np.round(uv[k, 1]).astype(int), np.round(uv[k, 0]).astype(int)] = 1
        return cv2.dilate(m, np.ones((5, 5), np.uint8))


g = np.random.default_rng(1)
piv = W.optical_pivot(C0, R0)
pts = np.concatenate([piv + g.normal(size=(1500, 3)) * [0.03, 0.03, 0.15] + [0, 0, 0.2],
                      piv + g.normal(size=(1000, 3)) * [0.15, 0.03, 0.03] + [0.05, 0, 0.3]])
mc = _MockCan(pts, C0, R0)
mc._set_camera(Ct, Rt)
recs = [(np.zeros(1), None, mc._robot_mask(96), None, None)]
Ci, Ri = Ct + np.array([0.03, 0.0, -0.02]), W.rotvec_to_mat(np.deg2rad([2.0, -2.0, 1.0])) @ Rt
Cr, Rr, o = PSRC.refine(mc, recs, Ci, Ri)
e0, e1 = W.pose_error(Ci, Ri, Ct, Rt), W.pose_error(Cr, Rr, Ct, Rt)
assert o["refine_loss_final"] < 0.5 * o["refine_loss_init"] and e1[0] < e0[0] and e1[1] < e0[1], (e0, e1, o)
assert np.allclose(mc.C, Ct) and np.allclose(mc.R, Rt), "refine must restore the camera"
print("dense refinement ok: %.1f cm %.1f deg -> %.1f cm %.1f deg" % (*e0, *e1))
