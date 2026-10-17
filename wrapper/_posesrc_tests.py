"""Local tests for wrapper/scripts/droid_posesrc.py (no GPU, no SAM 2, no CtRNet-X weights).
uv run --no-project --with numpy --with scipy --with opencv-python-headless --with mujoco python wrapper/_posesrc_tests.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import droid_posesrc as P  # noqa: E402
from droid_selfcal import Franka, euler_xyz_to_mat, look_at, pose_err  # noqa: E402

MEN = os.path.join(HERE, "..", "vendor", "mujoco_menagerie")


def test_ctr_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        C = rng.normal(size=3)
        R = euler_xyz_to_mat(rng.uniform(-3, 3, 3))
        C2, R2 = P.ctr_to_cam_pose(P.cam_pose_to_ctr(C, R))
        e = pose_err(C, R, C2, R2)
        assert e[0] < 1e-6 and e[1] < 1e-4, e
    print("ctrnetx pose round trip ok")


def test_prompts():
    m = np.zeros((P.H, P.W), np.uint8); m[60:120, 100:200] = 1
    box, pts, lab = P.prompts_from_render(m, np.random.default_rng(0))
    assert box[0] < 100 and box[2] > 199 and lab.sum() == 5 and (lab == 0).sum() == 4
    for (x, y), l in zip(pts, lab):
        assert m[int(y), int(x)] == l, (x, y, l)
    print("prompts ok: positives inside, negatives outside the render")


class FakeSAM:
    """Returns the TRUE-pose render (score 0.9) and a decoy blob (score 0.5): the loop must pick by score."""
    def __init__(self, rob, d, C, R, fy):
        self.masks = {i: rob.mask(q, C, R, fy) for i, q in enumerate(d["q"])}
        self.i = 0

    def segment(self, img, box, pts, lab):
        m = self.masks[self.i % len(self.masks)]; self.i += 1
        decoy = np.zeros_like(m); decoy[:40, :60] = 1
        return np.stack([m, decoy]), np.array([0.9, 0.5])


def test_sam2_loop_converges():
    rob = Franka(MEN, W=P.W, H=P.H, hand=True)
    rng = np.random.default_rng(0)
    q0 = np.array([0, -0.6, 0, -2.2, 0, 1.6, 0.8])
    qs = np.stack([q0 + rng.normal(0, 0.25, 7) for _ in range(6)])
    target = np.array([0.45, 0.0, 0.25])
    Cg = target + np.array([0.15, -0.6, 0.35]); Rg = look_at(Cg, target)
    fy = 130.0
    d = {"q": qs, "img": [None] * len(qs)}
    seg = FakeSAM(rob, d, Cg, Rg, fy)
    C0 = Cg + np.array([0.06, -0.05, 0.03]); R0 = euler_xyz_to_mat(np.deg2rad([3, -4, 2])) @ Rg
    e0 = pose_err(C0, R0, Cg, Rg)
    K = np.array([[fy, 0, P.W / 2], [0, fy, P.H / 2], [0, 0, 1.0]])
    C, R, hist = P.sam2_loop(rob, seg, d, K, C0, R0, rng, rounds=3)
    e = pose_err(C, R, Cg, Rg)
    print(f"sam2 loop with correct masks: {e0[0]:.1f} cm / {e0[1]:.1f} deg -> {e[0]:.1f} cm / {e[1]:.1f} deg, rounds {len(hist)}")
    assert e[0] < e0[0] / 2, (e0, e)


def test_panda_fk_matches_mujoco():
    """The URDF forward kinematics behind the CtRNet-X keypoints must agree with MuJoCo menagerie's Franka."""
    import mujoco
    urdf = os.path.join(HERE, "..", "vendor", "CtRNet-X", "urdfs", "Panda", "panda.urdf")
    if not os.path.exists(urdf):
        print("panda fk test skipped (no CtRNet-X checkout)"); return
    kp = P.PandaKeypoints(urdf)
    m = mujoco.MjModel.from_xml_path(os.path.join(MEN, "franka_emika_panda", "panda_nohand.xml"))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, 7)
        d.qpos[:7] = q; mujoco.mj_forward(m, d)
        F = kp.link_frames(q)
        for k in (6, 7):
            worst = max(worst, np.linalg.norm(F[k][:3, 3] - d.body(f"link{k}").xpos))
    print(f"panda FK vs mujoco: worst link6/link7 origin gap {worst * 1000:.2f} mm")
    assert worst < 1e-3, worst


if __name__ == "__main__":
    test_panda_fk_matches_mujoco()
    test_ctr_roundtrip()
    test_prompts()
    test_sam2_loop_converges()
    print("posesrc tests: all passed")
