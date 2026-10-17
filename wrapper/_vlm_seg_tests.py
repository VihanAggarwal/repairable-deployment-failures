"""VLM-selected robot masks (droid_vlm_seg.py): synthetic tests, no SAM, no API, no GPU.

    uv run --no-project --with numpy --with opencv-python-headless --with pillow python wrapper/_vlm_seg_tests.py

  V1  overlapping masks become a complete partition; every pixel lands in the smallest mask that holds it
  V2  the numbered view has the VLM's size and a number drawn inside every region
  V3  ask(): ids the frame does not have are dropped, robot wins over mixed, a refusal gives an error row
  V4  select rows -> assemble: robot regions in, mixed regions out unless a zoomed crop answered for them; the old
      mask is kept as mask_owl; an unanswered frame gets an empty mask
"""
import json
import os
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import droid_vlm_seg as V  # noqa: E402

H, W = 180 * V.UP, 320 * V.UP
yy, xx = np.mgrid[:H, :W]
wall = np.ones((H, W), bool)                                        # the largest mask: everything
arm = (xx > 300) & (xx < 420) & (yy > 60) & (yy < 480)               # inside the wall
joint = (xx - 360) ** 2 + (yy - 200) ** 2 < 30 ** 2                   # inside the arm
speck = (xx < 6) & (yy < 6)                                           # too small to number

# V1
lab, lost = V.regions_from_masks([joint, wall, arm, speck], (H, W))
assert lab.max() == 3 and lab[200, 360] == 3 and lab[100, 360] == 2 and lab[10, 600] == 1, np.unique(lab)
assert lab[2, 2] == 0 and 0 < lost < 1e-3, "the speck is not selectable and is reported as lost area"
lab2, _ = V.regions_from_masks([arm], (H, W))
assert set(np.unique(lab2)) == {1, 2}, "what no mask covers becomes a region of its own"
print("V1 pass")

# V2
img = np.full((180, 320, 3), 120, np.uint8)
plain, marked = V.views(img, lab)
assert plain.shape == marked.shape == (720, 1280, 3) and (marked != plain).any()
crop_plain, crop_marked = V.views(img[:, :90], lab[:, :270], inside=lab[:, :270] == 2)
assert crop_plain.shape[0] == V.SEND_W, "a tall crop is scaled by its long side"
print("V2 pass")


# V3
class FakeClient:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.seen = text, stop, None
        self.messages = types.SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.seen = kw
        return types.SimpleNamespace(stop_reason=self.stop, usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
                                     content=[types.SimpleNamespace(type="thinking", thinking=""),
                                              types.SimpleNamespace(type="text", text=self.text)])


a = types.SimpleNamespace(model="claude-opus-5", effort="medium")
c = FakeClient(json.dumps(dict(robot_visible=True, robot_ids=[2, 3, 99], mixed_ids=[1, 3])))
r = V.ask(c, a, img, lab)
assert r["robot_ids"] == [2, 3] and r["mixed_ids"] == [1] and r["unknown_ids"] == [99] and r["tokens"] == dict(input=11, output=7), r
kinds = [b["type"] for b in c.seen["messages"][0]["content"]]
assert kinds == ["text", "image", "text", "image", "text"] and c.seen["output_config"]["effort"] == "medium"
assert c.seen["output_config"]["format"]["schema"]["additionalProperties"] is False
V.ask(c, types.SimpleNamespace(model="claude-haiku-4-5", effort="medium"), img, lab)
assert "effort" not in c.seen["output_config"], "Haiku 4.5 rejects effort"
assert "error" in V.ask(FakeClient("", stop="refusal"), a, img, lab)
print("V3 pass")

# V4
with tempfile.TemporaryDirectory() as tmp:
    a = types.SimpleNamespace(in_dir=os.path.join(tmp, "in"), work_dir=os.path.join(tmp, "work"), out_dir=os.path.join(tmp, "out"),
                              limit=0, mixed=0)
    os.makedirs(a.in_dir), os.makedirs(a.work_dir)
    old = np.zeros((2, 180, 320), np.uint8)
    old[:, :50] = 1
    np.savez_compressed(os.path.join(a.in_dir, "cam.npz"), img=np.stack([img, img]), mask=old, q=np.zeros((2, 7)))
    np.savez_compressed(os.path.join(a.work_dir, "cam.props.npz"), labels=np.stack([lab, lab]))
    sub = np.zeros((40, 40), np.int16)
    sub[:, :20] = 1                                                   # the left half of the crop is robot
    np.savez_compressed(os.path.join(a.work_dir, "cam.refine.npz"), **{"lab_0_1": sub, "box_0_1": np.array([600, 30, 720, 150])})
    with open(os.path.join(a.work_dir, "vlm_select.jsonl"), "w") as f:
        f.write(json.dumps(dict(file="cam.npz", frame=0, region=0, robot_visible=True, robot_ids=[2, 3], mixed_ids=[1])) + "\n")
        f.write(json.dumps(dict(file="cam.npz", frame=0, region=1, robot_visible=True, robot_ids=[1], mixed_ids=[])) + "\n")
        f.write(json.dumps(dict(file="cam.npz", frame=1, region=0, error="refusal")) + "\n")
    V.assemble(a)
    with np.load(os.path.join(a.out_dir, "cam.npz")) as z:              # closed before the folder is removed (Windows)
        d = {k: z[k] for k in z.files}
    m = d["mask"][0]
    assert (d["mask_owl"] == old).all() and d["mask"][1].sum() == 0, "old mask kept; the unanswered frame is empty"
    assert m[100, 120] == 1 and m[100, 20] == 0, "arm in, wall out"
    assert m[30, 205] == 1 and m[30, 235] == 0, "the zoomed crop adds its robot half of the mixed region only"
print("V4 pass")
print("all VLM segmenter tests pass")

# V5: a mixed region is split by the robot's own motion (assemble --mixed 2): inside it, only the pixels that moved
with tempfile.TemporaryDirectory() as tmp:
    a = types.SimpleNamespace(in_dir=os.path.join(tmp, "in"), work_dir=os.path.join(tmp, "work"), out_dir=os.path.join(tmp, "out"),
                              limit=0, mixed=2)
    os.makedirs(a.in_dir), os.makedirs(a.work_dir)
    frames = np.full((4, 180, 320, 3), 120, np.uint8)
    for t in range(4):
        frames[t, 20:60, 200 + 10 * t:230 + 10 * t] = 250            # a bright link moving across the wall (region 1)
    np.savez_compressed(os.path.join(a.in_dir, "cam.npz"), img=frames, mask=np.zeros((4, 180, 320), np.uint8), q=np.zeros((4, 7)))
    np.savez_compressed(os.path.join(a.work_dir, "cam.props.npz"), labels=np.stack([lab] * 4))
    with open(os.path.join(a.work_dir, "vlm_select.jsonl"), "w") as f:
        for t in range(4):
            f.write(json.dumps(dict(file="cam.npz", frame=t, region=0, robot_visible=True, robot_ids=[2, 3], mixed_ids=[1])) + "\n")
    V.assemble(a)
    with np.load(os.path.join(a.out_dir, "cam.npz")) as z:
        m = z["mask"]
    assert m[0, 40, 205] == 1 and m[3, 40, 250] == 1, "the moving link inside the mixed region is added"
    assert m[0, 150, 300] == 0 and m[0, 100, 120] == 1, "the still wall stays out, the chosen regions stay in"
print("V5 pass")
