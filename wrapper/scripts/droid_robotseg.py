"""BASELINE for the DROID self-calibration: robot masks from RobotSeg (Mei et al., CVPR 2026, github.com/showlab/RobotSeg),
a SAM 2 model trained to segment robots, in place of the masks our VLM picks out of a SAM partition. Same cameras, same
frames, same calibration search afterwards, so the only thing that changes is where the mask comes from.

Each camera file (<in_dir>/*.npz: img (T, H, W, 3) RGB, the T frames of one episode in time order) is written out as a
short JPEG video, RobotSeg runs in its AUTO mode (no click, no box: `add_new_robot` on frame 0, category `robot`), and the
file is copied to --out_dir with `mask` replaced (the old one is kept as `mask_owl`).

Run with the working directory at <RobotSeg>/test, because the repository resolves its config and checkpoint relative to it:
    cd /content/RobotSeg/test && python /content/scripts/droid_robotseg.py --in_dir ... --out_dir ... [--limit 2]
"""
import argparse
import glob
import os
import shutil
import tempfile

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N cameras only (0 = all): use 2 for the smoke test")
    ap.add_argument("--category", default="robot", choices=["robot", "arm", "gripper"])
    ap.add_argument("--yaml", default="robotseg-infer")
    ap.add_argument("--ckpt", default="../checkpoints/robotseg.pt")
    a = ap.parse_args()
    assert os.path.abspath(a.in_dir) != os.path.abspath(a.out_dir), "--out_dir must differ from --in_dir: the old masks are kept"
    assert os.path.exists(a.ckpt), f"RobotSeg checkpoint not found at {os.path.abspath(a.ckpt)}"
    os.makedirs(a.out_dir, exist_ok=True)

    import cv2
    import torch
    from robotseg.build_robotseg import build_robotseg_video_predictor
    predictor = build_robotseg_video_predictor(f"../robotseg/configs/{a.yaml}", a.ckpt)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    files = sorted(glob.glob(os.path.join(a.in_dir, "*.npz")))
    files = files[:a.limit] if a.limit else files
    for p in files:
        out = os.path.join(a.out_dir, os.path.basename(p))
        if os.path.exists(out):
            continue
        with np.load(p, allow_pickle=True) as z:
            d = dict(z)
        imgs = d["img"]
        tmp = tempfile.mkdtemp()
        try:
            for t, im in enumerate(imgs):
                cv2.imwrite(os.path.join(tmp, f"{t:05d}.jpg"), np.ascontiguousarray(im[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 97])
            masks = np.zeros(imgs.shape[:3], np.uint8)
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                state = predictor.init_state(video_path=tmp, async_loading_frames=False, offload_video_to_cpu=False,
                                             offload_state_to_cpu=False)
                predictor.add_new_robot(inference_state=state, frame_idx=0, obj_id=0, robot=a.category)
                for t, ids, logits in predictor.propagate_in_video(inference_state=state, robot=a.category):
                    for i, oid in enumerate(ids):
                        if oid == 0:
                            m = np.squeeze((logits[i] > 0.0).detach().cpu().numpy()).astype(np.uint8)
                            if m.shape != imgs.shape[1:3]:
                                m = cv2.resize(m, (imgs.shape[2], imgs.shape[1]), interpolation=cv2.INTER_NEAREST)
                            masks[t] = m
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if "mask" in d:
            d["mask_owl"] = d["mask"]
        d["mask"] = masks
        np.savez_compressed(out, **d)
        print(f"{os.path.basename(p)}: robot pixel fraction {masks.mean():.3f}"
              + (f" (zero-shot detector masks {d['mask_owl'].mean():.3f})" if "mask_owl" in d else "") + f", frames with a mask {int((masks.reshape(len(masks), -1).sum(1) > 0).sum())}/{len(masks)}", flush=True)


if __name__ == "__main__":
    main()
