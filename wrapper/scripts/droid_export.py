"""Export DROID-100 (real Franka data, public GCS sample) to per-episode npz: both exterior cameras, wrist
camera, joint and gripper positions, recorded joint-velocity and gripper actions, and the instruction.
Runs in Colab's own Python (TensorFlow + tensorflow_datasets preinstalled)."""
import argparse
import glob
import os

import numpy as np


def find_builder_dir(local_root):
    hits = glob.glob(os.path.join(local_root, "**", "dataset_info.json"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    import tensorflow as tf
    hits = tf.io.gfile.glob("gs://gresearch/robotics/droid_100/*/dataset_info.json")
    assert hits, "droid_100 not found locally or on GCS"
    return os.path.dirname(hits[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_root", default="/content/droid_100")
    ap.add_argument("--out", default="/content/droid_npz")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    d = find_builder_dir(a.local_root)
    print("reading", d)
    ds = tfds.builder_from_directory(d).as_dataset(split="train")
    os.makedirs(a.out, exist_ok=True)
    kept = 0
    for i, ep in enumerate(ds):
        if kept >= a.episodes:
            break
        steps = list(ep["steps"].as_numpy_iterator())
        if i == 0:
            print("observation keys:", sorted(steps[0]["observation"].keys()))
            print("action_dict keys:", sorted(steps[0]["action_dict"].keys()))
        lang = steps[0]["language_instruction"]
        lang = lang.decode() if isinstance(lang, bytes) else str(lang)
        if not lang.strip() or len(steps) < 60:
            continue
        sel = steps[::a.stride]
        o = lambda k: np.stack([s["observation"][k] for s in sel])
        np.savez_compressed(
            os.path.join(a.out, f"ep{kept:03d}.npz"),
            ext1=o("exterior_image_1_left"), ext2=o("exterior_image_2_left"), wrist=o("wrist_image_left"),
            q=o("joint_position").astype(np.float32), g=o("gripper_position").astype(np.float32),
            jv=np.stack([s["action_dict"]["joint_velocity"] for s in sel]).astype(np.float32),
            ga=np.stack([s["action_dict"]["gripper_position"] for s in sel]).astype(np.float32),
            lang=np.array(lang))
        kept += 1
        print(f"episode {i} -> ep{kept-1:03d}: {len(steps)} steps, {len(sel)} kept | {lang}", flush=True)
    print("exported", kept, "episodes to", a.out)


if __name__ == "__main__":
    main()
