"""Monocular METRIC depth served from a python whose transformers can read the model (Colab's system python), for a
venv that cannot (OFT pins transformers 4.40). Started by wrapper_libero.MetricDepth; not run by hand.

Protocol on stdin / stdout, one line each: prints `ready <model name>` once, then for every line `<in.npy> <out.npy>`
reads an HxWx3 uint8 image, writes HxW float32 depth in metres at the same size, and prints `ok` (or `error ...`).
"""
import sys

import numpy as np

MODELS = ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "Intel/zoedepth-nyu-kitti")


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    last = None
    for name in MODELS:
        try:
            proc = AutoImageProcessor.from_pretrained(name)
            model = AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval()
            break
        except Exception as e:
            last = e
    else:
        print(f"error no metric depth model loads: {last!r}".replace("\n", " "), flush=True)
        return
    print(f"ready {name}", flush=True)
    for line in sys.stdin:
        try:
            src, dst = line.split()
            img = np.load(src)
            with torch.no_grad():
                pred = model(**proc(images=img, return_tensors="pt").to(dev)).predicted_depth
                pred = F.interpolate(pred[:, None].float(), size=img.shape[:2], mode="bicubic", align_corners=False)[0, 0]
            np.save(dst, pred.cpu().numpy().astype(np.float32))
            print("ok", flush=True)
        except Exception as e:
            print(f"error {e!r}".replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
