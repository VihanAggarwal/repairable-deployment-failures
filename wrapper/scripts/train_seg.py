"""Train the robot segmenter used by the learned-sensing wrapper (replaces the renderer's perfect robot mask).

Data: collect_canon.py shards (held-out LIBERO-Plus tasks, many camera poses and arm configurations), which store the
observed frame `x` and the robot mask `r` in observation orientation. As with any deployed robot segmenter
(e.g. CtRNet), labels come from rendering the robot's known geometry in simulation at training time; nothing about
the evaluation tasks is used. Reports IoU on held-out tasks.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon_net import block  # noqa: E402


class SegNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        ch = [width, 2 * width, 4 * width, 8 * width]
        self.enc = nn.ModuleList()
        prev = 3
        for c in ch:
            self.enc.append(block(prev, c))
            prev = c
        self.dec = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.dec.append(block(prev + c, c))
            prev = c
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, x):
        skips, h = [], x
        for i, b in enumerate(self.enc):
            h = b(h)
            if i < len(self.enc) - 1:
                skips.append(h)
                h = F.max_pool2d(h, 2)
        for b, s in zip(self.dec, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = b(torch.cat([h, s], 1))
        return self.head(h)


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    net = SegNet(**ck.get("net_args", {}))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def predict(net, img_uint8, device="cuda", thr=0.5):
    x = torch.from_numpy(np.ascontiguousarray(img_uint8)).to(device).permute(2, 0, 1)[None].float() / 255.0
    return (torch.sigmoid(net(x))[0, 0] > thr).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    X, Rm, T = [], [], []
    for p in sorted(glob.glob(os.path.join(a.data_dir, "task_*.npz"))):
        d = np.load(p)
        if "r" not in d:
            continue
        X.append(d["x"]); Rm.append(d["r"]); T += [json.loads(str(d["meta"]))["task_id"]] * len(d["x"])
    assert X, "no shards with robot masks (`r`): re-run collect_canon.py (it now stores them)"
    X, Rm, T = np.concatenate(X), np.concatenate(Rm), np.array(T)
    rng = np.random.default_rng(0)
    tasks = np.unique(T)
    val = set(rng.permutation(tasks)[:max(1, len(tasks) // 7)].tolist())
    va = np.array([t in val for t in T])
    print(f"{len(X)} frames, {len(tasks)} tasks, val tasks {len(val)}, robot pixel fraction {Rm.mean():.3f}", flush=True)
    net = SegNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    pos_w = torch.tensor([(1 - Rm.mean()) / max(Rm.mean(), 1e-3)], device=dev).clamp(max=10.0)
    tr = np.where(~va)[0]
    assert len(tr) >= a.bs, f"only {len(tr)} training frames for batch size {a.bs}: collect more shards"
    best, loss = -1.0, torch.tensor(float("nan"))
    for ep in range(a.epochs):
        net.train()
        rng.shuffle(tr)
        for s in range(0, len(tr) - a.bs + 1, a.bs):
            b = tr[s:s + a.bs]
            x = torch.from_numpy(X[b]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            y = torch.from_numpy(Rm[b]).to(dev).float()[:, None]
            if rng.random() < 0.5:                                   # photometric augmentation: gain and noise
                x = (x * torch.empty(len(b), 3, 1, 1, device=dev).uniform_(0.6, 1.4)
                     + torch.randn_like(x) * float(rng.uniform(0, 0.08))).clamp(0, 1)
            loss = F.binary_cross_entropy_with_logits(net(x), y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        inter = union = 0.0
        with torch.no_grad():
            for i in np.where(va)[0]:
                x = torch.from_numpy(X[i:i + 1]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                pr = torch.sigmoid(net(x))[0, 0].cpu().numpy() > 0.5
                gt = Rm[i] > 0
                inter += (pr & gt).sum(); union += (pr | gt).sum()
        iou = inter / max(union, 1)
        print(f"epoch {ep + 1:3d} loss {float(loss):.4f} val IoU {iou:.3f}", flush=True)
        if iou > best:
            best = iou
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save(dict(state_dict=net.state_dict(), net_args={}, val_iou=float(iou), epoch=ep + 1), a.out)
    print(f"best val IoU {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
