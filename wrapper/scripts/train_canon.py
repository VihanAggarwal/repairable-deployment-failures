"""Train the learned canonicalizer (claim 3) on the self-labelled pairs from collect_canon.py.

Targets (--target):
  reproj   the observation reprojected with the wrapper's own camera estimate (self-labelled). The network then
           distills the geometric pipeline into one RGB-only pass; its ceiling is the `full` arm.
  oracle   the true nominal-camera render stored by collect_canon.py (privileged at training time only). Every pixel
           is supervised, so the network can fill what reprojection cannot see; this is the arm that can beat `full`.
Inputs (--input):
  rgb      the raw frame only (learning alone; the 2026-09-16 result)
  reproj   the wrapper's reprojection + hole mask + raw frame (7 channels): measurement plus learning. The network
           only has to complete and repair, which is what the reprojection gets wrong.
Loss: L1 on observed pixels (weight 1) and on holes (weight --hole_w; every pixel counts as observed for oracle), plus
a feature loss in a frozen DINOv2-small on patches that are mostly observed. Split by task, so validation scenes and
camera poses are unseen. Reports PSNR against the chosen target for the network and for the do-nothing baseline.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon_net import CanonNet  # noqa: E402


def net_input(d, mode):
    """uint8 HxWxC network input from a shard: the raw frame, or reprojection + hole mask + raw frame."""
    if mode == "rgb":
        return d["x"]
    hole = (d["m"].astype(np.uint8) * 255)[..., None]
    return np.concatenate([d["y"], hole, d["x"]], -1)


def load_pairs(data_dir, target="reproj", mode="rgb"):
    X, Y, M, T, META = [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(data_dir, "task_*.npz"))):
        d = np.load(p)
        meta = json.loads(str(d["meta"]))
        if target == "oracle":
            if "o" not in d.files:
                sys.exit(f"{p} has no oracle render: re-run collect_canon.py into a fresh --out_dir (shards from "
                         f"before 2026-09-16 carry only the reprojection target)")
            X.append(net_input(d, mode)); Y.append(d["o"]); M.append(np.ones(d["m"].shape, bool))
        else:
            X.append(net_input(d, mode)); Y.append(d["y"]); M.append(d["m"])
        T += [meta["task_id"]] * len(d["x"])
        META.append(meta)
    assert X, f"no task_*.npz in {data_dir}"
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M), np.array(T), META


def to_t(a, dev):
    return torch.from_numpy(a).to(dev).permute(0, 3, 1, 2).float() / 255.0


def masked_psnr(pred, y, m):
    mse = ((pred - y) ** 2 * m).sum() / (m.sum() * 3 + 1e-6)
    return float(10 * torch.log10(1.0 / (mse + 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hole_w", type=float, default=0.1)
    ap.add_argument("--feat_w", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--target", default="reproj", choices=["reproj", "oracle"])
    ap.add_argument("--input", default="rgb", choices=["rgb", "reproj"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    X, Y, M, T, META = load_pairs(a.data_dir, a.target, a.input)
    net_args = dict(in_ch=X.shape[-1], max_flow=0.15 if a.input == "reproj" else 0.6)
    tasks = np.unique(T)
    rng = np.random.default_rng(0)
    val_tasks = set(rng.permutation(tasks)[:max(1, int(a.val_frac * len(tasks)))].tolist())
    va = np.array([t in val_tasks for t in T])
    cat = {m["task_id"]: m["category"] for m in META}
    print(f"target {a.target}, input {a.input} ({X.shape[-1]} ch): {len(X)} pairs from {len(tasks)} tasks; "
          f"train {int((~va).sum())} / val {int(va.sum())} "
          f"(val tasks {len(val_tasks)}); mean observed fraction {M.mean():.2f}", flush=True)

    net = CanonNet(**net_args).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * max(1, int((~va).sum()) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    dino = None
    if a.feat_w > 0:
        from transformers import AutoModel
        dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        for p in dino.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def feats(img):
        z = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        return dino(pixel_values=(z - mean) / std).last_hidden_state[:, 1:]          # (B, 256, D)

    tr_idx = np.where(~va)[0]
    best = -1.0
    for ep in range(a.epochs):
        net.train()
        rng.shuffle(tr_idx)
        tot = 0.0
        for s in range(0, len(tr_idx) - a.bs + 1, a.bs):
            b = tr_idx[s:s + a.bs]
            x, y = to_t(X[b], dev), to_t(Y[b], dev)
            m = torch.from_numpy(M[b]).to(dev).float()[:, None]
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                pred, _ = net(x)
                pred = pred.float()
                w = m + a.hole_w * (1 - m)
                loss = (torch.abs(pred - y) * w).sum() / (w.sum() * 3)
                if dino is not None:
                    fp, fy = feats(pred), feats(y)
                    pm = (F.adaptive_avg_pool2d(m, 16).flatten(1) > 0.9).float()
                    cos = 1 - F.cosine_similarity(fp.float(), fy.float(), dim=-1)
                    loss = loss + a.feat_w * (cos * pm).sum() / (pm.sum() + 1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        net.eval()
        res = {"camera": [[], []], "language": [[], []]}
        with torch.no_grad():
            for i in np.where(va)[0]:
                x, y = to_t(X[i:i + 1], dev), to_t(Y[i:i + 1], dev)
                m = torch.from_numpy(M[i:i + 1]).to(dev).float()[:, None]
                pred, _ = net(x)
                c = cat[int(T[i])]
                res[c][0].append(masked_psnr(pred, y, m))
                res[c][1].append(masked_psnr(x[:, :3], y, m))     # do-nothing baseline: the frame the net starts from
        line = " | ".join(f"{c}: net {np.mean(v[0]):.1f} dB vs raw {np.mean(v[1]):.1f} dB" for c, v in res.items() if v[0])
        print(f"epoch {ep + 1:3d} loss {tot / max(1, len(tr_idx) // a.bs):.4f} | val PSNR vs {a.target} target: {line}", flush=True)
        score = np.mean(res["camera"][0]) if res["camera"][0] else -np.mean([0])
        if score > best:
            best = score
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save(dict(state_dict=net.state_dict(), net_args=net_args, val_tasks=sorted(val_tasks), epoch=ep + 1,
                            val_camera_psnr=float(score), target=a.target, input=a.input), a.out)
    print(f"best val camera PSNR {best:.1f} dB -> {a.out}")


if __name__ == "__main__":
    main()
