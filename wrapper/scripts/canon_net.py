"""Learned canonicalizer: an RGB frame from an unknown camera -> the same scene seen from the nominal (training)
camera. Warp + fill: the network predicts a dense sampling field for the input image, a fill image, and a blend
weight, so large viewpoint changes are a geometric resampling rather than something a U-Net must copy through skips.

Two input modes (in_ch):
  3   the raw RGB frame only: the network must infer the camera pose from pixels and warp accordingly. Learning
      alone; on held-out poses this reached +1.4 dB (2026-09-16).
  7   the frame REPROJECTED by the wrapper's own self-calibration (3) + its hole mask (1) + the raw frame (3): the
      measured geometry is given, the network completes holes and repairs resampling artefacts. Measurement plus
      learning. The warp head is then a small residual (max_flow 0.15)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class CanonNet(nn.Module):
    def __init__(self, width=32, max_flow=0.6, in_ch=3):
        super().__init__()
        ch = [width, 2 * width, 4 * width, 8 * width, 8 * width]
        self.max_flow, self.in_ch = max_flow, in_ch
        self.enc = nn.ModuleList()
        prev = in_ch + 2                                           # input channels + 2 coordinate channels
        for c in ch:
            self.enc.append(block(prev, c))
            prev = c
        self.dec = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.dec.append(block(prev + c, c))
            prev = c
        self.head = nn.Conv2d(prev, 6, 1)                          # flow (2), fill RGB (3), blend (1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[5] = 4.0                                # start as identity: blend fully toward the warp

    def forward(self, x):
        """x: (B, in_ch, H, W) in [0, 1]; the first 3 channels are the image that gets warped. Returns (image in
        [0, 1], blend weight)."""
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                                torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype), indexing="ij")
        h = torch.cat([x, xx.expand(B, 1, H, W), yy.expand(B, 1, H, W)], 1)
        skips = []
        for i, b in enumerate(self.enc):
            h = b(h)
            if i < len(self.enc) - 1:
                skips.append(h)
                h = F.max_pool2d(h, 2)
        for b, s in zip(self.dec, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = b(torch.cat([h, s], 1))
        out = self.head(h)
        flow = torch.tanh(out[:, :2]) * self.max_flow
        grid = torch.stack([xx, yy], -1).expand(B, H, W, 2) + flow.permute(0, 2, 3, 1)
        warped = F.grid_sample(x[:, :3], grid, mode="bilinear", padding_mode="border", align_corners=True)
        fill = torch.sigmoid(out[:, 2:5])
        alpha = torch.sigmoid(out[:, 5:6])
        return alpha * warped + (1 - alpha) * fill, alpha


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    net = CanonNet(**ck.get("net_args", {}))
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def apply(net, img_uint8, device="cuda"):
    """uint8 HxWxC numpy (C = net.in_ch) -> uint8 HxWx3 numpy."""
    import numpy as np
    x = torch.from_numpy(np.ascontiguousarray(img_uint8)).to(device).permute(2, 0, 1)[None].float() / 255.0
    y, _ = net(x)
    return (y[0].permute(1, 2, 0).clamp(0, 1) * 255.0).round().byte().cpu().numpy()
