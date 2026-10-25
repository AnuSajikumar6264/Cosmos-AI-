"""Point-target detection as heat-map regression (objects-as-points / CenterNet style).

``HeatmapUNet``      U-Net trained from scratch; ``ResNetUNet`` the same decoder on an
                     ImageNet-pretrained ResNet-18 encoder (torchvision weights), its first
                     convolution widened to the input channels by repeating the mean RGB filter.
Heads                ``heat`` (one logit map per output channel: per frame for SpotGEO, per
                     class for the COSMOS synthetic scenes), ``offset`` (sub-pixel dx, dy) and
                     optionally ``wh`` (box size, log-pixels) for box metrics.
Loss                 penalty-reduced focal loss on Gaussian-splatted targets (Law & Deng 2018;
                     Zhou et al. 2019) + L1 on offsets / sizes at object pixels.
Decoding             3 x 3 max-pool peaks above a threshold, + offset; threshold chosen on
                     validation, never on test.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                         nn.SiLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.SiLU(inplace=True))


class _Heads(nn.Module):
    def __init__(self, c, n_heat, wh):
        super().__init__()
        self.heat = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, n_heat, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)       # prior 0.1, as in CenterNet
        self.offset = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(c, 2 * n_heat, 1))
        self.wh = (nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(c, 2 * n_heat, 1)) if wh else None)

    def forward(self, h):
        out = {"heat": self.heat(h), "offset": self.offset(h)}
        if self.wh is not None:
            out["wh"] = self.wh(h)
        return out


class HeatmapUNet(nn.Module):
    def __init__(self, in_ch: int, n_heat: int, base: int = 32, depth: int = 4,
                 wh: bool = False):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_ch
        for o in ch:
            self.down.append(_block(c, o))
            c = o
        self.mid = _block(c, c * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = c * 2
        for o in reversed(ch):
            self.up.append(nn.ConvTranspose2d(c, o, 2, stride=2))
            self.dec.append(_block(2 * o, o))
            c = o
        self.heads = _Heads(c, n_heat, wh)
        self.stride_multiple = 2 ** depth

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return self.heads(x)


class ResNetUNet(nn.Module):
    """U-Net decoder on a (pretrained) ResNet-18; full-resolution output via a stem branch."""

    def __init__(self, in_ch: int, n_heat: int, pretrained: bool = True, wh: bool = False):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        conv = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(r.conv1.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
                              * 3.0 / in_ch)
        self.stem = nn.Sequential(conv, r.bn1, r.relu)            # /2, 64
        self.pool = r.maxpool                                     # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.full = _block(in_ch, 32)                             # /1 branch
        self.u4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d4 = _block(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = _block(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = _block(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.d1 = _block(128, 64)
        self.u0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d0 = _block(64, 32)
        self.heads = _Heads(32, n_heat, wh)
        self.stride_multiple = 32

    def forward(self, x):
        f0 = self.full(x)
        s1 = self.stem(x)
        e1 = self.l1(self.pool(s1))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        d = self.d4(torch.cat([self.u4(e4), e3], 1))
        d = self.d3(torch.cat([self.u3(d), e2], 1))
        d = self.d2(torch.cat([self.u2(d), e1], 1))
        d = self.d1(torch.cat([self.u1(d), s1], 1))
        d = self.d0(torch.cat([self.u0(d), f0], 1))
        return self.heads(d)


def build(kind: str, in_ch: int, n_heat: int, wh: bool = False) -> nn.Module:
    if kind == "unet":
        return HeatmapUNet(in_ch, n_heat, wh=wh)
    if kind == "resnet18_pretrained":
        return ResNetUNet(in_ch, n_heat, pretrained=True, wh=wh)
    if kind == "resnet18_scratch":
        return ResNetUNet(in_ch, n_heat, pretrained=False, wh=wh)
    raise KeyError(kind)


# ============================================================================ targets / loss
def render(points_per_channel: list, shape: tuple, sigma: float = 1.5, sizes=None):
    """Gaussian heat-map targets, offset targets and the object-pixel mask.

    ``points_per_channel[c]`` is a (k, 2) array of (x, y) in pixels.  Returns ``heat``
    (C, H, W), ``off`` (2C, H, W), ``wh`` (2C, H, W) and ``mask`` (C, H, W).
    """
    C = len(points_per_channel)
    Hh, Ww = shape
    heat = np.zeros((C, Hh, Ww), np.float32)
    off = np.zeros((2 * C, Hh, Ww), np.float32)
    whm = np.zeros((2 * C, Hh, Ww), np.float32)
    mask = np.zeros((C, Hh, Ww), np.float32)
    r = int(np.ceil(3 * sigma))
    for c, pts in enumerate(points_per_channel):
        for k, (x, y) in enumerate(np.asarray(pts, np.float64).reshape(-1, 2)):
            ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))   # nearest pixel centre
            if not (0 <= ix < Ww and 0 <= iy < Hh):
                continue
            x0, x1, y0, y1 = max(ix - r, 0), min(ix + r + 1, Ww), max(iy - r, 0), min(iy + r + 1, Hh)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            heat[c, y0:y1, x0:x1] = np.maximum(heat[c, y0:y1, x0:x1], g)
            heat[c, iy, ix] = 1.0
            off[2 * c, iy, ix], off[2 * c + 1, iy, ix] = x - ix, y - iy
            if sizes is not None:
                w_, h_ = sizes[c][k]
                whm[2 * c, iy, ix], whm[2 * c + 1, iy, ix] = np.log(max(w_, 1.0)), np.log(max(h_, 1.0))
            mask[c, iy, ix] = 1.0
    return heat, off, whm, mask


def focal_loss(logits, target, alpha: float = 2.0, beta: float = 4.0):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def detection_loss(out, heat, off, mask, wh=None):
    l_heat = focal_loss(out["heat"], heat)
    m2 = mask.repeat_interleave(2, dim=1)
    n = mask.sum().clamp(min=1.0)
    l_off = (F.l1_loss(out["offset"], off, reduction="none") * m2).sum() / n
    loss = l_heat + l_off
    if wh is not None and "wh" in out:
        loss = loss + 0.1 * (F.l1_loss(out["wh"], wh, reduction="none") * m2).sum() / n
    return loss, {"heat": float(l_heat.detach()), "offset": float(l_off.detach())}


# ============================================================================ decoding
def decode(out: dict, thr: float, max_det: int = 30, nms: int = 3) -> list:
    """Per image and channel: (k, 3 or 5) arrays of x, y, score (+ w, h)."""
    heat = torch.sigmoid(out["heat"])
    peak = F.max_pool2d(heat, nms, 1, nms // 2) == heat
    heat = heat * peak
    B, C, Hh, Ww = heat.shape
    res = []
    for b in range(B):
        per = []
        for c in range(C):
            h = heat[b, c]
            idx = torch.nonzero(h > thr)
            if idx.numel() == 0:
                per.append(np.zeros((0, 5 if "wh" in out else 3), np.float32))
                continue
            sc = h[idx[:, 0], idx[:, 1]]
            order = torch.argsort(sc, descending=True)[:max_det]
            idx, sc = idx[order], sc[order]
            ox = out["offset"][b, 2 * c, idx[:, 0], idx[:, 1]]
            oy = out["offset"][b, 2 * c + 1, idx[:, 0], idx[:, 1]]
            cols = [idx[:, 1].float() + ox, idx[:, 0].float() + oy, sc]
            if "wh" in out:
                cols += [torch.exp(out["wh"][b, 2 * c, idx[:, 0], idx[:, 1]]),
                         torch.exp(out["wh"][b, 2 * c + 1, idx[:, 0], idx[:, 1]])]
            per.append(torch.stack(cols, 1).float().cpu().numpy())
        res.append(per)
    return res
