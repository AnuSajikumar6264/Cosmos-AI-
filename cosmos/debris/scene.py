"""FEATURE 2 (b) — synthetic observation scenes, and YOLO-format dataset export.

Two frames of the same star field are rendered a short interval apart.  Stars stay put;
debris moves.  Each frame gets its own Poisson and read noise, and the pair gets a small
pointing jitter, so registration is a real problem rather than a formality.

The domain-randomisation and YOLO-dataset approach follows the reference pipeline at
https://github.com/jaikr-dev/space-debris-detection-pipeline — images standardised to
640x640, labels as ``class cx cy w h`` normalised, a ``data.yaml``, a train/val split and
an augmentation summary.  Renders here are analytic rather than Blender, so the whole
dataset builds in seconds and needs no external tool.

    ds = generate_dataset(n_scenes=40)
    export_yolo_dataset(ds)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import feature_dir, rng_of

#: object classes.  The reference pipeline used (Satellite, Envisat, Hubble, Falcon 9 F&S);
#: here the classes are the debris categories the catalogue itself distinguishes.
CLASSES = ["satellite", "rocket_body", "fragment", "panel"]
#: PSF sigma [px] and total flux [ADU] per class.  What a detector responds to is peak
#: surface brightness, flux / (2 pi sigma^2), so these two must be chosen together.
CLASS_PROPS = {
    "satellite":   dict(sigma=(1.2, 1.8), flux=(2000.0, 4500.0)),
    "rocket_body": dict(sigma=(1.5, 2.2), flux=(2600.0, 6000.0)),
    "fragment":    dict(sigma=(0.8, 1.2), flux=(900.0, 2000.0)),
    "panel":       dict(sigma=(1.0, 1.6), flux=(1400.0, 3000.0)),
}
IMG_SIZE = 640


def _gauss_stamp(img, x, y, sigma, flux):
    """Add one Gaussian point source, clipped to the image, in place."""
    h, w = img.shape
    rad = int(max(3, np.ceil(3.5 * sigma)))
    x0, x1 = max(0, int(x) - rad), min(w, int(x) + rad + 1)
    y0, y1 = max(0, int(y) - rad), min(h, int(y) + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    s = g.sum()
    if s > 0:
        img[y0:y1, x0:x1] += flux * g / s


@dataclass
class Scene:
    """One frame pair plus the ground truth for the moving objects."""
    frame_a: np.ndarray
    frame_b: np.ndarray
    truth: pd.DataFrame                # cx, cy (in frame B), w, h, cls, cls_name
    jitter: tuple                      # (dy, dx) applied to frame B, in pixels
    n_stars: int
    seed: int

    def boxes(self) -> np.ndarray:
        """Truth boxes as ``(cx, cy, w, h)`` in pixels."""
        return self.truth[["cx", "cy", "w", "h"]].values.astype(float)


@dataclass
class SceneDataset:
    """A collection of scenes, plus the augmentation record."""
    scenes: list = field(default_factory=list)
    augmentations: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.scenes)

    def summary(self) -> dict:
        n_obj = sum(len(s.truth) for s in self.scenes)
        cls_counts = pd.concat([s.truth for s in self.scenes]).cls_name.value_counts()
        return {"scenes": len(self.scenes), "objects": n_obj,
                "objects_per_scene": round(n_obj / max(len(self.scenes), 1), 2),
                "image_size": IMG_SIZE, "classes": len(CLASSES),
                **{f"n_{k}": int(v) for k, v in cls_counts.items()},
                "mean_stars": int(np.mean([s.n_stars for s in self.scenes]))}


def make_scene(size: int = IMG_SIZE, n_stars: int = 140, n_objects: int = 8,
               jitter_px: float = 1.0, psf_sigma: float = 1.6, dt_s: float = 4.0,
               limb_glow: bool = True, seed=0) -> Scene:
    """Render one frame pair.

    Parameters
    ----------
    n_stars, n_objects
        Static sources and moving objects.
    jitter_px
        Standard deviation of the pointing shift between the two frames.
    dt_s
        Interval between frames; sets how far the objects move.
    """
    rng = rng_of(seed)
    bg = 40.0                              # sky background [ADU]
    a = np.full((size, size), bg, np.float64)

    stars = rng.random((n_stars, 2)) * size
    star_flux = 10.0 ** rng.uniform(2.0, 3.95, n_stars)   # many stars outshine the movers
    for (sx, sy), f in zip(stars, star_flux):
        _gauss_stamp(a, sx, sy, psf_sigma, f)

    if limb_glow:                          # Earth limb brightening in one corner
        yy, xx = np.mgrid[0:size, 0:size]
        r = np.hypot(xx - size * 0.18, yy - size * 0.78)
        a += 110.0 * np.exp(-(r / (size * 0.16)) ** 2)

    b = a.copy()

    # moving objects: same source drawn at two positions
    rows = []
    for _ in range(n_objects):
        cname = CLASSES[rng.integers(len(CLASSES))]
        p = CLASS_PROPS[cname]
        sig = rng.uniform(*p["sigma"])
        flux = rng.uniform(*p["flux"])
        x0, y0 = rng.uniform(0.08, 0.92, 2) * size
        speed = rng.uniform(1.6, 6.5)                      # px per second
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = speed * dt_s * np.cos(ang), speed * dt_s * np.sin(ang)
        x1, y1 = x0 + dx, y0 + dy
        if not (0 < x1 < size and 0 < y1 < size):
            x1, y1 = x0 - dx, y0 - dy
        _gauss_stamp(a, x0, y0, sig, flux)
        _gauss_stamp(b, x1, y1, sig, flux)
        box = max(5.0, 6.0 * sig)
        rows.append({"cx": x1, "cy": y1, "w": box, "h": box,
                     "cls": CLASSES.index(cname), "cls_name": cname,
                     "flux": flux, "sigma": sig, "speed_px_s": speed})

    # pointing jitter on frame B, then independent noise on both
    jy, jx = rng.normal(0.0, jitter_px, 2)
    b = _shift(b, jy, jx)
    a = rng.poisson(np.clip(a, 0, None)).astype(np.float64) + rng.normal(0, 1.5, a.shape)
    b = rng.poisson(np.clip(b, 0, None)).astype(np.float64) + rng.normal(0, 1.5, b.shape)

    truth = pd.DataFrame(rows)
    # Two coordinate frames matter and mixing them up costs every true positive:
    #   cx,  cy   position on the REGISTERED / difference image (frame A's grid)
    #   cx_b, cy_b position on the RAW jittered frame B, which is what a single-frame
    #              detector actually sees
    if len(truth):
        truth["cx_b"] = truth.cx + jx
        truth["cy_b"] = truth.cy + jy
    return Scene(a, b, truth, (float(jy), float(jx)), n_stars, int(seed))


def _shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Sub-pixel shift by bilinear interpolation (no SciPy dependency)."""
    iy, ix = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - iy, dx - ix
    out = np.zeros_like(img)
    for wy, oy in ((1 - fy, iy), (fy, iy + 1)):
        for wx, ox in ((1 - fx, ix), (fx, ix + 1)):
            if wy * wx == 0.0:
                continue
            out += wy * wx * np.roll(np.roll(img, oy, axis=0), ox, axis=1)
    return out


def generate_dataset(n_scenes: int = 40, n_objects=(5, 12), n_stars=(90, 220),
                     jitter_px: float = 1.0, seed=0) -> SceneDataset:
    """Domain-randomised set of scenes: star count, object count and jitter all vary."""
    rng = rng_of(seed)
    scenes = []
    for i in range(n_scenes):
        scenes.append(make_scene(
            n_stars=int(rng.integers(*n_stars)),
            n_objects=int(rng.integers(*n_objects)),
            jitter_px=float(jitter_px * rng.uniform(0.4, 1.8)),
            psf_sigma=float(rng.uniform(1.3, 2.1)),
            dt_s=float(rng.uniform(2.5, 6.0)),
            seed=int(rng.integers(1 << 30))))
    aug = {"horizontal_flip": True, "vertical_flip": True, "brightness_jitter": "±15 %",
           "contrast_jitter": "±10 %", "gaussian_noise": "sigma 1-4 ADU",
           "domain_randomisation": "star count, PSF width, pointing jitter, frame interval"}
    return SceneDataset(scenes, aug)


def _augment(img: np.ndarray, boxes: np.ndarray, rng) -> tuple:
    """Apply flips and photometric jitter, keeping labels consistent."""
    h, w = img.shape
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] = w - boxes[:, 0]
    if rng.random() < 0.5:
        img = img[::-1, :].copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 1] = h - boxes[:, 1]
    img = img * rng.uniform(0.85, 1.15) + rng.normal(0, rng.uniform(1.0, 4.0), img.shape)
    return img, boxes


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile stretch to 8-bit, which is what a detector actually sees."""
    lo, hi = np.percentile(img, 1.0), np.percentile(img, 99.8)
    return np.clip((img - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def export_yolo_dataset(ds: SceneDataset, out_dir: str | None = None, val_frac: float = 0.25,
                        augment_train: bool = True, seed=0) -> dict:
    """Write a YOLO-format dataset: ``images/``, ``labels/``, ``data.yaml``.

    Label lines are ``class cx cy w h`` with all coordinates normalised to ``[0, 1]``,
    exactly as Ultralytics expects, so the reference pipeline can train on this directly.
    """
    from PIL import Image
    rng = rng_of(seed)
    out_dir = out_dir or feature_dir("debris", "vision", "yolo_dataset")
    n_val = max(1, int(round(val_frac * len(ds))))
    counts = {"train": 0, "val": 0}
    labels_written = 0

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)

    for i, sc in enumerate(ds.scenes):
        split = "val" if i < n_val else "train"
        # the detector is trained on the DIFFERENCE image: that is the input where a
        # fragment is separable from a star at all
        from .detect import register_and_difference
        diff, _shift_est = register_and_difference(sc.frame_a, sc.frame_b)
        img, boxes = diff, sc.boxes()
        if split == "train" and augment_train:
            img, boxes = _augment(img, boxes, rng)

        u8 = to_uint8(img)
        h, w = u8.shape
        Image.fromarray(u8).resize((IMG_SIZE, IMG_SIZE)).save(
            os.path.join(out_dir, split, "images", f"scene_{i:04d}.png"))
        sx, sy = IMG_SIZE / w, IMG_SIZE / h
        lines = []
        for (cx, cy, bw, bh), cls in zip(boxes, sc.truth.cls.values):
            lines.append(f"{int(cls)} {cx * sx / IMG_SIZE:.6f} {cy * sy / IMG_SIZE:.6f} "
                         f"{bw * sx / IMG_SIZE:.6f} {bh * sy / IMG_SIZE:.6f}")
        with open(os.path.join(out_dir, split, "labels", f"scene_{i:04d}.txt"), "w") as f:
            f.write("\n".join(lines))
        counts[split] += 1
        labels_written += len(lines)

    yaml = (f"path: {out_dir}\ntrain: train/images\nval: val/images\n"
            f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(yaml)
    with open(os.path.join(out_dir, "augmentation_summary.json"), "w") as f:
        json.dump(ds.augmentations, f, indent=2)

    return {"path": out_dir, "train_images": counts["train"], "val_images": counts["val"],
            "labels": labels_written, "image_size": IMG_SIZE, "classes": CLASSES,
            "data_yaml": os.path.join(out_dir, "data.yaml")}


def save_preview(sc: Scene, path: str | None = None) -> str:
    """Save the raw pair and the difference as one PNG, for the report."""
    from PIL import Image
    from .detect import register_and_difference
    diff, _ = register_and_difference(sc.frame_a, sc.frame_b)
    strip = np.concatenate([to_uint8(sc.frame_a), to_uint8(sc.frame_b), to_uint8(diff)], axis=1)
    path = path or os.path.join(feature_dir("debris", "vision", "frames"), "preview.png")
    Image.fromarray(strip).save(path)
    return path
