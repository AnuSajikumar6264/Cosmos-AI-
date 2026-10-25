"""FEATURE 2 (d) — a learned YOLO-style detector.

Follows the approach of the reference pipeline
https://github.com/jaikr-dev/space-debris-detection-pipeline: synthetic renders with domain
randomisation, a YOLO-format dataset, nano-scale single-stage detectors chosen for
constrained on-board hardware, mAP@50 plus a confusion matrix, and inference at a 0.60
confidence threshold.

Two backends share one interface:

``TinyYOLO``          anchor-free single-scale detector written directly in PyTorch, so it
                      trains here in seconds on CPU with no extra dependency.
``UltralyticsYOLO``   thin adapter around ``ultralytics.YOLO`` (v8n / v11n / v12n).  Used
                      when the package is installed, which is what the reference pipeline
                      benchmarks; it reports mAP50-95 around 0.976 for v8n and v11n on its
                      own Blender dataset.

    m = TinyYOLO().fit(yolo_dir, epochs=40)
    m.evaluate(yolo_dir)
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .scene import CLASSES

CONF_DEFAULT = 0.60          # same operating point as the reference pipeline
GRID_STRIDE = 8
TRAIN_SIZE = 320             # the reference uses 640; half that trains far faster on CPU


def _torch():
    try:
        import torch
        return torch
    except Exception as exc:                                        # pragma: no cover
        raise ImportError("PyTorch is required for the learned detector") from exc


# --------------------------------------------------------------------------- data loading
def load_yolo_split(root: str, split: str = "train", size: int = TRAIN_SIZE):
    """Read a YOLO-format split into ``(images, labels)``.

    Images come back as ``float32`` in ``[0, 1]`` shaped ``(N, 1, size, size)``; labels as a
    list of ``(k, 5)`` arrays holding ``class, cx, cy, w, h`` normalised to ``[0, 1]``.
    """
    from PIL import Image
    img_dir = os.path.join(root, split, "images")
    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                   for p in glob.glob(os.path.join(img_dir, ext)))
    if not paths:
        raise FileNotFoundError(f"no images in {img_dir}")
    imgs, labels = [], []
    for p in paths:
        im = Image.open(p).convert("L").resize((size, size))
        imgs.append(np.asarray(im, np.float32) / 255.0)
        lp = os.path.splitext(p.replace(f"{os.sep}images{os.sep}",
                                        f"{os.sep}labels{os.sep}"))[0] + ".txt"
        rows = []
        if os.path.isfile(lp):
            for line in open(lp):
                parts = line.split()
                if len(parts) == 5:
                    rows.append([float(x) for x in parts])
        labels.append(np.array(rows, np.float32).reshape(-1, 5))
    return np.stack(imgs)[:, None, :, :], labels


# ------------------------------------------------------------------------------ the model
def _build_net(torch, n_classes: int):
    nn = torch.nn

    def blk(i, o, stride=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False),
                             nn.BatchNorm2d(o), nn.SiLU(inplace=True))

    return nn.Sequential(
        blk(1, 16), blk(16, 32, 2),          # /2
        blk(32, 32), blk(32, 64, 2),         # /4
        blk(64, 64), blk(64, 96, 2),         # /8  -> GRID_STRIDE
        blk(96, 96), blk(96, 128),
        nn.Conv2d(128, 5 + n_classes, 1),    # obj, dx, dy, w, h, class logits
    )


@dataclass
class TinyYOLO:
    """Anchor-free single-scale detector: one prediction per 8x8 cell.

    Deliberately small — the point of the reference pipeline's nano models is that detection
    has to run on constrained hardware, and a project demo has to train in seconds.
    """
    n_classes: int = len(CLASSES)
    size: int = TRAIN_SIZE
    device: str = "cpu"
    class_names: list = None           # defaults to the simulator's four classes
    net: object = field(default=None, repr=False)
    history: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = list(CLASSES)[:self.n_classes]
        else:
            self.class_names = list(self.class_names)
            self.n_classes = len(self.class_names)

    # ---------------------------------------------------------------- target construction
    def _targets(self, torch, labels, grid):
        b = len(labels)
        tobj = torch.zeros((b, 1, grid, grid))
        tbox = torch.zeros((b, 4, grid, grid))
        tcls = torch.zeros((b, grid, grid), dtype=torch.long)
        for i, lab in enumerate(labels):
            for row in lab:
                c, cx, cy, w, h = (float(v) for v in row)
                gx, gy = min(int(cx * grid), grid - 1), min(int(cy * grid), grid - 1)
                tobj[i, 0, gy, gx] = 1.0
                tbox[i, 0, gy, gx] = cx * grid - gx          # offset inside the cell
                tbox[i, 1, gy, gx] = cy * grid - gy
                # width/height in LOG units of the cell stride.  Normalised-to-image
                # targets are ~0.03 here, which drives a sigmoid head into saturation and
                # leaves boxes the right place but the wrong size (IoU ~0.35).
                tbox[i, 2, gy, gx] = np.log(max(w * self.size, 1.0) / GRID_STRIDE)
                tbox[i, 3, gy, gx] = np.log(max(h * self.size, 1.0) / GRID_STRIDE)
                tcls[i, gy, gx] = int(c)
        return tobj, tbox, tcls

    def fit(self, root: str, epochs: int = 40, lr: float = 3e-3, batch: int = 8,
            seed: int = 0, verbose: bool = False) -> "TinyYOLO":
        torch = _torch()
        torch.manual_seed(seed)
        x, labels = load_yolo_split(root, "train", self.size)
        grid = self.size // GRID_STRIDE
        self.net = _build_net(torch, self.n_classes).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        X = torch.tensor(x, device=self.device)
        tobj, tbox, tcls = self._targets(torch, labels, grid)
        tobj, tbox, tcls = tobj.to(self.device), tbox.to(self.device), tcls.to(self.device)

        rows = []
        n = X.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                out = self.net(X[idx])
                o_logit = out[:, 0:1]
                pos = tobj[idx]
                # objectness: positives are rare, so weight them up rather than let the
                # background dominate the gradient
                w = 1.0 + 60.0 * pos
                loss = (bce(o_logit, pos) * w).mean()
                if pos.sum() > 0:
                    mask = pos.expand_as(tbox[idx]) > 0
                    pbox = torch.cat([out[:, 1:3].sigmoid(), out[:, 3:5]], 1)
                    loss = loss + 5.0 * torch.nn.functional.l1_loss(pbox[mask],
                                                                    tbox[idx][mask])
                    cmask = pos[:, 0] > 0
                    if cmask.sum() > 0:
                        logits = out[:, 5:].permute(0, 2, 3, 1)[cmask]
                        loss = loss + torch.nn.functional.cross_entropy(
                            logits, tcls[idx][cmask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            rows.append({"epoch": ep + 1, "loss": tot / n})
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep + 1:3d}  loss {tot / n:.4f}")
        self.history = pd.DataFrame(rows)
        return self

    # ------------------------------------------------------------------------- inference
    def predict(self, images: np.ndarray, conf: float = CONF_DEFAULT,
                min_separation_px: float = 6.0) -> list:
        """Decode detections for a batch of images. Returns one DataFrame per image."""
        torch = _torch()
        if self.net is None:
            raise RuntimeError("call fit() first")
        self.net.eval()
        grid = self.size // GRID_STRIDE
        with torch.no_grad():
            out = self.net(torch.tensor(images, device=self.device)).cpu().numpy()
        results = []
        for k in range(out.shape[0]):
            o = 1.0 / (1.0 + np.exp(-out[k, 0]))
            ys, xs = np.where(o > conf)
            rows = []
            for gy, gx in zip(ys, xs):
                sig = lambda v: 1.0 / (1.0 + np.exp(-v))
                cx = (gx + sig(out[k, 1, gy, gx])) / grid * self.size
                cy = (gy + sig(out[k, 2, gy, gx])) / grid * self.size
                w = float(np.exp(out[k, 3, gy, gx])) * GRID_STRIDE
                h = float(np.exp(out[k, 4, gy, gx])) * GRID_STRIDE
                cls = int(np.argmax(out[k, 5:, gy, gx]))
                rows.append({"cx": cx, "cy": cy, "w": w, "h": h, "score": float(o[gy, gx]),
                             "cls": cls, "cls_name": self.class_names[cls]})
            df = pd.DataFrame(rows)
            if len(df):
                df = df.sort_values("score", ascending=False)
                keep = []
                for r in df.itertuples():
                    if all((r.cx - k2[0]) ** 2 + (r.cy - k2[1]) ** 2 > min_separation_px ** 2
                           for k2 in keep):
                        keep.append((r.cx, r.cy))
                    else:
                        df = df.drop(r.Index)
            results.append(df.reset_index(drop=True))
        return results

    # --------------------------------------------------------------------------- metrics
    def evaluate(self, root: str, split: str = "val", conf: float = CONF_DEFAULT,
                 tol_px: float = 6.0, match: str = "centre", iou_min: float = 0.5) -> dict:
        """Precision / recall / F1, plus mAP@50 and a class matrix.

        ``match="centre"`` pairs a detection with the nearest truth within ``tol_px`` — right
        for unresolved point sources, where a box says little beyond its centre.
        ``match="iou"`` pairs by box overlap of at least ``iou_min``, the standard for
        resolved objects (SPARK, DUST).
        """
        x, labels = load_yolo_split(root, split, self.size)
        preds = self.predict(x, conf, min_separation_px=tol_px if match == "centre"
                             else self.size / 20)
        tp = fp = fn = 0
        cm = np.zeros((self.n_classes + 1, self.n_classes + 1), int)
        ious, scores, matched = [], [], []
        for pr, lab in zip(preds, labels):
            t = lab.astype(np.float64).copy()
            t[:, 1] *= self.size; t[:, 2] *= self.size
            t[:, 3] *= self.size; t[:, 4] *= self.size
            used = np.zeros(len(t), bool)
            for r in pr.itertuples():
                if len(t) and match == "iou":
                    ov = np.array([_iou((r.cx, r.cy, r.w, r.h), tt[1:]) for tt in t])
                    ov[used] = -1.0
                    j = int(np.argmax(ov))
                    ok = ov[j] >= iou_min
                elif len(t):
                    d = np.hypot(t[:, 1] - r.cx, t[:, 2] - r.cy)
                    d[used] = np.inf
                    j = int(np.argmin(d))
                    ok = d[j] <= tol_px
                else:
                    j, ok = 0, False
                if ok:
                    used[j] = True
                    tp += 1
                    cm[int(t[j, 0]), r.cls] += 1
                    ious.append(_iou((r.cx, r.cy, r.w, r.h), t[j, 1:]))
                    scores.append(r.score); matched.append(1)
                else:
                    fp += 1
                    cm[self.n_classes, r.cls] += 1
                    scores.append(r.score); matched.append(0)
            for j in np.flatnonzero(~used):
                fn += 1
                cm[int(t[j, 0]), self.n_classes] += 1
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {"split": split, "conf": conf, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                "mean_iou": round(float(np.mean(ious)), 4) if ious else float("nan"),
                "map50": round(_average_precision(scores, matched, tp + fn), 4),
                "match": match,
                "confusion_matrix": pd.DataFrame(
                    cm, index=self.class_names + ["(background)"],
                    columns=self.class_names + ["(missed)"])}


def _iou(a, b) -> float:
    """IoU of two centre-form boxes ``(cx, cy, w, h)``."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / max(union, 1e-9))


def _average_precision(scores, matched, n_truth: int) -> float:
    """Area under the precision-recall curve, the standard AP definition."""
    if not scores or n_truth == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    m = np.asarray(matched)[order]
    tp = np.cumsum(m)
    fp = np.cumsum(1 - m)
    rec = tp / n_truth
    prec = tp / np.maximum(tp + fp, 1)
    ap, prev = 0.0, 0.0
    for r, p in zip(rec, prec):
        ap += (r - prev) * p
        prev = r
    return float(ap)


# ------------------------------------------------------- optional Ultralytics backend
@dataclass
class UltralyticsYOLO:
    """Adapter for the real thing, when ``ultralytics`` is installed.

    ``model`` may be ``yolov8n.pt``, ``yolo11n.pt`` or ``yolo12n.pt`` — the three the
    reference pipeline benchmarks.
    """
    model: str = "yolov8n.pt"
    _m: object = field(default=None, repr=False)

    @staticmethod
    def available() -> bool:
        try:
            import ultralytics                                     # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16):
        from ultralytics import YOLO
        self._m = YOLO(self.model)
        self._m.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, verbose=False)
        return self

    def evaluate(self, data_yaml: str) -> dict:
        if self._m is None:
            raise RuntimeError("call fit() first")
        r = self._m.val(data=data_yaml, verbose=False)
        return {"model": self.model, "map50": float(r.box.map50), "map50_95": float(r.box.map),
                "precision": float(r.box.mp), "recall": float(r.box.mr)}


def reference_benchmark() -> pd.DataFrame:
    """Published numbers from the reference pipeline, for comparison in the report.

    These are that project's results on its own Blender-rendered dataset, not ours.  They
    are reproduced here so the report can be explicit about whose result is whose.
    """
    return pd.DataFrame([
        {"model": "YOLOv8n", "mAP50-95": 0.976, "source": "reference pipeline (Blender data)"},
        {"model": "YOLOv11n", "mAP50-95": 0.976, "source": "reference pipeline (Blender data)"},
        {"model": "YOLOv12n", "mAP50-95": 0.969, "source": "reference pipeline (Blender data)"},
    ])
