"""Box-detection metrics (COCO-style) for the COSMOS synthetic scenes.

The original :meth:`cosmos.debris.TinyYOLO.evaluate` reports a class-agnostic, centre-distance
AP above a fixed confidence and calls it "mAP@50"; the audit flagged that label.  This module
computes the standard quantities instead:

``coco_map``   per-class AP at IoU 0.50, 0.55 ... 0.95 with 101-point interpolated precision,
               greedy score-ordered matching (each truth used once); mAP@50 and mAP@50-95
               are the means over classes; detections of ALL confidences are used
``prf_at``     precision / recall / F1 / mean IoU at one confidence threshold and IoU 0.5
``small_object_ap``  AP@50 restricted to truths below an area threshold (COCO "small")

Boxes are ``(cx, cy, w, h)`` in pixels; detections carry ``score`` and ``cls``.
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between centre-form boxes a (n, 4) and b (m, 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a0, a1 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b0, b1 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    lt = np.maximum(a0[:, None], b0[None])
    rb = np.minimum(a1[:, None], b1[None])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a, area_b = a[:, 2] * a[:, 3], b[:, 2] * b[:, 3]
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-12)


def _ap(scores, matched, n_truth) -> float:
    if n_truth == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    m = np.asarray(matched, float)[order]
    tp, fp = np.cumsum(m), np.cumsum(1 - m)
    rec = tp / n_truth
    prec = tp / np.maximum(tp + fp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]           # monotone envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def _match(dets, truths, cls, thr, area_max=None):
    """Score-ordered greedy matching for one class at one IoU threshold, over all images."""
    scores, matched, n_truth = [], [], 0
    for d, t in zip(dets, truths):
        dc = d[d[:, 5] == cls] if len(d) else d
        tc = t[t[:, 0] == cls] if len(t) else t
        tb = tc[:, 1:5] if len(tc) else np.zeros((0, 4))
        keep_t = np.ones(len(tb), bool)
        if area_max is not None and len(tb):
            keep_t = tb[:, 2] * tb[:, 3] <= area_max
        n_truth += int(keep_t.sum())
        if len(dc) == 0:
            continue
        dc = dc[np.argsort(-dc[:, 4], kind="mergesort")]
        iou = iou_matrix(dc[:, :4], tb)
        used = np.zeros(len(tb), bool)
        for i in range(len(dc)):
            j = -1
            if len(tb):
                cand = np.where(~used, iou[i], -1.0)
                j = int(np.argmax(cand))
                if cand[j] < thr:
                    j = -1
            if j >= 0:
                used[j] = True
                if keep_t[j]:
                    scores.append(dc[i, 4])
                    matched.append(1)
            else:
                if area_max is None or dc[i, 2] * dc[i, 3] <= area_max:
                    scores.append(dc[i, 4])
                    matched.append(0)
    return scores, matched, n_truth


def coco_map(dets: list, truths: list, n_classes: int, area_max=None) -> dict:
    """``dets[i]`` (k, 6) = cx, cy, w, h, score, cls; ``truths[i]`` (n, 5) = cls, cx, cy, w, h."""
    ap = np.full((n_classes, len(IOU_THRESHOLDS)), np.nan)
    for c in range(n_classes):
        for k, thr in enumerate(IOU_THRESHOLDS):
            s, m, n = _match(dets, truths, c, thr, area_max)
            ap[c, k] = _ap(s, m, n)
    return {"map50": float(np.nanmean(ap[:, 0])), "map50_95": float(np.nanmean(ap)),
            "ap50_per_class": ap[:, 0].tolist(), "ap50_95_per_class": np.nanmean(ap, 1).tolist()}


def prf_at(dets: list, truths: list, conf: float, iou_thr: float = 0.5,
           class_aware: bool = True) -> dict:
    tp = fp = fn = 0
    ious = []
    for d, t in zip(dets, truths):
        d = d[d[:, 4] >= conf] if len(d) else d
        d = d[np.argsort(-d[:, 4])] if len(d) else d
        tb = t[:, 1:5] if len(t) else np.zeros((0, 4))
        used = np.zeros(len(tb), bool)
        iou = iou_matrix(d[:, :4], tb) if len(d) else np.zeros((0, len(tb)))
        for i in range(len(d)):
            cand = np.where(~used, iou[i], -1.0) if len(tb) else np.array([-1.0])
            if class_aware and len(tb):
                cand = np.where(t[:, 0] == d[i, 5], cand, -1.0)
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                used[j] = True
                tp += 1
                ious.append(cand[j])
            else:
                fp += 1
        fn += int((~used).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
            "tp": tp, "fp": fp, "fn": fn}
