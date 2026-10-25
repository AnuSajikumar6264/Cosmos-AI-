"""FEATURE 2 (c) — classical detection: register, difference, then find what moved.

A single frame cannot separate a distant fragment from a star: both are unresolved point
sources through the same PSF, and the fragment is usually the fainter one.  No threshold
fixes that, because the information is not in the frame.  Motion is the discriminator, so
the pipeline registers the pair to sub-pixel accuracy, subtracts, and detects on what is
left.

One sign convention matters.  The phase-correlation peak sits at **-d**, so using ``+d``
shifts the frames further apart instead of aligning them, leaving star residuals everywhere
and burying real detections.  :func:`estimate_shift` returns the shift that must be applied
to frame B to bring it onto frame A.

    diff, shift = register_and_difference(a, b)
    det = dog_detect(diff)
    score_detections(det, truth)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def estimate_shift(a: np.ndarray, b: np.ndarray, upsample: bool = True) -> tuple:
    """Sub-pixel CORRECTION ``(dy, dx)`` to apply to frame B to bring it onto frame A.

    Sign convention, which is the easiest thing in this whole pipeline to get backwards: if
    frame B's content sits ``+j`` pixels from frame A's, this returns ``-j``, so that
    ``apply_shift(b, *estimate_shift(a, b))`` aligns.  Returning ``+j`` instead shifts the
    frames further apart, and with small jitter the error is invisible — alignment is merely
    twice as bad, detections still land inside a 3 px tolerance — so it survives casual
    testing and only collapses once the jitter exceeds about a pixel.
    See :func:`check_shift_sign` for the test that pins it down.
    """
    a = a - a.mean()
    b = b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.rfft2(a * win)
    fb = np.fft.rfft2(b * win)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    r = np.fft.irfft2(cross / np.maximum(mag, 1e-12), s=a.shape)
    py, px = np.unravel_index(np.argmax(r), r.shape)

    def _parabolic(v_lo, v_mid, v_hi):
        den = v_lo - 2.0 * v_mid + v_hi
        return 0.0 if abs(den) < 1e-12 else 0.5 * (v_lo - v_hi) / den

    dy, dx = float(py), float(px)
    if upsample:
        h, w = r.shape
        dy += _parabolic(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
        dx += _parabolic(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    h, w = r.shape
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    # the cross-power peak already sits at the negated displacement, which IS the
    # correction that has to be applied to frame B
    return dy, dx


def apply_shift(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    from .scene import _shift
    return _shift(img, dy, dx)


def register_and_difference(a: np.ndarray, b: np.ndarray) -> tuple:
    """Align B onto A, subtract, and return ``(difference, (dy, dx))``.

    The difference is returned SIGNED.  A mover appears as a positive blob at its new
    position and a negative one at the old, and the detector only looks for positive peaks.
    Clipping to positive here would be a mistake: it makes the residual one-sided, so the
    median-absolute-deviation threshold inside :func:`dog_detect` collapses to zero and the
    detector fires on noise.

    The border is zeroed.  Sub-pixel shifting is done with a circular roll, so after a shift
    of d pixels a d-wide strip at each edge holds content wrapped from the opposite side,
    which differences against something unrelated and produces residuals far larger than any
    real object.  Left in place they flood the detector and, worse, inflate the local noise
    estimate everywhere else, which blinds it.  The strip carries no usable signal, so it is
    masked rather than salvaged.
    """
    dy, dx = estimate_shift(a, b)
    b_reg = apply_shift(b, dy, dx)
    diff = b_reg - a
    m = int(np.ceil(max(abs(dy), abs(dx)))) + 3
    if m > 0:
        diff[:m, :] = 0.0
        diff[-m:, :] = 0.0
        diff[:, :m] = 0.0
        diff[:, -m:] = 0.0
    return diff, (dy, dx)


def photon_noise_map(a: np.ndarray, b: np.ndarray, read_noise: float = 1.5) -> np.ndarray:
    """Per-pixel standard deviation of the DIFFERENCE image, from photon statistics.

    Shot noise grows with the signal, so the pixels under a bright star are intrinsically
    noisier than empty sky.  Judging a difference image against one global (or even
    block-median) threshold therefore flags bright-star positions as high-significance
    detections, which is the dominant false positive in this pipeline.  Weighting by this
    map is what removes them.
    """
    var = np.clip(a, 0.0, None) + np.clip(b, 0.0, None) + 2.0 * read_noise ** 2
    return np.sqrt(var)


def _local_sigma(resp: np.ndarray, block: int = 64) -> np.ndarray:
    """Per-block robust noise estimate, upsampled back to the image grid.

    A single global threshold is wrong on a difference image because the noise is not
    stationary: the Earth-limb corner carries far more photon noise than empty sky, so a
    global cut either floods that corner with false positives or goes blind everywhere else.
    """
    h, w = resp.shape
    ph, pw = (-h) % block, (-w) % block
    dev = np.pad(np.abs(resp - np.median(resp)), ((0, ph), (0, pw)), mode="edge")
    H, W = dev.shape
    med = np.median(dev.reshape(H // block, block, W // block, block), axis=(1, 3)) * 1.4826
    up = np.repeat(np.repeat(med, block, axis=0), block, axis=1)[:h, :w]
    return np.maximum(up, 1e-9)


def _blur_sq(x: np.ndarray, sigma: float) -> np.ndarray:
    """Blur with the SQUARED Gaussian kernel — how a variance map transforms under a blur."""
    k = max(1, int(round(sigma * 2.5)) | 1)
    ker = np.exp(-0.5 * ((np.arange(k) - k // 2) / max(sigma, 1e-6)) ** 2)
    ker = (ker / ker.sum()) ** 2
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 0, x)
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, out)


def _dog(img: np.ndarray, s1: float, s2: float) -> np.ndarray:
    """Difference of Gaussians via separable box-blur approximation (numpy only)."""
    def blur(x, sigma):
        k = max(1, int(round(sigma * 2.5)) | 1)
        ker = np.exp(-0.5 * ((np.arange(k) - k // 2) / max(sigma, 1e-6)) ** 2)
        ker /= ker.sum()
        out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 0, x)
        return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, out)
    return blur(img, s1) - blur(img, s2)


@dataclass
class Detections:
    """Detected sources: centre, size and score."""
    frame: pd.DataFrame

    def __len__(self):
        return len(self.frame)

    def boxes(self) -> np.ndarray:
        if not len(self.frame):
            return np.zeros((0, 4))
        return self.frame[["cx", "cy", "w", "h"]].values.astype(float)


def dog_detect(img: np.ndarray, threshold: float | None = None, sigmas=(1.2, 2.4, 4.0),
               min_separation: float = 6.0, max_detections: int = 200,
               nsigma: float = 7.0, reject_dipoles: bool = True,
               noise_map: np.ndarray | None = None) -> Detections:
    """Multi-scale blob detection with sub-pixel centroids.

    ``threshold`` defaults to a robust ``median + nsigma * MAD`` of the DoG response, so it
    adapts to the noise level of whatever image it is given.

    ``reject_dipoles`` removes the dominant false positive on a difference image.  A bright
    star that is misregistered by even a tenth of a pixel does not cancel: it leaves a
    positive and a negative lobe *side by side*.  A genuine mover also leaves a negative
    lobe, but at its previous position, which is many pixels away.  So a candidate with a
    strong negative pixel within a couple of PSF widths is a registration artefact, not an
    object.
    """
    rows = []
    for s in sigmas:
        resp = _dog(img, s, s * 1.7)
        med = np.median(resp)
        if threshold is not None:
            snr = resp - threshold
            hit = resp > threshold
        else:
            if noise_map is not None:
                # propagate the pixel noise through the DoG kernel, then self-calibrate so
                # that quiet sky reads exactly one sigma
                sm = np.sqrt(np.maximum(_blur_sq(noise_map ** 2, s), 1e-12))
                ratio = np.abs(resp - med) / sm
                sm = sm * max(float(np.median(ratio)) / 0.6745, 1e-9)
            else:
                sm = _local_sigma(resp)
            snr = (resp - med) / np.maximum(sm, 1e-9)
            hit = snr > nsigma
        ys, xs = np.where(hit)
        if ys.size == 0:
            continue
        order = np.argsort(-snr[ys, xs])[:max_detections * 4]
        for y, x in zip(ys[order], xs[order]):
            rows.append({"cx": float(x), "cy": float(y), "score": float(snr[y, x]),
                         "sigma": s, "w": 6.0 * s * 0.5, "h": 6.0 * s * 0.5})
    if not rows:
        return Detections(pd.DataFrame(columns=["cx", "cy", "score", "sigma", "w", "h"]))

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    # greedy non-maximum suppression on centre distance
    keep = []
    for r in df.itertuples():
        if all((r.cx - k[0]) ** 2 + (r.cy - k[1]) ** 2 > min_separation ** 2 for k in keep):
            keep.append((r.cx, r.cy, r.score, r.sigma, r.w, r.h))
        if len(keep) >= max_detections:
            break
    out = pd.DataFrame(keep, columns=["cx", "cy", "score", "sigma", "w", "h"])

    # refine each centre with an intensity-weighted centroid in a small window
    cx, cy = [], []
    for r in out.itertuples():
        x0, x1 = int(max(0, r.cx - 3)), int(min(img.shape[1], r.cx + 4))
        y0, y1 = int(max(0, r.cy - 3)), int(min(img.shape[0], r.cy + 4))
        patch = np.clip(img[y0:y1, x0:x1], 0, None)
        s = patch.sum()
        if s <= 0:
            cx.append(r.cx); cy.append(r.cy); continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx.append(float((patch * xx).sum() / s))
        cy.append(float((patch * yy).sum() / s))
    out["cx"], out["cy"] = cx, cy

    if reject_dipoles and len(out):
        keep_mask = []
        noise = 1.4826 * np.median(np.abs(img - np.median(img)))
        for r in out.itertuples():
            rad = int(max(2, round(2.0 * r.sigma)))
            x0, x1 = int(max(0, r.cx - rad)), int(min(img.shape[1], r.cx + rad + 1))
            y0, y1 = int(max(0, r.cy - rad)), int(min(img.shape[0], r.cy + rad + 1))
            local = img[y0:y1, x0:x1]
            keep_mask.append(local.min() > -3.0 * max(noise, 1e-9))
        out = out[np.array(keep_mask)].reset_index(drop=True)
    return Detections(out)


def score_detections(det: Detections, truth: pd.DataFrame, tol_px: float = 3.0,
                     cols=("cx", "cy")) -> dict:
    """Greedy nearest-neighbour matching, then precision / recall / F1.

    A prediction matches a truth object if their centres are within ``tol_px``.  Each truth
    object can be matched at most once, so duplicate detections count as false positives.
    """
    p = det.boxes()
    cols = list(cols) if all(c in truth.columns for c in cols) else ["cx", "cy"]
    t = truth[cols].values.astype(float) if len(truth) else np.zeros((0, 2))
    used = np.zeros(len(t), bool)
    tp, errs = 0, []
    order = np.argsort(-det.frame.score.values) if len(det.frame) else []
    for i in order:
        if not len(t):
            break
        d = np.hypot(t[:, 0] - p[i, 0], t[:, 1] - p[i, 1])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_px:
            used[j] = True
            tp += 1
            errs.append(float(d[j]))
    fp, fn = len(p) - tp, len(t) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"n_truth": len(t), "n_pred": len(p), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "mean_position_error_px": round(float(np.mean(errs)), 3) if errs else float("nan"),
            "tol_px": tol_px}


def single_frame_baseline(scene, threshold: float | None = None) -> dict:
    """Detect on the RAW frame — the control that shows the problem is ill-posed.

    Stars outnumber and outshine the debris, so this returns many detections and almost no
    real ones no matter where the threshold is set.
    """
    det = dog_detect(scene.frame_b - np.median(scene.frame_b), threshold)
    return score_detections(det, scene.truth, cols=("cx_b", "cy_b"))


def pair_detection(scene, threshold: float | None = None) -> dict:
    """Detect on the registered difference — the same detector, a solvable input."""
    diff, shift = register_and_difference(scene.frame_a, scene.frame_b)
    det = dog_detect(diff, threshold,
                     noise_map=photon_noise_map(scene.frame_a, scene.frame_b))
    out = score_detections(det, scene.truth)
    # the correction should be minus the jitter that was applied when the scene was built
    out["shift_error_px"] = round(float(np.hypot(shift[0] + scene.jitter[0],
                                                shift[1] + scene.jitter[1])), 4)
    return out


def evaluate_dataset(ds, threshold: float | None = None) -> pd.DataFrame:
    """Run both detectors over every scene and return one row per scene."""
    rows = []
    for i, sc in enumerate(ds.scenes):
        s = single_frame_baseline(sc, threshold)
        p = pair_detection(sc, threshold)
        rows.append({"scene": i, "n_truth": s["n_truth"],
                     "single_pred": s["n_pred"], "single_tp": s["tp"], "single_f1": s["f1"],
                     "pair_pred": p["n_pred"], "pair_tp": p["tp"], "pair_f1": p["f1"],
                     "pair_precision": p["precision"], "pair_recall": p["recall"],
                     "pos_err_px": p["mean_position_error_px"],
                     "shift_err_px": p["shift_error_px"]})
    return pd.DataFrame(rows)


def aggregate(scores: pd.DataFrame) -> pd.DataFrame:
    """Pool the per-scene counts into one precision / recall / F1 per detector."""
    rows = []
    for tag in ("single", "pair"):
        tp = int(scores[f"{tag}_tp"].sum())
        pred = int(scores[f"{tag}_pred"].sum())
        truth = int(scores.n_truth.sum())
        prec, rec = tp / max(pred, 1), tp / max(truth, 1)
        rows.append({"detector": "single frame" if tag == "single" else "registered difference",
                     "truth": truth, "predicted": pred, "true_positives": tp,
                     "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4)})
    return pd.DataFrame(rows)


def check_shift_sign(size: int = 256, dy: float = 3.4, dx: float = -2.1) -> dict:
    """Shift a random field by a known amount and confirm registration undoes it.

    This is the guard for the sign convention in :func:`estimate_shift`.  It fails loudly if
    the sign flips, which a detection F1 score does not when the jitter is small.
    """
    from .scene import _shift
    rng = np.random.default_rng(0)
    a = rng.normal(100.0, 10.0, (size, size))
    a = _dog(a, 1.0, 3.0) * 50.0 + 100.0            # give it structure to correlate on
    b = _shift(a, dy, dx)
    est = estimate_shift(a, b)
    residual = estimate_shift(a, apply_shift(b, *est))
    return {"applied": (dy, dx), "estimated_correction": (round(est[0], 4), round(est[1], 4)),
            "expected_correction": (-dy, -dx),
            "correction_error_px": round(float(np.hypot(est[0] + dy, est[1] + dx)), 4),
            "residual_after_registration_px": round(float(np.hypot(*residual)), 4)}
