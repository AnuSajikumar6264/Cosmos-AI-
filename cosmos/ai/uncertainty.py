"""Distribution-free uncertainty shared by every COSMOS-AI model.

The re-entry footprint code (:mod:`cosmos.reentry.montecarlo`) already uses split conformal
regions for 2-D impact points; these are the same ideas for learned models.

conformal p-values     :func:`conformal_pvalues` — ``(1 + #{cal >= s}) / (n + 1)``, valid
                       (super-uniform) when the test score is exchangeable with the
                       calibration scores (Vovk et al. 2005; Bates et al., Ann. Statist. 2023)
multiplicity           :func:`bh_reject` (Benjamini-Hochberg per row) and :func:`simes`
                       (the matching global p-value: BH rejects something iff Simes <= q)
intervals              :func:`split_conformal_quantile`, :func:`cqr_interval` (Romano et al.
                       2019), :func:`interval_coverage`
selective prediction   :func:`risk_coverage` — risk as the least-confident cases are withheld

Guarantee classes (stated so that nothing is over-claimed): conformal p-values and intervals
are MARGINAL and need exchangeability.  Telemetry is serially dependent and drifts, so on
telemetry the guarantee is approximate; COSMOS-AI measures the realised rate instead of
assuming it.
"""
from __future__ import annotations

import numpy as np


def conformal_pvalues(cal_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Conformal p-values of ``scores`` against ascending-sorted calibration scores."""
    cal_sorted = np.asarray(cal_sorted)
    n = cal_sorted.size
    ge = n - np.searchsorted(cal_sorted, scores, side="left")    # #{cal >= s}
    return (1.0 + ge) / (n + 1.0)


def bh_reject(P: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg at level ``q`` along the last axis; returns the rejection mask."""
    P = np.asarray(P, float)
    m = P.shape[-1]
    order = np.argsort(P, axis=-1)
    ps = np.take_along_axis(P, order, axis=-1)
    thr = q * np.arange(1, m + 1) / m
    below = ps <= thr
    # largest k with p_(k) <= k q / m; reject the k smallest
    k = np.where(below.any(-1), m - np.argmax(below[..., ::-1], axis=-1), 0)
    rank = np.argsort(order, axis=-1)                              # inverse permutation
    return rank < k[..., None]


def simes(P: np.ndarray) -> np.ndarray:
    """Simes global p-value ``min_i m p_(i) / i`` along the last axis."""
    P = np.sort(np.asarray(P, float), axis=-1)
    m = P.shape[-1]
    return np.min(P * m / np.arange(1, m + 1), axis=-1)


def split_conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest calibration score (inf if n is too small)."""
    s = np.sort(np.asarray(cal_scores, float))
    n = s.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if k <= n else float("inf")


def cqr_interval(lo_cal, hi_cal, y_cal, lo, hi, alpha: float = 0.1):
    """Conformalised quantile regression: widen [lo, hi] by the calibrated margin."""
    e = np.maximum(np.asarray(lo_cal) - y_cal, np.asarray(y_cal) - hi_cal)
    qhat = split_conformal_quantile(e, alpha)
    return np.asarray(lo) - qhat, np.asarray(hi) + qhat, qhat


def interval_coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def risk_coverage(confidence: np.ndarray, correct: np.ndarray, n_points: int = 20) -> dict:
    """Selective prediction: error rate among the most confident fraction, per coverage."""
    order = np.argsort(-np.asarray(confidence, float))
    c = np.asarray(correct, float)[order]
    n = c.size
    cov = np.linspace(1.0 / n_points, 1.0, n_points)
    risk = np.array([1.0 - c[:max(1, int(round(f * n)))].mean() for f in cov])
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(
        np.trapz(risk, cov))
    return {"coverage": cov, "risk": risk, "aurc": aurc}
