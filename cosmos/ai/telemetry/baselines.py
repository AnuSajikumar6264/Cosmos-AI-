"""Non-deep baselines for ESA-ADB on the official protocol, producing per-channel binary alarms.

``global_std``      exact port of the benchmark's ``TimeEval-algorithms/std`` (mean and std of
                    each channel's NOMINAL training samples; alarm when |x - mean| > tol * std;
                    constant channels get std = 1).  Running it is how the COSMOS-AI
                    preprocessing + metric adapter is checked against DMLR Table 4.
``out_of_limits``   the operational status quo: alarm when a sample leaves the [min, max] range
                    of the channel's NOMINAL training samples (no free parameter).  Brancato et
                    al. (AST 2026, doi:10.1016/j.ast.2026.113957) use an out-of-limit detector
                    as their reference; their exact limits are not public, so this is ours.
``cosmos_monitor``  the existing COSMOS real-data monitor (:mod:`cosmos.health.realdata`):
                    AR(8) forecast residuals, 30-day trailing robust baseline, per-channel
                    nominal calibration, EWMA, alarm at 0.5 held for 5 samples.  Here it is fitted
                    on the official TRAINING half (not COSMOS's own first 20 %) and run online
                    over the test half; per-channel alarms are OR-ed into the global alarm, as
                    the official pipeline does for every algorithm.
"""
from __future__ import annotations

import numpy as np

from ...health import realdata as R
from ..data.preprocess_telemetry import ESASplit


def global_std(train: ESASplit, test: ESASplit, channels: list, tol: float = 3.0):
    """Official GlobalSTD; returns ``(alarms (T, C) uint8, means, stds)``."""
    jtr, jte = train.channel_index(channels), test.channel_index(channels)
    means, stds = np.zeros(len(channels)), np.ones(len(channels))
    for k, (a, b) in enumerate(zip(jtr, jte)):
        x = np.asarray(train.X[:, a], np.float32)
        nom = np.asarray(train.L[:, a]) == 0
        # the official code computes in float32 via numpy; keep the same precision
        means[k] = np.mean(x[nom])
        sd = np.std(x[nom])
        stds[k] = 1.0 if sd == 0 else sd
    X = np.asarray(test.X[:, jte], np.float32)
    alarms = ((X > means + tol * stds) | (X < means - tol * stds)).astype(np.uint8)
    return alarms, means, stds


def out_of_limits(train: ESASplit, test: ESASplit, channels: list, nominal_only: bool = True):
    """Alarm outside each channel's training [min, max]; returns ``(alarms (T, C) uint8, lo, hi)``."""
    jtr, jte = train.channel_index(channels), test.channel_index(channels)
    lo, hi = np.zeros(len(channels)), np.zeros(len(channels))
    for k, a in enumerate(jtr):
        x = np.asarray(train.X[:, a], np.float32)
        keep = (np.asarray(train.L[:, a]) == 0) if nominal_only else np.ones(len(x), bool)
        lo[k], hi[k] = np.min(x[keep]), np.max(x[keep])
    X = np.asarray(test.X[:, jte], np.float32)
    return ((X < lo) | (X > hi)).astype(np.uint8), lo, hi


def _per_channel_calibrated(z: np.ndarray, fit_rows: slice, q: float = 0.999,
                            alpha: float = 0.08) -> np.ndarray:
    """:func:`realdata.calibrated_scores`, applied to every channel on its own."""
    c = np.maximum(np.quantile(z[fit_rows], q, axis=0), 1.0)
    zt = z / c
    out = np.empty_like(zt)
    for j in range(zt.shape[1]):
        raw = R.ewma(zt[:, j], alpha)
        out[:, j] = np.clip(raw / (2.0 * max(np.quantile(raw[fit_rows], q), 1e-9)), 0, 1)
    return out


def cosmos_monitor(train: ESASplit, test: ESASplit, channels: list, residuals: bool = True,
                   threshold: float = 0.5, persistence: int = 5, q: float = 0.999,
                   window_days: float = 30.0, lag_days: float = 1.0) -> np.ndarray:
    """The COSMOS monitor on the official split; returns test alarms (T, C) uint8."""
    jtr, jte = train.channel_index(channels), test.channel_index(channels)
    # one contiguous record: the last training grid point coincides with the first test one
    Xtr = np.asarray(train.X[:-1, jtr], np.float64)
    Xte = np.asarray(test.X[:, jte], np.float64)
    X = np.vstack([Xtr, Xte])
    n_fit = Xtr.shape[0]
    cat = np.asarray(train.categorical)[jtr]
    cad = float(train.rule_s)
    z = np.zeros(X.shape)
    num = ~cat
    base = R.fit_baseline(X[:n_fit], cat)
    if cat.any():
        z[:, cat] = R.channel_z(X, base)[:, cat]
    S = R.ar_residuals(X[:, num], n_fit) if residuals else X[:, num]
    sbase = R.fit_baseline(S[:n_fit])
    med, sig = R.rolling_baseline(S, cad, window_days, lag_days, fit=sbase)
    z[:, num] = R.channel_z(S, sbase, 6.0, med, sig)
    warm = min(int((window_days + lag_days) * 86400 / cad), n_fit // 2)
    score = _per_channel_calibrated(z, slice(warm, n_fit), q)[n_fit:]
    alarms = np.zeros(score.shape, np.uint8)
    for j in range(score.shape[1]):
        for a, b in R.alarm_segments(score[:, j], threshold, persistence):
            alarms[a:b + 1, j] = 1
    return alarms
