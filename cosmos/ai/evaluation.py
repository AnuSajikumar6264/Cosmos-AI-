"""Metrics shared by every COSMOS-AI study, each written to be checked against a hand case.

classification   :func:`classification_metrics` — the OPSSAT-AD set (accuracy, precision,
                 recall, F1, MCC, AUC-PR, AUC-ROC), plus false-alarm rate
calibration      :func:`expected_calibration_error`, :func:`brier_score`, reliability bins
uncertainty      :func:`bootstrap_ci` (i.i.d. or moving-block), :func:`paired_bootstrap_diff`
survival         :func:`harrell_c`, :func:`ipcw_brier` — censoring-aware; ``lifelines`` is not
                 used because it pins pandas < 3, which the rest of COSMOS relies on
efficiency       :func:`measure_inference` — latency, throughput, peak device memory
"""
from __future__ import annotations

import time

import numpy as np


# ============================================================================ classification
def classification_metrics(y_true, y_pred, y_score=None, digits: int | None = None) -> dict:
    """Binary metrics with anomaly = 1 as the positive class (the OPSSAT-AD convention)."""
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 matthews_corrcoef, precision_score, recall_score,
                                 roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, zero_division=0),
           "recall": recall_score(y_true, y_pred, zero_division=0),
           "f1": f1_score(y_true, y_pred, zero_division=0),
           "mcc": matthews_corrcoef(y_true, y_pred),
           "false_alarm_rate": fp / max(fp + tn, 1)}
    if y_score is not None:
        s = np.asarray(y_score, float)
        out["auc_pr"] = average_precision_score(y_true, s)
        out["auc_roc"] = roc_auc_score(y_true, s)
    out = {k: float(v) for k, v in out.items()}
    if digits is not None:
        out = {k: round(v, digits) for k, v in out.items()}
    return out


# ============================================================================ calibration
def brier_score(y_true, p) -> float:
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    return float(np.mean((p - y_true) ** 2))


def reliability_bins(y_true, p, n_bins: int = 10) -> dict:
    """Equal-width bins on [0, 1]: mean confidence, observed frequency and count per bin."""
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf = np.array([p[b == k].mean() if (b == k).any() else np.nan for k in range(n_bins)])
    freq = np.array([y_true[b == k].mean() if (b == k).any() else np.nan
                     for k in range(n_bins)])
    cnt = np.array([(b == k).sum() for k in range(n_bins)])
    return {"edges": edges, "confidence": conf, "frequency": freq, "count": cnt}


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """ECE for a binary probability: sum_k (n_k / n) |freq_k - conf_k|."""
    r = reliability_bins(y_true, p, n_bins)
    m = r["count"] > 0
    return float(np.sum(r["count"][m] / r["count"].sum()
                        * np.abs(r["frequency"][m] - r["confidence"][m])))


# ============================================================================ bootstrap
def bootstrap_ci(stat, *arrays, n_boot: int = 2000, alpha: float = 0.05, block: int = 1,
                 seed: int = 0) -> tuple:
    """Percentile CI of ``stat(*arrays)``; ``block > 1`` resamples contiguous blocks.

    Moving blocks keep the serial dependence of time series inside each resample, which
    resampling single points destroys (and which makes i.i.d. CIs too narrow).
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    point = float(stat(*arrays))
    vals = []
    for _ in range(n_boot):
        if block <= 1:
            idx = rng.integers(0, n, n)
        else:
            starts = rng.integers(0, max(n - block + 1, 1), int(np.ceil(n / block)))
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        try:
            vals.append(float(stat(*[a[idx] for a in arrays])))
        except ValueError:                                   # e.g. one class only
            continue
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(stat, y, a, b, n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 0) -> dict:
    """CI of ``stat(y, a) - stat(y, b)`` resampling the SAME rows for both models."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    n = len(y)
    d0 = float(stat(y, a) - stat(y, b))
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            ds.append(float(stat(y[i], a[i]) - stat(y[i], b[i])))
        except ValueError:
            continue
    ds = np.asarray(ds)
    lo, hi = np.percentile(ds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": d0, "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))}


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict of named tests."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    out, run = {}, 0.0
    for k, nme in enumerate(names):
        run = max(run, min(1.0, (m - k) * pvalues[nme]))
        out[nme] = run
    return out


# ============================================================================ survival
def harrell_c(time, event, risk) -> float:
    """Harrell's concordance: among comparable pairs, how often higher risk fails first.

    A pair (i, j) is comparable when the earlier time is an observed event.  Ties in risk
    count one half.  ``risk`` is "higher = sooner"; pass ``-predicted_time`` for times.
    """
    t = np.asarray(time, float)
    e = np.asarray(event, bool)
    r = np.asarray(risk, float)
    order = np.argsort(t, kind="mergesort")
    t, e, r = t[order], e[order], r[order]
    num = den = 0.0
    for i in np.flatnonzero(e):
        later = t > t[i]
        if not later.any():
            continue
        den += later.sum()
        num += (r[i] > r[later]).sum() + 0.5 * (r[i] == r[later]).sum()
    return float(num / den) if den else float("nan")


def km_censoring(time, event):
    """Kaplan-Meier estimate of the CENSORING survival G(t), as a step function."""
    t = np.asarray(time, float)
    c = ~np.asarray(event, bool)                    # censoring is the "event" here
    ut = np.unique(t)
    surv, s = [], 1.0
    for u in ut:
        at_risk = (t >= u).sum()
        d = (c & (t == u)).sum()
        s *= 1.0 - d / max(at_risk, 1)
        surv.append(s)
    ut, surv = np.asarray(ut), np.asarray(surv)

    def G(x):
        x = np.atleast_1d(np.asarray(x, float))
        k = np.searchsorted(ut, x, side="right") - 1
        return np.where(k >= 0, surv[np.clip(k, 0, None)], 1.0)
    return G


def ipcw_brier(time, event, surv_prob_at_h, horizon: float, G=None) -> float:
    """Inverse-probability-of-censoring-weighted Brier score at one horizon (Graf et al.).

    ``surv_prob_at_h`` is the predicted P(T > horizon).  Subjects censored before the horizon
    contribute nothing; the rest are weighted by 1 / G, the censoring survival.
    """
    t = np.asarray(time, float)
    e = np.asarray(event, bool)
    s = np.asarray(surv_prob_at_h, float)
    G = G or km_censoring(t, e)
    died = (t <= horizon) & e
    alive = t > horizon
    w_died = 1.0 / np.maximum(G(t - 1e-9), 1e-9)
    w_alive = 1.0 / np.maximum(G(np.full_like(t, horizon)), 1e-9)
    loss = np.where(died, w_died * s ** 2, 0.0) + np.where(alive, w_alive * (1 - s) ** 2, 0.0)
    return float(loss.mean())


# ============================================================================ efficiency
def measure_inference(fn, batch, device: str = "cpu", warmup: int = 3, repeats: int = 20,
                      n_items: int | None = None) -> dict:
    """Median latency per call, throughput (items/s) and peak device memory of ``fn(batch)``."""
    import torch
    sync = (torch.xpu.synchronize if device.startswith("xpu") else
            (lambda: None))
    if device.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            fn(batch)
        sync()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(batch)
            sync()
            times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    n = n_items or (len(batch) if hasattr(batch, "__len__") else 1)
    out = {"latency_ms_per_batch": 1e3 * med, "batch_items": int(n),
           "latency_ms_per_item": 1e3 * med / max(n, 1), "throughput_items_s": n / med}
    if device.startswith("xpu"):
        out["peak_memory_mb"] = torch.xpu.max_memory_allocated(device) / 2**20
    return out
