"""Event-level tables and the moving-block bootstrap re-implementation of corrected
event-wise F0.5 (validated against the official values in ``report.esa_bootstrap``).

Kept apart from :mod:`cosmos.ai.report` so that the pre-registered analysis depends on a
small, frozen file and not on the figure code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _event_table(mission, alarms_global, ts, lab):
    """Per test event (anomaly + rare): detected; FP runs; alarmed nominal time — per sample."""
    from .alarms import runs
    t = pd.DatetimeIndex(ts).as_unit("ns").asi8
    sel = lab[lab.Category.isin(["Anomaly", "Rare Event"])]
    ev = sel.groupby("ID").agg(s=("StartTime", "min"), e=("EndTime", "max"))
    ls = lab.StartTime.to_numpy().astype("datetime64[ns]").astype(np.int64)
    le = lab.EndTime.to_numpy().astype("datetime64[ns]").astype(np.int64)
    in_lab = np.zeros(len(t), bool)
    for a, b in zip(ls, le):
        in_lab[np.searchsorted(t, a):np.searchsorted(t, b, "right")] = True
    cs = np.r_[0, np.cumsum(alarms_global)]
    es = ev.s.to_numpy().astype("datetime64[ns]").astype(np.int64)
    ee = ev.e.to_numpy().astype("datetime64[ns]").astype(np.int64)
    i0, i1 = np.searchsorted(t, es), np.searchsorted(t, ee, "right")
    det = (cs[np.minimum(i1, len(t))] - cs[i0]) > 0
    s, e = runs(alarms_global)
    fp_start = [a for a, b in zip(s, e) if not in_lab[a:b].any()]
    return {"t": t, "ev_time": es, "ev_det": det, "fp_time": t[np.array(fp_start, int)]
            if fp_start else np.zeros(0, np.int64),
            "nominal": ~in_lab, "alarm_nominal": alarms_global.astype(bool) & ~in_lab}


def _ewf(tab, blocks, beta=0.5):
    det = tab["ev_det"][blocks["ev"]]
    tp, nev = det.sum(), len(det)
    fp = len(blocks["fp"])
    rec = tp / max(nev, 1)
    prec = tp / max(tp + fp, 1)
    nom = tab["nominal_blocks"][blocks["b"]].sum()
    fa = tab["alarm_nominal_blocks"][blocks["b"]].sum()
    prec *= 1 - fa / max(nom, 1)
    return (1 + beta ** 2) * prec * rec / max(beta ** 2 * prec + rec, 1e-12)


def _blockify(tab, block_days=30):
    t = tab["t"]
    bl = ((t - t[0]) // int(block_days * 86400e9)).astype(int)
    nb = bl.max() + 1
    tab["nominal_blocks"] = np.bincount(bl, weights=tab["nominal"], minlength=nb)
    tab["alarm_nominal_blocks"] = np.bincount(bl, weights=tab["alarm_nominal"], minlength=nb)
    tab["ev_block"] = ((tab["ev_time"] - t[0]) // int(block_days * 86400e9)).astype(int)
    tab["fp_block"] = ((tab["fp_time"] - t[0]) // int(block_days * 86400e9)).astype(int)
    tab["nb"] = nb
    return tab


def _sample(tab, idx):
    ev = np.concatenate([np.flatnonzero(tab["ev_block"] == b) for b in idx])
    fp = np.concatenate([np.flatnonzero(tab["fp_block"] == b) for b in idx])
    return {"ev": ev, "fp": fp, "b": idx}
