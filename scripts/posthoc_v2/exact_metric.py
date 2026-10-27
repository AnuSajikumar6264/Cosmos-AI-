"""P3: a block-decomposable form of the official ESA-ADB corrected event-wise F-score that is
exact on the full test half, and the block bootstrap built on it.

The official definition (cosmos/ai/telemetry/esa_metrics/ESA_ADB_metrics.py, ESAScores.score):

* predicted events are the alarm runs, as intervals [t_start, t_first_zero) (closed at the last
  sample when the run reaches the end of the record);
* an event (one ID, possibly several label rows) is a true positive if any predicted interval
  intersects any of its closed label intervals; otherwise it is a false negative;
* a predicted interval is a false positive if it intersects no label interval of any category;
* precision is multiplied by TNR = 1 - (alarm time inside nominal time) / (nominal time), in
  seconds, where nominal time is the full range minus the union of all label intervals.

Here every quantity is accumulated per 30-day block (events and false positives by start time,
nominal and alarmed-nominal seconds by clipping to the block), so the sum over all blocks
reproduces the official value and a resample of blocks gives a bootstrap replicate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cosmos.ai.telemetry.alarms import runs

DAY_NS = 86400 * 10 ** 9


def _ns(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x)).to_numpy().astype("datetime64[ns]").astype(np.int64)


def _union(starts: np.ndarray, ends: np.ndarray):
    """Union of closed intervals as disjoint sorted (start, end)."""
    if starts.size == 0:
        return starts, ends
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    us, ue = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= ue[-1]:
            ue[-1] = max(ue[-1], b)
        else:
            us.append(a)
            ue.append(b)
    return np.asarray(us, np.int64), np.asarray(ue, np.int64)


def _intersect(as_, ae, bs, be):
    """Intersection of two disjoint sorted interval sets (measure only matters)."""
    i = j = 0
    os_, oe = [], []
    while i < as_.size and j < bs.size:
        lo, hi = max(as_[i], bs[j]), min(ae[i], be[j])
        if lo < hi:
            os_.append(lo)
            oe.append(hi)
        if ae[i] < be[j]:
            i += 1
        else:
            j += 1
    return np.asarray(os_, np.int64), np.asarray(oe, np.int64)


def _per_block(s: np.ndarray, e: np.ndarray, t0: int, width: int, nb: int) -> np.ndarray:
    """Length (ns) of the union of disjoint intervals inside each block [t0 + b w, t0 + (b+1) w)."""
    out = np.zeros(nb, np.float64)
    for a, b in zip(s, e):
        ba, bb = int((a - t0) // width), int((b - t0) // width)
        for k in range(max(ba, 0), min(bb, nb - 1) + 1):
            lo, hi = max(a, t0 + k * width), min(b, t0 + (k + 1) * width)
            if hi > lo:
                out[k] += hi - lo
    return out


def event_table(g: np.ndarray, ts, lab: pd.DataFrame, cats=("Anomaly", "Rare Event"),
                block_days: int = 30) -> dict:
    """Per-block counts for one global alarm array ``g`` on timestamps ``ts``."""
    g = np.asarray(g, bool)
    t = pd.DatetimeIndex(ts).as_unit("ns").asi8
    n = t.size
    ls, le = _ns(lab.StartTime), _ns(lab.EndTime)
    full0, full1 = min(int(ls.min()), int(t[0])), max(int(le.max()), int(t[-1]))
    s, e = runs(g)
    ps = t[s]
    closed = e == n
    pe = np.where(closed, t[-1], t[np.minimum(e, n - 1)])
    # --- events: true positive if any predicted interval meets any of its label intervals
    sel = lab.Category.isin(cats).to_numpy()
    ids = lab.ID.to_numpy()[sel]
    gs, ge = ls[sel], le[sel]
    hit = np.zeros(gs.size, bool)
    if ps.size:
        j = np.searchsorted(ps, ge, side="right") - 1          # last run starting at or before ge
        ok = j >= 0
        jj = np.where(ok, j, 0)
        reach = np.where(closed[jj], pe[jj] >= gs, pe[jj] > gs)
        hit = ok & reach
    ev = pd.DataFrame({"ID": ids, "s": gs, "hit": hit}).groupby("ID").agg(s=("s", "min"), det=("hit", "any"))
    # --- false positives: runs that touch no label interval of any category
    us, ue = _union(ls, le)
    if ps.size and us.size:
        k = np.searchsorted(us, pe, side="right") - 1
        k_open = np.searchsorted(us, pe, side="left") - 1
        kk = np.where(closed, k, k_open)
        okk = kk >= 0
        touch = okk & (ue[np.where(okk, kk, 0)] >= ps)
    else:
        touch = np.zeros(ps.size, bool)
    fp_start = ps[~touch]
    # --- nominal and alarmed-nominal time per block (continuous time, ns)
    width = block_days * DAY_NS
    nb = int((full1 - full0) // width) + 1
    edges_s = np.array([full0], np.int64)
    edges_e = np.array([full1], np.int64)
    total = _per_block(edges_s, edges_e, full0, width, nb)
    lab_in = _per_block(*_intersect(us, ue, edges_s, edges_e), full0, width, nb)
    pred_s, pred_e = ps, pe
    alarm = _per_block(pred_s, pred_e, full0, width, nb)
    alarm_lab = _per_block(*_intersect(pred_s, pred_e, us, ue), full0, width, nb)
    tab = {"nb": nb, "nominal": total - lab_in, "alarm_nominal": alarm - alarm_lab,
           "ev_block": ((ev.s.to_numpy() - full0) // width).astype(int), "ev_det": ev.det.to_numpy(),
           "fp_block": ((fp_start - full0) // width).astype(int)}
    tab["ev_count"] = np.bincount(tab["ev_block"], minlength=nb)
    tab["tp_count"] = np.bincount(tab["ev_block"], weights=tab["ev_det"], minlength=nb)
    tab["fp_count"] = np.bincount(tab["fp_block"], minlength=nb)
    return tab


def f_beta(tab: dict, idx=None, beta: float = 0.5) -> float:
    """Corrected event-wise F_beta for blocks ``idx`` (all blocks once if None)."""
    if idx is None:
        idx = np.arange(tab["nb"])
    tp, nev, fp = tab["tp_count"][idx].sum(), tab["ev_count"][idx].sum(), tab["fp_count"][idx].sum()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    nom = tab["nominal"][idx].sum()
    prec *= 1 - tab["alarm_nominal"][idx].sum() / nom
    rec = tp / max(nev, 1)
    return float((1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))


def compare(tabs_a: list, tabs_b: list, rng, n_boot: int = 2000) -> dict:
    """Paired hierarchical block bootstrap with the draw order of the frozen ``oof_benchmark.compare``."""
    nb = tabs_a[0]["nb"]
    pa = float(np.mean([f_beta(t) for t in tabs_a]))
    pb = float(np.mean([f_beta(t) for t in tabs_b]))
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nb, nb)
        sa = rng.integers(0, len(tabs_a), len(tabs_a))
        sb = rng.integers(0, len(tabs_b), len(tabs_b))
        d[i] = (np.mean([f_beta(tabs_a[j], idx) for j in sa])
                - np.mean([f_beta(tabs_b[j], idx) for j in sb]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / n_boot)
    return {"A": pa, "B": pb, "diff": pa - pb, "ci95_low": float(lo), "ci95_high": float(hi),
            "p_boot": float(p), "n_boot": n_boot, "replicates": d}
