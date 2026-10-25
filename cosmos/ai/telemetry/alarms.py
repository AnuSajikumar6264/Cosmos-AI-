"""From per-channel non-conformity scores to binary event alarms (the ESA-ADB output format).

Calibrators (each turns scores (T, C) into "hot" cells (T, C)):

``threshold``     per-channel percentile threshold on nominal calibration scores, no
                  multiplicity control across channels (the usual practice)
``conformal_bh``  per-channel conformal p-values against nominal calibration scores, then
                  Benjamini-Hochberg across channels at every time step (Bates et al. 2023)
``conformal_bh_rolling``  as above, but each channel's calibration pool is refreshed every
                  day from the trailing ``window_days`` of its own UNLABELLED test-period
                  scores, ``lag_days`` old (online recalibration for drift; Barber et al. 2023
                  argue for weighting recent data, this is the simplest form)

Event logic shared by all calibrators: a global alarm needs ``k`` consecutive hot steps on any
channel; alarm runs closer than ``merge`` steps are joined; within an alarm run, the channels
that were hot at any step are the reported (attributed) channels.

Selection: the calibrator's level and ``k`` are chosen on the calibration period with
cross-fitting (calibrate on one half, score the other with the official corrected
event-wise F0.5, both ways, average), so the test half is never used for any choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..uncertainty import bh_reject, conformal_pvalues


# ============================================================================ event logic
def runs(mask: np.ndarray):
    """``(starts, ends)`` of True runs (ends exclusive)."""
    m = np.asarray(mask, np.int8)
    d = np.diff(np.r_[0, m, 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def persist_and_merge(hot_any: np.ndarray, k: int, merge: int) -> np.ndarray:
    """Keep runs of at least ``k`` steps, then fill gaps shorter than ``merge`` steps."""
    s, e = runs(hot_any)
    keep = (e - s) >= k
    s, e = s[keep], e[keep]
    out = np.zeros(hot_any.shape[0], bool)
    if s.size == 0:
        return out
    # start of a kept run marks the alarm from where the run became "persistent"
    s = s + (k - 1)
    gs, ge = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a - ge[-1] < merge:
            ge[-1] = b
        else:
            gs.append(a)
            ge.append(b)
    for a, b in zip(gs, ge):
        out[a:b] = True
    return out


def attribute(alarm: np.ndarray, hot: np.ndarray, back: int = 0) -> np.ndarray:
    """Per-channel alarms: each global alarm run, for the channels hot inside it.

    ``back`` extends the look-up window before the run start (the ``k - 1`` hot steps that
    established persistence).
    """
    out = np.zeros(hot.shape, np.uint8)
    s, e = runs(alarm)
    if s.size == 0:
        return out
    cs = np.vstack([np.zeros((1, hot.shape[1]), np.int64), np.cumsum(hot, 0, dtype=np.int64)])
    for a, b in zip(s, e):
        chans = (cs[b] - cs[max(a - back, 0)]) > 0
        out[a:b, chans] = 1
    return out


# ============================================================================ calibrators
def _nominal_columns(S_cal: np.ndarray, L_cal: np.ndarray, stride: int = 1) -> list:
    """Sorted finite nominal calibration scores per channel."""
    out = []
    for c in range(S_cal.shape[1]):
        v = S_cal[::stride, c][(L_cal[::stride, c] == 0) & np.isfinite(S_cal[::stride, c])]
        out.append(np.sort(v.astype(np.float32)))
    return out


@dataclass
class Calibrator:
    kind: str = "conformal_bh"
    level: float = 1e-3            # q for BH, tail fraction a for thresholds
    k: int = 1
    merge: int = 120
    window_days: float = 30.0
    lag_days: float = 1.0
    rule_s: int = 30

    def fit(self, S_cal: np.ndarray, L_cal: np.ndarray) -> "Calibrator":
        self.cal_ = _nominal_columns(S_cal, L_cal)
        if self.kind == "threshold":
            self.tau_ = np.array([np.quantile(v, 1 - self.level) if self.level > 0
                                  else v.max() for v in self.cal_], np.float32)
        return self

    def pvalues(self, S: np.ndarray) -> np.ndarray:
        S = np.where(np.isfinite(S), S, -np.inf)
        P = np.ones(S.shape, np.float32)
        if self.kind == "conformal_bh_rolling":
            day = int(86400 / self.rule_s)
            w, lag = int(self.window_days * day), int(self.lag_days * day)
            for c in range(S.shape[1]):
                for a in range(0, S.shape[0], day):
                    b = min(a + day, S.shape[0])
                    lo, hi = a - lag - w, a - lag
                    if lo < 0:
                        pool = self.cal_[c]
                    else:
                        v = S[lo:hi:2, c]
                        pool = np.sort(v[np.isfinite(v)])
                    P[a:b, c] = conformal_pvalues(pool, S[a:b, c])
            return P
        for c in range(S.shape[1]):
            P[:, c] = conformal_pvalues(self.cal_[c], S[:, c])
        return P

    def hot(self, S: np.ndarray) -> np.ndarray:
        if self.kind == "threshold":
            return np.nan_to_num(S, nan=-np.inf) > self.tau_[None, :]
        return bh_reject(self.pvalues(S), self.level)

    def alarms(self, S: np.ndarray):
        """``(global alarm (T,) bool, per-channel alarms (T, C) uint8)``."""
        H = self.hot(S)
        g = persist_and_merge(H.any(1), self.k, self.merge)
        return g, attribute(g, H, back=self.k - 1)


#: selection grids (equal size for the two calibrator families)
GRIDS = {"threshold": [1e-3, 1e-4, 1e-5, 0.0],
         "conformal_bh": [1e-2, 1e-3, 3e-4, 1e-4],
         "conformal_bh_rolling": [1e-2, 1e-3, 3e-4, 1e-4]}
K_GRID = [1, 3, 10, 30]


def select_calibrator(kind: str, S_cal: np.ndarray, L_cal: np.ndarray, score_fn,
                      rule_s: int, merge: int, levels=None, ks=None) -> tuple:
    """Cross-fitted grid search on the calibration period.

    ``score_fn(alarms_global, alarms_channels, rows)`` must return the official corrected
    event-wise F0.5 for the calibration rows ``rows``.  Returns ``(best Calibrator fitted on
    the WHOLE calibration period, table of all grid points)``.
    """
    levels = levels or GRIDS[kind]
    ks = ks or K_GRID
    n = S_cal.shape[0]
    halves = [np.arange(0, n // 2), np.arange(n // 2, n)]
    rows = []
    base = "conformal_bh" if kind == "conformal_bh_rolling" else kind   # no test data here
    for lev in levels:
        for k in ks:
            f, frac = [], []
            for i in (0, 1):
                fit_rows, ev_rows = halves[1 - i], halves[i]
                c = Calibrator(base, lev, k, merge, rule_s=rule_s).fit(S_cal[fit_rows],
                                                                      L_cal[fit_rows])
                g, ch = c.alarms(S_cal[ev_rows])
                f.append(score_fn(g, ch, ev_rows))
                frac.append(float(g.mean()))
            rows.append({"kind": kind, "level": lev, "k": k, "cv_EW_F0.5": float(np.mean(f)),
                         "cv_alarm_fraction": float(np.mean(frac)),
                         "half_scores": f})
    rows.sort(key=lambda r: (-r["cv_EW_F0.5"], r["cv_alarm_fraction"]))
    best = rows[0]
    cal = Calibrator(kind, best["level"], best["k"], merge, rule_s=rule_s).fit(S_cal, L_cal)
    return cal, rows
