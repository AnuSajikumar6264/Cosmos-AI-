"""FEATURE 1 (b) — anomaly detection and subsystem health scoring.

The monitor learns each channel's normal behaviour from an early "healthy" window, then
scores every later sample.  Two detectors are available and share one interface:

``ewma``    robust z-score per channel, smoothed with an exponentially weighted mean.
            Pure numpy, deterministic, no training beyond the baseline statistics.
``iforest`` scikit-learn IsolationForest over all channels jointly (used when available).

    hm = HealthMonitor().fit(ts, train_frac=0.4)
    res = hm.score(ts)
    res.health_at(-1).worst()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .telemetry import SUBSYSTEMS, TelemetrySeries

#: health bands, applied to the worst subsystem degradation
BANDS = [(0.20, "NOMINAL"), (0.45, "WATCH"), (0.70, "DEGRADED"), (1.01, "CRITICAL")]


def band_of(x: float) -> str:
    for hi, name in BANDS:
        if x < hi:
            return name
    return "CRITICAL"


@dataclass
class SubsystemHealth:
    """Degradation in ``[0, 1]`` for each of the eight subsystems."""
    values: dict

    def __post_init__(self):
        self.values = {k: float(np.clip(self.values.get(k, 0.0), 0.0, 1.0))
                       for k in SUBSYSTEMS}

    def __getitem__(self, k):
        return self.values[k]

    def worst(self):
        k = max(self.values, key=self.values.get)
        return k, self.values[k]

    def mean(self) -> float:
        return float(np.mean(list(self.values.values())))

    def band(self) -> str:
        return band_of(self.worst()[1])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"subsystem": list(self.values),
                            "degradation": [round(v, 3) for v in self.values.values()],
                            "band": [band_of(v) for v in self.values.values()]})

    @classmethod
    def nominal(cls, level: float = 0.03):
        return cls({k: level for k in SUBSYSTEMS})


@dataclass
class MonitorResult:
    """Per-sample anomaly score plus per-sample, per-subsystem degradation."""
    t_s: np.ndarray
    anomaly: np.ndarray                     # 0..1 aggregate score
    degradation: pd.DataFrame               # columns = subsystems, rows = samples
    z: pd.DataFrame                         # per-channel robust z-scores
    method: str = "ewma"
    train_n: int = 0
    detected_index: int = -1                # first sustained alarm
    threshold: float = 0.5

    def health_at(self, i: int = -1) -> SubsystemHealth:
        return SubsystemHealth(self.degradation.iloc[i].to_dict())

    def frame(self) -> pd.DataFrame:
        out = self.degradation.copy()
        out.insert(0, "anomaly_score", self.anomaly)
        out.insert(0, "t_s", self.t_s)
        return out

    def summary(self) -> dict:
        sub, val = self.health_at(-1).worst()
        return {"method": self.method, "train_samples": self.train_n,
                "final_anomaly_score": round(float(self.anomaly[-1]), 3),
                "worst_subsystem": sub, "worst_degradation": round(val, 3),
                "band": self.health_at(-1).band(),
                "first_alarm_index": int(self.detected_index)}


def _robust_stats(a: np.ndarray):
    """Median and a sigma estimate from the MAD, which ignores the fault tail."""
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0)
    sig = np.maximum(1.4826 * mad, 1e-9)
    return med, sig


def _ewma(a: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out = np.empty_like(a)
    acc = a[0]
    for i, v in enumerate(a):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


@dataclass
class HealthMonitor:
    """Fit on a healthy window, then score the whole run."""
    method: str = "ewma"
    z_saturate: float = 6.0          # |z| that counts as fully degraded
    alpha: float = 0.08              # EWMA smoothing
    alarm_threshold: float = 0.5
    alarm_persistence: int = 5       # samples above threshold before declaring an alarm
    _med: np.ndarray = field(default=None, repr=False)
    _sig: np.ndarray = field(default=None, repr=False)
    _cols: list = field(default_factory=list, repr=False)
    _iforest: object = field(default=None, repr=False)
    train_n: int = 0

    def fit(self, ts: TelemetrySeries, train_frac: float = 0.4) -> "HealthMonitor":
        n = max(20, int(round(train_frac * ts.n)))
        if ts.fault_onset_index > 0:
            n = min(n, ts.fault_onset_index)          # never train on faulty samples
        self._cols = ts.channels
        train = ts.frame[self._cols].values[:n]
        self._med, self._sig = _robust_stats(train)
        self.train_n = n
        if self.method == "iforest":
            try:
                from sklearn.ensemble import IsolationForest
                self._iforest = IsolationForest(n_estimators=200, contamination=0.02,
                                                random_state=0).fit(
                    (train - self._med) / self._sig)
            except Exception:
                self.method = "ewma"                  # graceful fallback
                self._iforest = None
        return self

    def score(self, ts: TelemetrySeries) -> MonitorResult:
        if self._med is None:
            self.fit(ts)
        x = ts.frame[self._cols].values
        z = (x - self._med) / self._sig
        zdf = pd.DataFrame(z, columns=self._cols)

        # per-subsystem degradation: worst channel, saturated and smoothed
        deg = {}
        for sub, cols in ts.channel_map.items():
            worst = np.max(np.abs(zdf[cols].values), axis=1)
            deg[sub] = np.clip(_ewma(worst, self.alpha) / self.z_saturate, 0.0, 1.0)
        degdf = pd.DataFrame(deg)[SUBSYSTEMS]

        if self.method == "iforest" and self._iforest is not None:
            raw = -self._iforest.score_samples(z)            # higher = more anomalous
            base = np.median(raw[:self.train_n])
            spread = np.maximum(np.std(raw[:self.train_n]), 1e-9)
            anomaly = np.clip((raw - base) / (6.0 * spread), 0.0, 1.0)
            anomaly = _ewma(anomaly, self.alpha)
        else:
            anomaly = np.clip(_ewma(np.max(np.abs(z), axis=1), self.alpha) / self.z_saturate,
                              0.0, 1.0)

        hot = anomaly > self.alarm_threshold
        det = -1
        run = 0
        for i, h in enumerate(hot):
            run = run + 1 if h else 0
            if run >= self.alarm_persistence:
                det = i - self.alarm_persistence + 1
                break
        return MonitorResult(ts.frame["t_s"].values, anomaly, degdf, zdf, self.method,
                             self.train_n, det, self.alarm_threshold)


def detection_delay(ts: TelemetrySeries, res: MonitorResult) -> dict:
    """How long after fault onset the monitor raised a sustained alarm."""
    if ts.fault_onset_index < 0:
        return {"fault": False, "false_alarm": res.detected_index >= 0}
    if res.detected_index < 0:
        return {"fault": True, "detected": False}
    d = res.detected_index - ts.fault_onset_index
    return {"fault": True, "detected": True, "onset_index": ts.fault_onset_index,
            "alarm_index": int(res.detected_index), "delay_samples": int(d),
            "delay_minutes": round(d * ts.dt_s / 60.0, 1),
            "before_onset": bool(d < 0)}
