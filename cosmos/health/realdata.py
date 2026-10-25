"""FEATURE 1 (d) — real spacecraft telemetry: NASA SMAP / MSL and the ESA Anomaly Dataset.

The simulator in :mod:`.telemetry` is where the monitor was designed.  This module is where
it is tested on telemetry nobody in this project wrote, with anomalies labelled by the
missions' own engineers.

NASA SMAP / MSL   Hundman et al., KDD 2018 (the "Telemanom" release).  82 channels — 55 from
                  the SMAP satellite, 27 from the Curiosity rover — each with a nominal
                  training split and a test split whose anomaly windows are labelled.
ESA-ADB           ESA Anomaly Dataset, Zenodo record 12528696.  Three missions, 224 channels
                  over 14, 3.5 and 9 years.  Every channel belongs to a named subsystem and
                  every labelled event has a category (Anomaly, Rare Event, Communication
                  Gap, Invalid Segment), which is what makes subsystem attribution testable.

The ESA archives store each channel as a pickled pandas DataFrame.  Unpickling can execute
code, so channels are read through :class:`_SafeUnpickler`, which reconstructs only the numpy
and pandas types those files actually contain, and are converted once by
:func:`prepare_esa_mission` into plain ``.npz`` arrays on a regular time grid.  Nothing else
in the project touches the pickles.

How the simulator's monitor had to change for real data — each step is ablated in
:func:`esa_ablation`, and none of them uses a label:

1. **Rolling baseline.**  Over years telemetry drifts; a baseline frozen at the start of a
   mission alarms on everything later.  Location and scale follow a trailing 30-day window.
2. **Nominal calibration.**  With 50-100 channels, the plain maximum |z| is above the alarm
   level nearly all the time.  Each channel is rescaled by its own 99.9th-percentile |z| in
   the fitting window, and the alarm level is the fitting window's own 99.9th percentile of
   the aggregate score — so 0.5 again means "at the edge of normal".
3. **Forecast residuals.**  Most labelled ESA events are contextual: values inside the normal
   range that are wrong for that moment.  A per-channel linear AR(8) forecaster, fitted on
   the fitting window only, supplies residuals and the monitor scores those.

    chans = load_smap_msl()
    per_channel, summary = evaluate_smap_msl(chans)
    prepare_esa_mission(1)                       # once, about 30 s
    result = evaluate_esa_mission(1)
    esa_ablation()                               # the four steps above, on all missions
"""
from __future__ import annotations

import ast
import io
import os
import pickle
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import feature_dir
from .monitor import _ewma

#: the ESA mission archives as published on Zenodo (record 12528696); checksums live in
#: :mod:`cosmos.datasets`, which downloads and verifies them
ESA_ARCHIVES = {1: "ESA-Mission1.zip", 2: "ESA-Mission2.zip", 3: "ESA-Mission3.zip"}

#: resampling cadence per mission.  Mission 2's events are short (median about an hour), so it
#: is kept finer; missions 1 and 3 span 14 and 9 years.
ESA_CADENCE = {1: "5min", 2: "2min", 3: "5min"}

#: labelled categories that are events to detect, and categories during which alarms are
#: not counted at all (the data there is missing or invalid, not anomalous)
EVENT_CATEGORIES = ("Anomaly", "Rare Event")
IGNORE_CATEGORIES = ("Communication Gap", "Invalid Segment")


# =============================================================== loading: NASA SMAP / MSL
@dataclass
class RealChannel:
    """One labelled telemetry channel with a nominal training split and a test split."""
    chan_id: str
    spacecraft: str
    train: np.ndarray                      # (n_train,) telemetry value
    test: np.ndarray                       # (n_test,)
    anomalies: list                        # [(start, end)] test indices, end inclusive
    classes: list = field(default_factory=list)


def smap_msl_dir() -> str:
    return feature_dir("health", "external", "smap_msl")


def load_smap_msl(root: str | None = None, spacecraft: str | None = None) -> list:
    """Read the Telemanom release: ``labeled_anomalies.csv`` plus ``train/`` and ``test/``.

    Column 0 of each array is the telemetry value; the remaining columns are one-hot encoded
    commands, which the monitor does not use.
    """
    root = root or smap_msl_dir()
    lab_path = os.path.join(root, "labeled_anomalies.csv")
    if not os.path.isfile(lab_path):
        raise FileNotFoundError(f"{lab_path} not found — run: python -m cosmos.datasets fetch "
                                "smap_msl")
    lab = pd.read_csv(lab_path)
    out = []
    for r in lab.itertuples():
        if spacecraft and r.spacecraft != spacecraft:
            continue
        tr = np.load(os.path.join(root, "train", f"{r.chan_id}.npy"), allow_pickle=False)
        te = np.load(os.path.join(root, "test", f"{r.chan_id}.npy"), allow_pickle=False)
        seqs = [tuple(int(v) for v in s) for s in ast.literal_eval(r.anomaly_sequences)]
        cls = [c.strip() for c in str(r._4).strip("[]").split(",")]
        out.append(RealChannel(r.chan_id, r.spacecraft, tr[:, 0].astype(float),
                               te[:, 0].astype(float), seqs, cls))
    return out


# ====================================================== loading: ESA Anomaly Dataset (ESA-ADB)
class _SafeUnpickler(pickle.Unpickler):
    """Reconstruct only the numpy / pandas objects the ESA channel files contain.

    The allow-list was built by reading the opcode stream of all 224 channel pickles (which
    executes nothing) and listing every global they reference.  Anything else is refused.
    """
    ALLOWED = {
        ("builtins", "slice"),
        ("numpy", "dtype"), ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.arrays.datetimes", "DatetimeArray"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
        ("pandas.core.indexes.datetimes", "DatetimeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        # the same classes under the public paths pandas 3 writes when re-pickling
        ("pandas", "DataFrame"), ("pandas.arrays", "DatetimeArray"),
        ("pandas.arrays", "StringArray"), ("pandas", "StringDtype"),
        ("pandas", "Index"), ("pandas", "DatetimeIndex"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a data file")
        return super().find_class(module, name)


def safe_read_pickle(raw: bytes) -> pd.DataFrame:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)      # numpy.core -> numpy._core
        return _SafeUnpickler(io.BytesIO(raw)).load()


def esa_dir() -> str:
    return feature_dir("health", "external", "esa_ad")


def _esa_zip(mission: int, root: str | None = None) -> zipfile.ZipFile:
    path = os.path.join(root or esa_dir(), ESA_ARCHIVES[mission])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found — run: python -m cosmos.datasets fetch "
                                "esa_ad")
    return zipfile.ZipFile(path)


def _esa_channel(args):
    """Worker: one channel pickle -> its median per bin.

    Bins are aligned to the Unix epoch, so every channel lands on the same grid without the
    workers having to agree on a start time first.
    """
    zip_path, member, cadence = args
    with zipfile.ZipFile(zip_path) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read(member)))
        df = safe_read_pickle(inner.read(inner.namelist()[0]))
    s = df.iloc[:, 0]
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    if s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        # status words "value_k": keep the state code, and the highest state seen in each
        # bin so that a rare state is not averaged away
        codes = s.astype(str).str.removeprefix("value_").astype("float64")
        return df.columns[0], codes.resample(cadence, origin="epoch").max(), True
    return (df.columns[0], s.astype("float64").resample(cadence, origin="epoch").median(),
            False)


def prepare_esa_mission(mission: int, cadence: str | None = None, root: str | None = None,
                        workers: int = 16, verbose: bool = True) -> str:
    """Convert one mission archive into ``prepared/mission{m}.npz`` plus ``events.csv``.

    The grid runs from the first to the last sample of the mission at ``cadence`` (median per
    bin; empty bins stay NaN).  Labels are written next to it with their categories joined in.
    """
    from concurrent.futures import ProcessPoolExecutor

    cadence = cadence or ESA_CADENCE[mission]
    z = _esa_zip(mission, root)
    base = f"ESA-Mission{mission}"
    channels = pd.read_csv(z.open(f"{base}/channels.csv"))
    labels = pd.read_csv(z.open(f"{base}/labels.csv"))
    types = pd.read_csv(z.open(f"{base}/anomaly_types.csv"))
    labels["StartTime"] = pd.to_datetime(labels.StartTime, utc=True).dt.tz_localize(None)
    labels["EndTime"] = pd.to_datetime(labels.EndTime, utc=True).dt.tz_localize(None)

    members = [f"{base}/channels/{c}.zip" for c in channels.Channel]
    zip_path = z.filename
    z.close()

    series, categorical = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = [(zip_path, m, cadence) for m in members]
        for i, (name, ser, is_cat) in enumerate(ex.map(_esa_channel, jobs)):
            series[name], categorical[name] = ser, is_cat
            if verbose and (i + 1) % 20 == 0:
                print(f"  mission {mission}: {i + 1}/{len(jobs)} channels")
    # one grid covering every channel and every label
    t0 = min([s.index.min() for s in series.values()] + [labels.StartTime.min()]).floor(cadence)
    t1 = max([s.index.max() for s in series.values()] + [labels.EndTime.max()]).ceil(cadence)
    grid = pd.date_range(t0, t1, freq=cadence)
    X = np.column_stack([series[c].reindex(grid).to_numpy(np.float32)
                         for c in channels.Channel])

    out_dir = os.path.join(esa_dir(), "prepared")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mission{mission}.npz")
    np.savez_compressed(path, X=X, t0=t0.to_datetime64(), cadence=cadence,
                        channels=channels.Channel.to_numpy(str),
                        subsystems=channels.Subsystem.to_numpy(str),
                        target=(channels.Target == "YES").to_numpy(),
                        categorical=np.array([categorical[c] for c in channels.Channel]))
    ev = labels.merge(types, on="ID", how="left")
    ev.to_csv(os.path.join(out_dir, f"mission{mission}_labels.csv"), index=False)
    if verbose:
        print(f"  wrote {path}  ({X.shape[0]:,} samples x {X.shape[1]} channels at {cadence})")
    return path


@dataclass
class ESAMission:
    """One prepared ESA mission on a regular grid."""
    mission: int
    t: pd.DatetimeIndex
    X: np.ndarray                          # (n, channels) float32, NaN where no data
    channels: np.ndarray
    subsystems: np.ndarray
    target: np.ndarray
    categorical: np.ndarray
    labels: pd.DataFrame                   # one row per (event, channel), with category

    @property
    def cadence_s(self) -> float:
        return float((self.t[1] - self.t[0]).total_seconds())

    def events(self) -> pd.DataFrame:
        """One row per labelled event: span, category and the subsystems it touches."""
        sub = dict(zip(self.channels, self.subsystems))
        lab = self.labels.assign(Subsystem=self.labels.Channel.map(sub))
        return (lab.groupby("ID")
                .agg(start=("StartTime", "min"), end=("EndTime", "max"),
                     category=("Category", "first"),
                     subsystems=("Subsystem", lambda v: frozenset(v.dropna())),
                     n_channels=("Channel", "nunique"))
                .reset_index())


def load_esa_mission(mission: int) -> ESAMission:
    """Load a mission written by :func:`prepare_esa_mission` (no pickles involved)."""
    d = os.path.join(esa_dir(), "prepared")
    path = os.path.join(d, f"mission{mission}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found — run prepare_esa_mission({mission})")
    z = np.load(path, allow_pickle=False)
    n = z["X"].shape[0]
    t = pd.date_range(pd.Timestamp(z["t0"].item()), periods=n, freq=str(z["cadence"]))
    labels = pd.read_csv(os.path.join(d, f"mission{mission}_labels.csv"),
                         parse_dates=["StartTime", "EndTime"])
    return ESAMission(mission, t, z["X"], z["channels"], z["subsystems"], z["target"],
                      z["categorical"], labels)


# ============================================================== scoring real telemetry
@dataclass
class Baseline:
    """Per-channel normal behaviour learned from a fitting window, without labels."""
    med: np.ndarray
    sig: np.ndarray
    categorical: np.ndarray
    common_states: list                    # per channel: states seen >= 1 % of the time


def fit_baseline(X_fit: np.ndarray, categorical: np.ndarray | None = None,
                 rare_state: float = 0.01) -> Baseline:
    """Robust location and scale per channel, as in :class:`HealthMonitor`.

    Real channels are often discrete or multimodal, where the MAD collapses to zero and every
    ordinary level change would read as infinite z.  The scale is therefore the larger of the
    MAD estimate and the Gaussian-equivalent scale of the central 98 % range; both agree on
    Gaussian noise.  Status channels instead learn which states are common.
    """
    X_fit = np.asarray(X_fit, float)
    if X_fit.ndim == 1:
        X_fit = X_fit[:, None]
    cat = np.zeros(X_fit.shape[1], bool) if categorical is None else np.asarray(categorical)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN channels
            med = np.nanmedian(X_fit, axis=0)
            mad = np.nanmedian(np.abs(X_fit - med), axis=0)
            q01, q99 = np.nanpercentile(X_fit, [1, 99], axis=0)
    sig = np.maximum.reduce([1.4826 * mad, (q99 - q01) / (2 * 2.3263),
                             1e-6 * (np.abs(med) + 1.0)])
    common = []
    for j in range(X_fit.shape[1]):
        if cat[j]:
            v = X_fit[:, j][~np.isnan(X_fit[:, j])]
            states, counts = np.unique(v, return_counts=True)
            common.append(set(states[counts >= rare_state * max(len(v), 1)].tolist()))
        else:
            common.append(set())
    return Baseline(np.nan_to_num(med), np.nan_to_num(sig, nan=1.0), cat, common)


def rolling_baseline(X: np.ndarray, cadence_s: float, window_days: float = 30.0,
                     lag_days: float = 1.0, fit: Baseline | None = None):
    """Location and scale that follow the mission: robust statistics of the trailing window.

    At each sample they come from the ``window_days`` before it, ending ``lag_days`` earlier
    so that an ongoing anomaly is not absorbed straight away.  Statistics are computed on
    hourly medians (they change slowly) and held within each hour.  Before the first window
    fills, the fitted baseline is used.  Returns per-sample ``(med, sig)`` shaped like ``X``.
    """
    import warnings
    n, c = X.shape
    per_h = max(int(round(3600.0 / cadence_s)), 1)
    nh = int(np.ceil(n / per_h))
    pad = np.full((nh * per_h - n, c), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN hours
        H = pd.DataFrame(np.nanmedian(np.vstack([X, pad]).reshape(nh, per_h, c), axis=1))
    w, lag = int(window_days * 24), int(lag_days * 24)
    roll = H.rolling(w, min_periods=w // 4)
    med = roll.median().shift(lag).to_numpy()
    q = {k: roll.quantile(k).shift(lag).to_numpy() for k in (0.01, 0.25, 0.75, 0.99)}
    sig = np.maximum.reduce([(q[0.75] - q[0.25]) / 1.349, (q[0.99] - q[0.01]) / (2 * 2.3263),
                             1e-6 * (np.abs(med) + 1.0)])
    if fit is not None:
        empty = np.isnan(med)
        med = np.where(empty, fit.med, med)
        sig = np.where(empty | np.isnan(sig), fit.sig, sig)
    return np.repeat(med, per_h, axis=0)[:n], np.repeat(sig, per_h, axis=0)[:n]


def channel_z(X: np.ndarray, base: Baseline, z_saturate: float = 6.0,
              med: np.ndarray | None = None, sig: np.ndarray | None = None) -> np.ndarray:
    """|z| per sample and channel; status channels score ``z_saturate`` in a rare state."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    z = np.abs((X - (base.med if med is None else med)) / (base.sig if sig is None else sig))
    for j in np.flatnonzero(base.categorical):
        z[:, j] = np.where(np.isin(X[:, j], list(base.common_states[j])), 0.0, z_saturate)
    z[np.isnan(X) | np.isnan(z)] = 0.0       # a missing sample is no evidence either way
    return z


def ar_residuals(X: np.ndarray, n_fit: int, order: int = 8, ridge: float = 1e-3) -> np.ndarray:
    """One-step residuals of a per-channel linear AR(``order``) forecaster.

    Fitted by ridge least squares on the first ``n_fit`` samples only, then run over the whole
    record.  Gaps are bridged by carrying the last value forward for the lags; residuals at
    missing samples are NaN.
    """
    X = np.asarray(X, float)
    n, c = X.shape
    E = np.full((n, c), np.nan)
    for j in range(c):
        x = X[:, j]
        if np.all(np.isnan(x)):
            continue
        xf = pd.Series(x).ffill().bfill().to_numpy()
        mu = np.nanmedian(xf[:n_fit])
        L = np.full((n, order), np.nan)
        for k in range(1, order + 1):
            L[k:, k - 1] = xf[:-k] - mu
        A = np.c_[L, np.ones(n)]
        ok = ~np.isnan(L).any(axis=1) & ~np.isnan(x)
        ok_fit = ok.copy()
        ok_fit[n_fit:] = False
        if ok_fit.sum() < 10 * order:
            continue
        w = np.linalg.solve(A[ok_fit].T @ A[ok_fit] + ridge * np.eye(order + 1),
                            A[ok_fit].T @ (xf[ok_fit] - mu))
        e = (xf - mu) - np.nan_to_num(A) @ w
        e[~ok] = np.nan
        E[:, j] = e
    return E


def ewma(a: np.ndarray, alpha: float) -> np.ndarray:
    """The monitor's EWMA, vectorised (identical to :func:`.monitor._ewma`)."""
    return pd.Series(np.asarray(a, float)).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def calibrated_scores(z: np.ndarray, fit_rows, groups: np.ndarray | None = None,
                      q: float = 0.999, alpha: float = 0.08):
    """Aggregate score and per-group degradation, calibrated to the fitting window.

    Each channel's |z| is divided by its own ``q``-quantile in ``fit_rows``; the EWMA of the
    channel maximum is then divided by twice ITS ``q``-quantile in ``fit_rows``.  A score of
    0.5 therefore marks the edge of normal for that mission, the alarm level the simulator's
    monitor already uses.  Groups (subsystems) are calibrated the same way, each against its
    own normal level, so a subsystem with many channels is not favoured.
    """
    c = np.maximum(np.quantile(z[fit_rows], q, axis=0), 1.0)
    zt = z / c

    def cal(cols):
        raw = ewma(zt[:, cols].max(axis=1), alpha)
        return np.clip(raw / (2.0 * max(np.quantile(raw[fit_rows], q), 1e-9)), 0.0, 1.0)

    agg = cal(np.arange(z.shape[1]))
    deg = pd.DataFrame({g: cal(np.flatnonzero(groups == g)) for g in pd.unique(groups)}
                       if groups is not None else {})
    return agg, deg


def score_matrix(z: np.ndarray, groups: np.ndarray | None = None, alpha: float = 0.08,
                 z_saturate: float = 6.0):
    """The simulator's monitor, uncalibrated: equation (1) applied to real channels."""
    agg = np.clip(ewma(z.max(axis=1), alpha) / z_saturate, 0.0, 1.0)
    deg = pd.DataFrame({g: np.clip(ewma(z[:, groups == g].max(axis=1), alpha) / z_saturate,
                                   0.0, 1.0) for g in pd.unique(groups)}
                       if groups is not None else {})
    return agg, deg


#: the variants of :func:`esa_ablation`, in the order each change was added
VARIANTS = {
    "simulator monitor":       dict(baseline="fixed", calibrate=False, residuals=False),
    "+ rolling baseline":      dict(baseline="rolling", calibrate=False, residuals=False),
    "+ nominal calibration":   dict(baseline="rolling", calibrate=True, residuals=False),
    "+ AR forecast residuals": dict(baseline="rolling", calibrate=True, residuals=True),
}
DEFAULT_VARIANT = "+ AR forecast residuals"


def score_real(X: np.ndarray, n_fit: int, cadence_s: float, categorical: np.ndarray,
               groups: np.ndarray | None, baseline: str = "rolling", calibrate: bool = True,
               residuals: bool = True, q: float = 0.999, alpha: float = 0.08,
               z_saturate: float = 6.0, window_days: float = 30.0, lag_days: float = 1.0):
    """Score a real telemetry matrix; returns ``(aggregate, degradation per group)``.

    ``n_fit`` samples at the start are the fitting window.  The defaults are the method used
    for every reported real-data result; the other settings reproduce the ablation steps.
    """
    X = np.asarray(X, float)
    cat = np.asarray(categorical, bool)
    num = ~cat
    z = np.zeros(X.shape)
    base = fit_baseline(X[:n_fit], cat)
    if cat.any():
        z[:, cat] = channel_z(X, base, z_saturate)[:, cat]
    S = ar_residuals(X[:, num], n_fit) if residuals else X[:, num]
    sbase = fit_baseline(S[:n_fit])
    if baseline == "rolling" and S.shape[0] > n_fit:
        med, sig = rolling_baseline(S, cadence_s, window_days, lag_days, fit=sbase)
        z[:, num] = channel_z(S, sbase, z_saturate, med, sig)
    else:
        z[:, num] = channel_z(S, sbase, z_saturate)
    if not calibrate:
        return score_matrix(z, groups, alpha, z_saturate)
    # calibrate only where the trailing window was already full
    warm = min(int((window_days + lag_days) * 86400 / cadence_s), n_fit // 2) \
        if baseline == "rolling" else 0
    return calibrated_scores(z, slice(warm, n_fit), groups, q, alpha)


# ================================================================= alarms and event scores
def alarm_segments(score: np.ndarray, threshold: float = 0.5, persistence: int = 5,
                   mask: np.ndarray | None = None) -> list:
    """Sustained alarms: runs above ``threshold`` lasting ``persistence`` samples or more.

    Returns ``[(start, end)]`` with ``end`` inclusive.  Samples where ``mask`` is False never
    alarm (communication gaps, invalid segments, the fitting window).
    """
    hot = np.asarray(score) > threshold
    if mask is not None:
        hot &= mask
    edges = np.flatnonzero(np.diff(np.r_[0, hot.astype(np.int8), 0]))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])
            if e - s >= persistence]


def episodes(alarms: list, merge_gap: int = 0) -> list:
    """Group alarms closer than ``merge_gap`` samples into one episode (a list of alarms).

    An operator sees a flickering excursion as one alarm, not many; grouping changes how
    alarms are COUNTED, never which samples alarmed.  The gap is fixed in advance (100
    samples) — tuning it upward games the metric, since one endless episode touches every
    event.
    """
    out = []
    for a in alarms:
        if out and a[0] - out[-1][-1][1] <= merge_gap:
            out[-1].append(a)
        else:
            out.append([a])
    return out


def event_metrics(alarms: list, events: list, merge_gap: int = 0) -> dict:
    """Event-wise scoring, the convention of the SMAP / MSL release.

    An event counts as detected if an alarming sample overlaps it; an alarm episode is a
    false positive if none of its alarms overlaps an event.  Precision is over episodes,
    recall over events.
    """
    def overlaps(a, b):
        return a[0] <= b[1] and b[0] <= a[1]
    hit_ev = [any(overlaps(a, e) for a in alarms) for e in events]
    eps = episodes(alarms, merge_gap)
    good = [any(overlaps(a, e) for a in ep for e in events) for ep in eps]
    tp_e, fn = int(sum(hit_ev)), int(len(events) - sum(hit_ev))
    tp_a, fp = int(sum(good)), int(len(eps) - sum(good))
    p = tp_a / max(tp_a + fp, 1)
    r = tp_e / max(tp_e + fn, 1)
    return {"events": len(events), "detected": tp_e, "missed": fn, "alarms": len(eps),
            "true_alarms": tp_a, "false_alarms": fp, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12), "hit": hit_ev}


MERGE_GAP = 100          # samples; fixed in advance, see :func:`episodes`


# ============================================================ evaluation: SMAP / MSL
def evaluate_smap_msl(channels: list | None = None, variant: str = DEFAULT_VARIANT,
                      threshold: float = 0.5, persistence: int = 5,
                      merge_gap: int = MERGE_GAP, q: float = 0.999):
    """Fit on each channel's nominal split, score its test split, score events.

    The train and test splits are joined so that the forecaster and the calibration see the
    nominal split as their fitting window.  These records are days long, so the rolling
    baseline does not apply; the variant's other settings do.  Returns ``(per_channel,
    summary)``; the summary pools events and alarms per spacecraft.
    """
    channels = channels if channels is not None else load_smap_msl()
    v = dict(VARIANTS[variant], baseline="fixed")
    rows = []
    for ch in channels:
        X = np.r_[ch.train, ch.test][:, None]
        nf = len(ch.train)
        agg, _ = score_real(X, nf, 60.0, np.array([False]), None, q=q, **v)
        m = event_metrics(alarm_segments(agg[nf:], threshold, persistence), ch.anomalies,
                          merge_gap)
        m.pop("hit")
        rows.append({"chan_id": ch.chan_id, "spacecraft": ch.spacecraft, **m})
    per = pd.DataFrame(rows)
    return per, _pool(per, "spacecraft")


def _pool(per: pd.DataFrame, by: str) -> pd.DataFrame:
    g = per.groupby(by)[["events", "detected", "missed", "alarms", "true_alarms",
                         "false_alarms"]].sum()
    g.loc["all"] = g.sum()
    g["precision"] = g.true_alarms / g.alarms.clip(lower=1)
    g["recall"] = g.detected / g.events.clip(lower=1)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall).clip(lower=1e-12)
    return g.round(3)


# ======================================================= evaluation: ESA Anomaly Dataset
def evaluate_esa_mission(mission, variant: str = DEFAULT_VARIANT, fit_frac: float = 0.2,
                         threshold: float = 0.5, persistence: int = 5,
                         merge_gap: int = MERGE_GAP, q: float = 0.999) -> dict:
    """Detection and subsystem attribution on one ESA mission, without using its labels.

    The first ``fit_frac`` of the mission is the fitting window and is not scored.  Labelled
    anomalies and rare events are the events to detect; samples inside communication gaps
    and invalid segments never alarm.

    Event-wise precision (episodes that touch an event) can be gamed by one alarm that never
    ends, so the alarm duty cycle and the time-wise precision (share of alarmed time inside
    events) are reported with it.  Attribution: for each detected event, the subsystem with
    the highest mean degradation over the event is compared with the subsystems of the
    channels the operators labelled, next to two baselines — always blaming the subsystem
    with most channels, and blaming one at random.
    """
    m = mission if isinstance(mission, ESAMission) else load_esa_mission(int(mission))
    n = len(m.t)
    n_fit = int(fit_frac * n)
    agg, deg = score_real(m.X, n_fit, m.cadence_s, m.categorical, m.subsystems, q=q,
                          **VARIANTS[variant])

    ev = m.events()
    ev["i0"] = np.clip(m.t.searchsorted(ev.start), 0, n - 1)
    ev["i1"] = np.clip(m.t.searchsorted(ev.end), 0, n - 1)
    usable = np.ones(n, bool)
    usable[:n_fit] = False
    for r in ev[ev.category.isin(IGNORE_CATEGORIES)].itertuples():
        usable[r.i0:r.i1 + 1] = False
    test_ev = ev[ev.category.isin(EVENT_CATEGORIES) & (ev.i0 >= n_fit)].reset_index(drop=True)

    alarms = alarm_segments(agg, threshold, persistence, usable)
    spans = list(zip(test_ev.i0, test_ev.i1))
    met = event_metrics(alarms, spans, merge_gap)
    test_ev["detected"] = met.pop("hit")

    in_event = np.zeros(n, bool)
    for a, b in spans:
        in_event[a:b + 1] = True
    alarmed = np.zeros(n, bool)
    for a, b in alarms:
        alarmed[a:b + 1] = True

    subs = list(deg.columns)
    D = deg.to_numpy()
    test_ev["blamed"] = [subs[int(np.argmax(D[r.i0:r.i1 + 1].mean(axis=0)))]
                         for r in test_ev.itertuples()]
    test_ev["attributed"] = [b in s for b, s in zip(test_ev.blamed, test_ev.subsystems)]
    det = test_ev[test_ev.detected]
    majority = pd.Series(m.subsystems).value_counts().index[0]
    years = (n - n_fit) * m.cadence_s / (365.25 * 86400)
    nan = float("nan")

    def share(mask_rows):
        return round(float(mask_rows.mean()), 3) if len(mask_rows) else nan

    return {"mission": m.mission, "variant": variant, "channels": int(m.X.shape[1]),
            "subsystems": len(subs), "test_years": round(years, 2), **met,
            "recall_anomaly": _rate(test_ev, "Anomaly"),
            "recall_rare_event": _rate(test_ev, "Rare Event"),
            "false_alarms_per_year": round(met["false_alarms"] / max(years, 1e-9), 2),
            "alarm_duty": round(float(alarmed[usable].mean()), 4),
            "event_time_share": round(float(in_event[usable].mean()), 4),
            "time_precision": round(float(in_event[alarmed].mean()), 3) if alarmed.any()
            else nan,
            "attribution": share(det.attributed),
            "attribution_majority": share(det.subsystems.apply(lambda v: majority in v)),
            "attribution_random": round(float(det.subsystems.apply(
                lambda v: len(v) / len(subs)).mean()), 3) if len(det) else nan,
            "events_table": test_ev}


def _rate(ev: pd.DataFrame, category: str) -> float:
    sel = ev[ev.category == category]
    return round(float(sel.detected.mean()), 3) if len(sel) else float("nan")


def evaluate_esa(missions=(1, 2, 3), **kw) -> pd.DataFrame:
    """All ESA missions in one table (the per-event tables are dropped)."""
    rows = []
    for mi in missions:
        r = evaluate_esa_mission(mi, **kw)
        r.pop("events_table")
        rows.append(r)
    return pd.DataFrame(rows).set_index("mission")


def esa_ablation(missions=(1, 2, 3)) -> pd.DataFrame:
    """Each change the real data forced, added one at a time (see the module docstring)."""
    cols = ["variant", "recall", "precision", "f1", "recall_anomaly", "false_alarms_per_year",
            "alarm_duty", "time_precision", "attribution", "attribution_majority"]
    frames = [evaluate_esa(missions, variant=v).reset_index()[["mission"] + cols]
              for v in VARIANTS]
    return pd.concat(frames, ignore_index=True)
