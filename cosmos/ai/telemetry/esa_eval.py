"""Official ESA-ADB scoring of binary per-channel alarms, as the benchmark's TimeEval fork does it.

    scores = official_scores(mission, alarms, channel_names)

What is reproduced from ``timeeval/core/experiments.py`` (commit 67194d3):

* labels: ``labels.csv`` joined with ``anomaly_types.csv``, kept only if the event lies
  inside the test range, then restricted to the scored (target) channels;
* global prediction = OR over channels, scored with ``ESAScores(beta=0.5)`` twice — for
  {Anomaly, Rare Event} and for {Anomaly} alone — and with the global ``ADTQC``;
* per-channel predictions scored with ``ChannelAwareFScore(beta=0.5)`` and the
  channel -> subsystem map from ``channels.csv``.

The metric code itself is the vendored, unmodified copy in :mod:`.esa_metrics`.  A binary
stream is handed over as its change points plus first and last sample, which yields exactly
the same intervals as the full 7-million-sample array (checked in the unit tests) and keeps
the official interval code fast.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from ..data.preprocess_telemetry import (ESASplit, load_channels_table, load_esa_split,
                                         official_test_labels)
from .esa_metrics import ADTQC, ChannelAwareFScore, ESAScores

CATEGORY_SETS = {"anomaly_and_rare": ["Rare Event", "Anomaly"], "anomaly": ["Anomaly"]}


def compress_binary(ts: pd.DatetimeIndex | np.ndarray, y: np.ndarray) -> np.ndarray:
    """``[(timestamp, value)]`` at the first sample, every change, and the last sample."""
    y = np.asarray(y).astype(np.uint8)
    keep = np.zeros(y.size, bool)
    keep[0] = keep[-1] = True
    keep[1:] |= y[1:] != y[:-1]
    t = pd.DatetimeIndex(ts)[keep]
    out = np.empty((keep.sum(), 2), dtype=object)
    out[:, 0] = list(t)
    out[:, 1] = y[keep].astype(int)
    return out


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):     # the official code prints lists
        return fn(*a, **kw)


def period_labels(mission: int, t0, t1) -> pd.DataFrame:
    """Labels with types for events lying entirely inside [t0, t1] (the TimeEval rule)."""
    import os
    from ..data.preprocess_telemetry import processed_dir
    lab = pd.read_csv(os.path.join(processed_dir(mission), "labels_with_types.csv"),
                      parse_dates=["StartTime", "EndTime"])
    lab = lab[(lab.StartTime >= pd.Timestamp(t0)) & (lab.EndTime <= pd.Timestamp(t1))]
    return lab[["ID", "Channel", "StartTime", "EndTime", "Category", "Dimensionality",
                "Locality", "Length"]].reset_index(drop=True)


def score_alarms(mission: int, ts: pd.DatetimeIndex, alarms: np.ndarray, channel_names: list,
                 labels: pd.DataFrame, per_channel: bool = True,
                 categories=("anomaly_and_rare", "anomaly")) -> dict:
    """Official metrics for per-channel alarms (T, C) on timestamps ``ts``."""
    alarms = np.asarray(alarms)
    assert alarms.shape == (len(ts), len(channel_names)), (alarms.shape, len(ts))
    lab = labels[labels.Channel.isin(channel_names)]
    glob = alarms.max(axis=1)
    y_glob = compress_binary(ts, glob)
    out = {"alarm_fraction": float(glob.mean()), "n_events_scored": int(lab.ID.nunique())}
    if lab.empty:
        return out
    if not glob.any():
        # no alarm at all: the official definitions give precision = recall = F = 0 (the
        # official affiliation code would fail on an empty prediction list)
        for key in categories:
            out.update({f"{key}/EW_precision": 0.0, f"{key}/EW_recall": 0.0,
                        f"{key}/EW_F_0.50": 0.0, f"{key}/alarming_precision": 0.0})
        return out
    chans = load_channels_table(mission) if per_channel else None
    for key in categories:
        cats = CATEGORY_SETS[key]
        if not lab.Category.isin(cats).any():
            continue
        r = _quiet(ESAScores(betas=0.5, select_labels={"Category": cats}).score,
                   lab.drop(columns=["Channel"]), y_glob.copy())
        out.update({f"{key}/{k}": float(v) for k, v in r.items()})
        lg = lab.copy()
        lg["Channel"] = "global"
        r = _quiet(ADTQC(select_labels={"Category": cats}).score, lg, {"global": y_glob.copy()})
        out.update({f"{key}/ADTQC_{k}": float(v) for k, v in r.items()})
        if per_channel:
            subs = {s: [*v] for s, v in chans.groupby("Subsystem")["Channel"]}
            y_ch = {c: compress_binary(ts, alarms[:, j]) for j, c in enumerate(channel_names)}
            r = _quiet(ChannelAwareFScore(beta=0.5, select_labels={"Category": cats}).score,
                       lab, y_ch, subs)
            out.update({f"{key}/{k}": float(v) for k, v in r.items()})
    return out


def official_scores(mission: int, alarms: np.ndarray, channel_names: list,
                    split: ESASplit | None = None, labels: pd.DataFrame | None = None,
                    per_channel: bool = True) -> dict:
    """Score ``alarms`` (T, C) uint8 on the official test split; returns a flat dict."""
    split = split or load_esa_split(mission, "test")
    lab = official_test_labels(mission, split) if labels is None else labels
    return score_alarms(mission, split.timestamps(), alarms, channel_names, lab, per_channel)


def false_alarm_stats(ts: pd.DatetimeIndex, glob: np.ndarray, labels: pd.DataFrame) -> dict:
    """Alarm runs that touch no labelled event, per year, and runs that touch ONLY rare events."""
    from .alarms import runs
    s, e = runs(np.asarray(glob, bool))
    t = pd.DatetimeIndex(ts).as_unit("ns").asi8
    starts, ends = t[s], t[np.minimum(e, len(t) - 1)]
    lab = labels
    # explicit ns: pandas 3 may parse CSV datetimes at us resolution
    ls = lab.StartTime.to_numpy().astype("datetime64[ns]").astype(np.int64)
    le = lab.EndTime.to_numpy().astype("datetime64[ns]").astype(np.int64)
    cat = lab.Category.to_numpy()
    n_false = n_rare_only = n_anom = 0
    for a, b in zip(starts, ends):
        hit = (ls <= b) & (le >= a)
        if not hit.any():
            n_false += 1
        elif (cat[hit] == "Anomaly").any():
            n_anom += 1
        elif (cat[hit] == "Rare Event").any():
            n_rare_only += 1
    years = (t[-1] - t[0]) / (365.25 * 86400e9)
    return {"alarm_runs": int(len(s)), "false_alarm_runs": n_false,
            "false_alarms_per_year": n_false / years, "rare_event_only_runs": n_rare_only,
            "anomaly_runs": n_anom}


def headline(scores: dict, key: str = "anomaly_and_rare") -> dict:
    """The columns of DMLR Table 4 / 16-17, under short names."""
    g = lambda k: scores.get(f"{key}/{k}", float("nan"))
    return {"EW_precision": g("EW_precision"), "EW_recall": g("EW_recall"),
            "EW_F0.5": g("EW_F_0.50"), "alarming_precision": g("alarming_precision"),
            "channel_F0.5": g("channel_F0.50"), "subsystem_F0.5": g("subsystem_F0.50"),
            "ADTQC": g("ADTQC_Total"), "AFF_F0.5": g("AFF_F_0.50"),
            "alarm_fraction": scores.get("alarm_fraction", float("nan"))}
