"""Event table for collision-risk forecasting (ESA Kelvins CAC, Zenodo 4463683).

Each event (``event_id``) is a time series of Conjunction Data Messages (CDMs).  The task, as in
the competition, is to predict the risk (log10 collision probability) of the **final** CDM of an
event from the CDMs released **at least 2 days before** the time of closest approach (TCA).

Training events are filtered to mirror the test inputs: at least one CDM with
``time_to_tca >= 2`` (days) and a final CDM with ``time_to_tca < 1``.  The official test inputs
(``test_data.csv``) already contain only CDMs with ``time_to_tca >= 2``; their targets are in
``test_data_private.csv`` (``true_risk``), released after the competition.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ...common import feature_dir

RAW = ("debris", "kelvins_cac", "raw", "extracted", "Collision Avoidance Challenge - Dataset",
       "kelvins_competition_data")
HIGH_RISK = -6.0
RISK_FLOOR = -30.0


def _dir():
    return feature_dir(*RAW)


def load_raw() -> dict:
    d = _dir()
    return {"train": pd.read_csv(os.path.join(d, "train_data.csv")),
            "test": pd.read_csv(os.path.join(d, "test_data.csv")),
            "test_private": pd.read_csv(os.path.join(d, "test_data_private.csv"))}


NUMERIC_EXCLUDE = {"event_id", "risk", "true_risk", "c_object_type", "mission_id"}


def event_features(cdm: pd.DataFrame) -> pd.DataFrame:
    """One row per event from its input CDMs (sorted so that the last row is the latest CDM)."""
    cdm = cdm.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    g = cdm.groupby("event_id", sort=True)
    last = g.tail(1).set_index("event_id")
    num = [c for c in cdm.columns if c not in NUMERIC_EXCLUDE and pd.api.types.is_numeric_dtype(cdm[c])]
    base = last[num].add_prefix("last_")
    out = {}
    out["latest_risk"] = last["risk"]
    out["mission_id"] = last["mission_id"]
    out["c_object_type"] = last["c_object_type"].fillna("NA")
    r = cdm["risk"]
    out["n_cdm"] = g.size()
    out["risk_max"] = g["risk"].max()
    out["risk_min"] = g["risk"].min()
    out["risk_mean"] = g["risk"].mean()
    out["risk_first"] = g["risk"].first()
    out["risk_change"] = last["risk"] - out["risk_first"]
    out["tca_first"] = g["time_to_tca"].first()
    out["span_days"] = out["tca_first"] - last["time_to_tca"]
    # slope of risk versus time to TCA over the last (up to) three CDMs
    def slope(s):
        t, y = s["time_to_tca"].to_numpy()[-3:], s["risk"].to_numpy()[-3:]
        if len(t) < 2 or np.ptp(t) < 1e-6:
            return 0.0
        return float(np.polyfit(-t, y, 1)[0])
    out["risk_slope3"] = g[["time_to_tca", "risk"]].apply(slope)
    prev = g["risk"].apply(lambda s: s.iloc[-2] if len(s) >= 2 else np.nan)
    out["risk_last_step"] = last["risk"] - prev
    md = g["miss_distance"]
    out["miss_log_ratio"] = np.log10(np.maximum(last["miss_distance"], 1e-3) / np.maximum(md.first(), 1e-3))
    for c in ("c_sigma_r", "c_sigma_t", "c_sigma_n", "t_sigma_r", "t_sigma_t", "t_sigma_n"):
        if c in cdm:
            out[f"{c}_log_ratio"] = np.log10(np.maximum(last[c], 1e-6) / np.maximum(g[c].first(), 1e-6))
    return pd.concat([base, pd.DataFrame(out)], axis=1)


def build() -> dict:
    raw = load_raw()
    tr = raw["train"]
    fin = tr.sort_values("time_to_tca").groupby("event_id").first()
    has_early = tr[tr["time_to_tca"] >= 2].groupby("event_id").size()
    ok = fin.index[(fin["time_to_tca"] < 1)].intersection(has_early.index)
    tr_in = tr[tr["event_id"].isin(ok) & (tr["time_to_tca"] >= 2)]
    Xtr = event_features(tr_in)
    Xtr["y_final"] = fin.loc[Xtr.index, "risk"]
    te = raw["test"]
    Xte = event_features(te)
    tp = raw["test_private"].set_index("event_id")
    Xte["y_final"] = tp.loc[Xte.index, "true_risk"]
    log = {"train_events_raw": int(tr.event_id.nunique()), "train_events_kept": int(len(Xtr)),
           "test_events": int(len(Xte)), "train_high_risk": int((Xtr.y_final >= HIGH_RISK).sum()),
           "test_high_risk": int((Xte.y_final >= HIGH_RISK).sum())}
    return {"train": Xtr, "test": Xte, "log": log}
