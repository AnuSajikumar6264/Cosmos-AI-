"""Catalogue and re-entry data for COSMOS-AI: a leakage-free landmark dataset, and ESA's
public list of recovered debris.

Landmark re-entry dataset
-------------------------
The SATCAT's orbit fields of a DECAYED object are its last-known values (the audit measured a
median perigee of 199 km for decayed vs 623 km for in-orbit objects), so any model trained on
them learns the label.  Without a credentialed element-set history (Space-Track) the only
leakage-free design is a LANDMARK built from two dated snapshots:

* tau = the first snapshot's data date (2026-08-16 for the copy shipped with COSMOS);
* population at risk = objects in orbit at tau, Earth-orbiting, with orbit fields;
* features = ONLY what the tau snapshot says (perigee, apogee, inclination, period, RCS,
  object type, owner, launch site, age at tau) plus, where present, the tau GP element set
  (BSTAR, mean motion, its derivative, eccentricity; EPOCH must be <= tau);
* outcome = decay date from the later snapshot: event if tau < decay <= follow-up date, else
  right-censored at the follow-up date (administrative censoring, identical for everyone);
* excluded and counted: decays dated <= tau but reported only later (the event precedes the
  landmark), objects launched after tau, objects missing from the later snapshot.

:func:`leakage_report` checks these rules on the built table.  Splits are by LAUNCH
(international designator ``YYYY-NNN``), so pieces of one launch or break-up never straddle
train and test.

Recovered debris (ESA re-entry portal)
--------------------------------------
``/api/home`` of reentry.esoc.esa.int publicly serves the "recovered debris" page, a curated
table of fragments found on the ground (COSPAR ID, re-entry date, latitude/longitude, mass,
shape, material).  :func:`parse_recovered_debris` turns it into a CSV.  It is real ground
truth for WHERE debris was found, not for predicted impact points: without the re-entry
state vector (TIP messages need a Space-Track account) no impact prediction can be scored
against it, and COSMOS-AI does not claim one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from ...common import feature_dir

TAU = "2026-08-16"
FOLLOW = "2026-10-23"
SNAPSHOT_TAU = ("debris", "celestrak")                            # the shipped copy
SNAPSHOT_FOLLOW = ("debris", "celestrak", "snapshots", "2026-10-24")
GP_TAU_FILES = ("gp-active.csv", "gp-cosmos-1408-debris.csv", "gp-cosmos-2251-debris.csv",
                "gp-fengyun-1c-debris.csv", "gp-iridium-33-debris.csv")
NUMERIC = ["PERIGEE", "APOGEE", "INCLINATION", "PERIOD", "RCS"]


def _satcat(parts) -> pd.DataFrame:
    return pd.read_csv(os.path.join(feature_dir(*parts), "satcat.csv"), low_memory=False)


def launch_group(object_id: str) -> str:
    """``1999-025ABC`` -> ``1999-025`` (one launch / one parent event)."""
    m = re.match(r"^(\d{4}-\d{3})", str(object_id))
    return m.group(1) if m else str(object_id)


def group_split(groups: pd.Series, fracs=(0.6, 0.2, 0.2), salt: str = "cosmos-ai-reentry"):
    """Deterministic hash split of groups into train / val / test."""
    def h(g):
        return int(hashlib.sha256(f"{salt}:{g}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    u = groups.map(h)
    a, b = fracs[0], fracs[0] + fracs[1]
    return np.where(u < a, "train", np.where(u < b, "val", "test"))


def build_landmark(tau: str = TAU, follow: str = FOLLOW, write: bool = True) -> tuple:
    """Build the landmark table; returns ``(table, exclusion counts)``."""
    t0, t1 = pd.Timestamp(tau), pd.Timestamp(follow)
    a = _satcat(SNAPSHOT_TAU)
    b = _satcat(SNAPSHOT_FOLLOW)[["NORAD_CAT_ID", "DECAY_DATE", "OPS_STATUS_CODE"]]
    a["launch_date"] = pd.to_datetime(a.LAUNCH_DATE, errors="coerce")
    a["decay_tau"] = pd.to_datetime(a.DECAY_DATE, errors="coerce")
    ex = {"tau_snapshot_objects": int(len(a))}
    a = a[a.decay_tau.isna()]
    ex["in_orbit_at_tau"] = int(len(a))
    a = a[a.ORBIT_CENTER.fillna("EA") == "EA"]
    ex["earth_orbiting"] = int(len(a))
    for c in NUMERIC:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[a.PERIGEE.notna() & a.APOGEE.notna() & a.INCLINATION.notna() & a.PERIOD.notna()]
    ex["with_orbit_fields"] = int(len(a))
    a = a[a.launch_date <= t0]
    ex["launched_by_tau"] = int(len(a))
    m = a.merge(b.assign(in_follow=1), on="NORAD_CAT_ID", how="left",
                suffixes=("", "_follow"))
    ex["missing_from_follow_snapshot"] = int(m.in_follow.isna().sum())
    m["decay_follow"] = pd.to_datetime(m.DECAY_DATE_follow, errors="coerce")
    late = m.decay_follow.notna() & (m.decay_follow <= t0)
    ex["decayed_before_tau_reported_late"] = int(late.sum())
    m = m[~late & m.in_follow.notna()].copy()
    ex["population_at_risk"] = int(len(m))
    ev = m.decay_follow.notna() & (m.decay_follow <= t1)
    m["event"] = ev.astype(int)
    m["time_days"] = np.where(ev, (m.decay_follow - t0).dt.days.astype(float),
                              float((t1 - t0).days))
    ex["events"] = int(ev.sum())
    ex["follow_up_days"] = int((t1 - t0).days)
    # ---- features known at tau ------------------------------------------------------
    m["age_days"] = (t0 - m.launch_date).dt.days.astype(float)
    m["mean_alt_km"] = 0.5 * (m.PERIGEE + m.APOGEE)
    m["ecc_proxy"] = (m.APOGEE - m.PERIGEE) / (2 * 6378.137 + m.APOGEE + m.PERIGEE)
    m["log_rcs"] = np.log10(m.RCS.clip(lower=1e-4))
    m["rcs_missing"] = m.RCS.isna().astype(int)
    gp = pd.concat([pd.read_csv(os.path.join(feature_dir(*SNAPSHOT_TAU), f))
                    for f in GP_TAU_FILES], ignore_index=True)
    gp["EPOCH"] = pd.to_datetime(gp.EPOCH, errors="coerce")
    gp = gp[gp.EPOCH <= t0 + pd.Timedelta(days=1)]                # tau's own data date
    gp = gp.sort_values("EPOCH").drop_duplicates("NORAD_CAT_ID", keep="last")
    gp = gp[["NORAD_CAT_ID", "EPOCH", "BSTAR", "MEAN_MOTION", "MEAN_MOTION_DOT",
             "ECCENTRICITY"]].rename(columns={"EPOCH": "gp_epoch", "BSTAR": "gp_bstar",
                                              "MEAN_MOTION": "gp_mean_motion",
                                              "MEAN_MOTION_DOT": "gp_ndot",
                                              "ECCENTRICITY": "gp_ecc"})
    m = m.merge(gp, on="NORAD_CAT_ID", how="left")
    m["gp_available"] = m.gp_epoch.notna().astype(int)
    m["gp_age_days"] = (t0 + pd.Timedelta(days=1) - m.gp_epoch).dt.total_seconds() / 86400
    m["launch_group"] = m.OBJECT_ID.map(launch_group)
    m["split"] = group_split(m.launch_group)
    m["uncontrolled"] = m.OBJECT_TYPE.isin(["DEB", "R/B"]).astype(int)
    keep = ["NORAD_CAT_ID", "OBJECT_ID", "OBJECT_NAME", "OBJECT_TYPE", "OWNER", "LAUNCH_SITE",
            "launch_group", "split", "uncontrolled", "PERIGEE", "APOGEE", "INCLINATION",
            "PERIOD", "RCS", "log_rcs", "rcs_missing", "age_days", "mean_alt_km", "ecc_proxy",
            "gp_available", "gp_bstar", "gp_mean_motion", "gp_ndot", "gp_ecc", "gp_age_days",
            "gp_epoch", "time_days", "event", "decay_follow"]
    out = m[keep].reset_index(drop=True)
    if write:
        d = feature_dir("reentry", "derived")
        out.to_csv(os.path.join(d, f"landmark_{tau}_to_{follow}.csv"), index=False)
        with open(os.path.join(d, f"landmark_{tau}_to_{follow}_exclusions.json"), "w") as f:
            json.dump(ex, f, indent=2)
    return out, ex


FEATURES = ["PERIGEE", "APOGEE", "INCLINATION", "PERIOD", "log_rcs", "rcs_missing", "age_days",
            "mean_alt_km", "ecc_proxy", "gp_available", "gp_bstar", "gp_mean_motion",
            "gp_ndot", "gp_ecc", "gp_age_days"]
CATEGORICAL = ["OBJECT_TYPE", "OWNER", "LAUNCH_SITE"]


def leakage_report(df: pd.DataFrame, tau: str = TAU) -> dict:
    """Checks that must all be True for the table to be usable."""
    t0 = pd.Timestamp(tau)
    ep = pd.to_datetime(df.gp_epoch, errors="coerce")
    g = df.groupby("launch_group").split.nunique()
    return {"no_gp_epoch_after_tau_day": bool((ep.dropna() <= t0 + pd.Timedelta(days=1)).all()),
            "no_event_at_or_before_tau": bool((df.time_days[df.event == 1] > 0).all()),
            "launch_groups_in_one_split": bool((g == 1).all()),
            "no_decay_field_in_features": not any("decay" in f.lower() for f in FEATURES),
            "censoring_time_constant": bool(df.time_days[df.event == 0].nunique() == 1)}


# ============================================================================ recovered debris
def parse_recovered_debris(api_json: str | None = None) -> pd.DataFrame:
    """ESA's recovered-debris table (rows continue the previous object when cells are merged)."""
    import html as _html
    api_json = api_json or os.path.join(feature_dir("reentry", "historical"), "raw",
                                        "esa_reentry_portal", "api_home.json")
    with open(api_json) as f:
        h = json.load(f)["recoveredDebris"]["htmlContent"]
    table = re.findall(r"<table.*?</table>", h, flags=re.S)[1]
    clean = lambda c: re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
    rows = [[clean(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=re.S)]
            for r in re.findall(r"<tr.*?</tr>", table, flags=re.S)]
    cols = ["cospar_id", "name", "reentry_date", "location", "country", "mass_kg", "shape",
            "size_mm", "material", "debris_type", "object_class"]
    out, cur = [], dict.fromkeys(cols, "")
    for r in rows[1:]:
        if len(r) == 11:
            cur = dict(zip(cols, r))
        elif len(r) == 7:
            cur = {**cur, **dict(zip(["location", "country", "mass_kg", "shape", "size_mm",
                                      "material", "debris_type"], r))}
        elif len(r) == 6:
            cur = {**cur, **dict(zip(["location", "mass_kg", "shape", "size_mm", "material",
                                      "debris_type"], r))}
        elif len(r) == 5:
            cur = {**cur, **dict(zip(["mass_kg", "shape", "size_mm", "material",
                                      "debris_type"], r))}
        else:
            continue
        out.append({**cur, "source_row_cells": len(r)})
    df = pd.DataFrame(out)
    ll = df.location.str.extract(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
    df["lat"], df["lon"] = pd.to_numeric(ll[0]), pd.to_numeric(ll[1])
    df["mass_kg"] = pd.to_numeric(df.mass_kg.str.replace(",", "."), errors="coerce")
    df["reentry_date"] = pd.to_datetime(df.reentry_date, errors="coerce")
    return df


def recovered_vs_inclination(rec: pd.DataFrame, satcat: pd.DataFrame) -> pd.DataFrame:
    """A physical consistency check: a fragment cannot land poleward of its orbit inclination.

    (For retrograde orbits the bound is 180 deg - i.)  This validates the record linkage
    (COSPAR ID -> catalogue object), not any COSMOS prediction.
    """
    s = satcat[["OBJECT_ID", "INCLINATION", "OBJECT_TYPE", "DECAY_DATE"]].copy()
    s["INCLINATION"] = pd.to_numeric(s.INCLINATION, errors="coerce")
    m = rec.merge(s, left_on="cospar_id", right_on="OBJECT_ID", how="left")
    bound = np.where(m.INCLINATION > 90, 180 - m.INCLINATION, m.INCLINATION)
    m["lat_bound_deg"] = bound
    m["within_inclination_band"] = np.abs(m.lat) <= bound + 1.0
    return m
