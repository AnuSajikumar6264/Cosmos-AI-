"""FEATURE 2 (a) — the real debris catalogue.

Loads the CelesTrak SATCAT (the public US Space Force catalogue, no credentials needed) and
derives the quantities the project validates against:

* how debris is distributed with altitude (spatial density per 100 km shell);
* the four large break-up clouds, matched by international designator prefix;
* measured orbital lifetimes, from ``DECAY_DATE`` minus the break-up date.

One trap is worth knowing: the ``gp-*.csv`` element-set files contain only objects that are
STILL IN ORBIT.  Building a break-up cloud from them keeps survivors only, which biases
every lifetime towards infinity.  Clouds must come from SATCAT by designator, which is what
:func:`load_breakup_cloud` does.

    cat = load_satcat()
    spatial_density(cat).head()
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common import R_EARTH, catalogue_dir, feature_dir


@dataclass(frozen=True)
class BreakupEvent:
    """A documented on-orbit break-up whose debris can be followed in the catalogue."""
    key: str
    name: str
    date: str
    altitude_km: float
    designator_prefix: str
    cause: str


#: the four large, well-documented break-ups used throughout the project
BREAKUP_EVENTS = {
    "cosmos-1408": BreakupEvent("cosmos-1408", "Cosmos 1408", "2021-11-15", 480.0,
                                "1982-092", "ASAT test"),
    "iridium-33": BreakupEvent("iridium-33", "Iridium 33", "2009-02-10", 789.0,
                               "1997-051", "collision"),
    "cosmos-2251": BreakupEvent("cosmos-2251", "Cosmos 2251", "2009-02-10", 789.0,
                                "1993-036", "collision"),
    "fengyun-1c": BreakupEvent("fengyun-1c", "Fengyun-1C", "2007-01-11", 865.0,
                               "1999-025", "ASAT test"),
}


def load_satcat(path: str | None = None) -> pd.DataFrame:
    """Load the SATCAT and add convenience columns.

    Adds ``perigee_km``, ``apogee_km``, ``mean_alt_km``, ``decayed`` (bool),
    ``rcs_m2`` and ``launch_year``.
    """
    path = path or os.path.join(catalogue_dir(), "satcat.csv")
    df = pd.read_csv(path, low_memory=False)
    df["perigee_km"] = pd.to_numeric(df.get("PERIGEE"), errors="coerce")
    df["apogee_km"] = pd.to_numeric(df.get("APOGEE"), errors="coerce")
    df["mean_alt_km"] = 0.5 * (df.perigee_km + df.apogee_km)
    df["inclination_deg"] = pd.to_numeric(df.get("INCLINATION"), errors="coerce")
    df["rcs_m2"] = pd.to_numeric(df.get("RCS"), errors="coerce")
    df["decay_date"] = pd.to_datetime(df.get("DECAY_DATE"), errors="coerce")
    df["launch_date"] = pd.to_datetime(df.get("LAUNCH_DATE"), errors="coerce")
    df["decayed"] = df.decay_date.notna()
    df["launch_year"] = df.launch_date.dt.year
    return df


def catalogue_overview(cat: pd.DataFrame) -> pd.DataFrame:
    """Counts by object type, and how many of each have already re-entered."""
    g = cat.groupby("OBJECT_TYPE", dropna=False)
    out = pd.DataFrame({"objects": g.size(), "decayed": g.decayed.sum()})
    out["still_in_orbit"] = out.objects - out.decayed
    out["percent_decayed"] = (100 * out.decayed / out.objects).round(1)
    return out.sort_values("objects", ascending=False)


def spatial_density(cat: pd.DataFrame, lo_km: float = 200.0, hi_km: float = 1600.0,
                    step_km: float = 100.0, object_type: str = "DEB") -> pd.DataFrame:
    """Measured number density of catalogued objects per 100 km shell [1/km^3]."""
    sel = cat[(cat.OBJECT_TYPE == object_type) & (~cat.decayed) & cat.mean_alt_km.notna()]
    edges = np.arange(lo_km, hi_km + step_km, step_km)
    rows = []
    re_km = R_EARTH / 1e3
    for a, b in zip(edges[:-1], edges[1:]):
        n = int(((sel.mean_alt_km >= a) & (sel.mean_alt_km < b)).sum())
        vol = 4.0 / 3.0 * np.pi * ((re_km + b) ** 3 - (re_km + a) ** 3)
        rows.append({"alt_lo_km": a, "alt_hi_km": b, "alt_mid_km": 0.5 * (a + b),
                     "count": n, "shell_volume_km3": vol, "density_per_km3": n / vol})
    return pd.DataFrame(rows)


def load_breakup_cloud(key: str, cat: pd.DataFrame | None = None) -> pd.DataFrame:
    """All catalogued debris from one break-up, matched by international designator.

    Adds ``lifetime_years`` (decay date minus break-up date) and ``observed`` — the latter
    is ``False`` for fragments still in orbit, whose lifetime is right-censored.
    """
    if key not in BREAKUP_EVENTS:
        raise KeyError(f"unknown break-up {key!r}; expected {sorted(BREAKUP_EVENTS)}")
    ev = BREAKUP_EVENTS[key]
    cat = load_satcat() if cat is None else cat
    desig = cat.get("OBJECT_ID", pd.Series(dtype=str)).astype(str)
    sel = cat[desig.str.startswith(ev.designator_prefix) & (cat.OBJECT_TYPE == "DEB")].copy()
    t0 = pd.Timestamp(ev.date)
    now = pd.Timestamp("2026-08-29")
    sel["event"] = ev.name
    sel["event_altitude_km"] = ev.altitude_km
    sel["observed"] = sel.decayed
    end = sel.decay_date.where(sel.decayed, now)
    sel["lifetime_years"] = (end - t0).dt.total_seconds() / (365.25 * 86400.0)
    return sel[sel.lifetime_years > 0].reset_index(drop=True)


def all_breakup_clouds(cat: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every fragment from all four break-ups, concatenated. This is the training set."""
    cat = load_satcat() if cat is None else cat
    return pd.concat([load_breakup_cloud(k, cat) for k in BREAKUP_EVENTS],
                     ignore_index=True)


def cloud_overview(clouds: pd.DataFrame) -> pd.DataFrame:
    """One row per break-up: fragments, how many have come down, measured median lifetime."""
    rows = []
    for name, g in clouds.groupby("event", sort=False):
        km = kaplan_meier(g.lifetime_years.values, g.observed.values)
        rows.append({"event": name, "altitude_km": float(g.event_altitude_km.iloc[0]),
                     "fragments": len(g), "decayed": int(g.observed.sum()),
                     "percent_decayed": round(100.0 * g.observed.mean(), 1),
                     "median_lifetime_years": km["median"],
                     "censored": int((~g.observed).sum())})
    return pd.DataFrame(rows).sort_values("altitude_km").reset_index(drop=True)


def kaplan_meier(time: np.ndarray, observed: np.ndarray) -> dict:
    """Kaplan-Meier survival curve, which handles fragments still in orbit correctly.

    Returns the curve plus the median survival time, or NaN when more than half the
    population is still up and the median is therefore not reached.
    """
    time = np.asarray(time, float)
    observed = np.asarray(observed, bool)
    order = np.argsort(time)
    t, o = time[order], observed[order]
    n = t.size
    surv, times, at_risk = [], [], n
    s = 1.0
    for i in range(n):
        if o[i]:
            s *= 1.0 - 1.0 / max(at_risk, 1)
            times.append(t[i])
            surv.append(s)
        at_risk -= 1
    times, surv = np.array(times), np.array(surv)
    median = float(times[np.argmax(surv <= 0.5)]) if surv.size and surv.min() <= 0.5 else np.nan
    return {"times": times, "survival": surv, "median": median,
            "n": n, "n_observed": int(o.sum())}


def export_derived(cat: pd.DataFrame | None = None) -> dict:
    """Write the derived per-feature datasets to disk and report where they went.

    * ``feature2_debris/celestrak/spatial_density.csv`` — measured debris density profile
    * ``feature3_reentry/derived/measured_lifetimes.csv`` — the real training label
    """
    cat = load_satcat() if cat is None else cat
    dens = spatial_density(cat)
    clouds = all_breakup_clouds(cat)
    p1 = os.path.join(feature_dir("debris", "celestrak"), "spatial_density.csv")
    p2 = os.path.join(feature_dir("reentry", "derived"), "measured_lifetimes.csv")
    dens.to_csv(p1, index=False)
    cols = ["OBJECT_NAME", "OBJECT_ID", "event", "event_altitude_km", "perigee_km",
            "apogee_km", "rcs_m2", "lifetime_years", "observed"]
    clouds[cols].to_csv(p2, index=False)
    return {"spatial_density": p1, "measured_lifetimes": p2,
            "n_density_shells": len(dens), "n_fragments": len(clouds),
            "n_observed": int(clouds.observed.sum()),
            "n_censored": int((~clouds.observed).sum())}
