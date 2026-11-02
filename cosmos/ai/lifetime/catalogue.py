"""Leakage-controlled cohort table for catalogue-wide re-entry prediction.

One row per catalogued object whose GCAT *canonical orbit* (an orbit "representative of the
object's early operations, after initial positioning", J. McDowell) lies in or crosses low
Earth orbit (perigee below 2000 km).  The prediction time ``t0`` is the epoch of that orbit
(GCAT ``ODate``); everything used as an input must be known at ``t0``:

* orbit at ``t0`` (perigee, apogee, inclination) and design properties (mass, length,
  diameter, span, shape; object class);
* space weather from CelesTrak ``SW-All.csv``: only *observed* rows of days that ended before
  ``t0`` (trailing 81-day and 365-day F10.7 means, trailing 81-day Ap, 1-year F10.7 trend);
* family evidence (the debris-intelligence link): siblings from the same launch and, for
  fragmentation debris, the same breakup parent, counting **only siblings catalogued by
  ``t0`` and decays that happened before ``t0``**.

The table also carries the radar cross-section from the CelesTrak SATCAT snapshot
(``rcs_m2``).  It is **not** an input known at ``t0``: whether a value was ever published
depends on how long the object stayed in orbit (objects that decayed within weeks mostly
have none, and no values are published for objects catalogued after about 2018).  It is used
only for the class-level size prior of the physics expert (training objects only) and for a
labelled leak demonstration.

Outcome: natural decay (GCAT status ``R``) at ``DDate``.  Controlled endings (de-orbit,
landing, docking, attachment, explosion) are censored at their ``DDate``; objects still in
orbit are censored at the catalogue snapshot.  :func:`censor_at` applies administrative
censoring at a cut-off date, so that a model can be trained only on outcomes known at that
date (the temporal protocol of ``run.py``).

    python -m cosmos.ai.lifetime.catalogue          # writes data/feature3_reentry/processed/
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from ...common import feature_dir

SNAPSHOT = "2026-10-27"                               # GCAT and CelesTrak retrieval date
GCAT_DIR = ("debris", "gcat", "raw", SNAPSHOT)
CELESTRAK_DIR = ("debris", "celestrak", "snapshots", SNAPSHOT)
MAX_PERIGEE_KM = 2000.0
TYPES = ("P", "R", "C", "D")                          # payload, rocket stage, component, fragment
EVENT_STATUS = {"R"}                                  # destroyed after natural decay
CENSOR_AT_DDATE = {"D", "L", "LF", "DK", "GRP", "ATT", "AR", "AR IN", "AS", "AL", "AL IN",
                   "TFR", "TFR E", "TFR IN", "E", "C", "N", "NA", "UDK", "REL", "F", "AF",
                   "S", "EVA DP", "EVA RP", "REFLT"}
CENSOR_AT_SNAPSHOT = {"O", "AO", "AO IN"}
EXCLUDE_STATUS = {"OX", "DSO", "DSA", "DSA IN", "EO", "EAO", "OI", "OE", "EN", "LO", "LOA", "TO",
                  "TOA", "DEP", "TX", "LEASE"}        # lost, deep space, other bodies
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                                      "Sep", "Oct", "Nov", "Dec"], start=1)}
_VAGUE = re.compile(r"^\s*(\d{4})(?:\s+([A-Z][a-z]{2}))?(?:\s+(\d{1,2}(?:\.\d+)?))?"
                    r"(?:\s+(\d{2})(\d{2})(?::(\d{2}))?)?\s*(\?)?\s*$")


# ------------------------------------------------------------------------------ parsing
def parse_vague(s) -> tuple:
    """GCAT vague date -> (timestamp, precision in days, uncertain flag).

    ``'1957 Dec  1 1000?'`` -> (1957-12-01 10:00, 1/24, True); ``'2001 Mar'`` -> (2001-03-15,
    15, False); missing ('-') -> (NaT, nan, False).
    """
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    m = _VAGUE.match(s)
    if not m:
        return pd.NaT, np.nan, False
    y, mon, day, hh, mm, ss, q = m.groups()
    if mon is None:
        return pd.Timestamp(int(y), 7, 2), 183.0, bool(q)
    if day is None:
        return pd.Timestamp(int(y), MONTHS[mon], 15), 15.0, bool(q)
    d = float(day)
    t = pd.Timestamp(int(y), MONTHS[mon], 1) + pd.Timedelta(days=d - 1)
    prec = 1.0
    if hh is not None:
        t += pd.Timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss or 0))
        prec = 1 / 24
    return t, prec, bool(q)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"-": np.nan, "": np.nan}), errors="coerce")


def load_gcat() -> pd.DataFrame:
    p = os.path.join(feature_dir(*GCAT_DIR), "satcat.tsv")
    df = pd.read_csv(p, sep="\t", dtype=str, comment="#", header=None, low_memory=False)
    with open(p) as f:
        cols = f.readline().lstrip("#").rstrip("\n").split("\t")
    df.columns = cols
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def load_space_weather() -> pd.DataFrame:
    """Daily observed space weather (rows flagged PRD/PRM are predictions and are dropped)."""
    p = os.path.join(feature_dir(*CELESTRAK_DIR), "SW-All.csv")
    sw = pd.read_csv(p, parse_dates=["DATE"])
    sw = sw[sw["F10.7_DATA_TYPE"].isin(["OBS", "INT"])].copy()
    sw = sw.set_index("DATE").sort_index()
    f = sw["F10.7_OBS"].astype(float)
    ap = sw["AP_AVG"].astype(float)
    out = pd.DataFrame({
        "f107_last81": f.rolling(81, min_periods=60).mean(),      # trailing: ends on the day
        "f107_last365": f.rolling(365, min_periods=300).mean(),
        "ap_last81": ap.rolling(81, min_periods=60).mean(),
        "f107_daily": f,
    })
    out["f107_trend365"] = out["f107_last81"] - out["f107_last81"].shift(365)
    return out


# ------------------------------------------------------------------------------ build
def build(write: bool = True) -> pd.DataFrame:
    g = load_gcat()
    n0 = len(g)
    log = {"gcat_rows": n0}
    g["type1"] = g["Type"].str[:1]
    keep = g["Primary"].eq("Earth") & g["type1"].isin(TYPES)
    log["drop_not_earth_or_type"] = int((~keep).sum())
    g = g[keep].copy()
    st = g["Status"]
    keep = ~st.isin(EXCLUDE_STATUS)
    log["drop_status_" + "_".join(sorted(EXCLUDE_STATUS & set(st[~keep])))] = int((~keep).sum())
    g = g[keep].copy()

    for c, name in (("ODate", "t0"), ("DDate", "t_end"), ("LDate", "t_launch")):
        parsed = g[c].map(parse_vague)
        g[name] = pd.to_datetime([p[0] for p in parsed])
        g[name + "_prec_d"] = [p[1] for p in parsed]
        g[name + "_uncertain"] = [p[2] for p in parsed]
    for c in ("Perigee", "Apogee", "Inc", "Mass", "DryMass", "TotMass", "Length", "Diameter", "Span"):
        g[c.lower()] = _num(g[c])
    for c in ("mass", "drymass", "totmass", "length", "diameter", "span"):
        g.loc[g[c] <= 0, c] = np.nan                    # GCAT uses 0 for unknown
    g["mass_uncertain"] = g["MassFlag"].eq("?")
    g["orbit_uncertain"] = g["PF"].eq("?") | g["AF"].eq("?") | g["IF"].eq("?")

    ok = g["t0"].notna() & (g["t0_prec_d"] <= 1.0) & g["perigee"].notna() & g["apogee"].notna() & g["inc"].notna()
    log["drop_no_dayprecise_orbit"] = int((~ok).sum())
    g = g[ok].copy()
    lowp = g["perigee"] < MAX_PERIGEE_KM
    log["drop_perigee_ge_2000km"] = int((~lowp).sum())
    g = g[lowp].copy()

    snap = pd.Timestamp(SNAPSHOT)
    ev = g["Status"].isin(EVENT_STATUS)
    cd = g["Status"].isin(CENSOR_AT_DDATE)
    cs = g["Status"].isin(CENSOR_AT_SNAPSHOT)
    end = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns]")
    end[ev | cd] = g.loc[ev | cd, "t_end"]
    end[cs] = snap
    bad_end = (ev | cd) & (g["t_end"].isna() | (g["t_end_prec_d"] > 1.0))
    log["drop_event_or_end_without_dayprecise_date"] = int(bad_end.sum())
    g, end, ev = g[~bad_end].copy(), end[~bad_end], ev[~bad_end]
    g["event"] = ev.astype(int)
    g["time_days"] = (end - g["t0"]).dt.total_seconds() / 86400.0
    neg = g["time_days"] < 0
    log["drop_end_before_t0"] = int(neg.sum())
    g = g[~neg].copy()
    g["end_kind"] = np.where(g["event"].eq(1), "natural_decay",
                             np.where(g["Status"].isin(CENSOR_AT_SNAPSHOT), "in_orbit", "controlled_or_other"))

    # ---- orbit and physical features (known at t0)
    RE = 6378.137
    a = RE + (g["perigee"] + g["apogee"]) / 2
    g["sma_km"] = a
    g["ecc"] = (g["apogee"] - g["perigee"]) / (2 * a)
    g["mean_alt_km"] = (g["perigee"] + g["apogee"]) / 2
    g["age_at_t0_d"] = (g["t0"] - g["t_launch"]).dt.total_seconds() / 86400.0
    L, D, S = g["length"], g["diameter"], g["span"]
    # average tumbling cross-section of a cylinder (Sentman/area-averaging: A = pi D^2/4 * 1/2
    # + L D * pi/4) as a design-area proxy; span adds deployed appendages
    cyl = 0.5 * np.pi * D ** 2 / 4 + np.pi / 4 * L * D
    g["area_design_m2"] = np.where(S > L, np.maximum(cyl, 0.25 * L * S), cyl)
    g["am_design"] = g["area_design_m2"] / g["mass"]
    g["shape"] = g["Shape"].replace({"-": "unknown", "": "unknown"}).fillna("unknown")

    # ---- CelesTrak SATCAT: radar cross-section (m^2); NOT an input at t0 (see the docstring)
    sc = pd.read_csv(os.path.join(feature_dir(*CELESTRAK_DIR), "satcat.csv"),
                     usecols=["NORAD_CAT_ID", "RCS", "OBJECT_TYPE"])
    sc = sc.drop_duplicates("NORAD_CAT_ID").set_index("NORAD_CAT_ID")
    nid = pd.to_numeric(g["Satcat"], errors="coerce")
    g["norad"] = nid
    g["rcs_m2"] = nid.map(sc["RCS"]).astype(float)
    g.loc[g["rcs_m2"] <= 0, "rcs_m2"] = np.nan

    # ---- space weather at t0 (trailing, observed only).  t0 is a day-precise epoch stored as
    # 00:00, and a day's F10.7 (measured near 20 UT) and daily Ap are complete only at its end,
    # so the trailing windows end on the last day before t0
    sw = load_space_weather()
    day = g["t0"].dt.normalize()
    idx = sw.index.searchsorted(day, side="left") - 1      # last observed day < t0
    idx = np.clip(idx, 0, len(sw) - 1)
    for c in ("f107_last81", "f107_last365", "ap_last81", "f107_trend365"):
        v = sw[c].to_numpy()[idx]
        v[(day < sw.index[0]).to_numpy()] = np.nan
        g[c] = v

    # ---- family evidence (causal at t0)
    # fragment parent class (rocket stage or not) selects the breakup-model branch
    full = load_gcat()
    full_type = full.set_index("JCAT")["Type"].str[:1]
    parent = g["Parent"].str.replace("*", "", regex=False).str.split("/").str[0]
    g["parent_type"] = parent.map(full_type).fillna("none")
    g["parent_is_rocket"] = g["parent_type"].eq("R")
    g["launch_family"] = g["Launch_Tag"]
    g["breakup_family"] = np.where(g["type1"].eq("D"), g["Parent"].str.replace("*", "", regex=False), "")
    for fam in ("launch_family", "breakup_family"):
        feats = family_features(g, fam)
        for c, v in feats.items():
            g[f"{fam}_{c}"] = v

    cols = ["JCAT", "Satcat", "norad", "Name", "Launch_Tag", "Piece", "type1", "Status", "end_kind", "Owner",
            "State", "Parent", "parent_type", "parent_is_rocket", "t_launch", "t0", "t_end", "event", "time_days",
            "perigee", "apogee", "inc", "sma_km", "ecc", "mean_alt_km", "age_at_t0_d",
            "mass", "length", "diameter", "span", "shape", "area_design_m2", "am_design", "rcs_m2",
            "mass_uncertain", "orbit_uncertain", "t0_uncertain",
            "f107_last81", "f107_last365", "ap_last81", "f107_trend365",
            "launch_family", "breakup_family"] + [c for c in g.columns if c.startswith(("launch_family_", "breakup_family_"))]
    out = g[cols].sort_values("t0").reset_index(drop=True)
    log["rows_out"] = len(out)
    log["events"] = int(out["event"].sum())
    log["end_kind"] = out["end_kind"].value_counts().to_dict()
    if write:
        from ..data.download import file_hashes, utc_now, write_manifest
        d = feature_dir("reentry", "processed")
        p = os.path.join(d, f"lifetime_cohort_{SNAPSHOT}.csv.gz")
        out.to_csv(p, index=False)
        with open(os.path.join(d, f"lifetime_cohort_{SNAPSHOT}_build_log.json"), "w") as f:
            json.dump(log, f, indent=1, default=str)
        write_manifest(d, {
            "dataset": "reentry_processed", "class": "DERIVED (from REAL catalogue data)", "status": "generated",
            "generator": "python -m cosmos.ai.lifetime.catalogue",
            "inputs": [f"data/feature2_debris/gcat/raw/{SNAPSHOT}/satcat.tsv",
                       f"data/feature2_debris/celestrak/snapshots/{SNAPSHOT}/satcat.csv",
                       f"data/feature2_debris/celestrak/snapshots/{SNAPSHOT}/SW-All.csv"],
            "notes": ["One row per object with a LEO-crossing GCAT canonical orbit; see the docstring of "
                      "cosmos/ai/lifetime/catalogue.py and docs/TARGET_DEFINITIONS.md.",
                      "msis21_density_table.npz: NRLMSIS 2.1 densities (pymsis 0.13.0) used by the physics expert."]},
            [{"path": os.path.basename(p), "bytes": os.path.getsize(p), **file_hashes(p), "retrieved_utc": utc_now()}])
    return out


def family_features(g: pd.DataFrame, fam: str) -> dict:
    """Causal sibling statistics at each object's t0 within ``fam`` (excluding the object).

    n_known      siblings catalogued (canonical-orbit epoch) on or before t0
    n_decayed    siblings that decayed naturally before t0
    frac_decayed n_decayed / n_known
    mean_log_life  mean log10(lifetime in days) of the siblings that decayed before t0
    """
    n = len(g)
    n_known = np.zeros(n)
    n_dec = np.zeros(n)
    mean_ll = np.full(n, np.nan)
    pos = {k: i for i, k in enumerate(g.index)}
    t0 = g["t0"].to_numpy("datetime64[s]").astype(np.int64)
    tend = g["t_end"].to_numpy("datetime64[s]")
    ev = g["event"].to_numpy() == 1
    life = np.log10(np.maximum(g["time_days"].to_numpy(), 1e-3))
    for key, idx in g.groupby(fam).indices.items():
        if key in ("", None) or (isinstance(key, float) and np.isnan(key)) or len(idx) < 2:
            continue
        ti = t0[idx]
        known_sorted = np.sort(ti)
        n_known[idx] = np.searchsorted(known_sorted, ti, side="right") - 1          # minus self
        di = idx[ev[idx]]
        if len(di):
            dd = tend[di].astype("datetime64[s]").astype(np.int64)
            order = np.argsort(dd)
            dd_s, ll_s = dd[order], life[di][order]
            csum = np.concatenate([[0.0], np.cumsum(ll_s)])
            k = np.searchsorted(dd_s, ti, side="left")                             # decays strictly before t0
            n_dec[idx] = k
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_ll[idx] = np.where(k > 0, csum[k] / np.maximum(k, 1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_known > 0, n_dec / np.maximum(n_known, 1), np.nan)
    return {"n_known": n_known, "n_decayed": n_dec, "frac_decayed": frac, "mean_log_life": mean_ll}


def censor_at(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Administrative censoring: outcomes after ``cutoff`` are unknown at ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    out = df[df["t0"] < cutoff].copy()
    end = out["t0"] + pd.to_timedelta(out["time_days"], unit="D")
    late = end > cutoff
    out.loc[late, "event"] = 0
    out.loc[late, "time_days"] = (cutoff - out.loc[late, "t0"]).dt.total_seconds() / 86400.0
    return out


if __name__ == "__main__":
    df = build()
    print(df.shape, int(df.event.sum()), "events")
    print(df.end_kind.value_counts().to_dict())
