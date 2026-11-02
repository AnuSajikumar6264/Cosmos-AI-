"""Digital-twin engine: build twins for the objects in orbit at ``as_of`` and run the modules.

    python -m cosmos.ai.twin.engine --as-of 2026-10-27

What it does, in order:

1. **State.** Objects in orbit at ``as_of`` with perigee below 2000 km, from the CelesTrak
   SATCAT snapshot of that date (current orbit, operational status), joined to GCAT for design
   properties, mass flags and family links; space weather from ``SW-All.csv`` (observed rows of
   days before ``as_of``).
2. **Re-entry module, uncontrolled objects** (fragments, rocket stages, components).

   The lifetime study validates forecasts made at an object's canonical-orbit epoch. An object
   still in orbit today has survived since that epoch, and that survival selects low
   area-to-mass ratios, which neither the physics prior nor the learned experts know about.

   The twin therefore uses the physics expert with an area-to-mass posterior conditioned on the
   object's catalogue history: the canonical orbit, survival to ``as_of`` and the current orbit
   (``physics.history_posterior``).

   The validated lifetime pipeline is computed alongside for comparison and reported in every
   record. It is the study's learner and settings as a 5-seed ensemble, with the gate fitted on
   training out-of-fold predictions and Mondrian CQR and isotonic maps fitted on tune-B, all
   refitted with every outcome known at ``as_of``.

   Both variants, and the physics prior alone, are tested prospectively in
   ``cosmos.ai.twin.prospective`` (T1).
3. **Health coupling (C3), payloads.** Operational payloads are in the operator-controlled
   regime: the twin abstains from a drag forecast and reports the previous-window payload base
   rate for each horizon (the status-aware policy tested prospectively in C3).  Non-operational
   payloads are in the natural-decay regime and get the physics expert's probabilities (the C3
   test validates them at 72 days); payloads without a status are reported as a separate
   ``unknown`` regime with physics probabilities.  The learned experts were not validated on
   payloads and are not used for them.
4. **Debris coupling (C2).** For each breakup cloud with at least 30 fragments in orbit, the
   expected number still in orbit after each horizon, from the calibrated per-fragment
   probabilities.
5. **Conjunction module** (optional): if CDM sequences are supplied for an object, the gated
   collision-risk forecaster is applied (see ``cosmos.ai.conjunction``).

The output is a set of object reports (JSON lines), a cloud table and a population summary;
every module output carries the validation that supports it.  Because the forecast is made as
of a date and hashed, it can be verified later against the catalogue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from ..lifetime import fusion as FU
from ..lifetime import models as MO
from ..lifetime import physics as PH
from ..lifetime.catalogue import _num, censor_at, family_features, load_gcat, load_space_weather, parse_vague
from .couplings import previous_window_rate
from .state import ModuleOutput, ObjectTwin

HORIZONS = {"30d": 30.0, "90d": 90.0, "1yr": 365.25}
# SATCAT snapshot folder per as-of date (the 2026-08 snapshot, data date 2026-08-16, is the
# original COSMOS download in the folder root)
SNAPSHOT_DIRS = {"2026-08-16": ("debris", "celestrak")}


def snapshot_dir(as_of: str) -> str:
    return feature_dir(*SNAPSHOT_DIRS.get(as_of, ("debris", "celestrak", "snapshots", as_of)))
VALIDATION = {
    "uncontrolled": "validated-real: GCAT temporal tests (origins 2011, 2016) at canonical-orbit epochs; "
                    "applied to the current orbit (prospective check pending)",
    "payload_natural": "physics expert only; prospective test C3 (72 d from 2026-08-16) for non-operational payloads",
    "payload_unknown": "physics expert only; regime unknown (no operational status); not validated",
    "payload_controlled": "abstains from a drag forecast; previous-window payload base rate (C3 status-aware policy)",
}


def _md5_fold(s: pd.Series, k: int) -> np.ndarray:
    return s.astype(str).map(lambda v: int(hashlib.md5(v.encode()).hexdigest(), 16) % k).to_numpy()


def current_state(as_of: str) -> pd.DataFrame:
    """In-orbit LEO-crossing objects at ``as_of`` with features in the lifetime-study format."""
    sc = pd.read_csv(os.path.join(snapshot_dir(as_of), "satcat.csv"))
    t0 = pd.Timestamp(as_of)
    sc = sc[sc["DECAY_DATE"].isna() & sc["PERIGEE"].notna() & (sc["PERIGEE"] < 2000) & sc["ORBIT_CENTER"].eq("EA")]
    sc = sc[pd.to_datetime(sc["LAUNCH_DATE"], errors="coerce") <= t0].reset_index(drop=True)
    g = load_gcat()
    g["norad"] = pd.to_numeric(g["Satcat"], errors="coerce")
    g = g.dropna(subset=["norad"]).drop_duplicates("norad").set_index("norad")
    d = pd.DataFrame({"norad": sc["NORAD_CAT_ID"].to_numpy(), "Name": sc["OBJECT_NAME"].to_numpy(),
                      "ops_status": sc["OPS_STATUS_CODE"].to_numpy(),
                      "perigee": sc["PERIGEE"].astype(float).to_numpy(), "apogee": sc["APOGEE"].astype(float).to_numpy(),
                      "inc": sc["INCLINATION"].astype(float).to_numpy(), "rcs_m2": pd.to_numeric(sc["RCS"], errors="coerce").to_numpy()})
    d.loc[d["rcs_m2"] <= 0, "rcs_m2"] = np.nan                  # reported only; not a model input
    d["JCAT"] = d["norad"].map(g["JCAT"]).fillna("N" + d["norad"].astype(str))   # key of the physics random streams
    gt = g["Type"].str[:1]
    cls = d["norad"].map(gt)
    fallback = sc["OBJECT_TYPE"].map({"PAY": "P", "R/B": "R", "DEB": "D"}).fillna("D").to_numpy()
    d["type1"] = np.where(cls.isin(["P", "R", "C", "D"]), cls, fallback)
    for c in ("Mass", "Length", "Diameter", "Span"):
        v = _num(g[c])
        v[v <= 0] = np.nan
        d[c.lower()] = d["norad"].map(v)
    d["Shape"] = d["norad"].map(g["Shape"]).fillna("unknown")
    d["shape"] = d["Shape"].replace({"-": "unknown", "": "unknown"})
    d["Launch_Tag"] = d["norad"].map(g["Launch_Tag"]).fillna(sc["OBJECT_ID"].astype(str).str[:8])
    d["Parent"] = d["norad"].map(g["Parent"]).fillna("")
    full_type = g.reset_index().set_index("JCAT")["Type"].str[:1]
    parent = d["Parent"].astype(str).str.replace("*", "", regex=False).str.split("/").str[0]
    d["parent_type"] = parent.map(full_type).fillna("none")
    d["parent_is_rocket"] = d["parent_type"].eq("R")
    d["t0"] = t0
    d["t_launch"] = pd.to_datetime(sc["LAUNCH_DATE"], errors="coerce")
    RE = 6378.137
    a = RE + (d["perigee"] + d["apogee"]) / 2
    d["sma_km"], d["ecc"], d["mean_alt_km"] = a, (d["apogee"] - d["perigee"]) / (2 * a), (d["perigee"] + d["apogee"]) / 2
    d["age_at_t0_d"] = (t0 - d["t_launch"]).dt.total_seconds() / 86400
    L, D, S = d["length"], d["diameter"], d["span"]
    cyl = 0.5 * np.pi * D ** 2 / 4 + np.pi / 4 * L * D
    d["area_design_m2"] = np.where(S > L, np.maximum(cyl, 0.25 * L * S), cyl)
    d["am_design"] = d["area_design_m2"] / d["mass"]
    d["mass_uncertain"] = d["norad"].map(g["MassFlag"]).eq("?")            # as in the training cohort
    # catalogue history: GCAT canonical orbit and its epoch (day-precise only)
    od = d["norad"].map(g["ODate"]).map(parse_vague)
    d["t0_canonical"] = pd.to_datetime([p[0] if p[1] <= 1.0 else pd.NaT for p in od])
    d["perigee0"] = d["norad"].map(_num(g["Perigee"]))
    d["apogee0"] = d["norad"].map(_num(g["Apogee"]))
    d["orbit_uncertain"] = False                                          # current CelesTrak orbit, not GCAT's flagged one
    sw = load_space_weather()
    i = sw.index.searchsorted(t0, side="left") - 1                        # last complete day before as_of
    for c in ("f107_last81", "f107_last365", "ap_last81", "f107_trend365"):
        d[c] = float(sw[c].iloc[i])
    # family evidence as of as_of: siblings from the lifetime cohort (catalogue history)
    hist = pd.read_csv(os.path.join(feature_dir("reentry", "processed"), "lifetime_cohort_2026-10-27.csv.gz"),
                       usecols=["norad", "Launch_Tag", "type1", "Parent", "t0", "t_end", "event", "time_days"],
                       parse_dates=["t0", "t_end"])
    d["launch_family"] = d["Launch_Tag"]
    d["breakup_family"] = np.where(d["type1"].eq("D"), d["Parent"].astype(str).str.replace("*", "", regex=False), "")
    hist = hist[~hist["norad"].isin(set(d["norad"]))]           # current objects enter once, as of as_of
    hist["launch_family"] = hist["Launch_Tag"]
    hist["breakup_family"] = np.where(hist["type1"].eq("D"), hist["Parent"].astype(str).str.replace("*", "", regex=False), "")
    combo = pd.concat([hist.assign(_cur=False), d[["norad", "launch_family", "breakup_family", "t0"]].assign(
        _cur=True, event=0, time_days=0.0, t_end=pd.NaT)], ignore_index=True)
    for fam in ("launch_family", "breakup_family"):
        feats = family_features(combo, fam)
        cur = combo["_cur"].to_numpy()
        for c, v in feats.items():
            d[f"{fam}_{c}"] = v[cur]
    return d.reset_index(drop=True)


def horizon_labels(df: pd.DataFrame, h: float) -> np.ndarray:
    return np.where(df["event"].eq(1) & (df["time_days"] <= h), 1.0, np.where(df["time_days"] >= h, 0.0, np.nan))


def physics_prior(df_all: pd.DataFrame, train: pd.DataFrame, tr_end) -> dict:
    """Class pools of the physics expert as validated (uncontrolled training population), plus the
    payload design-A/m pool from the payloads of the same training period: the twin also
    forecasts payloads, which must not fall back to the fragment breakup law."""
    prior = PH.class_prior_from(train)
    pay = df_all[df_all["type1"].eq("P") & df_all["time_days"].notna() & (df_all["t0"] < tr_end)]
    prior["am"].update({k: v for k, v in PH.class_prior_from(pay)["am"].items() if k == "P"})
    return prior


def physics_prior_as_of(as_of: str) -> dict:
    """The physics prior that ``fit_as_of`` builds, without fitting the learned expert."""
    from ..lifetime import run as LR
    O = pd.Timestamp(as_of)
    df_all = LR.load_cohort()
    tr_end, va_end = O - pd.DateOffset(years=7), O - pd.DateOffset(years=5)
    df = df_all[df_all["type1"].isin(LR.POPULATIONS["uncontrolled"]) & df_all["time_days"].notna()]
    dev = censor_at(df[df["t0"] < va_end], O)
    return physics_prior(df_all, LR.add_targets(dev[dev["t0"] < tr_end]).reset_index(drop=True), tr_end)


def fit_as_of(as_of: str, learner: str, params: dict, seeds=(0, 1, 2, 3, 4), n_samples: int = 32,
              horizons: dict = HORIZONS) -> dict:
    """Re-entry module for uncontrolled objects, refitted with every outcome known at ``as_of``
    in the configuration validated by the lifetime study (see the module docstring)."""
    from sklearn.isotonic import IsotonicRegression
    from ..lifetime import run as LR
    O = pd.Timestamp(as_of)
    df_all = LR.load_cohort()
    tr_end, va_end = O - pd.DateOffset(years=7), O - pd.DateOffset(years=5)
    df = df_all[df_all["type1"].isin(LR.POPULATIONS["uncontrolled"]) & df_all["time_days"].notna()]
    dev = censor_at(df[df["t0"] < va_end], O)
    train = LR.add_targets(dev[dev["t0"] < tr_end]).reset_index(drop=True)
    prior = physics_prior(df_all, train, tr_end)
    tune = LR.add_targets(dev[dev["t0"] >= tr_end]).reset_index(drop=True)
    fam = _md5_fold(tune["launch_family"], 2)
    S = {"train": train, "tuneA": tune[fam == 0].reset_index(drop=True), "tuneB": tune[fam == 1].reset_index(drop=True)}
    feat, qp, src = {}, {}, {}
    for k, part in S.items():
        r = PH.predict(part, prior, n_samples=n_samples, seed=0)
        feat[k], qp[k], src[k] = MO.physics_features(r["samples_days"]), MO._monotone(MO.physics_quantiles(r["samples_days"])), r["am_source"]
    X, shapes = {}, None
    for k in S:
        X[k], shapes = MO.design_matrix(S[k], feat[k], LR.FULL, shapes)
    cols = list(X["train"].columns)
    X = {k: MO.align_columns(v, cols) for k, v in X.items()}
    m = {k: S[k]["known5"].to_numpy() for k in S}
    z = {k: S[k]["z"].to_numpy() for k in S}
    fit = lambda s, rows: MO.LEARNERS[learner](seed=s).fit(X["train"][rows], z["train"][rows], X["tuneA"][m["tuneA"]],  # noqa: E731
                                                           z["tuneA"][m["tuneA"]], params)
    models = [fit(s, m["train"]) for s in seeds]
    fold = _md5_fold(train["launch_family"], 5)
    oof = np.full((len(train), len(MO.QUANTILES)), np.nan)
    for k in range(5):
        oof[fold == k] = fit(0, (fold != k) & m["train"]).predict(X["train"][fold == k])
    R = {k: FU.reliability_features(S[k], feat[k], src[k]) for k in S}
    gate = FU.Gate().fit(R["train"][m["train"]], qp["train"][m["train"]], oof[m["train"]], z["train"][m["train"]])
    M = {"prior": prior, "shapes": shapes, "cols": cols, "models": models, "gate": gate, "learner": learner,
         "train_n": int(m["train"].sum()), "tuneB_n": int(m["tuneB"].sum())}
    qB = fuse(M, S["tuneB"], feat["tuneB"], qp["tuneB"], src["tuneB"], X["tuneB"])
    G = FU.mondrian_groups(S["tuneB"])
    M["cqr"] = FU.MondrianCQR().fit(qB[m["tuneB"]], z["tuneB"][m["tuneB"]], G[m["tuneB"]].reset_index(drop=True))
    M["iso"] = {}
    for hk, h in horizons.items():
        y = horizon_labels(S["tuneB"], h)
        p, ok = FU.horizon_prob(qB, h), ~np.isnan(y)
        M["iso"][hk] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[ok], y[ok])
    return M


def fuse(M: dict, df: pd.DataFrame, pf: pd.DataFrame, qp: np.ndarray, src: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    qm = np.sort(np.mean([mm.predict(X) for mm in M["models"]], 0), 1)
    return MO._monotone(M["gate"].fuse(FU.reliability_features(df, pf, src), qp, qm))


def study_learner() -> tuple:
    """The learner selected in the lifetime study (origin 2016) and its tuned settings."""
    s = json.load(open(os.path.join(output_dir("benchmarks", "reentry_lifetime", "origin2016_uncontrolled"), "summary.json")))
    best = s["best_learner_on_tuneA"]
    return best, s["fits"][best]["params"]


def forecast(as_of: str, learner: str, params: dict, n_samples: int = 32, seeds=(0, 1, 2, 3, 4),
             horizons: dict = HORIZONS) -> tuple:
    """Per-object re-entry forecasts at ``as_of`` for every variant, and the twin's choice.

    Variants: ``lifetime`` (the validated lifetime pipeline; uncontrolled objects), ``physics_prior``
    (physics expert with the class priors), ``physics_history`` (physics expert with the
    area-to-mass posterior conditioned on the object's catalogue history; uncontrolled objects).
    The twin's choice, fixed before any test: physics_history for fragments, rocket stages and
    components (objects in orbit today have survived since their canonical orbit, which the
    lifetime pipeline, fitted at canonical-orbit epochs, does not know); physics_prior for
    non-operational and unknown-status payloads (their past manoeuvres make the decay history
    uninformative; C3 tests this reading); the previous-window payload base rate for operational
    payloads (C3 policy).  Returns (table, M, info)."""
    from ..lifetime import run as LR
    M = fit_as_of(as_of, learner, params, seeds=seeds, n_samples=n_samples, horizons=horizons)
    d = current_state(as_of)
    unc = d["type1"].isin(LR.POPULATIONS["uncontrolled"]).to_numpy()
    r2 = PH.predict(d, M["prior"], n_samples=n_samples, seed=0)
    pf = MO.physics_features(r2["samples_days"])
    qp = MO._monotone(MO.physics_quantiles(r2["samples_days"]))
    du = d[unc].reset_index(drop=True)
    X = MO.align_columns(MO.design_matrix(du, pf[unc].reset_index(drop=True), LR.FULL, M["shapes"])[0], M["cols"])
    q1 = fuse(M, du, pf[unc].reset_index(drop=True), qp[unc], r2["am_source"][unc], X)
    lo80, hi80 = M["cqr"].interval(q1, FU.mondrian_groups(du), 0.8)
    lo90, hi90 = M["cqr"].interval(q1, FU.mondrian_groups(du), 0.9)
    post, hinfo = PH.history_posterior(du, M["prior"], as_of, load_space_weather(), n_out=n_samples)
    r3 = PH.predict(du, M["prior"], log_am=post)
    snap = pd.read_csv(os.path.join(snapshot_dir(as_of), "satcat.csv"))
    base = {k: {"perigee_lt_600": previous_window_rate(snap, as_of, h, (0.0, 600.0)),
                "perigee_600_2000": previous_window_rate(snap, as_of, h, (600.0, 2000.0))} for k, h in horizons.items()}
    regime = np.array([ObjectTwin(object_id="", name="", object_class=c, as_of=as_of, orbit={}, physical={},
                                  operational_status=None if pd.isna(s) else str(s), family={}, space_weather={}).regime()
                       for c, s in zip(d["type1"], d["ops_status"])])
    T = d[["norad", "JCAT", "Name", "type1", "ops_status", "perigee", "apogee", "inc", "Launch_Tag", "breakup_family",
           "launch_family", "breakup_family_n_decayed", "mass", "rcs_m2", "f107_last81", "ap_last81", "t0_canonical"]].copy()
    T["regime"], T["am_source"] = regime, r2["am_source"]
    iu = np.flatnonzero(unc)
    T["history_kind"], T["history_ess"], T["history_days"] = "", np.nan, np.nan
    T.loc[iu, "history_kind"] = hinfo["kind"].to_numpy()
    T.loc[iu, "history_ess"] = hinfo["ess"].to_numpy()
    T.loc[iu, "history_days"] = hinfo["history_days"].to_numpy()
    med = {"physics_prior": np.median(r2["samples_days"], 1)}
    med["lifetime"], med["physics_history"] = np.full(len(d), np.nan), np.full(len(d), np.nan)
    med["lifetime"][iu] = 10 ** q1[:, 3]
    med["physics_history"][iu] = np.median(r3["samples_days"], 1)
    for k, h in horizons.items():
        T[f"p_physics_prior_{k}"] = (r2["samples_days"] <= h).mean(1)
        T[f"p_lifetime_{k}"], T[f"p_physics_history_{k}"], T[f"p_lifetime_raw_{k}"] = np.nan, np.nan, np.nan
        T.loc[iu, f"p_lifetime_raw_{k}"] = FU.horizon_prob(q1, h)          # for rank metrics (isotonic ties)
        T.loc[iu, f"p_lifetime_{k}"] = M["iso"][k].predict(FU.horizon_prob(q1, h))
        T.loc[iu, f"p_physics_history_{k}"] = (r3["samples_days"] <= h).mean(1)
        band = np.where(T["perigee"] < 600, "perigee_lt_600", "perigee_600_2000")
        T[f"p_{k}"] = np.where(unc, T[f"p_physics_history_{k}"],
                               np.where(regime == "operator_controlled", [base[k][b] for b in band], T[f"p_physics_prior_{k}"]))
    for v, x in med.items():
        T[f"median_days_{v}"] = x
    # 80/90 % intervals: Mondrian CQR for the lifetime pipeline, Monte-Carlo quantiles for physics
    T["lifetime_lo90_days"], T["lifetime_hi90_days"] = np.nan, np.nan
    T.loc[iu, "lifetime_lo90_days"], T.loc[iu, "lifetime_hi90_days"] = 10 ** lo90, 10 ** hi90
    T["lifetime_lo80_days"], T["lifetime_hi80_days"] = np.nan, np.nan
    T.loc[iu, "lifetime_lo80_days"], T.loc[iu, "lifetime_hi80_days"] = 10 ** lo80, 10 ** hi80
    s_hist = np.full((len(d), n_samples), np.nan)
    s_hist[iu] = r3["samples_days"]
    S_tw = np.where(unc[:, None], s_hist, r2["samples_days"])
    T["lo90_days"], T["hi90_days"] = np.quantile(S_tw, 0.05, 1), np.quantile(S_tw, 0.95, 1)
    T["lo80_days"], T["hi80_days"] = np.quantile(S_tw, 0.10, 1), np.quantile(S_tw, 0.90, 1)
    T["median_days"] = np.median(S_tw, 1)
    T.loc[regime == "operator_controlled", ["median_days", "lo90_days", "hi90_days", "lo80_days", "hi80_days"]] = np.nan
    return T, M, {"base_rates": base, "history_kinds": hinfo["kind"].value_counts().to_dict()}


VARIANT_LABEL = {"physics_history": "physics expert, area-to-mass conditioned on the catalogue history",
                 "physics_prior": "physics expert (NRLMSIS 2.1, Monte-Carlo A/m from class priors)",
                 "lifetime": "physics + learned, reliability-gated (lifetime pipeline)"}
VALIDATION_T1 = "prospective test T1 (72 d from 2026-08-16); fixed before the test"


def run(as_of: str, learner: str | None = None, params: dict | None = None, n_samples: int = 32, out=None,
        seeds=(0, 1, 2, 3, 4)):
    t0 = time.time()
    if learner is None:
        learner, params = study_learner()
    T, M, info = forecast(as_of, learner, params, n_samples=n_samples, seeds=seeds)
    od = output_dir("twin", f"as_of_{as_of}") if out is None else out
    os.makedirs(od, exist_ok=True)
    regimes = T["regime"].to_numpy()
    with open(os.path.join(od, "twins.jsonl"), "w") as f:
        for _, row in T.iterrows():
            tw = ObjectTwin(object_id=str(int(row["norad"])), name=str(row["Name"]), object_class=row["type1"], as_of=as_of,
                            orbit={"perigee_km": row["perigee"], "apogee_km": row["apogee"], "inc_deg": row["inc"],
                                   "canonical_epoch": None if pd.isna(row["t0_canonical"]) else str(row["t0_canonical"].date())},
                            physical={"mass_kg": row["mass"], "area_to_mass_source": row["am_source"], "rcs_m2": row["rcs_m2"]},
                            operational_status=None if pd.isna(row["ops_status"]) else str(row["ops_status"]),
                            family={"launch": row["launch_family"], "breakup_parent": row["breakup_family"],
                                    "breakup_siblings_decayed": float(row["breakup_family_n_decayed"])},
                            space_weather={"f107_last81": row["f107_last81"], "ap_last81": row["ap_last81"]})
            reg, unc_i = row["regime"], row["type1"] in ("D", "R", "C")
            vals = {f"p_reentry_{k}": float(row[f"p_{k}"]) for k in HORIZONS}
            if reg == "operator_controlled":
                vals.update({f"drag_only_p_reentry_{k}": float(row[f"p_physics_prior_{k}"]) for k in HORIZONS})
                tw.attach(ModuleOutput("reentry", "previous-window payload base rate", VALIDATION["payload_controlled"],
                                       vals, {}, abstain=True,
                                       reason="operational payload: operator-controlled regime, drag physics does not determine re-entry (C3)"))
            else:
                vals.update({"median_days": float(row["median_days"]), "capped_at_days": PH.CAP_DAYS})
                unc_d = {"interval80_days": [float(row["lo80_days"]), float(row["hi80_days"])],
                         "interval90_days": [float(row["lo90_days"]), float(row["hi90_days"])],
                         "interval_kind": "physics Monte-Carlo quantiles (not conformalised)"}
                if unc_i:
                    unc_d.update({"history_kind": row["history_kind"], "history_ess": float(row["history_ess"])})
                    tw.attach(ModuleOutput("reentry", VARIANT_LABEL["physics_history"], VALIDATION_T1, vals, unc_d))
                    tw.attach(ModuleOutput("reentry-comparison", VARIANT_LABEL["lifetime"], VALIDATION["uncontrolled"],
                                           {f"p_reentry_{k}": float(row[f"p_lifetime_{k}"]) for k in HORIZONS} |
                                           {"median_days": float(row["median_days_lifetime"])},
                                           {"interval90_days": [float(row["lifetime_lo90_days"]), float(row["lifetime_hi90_days"])],
                                            "interval_kind": "Mondrian CQR"}))
                else:
                    tw.attach(ModuleOutput("reentry", VARIANT_LABEL["physics_prior"],
                                           VALIDATION["payload_natural" if reg == "natural_decay" else "payload_unknown"], vals, unc_d))
            f.write(json.dumps(tw.to_dict(), default=float) + "\n")
    nat, unk = regimes == "natural_decay", regimes == "unknown"
    unc = T["type1"].isin(["D", "R", "C"]).to_numpy()
    frag = T["type1"].eq("D").to_numpy() & T["breakup_family"].astype(str).ne("").to_numpy()
    clouds = []
    for parent, idx in T[frag].groupby("breakup_family").indices.items():
        idx = np.flatnonzero(frag)[idx]
        if len(idx) < 30:
            continue
        clouds.append({"breakup_parent": parent, "fragments_in_orbit": int(len(idx)),
                       **{f"expected_remaining_{k}": float(len(idx) - T[f"p_{k}"].to_numpy()[idx].sum()) for k in HORIZONS}})
    pd.DataFrame(clouds).sort_values("fragments_in_orbit", ascending=False).to_csv(os.path.join(od, "c2_clouds.csv"), index=False)
    T.to_csv(os.path.join(od, "forecast_table.csv.gz"), index=False)
    summary = {"as_of": as_of, "objects": int(len(T)),
               "regime_counts": {"natural_decay": int(nat.sum()), "operator_controlled": int((regimes == "operator_controlled").sum()),
                                 "unknown": int(unk.sum())},
               "natural_decay_by_class": pd.Series(T["type1"].to_numpy()[nat]).value_counts().to_dict(),
               "learner": learner, "seeds": len(M["models"]), "train_n": M["train_n"], "tuneB_n": M["tuneB_n"],
               "expected_reentries_natural_decay": {k: float(T[f"p_{k}"].to_numpy()[nat].sum()) for k in HORIZONS},
               "expected_reentries_unknown_regime": {k: float(T[f"p_{k}"].to_numpy()[unk].sum()) for k in HORIZONS},
               "expected_reentries_uncontrolled_by_variant": {v: {k: float(np.nansum(T[f"p_{v}_{k}"].to_numpy()[unc])) for k in HORIZONS}
                                                             for v in ("physics_history", "physics_prior", "lifetime")},
               "history_kinds": info["history_kinds"], "payload_base_rates_previous_window": info["base_rates"],
               "objects_p30d_above_0.5_natural_decay": int(((T["p_30d"].to_numpy() > 0.5) & nat).sum()),
               "clouds_with_30_fragments": len(clouds), "seconds": time.time() - t0,
               "verification": "compare with decays in a later CelesTrak SATCAT snapshot"}
    with open(os.path.join(od, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    T[nat].sort_values("p_30d", ascending=False).head(50)[
        ["norad", "Name", "type1", "perigee", "apogee", "p_30d", "median_days", "lo90_days", "hi90_days"]].to_csv(
        os.path.join(od, "top50_reentry_30d.csv"), index=False)
    h = hashlib.sha256(open(os.path.join(od, "twins.jsonl"), "rb").read()).hexdigest()
    with open(os.path.join(od, "FORECAST_SHA256.txt"), "w") as f:
        f.write(h + "  twins.jsonl\n")
    print(json.dumps(summary, indent=1, default=float))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-10-27")
    a = ap.parse_args(argv)
    run(a.as_of)


if __name__ == "__main__":
    main()
