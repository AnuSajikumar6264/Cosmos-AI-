"""Cross-module couplings of the COSMOS-AI digital twin, and their tests on real data.

    python -m cosmos.ai.twin.couplings c2      # re-entry -> debris environment (cloud survival)
    python -m cosmos.ai.twin.couplings c3      # health -> re-entry regime (prospective)
    python -m cosmos.ai.twin.couplings c4      # health x debris -> collision response (demonstration)

C2  Each fragment's predicted time-to-re-entry distribution (lifetime module, test fold) gives a
    survival probability S_i(tau).  The expected number of a cloud's fragments still in orbit
    at ``k`` years after the cloud's first catalogued fragment is sum_i S_i(D_k - t0_i) over the
    fragments catalogued by D_k; it is compared with the catalogue count.  Fragments are those
    of the lifetime test folds, so no fragment outcome was used in fitting.  S_i is read in two
    ways: *raw*, by linear interpolation of the quantile function (whose lower tail spreads 5 %
    of the mass evenly in log-time down to 0.5 day, so even a fragment predicted to outlive the
    cap gets S of about 0.96 at 1 yr), and *calibrated*, by isotonic maps of P(T <= a) fitted on
    the lifetime study's tune-B fragments at a grid of ages ``a`` and interpolated in log-age.

C3  Operational status is the catalogue-level health evidence available for every payload
    (CelesTrak OPS_STATUS_CODE: '+' operational, 'P' partially operational, '-' non-operational).
    The twin's rule: an operational payload can manoeuvre, so drag physics does not determine
    its re-entry; a non-operational payload decays naturally.  As-of date: the 2026-08 SATCAT
    snapshot (data date 2026-08-16); outcomes: decay dates in the 2026-10-27 snapshot.
    *Prospective* policies use only information available at tau: drag physics for every payload
    (status-blind), the previous-window payload base rate for every payload (status-blind), and
    the status-aware twin (base rate for operational payloads, physics otherwise).  The Platt
    recalibrations (status-blind and status-specific) are fitted on outcomes of the follow-up
    window itself (cross-fitted in two folds by launch); they are reported as an in-window
    diagnostic of attainable calibration, not as forecasts.  Confidence intervals resample
    launches (outcomes cluster by launch).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from ..evaluation import brier_score, expected_calibration_error

TAU_C3 = "2026-08-16"
FOLLOW_C3 = "2026-10-27"
AGE_GRID = np.geomspace(10.0, 1826.25, 25)          # days; C2 survival calibration ages


def launch_fold(keys: pd.Series, k: int) -> np.ndarray:
    """Deterministic fold of each launch (md5, as everywhere else in the project)."""
    return keys.astype(str).map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % k).to_numpy()


# ------------------------------------------------------------------------------ C2
def survival_from_quantiles(q: np.ndarray, tau_days: np.ndarray) -> np.ndarray:
    """Raw S(tau) = P(T > tau) per object from its quantiles of log10 capped lifetime."""
    from ..lifetime.fusion import horizon_prob
    out = np.empty((len(q), len(tau_days)))
    for j, t in enumerate(tau_days):
        out[:, j] = 1 - horizon_prob(q, max(float(t), 0.51))
    return out


def survival_calibrator(q: np.ndarray, time_days: np.ndarray, event: np.ndarray) -> list:
    """Isotonic maps raw P(T <= a) -> observed, one per age in AGE_GRID (calibration objects)."""
    from sklearn.isotonic import IsotonicRegression
    from ..lifetime.fusion import horizon_prob
    maps = []
    for a in AGE_GRID:
        y = np.where((event == 1) & (time_days <= a), 1.0, np.where(time_days >= a, 0.0, np.nan))
        p, ok = horizon_prob(q, a), ~np.isnan(y)
        maps.append(IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[ok], y[ok]))
    return maps


def calibrated_survival(maps: list, q: np.ndarray, age_days: np.ndarray) -> np.ndarray:
    """S(age) per object: calibrated P(T <= a) at the two neighbouring grid ages, interpolated
    linearly in log-age (ages outside the grid are clipped to it)."""
    from ..lifetime.fusion import horizon_prob
    lg = np.log(AGE_GRID)
    la = np.log(np.clip(age_days, AGE_GRID[0], AGE_GRID[-1]))
    j = np.clip(np.searchsorted(lg, la) - 1, 0, len(AGE_GRID) - 2)
    w = (la - lg[j]) / (lg[j + 1] - lg[j])
    F = np.empty(len(q))
    for jj in np.unique(j):
        s = j == jj
        f0 = maps[jj].predict(horizon_prob(q[s], AGE_GRID[jj]))
        f1 = maps[jj + 1].predict(horizon_prob(q[s], AGE_GRID[jj + 1]))
        F[s] = (1 - w[s]) * f0 + w[s] * f1
    return 1 - F


def c2_cloud_survival(origin: str = "2016", years=(1, 2, 3, 4, 5), min_fragments: int = 30) -> pd.DataFrame:
    from ..lifetime import run as LR
    pdir = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_uncontrolled")
    P = np.load(os.path.join(pdir, "test_predictions.npz"))
    B = np.load(os.path.join(pdir, "tuneB_predictions.npz"), allow_pickle=True)
    S = LR.split(LR.load_cohort(), origin, "uncontrolled")
    t = S["test"].reset_index(drop=True)
    models = sorted({k.split("__")[0] for k in P.files if k.endswith("__q")} & {k.split("__")[0] for k in B.files})
    fragB = B["type1"] == "D"                                   # calibrate on tune-B fragments
    maps = {m: survival_calibrator(B[f"{m}__q"][fragB], B["time_days"][fragB], B["event"][fragB]) for m in models}
    d = t[t["type1"].eq("D") & t["breakup_family"].astype(str).ne("")]
    rows = []
    t_end = t["t0"] + pd.to_timedelta(t["time_days"], unit="D")
    for fam, idx in d.groupby("breakup_family").indices.items():
        idx = d.index[idx]
        if len(idx) < min_fragments:
            continue
        start = t.loc[idx, "t0"].min()
        for k in years:
            Dk = start + pd.Timedelta(days=365.25 * k)
            cat = idx[t.loc[idx, "t0"] <= Dk]
            if len(cat) == 0:
                continue
            age = (Dk - t.loc[cat, "t0"]).dt.total_seconds().to_numpy() / 86400.0
            if age.max() > LR.CAP + 1:
                continue                                   # beyond the 5-year predictive cap
            decayed = (t.loc[cat, "event"].eq(1) & (t_end[cat] <= Dk)).to_numpy()
            censored_before = (t.loc[cat, "event"].eq(0) & (t_end[cat] < Dk)).to_numpy()
            obs = int((~decayed & ~censored_before).sum())
            r = {"origin": origin, "cloud": fam, "years": k, "catalogued": int(len(cat)), "observed_in_orbit": obs}
            for m in models:
                q = P[f"{m}__q"][cat]
                s = np.array([survival_from_quantiles(q[i:i + 1], np.array([age[i]]))[0, 0] for i in range(len(cat))])
                r[f"pred_{m}"] = float(s[~censored_before].sum())
                r[f"predcal_{m}"] = float(calibrated_survival(maps[m], q, age)[~censored_before].sum())
            rows.append(r)
    out = pd.DataFrame(rows)
    return out


def c2_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    preds = [c for c in df.columns if c.startswith(("pred_", "predcal_"))]
    for c in preds:
        d = df[df[c].notna()]                       # a model may exist at one origin only (ablations)
        err = d[c] - d["observed_in_orbit"]
        frac = err / d["catalogued"]
        rows.append({"model": c.split("_", 1)[1], "reading": "calibrated" if c.startswith("predcal_") else "raw",
                     "origins": ",".join(sorted(d["origin"].astype(str).unique())),
                     "n_cloud_years": int(len(d)), "n_clouds": int(d["cloud"].nunique()), "mae_count": float(err.abs().mean()),
                     "mae_fraction": float(frac.abs().mean()), "bias_fraction": float(frac.mean()),
                     "within_10pct": float((frac.abs() <= 0.10).mean())})
    return pd.DataFrame(rows).sort_values(["reading", "mae_fraction"])


C2_PAIRS = (("pred_physics_only", "predcal_tabicl"), ("pred_physics_only", "predcal_physics_only"),
            ("pred_cosmos_gated", "predcal_cosmos_gated"), ("pred_tabicl", "predcal_tabicl"),
            ("pred_physics_only", "pred_cosmos_gated"))


def c2_bootstrap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 20261027) -> pd.DataFrame:
    """Cloud-cluster bootstrap (the cloud-years of one cloud are resampled together): CI of each
    mean absolute fraction error, and paired CIs of differences for C2_PAIRS (< 0: first better)."""
    from ..lifetime import run as LR
    err = {c: ((df[c] - df["observed_in_orbit"]) / df["catalogued"]).abs().to_numpy()
           for c in df.columns if c.startswith(("pred_", "predcal_"))}
    clouds = (df["origin"].astype(str) + "/" + df["cloud"].astype(str)).to_numpy()
    rows = []
    for c, e in err.items():
        ok = np.isfinite(e)
        if ok.sum() < 10:
            continue
        lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(e[ok][idx].mean()), clouds[ok], n_boot=n_boot, seed=seed)
        rows.append({"comparison": c, "mae_fraction": float(e[ok].mean()), "ci95_lo": lo, "ci95_hi": hi})
    for a, b in C2_PAIRS:
        if a in err and b in err:
            ok = np.isfinite(err[a]) & np.isfinite(err[b])
            dv = err[a][ok] - err[b][ok]
            lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(dv[idx].mean()), clouds[ok], n_boot=n_boot, seed=seed)
            rows.append({"comparison": f"{a} - {b}", "mae_fraction": float(dv.mean()), "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ C3
def c3_dataset() -> pd.DataFrame:
    from ..lifetime.catalogue import load_gcat, load_space_weather, _num
    a = pd.read_csv(os.path.join(feature_dir("debris", "celestrak"), "satcat.csv"))
    b = pd.read_csv(os.path.join(feature_dir("debris", "celestrak", "snapshots", "2026-10-27"), "satcat.csv"),
                    usecols=["NORAD_CAT_ID", "DECAY_DATE"])
    tau, fol = pd.Timestamp(TAU_C3), pd.Timestamp(FOLLOW_C3)
    a = a[a["DECAY_DATE"].isna() & a["OBJECT_TYPE"].eq("PAY") & a["ORBIT_CENTER"].eq("EA")
          & (pd.to_datetime(a["LAUNCH_DATE"]) <= tau)]
    a = a[a["PERIGEE"].notna() & (a["PERIGEE"] < 600)].copy()
    dec = b.set_index("NORAD_CAT_ID")["DECAY_DATE"]
    a["decay"] = pd.to_datetime(a["NORAD_CAT_ID"].map(dec), errors="coerce")
    a = a[~(a["decay"] <= tau)]                                      # decayed before tau, reported late
    a["event"] = (a["decay"].notna() & (a["decay"] <= fol)).astype(int)
    a["time_days"] = np.where(a["event"].eq(1), (a["decay"] - tau).dt.days, (fol - tau).days).astype(float)
    g = load_gcat()
    g["Satcat_n"] = pd.to_numeric(g["Satcat"], errors="coerce")
    g = g.drop_duplicates("Satcat_n").set_index("Satcat_n")
    for c in ("Mass", "Length", "Diameter", "Span"):
        v = _num(g[c])
        v[v <= 0] = np.nan
        a[c.lower()] = a["NORAD_CAT_ID"].map(v)
    a["gcat_status"] = a["NORAD_CAT_ID"].map(g["Status"])            # outcome-side: used only to describe
    L, D, S = a["length"], a["diameter"], a["span"]
    cyl = 0.5 * np.pi * D ** 2 / 4 + np.pi / 4 * L * D
    a["area_design_m2"] = np.where(S > L, np.maximum(cyl, 0.25 * L * S), cyl)
    a["am_design"] = a["area_design_m2"] / a["mass"]
    sw = load_space_weather()
    i = sw.index.searchsorted(tau, side="left") - 1                  # last complete day before tau
    a["f107_last81"], a["ap_last81"] = float(sw["f107_last81"].iloc[i]), float(sw["ap_last81"].iloc[i])
    a = a.rename(columns={"PERIGEE": "perigee", "APOGEE": "apogee", "INCLINATION": "inc", "RCS": "rcs_m2"})
    a["norad"] = a["NORAD_CAT_ID"]                                   # key of the physics expert's random streams
    a["type1"] = "P"
    a["parent_is_rocket"] = False
    a["status_group"] = a["OPS_STATUS_CODE"].map({"+": "operational", "P": "operational", "B": "operational",
                                                  "S": "operational", "X": "operational", "-": "non-operational"}).fillna("unknown")
    return a.reset_index(drop=True)


def previous_window_rate(sc: pd.DataFrame, end, days: float, perigee=(0.0, 600.0)) -> float:
    """Share of Earth-orbiting payloads in orbit at ``end - days`` (perigee band from the snapshot)
    that decayed in (end - days, end]: a base rate that uses only outcomes before ``end``."""
    end = pd.Timestamp(end)
    w0 = end - pd.Timedelta(days=days)
    dec = pd.to_datetime(sc["DECAY_DATE"], errors="coerce")
    h = sc[sc["OBJECT_TYPE"].eq("PAY") & sc["ORBIT_CENTER"].eq("EA") & sc["PERIGEE"].ge(perigee[0])
           & sc["PERIGEE"].lt(perigee[1]) & (pd.to_datetime(sc["LAUNCH_DATE"], errors="coerce") <= w0)
           & (dec.isna() | (dec > w0))]
    d = dec[h.index]
    return float(((d > w0) & (d <= end)).mean())


def c3_run(n_samples: int = 32, n_boot: int = 2000) -> dict:
    from ..lifetime import physics as PH
    from ..lifetime import run as LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    a = c3_dataset()
    H = float((pd.Timestamp(FOLLOW_C3) - pd.Timestamp(TAU_C3)).days)
    cohort = pd.read_csv(os.path.join(feature_dir("reentry", "processed"), "lifetime_cohort_2026-10-27.csv.gz"),
                         usecols=["type1", "rcs_m2", "am_design", "t0"], parse_dates=["t0"])
    prior = PH.class_prior_from(cohort[cohort["t0"] < TAU_C3])
    r = PH.predict(a, prior, n_samples=n_samples, seed=0, cap_days=400.0)
    p_phys = (r["samples_days"] <= H).mean(1)
    a["p_physics"] = p_phys
    a["am_source"] = r["am_source"]
    y = a["event"].to_numpy()
    launch = a["OBJECT_ID"].astype(str).str[:8]
    out = {"tau": TAU_C3, "follow_up": FOLLOW_C3, "horizon_days": H, "n": int(len(a)), "events": int(y.sum()),
           "launches": int(launch.nunique()), "launches_with_events": int(launch[y == 1].nunique()),
           "am_source": pd.Series(r["am_source"]).value_counts().to_dict(), "by_status": {}}
    for grp, sub in a.groupby("status_group"):
        yy, pp = sub["event"].to_numpy(), sub["p_physics"].to_numpy()
        e = {"n": int(len(sub)), "events": int(yy.sum()), "event_rate": float(yy.mean()),
             "mean_p_physics": float(pp.mean()), "brier_physics": brier_score(yy, pp),
             "ece_physics": expected_calibration_error(yy, pp),
             "gcat_end_status_of_decays": sub.loc[sub["event"].eq(1), "gcat_status"].value_counts().to_dict(),
             "by_am_source": {s: {"n": int(len(v)), "events": int(v["event"].sum()), "mean_p_physics": float(v["p_physics"].mean())}
                              for s, v in sub.groupby("am_source")}}
        if 0 < yy.sum() < len(yy):
            e.update({"auc_roc_physics": float(roc_auc_score(yy, pp)), "auc_pr_physics": float(average_precision_score(yy, pp))})
        out["by_status"][grp] = e
    # ---- prospective policies (information available at tau only).  The base rate is the decay
    # share of LEO payloads over the window of the same length that ended at tau; the decayed
    # payloads' status at the start of that window is not in the snapshot, so it is the rate of
    # all payloads, applied by the status-aware twin to the operational ones
    prev = pd.read_csv(os.path.join(feature_dir("debris", "celestrak"), "satcat.csv"))
    base = previous_window_rate(prev, TAU_C3, H)
    op = a["status_group"].eq("operational").to_numpy()
    policies = {"physics_status_blind": p_phys, "base_rate_status_blind": np.full(len(a), base),
                "twin_status_aware_baserate": np.where(op, base, p_phys)}
    # ---- in-window diagnostic (NOT prospective): Platt maps on logit(p_physics) fitted on the
    # follow-up window's outcomes, cross-fitted in two folds by launch (no payload is calibrated on
    # itself or on a launch sibling), status-blind and one map per health status
    fold = launch_fold(launch, 2)
    lg = np.log(np.clip(p_phys, 1e-4, 1 - 1e-4) / (1 - np.clip(p_phys, 1e-4, 1 - 1e-4)))
    grp = a["status_group"].to_numpy()
    p_cal, p_blindcal = np.empty(len(a)), np.empty(len(a))
    for k in (0, 1):
        m = LogisticRegression(C=1e3).fit(lg[fold != k, None], y[fold != k])
        p_blindcal[fold == k] = m.predict_proba(lg[fold == k, None])[:, 1]
        for gname in np.unique(grp):
            fit, app = (fold != k) & (grp == gname), (fold == k) & (grp == gname)
            if app.sum() == 0:
                continue
            if fit.sum() < 20 or y[fit].min() == y[fit].max():
                p_cal[app] = p_phys[app]
                continue
            m = LogisticRegression(C=1e3).fit(lg[fit, None], y[fit])
            p_cal[app] = m.predict_proba(lg[app, None])[:, 1]
    in_window = {"physics_recalibrated_status_blind": p_blindcal, "twin_status_aware_recalibrated": p_cal}
    out["payload_base_rate_previous_window"] = base
    for kind, pols in (("prospective", policies), ("in_window_cross_fitted", in_window)):
        out[kind] = {}
        for name, p in pols.items():
            e = {"brier": brier_score(y, p), "ece": expected_calibration_error(y, p), "mean_p": float(p.mean())}
            if np.ptp(p) > 0:
                e.update({"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p))})
            out[kind][name] = e
    out["event_rate"] = float(y.mean())
    # ---- launch-cluster bootstrap of Brier differences (< 0: the first policy is better)
    allp = {**policies, **in_window}
    sq = {k: (v - y) ** 2 for k, v in allp.items()}
    diffs = {"twin_status_aware_baserate_minus_physics_status_blind": ("twin_status_aware_baserate", "physics_status_blind"),
             "twin_status_aware_baserate_minus_base_rate_status_blind": ("twin_status_aware_baserate", "base_rate_status_blind"),
             "in_window_recal_status_aware_minus_recal_blind": ("twin_status_aware_recalibrated", "physics_recalibrated_status_blind"),
             "in_window_recal_status_aware_minus_physics_raw": ("twin_status_aware_recalibrated", "physics_status_blind")}
    launch_arr = launch.to_numpy()
    out["brier_difference_ci95_launch_bootstrap"] = {}
    for name, (p1, p2) in diffs.items():
        dv = sq[p1] - sq[p2]
        lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(dv[idx].mean()), launch_arr, n_boot=n_boot)
        out["brier_difference_ci95_launch_bootstrap"][name] = {"mean": float(dv.mean()), "ci95": [lo, hi]}
    a["p_twin_status_aware_baserate"] = policies["twin_status_aware_baserate"]
    a["p_twin_recalibrated_in_window"] = p_cal
    a["p_recalibrated_blind_in_window"] = p_blindcal
    per = {}
    for gname in ("non-operational", "operational", "unknown"):
        s = grp == gname
        if not s.any():
            continue
        per[gname] = {pol: {"mean_p": float(p[s].mean()), "event_rate": float(y[s].mean()), "brier": brier_score(y[s], p[s]),
                            "ece": expected_calibration_error(y[s], p[s])} for pol, p in allp.items()}
    out["by_status_policy"] = per
    # Platt coefficients on all payloads (in-window; for transparency, not used by the twin)
    out["platt_in_window_full_fit"] = {}
    for gname in ["all"] + sorted(np.unique(grp)):
        s = np.ones(len(a), bool) if gname == "all" else grp == gname
        if s.sum() >= 20 and 0 < y[s].sum() < s.sum():
            m = LogisticRegression(C=1e3).fit(lg[s, None], y[s])
            out["platt_in_window_full_fit"][gname] = {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0, 0])}
    d = output_dir("benchmarks", "twin_couplings")
    a[["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_ID", "OPS_STATUS_CODE", "status_group", "perigee", "apogee", "event", "time_days",
       "p_physics", "p_twin_status_aware_baserate", "p_twin_recalibrated_in_window", "p_recalibrated_blind_in_window",
       "am_source", "gcat_status"]].to_csv(os.path.join(d, "c3_payloads.csv"), index=False)
    with open(os.path.join(d, "c3_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


# ------------------------------------------------------------------------------ C4 (demonstration)
def c4_escalation(p_high: float, risk_interval: tuple, regime: str, health_alarm: bool | None) -> dict:
    """Twin rule for the collision response of a payload.

    A forecast high-risk conjunction (calibrated P(high) >= 0.5, or the upper end of the 90 %
    interval at or above -6) normally triggers manoeuvre planning.  If the payload cannot be
    relied on to manoeuvre -- non-operational status, or an active health alarm -- the twin
    escalates instead: the conjunction is reported as one the operator may not be able to avoid.
    This rule is not validated: no public dataset pairs conjunction messages with the health of
    the same spacecraft."""
    high = (p_high >= 0.5) or (risk_interval[1] >= -6.0)
    if not high:
        action = "monitor"
    elif regime == "natural_decay" or health_alarm:
        action = "escalate: high-risk conjunction, manoeuvre capability uncertain"
    else:
        action = "plan avoidance manoeuvre"
    return {"high_risk": bool(high), "action": action, "validated": False}


def c4_demo() -> dict:
    """Apply the C4 rule to real module outputs that come from *different* objects (anonymised
    CDMs cannot be linked to catalogue objects): the highest-risk Kelvins test forecast attached to
    one operational and one non-operational payload twin.  Demonstration only."""
    from ..conjunction import run as CR
    P = np.load(os.path.join(output_dir("benchmarks", "conjunction_kelvins_cac"), "test_predictions.npz"))
    q = P["cosmos_gated__q"]
    p = CR.p_high(q)
    i = int(np.argmax(p))
    a = c3_dataset()
    # operational status does not imply propulsion (e.g. UoSAT-2 is operational without any);
    # the demonstration therefore uses a constellation satellite known to carry propulsion
    ops = a[a["status_group"].eq("operational")]
    star = ops[ops["OBJECT_NAME"].astype(str).str.contains("STARLINK", case=False)]
    op = (star if len(star) else ops).iloc[0]
    nop = a[a["status_group"].eq("non-operational")].iloc[0]
    out = {"note": "DEMONSTRATION: conjunction forecast and payload states come from different objects. "
                   "Operational status is used as a proxy for manoeuvre capability; it is not one "
                   "(many operational satellites carry no propulsion), and no public source used here "
                   "states propulsion capability.",
           "conjunction_test_event_index": i, "p_high_uncalibrated": float(p[i]),
           "final_risk_quantiles_log10": q[i].tolist(), "true_final_risk": float(P["y"][i])}
    for tag, row in (("operational_payload", op), ("non_operational_payload", nop)):
        reg = "operator_controlled" if row["status_group"] == "operational" else "natural_decay"
        out[tag] = {"norad": int(row["NORAD_CAT_ID"]), "name": str(row["OBJECT_NAME"]), "ops_status": str(row["OPS_STATUS_CODE"]),
                    **c4_escalation(float(p[i]), (float(q[i, 0]), float(q[i, -1])), reg, None)}
    with open(os.path.join(output_dir("benchmarks", "twin_couplings"), "c4_demo.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    what = argv[0] if argv else "c3"
    d = output_dir("benchmarks", "twin_couplings")
    if what == "c3":
        print(json.dumps(c3_run(), indent=1, default=float))
    elif what == "c4":
        print(json.dumps(c4_demo(), indent=1, default=float))
    elif what == "c2":
        frames = []
        for o in ("2016", "2011"):
            try:
                frames.append(c2_cloud_survival(o))
            except FileNotFoundError as e:
                print("skip", o, e)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(os.path.join(d, "c2_cloud_survival.csv"), index=False)
        s = c2_summary(df)
        s.to_csv(os.path.join(d, "c2_summary.csv"), index=False)
        print(s.to_string(index=False))
        bs = c2_bootstrap(df)
        bs.to_csv(os.path.join(d, "c2_bootstrap.csv"), index=False)
        print(bs.to_string(index=False))


if __name__ == "__main__":
    main()
