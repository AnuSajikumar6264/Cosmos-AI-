"""Diagnostics of the re-entry study that use information unavailable at prediction time.

    python -m cosmos.ai.lifetime.diagnostics --origin 2016

``oracle_solar``: the physics expert is re-run on the test objects with the *realised* solar
activity along each trajectory (CelesTrak 81-day centred F10.7, observed rows) instead of the
value held from t0.  The gap between the two shows how much of the physics error is due to the
unknown future solar activity.  Never used for model selection or reported as a forecast.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from . import models as MO
from . import physics as PH
from . import run as LR
from .catalogue import CELESTRAK_DIR


def realised_f107():
    sw = pd.read_csv(os.path.join(feature_dir(*CELESTRAK_DIR), "SW-All.csv"), parse_dates=["DATE"])
    sw = sw[sw["F10.7_DATA_TYPE"].isin(["OBS", "INT"])].set_index("DATE").sort_index()
    f = sw["F10.7_OBS_CENTER81"].astype(float).ffill()
    day0 = f.index[0]
    return day0, f.to_numpy()


def oracle_solar(origin="2016", population="uncontrolled", n_samples=32) -> dict:
    S = LR.split(LR.load_cohort(), origin, population)
    t = S["test"].reset_index(drop=True)
    prior = PH.class_prior_from(S["train"])
    day0, f = realised_f107()
    t0d = ((t["t0"] - day0).dt.total_seconds() / 86400).to_numpy()
    t0_el = np.repeat(t0d, n_samples)

    def path(active, elapsed):
        k = np.clip((t0_el[active] + elapsed).astype(int), 0, len(f) - 1)
        return f[k]
    res = {}
    for name, fp in (("persistence", None), ("oracle_realised_f107", path)):
        r = PH.predict(t, prior, n_samples=n_samples, seed=0, oracle_f107=fp)
        q = MO.physics_quantiles(r["samples_days"])
        m = t["known5"].to_numpy()
        res[name] = LR.regression_metrics(t["z"].to_numpy()[m], q[m])
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    with open(os.path.join(out, "diagnostic_oracle_solar.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2016")
    a = ap.parse_args(argv)
    print(json.dumps(oracle_solar(a.origin), indent=1, default=float))



def interval_rules(origin="2016", population="uncontrolled") -> dict:
    """Coverage and width of the operational +/-20 % and +/-30 % residual-lifetime windows
    (Pardini & Anselmo 2018: about 91 % and 95 % coverage in IADC campaigns near re-entry)
    around the physics and fused medians, next to the conformal 80 % and 90 % intervals."""
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    P = np.load(os.path.join(out, "test_predictions.npz"))
    z, m = P["z"], P["known5"]
    res = {}
    for name in ("physics_only", "cosmos_gated"):
        q = P[f"{name}__q"]
        med = q[:, 3]
        for frac in (0.2, 0.3):
            lo = np.log10(np.clip(10 ** med * (1 - frac), 0.5, None))
            hi = np.log10(10 ** med * (1 + frac))
            cov = (z >= lo) & (z <= hi)
            res[f"{name}_pm{int(frac * 100)}"] = {"coverage": float(cov[m].mean()), "width_log10": float((hi - lo)[m].mean()),
                                                   "coverage_decayed": float(cov[m & (z < MO.Z_CAP - 1e-9)].mean())}
        for lev in (80, 90):
            lo, hi = P[f"{name}__lo{lev}"], P[f"{name}__hi{lev}"]
            cov = (z >= lo) & (z <= hi)
            res[f"{name}_conformal{lev}"] = {"coverage": float(cov[m].mean()), "width_log10": float((hi - lo)[m].mean()),
                                             "coverage_decayed": float(cov[m & (z < MO.Z_CAP - 1e-9)].mean())}
    with open(os.path.join(out, "interval_rules.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    main()


def class_breakdown(origin="2016", population="uncontrolled") -> pd.DataFrame:
    """Post-hoc: test pinball loss of every candidate by object class (fragments, rocket stages,
    components), from the saved test predictions.  Added after the test results were seen; it
    explains them and selects nothing."""
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    P = np.load(os.path.join(out, "test_predictions.npz"))
    t = LR.split(LR.load_cohort(), origin, population)["test"].reset_index(drop=True)
    z, m = P["z"], P["known5"]
    rows = []
    for name in sorted({k.split("__")[0] for k in P.files if k.endswith("__q")}):
        q = P[f"{name}__q"]
        for c in ("D", "R", "C"):
            s = m & t["type1"].eq(c).to_numpy()
            rows.append({"model": name, "type1": c, "n": int(s.sum()), "pinball": MO.pinball(z[s], q[s])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "class_breakdown_test.csv"), index=False)
    return df


def ablation_effects(origin="2016", population="uncontrolled") -> pd.DataFrame:
    """Paired cluster-bootstrap CI (launch families) of the test pinball difference between each
    feature-group ablation of the selected learner and the full learner (> 0: removing the group
    hurts; for plus_rcs_2026, > 0 means adding the leaky covariate hurts)."""
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    P = np.load(os.path.join(out, "test_predictions.npz"))
    best = json.load(open(os.path.join(out, "summary.json")))["best_learner_on_tuneA"]
    t = LR.split(LR.load_cohort(), origin, population)["test"].reset_index(drop=True)
    z, m = P["z"], P["known5"]
    fam = t["launch_family"].astype(str).to_numpy()[m]

    def loss(q):
        d = z[m, None] - q[m]
        return np.mean(np.maximum(MO.QUANTILES * d, (MO.QUANTILES - 1) * d), 1)
    full = loss(P[f"{best}__q"])
    rows = []
    for k in sorted(P.files):
        if k.startswith(f"{best}_") and k.endswith("__q") and k != f"{best}__q" and "marginal" not in k:
            diff = loss(P[k]) - full
            lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(diff[idx].mean()), fam)
            rows.append({"learner": best, "ablation": k[len(best) + 1:-3], "pinball_ablation_minus_full": float(diff.mean()),
                         "ci95_lo": lo, "ci95_hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "ablation_effects_test.csv"), index=False)
    return df


PAIRS = (("physics_only", "tabicl"), ("physics_only", "kumo"), ("physics_only", "tabm"), ("physics_only", "cosmos_gated"),
         ("tabicl", "kumo"), ("tabicl", "xgboost"), ("kumo", "xgboost"), ("tabicl", "cosmos_gated"),
         ("kumo", "cosmos_gated"), ("tabicl", "tabm"), ("kumo", "tabm"))


def pairwise_effects(origin="2016", population="uncontrolled", pairs=PAIRS) -> pd.DataFrame:
    """Post-hoc head-to-head comparisons (added after the test results were seen): paired cluster
    bootstrap over launch families of the test pinball difference a - b (< 0: a is better)."""
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    P = np.load(os.path.join(out, "test_predictions.npz"))
    t = LR.split(LR.load_cohort(), origin, population)["test"].reset_index(drop=True)
    z, m = P["z"], P["known5"]
    fam = t["launch_family"].astype(str).to_numpy()[m]

    def loss(q):
        d = z[m, None] - q[m]
        return np.mean(np.maximum(MO.QUANTILES * d, (MO.QUANTILES - 1) * d), 1)
    rows = []
    for a, b in pairs:
        if f"{a}__q" not in P.files or f"{b}__q" not in P.files:
            continue
        diff = loss(P[f"{a}__q"]) - loss(P[f"{b}__q"])
        lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(diff[idx].mean()), fam)
        rows.append({"a": a, "b": b, "pinball_a_minus_b": float(diff.mean()), "ci95_lo": lo, "ci95_hi": hi})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "pairwise_test.csv"), index=False)
    return df


def km_descriptive(origin="2016", population="uncontrolled") -> pd.DataFrame:
    """Kaplan-Meier descriptive survival of the test cohort, overall and per object class.

    Uses the censoring-aware estimator of ``cosmos.debris.catalogue`` on the observed time to
    natural decay, with objects still in orbit (or ended without natural decay) as right-censored.
    Descriptive only: it uses no model output, and it is the reference against which the learned
    quantile models are read.
    """
    from ...debris.catalogue import kaplan_meier
    t = LR.split(LR.load_cohort(), origin, population)["test"].reset_index(drop=True)
    days, ev = t["time_days"].astype(float).to_numpy(), t["event"].astype(bool).to_numpy()
    rows, curves = [], {}
    for lab, sel in [("all", np.ones(len(t), bool))] + [(c, t["type1"].eq(c).to_numpy()) for c in ("D", "R", "C")]:
        if sel.sum() < 10:
            continue
        km = kaplan_meier(days[sel], ev[sel])
        curves[lab] = (km["times"], km["survival"])
        rows.append({"group": lab, "n": km["n"], "events": km["n_observed"],
                     "censored": km["n"] - km["n_observed"],
                     "censored_fraction": 1 - km["n_observed"] / max(km["n"], 1),
                     "median_days": km["median"],
                     "surv_30d": float(np.interp(30.0, km["times"], km["survival"], left=1.0)),
                     "surv_1yr": float(np.interp(365.25, km["times"], km["survival"], left=1.0)),
                     "surv_5yr": float(np.interp(1826.25, km["times"], km["survival"], left=1.0))})
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "km_descriptive_test.csv"), index=False)
    np.savez_compressed(os.path.join(out, "km_curves_test.npz"),
                        **{f"{k}__{w}": v for k, (ts, sv) in curves.items() for w, v in (("t", ts), ("s", sv))})
    return df


def temporal_coverage(origin="2016", population="uncontrolled", models=("physics_only", "cosmos_gated", "BEST")) -> pd.DataFrame:
    """Empirical 90 % conformal coverage of the test cohort by calendar year of t0.

    Conditional coverage over time: conformal prediction guarantees marginal coverage under
    exchangeability, so a drift in coverage across the test period is evidence against it.
    """
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    P = np.load(os.path.join(out, "test_predictions.npz"))
    s = json.load(open(os.path.join(out, "summary.json")))
    t = LR.split(LR.load_cohort(), origin, population)["test"].reset_index(drop=True)
    z, m = P["z"], P["known5"].astype(bool)
    year = pd.to_datetime(t["t0"]).dt.year.to_numpy()
    rows = []
    for name in models:
        mod = s["best_learner_on_tuneA"] if name == "BEST" else name
        lo, hi = f"{mod}__lo90", f"{mod}__hi90"
        if lo not in P.files:
            continue
        inside = (z >= P[lo]) & (z <= P[hi])
        for y in sorted(set(year[m])):
            k = m & (year == y)
            if k.sum() >= 25:
                rows.append({"model": mod, "year": int(y), "n": int(k.sum()),
                             "cov90": float(inside[k].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "temporal_coverage_test.csv"), index=False)
    return df
