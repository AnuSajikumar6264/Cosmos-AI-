"""As-of-date test T1 of the twin's re-entry forecasts, for every object class (not blind: the
window had been analysed earlier in the project; see the disclosure below).

    python -m cosmos.ai.twin.prospective

As of 2026-08-16 (the CelesTrak SATCAT snapshot with that data date; GCAT design data, family
evidence and space weather restricted to information before it), the twin forecasts
P(re-entry within 30 and 72 days) for every object in orbit with perigee below 2000 km.  The
outcomes are the decay dates in the 2026-10-27 snapshot.  Objects whose decay before the as-of
date was reported late are dropped, as in C3.

Variants, all specified before this test:
- ``lifetime``: the validated lifetime pipeline, refitted as of 2026-08-16;
- ``physics_prior``: the physics expert with class priors;
- ``physics_history``: the physics expert with the area-to-mass posterior conditioned on the
  catalogue history (the twin's choice for uncontrolled objects);
- the twin's per-regime choice (``twin``).

Payload regimes are reported as in C3.  Confidence intervals resample launches.

Disclosure: the aggregate number of decays in this window was seen while diagnosing the domain
shift of the lifetime pipeline on current orbits; that motivated the history-conditioned variant.
Its design (prior samples, likelihood width, fallbacks) was fixed before it was evaluated here.
The only change after the first evaluation is the correction of the payload prior (payloads
without design values had fallen back to the fragment breakup law), which changes payload
probabilities only.  The window had also been used by the earlier landmark benchmark (commit
2525367, cosmos/ai/survival/landmark.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from ..evaluation import brier_score, expected_calibration_error

AS_OF = "2026-08-16"
FOLLOW = "2026-10-27"
H_T1 = {"30d": 30.0, "72d": 72.0}
VARIANTS = ("physics_history", "physics_prior", "lifetime")


def outcomes(T: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_csv(os.path.join(feature_dir("debris", "celestrak", "snapshots", FOLLOW), "satcat.csv"),
                    usecols=["NORAD_CAT_ID", "DECAY_DATE"])
    dec = pd.to_datetime(T["norad"].map(b.set_index("NORAD_CAT_ID")["DECAY_DATE"]), errors="coerce")
    tau = pd.Timestamp(AS_OF)
    keep = ~(dec <= tau)
    T = T[keep.to_numpy()].copy()
    dec = dec[keep]
    for k, h in H_T1.items():
        T[f"y_{k}"] = (dec.notna() & (dec <= tau + pd.Timedelta(days=h))).astype(int).to_numpy()
    T["launch"] = T["Launch_Tag"].astype(str).str[:8]
    return T


def scores(y, p, p_rank=None) -> dict:
    """Calibration on ``p``; rank metrics on ``p_rank`` (the raw score where ``p`` is isotonic-calibrated,
    since the isotonic map ties scores).  The physics variants are shares of 32 Monte-Carlo samples, so
    their scores are tied too; sklearn resolves ties in average_precision_score pessimistically."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    p_rank = p if p_rank is None else p_rank
    e = {"n": int(len(y)), "events": int(y.sum()), "expected": float(p.sum()), "mean_p": float(p.mean()),
         "brier": brier_score(y, p), "ece": expected_calibration_error(y, p)}
    if 0 < y.sum() < len(y) and np.ptp(p_rank) > 0:
        e.update({"auc_roc": float(roc_auc_score(y, p_rank)), "auc_pr": float(average_precision_score(y, p_rank))})
    return e


def variant_scores(T, k, v, sel):
    y = T[f"y_{k}"].to_numpy()[sel]
    raw = f"p_{v}_raw_{k}"
    return scores(y, T[f"p_{v}_{k}"].to_numpy()[sel], T[raw].to_numpy()[sel] if raw in T else None)


def run(learner=None, params=None, seeds=(0, 1, 2, 3, 4)) -> dict:
    from ..lifetime import run as LR
    from . import engine as E
    if learner is None:
        learner, params = E.study_learner()
    T, M, info = E.forecast(AS_OF, learner, params, seeds=seeds, horizons=H_T1)
    T = outcomes(T)
    unc = T["type1"].isin(["D", "R", "C"]).to_numpy()
    out = {"as_of": AS_OF, "follow_up": FOLLOW, "learner": learner, "seeds": len(seeds), "history_kinds": info["history_kinds"],
           "objects": int(len(T)), "uncontrolled": int(unc.sum()), "by_horizon": {}}
    for k in H_T1:
        y = T[f"y_{k}"].to_numpy()
        r = {"uncontrolled": {v: variant_scores(T, k, v, unc) for v in VARIANTS}}
        for cls in ("D", "R", "C"):
            s = T["type1"].eq(cls).to_numpy()
            r[f"class_{cls}"] = {v: variant_scores(T, k, v, s) for v in VARIANTS}
        for reg in ("natural_decay", "operator_controlled", "unknown"):
            s = (T["regime"].eq(reg) & T["type1"].eq("P")).to_numpy()
            if s.any():
                r[f"payload_{reg}"] = {"twin": scores(y[s], T[f"p_{k}"].to_numpy()[s]),
                                       "physics_prior": scores(y[s], T[f"p_physics_prior_{k}"].to_numpy()[s])}
        r["all_twin"] = scores(y, T[f"p_{k}"].to_numpy())
        # launch-cluster bootstrap of Brier differences on uncontrolled objects (< 0: first is better)
        launch = T["launch"].to_numpy()[unc]
        sq = {v: (T[f"p_{v}_{k}"].to_numpy()[unc] - y[unc]) ** 2 for v in VARIANTS}
        r["brier_difference_ci95_launch_bootstrap"] = {}
        for a, b in (("physics_history", "physics_prior"), ("physics_history", "lifetime"), ("physics_prior", "lifetime")):
            dv = sq[a] - sq[b]
            lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(dv[idx].mean()), launch)
            r["brier_difference_ci95_launch_bootstrap"][f"{a}_minus_{b}"] = {"mean": float(dv.mean()), "ci95": [lo, hi]}
        # expected-over-observed count ratio with a launch-cluster CI
        r["expected_over_observed_ci95"] = {}
        for v in VARIANTS:
            pv, yv = T[f"p_{v}_{k}"].to_numpy()[unc], y[unc]
            lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(pv[idx].sum() / max(yv[idx].sum(), 1)), launch)
            r["expected_over_observed_ci95"][v] = {"ratio": float(pv.sum() / max(yv.sum(), 1)), "ci95": [lo, hi]}
        out["by_horizon"][k] = r
    d = output_dir("benchmarks", "twin_prospective_t1")
    T.to_csv(os.path.join(d, "t1_objects.csv.gz"), index=False)
    with open(os.path.join(d, "t1_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
