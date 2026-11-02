"""Evidence that radar-cross-section availability in the catalogue snapshot encodes the outcome.

    python -m cosmos.ai.lifetime.leak_check

For each origin and split: P(natural decay within 30 days | RCS missing) and | RCS present, the
share of objects without RCS, and the odds ratio of 30-day decay for a missing RCS, adjusted for
log perigee, log apogee, inclination, object class and decade of t0 (logistic regression, 95 % CI
from cluster-robust standard errors with launch families as clusters: fragments of one breakup
share their fate).  The
same adjusted odds ratio is given for design-data availability among components (class C), the
only class in which it is mixed.  Output: outputs/benchmarks/reentry_lifetime/leak_check.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ...common import output_dir
from . import run as LR


def adjusted_or(d: pd.DataFrame, flag: str) -> dict:
    import statsmodels.api as sm
    X = pd.DataFrame({"log_perigee": np.log10(d["perigee"].clip(lower=100)), "log_apogee": np.log10(d["apogee"].clip(lower=100)),
                      "inc": d["inc"] / 90.0, flag: d[flag].astype(float)})
    cats = [pd.get_dummies(d["t0"].dt.year // 10 * 10, prefix="dec", drop_first=True)]
    if d["type1"].nunique() > 1:
        cats.append(pd.get_dummies(d["type1"], prefix="cls", drop_first=True))
    X = pd.concat([X] + cats, axis=1).astype(float)
    groups = pd.factorize(d["launch_family"].astype(str))[0]
    m = sm.Logit(d["y_30d"].to_numpy(), sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    b, se = float(m.params[flag]), float(m.bse[flag])
    return {"odds_ratio": float(np.exp(b)), "ci95": [float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))], "n": int(len(d)),
            "clusters": int(groups.max() + 1), "se": "cluster-robust (launch family)"}


def run() -> dict:
    df = LR.load_cohort()
    out = {}
    for o in ("2016", "2011"):
        S = LR.split(df, o, "uncontrolled")
        for k in ("train", "tuneB", "test"):
            d = S[k][S[k]["y_30d"].notna()].copy()
            d["rcs_missing"] = d["rcs_m2"].isna()
            e = {"n": int(len(d)), "share_rcs_missing": float(d["rcs_missing"].mean()),
                 "p_decay30_given_missing": float(d.loc[d["rcs_missing"], "y_30d"].mean()) if d["rcs_missing"].any() else None,
                 "p_decay30_given_present": float(d.loc[~d["rcs_missing"], "y_30d"].mean()),
                 "n_missing": int(d["rcs_missing"].sum())}
            if k == "train":
                e["adjusted_or_rcs_missing"] = adjusted_or(d, "rcs_missing")
                c = d[d["type1"].eq("C")].copy()
                c["design_missing"] = ~np.isfinite(c["am_design"])
                e["components_adjusted_or_design_missing"] = adjusted_or(c, "design_missing")
            if k == "test":
                e["share_rcs_missing_t0_from_2018"] = float(d.loc[d["t0"] >= "2018-01-01", "rcs_missing"].mean()) \
                    if (d["t0"] >= "2018-01-01").any() else None
            out[f"o{o}.{k}"] = e
    p = os.path.join(output_dir("benchmarks", "reentry_lifetime"), "leak_check.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
