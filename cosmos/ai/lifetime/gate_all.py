"""Sensitivity analysis (added after the origin-2016 results were seen; not used for any
selection): the reliability gate applied to *every* learned expert, not only the one selected on
tune-A.

    python -m cosmos.ai.lifetime.gate_all --origin 2016

For each learner, out-of-fold predictions on the training period (5 folds grouped by launch
family, the learner's tuned settings from the main run) are used to fit its gate; the fused
model is calibrated on tune-B as in the main run and scored on the test cohort.  Output: the
learner alone versus the learner gated with the physics expert, with a paired cluster bootstrap
over launch families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from ...common import output_dir
from . import fusion as FU
from . import models as MO
from . import physics as PH
from . import run as LR


def run(origin="2016", population="uncontrolled", n_samples=32, learners=LR.LEARNER_SET):
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    cache = os.path.join(out, "learner_cache")
    S = LR.split(LR.load_cohort(), origin, population)
    prior = PH.class_prior_from(S["train"])
    phys = {}
    for k, d in S.items():
        r = PH.predict(d, prior, n_samples=n_samples, seed=0)
        phys[k] = {"feat": MO.physics_features(r["samples_days"]), "q": MO._monotone(MO.physics_quantiles(r["samples_days"])), "src": r["am_source"]}
    X, shapes = {}, None
    for k in ("train", "tuneA", "tuneB", "test"):
        X[k], shapes = MO.design_matrix(S[k], phys[k]["feat"], LR.FULL, shapes)
    cols = list(X["train"].columns)
    X = {k: MO.align_columns(v, cols) for k, v in X.items()}
    lab = {k: (S[k].reset_index(drop=True)["known5"].to_numpy(), S[k].reset_index(drop=True)["z"].to_numpy()) for k in S}
    (mtr, ztr), (mA, zA), (mB, zB), (mT, zT) = lab["train"], lab["tuneA"], lab["tuneB"], lab["test"]
    fam = S["train"].reset_index(drop=True)["launch_family"].astype(str)
    fold = fam.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 5).to_numpy()
    R = {k: FU.reliability_features(S[k], phys[k]["feat"], phys[k]["src"]) for k in ("train", "tuneB", "test")}
    G = {k: FU.mondrian_groups(S[k]) for k in ("tuneB", "test")}
    famT = S["test"].reset_index(drop=True)["launch_family"].astype(str).to_numpy()[mT]
    rows = []
    for name in learners:
        z_ = np.load(os.path.join(cache, f"{name}.npz"), allow_pickle=True)
        fit = json.loads(str(z_["fit_json"]))
        seeds = sorted({int(k.split("_")[0][1:]) for k in z_.files if k.startswith("s")})
        q = {k: np.sort(np.mean([z_[f"s{s}_{k}"] for s in seeds], 0), 1) for k in ("tuneB", "test")}
        oof = np.full((len(fam), len(MO.QUANTILES)), np.nan)
        for k in range(5):
            trn, hold = (fold != k) & mtr, fold == k
            m = MO.LEARNERS[name](seed=0).fit(X["train"][trn], ztr[trn], X["tuneA"][mA], zA[mA], fit["params"])
            oof[hold] = m.predict(X["train"][hold])
        gate = FU.Gate().fit(R["train"][mtr], phys["train"]["q"][mtr], oof[mtr], ztr[mtr])
        fused = {k: MO._monotone(gate.fuse(R[k], phys[k]["q"], q[k])) for k in ("tuneB", "test")}
        res = {}
        for tag, qq in (("alone", q), ("gated", fused)):
            cq = FU.MondrianCQR(mondrian=False).fit(qq["tuneB"][mB], zB[mB], G["tuneB"][mB].reset_index(drop=True))
            lo90, hi90 = cq.interval(qq["test"], G["test"], 0.9)
            res[tag] = LR.regression_metrics(zT[mT], qq["test"][mT], None, None, lo90[mT], hi90[mT])
            res[tag]["loss_vec"] = np.mean(np.maximum(MO.QUANTILES * (zT[mT, None] - qq["test"][mT]),
                                                      (MO.QUANTILES - 1) * (zT[mT, None] - qq["test"][mT])), 1)
        diff = res["alone"]["loss_vec"] - res["gated"]["loss_vec"]         # > 0: gating helps
        lo, hi, _ = LR.cluster_bootstrap(lambda idx: float(diff[idx].mean()), famT)
        rows.append({"learner": name, "pinball_alone": res["alone"]["pinball"], "pinball_gated": res["gated"]["pinball"],
                     "gain": float(diff.mean()), "gain_ci95_lo": lo, "gain_ci95_hi": hi,
                     "mae_log10_alone": res["alone"]["mae_log10"], "mae_log10_gated": res["gated"]["mae_log10"],
                     "cov90_alone_marginal": res["alone"]["cov90"], "cov90_gated_marginal": res["gated"]["cov90"],
                     "gate_weight_test_mean": float(gate.weight(R["test"]).mean()),
                     "oof_pinball": MO.pinball(ztr[mtr], oof[mtr])})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "gate_all_learners.csv"), index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2016")
    a = ap.parse_args(argv)
    print(run(a.origin).to_string(index=False))


if __name__ == "__main__":
    main()
