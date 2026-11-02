"""Censoring-aware learned expert: XGBoost accelerated-failure-time (AFT) survival model.

    python -m cosmos.ai.lifetime.survival --origin 2016

The capped-regression experts of :mod:`run` are trained on objects whose 5-year outcome is
known.  This expert instead uses every training object as a survival unit: a natural decay is
an exact event time, and an object censored by a controlled ending or by the administrative
cut-off (the origin date) contributes the interval ``[C, inf)``.  The model is fitted on
``ln T`` with a normal or logistic error distribution; its quantiles of ``T`` are capped at 5
years and converted to the common target ``z = log10 min(T, 5 yr)``.

Same splits, same features (``run.FULL``), same tuning budget (``models.N_TRIALS`` random
configurations scored by the pinball loss on tune-A) and the same test metrics as ``run.py``;
results are written next to the main results of the origin.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import logistic, norm

from ...common import output_dir
from . import fusion as FU
from . import models as MO
from . import physics as PH
from . import run as LR


class XGBAFT(MO._Bounded):
    name = "xgboost_aft"

    def __init__(self, seed=0, n_jobs=32):
        self.seed, self.n_jobs = seed, n_jobs

    def space(self, rng):
        return {"max_depth": int(rng.integers(3, 9)), "learning_rate": float(10 ** rng.uniform(-1.7, -0.7)),
                "subsample": float(rng.uniform(0.6, 1.0)), "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_weight": float(10 ** rng.uniform(0, 1.5)), "reg_lambda": float(10 ** rng.uniform(-1, 1)),
                "dist": str(rng.choice(["normal", "logistic"])), "scale": float(10 ** rng.uniform(-0.5, 0.4))}

    def fit(self, X, t_days, event, Xv, zv, params):
        import xgboost as xgb
        self.cols = list(X.columns)
        self.params = params
        # XGBoost's AFT objective takes times on the original scale (it applies ln internally)
        lo = np.clip(np.asarray(t_days, float), 0.5, None)
        hi = np.where(np.asarray(event) == 1, lo, np.inf)
        d = xgb.DMatrix(X, label_lower_bound=lo, label_upper_bound=hi)
        p = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "tree_method": "hist",
             "aft_loss_distribution": params["dist"], "aft_loss_distribution_scale": params["scale"],
             "max_depth": params["max_depth"], "eta": params["learning_rate"], "subsample": params["subsample"],
             "colsample_bytree": params["colsample_bytree"], "min_child_weight": params["min_child_weight"],
             "lambda": params["reg_lambda"], "seed": self.seed, "nthread": self.n_jobs}
        # early stopping on the pinball loss of the capped target on the tuning split
        best, best_it, self.bst = np.inf, 50, None
        bst = None
        for it in range(0, 2000, 50):
            bst = xgb.train(p, d, num_boost_round=50, xgb_model=bst)
            self.bst, self.n_iter = bst, it + 50
            s = MO.pinball(np.asarray(zv), self.predict(Xv))
            if s < best - 1e-5:
                best, best_it = s, it + 50
            elif it + 50 - best_it >= 200:
                break
        self.bst = xgb.train(p, d, num_boost_round=best_it)
        self.n_iter = best_it
        return self

    def predict(self, X):
        import xgboost as xgb
        mu = self.bst.predict(xgb.DMatrix(MO.align_columns(X, self.cols)), output_margin=True)
        dist = norm if self.params["dist"] == "normal" else logistic
        lnq = mu[:, None] + self.params["scale"] * dist.ppf(MO.QUANTILES)[None, :]
        z = np.log10(np.minimum(np.exp(lnq), PH.CAP_DAYS))
        return MO._monotone(z, self.lo, self.hi)


def run(origin="2016", population="uncontrolled", n_samples=32, n_trials=MO.N_TRIALS, seeds=LR.SEEDS):
    t_start = time.time()
    S = LR.split(LR.load_cohort(), origin, population)
    prior = PH.class_prior_from(S["train"])
    feat = {}
    for k, d in S.items():
        r = PH.predict(d, prior, n_samples=n_samples, seed=0)
        feat[k] = MO.physics_features(r["samples_days"])
    X, shapes = {}, None
    for k in ("train", "tuneA", "tuneB", "test"):
        X[k], shapes = MO.design_matrix(S[k], feat[k], LR.FULL, shapes)
    cols = list(X["train"].columns)
    X = {k: MO.align_columns(v, cols) for k, v in X.items()}
    tr = S["train"].reset_index(drop=True)
    A = S["tuneA"].reset_index(drop=True)
    mA = A["known5"].to_numpy()
    rng = np.random.default_rng(1000)
    trials, best = [], None
    for t in range(n_trials):
        m = XGBAFT(seed=0)
        p = m.space(rng)
        m.fit(X["train"], tr["time_days"].to_numpy(), tr["event"].to_numpy(), X["tuneA"][mA], A["z"].to_numpy()[mA], p)
        s = MO.pinball(A["z"].to_numpy()[mA], m.predict(X["tuneA"][mA]))
        trials.append({"params": p, "val_pinball": s, "n_iter": m.n_iter})
        if best is None or s < best[0]:
            best = (s, p)
    preds = []
    for sd in seeds:
        m = XGBAFT(seed=sd).fit(X["train"], tr["time_days"].to_numpy(), tr["event"].to_numpy(), X["tuneA"][mA],
                                A["z"].to_numpy()[mA], best[1])
        preds.append({k: m.predict(X[k]) for k in ("tuneB", "test")})
    q = {k: np.sort(np.mean([p_[k] for p_ in preds], 0), 1) for k in ("tuneB", "test")}
    B = S["tuneB"].reset_index(drop=True)
    T = S["test"].reset_index(drop=True)
    mB, mT = B["known5"].to_numpy(), T["known5"].to_numpy()
    G = {"tuneB": FU.mondrian_groups(B), "test": FU.mondrian_groups(T)}
    cq = FU.MondrianCQR().fit(q["tuneB"][mB], B["z"].to_numpy()[mB], G["tuneB"][mB].reset_index(drop=True))
    lo80, hi80 = cq.interval(q["test"], G["test"], 0.8)
    lo90, hi90 = cq.interval(q["test"], G["test"], 0.9)
    zT = T["z"].to_numpy()
    from ..evaluation import harrell_c
    res = LR.regression_metrics(zT[mT], q["test"][mT], lo80[mT], hi80[mT], lo90[mT], hi90[mT])
    res["c_index"] = harrell_c(T["time_days"].to_numpy(), T["event"].to_numpy(), -q["test"][:, 3])
    res["train_rows_used"] = int(len(tr))
    res["train_rows_censored"] = int((tr["event"] == 0).sum())
    seed_rows = [{"seed": s, "pinball": MO.pinball(zT[mT], p_["test"][mT])} for s, p_ in zip(seeds, preds)]
    out = output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    with open(os.path.join(out, "survival_aft.json"), "w") as f:
        json.dump({"test": res, "seeds": seed_rows, "best_params": best[1], "trials": trials,
                   "seconds": time.time() - t_start}, f, indent=1, default=float)
    np.savez_compressed(os.path.join(out, "survival_aft_test_predictions.npz"), q=q["test"], lo90=lo90, hi90=hi90,
                        lo80=lo80, hi80=hi80)
    print(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2016")
    ap.add_argument("--population", default="uncontrolled")
    a = ap.parse_args(argv)
    run(a.origin, a.population)


if __name__ == "__main__":
    main()
