"""Computational cost of the re-entry module, measured with fitting and prediction timed apart.

    PYTHONPATH=. COSMOS_DEVICE=xpu:0 python -m cosmos.ai.lifetime.cost

The timings inside ``run.py`` mix the two: its ``fit_seconds_per_seed`` wraps the seed loop, which
also predicts tune-A, tune-B and test and reuses the already-fitted seed-0 model, and its
``predict_ms_per_1000_objects`` divides one test-set call by its row count, which for an in-context
model is dominated by the fixed cost of encoding the training context.  This module therefore
measures, for the learner settings selected at an origin:

* ``fit_seconds``: one ``fit`` call on the training cohort, nothing else inside the timer;
* ``predict_seconds[n]``: one ``predict`` call on ``n`` rows, for several ``n``, so that the fixed
  and the marginal cost of a prediction are visible.  Rows are taken from the test cohort, repeated
  when ``n`` exceeds it, so every learner sees the same matrix.

The physics expert is timed the same way (its Monte-Carlo run is its whole cost; it has no fit).
Results go to ``outputs/benchmarks/reentry_lifetime/cost_timing.json``.  They are a single
measurement on a shared machine, so they are indicative, not a benchmark of the implementations.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from ..experiment import output_dir
from . import models as MO
from . import physics as PH
from . import run as LR

SIZES = (1000, 3000, 30000)
LEARNER_SET = LR.LEARNER_SET


def _rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """``n`` rows, repeating the frame when it is shorter (prediction cost does not depend on the
    values, so repeated rows measure the same work as a catalogue of that size)."""
    k = int(np.ceil(n / len(df)))
    return pd.concat([df] * k, ignore_index=True).iloc[:n] if k > 1 else df.iloc[:n].reset_index(drop=True)


def measure(origin: str = "2016", population: str = "uncontrolled", learners=LEARNER_SET, sizes=SIZES,
            n_samples: int = 32, out=None) -> dict:
    out = out or output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    S = LR.split(LR.load_cohort(), origin, population)
    res = {"origin": origin, "population": population, "sizes": list(sizes), "n_samples": n_samples,
           "host": {"node": platform.node(), "cpu_count": os.cpu_count(),
                    "torch_threads": os.environ.get("COSMOS_TORCH_THREADS"), "device_env": os.environ.get("COSMOS_DEVICE")},
           "train_rows": int(S["train"]["known5"].sum()), "models": {}}

    # physics expert: the Monte-Carlo propagation is its whole cost
    prior = PH.class_prior_from(S["train"])
    phys = {}
    for n in sizes:
        d = _rows(S["test"], n)
        t0 = time.time()
        PH.predict(d, prior, n_samples=n_samples, seed=0)
        phys[str(n)] = time.time() - t0
        print(f"  physics_only predict {n}: {phys[str(n)]:.1f} s", flush=True)
    res["models"]["physics_only"] = {"fit_seconds": None, "predict_seconds": phys, "device": "cpu",
                                     "note": f"Monte-Carlo propagation, {n_samples} samples per object; no fitting step"}

    # learned experts and baselines, with the settings selected at this origin
    fits = json.load(open(os.path.join(out, "summary.json")))["fits"]
    feat = {}
    for k in ("train", "tuneA", "test"):
        r = PH.predict(S[k], prior, n_samples=n_samples, seed=0)
        feat[k] = MO.physics_features(r["samples_days"])
    X, shapes = {}, None
    for k in ("train", "tuneA", "test"):
        X[k], shapes = MO.design_matrix(S[k], feat[k], LR.FULL, shapes)
    cols = list(X["train"].columns)
    X = {k: MO.align_columns(v, cols) for k, v in X.items()}
    mtr, ztr = S["train"]["known5"].to_numpy(), S["train"]["z"].to_numpy()
    mA, zA = S["tuneA"]["known5"].to_numpy(), S["tuneA"]["z"].to_numpy()
    for name in learners:
        if name not in fits:
            continue
        params = fits[name]["params"]
        model = MO.LEARNERS[name](seed=0)
        t0 = time.time()
        model.fit(X["train"][mtr], ztr[mtr], X["tuneA"][mA], zA[mA], params)
        fit_s = time.time() - t0
        pr = {}
        for n in sizes:
            Xn = _rows(X["test"], n)
            t0 = time.time()
            model.predict(Xn)
            pr[str(n)] = time.time() - t0
            print(f"  {name} predict {n}: {pr[str(n)]:.1f} s", flush=True)
        res["models"][name] = {"fit_seconds": fit_s, "predict_seconds": pr, "params": params,
                               "device": str(getattr(model, "device", "cpu")),
                               "context_rows": int(getattr(model, "context_rows", 0)) or None}
        print(f"  {name}: fit {fit_s:.1f} s", flush=True)
        if hasattr(model, "_release"):
            model._release()
    p = os.path.join(out, "cost_timing.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    print("wrote", p, flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2016")
    ap.add_argument("--learners", default=",".join(LEARNER_SET))
    a = ap.parse_args(argv)
    measure(a.origin, learners=tuple(a.learners.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
