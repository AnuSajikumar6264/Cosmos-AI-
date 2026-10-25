"""Gradient-boosted-tree baseline for the supervised event model (same features, same folds).

    python -m cosmos.ai.telemetry.event_gbt --mission 1 --seed 0

Each (step, channel) is one row.  Columns: the seven causal features of :mod:`event_model`,
two trailing maxima of ``log_s`` (16 and 256 steps, giving the trees some temporal context),
and the channel index (categorical).  The model is a 3-class XGBoost (nominal / anomaly /
rare event) with the same class weights as the network.  Training rows are subsampled, half
of them labelled.  It follows the same out-of-fold protocol and writes the same files, so
:mod:`oof_benchmark` treats both models alike.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ...common import output_dir
from ..experiment import Run
from . import oof
from .event_model import (EventConfig, LABEL_IGNORE, _allowed, mission_features)
from .forecasting import MissionData

N_ROWS = 2_000_000
PARAMS = {"objective": "multi:softprob", "num_class": 3, "max_depth": 8, "eta": 0.1,
          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
          "tree_method": "hist", "max_cat_to_onehot": 1, "nthread": 64}
ROUNDS = 400


def _trailing_max(x: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    # causal: the window ends at t (origin shifts the centred filter backwards)
    return maximum_filter1d(x, size=w, axis=0, origin=(w - 1) // 2, mode="nearest")


def tabular(F: np.ndarray) -> np.ndarray:
    """``(T, C, 10)`` float32: features + trailing maxima of log_s + channel index."""
    T, C, _ = F.shape
    out = np.empty((T, C, 10), np.float32)
    out[..., :7] = F
    out[..., 7] = _trailing_max(F[..., 1], 16)
    out[..., 8] = _trailing_max(F[..., 1], 256)
    out[..., 9] = np.arange(C, dtype=np.float32)[None, :]
    return out


def _dmatrix(X, y=None, w=None):
    import pandas as pd
    import xgboost as xgb
    cols = [f"f{i}" for i in range(X.shape[1] - 1)]
    df = pd.DataFrame(X[:, :-1], columns=cols)
    df["channel"] = pd.Categorical(X[:, -1].astype(np.int32))
    return xgb.DMatrix(df, label=y, weight=w, enable_categorical=True, nthread=64)


def train_one(Xt: np.ndarray, L: np.ndarray, allowed: np.ndarray, cfg: EventConfig, seed: int):
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    T, C, Fd = Xt.shape
    flat_ok = (allowed[:, None] & (L != LABEL_IGNORE)).ravel()
    lab = (L > 0).ravel() & flat_ok
    idx_l = np.flatnonzero(lab)
    idx_n = np.flatnonzero(flat_ok & ~lab)
    n_l = min(idx_l.size, N_ROWS // 2)
    pick = np.r_[rng.choice(idx_l, n_l, replace=False),
                 rng.choice(idx_n, N_ROWS - n_l, replace=False)]
    X = Xt.reshape(-1, Fd)[pick]
    y = L.ravel()[pick].astype(np.int32)
    w = np.asarray(cfg.class_weights, np.float32)[y]
    t0 = time.time()
    bst = xgb.train({**PARAMS, "seed": seed}, _dmatrix(X, y, w), ROUNDS)
    return bst, {"seconds": time.time() - t0, "n_rows": int(pick.size),
                 "labelled_rows": int(n_l)}


def predict(bst, Xt: np.ndarray, rows: tuple, block: int = 4_000_000):
    a, b = rows
    T, C, Fd = Xt.shape
    flat = Xt[a:b].reshape(-1, Fd)
    P = np.empty((flat.shape[0], 3), np.float32)
    for s in range(0, flat.shape[0], block):
        P[s:s + block] = bst.predict(_dmatrix(flat[s:s + block]))
    P = np.clip(P, 1e-7, 1.0)
    ev = np.log(P[:, 1] + P[:, 2]) - np.log(P[:, 0])
    an = np.log(P[:, 1]) - np.log1p(-np.minimum(P[:, 1], 1 - 1e-6))
    f16 = lambda v: np.clip(v, -60, 60).astype(np.float16).reshape(b - a, C)
    return f16(ev), f16(an)


def run(mission: int, channels: str = "target", seed: int = 0) -> str:
    cfg = EventConfig(mission=mission, channels=channels, seed=seed, device="cpu")
    D = MissionData(mission, channels)
    S_tr, S_te, meta = oof.load(mission, channels, "ar8")
    folds = [tuple(f) for f in meta["folds"]]
    purge = int(cfg.purge_days * 86400 / D.train.rule_s)
    F_tr, F_te = mission_features(D, S_tr, S_te)
    X_tr, X_te = tabular(F_tr), tabular(F_te)
    del F_tr, F_te
    n, C = X_tr.shape[:2]
    tag = f"m{mission}_{channels}_gbt_ar8_s{seed}"
    d = output_dir("benchmarks", "telemetry_esa_adb", "event_model", tag)
    ev_tr = np.zeros((n, C), np.float16)
    an_tr = np.zeros_like(ev_tr)
    info = {"folds": []}
    with Run(f"esa_event_{tag}", {"model": "xgboost", "params": PARAMS, "rounds": ROUNDS,
                                  "n_rows": N_ROWS, "mission": mission, "channels": channels,
                                  "class_weights": cfg.class_weights}, seed, "cpu",
             group="esa_event_model") as rn:
        for b in range(len(folds)):
            bst, inf = train_one(X_tr, D.Ltr, _allowed(n, folds, b, purge), cfg, seed * 100 + b)
            ev_tr[folds[b][0]:folds[b][1]], an_tr[folds[b][0]:folds[b][1]] = \
                predict(bst, X_tr, folds[b])
            info["folds"].append(inf)
            print(f"  {tag} fold {b}: trained in {inf['seconds']:.0f} s", flush=True)
        bst, inf = train_one(X_tr, D.Ltr, _allowed(n, folds, None, purge), cfg, seed * 100 + 99)
        bst.save_model(os.path.join(rn.dir, "checkpoints", "final.json"))
        ev_te, an_te = predict(bst, X_te, (0, X_te.shape[0]))
        info["final"] = inf
        for name, arr in (("event_train_oof", ev_tr), ("anomaly_train_oof", an_tr),
                          ("event_test", ev_te), ("anomaly_test", an_te)):
            np.save(os.path.join(d, f"{name}.npy"), arr)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"model": "xgboost", "params": PARAMS, "rounds": ROUNDS, "folds": folds,
                       "purge": purge, "run_id": rn.run_id, **info}, f, indent=1, default=str)
        rn.finish({"final_seconds": inf["seconds"]})
    print(f"{tag}: written to {d}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--channels", default="target")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.mission, a.channels, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
