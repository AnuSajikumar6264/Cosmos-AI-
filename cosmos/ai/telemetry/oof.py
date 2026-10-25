"""Out-of-fold forecaster scores over the whole official TRAINING half.

    python -m cosmos.ai.telemetry.oof --mission 1 --encoder ar8

Why this exists: :mod:`esa_benchmark` selects its alarm settings on the last months of the
training half only.  On Mission 1 that window holds 9 labelled events, so the selection is
noisy.  Out-of-fold (OOF) scores give every training sample a score from a model that never
saw it.  Selection and the supervised event model (:mod:`event_model`) can then use all
training labels: 105 events on Mission 1 and 282 on Mission 2.  The test half is not read
for any choice.

Folds are ``K`` contiguous blocks of the training half.  For block ``b`` the forecaster is fitted
on the nominal samples of the other blocks, minus ``purge_days`` on each side of ``b``.  Inputs
(the lagged values at each origin) are read from the record itself, which is what any
deployed forecaster does.  The test model is fitted on every nominal training sample, and
its test scores are written next to the OOF scores.

Output: ``outputs/benchmarks/telemetry_esa_adb/oof/m{mission}_{channels}_{encoder}/``
holding ``scores_train_oof.npy`` and ``scores_test.npy`` (float32, (T, C), NaN where no
forecast exists), ``folds.json`` and ``sigma.npy``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from ...common import output_dir
from .forecasting import DELTA, MissionData, origins

ORDER = 8
HORIZON = 16                         # = ForecastConfig.horizon, the shared origin schedule
MAX_FIT = 500_000                    # = _ar_forecast's subsample


def oof_dir(mission: int, channels: str, encoder: str) -> str:
    return output_dir("benchmarks", "telemetry_esa_adb", "oof",
                      f"m{mission}_{channels}_{encoder}")


def fold_bounds(n: int, K: int) -> list:
    e = np.linspace(0, n, K + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(e[:-1], e[1:])]


def _ar_fit(z: np.ndarray, usable: np.ndarray, rng, order: int = ORDER) -> np.ndarray:
    """Ridge AR(order) on samples whose target and lags are all ``usable``."""
    ok = usable.copy()
    for k in range(1, order + 1):
        ok[k:] &= usable[:-k]
    ok[:order] = False
    idx = np.flatnonzero(ok)
    if idx.size > MAX_FIT:
        idx = np.sort(idx[rng.permutation(idx.size)[:MAX_FIT]])
    if idx.size < 10 * order:
        w = np.zeros(order + 1)
        w[0] = 1.0                                   # fall back to persistence
        return w
    A = np.stack([z[idx - k] for k in range(1, order + 1)] + [np.ones(idx.size)], 1)
    return np.linalg.solve(A.T @ A + 1e-3 * np.eye(order + 1), A.T @ z[idx])


def _ar_predict(z: np.ndarray, o: np.ndarray, w: np.ndarray, H: int = HORIZON) -> np.ndarray:
    """Iterated H-step forecasts from origins ``o`` (same schedule as ``forecast_record``)."""
    order = w.size - 1
    n = z.size
    out = np.full(n, np.nan, np.float32)
    hist = np.stack([z[np.maximum(o - k + 1, 0)] for k in range(1, order + 1)], 1)
    for h in range(1, H + 1):
        pred = hist @ w[:order] + w[order]
        t = o + h
        ok = t < n
        out[t[ok]] = pred[ok]
        hist = np.concatenate([pred[:, None], hist[:, :-1]], 1)
    return out


def _sigma(r: np.ndarray, usable: np.ndarray) -> float:
    """``forecasting.residual_scale``'s robust scale on usable residuals."""
    v = np.abs(r[usable & np.isfinite(r)])
    if v.size == 0:
        return 1.0
    return float(max(1.4826 * np.median(v), np.percentile(v, 90) / 1.645, DELTA))


def ar8_oof(D: MissionData, K: int = 5, purge_days: float = 1.0, seed: int = 0) -> dict:
    """OOF AR(8) non-conformity on the training half and full-fit scores on the test half."""
    rng = np.random.default_rng(seed)
    n, C = D.Ztr.shape
    folds = fold_bounds(n, K)
    purge = int(purge_days * 86400 / D.train.rule_s)
    o_tr = origins(n, 1, HORIZON)
    Zte_rec, _L, _cov, warm = D.record("test")
    o_te = origins(Zte_rec.shape[0], warm, HORIZON)
    S_tr = np.full((n, C), np.nan, np.float32)
    S_te = np.full((Zte_rec.shape[0] - warm, C), np.nan, np.float32)
    sig_fold = np.ones((K, C), np.float32)
    sig_full = np.ones(C, np.float32)
    for c in range(C):
        z = D.Ztr[:, c].astype(np.float64)
        nominal = D.Ltr[:, c] == 0
        for b, (a, e) in enumerate(folds):
            usable = nominal.copy()
            usable[max(0, a - purge):min(n, e + purge)] = False
            w = _ar_fit(z, usable, rng)
            q = _ar_predict(z, o_tr, w)
            r = z - q
            sig_fold[b, c] = _sigma(r, usable)
            S_tr[a:e, c] = (np.abs(r[a:e]) / max(sig_fold[b, c], DELTA)).astype(np.float32)
        w = _ar_fit(z, nominal, rng)
        sig_full[c] = _sigma(z - _ar_predict(z, o_tr, w), nominal)
        zt = Zte_rec[:, c].astype(np.float64)
        q = _ar_predict(zt, o_te, w)
        S_te[:, c] = (np.abs(zt - q)[warm:] / max(sig_full[c], DELTA)).astype(np.float32)
    return {"S_tr": S_tr, "S_te": S_te, "folds": folds, "purge": purge,
            "sigma_fold": sig_fold, "sigma_full": sig_full}


def run(mission: int, channels: str = "target", encoder: str = "ar8", K: int = 5,
        purge_days: float = 1.0) -> str:
    if encoder != "ar8":
        raise NotImplementedError("OOF scores are implemented for the AR(8) encoder")
    t0 = time.time()
    D = MissionData(mission, channels)
    out = ar8_oof(D, K, purge_days)
    d = oof_dir(mission, channels, encoder)
    np.save(os.path.join(d, "scores_train_oof.npy"), out["S_tr"])
    np.save(os.path.join(d, "scores_test.npy"), out["S_te"])
    np.save(os.path.join(d, "sigma.npy"), np.vstack([out["sigma_fold"], out["sigma_full"]]))
    h = hashlib.sha256(out["S_tr"][::97].tobytes()).hexdigest()[:12]
    meta = {"mission": mission, "channels": channels, "encoder": encoder, "K": K,
            "purge_days": purge_days, "purge_steps": out["purge"], "folds": out["folds"],
            "n_train": int(out["S_tr"].shape[0]), "n_test": int(out["S_te"].shape[0]),
            "channel_names": D.channels, "seconds": time.time() - t0, "sha256_sample": h}
    with open(os.path.join(d, "folds.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"m{mission} {channels} {encoder}: OOF scores in {time.time() - t0:.0f} s -> {d}")
    return d


def load(mission: int, channels: str = "target", encoder: str = "ar8"):
    d = oof_dir(mission, channels, encoder)
    with open(os.path.join(d, "folds.json")) as f:
        meta = json.load(f)
    return (np.load(os.path.join(d, "scores_train_oof.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "scores_test.npy"), mmap_mode="r"), meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--channels", default="target")
    ap.add_argument("--encoder", default="ar8")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--purge-days", type=float, default=1.0)
    a = ap.parse_args(argv)
    run(a.mission, a.channels, a.encoder, a.folds, a.purge_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
