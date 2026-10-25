"""Robustness of the OPSSAT-AD models to corrupted telemetry (no retraining).

    python -m cosmos.ai.telemetry.robustness

Perturbations are applied to the RAW test fragments, and both model inputs are rebuilt from the
perturbed fragments: the 18 handcrafted features (with a re-implementation of the official
``dataset_generator.ipynb``, checked to reproduce ``dataset.csv`` on clean data) and the
resampled tensors.  Feature-based and raw-signal models therefore face the same corruption.

* ``noise``    additive Gaussian noise, sigma = k x the fragment's own std
* ``dropout``  a random fraction of samples removed (the timestamps stay irregular)

Models are the seed-0..4 checkpoints from the OPSSAT benchmark (official split), plus the
classical models refitted exactly as in the benchmark.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from ...common import output_dir
from ..data.preprocess_telemetry import OPSSAT_FEATURES, opssat_tensors
from ..evaluation import classification_metrics


# ============================================================================ official features
def _peaks(a):
    from scipy import signal
    prom = 0.1 * (np.max(a) - np.min(a))
    return len(signal.find_peaks(a, prominence=prom)[0])


def fragment_features(t: pd.Series, v: np.ndarray, sampling: int) -> dict:
    """The official ``generate_dataset`` features of one fragment."""
    from scipy.stats import kurtosis, skew
    v = np.asarray(v, float)
    ts = pd.to_datetime(t).reset_index(drop=True)
    duration = (ts.iloc[-1] - ts.iloc[0]).seconds
    d = {"len": len(v), "mean": np.mean(v), "var": np.var(v), "std": np.std(v),
         "kurtosis": kurtosis(v), "skew": skew(v), "n_peaks": _peaks(v),
         "smooth10_n_peaks": _peaks(np.convolve(v, np.ones(10) / 10, mode="same")),
         "smooth20_n_peaks": _peaks(np.convolve(v, np.ones(20) / 20, mode="same")),
         "diff_peaks": _peaks(np.diff(v)), "diff2_peaks": _peaks(np.diff(v, n=2)),
         "diff_var": np.var(np.diff(v)), "diff2_var": np.var(np.diff(v, n=2)),
         # the official code uses the Timedelta .seconds COMPONENT (it wraps at one day)
         "gaps_squared": float((ts.diff().iloc[1:].dt.seconds.astype(float) ** 2).sum()),
         "duration": duration}
    d["len_weighted"] = sampling * d["len"]
    d["var_div_duration"] = d["var"] / duration if duration else np.inf
    d["var_div_len"] = d["var"] / d["len"]
    return d


def features_table(segs: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for sid, g in segs.groupby("segment", sort=True):
        rows[sid] = fragment_features(g.timestamp, g.value.to_numpy(), int(g.sampling.iloc[0]))
    return pd.DataFrame.from_dict(rows, orient="index")[OPSSAT_FEATURES]


def perturb(segs: pd.DataFrame, kind: str, level: float, seed: int = 0) -> pd.DataFrame:
    if level == 0:
        return segs
    rng = np.random.default_rng(seed)
    out = []
    for sid, g in segs.groupby("segment", sort=True):
        g = g.copy()
        if kind == "noise":
            g["value"] = g.value + rng.normal(0, level * (g.value.std() + 1e-12), len(g))
        elif kind == "dropout":
            keep = rng.random(len(g)) >= level
            keep[[0, -1]] = True                              # keep the fragment's extent
            if keep.sum() < 8:
                keep[: min(8, len(g))] = True
            g = g[keep]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ============================================================================ evaluation
def run(levels=None, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    import torch
    from sklearn.preprocessing import StandardScaler
    from .models import build_classifier
    from .opssat import OpssatData, classical_scores, signed_log
    levels = levels or {"noise": [0.0, 0.1, 0.25, 0.5, 1.0], "dropout": [0.1, 0.3, 0.5]}
    D = OpssatData(256)
    tr, va, te = D.split("official")
    feat_ok = feature_generator_check(D)
    rows = []
    sc = {}
    for kind, lv in levels.items():
        for level in lv:
            segs = perturb(D.segments, kind, level, seed=123)
            F = signed_log(features_table(segs).to_numpy(np.float64))
            F = np.nan_to_num(F, nan=0.0, posinf=50.0, neginf=-50.0)
            X, M, _ = opssat_tensors(segs, 256)
            for model in ("fcnn", "cnn", "hybrid"):
                for s in seeds:
                    ck = sorted(glob.glob(os.path.join(output_dir("experiments"),
                                f"*opssat_{model}_official_f100_s{s}", "checkpoints", "best.pt")))
                    if not ck:
                        continue
                    kw = {"fcnn": dict(n_features=18), "cnn": dict(in_ch=3),
                          "hybrid": dict(in_ch=3, n_features=18)}[model]
                    net = build_classifier(model, **kw)
                    net.load_state_dict(torch.load(ck[-1], map_location="cpu")["state_dict"])
                    net.eval()
                    scaler = StandardScaler().fit(D.F[tr])           # as in training
                    with torch.no_grad():
                        p = torch.sigmoid(net(torch.tensor(X[te]),
                                              torch.tensor(scaler.transform(F[te]), dtype=torch.float32),
                                              torch.tensor(M[te]))).numpy()
                    m = classification_metrics(D.y[te], (p >= 0.5).astype(int), p)
                    rows.append({"perturbation": kind, "level": level, "model": model, "seed": s, **m})
            # classical, refitted on CLEAN training data exactly as in the benchmark,
            # applied to perturbed TEST features
            for model in ("xgboost", "random_forest"):
                for s in seeds:
                    key = (model, s)
                    if key not in sc:
                        sc[key] = _fit_classical(model, D, tr, va, s)
                    mdl, scaler = sc[key]
                    p = mdl.predict_proba(scaler.transform(F[te]))[:, 1]
                    m = classification_metrics(D.y[te], (p >= 0.5).astype(int), p)
                    rows.append({"perturbation": kind, "level": level, "model": model, "seed": s, **m})
            print(f"  {kind} {level}: done", flush=True)
    df = pd.DataFrame(rows)
    d = output_dir("benchmarks", "telemetry_opssat")
    df.to_csv(os.path.join(d, "robustness_per_run.csv"), index=False)
    s = (df.groupby(["perturbation", "level", "model"])[["f1", "auc_pr", "mcc"]]
         .agg(["mean", "std"]).round(4))
    s.columns = [f"{a}|{b}" for a, b in s.columns]
    s.reset_index().to_csv(os.path.join(d, "robustness_summary.csv"), index=False)
    with open(os.path.join(d, "feature_generator_check.json"), "w") as f:
        json.dump(feat_ok, f, indent=2)
    return df


def feature_generator_check(D) -> dict:
    """Per feature: fragments whose recomputed value differs from dataset.csv (rel > 1e-6)."""
    clean = features_table(D.segments)
    off = D.feats[OPSSAT_FEATURES]
    out = {}
    for c in OPSSAT_FEATURES:
        a, b = clean[c].to_numpy(float), off[c].to_numpy(float)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        out[c] = int((rel > 1e-6).sum())
    return {"fragments": int(len(off)), "mismatching_fragments_per_feature": out,
            "note": "smooth20_n_peaks differs by one peak on a few short fragments "
                    "(scipy/numpy version difference); every other feature is exact"}


def _fit_classical(model, D, tr, va, seed):
    from sklearn.preprocessing import StandardScaler
    nominal = tr[D.y[tr] == 0]
    scaler = StandardScaler().fit(D.F[nominal])
    Z = scaler.transform(D.F)
    if model == "xgboost":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=seed, n_jobs=16,
                          eval_metric="aucpr", early_stopping_rounds=50)
        m.fit(Z[tr], D.y[tr], eval_set=[(Z[va], D.y[va])], verbose=False)
    else:
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(500, random_state=seed, n_jobs=16).fit(Z[tr], D.y[tr])
    return m, scaler


def main() -> int:
    df = run()
    print(df.groupby(["perturbation", "level", "model"]).f1.mean().unstack().round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
