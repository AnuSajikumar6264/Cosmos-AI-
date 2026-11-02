"""Collision-risk forecasting experiment (debris-analysis module).

    python -m cosmos.ai.conjunction.run

Experts, all predicting the same quantiles of the final risk r (log10 collision probability):

* persistence: the latest available risk plus the training distribution of (final - latest)
  within the latest-risk bin (an honest persistence forecast with uncertainty);
* learned: TabM, TabICLv2 and Kumo-Tabular (models published 2022-2026, the candidates for the
  gated expert), and gradient-boosted trees and a quantile MLP as reference baselines, on event
  features;
* COSMOS gated fusion: g(r) * q_persistence + (1 - g(r)) * q_learned, g fitted on out-of-fold
  training predictions; Mondrian conformal calibration (groups: latest-risk bin) on the same
  out-of-fold predictions (cross-conformal);
* the official ESA test set is read only for scoring (it was scored more than once during
  development; see docs/REVIEW_AND_FIXES.md).

Out-of-fold (OOF) predictions come from 5 event-hash folds with nested model selection: for
fold k a tuned learner runs its random search (the same configuration draws in every fold) on
the other four folds, training on three and early-stopping and selecting on fold k + 1, and
the chosen model predicts fold k.  The test models are tuned the same way on all training
events (trained on folds 1-4, early-stopped and selected on fold 0) and averaged over seeds.
TabICLv2 and Kumo-Tabular are not tuned and ignore the validation set, so their OOF models take
all four other folds as context and their test models all training events.  Every quantile array (experts and
fusions, OOF and test) is sorted and snapped to the bounds [-30, 0] before anything is fitted
on or scored with it, so the point mass at the -30 floor is matched exactly.

Official metric of the Kelvins challenge (see docs/TARGET_DEFINITIONS.md): high-risk threshold
r >= -6; F2 on the high-risk classification and MSE over the true high-risk events, combined as
L = MSE_HR / F2 (lower is better).  P(high) = P(r >= -6) is read off each quantile function;
AUC-PR and AUC-ROC rank the raw P(high), while Brier, ECE and the decision rule use P(high)
calibrated by isotonic regression on the OOF predictions (its steps would tie the ranking).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from ...common import output_dir
from ..evaluation import brier_score, expected_calibration_error
from ..lifetime import models as MO
from ..lifetime.fusion import Gate, MondrianCQR, snap_bounds
from . import data as CD

BOUNDS = (CD.RISK_FLOOR, 0.0)
BINS = [-np.inf, -29.999, -15.0, -10.0, -6.0, np.inf]
SEEDS = (0, 1, 2, 3, 4)
UNTUNED = ("tabicl", "kumo")   # no search and no early stopping: fit ignores the validation set
LEARNER_SET = ("xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo")


SNAP_TOL = 1e-4                # log10 risk; above the float32 spacing at |-30| (1.9e-6)


def snap(q):
    """Quantiles sorted along tau and snapped to BOUNDS (values within SNAP_TOL of a bound set to it)."""
    return snap_bounds(np.sort(q, 1), *BOUNDS, tol=SNAP_TOL)


# ------------------------------------------------------------------------------ metric
def kelvins_score(y, yhat, threshold=CD.HIGH_RISK, clip_low=None) -> dict:
    """F2, MSE on true high-risk events, and L = MSE_HR / F2.  ``clip_low`` optionally raises
    predictions below it (the challenge clipped predictions at -6.001; see the docs)."""
    yhat = np.asarray(yhat, float)
    if clip_low is not None:
        yhat = np.maximum(yhat, clip_low)
    t, p = y >= threshold, yhat >= threshold
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    mse_hr = float(np.mean((y[t] - yhat[t]) ** 2)) if t.any() else float("nan")
    return {"F2": f2, "MSE_HR": mse_hr, "L": mse_hr / f2 if f2 > 0 else float("inf"),
            "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def p_high(q, threshold=CD.HIGH_RISK):
    """P(r >= threshold) from the quantile function (linear interpolation in tau)."""
    taus = np.concatenate([[0.0], MO.QUANTILES, [1.0]])
    out = np.empty(len(q))
    for i in range(len(q)):
        qq = np.concatenate([[BOUNDS[0] - 1e-6], q[i], [BOUNDS[1] + 1e-6]])
        qq = np.maximum.accumulate(qq + np.arange(qq.size) * 1e-9)
        out[i] = 1 - np.interp(threshold, qq, taus)
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------------------ experts
def design(df: pd.DataFrame, cols=None, logcols=None):
    """Event features; columns spanning many decades (covariance determinants and sigmas reach
    1e46) get a signed log10 transform, chosen on the training data and reused for test."""
    X = df.drop(columns=["y_final"]).copy()
    X = pd.get_dummies(X, columns=["c_object_type", "mission_id"], dtype=float)
    X = X.astype(float).replace([np.inf, -np.inf], np.nan)
    if logcols is None:
        logcols = [c for c in X.columns if X[c].abs().max() > 1e6]
    for c in logcols:
        if c in X:
            X[c] = np.sign(X[c]) * np.log10(1 + X[c].abs())
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X, logcols


def persistence_quantiles(latest_tr, y_tr, latest):
    b_tr = np.digitize(latest_tr, BINS)
    b = np.digitize(latest, BINS)
    resid = y_tr - latest_tr
    q = np.empty((len(latest), len(MO.QUANTILES)))
    allq = np.quantile(resid, MO.QUANTILES)
    for k in np.unique(b):
        sel = b_tr == k
        rq = np.quantile(resid[sel], MO.QUANTILES) if sel.sum() >= 30 else allq
        q[b == k] = latest[b == k, None] + rq
    return snap(q)


def reliability(df: pd.DataFrame) -> np.ndarray:
    lr = df["latest_risk"].to_numpy()
    cols = [np.log1p(df["n_cdm"].to_numpy()), df["last_time_to_tca"].to_numpy(),
            np.log1p(np.abs(df["risk_slope3"].to_numpy())), np.nan_to_num(df["risk_last_step"].to_numpy()),
            (lr <= -29.999).astype(float), ((lr > -10) & (lr < -6)).astype(float), (lr >= -6).astype(float),
            df["c_object_type"].eq("DEBRIS").to_numpy(float),
            np.log10(np.maximum(df["last_c_sigma_r"].to_numpy(), 1e-3)),
            np.log10(np.maximum(df["last_miss_distance"].to_numpy(), 1.0))]
    return np.nan_to_num(np.column_stack(cols))


def mondrian_key(df):
    return pd.Series(np.digitize(df["latest_risk"].to_numpy(), BINS)).astype(str)


# ------------------------------------------------------------------------------ driver
def _selected(trials, params) -> dict:
    """The trial the tuner kept (``params`` is that trial's own dict)."""
    t = next(t for t in trials if t["params"] is params)
    return {"trial": t["trial"], "val_pinball": t["val_pinball"], "params": params}


def run(learners=LEARNER_SET, seeds=SEEDS, n_trials=8, out=None):
    t_start = time.time()
    out = out or output_dir("benchmarks", "conjunction_kelvins_cac")
    B = CD.build()
    tr, te = B["train"], B["test"]
    log = {"data": B["log"]}
    fold = np.array([int(hashlib.md5(str(e).encode()).hexdigest(), 16) % 5 for e in tr.index])
    Xtr, logcols = design(tr)
    cols = list(Xtr.columns)
    Xte, _ = design(te, cols, logcols)
    log["log_transformed_columns"] = logcols
    ytr = tr["y_final"].to_numpy()                           # test labels: read only for scoring, below
    tune = fold == 0                                         # model selection split of the test models
    qp_te = persistence_quantiles(tr["latest_risk"].to_numpy(), ytr, te["latest_risk"].to_numpy())
    qp_oof = np.empty((len(tr), len(MO.QUANTILES)))
    for k in range(5):
        h = fold == k
        qp_oof[h] = persistence_quantiles(tr["latest_risk"].to_numpy()[~h], ytr[~h], tr["latest_risk"].to_numpy()[h])
    cand = {"persistence": {"oof": qp_oof, "test": qp_te}}
    fits = {}
    for name in learners:
        t0 = time.time()
        oof, sel = np.empty((len(tr), len(MO.QUANTILES))), []
        if name in UNTUNED:                                  # the tuner's single configuration; context = all of ~fold k
            params = MO.LEARNERS[name](seed=0).space(np.random.default_rng(1000))
            for k in range(5):
                h = fold == k
                oof[h] = MO.LEARNERS[name](seed=0).bounds(*BOUNDS).fit(Xtr[~h], ytr[~h], None, None, params).predict(Xtr[h])
                sel.append({"fold": k, "params": params})
            test_sel, test_fit = {"params": params}, (Xtr, ytr, None, None)
        else:
            for k in range(5):                               # nested: fold k is never seen by its model's selection
                h, es = fold == k, fold == (k + 1) % 5       # train on three folds, early-stop and select on the next
                m, p, trials = MO.tune_and_fit(name, Xtr[~h & ~es], ytr[~h & ~es], Xtr[es], ytr[es], seed=0,
                                               n_trials=n_trials, bounds=BOUNDS)
                oof[h] = m.predict(Xtr[h])
                sel.append({"fold": k, "select_fold": (k + 1) % 5, **_selected(trials, p)})
            _, params, trials = MO.tune_and_fit(name, Xtr[~tune], ytr[~tune], Xtr[tune], ytr[tune], seed=0,
                                                n_trials=n_trials, bounds=BOUNDS)
            test_sel, test_fit = _selected(trials, params), (Xtr[~tune], ytr[~tune], Xtr[tune], ytr[tune])
        te_seeds = [snap(MO.LEARNERS[name](seed=s).bounds(*BOUNDS).fit(*test_fit, params).predict(Xte)) for s in seeds]
        cand[name] = {"oof": snap(oof), "test": snap(np.mean(te_seeds, 0)), "test_seeds": te_seeds}
        fits[name] = {**test_sel, "oof_selection": sel, "seconds": time.time() - t0,
                      "oof_pinball": MO.pinball(ytr, cand[name]["oof"])}
        print(f"  {name}: OOF pinball {fits[name]['oof_pinball']:.4f} ({fits[name]['seconds']:.0f}s)", flush=True)
    # the expert is chosen among the models published 2022-2026 (TabM, TabICLv2, Kumo-Tabular);
    # the gradient-boosted trees and the plain MLP are reference baselines
    pool = [n for n in learners if n in MO.RECENT] or list(learners)
    best = min(pool, key=lambda n: fits[n]["oof_pinball"])
    log["best_learner_oof"] = best
    log["selection_pool"] = pool
    Rtr, Rte = reliability(tr), reliability(te)
    for mode, label in (("gated", "cosmos_gated"), ("constant", "stack_constant")):
        g = Gate(mode=mode).fit(Rtr, qp_oof, cand[best]["oof"], ytr)
        cand[label] = {"oof": snap(g.fuse(Rtr, qp_oof, cand[best]["oof"])),     # g q + (1 - g) q can be an ulp off q
                       "test": snap(g.fuse(Rte, qp_te, cand[best]["test"]))}
        log[f"gate_{label}"] = {"theta": g.theta.tolist(), "test_weight_mean": float(g.weight(Rte).mean())}
    yte = te["y_final"].to_numpy()                           # every candidate is fixed: test labels from here on
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "candidates.npz"), ytr=ytr, yte=yte,
                        **{f"{k}__{kk}": vv for k, v in cand.items() for kk, vv in v.items() if kk in ("oof", "test")})
    # ---- evaluation
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    ytr_hi, yte_hi = (ytr >= CD.HIGH_RISK).astype(float), (yte >= CD.HIGH_RISK).astype(float)
    Gtr = pd.DataFrame({"fine": mondrian_key(tr), "coarse": "all", "all": "all"})
    Gte = pd.DataFrame({"fine": mondrian_key(te), "coarse": "all", "all": "all"})
    rows = []
    latest = te["latest_risk"].to_numpy()
    rows.append({"model": "latest_risk_point", **kelvins_score(yte, latest),
                 "mse_all": float(np.mean((yte - latest) ** 2)), "mae_all": float(np.mean(np.abs(yte - latest)))})
    rows[-1].update({f"{k}_clip": v for k, v in kelvins_score(yte, latest, clip_low=-6.001).items()})
    for name, c in cand.items():
        q, qo = c["test"], c["oof"]
        med = q[:, 3]
        cq = MondrianCQR(levels=(0.8, 0.9), bounds=BOUNDS).fit(qo, ytr, Gtr)
        # endpoints snapped as well: an event at the -30 floor is covered iff its lower end is exactly -30
        lo80, hi80 = (snap_bounds(v, *BOUNDS, tol=SNAP_TOL) for v in cq.interval(q, Gte, 0.8))
        lo90, hi90 = (snap_bounds(v, *BOUNDS, tol=SNAP_TOL) for v in cq.interval(q, Gte, 0.9))
        po, pt = p_high(qo), p_high(q)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(po, ytr_hi)
        pc = iso.predict(pt)                                 # calibrated: Brier, ECE, decision rule; pt: ranking
        # decision rule chosen on out-of-fold training predictions: classify high-risk when the
        # calibrated P(high) exceeds the threshold that maximises the OOF F2
        grid = np.linspace(0.02, 0.9, 45)
        pco = iso.predict(po)
        f2s = [kelvins_score(ytr, np.where(pco >= t_, np.maximum(qo[:, 3], -6.0), np.minimum(qo[:, 3], -6.001)))["F2"] for t_ in grid]
        thr = float(grid[int(np.argmax(f2s))])
        dec = np.where(pc >= thr, np.maximum(med, -6.0), np.minimum(med, -6.001))
        r = {"model": name, **kelvins_score(yte, med)}
        r.update({f"{k}_clip": v for k, v in kelvins_score(yte, med, clip_low=-6.001).items()})
        r.update({f"{k}_decision": v for k, v in kelvins_score(yte, dec).items()})
        r.update({f"{k}_decision_clip": v for k, v in kelvins_score(yte, dec, clip_low=-6.001).items()})
        r.update({"decision_threshold": thr, "pinball": MO.pinball(yte, q), "mse_all": float(np.mean((yte - med) ** 2)),
                  "mae_all": float(np.mean(np.abs(yte - med))),
                  "auc_pr_high": float(average_precision_score(yte_hi, pt)), "auc_roc_high": float(roc_auc_score(yte_hi, pt)),
                  "brier_high": brier_score(yte_hi, pc), "ece_high": expected_calibration_error(yte_hi, pc),
                  "cov80": float(np.mean((yte >= lo80) & (yte <= hi80))), "cov90": float(np.mean((yte >= lo90) & (yte <= hi90))),
                  "width80": float(np.mean(hi80 - lo80)), "width90": float(np.mean(hi90 - lo90)),
                  "cov90_high": float(np.mean(((yte >= lo90) & (yte <= hi90))[yte_hi == 1]))})
        rows.append(r)
        if "test_seeds" in c:
            r["seed_L_mean"] = float(np.mean([kelvins_score(yte, s[:, 3])["L"] for s in c["test_seeds"]]))
            r["seed_L_sd"] = float(np.std([kelvins_score(yte, s[:, 3])["L"] for s in c["test_seeds"]]))
    res = pd.DataFrame(rows)
    # bootstrap CI of L for the main models (resampling events)
    rng = np.random.default_rng(20261027)
    boot = {}
    def ci(pred):
        vals = []
        for _ in range(2000):
            i = rng.integers(0, len(yte), len(yte))
            vals.append(kelvins_score(yte[i], pred[i], clip_low=-6.001)["L"])
        v = np.asarray(vals)
        fin = v[np.isfinite(v)]
        return {"ci95": [float(np.quantile(fin, 0.025)), float(np.quantile(fin, 0.975))] if fin.size else None,
                "share_infinite": float(np.mean(~np.isfinite(v)))}
    for name in ("persistence", best, "cosmos_gated"):
        boot[name] = ci(cand[name]["test"][:, 3])
    boot["latest_risk_point"] = ci(latest)
    log.update({"fits": fits, "bootstrap_L_ci95": boot, "seconds": time.time() - t_start})
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "test_results.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
    np.savez_compressed(os.path.join(out, "test_predictions.npz"), y=yte, latest=latest,
                        **{f"{k}__q": v["test"] for k, v in cand.items()})
    print(res[["model", "L", "L_clip", "F2", "MSE_HR", "L_decision_clip", "auc_pr_high", "cov90", "pinball"]].to_string(index=False))
    return res, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default=",".join(LEARNER_SET))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args(argv)
    run(tuple(a.learners.split(",")), tuple(range(a.seeds)), a.trials)


if __name__ == "__main__":
    main()


def gated_persistence_eval(out=None):
    """Use each model's calibrated P(high risk) to gate the persistence point forecast:
    predict the latest risk when P(high) >= threshold, else -6.001.  The threshold is chosen on
    out-of-fold training predictions by minimising the clipped official score; the test set is
    then scored once.  This is the persistence-anchored decision rule of the twin's debris
    module; it is compared with the latest-risk baseline (published value 0.694)."""
    from sklearn.isotonic import IsotonicRegression
    out = out or output_dir("benchmarks", "conjunction_kelvins_cac")
    C = np.load(os.path.join(out, "candidates.npz"))
    B = CD.build()
    tr, te = B["train"], B["test"]
    ytr, yte = C["ytr"], C["yte"]
    lat_tr, lat_te = tr["latest_risk"].to_numpy(), te["latest_risk"].to_numpy()
    rng = np.random.default_rng(20261027)
    idx = [rng.integers(0, len(yte), len(yte)) for _ in range(2000)]
    rows = []
    base = np.asarray([kelvins_score(yte[i], lat_te[i], clip_low=-6.001)["L"] for i in idx])
    for name in sorted({k.split("__")[0] for k in C.files if k.endswith("__oof")}):
        qo, qt = snap(C[f"{name}__oof"]), snap(C[f"{name}__test"])          # run() saves them snapped already
        po, pt = p_high(qo), p_high(qt)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(po, (ytr >= CD.HIGH_RISK).astype(float))
        pco, pct = iso.predict(po), iso.predict(pt)
        grid = np.unique(np.concatenate([np.linspace(0.01, 0.95, 95), pco]))
        Ls = [kelvins_score(ytr, np.where(pco >= t_, lat_tr, -6.001), clip_low=-6.001)["L"] for t_ in grid]
        thr = float(grid[int(np.nanargmin(Ls))])
        pred = np.where(pct >= thr, lat_te, -6.001)
        s = kelvins_score(yte, pred, clip_low=-6.001)
        diff = np.asarray([kelvins_score(yte[i], pred[i], clip_low=-6.001)["L"] for i in idx]) - base
        ok = np.isfinite(diff)
        rows.append({"model": f"{name}_gated_persistence", "threshold_oof": thr, **s,
                     "L_minus_latest_ci95": [float(np.quantile(diff[ok], 0.025)), float(np.quantile(diff[ok], 0.975))],
                     "p_better_than_latest": float(np.mean(diff[ok] < 0))})
    s0 = kelvins_score(yte, lat_te, clip_low=-6.001)
    rows.insert(0, {"model": "latest_risk_point", "threshold_oof": None, **s0,
                    "L_ci95": [float(np.quantile(base, 0.025)), float(np.quantile(base, 0.975))]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "gated_persistence_test.csv"), index=False)
    print(res[["model", "threshold_oof", "L", "F2", "MSE_HR", "tp", "fp", "fn"]].round(4).to_string(index=False))
    return res


PINBALL_PAIRS = (("kumo", "tabicl"), ("kumo", "xgboost"), ("kumo", "tabm"), ("kumo", "persistence"))


def pinball_pairs(out=None, pairs=PINBALL_PAIRS, n_boot=2000):
    """Post-hoc paired bootstrap over test events of the pinball-loss difference a - b
    (< 0: a is better), from the saved test quantiles.  Added after the test results were seen."""
    out = out or output_dir("benchmarks", "conjunction_kelvins_cac")
    P = np.load(os.path.join(out, "test_predictions.npz"))
    y = P["y"]

    def loss(q):
        d = y[:, None] - q
        return np.mean(np.maximum(MO.QUANTILES * d, (MO.QUANTILES - 1) * d), 1)
    rng = np.random.default_rng(20261027)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for a, b in pairs:
        diff = loss(P[f"{a}__q"]) - loss(P[f"{b}__q"])
        v = np.asarray([diff[i].mean() for i in idx])
        rows.append({"a": a, "b": b, "pinball_a_minus_b": float(diff.mean()),
                     "ci95_lo": float(np.quantile(v, 0.025)), "ci95_hi": float(np.quantile(v, 0.975))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "pinball_pairs_test.csv"), index=False)
    return res
