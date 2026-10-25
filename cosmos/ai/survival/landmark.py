"""Re-entry within a horizon, from a leakage-free landmark (see :mod:`..data.preprocess_catalogue`).

    python -m cosmos.ai.survival.landmark

Target: decay within 30 and within 68 days of tau = 2026-08-16 (the end of follow-up), and
the time to decay within the window.  All non-events are censored at the same date, so for
these horizons every label is observed (no subject is censored before the horizon) — the
models are still fitted with censoring-correct likelihoods so that the code is right when
landmarks with earlier censoring are added.

Models (one protocol: train = fit, val = early stopping / ensembling / Platt scaling of the
physics score, test = scored once; splits by launch so siblings never straddle):

``physics``             COSMOS orbital-lifetime model at the perigee altitude with the default
                        area-to-mass (``cosmos.debris.AM_DEFAULT``); risk = -log lifetime,
                        Platt-scaled on val for probabilities
``logreg``              logistic regression per horizon
``xgboost``             gradient-boosted trees per horizon (strong tabular baseline)
``xgboost_aft``         XGBoost accelerated-failure-time survival model (censoring-native)
``hazard_mlp``          discrete-time hazard MLP (7-day bins), deep ensemble of 5, with the
                        physics log-lifetime as an input feature
``hazard_mlp_nophys``   the same without the physics feature (ablation)

Metrics: AUC-PR, Brier, ECE (horizon probabilities); Harrell's C on time to decay; cluster
(launch-group) bootstrap CIs; split-conformal binary prediction sets (val-calibrated) with
their test coverage; risk-coverage curves using ensemble disagreement.  Results are split
into all objects and uncontrolled objects (DEB + R/B), because payload decays include
controlled de-orbits.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from ..data.preprocess_catalogue import (CATEGORICAL, FEATURES, FOLLOW, TAU, build_landmark,
                                         leakage_report)
from ..evaluation import brier_score, expected_calibration_error, harrell_c
from ..experiment import Run, count_parameters, pick_device, set_seed

HORIZONS = (30, 68)
BIN_DAYS = 7


def load() -> pd.DataFrame:
    p = os.path.join(feature_dir("reentry", "derived"), f"landmark_{TAU}_to_{FOLLOW}.csv")
    if not os.path.isfile(p):
        build_landmark()
    return pd.read_csv(p, parse_dates=["gp_epoch", "decay_follow"])


def physics_lifetime_days(df: pd.DataFrame, area_to_mass: float | None = None) -> np.ndarray:
    """COSMOS ``orbital_lifetime`` at each object's perigee (cached per 5 km altitude bin)."""
    from ...debris.lifetime import AM_DEFAULT, orbital_lifetime
    am = AM_DEFAULT if area_to_mass is None else area_to_mass
    alt = np.clip(df.PERIGEE.to_numpy(float), 125.0, 2000.0)
    bins = np.round(alt / 5.0) * 5.0
    u = np.unique(bins)
    life = {a: float(np.atleast_1d(orbital_lifetime(a, am))[0]) * 365.25 for a in u}
    return np.array([life[b] for b in bins])


def design(df: pd.DataFrame, train_mask: np.ndarray, with_physics: bool = True,
           quantile: bool = False):
    """Numeric features (median-imputed + missing flags, standardised on train) + one-hots.

    ``quantile=True`` maps every numeric feature through a rank-Gaussian transform fitted on
    the training rows (heavy-tailed orbital features otherwise dominate an MLP's inputs).
    """
    X = df[FEATURES].copy()
    for c in ["gp_bstar", "gp_mean_motion", "gp_ndot", "gp_ecc", "gp_age_days"]:
        X[c] = X[c].astype(float)
    X["gp_bstar"] = np.sign(X.gp_bstar) * np.log10(1 + 1e5 * np.abs(X.gp_bstar))
    X["gp_ndot"] = np.sign(X.gp_ndot) * np.log10(1 + 1e6 * np.abs(X.gp_ndot))
    if with_physics:
        X["phys_log_life"] = np.log10(np.clip(df["phys_life_days"], 1e-2, 1e7))
    med = X[train_mask].median()
    X = X.fillna(med)
    if quantile:
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                 random_state=0).fit(X[train_mask])
        X = pd.DataFrame(qt.transform(X), columns=X.columns, index=X.index)
    mu, sd = X[train_mask].mean(), X[train_mask].std().replace(0, 1)
    X = (X - mu) / sd
    for c in CATEGORICAL:
        top = df.loc[train_mask, c].value_counts().index[:12]
        for v in top:
            X[f"{c}={v}"] = (df[c] == v).astype(float)
    return X.to_numpy(np.float32), list(X.columns)


# ============================================================================ neural hazard
def _hazard_targets(t, e, n_bins):
    """Discrete-time likelihood pieces: at-risk mask (N, B) and event-bin one-hot (N, B)."""
    k = np.minimum((np.asarray(t) - 1) // BIN_DAYS, n_bins - 1).astype(int)
    at_risk = (np.arange(n_bins)[None, :] <= k[:, None]).astype(np.float32)
    ev = np.zeros_like(at_risk)
    ev[np.arange(len(k))[np.asarray(e) == 1], k[np.asarray(e) == 1]] = 1.0
    return at_risk, ev


def train_hazard_mlp(X, t, e, tr, va, seed, device, hidden=(128, 64), epochs=300, lr=1e-3):
    import torch
    from torch import nn
    set_seed(seed)
    n_bins = int(np.ceil(68 / BIN_DAYS))
    R, E = _hazard_targets(t, e, n_bins)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(0.1)]
        d = h
    net = nn.Sequential(*layers, nn.Linear(d, n_bins)).to(device)
    with torch.no_grad():                                    # start at the base hazard
        net[-1].bias.fill_(float(np.log(max(e[tr].mean() / n_bins, 1e-4))))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.as_tensor(X, device=device)
    Rt, Et = torch.as_tensor(R, device=device), torch.as_tensor(E, device=device)

    def nll(idx):
        logit = net(Xt[idx])
        # censoring-correct discrete-time likelihood: sum over at-risk bins of BCE(hazard)
        l = nn.functional.binary_cross_entropy_with_logits(logit, Et[idx], reduction="none")
        return (l * Rt[idx]).sum() / len(idx)

    rng = np.random.default_rng(seed)
    best, state, bad = np.inf, None, 0
    tr_t = torch.as_tensor(tr, device=device)
    for ep in range(epochs):
        net.train()
        perm = tr_t[torch.as_tensor(rng.permutation(len(tr)), device=device)]
        for i in range(0, len(perm), 512):
            loss = nll(perm[i:i + 512])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nll(torch.as_tensor(va, device=device)))
        if v < best - 1e-5:
            best, bad = v, 0
            state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 25:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        h = torch.sigmoid(net(Xt)).cpu().numpy()
    surv = np.cumprod(1 - h, axis=1)                         # S at end of each bin
    return net, surv, {"val_nll": best, "epochs": ep + 1}


def surv_at(surv: np.ndarray, days: float) -> np.ndarray:
    """S(days) by linear interpolation between bin ends (S(0) = 1)."""
    ends = np.arange(1, surv.shape[1] + 1) * BIN_DAYS
    full = np.c_[np.ones(len(surv)), surv]
    xs = np.r_[0, ends]
    return np.array([np.interp(days, xs, row) for row in full])


# ============================================================================ study
def _platt(score_va, y_va):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e3, max_iter=5000).fit(score_va[:, None], y_va)
    return lambda s: m.predict_proba(np.asarray(s)[:, None])[:, 1]


def evaluate(df, te, prob: dict, risk: np.ndarray) -> dict:
    out = {}
    for sub, mask in (("all", np.ones(len(te), bool)),
                      ("uncontrolled", df.uncontrolled.to_numpy()[te] == 1)):
        idx = te[mask]
        t, e = df.time_days.to_numpy()[idx], df.event.to_numpy()[idx]
        out[f"{sub}/n"] = int(len(idx))
        out[f"{sub}/events"] = int(e.sum())
        out[f"{sub}/c_index"] = harrell_c(t, e, risk[idx]) if e.sum() else float("nan")
        for h in HORIZONS:
            y = ((t <= h) & (e == 1)).astype(int)
            p = prob[h][idx]
            from sklearn.metrics import average_precision_score, roc_auc_score
            out[f"{sub}/h{h}/positives"] = int(y.sum())
            if y.sum() and y.sum() < len(y):
                out[f"{sub}/h{h}/auc_pr"] = float(average_precision_score(y, p))
                out[f"{sub}/h{h}/auc_roc"] = float(roc_auc_score(y, p))
            out[f"{sub}/h{h}/brier"] = brier_score(y, p)
            out[f"{sub}/h{h}/ece"] = expected_calibration_error(y, p)
    return out


def cluster_bootstrap_ci(df, te, p_a, p_b, h, n_boot=2000, seed=0):
    """CI of AUC-PR(a) - AUC-PR(b) resampling LAUNCH GROUPS of the test set."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    if len(p_a) == len(df):                                   # full-population arrays
        p_a, p_b = np.asarray(p_a)[te], np.asarray(p_b)[te]
    g = df.launch_group.to_numpy()[te]
    y = ((df.time_days.to_numpy()[te] <= h) & (df.event.to_numpy()[te] == 1)).astype(int)
    ug = np.unique(g)
    pos = {k: np.flatnonzero(g == k) for k in ug}
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([pos[k] for k in rng.choice(ug, len(ug))])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], p_a[idx])
                     - average_precision_score(y[idx], p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(average_precision_score(y, p_a) - average_precision_score(y, p_b)),
            "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(diffs)}


def conformal_sets(p_va, y_va, p_te, y_te, alpha=0.1) -> dict:
    """Split-conformal binary prediction sets (score = 1 - p(true class)), val-calibrated."""
    from ..uncertainty import split_conformal_quantile
    s = 1 - np.where(y_va == 1, p_va, 1 - p_va)
    q = split_conformal_quantile(s, alpha)
    inc1 = (1 - p_te) <= q
    inc0 = p_te <= q
    cov = np.where(y_te == 1, inc1, inc0).mean()
    size = (inc1.astype(int) + inc0.astype(int)).mean()
    pos = y_te == 1
    return {"alpha": alpha, "coverage": float(cov), "mean_set_size": float(size),
            "coverage_on_events": float(np.where(pos, inc1, inc0)[pos].mean()) if pos.any()
            else float("nan"), "singleton_fraction": float(((inc1 ^ inc0)).mean())}


def mondrian_sets(p_va, y_va, p_te, y_te, alpha=0.1) -> dict:
    """Label-conditional (Mondrian) conformal sets: coverage >= 1 - alpha WITHIN each class.

    Marginal sets are dominated by the 99 % non-events and can ignore every event while
    meeting their nominal coverage; calibrating each class separately cannot.
    """
    from ..uncertainty import split_conformal_quantile
    q1 = split_conformal_quantile(1 - p_va[y_va == 1], alpha)
    q0 = split_conformal_quantile(p_va[y_va == 0], alpha)
    inc1, inc0 = (1 - p_te) <= q1, p_te <= q0
    pos = y_te == 1
    return {"alpha": alpha, "q_event": q1, "q_nonevent": q0,
            "coverage_events": float(inc1[pos].mean()) if pos.any() else float("nan"),
            "coverage_nonevents": float(inc0[~pos].mean()),
            "mean_set_size": float((inc1.astype(int) + inc0.astype(int)).mean()),
            "both_labels_fraction": float((inc1 & inc0).mean()),
            "n_val_events": int((y_va == 1).sum())}


MLP_GRID = [dict(hidden=h, lr=lr, quantile=q) for h in ((128, 64), (256, 256, 128))
            for lr in (1e-3, 3e-4) for q in (False, True)]


def run_study(seeds=(0, 1, 2, 3, 4)) -> dict:
    df = load()
    leak = leakage_report(df)
    assert all(leak.values()), leak
    device = pick_device()
    df["phys_life_days"] = physics_lifetime_days(df)
    tr = np.flatnonzero(df.split == "train")
    va = np.flatnonzero(df.split == "val")
    te = np.flatnonzero(df.split == "test")
    t, e = df.time_days.to_numpy(), df.event.to_numpy()
    Y = {h: ((t <= h) & (e == 1)).astype(int) for h in HORIZONS}
    res, probs, timings = {}, {}, {}
    # ---- physics
    risk_phys = -np.log10(np.clip(df.phys_life_days.to_numpy(), 1e-2, None))
    probs["physics"] = {h: _platt(risk_phys[va], Y[h][va])(risk_phys) for h in HORIZONS}
    res["physics"] = evaluate(df, te, probs["physics"], risk_phys)
    # ---- tabular
    Xp, names = design(df, df.split.eq("train").to_numpy(), True)
    Xn, _ = design(df, df.split.eq("train").to_numpy(), False)
    from sklearn.linear_model import LogisticRegression
    probs["logreg"] = {}
    for h in HORIZONS:
        m = LogisticRegression(C=1.0, max_iter=10000).fit(Xp[tr], Y[h][tr])
        probs["logreg"][h] = m.predict_proba(Xp)[:, 1]
    res["logreg"] = evaluate(df, te, probs["logreg"], probs["logreg"][68])
    import xgboost as xgb
    pr = {h: [] for h in HORIZONS}
    aft = []
    t0 = time.time()
    for s in seeds:
        for h in HORIZONS:
            m = xgb.XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, random_state=s,
                                  eval_metric="aucpr", early_stopping_rounds=100, n_jobs=16)
            m.fit(Xp[tr], Y[h][tr], eval_set=[(Xp[va], Y[h][va])], verbose=False)
            pr[h].append(m.predict_proba(Xp)[:, 1])
        dtr = xgb.DMatrix(Xp[tr], label_lower_bound=t[tr],
                          label_upper_bound=np.where(e[tr] == 1, t[tr], np.inf))
        dva = xgb.DMatrix(Xp[va], label_lower_bound=t[va],
                          label_upper_bound=np.where(e[va] == 1, t[va], np.inf))
        b = xgb.train({"objective": "survival:aft", "eval_metric": "aft-nloglik",
                       "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0,
                       "tree_method": "hist", "learning_rate": 0.03, "max_depth": 4,
                       "subsample": 0.8, "seed": s}, dtr, 3000, evals=[(dva, "val")],
                      early_stopping_rounds=100, verbose_eval=False)
        aft.append(np.log(b.predict(xgb.DMatrix(Xp))))
    timings["xgboost_s"] = time.time() - t0
    probs["xgboost"] = {h: np.mean(pr[h], 0) for h in HORIZONS}
    res["xgboost"] = evaluate(df, te, probs["xgboost"], probs["xgboost"][68])
    risk_aft = -np.mean(aft, 0)
    probs["xgboost_aft"] = {h: _platt(risk_aft[va], Y[h][va])(risk_aft) for h in HORIZONS}
    res["xgboost_aft"] = evaluate(df, te, probs["xgboost_aft"], risk_aft)
    # ---- neural hazard ensembles: configuration chosen on val AUC-PR (seed 0), then 5 seeds
    from sklearn.metrics import average_precision_score
    trm = df.split.eq("train").to_numpy()
    designs = {(phys, q): design(df, trm, phys, q)[0] for phys in (True, False)
               for q in (False, True)}
    selection = {}
    for name, phys in (("hazard_mlp", True), ("hazard_mlp_nophys", False)):
        scores = []
        for cfg in MLP_GRID:
            _n, sv, _i = train_hazard_mlp(designs[(phys, cfg["quantile"])], t, e, tr, va, 0,
                                          device, hidden=cfg["hidden"], lr=cfg["lr"])
            scores.append(average_precision_score(Y[68][va], 1 - surv_at(sv, 68)[va]))
        k = int(np.argmax(scores))
        selection[name] = {"grid": [{**c, "val_auc_pr_h68": sc} for c, sc in
                                    zip(MLP_GRID, scores)], "selected": MLP_GRID[k]}
    for name, phys in (("hazard_mlp", True), ("hazard_mlp_nophys", False)):
        cfg = selection[name]["selected"]
        X = designs[(phys, cfg["quantile"])]
        S, infos = [], []
        with Run(f"reentry_{name}", {"features": names if phys else
                                     [n for n in names if n != "phys_log_life"],
                                     "seeds": list(seeds), "bin_days": BIN_DAYS, **cfg}, 0,
                 device, group="reentry_landmark") as run:
            t0 = time.time()
            for s in seeds:
                net, surv, info = train_hazard_mlp(X, t, e, tr, va, s, device,
                                                   hidden=cfg["hidden"], lr=cfg["lr"])
                S.append(surv)
                infos.append(info)
                run.save_checkpoint(net, f"member_{s}")
            surv = np.mean(S, 0)
            probs[name] = {h: 1 - surv_at(surv, h) for h in HORIZONS}
            members68 = np.stack([1 - surv_at(x, 68) for x in S])
            probs[name + "_std68"] = members68.std(0)
            res[name] = evaluate(df, te, probs[name], probs[name][68])
            res[name]["params_per_member"] = count_parameters(net)
            res[name]["train_time_s"] = time.time() - t0
            run.finish({k: v for k, v in res[name].items() if np.isscalar(v)},
                       members=infos)
    # ---- comparisons, conformal, selective prediction
    comp = {}
    for a, b in (("hazard_mlp", "xgboost"), ("hazard_mlp", "physics"),
                 ("xgboost", "physics"), ("hazard_mlp", "hazard_mlp_nophys")):
        for h in HORIZONS:
            comp[f"{a}_vs_{b}/h{h}"] = cluster_bootstrap_ci(df, te, probs[a][h], probs[b][h], h)
    conf = {m: {"marginal": conformal_sets(probs[m][68][va], Y[68][va], probs[m][68][te],
                                           Y[68][te]),
                "mondrian": mondrian_sets(probs[m][68][va], Y[68][va], probs[m][68][te],
                                          Y[68][te])}
            for m in ("hazard_mlp", "xgboost", "xgboost_aft", "physics")}
    from ..uncertainty import risk_coverage
    p = probs["hazard_mlp"][68][te]
    correct = ((p >= 0.5).astype(int) == Y[68][te]).astype(float)
    rc_unc = risk_coverage(-probs["hazard_mlp_std68"][te], correct)
    rc_margin = risk_coverage(np.abs(p - 0.5), correct)
    out = {"tau": TAU, "follow_up": FOLLOW, "leakage_report": leak,
           "split_sizes": {k: int((df.split == k).sum()) for k in ("train", "val", "test")},
           "split_events": {k: int(df.event[df.split == k].sum()) for k in
                            ("train", "val", "test")},
           "results": res, "comparisons": comp, "conformal_sets_h68": conf,
           "mlp_selection": selection,
           "selective_prediction_h68": {"by_ensemble_std": {"aurc": rc_unc["aurc"]},
                                        "by_margin": {"aurc": rc_margin["aurc"]}},
           "timings": timings}
    d = output_dir("benchmarks", "reentry_landmark")
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    np.savez_compressed(os.path.join(d, "test_predictions.npz"), test_index=te,
                        **{f"{m}_h{h}": v[h][te] for m, v in probs.items()
                           if isinstance(v, dict) for h in HORIZONS})
    rows = [{"model": m, **{k: v for k, v in r.items() if np.isscalar(v)}}
            for m, r in res.items()]
    pd.DataFrame(rows).to_csv(os.path.join(d, "results.csv"), index=False)
    return out


def main(argv=None) -> int:
    out = run_study()
    cols = ["all/h68/auc_pr", "all/h30/auc_pr", "all/c_index", "all/h68/brier",
            "uncontrolled/h68/auc_pr", "uncontrolled/events"]
    print(pd.DataFrame({m: {c: r.get(c) for c in cols} for m, r in out["results"].items()})
          .T.round(4).to_string())
    print(json.dumps(out["comparisons"], indent=1))
    print(json.dumps(out["conformal_sets_h68"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
