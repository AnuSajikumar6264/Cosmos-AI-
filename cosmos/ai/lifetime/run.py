"""Experiment driver for catalogue-wide re-entry prediction (see docs/TARGET_DEFINITIONS.md).

    python -m cosmos.ai.lifetime.run --origin 2016 --population uncontrolled

Protocol "as of the origin date O" (no information after O is used for fitting or selection):

* train  : canonical-orbit epoch t0 < O - 7 yr; outcomes censored at O
* tune   : O - 7 yr <= t0 < O - 5 yr; outcomes censored at O (5-yr outcome known); split by
           launch family into tune-A (early stopping, model selection, gate fitting) and tune-B
           (conformal and probability calibration)
* test   : O <= t0 < test end (ORIGINS below: 2011-01-01 to 2016-01-01 at origin 2011, and
           2016-01-01 to the 2026-10-27 snapshot minus 5 yr at origin 2016, so that the 5-yr
           follow-up is complete for every test object); read only for scoring (scored more than
           once during development, see docs/REVIEW_AND_FIXES.md).

Target: z = log10(min(T, 5 yr) / day), floored at 0.5 day; horizon events T <= H for
H in {30 d, 0.5 yr, 1 yr, 5 yr}.

Learners: the learned-expert candidates are models published 2022-2026 (``models.RECENT``:
TabM, TabICLv2, Kumo-Tabular); the gradient-boosted trees and the plain quantile MLP are fitted
and scored as classic reference baselines but are not eligible as the gated expert.  The
expert is the recent model with the lowest tune-A pinball loss.

Learner predictions are cached in ``learner_cache/`` under a fingerprint of the split (object
keys and labels), the feature matrices and the settings; a cache whose fingerprint differs is
refitted, never re-aligned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from ..evaluation import brier_score, expected_calibration_error, harrell_c
from . import fusion as FU
from . import models as MO
from . import physics as PH
from .catalogue import SNAPSHOT, censor_at

CAP = PH.CAP_DAYS
HORIZONS = {"30d": 30.0, "6mo": 182.625, "1yr": 365.25, "5yr": CAP}
ORIGINS = {"2016": ("2009-01-01", "2011-01-01", "2016-01-01", "2021-10-27"),
           "2011": ("2004-01-01", "2006-01-01", "2011-01-01", "2016-01-01")}
POPULATIONS = {"uncontrolled": ("D", "R", "C"), "all": ("D", "R", "C", "P")}
FULL = ("orbit", "age", "design", "space_weather", "family", "physics", "class")
# feature-group ablations, refitted with the best learner's tuned settings.  "plus_rcs_2026" adds
# the radar cross-section of the 2026 catalogue snapshot, whose availability depends on the
# outcome: a labelled leak demonstration, never a candidate for selection
ABLATIONS = {f"no_{g}": tuple(x for x in FULL if x != g) for g in ("physics", "family", "space_weather", "design")}
ABLATIONS["plus_rcs_2026"] = FULL + ("rcs",)
SEEDS = (0, 1, 2, 3, 4)
CACHE_VERSION = 2


def load_cohort() -> pd.DataFrame:
    p = os.path.join(feature_dir("reentry", "processed"), f"lifetime_cohort_{SNAPSHOT}.csv.gz")
    return pd.read_csv(p, low_memory=False, dtype={"Satcat": str, "JCAT": str},
                       parse_dates=["t0", "t_end", "t_launch"])


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    known = (d["event"].eq(1) & (d["time_days"] <= CAP)) | (d["time_days"] >= CAP)
    d["known5"] = known
    y = np.minimum(d["time_days"], CAP)
    d["z"] = np.log10(np.clip(y, 0.5, None))
    for k, h in HORIZONS.items():
        lab = np.where(d["event"].eq(1) & (d["time_days"] <= h), 1.0, np.where(d["time_days"] >= h, 0.0, np.nan))
        d[f"y_{k}"] = lab
    return d


def split(df: pd.DataFrame, origin: str, population: str) -> dict:
    tr_end, va_end, o, te_end = ORIGINS[origin]
    # 16 objects carry uncertain or erroneous GCAT status codes ('R?', 'L?', 'ERR'): their outcome is
    # undefined (no end time); none lies in a test cohort, and they have no known label anywhere
    d = df[df["type1"].isin(POPULATIONS[population]) & df["time_days"].notna()]
    dev = censor_at(d[d["t0"] < va_end], o)
    train = add_targets(dev[dev["t0"] < tr_end])
    tune = add_targets(dev[dev["t0"] >= tr_end])
    test = add_targets(d[(d["t0"] >= o) & (d["t0"] < te_end)])
    fam = tune["launch_family"].astype(str).map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 2)
    return {"train": train, "tuneA": tune[fam.eq(0)], "tuneB": tune[fam.eq(1)], "test": test}


# ------------------------------------------------------------------------------ metrics
def regression_metrics(z, q, lo80=None, hi80=None, lo90=None, hi90=None) -> dict:
    med = q[:, 3]
    yd, pd_ = 10 ** z, 10 ** med
    ae = np.abs(yd - pd_)
    out = {"n": int(len(z)), "pinball": MO.pinball(z, q), "mae_log10": float(np.mean(np.abs(z - med))),
           "rmse_log10": float(np.sqrt(np.mean((z - med) ** 2))), "mae_days": float(ae.mean()),
           "medae_days": float(np.median(ae)), "p90ae_days": float(np.quantile(ae, 0.9)),
           "median_rel_err": float(np.median(ae / np.maximum(yd, 1.0)))}
    dec = z < MO.Z_CAP - 1e-9                                   # decayed within the 5-yr horizon
    if dec.any():
        out.update({"n_decayed": int(dec.sum()), "mae_days_decayed": float(ae[dec].mean()),
                    "medae_days_decayed": float(np.median(ae[dec])), "p90ae_days_decayed": float(np.quantile(ae[dec], 0.9)),
                    "median_rel_err_decayed": float(np.median(ae[dec] / np.maximum(yd[dec], 1.0))),
                    "mae_log10_decayed": float(np.mean(np.abs(z[dec] - med[dec])))})
    for lev, lo, hi in ((80, lo80, hi80), (90, lo90, hi90)):
        if lo is None:
            continue
        a = 1 - lev / 100
        cov = (z >= lo) & (z <= hi)
        w = hi - lo
        out[f"cov{lev}"] = float(cov.mean())
        out[f"width{lev}_log10"] = float(np.mean(w))
        out[f"winkler{lev}"] = float(np.mean(w + 2 / a * (lo - z) * (z < lo) + 2 / a * (z - hi) * (z > hi)))
    return out


def horizon_metrics(y, p, p_raw=None) -> dict:
    """AUC-PR and AUC-ROC on the raw score ``p_raw`` (the isotonic map is monotone but ties
    scores, which changes rank metrics); Brier score and ECE on the calibrated ``p``."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    p_raw = p if p_raw is None else p_raw
    ok = ~np.isnan(y)
    y, p, p_raw = y[ok], p[ok], p_raw[ok]
    if y.min() == y.max():
        return {"n": int(len(y)), "positives": int(y.sum())}
    return {"n": int(len(y)), "positives": int(y.sum()), "auc_pr": float(average_precision_score(y, p_raw)),
            "auc_roc": float(roc_auc_score(y, p_raw)), "brier": brier_score(y, p),
            "ece": expected_calibration_error(y, p)}


def cluster_bootstrap(fn, groups, n_boot=2000, seed=20261027):
    """Percentile CI of ``fn(idx)`` resampling launch families with replacement."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(keys))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(fn(np.concatenate([members[k] for k in pick])))
    vals = np.asarray(vals)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), vals


# ------------------------------------------------------------------------------ driver
LEARNER_SET = ("xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo")


def run(origin="2016", population="uncontrolled", n_samples=32, learners=LEARNER_SET,
        seeds=SEEDS, n_trials=MO.N_TRIALS, ablations=True, out=None):
    t_start = time.time()
    out = out or output_dir("benchmarks", "reentry_lifetime", f"origin{origin}_{population}")
    os.makedirs(out, exist_ok=True)
    S = split(load_cohort(), origin, population)
    log = {"origin": origin, "population": population, "snapshot": SNAPSHOT, "dates": ORIGINS[origin],
           "sizes": {k: int(len(v)) for k, v in S.items()},
           "known5": {k: int(v["known5"].sum()) for k, v in S.items()},
           "events_5yr": {k: int(np.nansum(v["y_5yr"])) for k, v in S.items()}}
    print(json.dumps(log), flush=True)

    # ---- physics expert (Monte Carlo predictive distribution)
    prior = PH.class_prior_from(S["train"])
    phys = {}
    for k, d in S.items():
        r = PH.predict(d, prior, n_samples=n_samples, seed=0)
        phys[k] = {"feat": MO.physics_features(r["samples_days"]), "q": MO._monotone(MO.physics_quantiles(r["samples_days"])),
                   "src": r["am_source"], "seconds": r["seconds"]}
    log["physics_seconds"] = {k: v["seconds"] for k, v in phys.items()}
    log["prior_pool_sizes"] = {f"{kind}:{t}": int(len(v)) for kind in ("lc", "am") for t, v in prior[kind].items()}
    for k in ("train", "tuneB", "test"):
        log[f"am_source_{k}"] = pd.Series(phys[k]["src"]).value_counts().to_dict()

    def matrices(groups):
        Xs, shapes = {}, None
        for k in ("train", "tuneA", "tuneB", "test"):
            X, shapes = MO.design_matrix(S[k], phys[k]["feat"], groups, shapes)
            Xs[k] = X
        cols = list(Xs["train"].columns)
        return {k: MO.align_columns(v, cols) for k, v in Xs.items()}

    def lab(k):
        d = S[k].reset_index(drop=True)
        m = d["known5"].to_numpy()
        return m, d["z"].to_numpy()

    mtr, ztr = lab("train")
    mA, zA = lab("tuneA")
    mB, zB = lab("tuneB")
    mT, zT = lab("test")
    preds, fits = {}, {}

    cache_dir = os.path.join(out, "learner_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def fingerprint(name, groups, seeds_, fixed_params, X):
        h = hashlib.md5(json.dumps([CACHE_VERSION, name, list(groups), list(seeds_), n_trials, fixed_params],
                                   sort_keys=True, default=float).encode())
        for k in ("train", "tuneA", "tuneB", "test"):
            h.update(",".join(S[k]["JCAT"].astype(str)).encode())
            h.update(",".join(X[k].columns).encode())
            h.update(np.ascontiguousarray(X[k].to_numpy(float)).tobytes())
        for a in (ztr, mtr, zA, mA):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def fit_learner(name, groups, tag, seeds_, fixed_params=None):
        """Tune (or use ``fixed_params``) and fit over seeds; results are cached per tag under a
        fingerprint of data, features and settings, so that an interrupted run resumes without
        refitting finished learners and a stale cache is never used."""
        cpath = os.path.join(cache_dir, f"{tag}.npz")
        X = matrices(groups)
        fp = fingerprint(name, groups, seeds_, fixed_params, X)
        if os.path.isfile(cpath):
            z_ = np.load(cpath, allow_pickle=True)
            if "fingerprint" in z_.files and str(z_["fingerprint"]) == fp:
                per_seed = [{k: z_[f"s{s}_{k}"] for k in ("tuneA", "tuneB", "test")} for s in seeds_]
                ens = {k: np.sort(np.mean([p[k] for p in per_seed], 0), 1) for k in ("tuneA", "tuneB", "test")}
                preds[tag] = {"ensemble": ens, "seeds": per_seed}
                fits[tag] = json.loads(str(z_["fit_json"]))
                print(f"  {tag}: loaded from cache (tuneA pinball {MO.pinball(zA[mA], ens['tuneA'][mA]):.4f})", flush=True)
                return
            print(f"  {tag}: cache fingerprint differs; refitting", flush=True)
        if fixed_params is None:
            model, params, trials = MO.tune_and_fit(name, X["train"][mtr], ztr[mtr], X["tuneA"][mA], zA[mA],
                                                    seed=0, n_trials=n_trials)
        else:
            params, trials = fixed_params, []
            model = MO.LEARNERS[name](seed=0).fit(X["train"][mtr], ztr[mtr], X["tuneA"][mA], zA[mA], params)
        per_seed = []
        t0 = time.time()
        for s in seeds_:
            m = model if s == 0 else MO.LEARNERS[name](seed=s).fit(X["train"][mtr], ztr[mtr], X["tuneA"][mA], zA[mA], params)
            per_seed.append({k: m.predict(X[k]) for k in ("tuneA", "tuneB", "test")})
        fit_s = (time.time() - t0) / len(seeds_)
        t1 = time.time()
        _ = model.predict(X["test"])
        lat = (time.time() - t1) / max(len(X["test"]), 1) * 1000
        ens = {k: np.sort(np.mean([p[k] for p in per_seed], 0), 1) for k in ("tuneA", "tuneB", "test")}
        preds[tag] = {"ensemble": ens, "seeds": per_seed}
        fits[tag] = {"params": params, "trials": trials, "fit_seconds_per_seed": fit_s,
                     "predict_ms_per_1000_objects": lat * 1000, "n_features": X["train"].shape[1],
                     "tuned": fixed_params is None}
        np.savez_compressed(cpath, fit_json=json.dumps(fits[tag], default=float), fingerprint=fp,
                            **{f"s{s}_{k}": p[k] for s, p in zip(seeds_, per_seed) for k in ("tuneA", "tuneB", "test")})
        print(f"  {tag}: tuneA pinball {MO.pinball(zA[mA], ens['tuneA'][mA]):.4f} (params {params})", flush=True)

    for name in learners:
        fit_learner(name, FULL, name, seeds)
    pool = [n for n in learners if n in MO.RECENT] or list(learners)
    best = min(pool, key=lambda n: MO.pinball(zA[mA], preds[n]["ensemble"]["tuneA"][mA]))
    log["best_learner_on_tuneA"] = best
    log["selection_pool"] = pool
    log["tuneA_pinball"] = {n: MO.pinball(zA[mA], preds[n]["ensemble"]["tuneA"][mA]) for n in learners}

    # ---- out-of-fold predictions of the best learner on the training period (5 folds grouped by
    # launch family), so that the gate is fitted on the full diversity of the training data
    Xf = matrices(FULL)
    fam_tr = S["train"].reset_index(drop=True)["launch_family"].astype(str)
    fold = fam_tr.map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 5).to_numpy()
    oof = np.full((len(fam_tr), len(MO.QUANTILES)), np.nan)
    for k in range(5):
        trn, hold = (fold != k) & mtr, fold == k
        mk = MO.LEARNERS[best](seed=0).fit(Xf["train"][trn], ztr[trn], Xf["tuneA"][mA], zA[mA], fits[best]["params"])
        oof[hold] = mk.predict(Xf["train"][hold])
    print(f"  OOF ({best}) train pinball {MO.pinball(ztr[mtr], oof[mtr]):.4f}", flush=True)
    if ablations:                     # feature-group ablations with the best learner's tuned settings
        for ab, groups in ABLATIONS.items():
            fit_learner(best, groups, f"{best}_{ab}", seeds, fixed_params=fits[best]["params"])

    # ---- fusion candidates (gate fitted on tune-A, calibration on tune-B)
    R = {k: FU.reliability_features(S[k], phys[k]["feat"], phys[k]["src"]) for k in ("train", "tuneA", "tuneB", "test")}
    cand = {"physics_only": {k: phys[k]["q"] for k in ("tuneA", "tuneB", "test")}}
    for tag in preds:
        cand[tag] = preds[tag]["ensemble"]
    gate_info = {}
    for mode, label, where in (("gated", "cosmos_gated", "train_oof"), ("constant", "stack_constant", "train_oof"),
                               ("gated", "cosmos_gated_tuneA", "tuneA")):
        if where == "train_oof":
            gt = FU.Gate(mode=mode).fit(R["train"][mtr], phys["train"]["q"][mtr], oof[mtr], ztr[mtr])
        else:
            gt = FU.Gate(mode=mode).fit(R["tuneA"][mA], phys["tuneA"]["q"][mA], preds[best]["ensemble"]["tuneA"][mA], zA[mA])
        cand[label] = {k: MO._monotone(gt.fuse(R[k], phys[k]["q"], preds[best]["ensemble"][k])) for k in ("tuneA", "tuneB", "test")}
        gate_info[label] = {"theta": gt.theta.tolist(), "fitted_on": where,
                            "tuneA_pinball": MO.pinball(zA[mA], cand[label]["tuneA"][mA]), "test_weight_mean": float(gt.weight(R["test"]).mean()),
                            "test_weight_by_am_source": pd.Series(gt.weight(R["test"])).groupby(phys["test"]["src"]).mean().to_dict()}
    cand["equal_average"] = {k: MO._monotone(0.5 * phys[k]["q"] + 0.5 * preds[best]["ensemble"][k]) for k in ("tuneA", "tuneB", "test")}
    log["gates"] = gate_info

    # ---- calibration and evaluation
    G = {k: FU.mondrian_groups(S[k]) for k in ("tuneB", "test")}
    dT = S["test"].reset_index(drop=True)
    rows, horizon_rows, keep = [], [], {}
    from sklearn.isotonic import IsotonicRegression
    for name, q in cand.items():
        for mond in (True, False):
            if not mond and name not in (best, "cosmos_gated"):
                continue
            cq = FU.MondrianCQR(mondrian=mond).fit(q["tuneB"][mB], zB[mB], G["tuneB"][mB].reset_index(drop=True))
            lo80, hi80 = cq.interval(q["test"], G["test"], 0.8)
            lo90, hi90 = cq.interval(q["test"], G["test"], 0.9)
            tag = name if mond else f"{name}_marginal_conformal"
            r = regression_metrics(zT[mT], q["test"][mT], lo80[mT], hi80[mT], lo90[mT], hi90[mT])
            raw = regression_metrics(zT[mT], q["test"][mT], q["test"][mT, 1], q["test"][mT, 5], q["test"][mT, 0], q["test"][mT, 6])
            r.update({"raw_cov80": raw["cov80"], "raw_cov90": raw["cov90"],
                      "c_index": harrell_c(dT["time_days"].to_numpy(), dT["event"].to_numpy(), -q["test"][:, 3])})
            rows.append({"model": tag, **r})
            if mond:
                keep[name] = {"q": q["test"], "lo90": lo90, "hi90": hi90, "lo80": lo80, "hi80": hi80, "q_tuneB": q["tuneB"]}
                for hk, h in HORIZONS.items():
                    pB = FU.horizon_prob(q["tuneB"], h)
                    yB = S["tuneB"].reset_index(drop=True)[f"y_{hk}"].to_numpy()
                    okB = ~np.isnan(yB)
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pB[okB], yB[okB])
                    pT_raw = FU.horizon_prob(q["test"], h)
                    horizon_rows.append({"model": name, "horizon": hk,
                                         **horizon_metrics(dT[f"y_{hk}"].to_numpy(), iso.predict(pT_raw), pT_raw)})
    res = pd.DataFrame(rows).sort_values("pinball")
    hres = pd.DataFrame(horizon_rows)

    # ---- per-seed variability of the learned experts
    seed_rows = []
    for tag, p in preds.items():
        for s, ps in zip(seeds, p["seeds"]):
            seed_rows.append({"model": tag, "seed": s, "pinball": MO.pinball(zT[mT], ps["test"][mT]),
                              "mae_log10": float(np.mean(np.abs(zT[mT] - ps["test"][mT, 3])))})
    seeds_df = pd.DataFrame(seed_rows)

    # ---- paired cluster bootstrap: cosmos_gated vs each alternative (test, known 5-yr labels)
    fam = dT["launch_family"].astype(str).to_numpy()[mT]
    comps = {}
    zt = zT[mT]

    def loss_vec(q):
        d = zt[:, None] - q[mT]
        return np.mean(np.maximum(MO.QUANTILES * d, (MO.QUANTILES - 1) * d), 1)
    base = loss_vec(keep["cosmos_gated"]["q"])
    for name in keep:
        if name == "cosmos_gated":
            continue
        other = loss_vec(keep[name]["q"])
        diff = other - base                                     # > 0: cosmos_gated better
        lo, hi, _ = cluster_bootstrap(lambda idx: float(diff[idx].mean()), fam)
        comps[name] = {"pinball_other_minus_gated": float(diff.mean()), "ci95": [lo, hi]}
    ci = {}
    for name in ("cosmos_gated", best, "physics_only"):
        e = np.abs(zt - keep[name]["q"][mT, 3])
        lo, hi, _ = cluster_bootstrap(lambda idx: float(e[idx].mean()), fam)
        ci[name] = {"mae_log10": float(e.mean()), "ci95": [lo, hi]}

    # ---- subgroup results for the main models
    sub_rows = []
    srcT = pd.Series(phys["test"]["src"])[mT].to_numpy()
    typT = dT["type1"].to_numpy()[mT]
    for name in ("physics_only", best, "cosmos_gated"):
        qq = keep[name]
        for key, sel in [(f"type={t}", typT == t) for t in np.unique(typT)] + [(f"am_source={s}", srcT == s) for s in np.unique(srcT)]:
            if sel.sum() < 30:
                continue
            idx = np.flatnonzero(mT)[sel]
            r = regression_metrics(zT[idx], qq["q"][idx], qq["lo80"][idx], qq["hi80"][idx], qq["lo90"][idx], qq["hi90"][idx])
            sub_rows.append({"model": name, "subgroup": key, **r})
    sub = pd.DataFrame(sub_rows)

    log.update({"fits": {k: {kk: vv for kk, vv in v.items() if kk != "trials"} for k, v in fits.items()},
                "paired_vs_cosmos_gated": comps, "mae_log10_ci": ci, "seconds_total": time.time() - t_start})
    res.to_csv(os.path.join(out, "regression_test.csv"), index=False)
    hres.to_csv(os.path.join(out, "horizon_test.csv"), index=False)
    seeds_df.to_csv(os.path.join(out, "seeds_test.csv"), index=False)
    sub.to_csv(os.path.join(out, "subgroups_test.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
    with open(os.path.join(out, "tuning_trials.json"), "w") as f:
        json.dump({k: v["trials"] for k, v in fits.items()}, f, indent=1, default=float)
    np.savez_compressed(os.path.join(out, "test_predictions.npz"),
                        **{f"{k}__{kk}": vv for k, v in keep.items() for kk, vv in v.items() if kk != "q_tuneB"},
                        z=zT, known5=mT, time_days=dT["time_days"].to_numpy(), event=dT["event"].to_numpy())
    dB = S["tuneB"].reset_index(drop=True)            # calibration split (used by the C2 survival calibration)
    np.savez_compressed(os.path.join(out, "tuneB_predictions.npz"), **{f"{k}__q": v["q_tuneB"] for k, v in keep.items()},
                        z=zB, known5=mB, time_days=dB["time_days"].to_numpy(), event=dB["event"].to_numpy(),
                        type1=dB["type1"].astype(str).to_numpy())
    print(res[["model", "pinball", "mae_log10", "medae_days", "cov80", "cov90", "c_index"]].to_string(index=False), flush=True)
    return res, hres, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2016", choices=list(ORIGINS))
    ap.add_argument("--population", default="uncontrolled", choices=list(POPULATIONS))
    ap.add_argument("--learners", default=",".join(LEARNER_SET))
    ap.add_argument("--trials", type=int, default=MO.N_TRIALS)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--samples", type=int, default=32)
    a = ap.parse_args(argv)
    run(a.origin, a.population, n_samples=a.samples, learners=tuple(a.learners.split(",")),
        seeds=tuple(range(a.seeds)), n_trials=a.trials, ablations=not a.no_ablations)


if __name__ == "__main__":
    main()
