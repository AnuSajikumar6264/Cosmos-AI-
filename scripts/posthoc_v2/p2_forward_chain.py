"""P2: forward-chaining (pseudo-prospective) replication of H6 inside the training half.

    python scripts/posthoc_v2/p2_forward_chain.py --mission 1 --fold 0      # one fold per process
    python scripts/posthoc_v2/p2_forward_chain.py --mission 1 --summary     # pool the folds

For each fold only data before the fold start are used. Two protocols mirror A0 and A1:

* W (window):  AR(8) fitted on the history minus its last 12 (M1) / 6 (M2) months, out-of-sample
  residual scale on the next 3 / 1.5 months, pool and cross-fitted selection on the last 9 / 4.5
  months, exploratory grid (the rule of esa_benchmark / alarms.select_calibrator).
* H (history): out-of-fold AR(8) scores over the whole history (K = 5, 1-day purge), final model
  on all nominal history rows with in-sample residual scale, selection by the frozen cross-fit
  over the two halves of the history (oof_benchmark._score_task), pre-registered grid.

The AR(8) primitives are those of the frozen ``oof`` module. Normalisation statistics are
recomputed from the nominal samples of W's fitting part of the history, for both protocols.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import provenance, save_json, OUT  # noqa: E402
import exact_metric as XM  # noqa: E402

from cosmos.ai.telemetry import oof, oof_benchmark as OB  # noqa: E402
from cosmos.ai.telemetry.alarms import Calibrator, GRIDS, K_GRID, select_calibrator  # noqa: E402
from cosmos.ai.telemetry.esa_eval import false_alarm_stats, period_labels, score_alarms  # noqa: E402
from cosmos.ai.telemetry.forecasting import DELTA, MissionData, origins  # noqa: E402

FOLDS = {1: [("2003-01-01", "2004-01-01"), ("2004-01-01", "2005-01-01"),
             ("2005-01-01", "2006-01-01"), ("2006-01-01", "2007-01-01")],
         2: [("2001-01-01", "2001-04-01"), ("2001-04-01", "2001-07-01"), ("2001-07-01", "2001-10-01")]}
#: (months before the fold start) where W's es and cal periods begin
W_OFFSETS = {1: {"es": 12, "cal": 9}, 2: {"es": 6, "cal": 4.5}}
LAYERS = ("conformal_bh", "threshold")


def _month_shift(ts, months):
    days = int(round(months * 30.4375))
    return pd.Timestamp(ts) - pd.Timedelta(days=days)


def normalise(X, L, fit_rows):
    med = np.zeros(X.shape[1])
    scale = np.ones(X.shape[1])
    for k in range(X.shape[1]):
        v = X[fit_rows, k][L[fit_rows, k] == 0]
        if v.size == 0:
            v = X[fit_rows, k]
        med[k] = np.median(v)
        q1, q3 = np.percentile(v, [25, 75])
        s = (q3 - q1) / 1.349
        if s <= 1e-12:
            q01, q99 = np.percentile(v, [1, 99])
            s = (q99 - q01) / 4.653
        if s <= 1e-12:
            s = max(np.std(v), 1e-6 * (abs(med[k]) + 1.0), 1e-9)
        scale[k] = s
    return ((X - med) / scale).astype(np.float64)


def w_scores(Z, L, fit, es, upto):
    """Protocol W: scores on rows [0, upto) from a model fitted on ``fit`` rows, sigma from ``es`` rows."""
    rng = np.random.default_rng(0)
    n, C = upto, Z.shape[1]
    o = origins(n, 1, oof.HORIZON)
    S = np.full((n, C), np.nan, np.float32)
    for c in range(C):
        z = Z[:n, c]
        usable = np.zeros(n, bool)
        usable[fit[0]:fit[1]] = L[fit[0]:fit[1], c] == 0
        w = oof._ar_fit(z, usable, rng)
        r = z - oof._ar_predict(z, o, w)
        es_ok = np.zeros(n, bool)
        es_ok[es[0]:es[1]] = L[es[0]:es[1], c] == 0
        sig = oof._sigma(r, es_ok)
        S[:, c] = (np.abs(r) / max(sig, DELTA)).astype(np.float32)
    return S


def h_scores(Z, L, hist_end, upto, rule_s, K=5, purge_days=1.0):
    """Protocol H: OOF scores on [0, hist_end) and full-fit scores on [hist_end, upto)."""
    rng = np.random.default_rng(0)
    n, C = hist_end, Z.shape[1]
    folds = oof.fold_bounds(n, K)
    purge = int(purge_days * 86400 / rule_s)
    o_all = origins(upto, 1, oof.HORIZON)
    S_hist = np.full((n, C), np.nan, np.float32)
    S_eval = np.full((upto - n, C), np.nan, np.float32)
    for c in range(C):
        z_all = Z[:upto, c]
        z = z_all[:n]
        nominal = L[:n, c] == 0
        o_tr = o_all[o_all < n]
        for a, e in folds:
            usable = nominal.copy()
            usable[max(0, a - purge):min(n, e + purge)] = False
            w = oof._ar_fit(z, usable, rng)
            r = z - oof._ar_predict(z, o_tr, w)
            sig = oof._sigma(r, usable)
            S_hist[a:e, c] = (np.abs(r[a:e]) / max(sig, DELTA)).astype(np.float32)
        w = oof._ar_fit(z, nominal, rng)
        sig_full = oof._sigma(z - oof._ar_predict(z, o_tr, w), nominal)
        r_all = z_all - oof._ar_predict(z_all, o_all, w)
        S_eval[:, c] = (np.abs(r_all[n:]) / max(sig_full, DELTA)).astype(np.float32)
    return S_hist, S_eval


def run_fold(m, i):
    prov = provenance(__file__)
    t0 = time.time()
    D = MissionData(m, "target")
    rule = D.train.rule_s
    merge = int(OB.MERGE_H * 3600 / rule)
    ts = D.train.timestamps()
    s_ts, e_ts = FOLDS[m][i]
    f_start, f_end = D.train.index_of(s_ts), D.train.index_of(e_ts)
    es0 = D.train.index_of(str(_month_shift(s_ts, W_OFFSETS[m]["es"]).date()))
    cal0 = D.train.index_of(str(_month_shift(s_ts, W_OFFSETS[m]["cal"]).date()))
    X = np.asarray(D.train.X[:f_end][:, D.jtr], np.float64)
    L = np.asarray(D.Ltr[:f_end])
    Z = normalise(X, L, np.arange(0, es0))
    lab_all = period_labels(m, ts[0], ts[f_end - 1])
    lab_all = lab_all[lab_all.Channel.isin(D.channels)]
    lab_eval = lab_all[(lab_all.StartTime >= ts[f_start]) & (lab_all.EndTime <= ts[f_end - 1])]
    t_eval = ts[f_start:f_end]
    out = {**prov, "mission": m, "fold": i, "fold_start": s_ts, "fold_end": e_ts,
           "rows": {"es0": int(es0), "cal0": int(cal0), "start": int(f_start), "end": int(f_end)},
           "n_events_eval": int(lab_eval[lab_eval.Category.isin(["Anomaly", "Rare Event"])].ID.nunique()),
           "protocols": {}}
    # --- W
    SW = w_scores(Z, L, (0, es0), (es0, cal0), f_end)
    ts_cal = ts[cal0:f_start]
    lab_cal = lab_all[(lab_all.StartTime >= ts_cal[0]) & (lab_all.EndTime <= ts_cal[-1])]

    def score_fn(g, ch, rows):
        t = ts_cal[rows]
        lab = lab_cal[(lab_cal.StartTime >= t[0]) & (lab_cal.EndTime <= t[-1])]
        s = score_alarms(m, t, ch, D.channels, lab, per_channel=False, categories=("anomaly_and_rare",))
        return s.get("anomaly_and_rare/EW_F_0.50", 0.0)

    # --- H
    SH_hist, SH_eval = h_scores(Z, L, f_start, f_end, rule)
    lab_hist = lab_all[lab_all.EndTime <= ts[f_start - 1]]
    for layer in LAYERS:
        cal, _ = select_calibrator(layer, SW[cal0:f_start], L[cal0:f_start], score_fn, rule, merge,
                                   levels=GRIDS[layer], ks=K_GRID)
        gW, chW = cal.alarms(SW[f_start:f_end])
        OB.KS[:] = list(OB.KS)
        OB._G.update(S=SH_hist, L=L[:f_start], halves=[np.arange(0, f_start // 2), np.arange(f_start // 2, f_start)],
                     ts=ts[:f_start], lab=lab_hist, mission=m, channels=D.channels, rule=rule, merge=merge)
        tasks = [(layer, lev, h) for lev in OB.LEVELS[layer] for h in (0, 1)]
        with ProcessPoolExecutor(min(16, len(tasks))) as ex:
            rows = [r for part in ex.map(OB._score_task, tasks) for r in part]
        g = (pd.DataFrame(rows).groupby(["kind", "level", "k"], as_index=False)
             .agg(cv=("EW_F0.5", "mean"), alarm_fraction=("alarm_fraction", "mean")))
        b = g.sort_values(["cv", "alarm_fraction"], ascending=[False, True]).iloc[0]
        calH = Calibrator(layer, float(b.level), int(b.k), merge, rule_s=rule).fit(SH_hist, L[:f_start])
        gH, chH = calH.alarms(SH_eval)
        for name, gg, cc, c in (("W", gW, chW, cal), ("H", gH, chH, calH)):
            sc = score_alarms(m, t_eval, cc, D.channels, lab_eval, per_channel=False)
            fa = false_alarm_stats(t_eval, gg, lab_eval)
            tab = XM.event_table(gg, t_eval, lab_eval)
            out["protocols"][f"{name}_{layer}"] = {
                "level": float(c.level), "k": int(c.k), "EW_F0.5": sc.get("anomaly_and_rare/EW_F_0.50", 0.0),
                "anomaly_EW_F0.5": sc.get("anomaly/EW_F_0.50", 0.0),
                "EW_recall": sc.get("anomaly_and_rare/EW_recall", 0.0), "exact_EW_F0.5": XM.f_beta(tab), **fa,
                "table": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in tab.items()}}
            print(f"M{m} fold {i} {name} {layer}: level {c.level:g} k {c.k}: F0.5 "
                  f"{out['protocols'][f'{name}_{layer}']['EW_F0.5']:.4f} FA/yr {fa['false_alarms_per_year']:.1f}",
                  flush=True)
    out["seconds"] = time.time() - t0
    save_json(out, "p2_forward", f"m{m}_fold{i}.json")


def summary(m):
    prov = provenance(__file__)
    folds = [json.load(open(os.path.join(OUT, "p2_forward", f"m{m}_fold{i}.json"))) for i in range(len(FOLDS[m]))]
    res = {**prov, "mission": m, "layers": {}}
    rng = np.random.default_rng(20261024)
    for layer in LAYERS:
        per = []
        tabsH, tabsW = [], []
        for f in folds:
            h, w = f["protocols"][f"H_{layer}"], f["protocols"][f"W_{layer}"]
            per.append({"fold": f["fold"], "start": f["fold_start"], "n_events": f["n_events_eval"],
                        "H": h["EW_F0.5"], "W": w["EW_F0.5"], "diff": h["EW_F0.5"] - w["EW_F0.5"],
                        "H_fa_yr": h["false_alarms_per_year"], "W_fa_yr": w["false_alarms_per_year"],
                        "H_recall": h["EW_recall"], "W_recall": w["EW_recall"],
                        "H_level_k": [h["level"], h["k"]], "W_level_k": [w["level"], w["k"]]})
            tabsH.append({k: np.asarray(v) if isinstance(v, list) else v for k, v in h["table"].items()})
            tabsW.append({k: np.asarray(v) if isinstance(v, list) else v for k, v in w["table"].items()})

        def pool(ts_):
            keys = ("nominal", "alarm_nominal", "ev_count", "tp_count", "fp_count")
            p = {k: np.concatenate([t[k] for t in ts_]) for k in keys}
            p["nb"] = int(sum(t["nb"] for t in ts_))
            return p
        r = XM.compare([pool(tabsH)], [pool(tabsW)], rng)
        r.pop("replicates")
        diffs = [p["diff"] for p in per]
        res["layers"][layer] = {"folds": per, "mean_diff": float(np.mean(diffs)),
                                "n_folds_H_better": int(sum(d > 0 for d in diffs)), "pooled": r}
        need = 3 if m == 1 else 2
        res["layers"][layer]["replicates_H6"] = bool(np.mean(diffs) >= 0.05 and res["layers"][layer]["n_folds_H_better"] >= need)
        print(f"M{m} {layer}: mean H-W {np.mean(diffs):+.4f}; H better in {res['layers'][layer]['n_folds_H_better']}"
              f"/{len(per)}; pooled {r['diff']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
    save_json(res, "p2_forward", f"m{m}_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary(a.mission)
    else:
        run_fold(a.mission, a.fold)


if __name__ == "__main__":
    main()
