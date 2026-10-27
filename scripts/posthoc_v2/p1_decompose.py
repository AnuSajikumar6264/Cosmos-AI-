"""P1: one-factor-at-a-time decomposition of the H6 contrast (A1 - A0).

    python scripts/posthoc_v2/p1_decompose.py --mission 1 --channels target --layer conformal_bh

Factors (POSTHOC_PLAN_V2.md, P1): G grid, F AR(8) fit data, sigma residual scale, P calibration
pool, D selection data. Score levels: S0 = A0's scores; Sh = A1's AR(8) residuals divided by A0's
residual scale; S1 = A1's scores. Forward path c0..c5, reverse path c5, r1..r4, c0.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, provenance, save_json, out_path  # noqa: E402
import exact_metric as XM  # noqa: E402

from cosmos.ai.telemetry import oof, oof_benchmark as OB  # noqa: E402
from cosmos.ai.telemetry.alarms import Calibrator, GRIDS, K_GRID, select_calibrator  # noqa: E402
from cosmos.ai.telemetry.esa_eval import false_alarm_stats, period_labels, score_alarms  # noqa: E402
from cosmos.ai.telemetry.forecasting import (DELTA, ForecastConfig, MissionData, forecast_record,  # noqa: E402
                                             nonconformity, residual_scale)
from cosmos.ai.data.preprocess_telemetry import official_test_labels  # noqa: E402

G0 = {"levels": {k: list(v) for k, v in GRIDS.items()}, "ks": list(K_GRID)}
G1 = {"levels": {k: list(v) for k, v in OB.LEVELS.items()}, "ks": list(OB.KS)}


def a0_dir(m, ch):
    return sorted(glob.glob(os.path.join(ROOT, "outputs", "experiments", f"*_esa_m{m}_{ch}_ar8_s0_s0")))[-1]


def score_levels(D, m, ch):
    """S0 (A0), Sh (A1 residuals / A0 sigma) and S1 (A1), as (S_train_or_cal, S_test) pairs."""
    d = a0_dir(m, ch)
    S0_cal = np.load(os.path.join(d, "scores_cal.npy"))
    S0_te = np.load(os.path.join(d, "scores_test.npy"), mmap_mode="r")
    cfg = ForecastConfig(mission=m, encoder="ar8", channels=ch)
    sig0 = residual_scale(D, cfg, None)
    # check: A0's calibration-window scores are reproduced from the recomputed sigma
    Y, q25, q50, q75, _ = forecast_record(None, D, "cal", cfg)
    rec = nonconformity(Y, q25, q50, q75, sig0)
    ok = np.isfinite(rec) & np.isfinite(S0_cal)
    s0_check = float(np.nanmax(np.abs(rec[ok] - S0_cal[ok])))
    S1_tr, S1_te, meta = oof.load(m, ch, "ar8")
    sig = np.load(os.path.join(oof.oof_dir(m, ch, "ar8"), "sigma.npy"))
    sig_fold, sig_full = sig[:-1], sig[-1]
    S1_tr = np.asarray(S1_tr, np.float32)
    Sh_tr = np.empty_like(S1_tr)
    for b, (a, e) in enumerate(meta["folds"]):
        Sh_tr[a:e] = S1_tr[a:e] * (np.maximum(sig_fold[b], DELTA) / np.maximum(sig0, DELTA))[None, :]
    S1_te = np.asarray(S1_te, np.float32)
    Sh_te = (S1_te * (np.maximum(sig_full, DELTA) / np.maximum(sig0, DELTA))[None, :]).astype(np.float32)
    return {"S0": (S0_cal, np.asarray(S0_te, np.float32)), "Sh": (Sh_tr, Sh_te), "S1": (S1_tr, S1_te)}, \
        {"sigma0": sig0.tolist(), "sigma1_full": sig_full.tolist(), "s0_recompute_max_abs_diff": s0_check}


# ---------------------------------------------------------------------------- selection procedures
def sel_cal(D, m, layer, S_calrows, grid, merge):
    """P_cal, D_cal: the exploratory rule (alarms.select_calibrator) on the calibration window."""
    a, b = D.p["cal"]
    ts_cal = D.train.timestamps()[a:b]
    lab_cal = period_labels(m, ts_cal[0], ts_cal[-1])
    lab_cal = lab_cal[lab_cal.Channel.isin(D.channels)]
    L_cal = D.Ltr[a:b]

    def score_fn(g, ch, rows):
        t = ts_cal[rows]
        lab = lab_cal[(lab_cal.StartTime >= t[0]) & (lab_cal.EndTime <= t[-1])]
        s = score_alarms(m, t, ch, D.channels, lab, per_channel=False, categories=("anomaly_and_rare",))
        return s.get("anomaly_and_rare/EW_F_0.50", 0.0)

    cal, rows = select_calibrator(layer, np.asarray(S_calrows, np.float32), L_cal, score_fn, D.train.rule_s,
                                  merge, levels=grid["levels"][layer], ks=grid["ks"])
    return cal, [{k: v for k, v in r.items() if k != "half_scores"} for r in rows]


def sel_half(D, m, layer, S_tr, grid, merge, workers=24):
    """P_half, D_half: the frozen oof_benchmark rule (its own _score_task), on the given grid."""
    n = S_tr.shape[0]
    ts = D.train.timestamps()
    lab = period_labels(m, ts[0], ts[-1])
    lab = lab[lab.Channel.isin(D.channels)]
    OB.KS[:] = grid["ks"]                       # module-level k grid read by _score_task
    OB._G.update(S=S_tr, L=D.Ltr, halves=[np.arange(0, n // 2), np.arange(n // 2, n)], ts=ts, lab=lab,
                 mission=m, channels=D.channels, rule=D.train.rule_s, merge=merge)
    tasks = [(layer, lev, i) for lev in grid["levels"][layer] for i in (0, 1)]
    with ProcessPoolExecutor(min(workers, len(tasks))) as ex:
        rows = [r for part in ex.map(OB._score_task, tasks) for r in part]
    g = (pd.DataFrame(rows).groupby(["kind", "level", "k"], as_index=False)
         .agg(cv=("EW_F0.5", "mean"), alarm_fraction=("alarm_fraction", "mean")))
    b = g.sort_values(["cv", "alarm_fraction"], ascending=[False, True]).iloc[0]
    cal = Calibrator(layer, float(b.level), int(b.k), merge, rule_s=D.train.rule_s).fit(S_tr, D.Ltr)
    return cal, g.to_dict("records")


def _halfpool_task(args):
    lev, i = args
    G = _HP
    rows_ev = G["halves"][i]
    keep = np.ones(G["S"].shape[0], bool)
    keep[rows_ev] = False                           # the evaluated rows never enter the pool
    c = Calibrator(G["layer"], lev, 1, G["merge"], rule_s=G["rule"]).fit(G["S"][keep], G["L"][keep])
    H = c.hot(np.asarray(G["S"][rows_ev], np.float32))
    t = G["ts"][rows_ev]
    lab = G["lab"][(G["lab"].StartTime >= t[0]) & (G["lab"].EndTime <= t[-1])]
    out = []
    from cosmos.ai.telemetry.alarms import attribute, persist_and_merge
    for k in G["ks"]:
        g = persist_and_merge(H.any(1), k, G["merge"])
        ch = attribute(g, H, back=k - 1)
        s = score_alarms(G["m"], t, ch, G["channels"], lab, per_channel=False, categories=("anomaly_and_rare",))
        out.append({"level": lev, "k": k, "half": i, "EW_F0.5": s.get("anomaly_and_rare/EW_F_0.50", 0.0),
                    "alarm_fraction": float(g.mean())})
    return out


_HP = {}


def sel_halfpool_cal(D, m, layer, S_tr, grid, merge, workers=8):
    """P_half, D_cal: pool = all training-half nominal OOF scores except the evaluated
    calibration-window half; objective on calibration-window labels."""
    a, b = D.p["cal"]
    n = b - a
    ts = D.train.timestamps()
    lab = period_labels(m, ts[a], ts[b - 1])
    lab = lab[lab.Channel.isin(D.channels)]
    _HP.update(S=S_tr, L=D.Ltr, halves=[np.arange(a, a + n // 2), np.arange(a + n // 2, b)], ts=ts, lab=lab,
               m=m, channels=D.channels, rule=D.train.rule_s, merge=merge, layer=layer, ks=grid["ks"])
    tasks = [(lev, i) for lev in grid["levels"][layer] for i in (0, 1)]
    with ProcessPoolExecutor(min(workers, len(tasks))) as ex:
        rows = [r for part in ex.map(_halfpool_task, tasks) for r in part]
    g = (pd.DataFrame(rows).groupby(["level", "k"], as_index=False)
         .agg(cv=("EW_F0.5", "mean"), alarm_fraction=("alarm_fraction", "mean")))
    bst = g.sort_values(["cv", "alarm_fraction"], ascending=[False, True]).iloc[0]
    cal = Calibrator(layer, float(bst.level), int(bst.k), merge, rule_s=D.train.rule_s).fit(S_tr, D.Ltr)
    return cal, g.to_dict("records")


# ---------------------------------------------------------------------------- one cell
def run_cell(name, D, m, layer, scores, S_key, P, Dsel, grid, merge, ts_te, lab_te):
    t0 = time.time()
    S_tr, S_te = scores[S_key]
    if P == "cal" and Dsel == "cal":
        calrows = S_tr if S_key == "S0" else S_tr[D.p["cal"][0]:D.p["cal"][1]]
        cal, grid_rows = sel_cal(D, m, layer, calrows, grid, merge)
        # the exploratory rule fits the final pool on the calibration window only
    elif P == "half" and Dsel == "half":
        cal, grid_rows = sel_half(D, m, layer, S_tr, grid, merge)
    elif P == "half" and Dsel == "cal":
        cal, grid_rows = sel_halfpool_cal(D, m, layer, S_tr, grid, merge)
    else:
        raise ValueError((P, Dsel))
    g, ch = cal.alarms(np.asarray(S_te, np.float32))
    sc = score_alarms(m, ts_te, ch, D.channels, lab_te, per_channel=False)
    fa = false_alarm_stats(ts_te, g, lab_te)
    tab = XM.event_table(g, ts_te, lab_te)
    np.save(out_path("p1_decompose", f"m{m}_{D.channels_set}_{layer}", f"{name}_global.npy"), np.packbits(g))
    row = {"cell": name, "S": S_key, "P": P, "D": Dsel, "grid": "G0" if grid is G0 else "G1",
           "chosen_level": float(cal.level), "chosen_k": int(cal.k),
           "official_EW_F0.5": sc.get("anomaly_and_rare/EW_F_0.50", 0.0),
           "anomaly_EW_F0.5": sc.get("anomaly/EW_F_0.50", 0.0),
           "EW_precision": sc.get("anomaly_and_rare/EW_precision", 0.0),
           "EW_recall": sc.get("anomaly_and_rare/EW_recall", 0.0),
           "exact_EW_F0.5": XM.f_beta(tab), **fa, "seconds": time.time() - t0, "selection_grid": grid_rows}
    print(f"  {name}: S={S_key} P={P} D={Dsel} {row['grid']} -> level {cal.level:g} k {cal.k}: "
          f"F0.5 {row['official_EW_F0.5']:.4f} (exact {row['exact_EW_F0.5']:.4f}), FA/yr "
          f"{fa['false_alarms_per_year']:.1f}, {row['seconds']:.0f} s", flush=True)
    return row, tab


CELLS = [  # name, S, P, D, grid
    ("c0", "S0", "cal", "cal", G0), ("c1", "S0", "cal", "cal", G1), ("c2", "Sh", "cal", "cal", G1),
    ("c3", "S1", "cal", "cal", G1), ("c4", "S1", "half", "cal", G1), ("c5", "S1", "half", "half", G1),
    ("r1", "S1", "half", "half", G0), ("r2", "S1", "half", "cal", G0), ("r3", "S1", "cal", "cal", G0),
    ("r4", "Sh", "cal", "cal", G0)]
FORWARD = [("G", "c0", "c1"), ("F", "c1", "c2"), ("sigma", "c2", "c3"), ("P", "c3", "c4"), ("D", "c4", "c5")]
REVERSE = [("G", "r1", "c5"), ("D", "r2", "r1"), ("P", "r3", "r2"), ("sigma", "r4", "r3"), ("F", "c0", "r4")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--channels", default="target")
    ap.add_argument("--layer", default="conformal_bh", choices=("conformal_bh", "threshold"))
    a = ap.parse_args()
    prov = provenance(__file__)
    m, chs, layer = a.mission, a.channels, a.layer
    D = MissionData(m, chs)
    D.channels_set = chs
    merge = int(OB.MERGE_H * 3600 / D.train.rule_s)
    ts_te = D.test.timestamps()
    lab_te = official_test_labels(m, D.test)
    lab_te = lab_te[lab_te.Channel.isin(D.channels)]
    scores, sig_info = score_levels(D, m, chs)
    print(f"m{m} {chs} {layer}: A0 scores reproduced to {sig_info['s0_recompute_max_abs_diff']:.3g}", flush=True)
    rows, tabs = {}, {}
    for name, S_key, P, Dsel, grid in CELLS:
        rows[name], tabs[name] = run_cell(name, D, m, layer, scores, S_key, P, Dsel, grid, merge, ts_te, lab_te)
    # reproduction of the two endpoints against the frozen results
    a0_off = [r for r in __import__("json").load(open(os.path.join(
        ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "arms", f"m{m}_{chs}_ar8_s0.json")))["rows"]
        if r["calibrator"] == layer][0]["anomaly_and_rare/EW_F0.5"]
    a1_off = [r for r in __import__("json").load(open(os.path.join(
        ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "oof_protocol", "test", f"m{m}_{chs}_ar8.json")))
        if r["calibrator"] == layer][0]["anomaly_and_rare/EW_F0.5"]
    repro = {"A0_official": a0_off, "c0": rows["c0"]["official_EW_F0.5"],
             "A1_official": a1_off, "c5": rows["c5"]["official_EW_F0.5"]}
    repro["passed"] = bool(abs(repro["A0_official"] - repro["c0"]) < 1e-9 and abs(repro["A1_official"] - repro["c5"]) < 1e-9)
    print(f"reproduction: {repro}", flush=True)
    steps = []
    rng = np.random.default_rng(20261024)
    for path, seq in (("forward", FORWARD), ("reverse", REVERSE)):
        for factor, x, y in seq:
            r = XM.compare([tabs[y]], [tabs[x]], rng)
            r.pop("replicates")
            steps.append({"path": path, "factor": factor, "from": x, "to": y,
                          "delta_official": rows[y]["official_EW_F0.5"] - rows[x]["official_EW_F0.5"],
                          "delta_exact": r["diff"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
                          "p_boot": r["p_boot"]})
    verdict = {}
    for f in ("G", "F", "sigma", "P", "D"):
        fw = [s["delta_official"] for s in steps if s["factor"] == f and s["path"] == "forward"][0]
        rv = [s["delta_official"] for s in steps if s["factor"] == f and s["path"] == "reverse"][0]
        verdict[f] = ("material" if fw >= 0.05 and rv >= 0.05 else
                      "order-dependent" if abs(fw - rv) > 0.05 else "not material")
    save_json({**prov, "mission": m, "channels": chs, "layer": layer, "sigma_check": sig_info, "reproduction": repro,
               "cells": rows, "steps": steps, "verdict": verdict},
              "p1_decompose", f"m{m}_{chs}_{layer}.json")
    for s in steps:
        print(f"  {s['path']:7s} {s['factor']:5s} {s['from']}->{s['to']}: {s['delta_official']:+.4f} "
              f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]")
    print("verdict", verdict)


if __name__ == "__main__":
    main()
