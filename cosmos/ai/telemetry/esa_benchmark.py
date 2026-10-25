"""ESA-ADB benchmark driver (official protocol, full or lightweight channel sets).

    python -m cosmos.ai.telemetry.esa_benchmark --mission 1 --encoder tcn --seed 0
    python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --baseline global_std5

An *arm* is (mission, channel set, encoder, seed, telecommands on/off).  For each arm:

1. train the encoder on nominal ``fit`` windows (deep encoders), early-stop on ``es``;
2. forecast the ``cal`` period and the official test half on the shared origin schedule;
3. for every calibrator, select its level and persistence on ``cal`` by cross-fitting;
4. score the test half ONCE per calibrator with the official ESA-ADB metrics;
5. record realised false alarms per year, rare-event-only alarm runs, forecast-interval
   coverage, parameters, training time and inference time per telemetry-day.

Results: one JSON per arm in ``outputs/benchmarks/telemetry_esa_adb/arms/`` plus the run
folder in ``outputs/experiments/``; :func:`collect` merges them into ``results.csv``.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from ...common import output_dir
from ..experiment import Run, count_parameters, model_size_mb, pick_device
from .alarms import select_calibrator
from .esa_eval import (false_alarm_stats, headline, official_test_labels, period_labels,
                       score_alarms)
from .forecasting import (ForecastConfig, MissionData, forecast_record, load_chronos,
                          nonconformity, residual_scale, train_forecaster)

CALIBRATORS = ("threshold", "conformal_bh", "conformal_bh_rolling")
TRAINABLE = ("tcn", "patchtst")


def bench_dir(*parts) -> str:
    """A directory under ``outputs/benchmarks/telemetry_esa_adb`` (created on demand)."""
    return output_dir("benchmarks", "telemetry_esa_adb", *parts)


def bench_file(*parts) -> str:
    return os.path.join(bench_dir(*parts[:-1]), parts[-1])


def _row_metrics(scores: dict) -> dict:
    out = {}
    for key in ("anomaly_and_rare", "anomaly"):
        for k, v in headline(scores, key).items():
            out[f"{key}/{k}"] = v
    return out


def evaluate_scores(D: MissionData, S_cal, L_cal, S_te, tag: str, calibrators=CALIBRATORS,
                    merge_h: float = 1.0) -> list:
    """Steps 3-5 for one set of scores; returns one row per calibrator."""
    m = D.mission
    rule = D.train.rule_s
    merge = int(merge_h * 3600 / rule)
    a, b = D.p["cal"]
    ts_cal = D.train.timestamps()[a:b]
    lab_cal = period_labels(m, ts_cal[0], ts_cal[-1])
    lab_cal = lab_cal[lab_cal.Channel.isin(D.channels)]

    def score_fn(g, ch, rows):
        t = ts_cal[rows]
        lab = lab_cal[(lab_cal.StartTime >= t[0]) & (lab_cal.EndTime <= t[-1])]
        s = score_alarms(m, t, ch, D.channels, lab, per_channel=False,
                         categories=("anomaly_and_rare",))
        return s.get("anomaly_and_rare/EW_F_0.50", 0.0)

    ts_te = D.test.timestamps()
    lab_te = official_test_labels(m, D.test)
    lab_te = lab_te[lab_te.Channel.isin(D.channels)]
    rows = []
    for kind in calibrators:
        t0 = time.time()
        cal, grid = select_calibrator(kind, S_cal, L_cal, score_fn, rule, merge)
        t_sel = time.time() - t0
        t0 = time.time()
        g, ch = cal.alarms(S_te)
        t_alarm = time.time() - t0
        sc = score_alarms(m, ts_te, ch, D.channels, lab_te, per_channel=True)
        fa = false_alarm_stats(ts_te, g, lab_te)
        rows.append({"tag": tag, "calibrator": kind, "selected_level": cal.level,
                     "selected_k": cal.k, "merge_steps": merge,
                     "cv_EW_F0.5": grid[0]["cv_EW_F0.5"],
                     "selection_seconds": t_sel, "alarm_seconds": t_alarm,
                     **_row_metrics(sc), **fa, "grid": grid,
                     "raw_scores": {k: v for k, v in sc.items()}})
        print(f"    {tag} {kind:22s} level {cal.level:g} k {cal.k}: EW F0.5 "
              f"{rows[-1]['anomaly_and_rare/EW_F0.5']:.3f} (P {rows[-1]['anomaly_and_rare/EW_precision']:.3f} "
              f"R {rows[-1]['anomaly_and_rare/EW_recall']:.3f}), false alarms/yr "
              f"{fa['false_alarms_per_year']:.1f}", flush=True)
    return rows


def interval_coverage_nominal(Y, q25, q75, L) -> float:
    ok = np.isfinite(q25) & (L == 0)
    return float(((Y >= q25) & (Y <= q75))[ok].mean()) if ok.any() else float("nan")


def run_arm(cfg: ForecastConfig, calibrators=CALIBRATORS, save_scores: bool = True) -> dict:
    D = MissionData(cfg.mission, cfg.channels)
    tag = (f"m{cfg.mission}_{cfg.channels}_{cfg.encoder}"
           f"{'_tc' if cfg.telecommands else ''}_s{cfg.seed}")
    out = {"tag": tag, "config": dataclasses.asdict(cfg)}
    with Run(f"esa_{tag}", dataclasses.asdict(cfg), cfg.seed, cfg.device,
             group="esa_adb") as run:
        model, info = None, {}
        if cfg.encoder in TRAINABLE:
            model, info = train_forecaster(D, cfg, run)
            run.save_checkpoint(model, "best", normaliser={"median": D.med.tolist(),
                                                           "scale": D.scale.tolist(),
                                                           "channels": D.channels})
            info.update(params=count_parameters(model), size_mb=model_size_mb(model))
        elif cfg.encoder.startswith("chronos"):
            model = load_chronos(cfg.encoder, cfg.device)
            info.update(params=count_parameters(model.model),
                        size_mb=model_size_mb(model.model), zero_shot=True)
        sigma = residual_scale(D, cfg, model) if cfg.encoder in ("persistence", "ar8") else None
        t0 = time.time()
        Yc, q25c, q50c, q75c, Lc = forecast_record(model, D, "cal", cfg)
        t_cal = time.time() - t0
        t0 = time.time()
        Yt, q25t, q50t, q75t, Lt = forecast_record(model, D, "test", cfg)
        t_test = time.time() - t0
        test_days = len(Yt) * D.test.rule_s / 86400
        S_cal = nonconformity(Yc, q25c, q50c, q75c, sigma)
        S_te = nonconformity(Yt, q25t, q50t, q75t, sigma)
        info.update(cal_forecast_s=t_cal, test_forecast_s=t_test,
                    inference_s_per_telemetry_day=t_test / test_days,
                    n_channels=D.n_channels)
        if sigma is None:
            info["iqr_coverage_test_nominal"] = interval_coverage_nominal(Yt, q25t, q75t, Lt)
            info["iqr_coverage_cal_nominal"] = interval_coverage_nominal(Yc, q25c, q75c, Lc)
        mae = np.abs(Yt - q50t)[(Lt == 0) & np.isfinite(q50t)]
        info["test_nominal_mae_robust_units"] = float(mae.mean())
        info["test_nominal_median_ae_robust_units"] = float(np.median(mae))
        if save_scores:
            np.save(os.path.join(run.dir, "scores_cal.npy"), S_cal)
            np.save(os.path.join(run.dir, "scores_test.npy"), S_te)
        del Yc, q25c, q50c, q75c, Yt, q25t, q50t, q75t
        rows = evaluate_scores(D, S_cal, Lc, S_te, tag, calibrators)
        for r in rows:
            r.update({"mission": cfg.mission, "channels": cfg.channels,
                      "encoder": cfg.encoder, "telecommands": cfg.telecommands,
                      "seed": cfg.seed, "run_id": run.run_id, **info})
        out["rows"] = rows
        best = max(rows, key=lambda r: r["anomaly_and_rare/EW_F0.5"])
        run.finish({k: v for k, v in best.items() if np.isscalar(v)}, **info)
    with open(bench_file("arms", f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    return out


def missing_modality(run_dir: str, device: str | None = None) -> dict:
    """E10: a telecommand-gated arm with the telecommand stream REMOVED at test time.

    The checkpoint, the calibration scores and the calibrators selected with telecommands
    present are reused unchanged; only the test forecasts are recomputed with zeroed
    telecommand features (what modality dropout during training prepared the model for).
    """
    import torch
    import yaml
    from .alarms import Calibrator
    with open(os.path.join(run_dir, "config.yaml")) as f:
        c = yaml.safe_load(f)
    fields = {k: c[k] for k in ForecastConfig.__dataclass_fields__ if k in c}
    cfg = ForecastConfig(**{**fields, "device": device or pick_device()})
    assert cfg.telecommands, "not a telecommand arm"
    D = MissionData(cfg.mission, cfg.channels)
    from .forecasting import build_model
    model = build_model(cfg, D.n_channels, D.cov_tr.shape[1]).to(cfg.device)
    ck = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location=cfg.device)
    model.load_state_dict(ck["state_dict"])
    D.cov_te[:] = 0.0                                          # the modality is missing
    D.cov_tr[-4097:] = 0.0                                     # including the warm-up context
    Yt, q25t, q50t, q75t, _Lt = forecast_record(model, D, "test", cfg)
    S_te = nonconformity(Yt, q25t, q50t, q75t)
    S_cal = np.load(os.path.join(run_dir, "scores_cal.npy"))
    tag = (f"m{cfg.mission}_{cfg.channels}_{cfg.encoder}_tc_s{cfg.seed}")
    with open(bench_file("arms", f"{tag}.json")) as f:
        arm = json.load(f)
    a, b = D.p["cal"]
    ts_te = D.test.timestamps()
    lab_te = official_test_labels(cfg.mission, D.test)
    lab_te = lab_te[lab_te.Channel.isin(D.channels)]
    rows = []
    for r in arm["rows"]:
        cal = Calibrator(r["calibrator"], r["selected_level"], int(r["selected_k"]),
                         int(r["merge_steps"]), rule_s=D.train.rule_s).fit(S_cal, D.Ltr[a:b])
        g, ch = cal.alarms(S_te)
        sc = score_alarms(cfg.mission, ts_te, ch, D.channels, lab_te, per_channel=True)
        rows.append({"tag": tag + "_telecommands_removed_at_test", "calibrator": r["calibrator"],
                     "with_telecommands_EW_F0.5": r["anomaly_and_rare/EW_F0.5"],
                     **_row_metrics(sc), **false_alarm_stats(ts_te, g, lab_te),
                     "mission": cfg.mission, "encoder": cfg.encoder, "seed": cfg.seed})
        print(f"  {tag} {r['calibrator']}: EW F0.5 with TC {r['anomaly_and_rare/EW_F0.5']:.3f} "
              f"-> without {rows[-1]['anomaly_and_rare/EW_F0.5']:.3f}", flush=True)
    out = os.path.join(bench_dir("robustness"), f"{tag}_telecommands_removed.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return {"rows": rows, "path": out}


def telecommand_mask_baseline(run_dir: str, minutes=(30, 120)) -> list:
    """H3 rule baseline: drop alarm samples within ``minutes`` after any telecommand.

    Post hoc on an arm's saved scores and selected calibrators (no retraining).  The windows
    are fixed in advance.  What H3 is about is measured directly: alarm runs that touch only
    rare nominal events (to be suppressed) and anomaly recall (to be kept).
    """
    import yaml
    from .alarms import Calibrator, attribute, persist_and_merge
    with open(os.path.join(run_dir, "config.yaml")) as f:
        c = yaml.safe_load(f)
    D = MissionData(c["mission"], c.get("channels", "target"))
    S_cal = np.load(os.path.join(run_dir, "scores_cal.npy"))
    S_te = np.load(os.path.join(run_dir, "scores_test.npy"))
    tag = (f"m{c['mission']}_{c.get('channels', 'target')}_{c['encoder']}"
           f"{'_tc' if c.get('telecommands') else ''}_s{c['seed']}")
    with open(bench_file("arms", f"{tag}.json")) as f:
        arm = json.load(f)
    a, b = D.p["cal"]
    ts = D.test.timestamps()
    lab = official_test_labels(c["mission"], D.test)
    lab = lab[lab.Channel.isin(D.channels)]
    any_tc = D.test.TC.max(axis=1) > 0
    rows = []
    for r in arm["rows"]:
        if r["calibrator"] != "conformal_bh":
            continue
        cal = Calibrator(r["calibrator"], r["selected_level"], int(r["selected_k"]),
                         int(r["merge_steps"]), rule_s=D.train.rule_s).fit(S_cal, D.Ltr[a:b])
        H = cal.hot(S_te)
        for mins in (0,) + tuple(minutes):
            Hm = H.copy()
            if mins:
                w = int(mins * 60 / D.test.rule_s)
                after = np.convolve(any_tc.astype(int), np.ones(w + 1, int))[:len(any_tc)] > 0
                Hm[after] = False
            g = persist_and_merge(Hm.any(1), cal.k, cal.merge)
            ch = attribute(g, Hm, back=cal.k - 1)
            sc = score_alarms(c["mission"], ts, ch, D.channels, lab, per_channel=False)
            fa = false_alarm_stats(ts, g, lab)
            rows.append({"tag": tag, "mask_minutes_after_telecommand": mins,
                         "masked_fraction_of_time": float(after.mean()) if mins else 0.0,
                         **_row_metrics(sc), **fa})
            print(f"  {tag} mask {mins:3d} min: EW F0.5 {rows[-1]['anomaly_and_rare/EW_F0.5']:.3f}, "
                  f"anomaly recall {rows[-1]['anomaly/EW_recall']:.3f}, rare-only runs "
                  f"{fa['rare_event_only_runs']}", flush=True)
    out = os.path.join(bench_dir("robustness"), f"{tag}_telecommand_mask.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return rows


def run_baseline(mission: int, kind: str, channels: str = "target") -> dict:
    """GlobalSTD3/5 (official) and the COSMOS monitor, scored with the same code."""
    from .baselines import cosmos_monitor, global_std, out_of_limits
    from .esa_eval import false_alarm_stats as fas
    D = MissionData(mission, channels)
    tag = f"m{mission}_{channels}_{kind}"
    t0 = time.time()
    if kind.startswith("global_std"):
        al, _mu, _sd = global_std(D.train, D.test, D.channels, float(kind[-1]))
    elif kind in ("out_of_limits", "out_of_limits_all"):
        al, _lo, _hi = out_of_limits(D.train, D.test, D.channels, kind == "out_of_limits")
    elif kind == "cosmos_monitor":
        al = cosmos_monitor(D.train, D.test, D.channels)
    elif kind == "cosmos_monitor_no_ar":
        al = cosmos_monitor(D.train, D.test, D.channels, residuals=False)
    else:
        raise KeyError(kind)
    secs = time.time() - t0
    ts = D.test.timestamps()
    lab = official_test_labels(mission, D.test)
    lab = lab[lab.Channel.isin(D.channels)]
    sc = score_alarms(mission, ts, al, D.channels, lab, per_channel=True)
    row = {"tag": tag, "mission": mission, "channels": channels, "encoder": kind,
           "calibrator": "native", "seed": 0, "telecommands": False,
           **_row_metrics(sc), **fas(ts, al.max(1), lab), "fit_and_score_s": secs,
           "raw_scores": sc}
    with open(bench_file("arms", f"{tag}.json"), "w") as f:
        json.dump({"tag": tag, "rows": [row]}, f, indent=1, default=str)
    print(f"  {tag}: EW F0.5 {row['anomaly_and_rare/EW_F0.5']:.4f}  false alarms/yr "
          f"{row['false_alarms_per_year']:.1f}", flush=True)
    return row


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(bench_dir("arms"), "*.json"))):
        with open(p) as f:
            d = json.load(f)
        for r in d["rows"]:
            rows.append({k: v for k, v in r.items() if k not in ("grid", "raw_scores")})
    df = pd.DataFrame(rows)
    df.to_csv(bench_file("results.csv"), index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--channels", default="target")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--telecommands", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--calibrators", default=",".join(CALIBRATORS))
    ap.add_argument("--missing-modality", default=None, metavar="RUN_DIR",
                    help="re-score a telecommand arm with telecommands removed at test")
    a = ap.parse_args(argv)
    if a.missing_modality:
        missing_modality(a.missing_modality, a.device)
        return 0
    if a.baseline:
        run_baseline(a.mission, a.baseline, a.channels)
        return 0
    cfg = ForecastConfig(mission=a.mission, encoder=a.encoder, channels=a.channels,
                         seed=a.seed, telecommands=a.telecommands, steps=a.steps,
                         device=a.device or pick_device())
    run_arm(cfg, tuple(a.calibrators.split(",")))
    collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
