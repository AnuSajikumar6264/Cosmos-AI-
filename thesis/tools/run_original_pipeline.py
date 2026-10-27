"""Re-run the original COSMOS pipeline (cosmos/health, debris, reentry) with the CURRENT code and
store the outputs the thesis reports: thesis/evidence/original_pipeline_runs.json.

    python thesis/tools/run_original_pipeline.py

The demo notebook's saved outputs were produced by an earlier package version (1.0.0) and are
not used as evidence; everything here is regenerated with fixed seeds.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import cosmos  # noqa: E402

OUT = os.path.join(ROOT, "thesis", "evidence", "original_pipeline_runs.json")
R: dict = {"cosmos_version": cosmos.__version__, "generated_utc": pd.Timestamp.utcnow().isoformat()}


def jsonable(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def step(name, fn):
    t0 = time.time()
    try:
        R[name] = fn()
        R[name + "_seconds"] = time.time() - t0
        print(f"{name}: ok in {time.time() - t0:.0f} s", flush=True)
    except Exception as e:                                   # record failures, keep going
        R[name] = {"error": repr(e)}
        print(f"{name}: FAILED {e!r}", flush=True)
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=jsonable)


def default_run():
    res = cosmos.run_pipeline()
    return {"summary": [list(map(str, r)) for r in res.summary().to_records()] if hasattr(res.summary(), "to_records") else str(res.summary())}


def default_summary_table():
    res = cosmos.run_pipeline()
    s = res.summary()
    return s.to_dict("records") if isinstance(s, pd.DataFrame) else s


def monitor_grid():
    from cosmos.health import HealthMonitor, detection_delay, simulate_telemetry
    from cosmos.health.telemetry import CHANNELS, FAULTS
    subsystems = list(CHANNELS)
    rows = []
    for sub in subsystems:
        for fault in FAULTS:
            if fault == "stuck":
                continue
            ts = simulate_telemetry(faults={sub: fault}, seed=7)
            r = HealthMonitor().fit(ts).score(ts)
            d = detection_delay(ts, r)
            rows.append({"subsystem": sub, "fault": fault, "detected": bool(d["detected"]),
                         "delay_minutes": d.get("delay_minutes"), "final_anomaly": float(r.anomaly[-1]),
                         "worst_subsystem": str(r.health_at(-1).worst())})
    ts = simulate_telemetry(seed=7)
    r = HealthMonitor().fit(ts).score(ts)
    rows.append({"subsystem": "none", "fault": "nominal", "detected": bool(r.detected_index >= 0),
                 "delay_minutes": None, "final_anomaly": float(r.anomaly[-1])})
    return rows


def stuck_fault():
    from cosmos.health import simulate_telemetry
    try:
        simulate_telemetry(faults={"Propulsion System": "stuck"}, seed=7)
        return {"raises": False}
    except Exception as e:
        return {"raises": True, "error": repr(e)}


def scenarios():
    return cosmos.health.scenario_table().to_dict("records")


def failure_modes():
    table, _ = cosmos.compare_failure_modes(n_samples=600)
    return table.to_dict("records")


def catalogue():
    cat = cosmos.debris.load_satcat()
    ov = cosmos.debris.catalogue_overview(cat)
    clouds = cosmos.debris.all_breakup_clouds(cat)
    co = cosmos.debris.cloud_overview(clouds)
    cal = cosmos.debris.calibrate_area_to_mass(clouds)
    val = cosmos.debris.validate_lifetimes(clouds, cal["area_to_mass_censored"])
    dens = cosmos.debris.spatial_density(cat)
    rate = cosmos.debris.total_leo_collision_rate(cosmos.debris.collision_risk(dens))
    return {"overview": ov.reset_index().to_dict("records"), "clouds": co.to_dict("records"),
            "calibration": {k: v for k, v in cal.items() if k not in ("grid", "losses")},
            "validation": val.to_dict("records") if isinstance(val, pd.DataFrame) else val,
            "density": dens.to_dict("records") if isinstance(dens, pd.DataFrame) else dens,
            "leo_rate": rate}


def detection():
    ds = cosmos.debris.generate_dataset(n_scenes=16, seed=3)
    agg = cosmos.debris.aggregate(cosmos.debris.evaluate_dataset(ds))
    return agg.to_dict("records")


def zones():
    hz = cosmos.health.assess_hazard(0.85, cosmos.health.SubsystemHealth.nominal())
    cloud = cosmos.reentry.sample_cloud(parent_mass=1800.0, kind="explosion", rng=7)
    fp = cosmos.reentry.monte_carlo_footprint(hz, cloud, cosmos.reentry.DispersionConfig(n_samples=1200, seed=0))
    z = cosmos.reentry.zone_comparison(fp, n_splits=12, seed=0)
    return {"footprint": fp.summary() if isinstance(fp.summary(), dict) else str(fp.summary()),
            "cloud": cloud.summary() if isinstance(cloud.summary(), dict) else str(cloud.summary()),
            "zones": z.to_dict("records"), "risk": cosmos.reentry.assess_footprint(fp)}


def corridors():
    out = cosmos.corridor_risk(n_samples=300)
    return out.to_dict("records") if isinstance(out, pd.DataFrame) else out


if __name__ == "__main__":
    for name, fn in (("default_summary", default_summary_table), ("monitor_grid", monitor_grid),
                     ("stuck_fault", stuck_fault), ("scenario_table", scenarios),
                     ("detection_single_vs_pair", detection), ("catalogue", catalogue),
                     ("zones", zones), ("failure_modes", failure_modes), ("corridors", corridors)):
        step(name, fn)
    print("wrote", OUT)
