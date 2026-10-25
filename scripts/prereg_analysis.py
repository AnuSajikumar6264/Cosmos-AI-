"""Confirmatory analysis of research_design/preregistration/PREREG_TELEMETRY_V1.md.

    python scripts/prereg_analysis.py select        # training half only (any time)
    python scripts/prereg_analysis.py test          # gated by freeze.json
    python scripts/prereg_analysis.py stats         # H4-H6 (+ Mission 2 replication), Holm

``select`` runs the frozen selection for every arm whose scores exist.  ``test`` scores each
arm's test half once; an arm that already has a test file is not scored again.  ``stats``
reads the saved test alarms and writes
``outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.{json,md}``.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cosmos.ai.telemetry import oof_benchmark as OB  # noqa: E402

SEEDS = range(5)
EV = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "event_model")


def arms(mission: int, channels: str = "target") -> dict:
    """Arm name -> list of score sources (one per seed); missing seeds are left out."""
    out = {"A1": ["ar8"]}
    for name, pat in (("A2", f"m{mission}_{channels}_gbt_ar8_s{{s}}"),
                      ("A3", f"m{mission}_{channels}_event_ar8_all_c3_s{{s}}"),
                      ("A3_no_residual", f"m{mission}_{channels}_event_ar8_no_residual_c3_s{{s}}"),
                      ("A3_two_class", f"m{mission}_{channels}_event_ar8_all_c2_s{{s}}")):
        src = [f"event:{os.path.join(EV, pat.format(s=s))}" for s in SEEDS
               if os.path.isfile(os.path.join(EV, pat.format(s=s), "event_test.npy"))]
        if src:
            out[name] = src
    return out


def tag_of(mission, channels, source):
    return OB.load_source(mission, channels, source)[2]


def do_select(missions=(1, 2), channels="target"):
    for m in missions:
        for name, srcs in arms(m, channels).items():
            for src in srcs:
                t = tag_of(m, channels, src)
                if os.path.isfile(os.path.join(OB.result_dir("selection"), f"{t}.json")):
                    continue
                print(f"select M{m} {name} {t}", flush=True)
                OB.select(m, channels, src)


def do_test(missions=(1, 2), channels="target"):
    OB.check_freeze()
    for m in missions:
        a0 = f"m{m}_{channels}_ar8_calperiod"
        if not os.path.isfile(os.path.join(OB.result_dir("alarms"),
                                           f"{a0}_conformal_bh_global.npy")):
            for cal in ("conformal_bh", "threshold"):
                OB.calperiod_alarms(m, channels, cal)
        for name, srcs in arms(m, channels).items():
            for src in srcs:
                t = tag_of(m, channels, src)
                if os.path.isfile(os.path.join(OB.result_dir("test"), f"{t}.json")):
                    continue
                print(f"TEST M{m} {name} {t}", flush=True)
                OB.test(m, channels, src)


def _summary(mission, channels="target") -> pd.DataFrame:
    rows = []
    for name, srcs in arms(mission, channels).items():
        for src in srcs:
            p = os.path.join(OB.result_dir("test"), f"{tag_of(mission, channels, src)}.json")
            if os.path.isfile(p):
                for r in json.load(open(p)):
                    rows.append({"arm": name, **{k: v for k, v in r.items()
                                                 if not isinstance(v, (dict, list))}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["anomaly_and_rare/EW_F0.5", "anomaly_and_rare/EW_precision",
            "anomaly_and_rare/EW_recall", "anomaly/EW_F0.5", "false_alarms_per_year",
            "rare_event_only_runs", "anomaly_and_rare/ADTQC", "anomaly_and_rare/AFF_F0.5"]
    g = df.groupby(["arm", "calibrator"])[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(
        g.size().rename("n_seeds"))
    return out.reset_index()


def _official(mission, channels, tags) -> dict:
    """Seed-mean official primary metric per arm (conformal_bh); A0 from esa_benchmark."""
    out = {}
    for arm, ts in tags.items():
        if arm == "A0":
            p = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "arms",
                             f"m{mission}_{channels}_ar8_s0.json")
            rows = json.load(open(p))["rows"]
            out[arm] = [r for r in rows if r["calibrator"] == "conformal_bh"][0][
                "anomaly_and_rare/EW_F0.5"]
            continue
        v = []
        for t in ts:
            rows = json.load(open(os.path.join(OB.result_dir("test"), f"{t}.json")))
            v.append([r for r in rows if r["calibrator"] == "conformal_bh"][0][
                "anomaly_and_rare/EW_F0.5"])
        out[arm] = float(np.mean(v))
    return out


def do_stats(channels="target"):
    fr = OB.check_freeze()
    res = {"freeze_id": fr["freeze_id"], "missions": {}}
    for m in (1, 2):
        a = arms(m, channels)
        tags = {k: [tag_of(m, channels, s) for s in v] for k, v in a.items()}
        tags["A0"] = [f"m{m}_{channels}_ar8_calperiod"]
        comps = [(h, tags[x], tags[y], "conformal_bh") for h, x, y in
                 (("H4: A3 - A1", "A3", "A1"), ("H5: A3 - A2", "A3", "A2"),
                  ("H6: A1 - A0", "A1", "A0")) if x in tags and y in tags]
        rows = OB.compare(m, comps, channels)
        off = _official(m, channels, tags)
        for r in rows:
            x, y = r["comparison"].split(": ")[1].split(" - ")
            r["A_official"], r["B_official"] = off[x], off[y]
            r["diff_official"] = off[x] - off[y]
        if m == 1:
            from cosmos.ai.evaluation import holm
            adj = holm({r["comparison"]: r["p_boot"] for r in rows})
            for r in rows:
                r["p_holm"] = adj[r["comparison"]]
                sup = r["p_holm"] < 0.05 and r["diff_official"] >= 0.05
                ref = r["ci95_high"] < 0.05
                r["verdict"] = "supported" if sup else ("refuted" if ref else "inconclusive")
                r["n_seeds_A"], r["n_seeds_B"] = len(r["A"]), len(r["B"])
        else:
            for r in rows:
                r["verdict"] = "replication (no inference)"
        res["missions"][m] = {"tests": rows,
                              "summary": _summary(m, channels).to_dict("records")}
    d = OB.result_dir()
    with open(os.path.join(d, "CONFIRMATORY.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o if not isinstance(o, np.generic)
                  else o.item())
    lines = [f"# Confirmatory results (freeze {fr['freeze_id']})", ""]
    for m, r in res["missions"].items():
        lines += [f"## Mission {m} ({'primary' if m == 1 else 'replication'})", "",
                  "| Test | A (official) | B (official) | diff (official) | diff (re-impl.) | "
                  "95 % CI | p (boot) | p (Holm) | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for t in r["tests"]:
            lines.append(f"| {t['comparison']} | {t['A_official']:.3f} | {t['B_official']:.3f} | "
                         f"{t['diff_official']:+.3f} | {t['diff']:+.3f} | "
                         f"[{t['ci95_low']:+.3f}, {t['ci95_high']:+.3f}] | "
                         f"{t['p_boot']:.4f} | {t.get('p_holm', float('nan')):.4f} | "
                         f"{t['verdict']} |")
        lines.append("")
    with open(os.path.join(d, "CONFIRMATORY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    act = sys.argv[1] if len(sys.argv) > 1 else "select"
    {"select": do_select, "test": do_test, "stats": do_stats}[act]()
