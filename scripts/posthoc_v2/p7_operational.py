"""P7: operational metrics of the frozen arms (conformal-BH layer, test halves) from the stored
global alarm arrays: detection delay, alarm-run duration, runs per detected event, false-alarm
runs per 1,000 nominal hours, and detected / missed events by category.

    python scripts/posthoc_v2/p7_operational.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, provenance, save_json  # noqa: E402
import exact_metric as XM  # noqa: E402

from cosmos.ai.telemetry.alarms import runs  # noqa: E402
from cosmos.ai.data.preprocess_telemetry import ESA_PROTOCOL, load_esa_split, official_test_labels  # noqa: E402

PROT = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "oof_protocol")
H_NS = 3600 * 10 ** 9


def arm_metrics(g, ts, lab):
    t = pd.DatetimeIndex(ts).as_unit("ns").asi8
    n = t.size
    s, e = runs(g)
    ps = t[s]
    pe = np.where(e == n, t[-1], t[np.minimum(e, n - 1)])
    sel = lab[lab.Category.isin(["Anomaly", "Rare Event"])]
    gs, ge = XM._ns(sel.StartTime), XM._ns(sel.EndTime)
    ev = []
    for eid, grp in sel.assign(g_s=gs, g_e=ge).groupby("ID"):
        e0 = grp.g_s.min()
        best, nruns = None, set()
        for a, b in zip(grp.g_s, grp.g_e):
            j = np.flatnonzero((ps <= b) & (pe >= a))
            for jj in j:
                nruns.add(int(jj))
                o = max(ps[jj], a)
                best = o if best is None else min(best, o)
        ev.append({"ID": eid, "category": grp.Category.iloc[0], "detected": best is not None,
                   "delay_h": (best - e0) / H_NS if best is not None else np.nan, "n_runs": len(nruns)})
    ev = pd.DataFrame(ev)
    us, ue = XM._union(XM._ns(lab.StartTime), XM._ns(lab.EndTime))
    total = (max(int(ue.max()), int(t[-1])) - min(int(us.min()), int(t[0])))
    lab_ns = float((ue - us).sum())
    nominal_h = (total - lab_ns) / H_NS
    tab = XM.event_table(g, ts, lab)
    fp = int(tab["fp_count"].sum())
    det = ev[ev.detected]
    dur_h = (pe - ps) / H_NS
    q = lambda x, p: float(np.nanpercentile(x, p)) if len(x) else float("nan")  # noqa: E731
    return {"n_events": int(len(ev)), "detected": int(len(det)),
            "detected_anomalies": int(det.category.eq("Anomaly").sum()),
            "detected_rare": int(det.category.eq("Rare Event").sum()),
            "missed_anomalies": int((~ev.detected & ev.category.eq("Anomaly")).sum()),
            "missed_rare": int((~ev.detected & ev.category.eq("Rare Event")).sum()),
            "delay_h_median": q(det.delay_h, 50), "delay_h_q25": q(det.delay_h, 25), "delay_h_q75": q(det.delay_h, 75),
            "detected_within_1h": float((det.delay_h <= 1).mean()) if len(det) else float("nan"),
            "detected_within_24h": float((det.delay_h <= 24).mean()) if len(det) else float("nan"),
            "alarm_runs": int(len(ps)), "run_duration_h_median": q(dur_h, 50), "run_duration_h_q75": q(dur_h, 75),
            "runs_per_detected_event_median": q(det.n_runs, 50), "false_alarm_runs": fp,
            "false_alarms_per_1000_nominal_h": 1000 * fp / nominal_h, "nominal_hours": nominal_h}


def main():
    prov = provenance(__file__)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prereg_analysis as PA
    out = {**prov, "missions": {}}
    for m in (1, 2):
        sp = load_esa_split(m, "test")
        ts = sp.timestamps()
        lab = official_test_labels(m, sp)
        lab = lab[lab.Channel.isin(ESA_PROTOCOL[m]["target"])]
        tags = {k: [PA.tag_of(m, "target", s) for s in v] for k, v in PA.arms(m, "target").items()}
        tags["A0"] = [f"m{m}_target_ar8_calperiod"]
        res = {}
        for arm in ("A0", "A1", "A2", "A3"):
            per = []
            for t in tags[arm]:
                g = np.unpackbits(np.load(os.path.join(PROT, "alarms", f"{t}_conformal_bh_global.npy")))[:len(ts)]
                per.append({"tag": t, **arm_metrics(g.astype(bool), ts, lab)})
            df = pd.DataFrame(per)
            res[arm] = {"runs": per, "mean": {c: float(df[c].mean()) for c in df.columns if c != "tag"}}
            mm = res[arm]["mean"]
            print(f"M{m} {arm}: detected {mm['detected']:.1f}/{mm['n_events']:.0f}; median delay "
                  f"{mm['delay_h_median']:.1f} h; median run {mm['run_duration_h_median']:.2f} h; "
                  f"FA per 1000 nominal h {mm['false_alarms_per_1000_nominal_h']:.3f}", flush=True)
        out["missions"][m] = res
    save_json(out, "p7_operational", "summary.json")


if __name__ == "__main__":
    main()
