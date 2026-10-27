"""P3: check the exact block-decomposable metric against every stored official value, then
recompute the confirmatory comparisons H4-H6 with it (same blocks, seed and draw order).

    python scripts/posthoc_v2/p3_exact_bootstrap.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, provenance, save_json  # noqa: E402
import exact_metric as XM  # noqa: E402

from cosmos.ai.data.preprocess_telemetry import ESA_PROTOCOL, load_esa_split, official_test_labels  # noqa: E402

PROT = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "oof_protocol")
_CACHE = {}


def split_info(mission, channels):
    key = (mission, channels)
    if key not in _CACHE:
        sp = load_esa_split(mission, "test")
        lab = official_test_labels(mission, sp)
        chans = ESA_PROTOCOL[mission]["lightweight" if channels == "lightweight" else "target"]
        _CACHE[key] = (sp.timestamps(), lab[lab.Channel.isin(chans)])
    return _CACHE[key]


def alarms(tag, kind, n):
    return np.unpackbits(np.load(os.path.join(PROT, "alarms", f"{tag}_{kind}_global.npy")))[:n].astype(bool)


def main():
    prov = provenance(__file__)
    checks = []
    for f in sorted(glob.glob(os.path.join(PROT, "test", "*.json"))):
        tag = os.path.basename(f)[:-5]
        m = int(tag[1])
        ch = "lightweight" if "_lightweight_" in tag else "target"
        ts, lab = split_info(m, ch)
        for row in json.load(open(f)):
            kind = row["calibrator"]
            g = alarms(tag, kind, len(ts))
            for cats, key in ((("Anomaly", "Rare Event"), "anomaly_and_rare"), (("Anomaly",), "anomaly")):
                tab = XM.event_table(g, ts, lab, cats)
                ex, off = XM.f_beta(tab), float(row[f"{key}/EW_F0.5"])
                checks.append({"tag": tag, "calibrator": kind, "metric": key, "exact": ex, "official": off,
                               "abs_diff": abs(ex - off)})
    # A0 (calibration-window protocol): official value from the exploratory arm file
    for m in (1, 2):
        ts, lab = split_info(m, "target")
        row = [r for r in json.load(open(os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb",
                                                      "arms", f"m{m}_target_ar8_s0.json")))["rows"]
               if r["calibrator"] == "conformal_bh"][0]
        g = alarms(f"m{m}_target_ar8_calperiod", "conformal_bh", len(ts))
        ex = XM.f_beta(XM.event_table(g, ts, lab))
        checks.append({"tag": f"m{m}_target_ar8_calperiod", "calibrator": "conformal_bh",
                       "metric": "anomaly_and_rare", "exact": ex, "official": row["anomaly_and_rare/EW_F0.5"],
                       "abs_diff": abs(ex - row["anomaly_and_rare/EW_F0.5"])})
    maxd = max(c["abs_diff"] for c in checks)
    passed = bool(maxd < 1e-9)
    save_json({**prov, "n_checks": len(checks), "max_abs_diff": maxd, "passed": passed, "checks": checks},
              "p3_exact", "exactness.json")
    print(f"exactness: {len(checks)} checks, max |exact - official| = {maxd:.3g}, passed = {passed}")
    if not passed:
        bad = sorted(checks, key=lambda c: -c["abs_diff"])[:10]
        for c in bad:
            print("  ", c)
    # H4-H6 recomputed with the exact metric (frozen arms, draw order of oof_benchmark.compare)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prereg_analysis as PA  # frozen driver: arm and tag definitions only
    res = {}
    for m in (1, 2):
        ts, lab = split_info(m, "target")
        a = PA.arms(m, "target")
        tags = {k: [PA.tag_of(m, "target", s) for s in v] for k, v in a.items()}
        tags["A0"] = [f"m{m}_target_ar8_calperiod"]
        tabs = {k: [XM.event_table(alarms(t, "conformal_bh", len(ts)), ts, lab) for t in v]
                for k, v in tags.items() if k in ("A0", "A1", "A2", "A3")}
        rng = np.random.default_rng(20261024)
        rows = []
        for h, x, y in (("H4: A3 - A1", "A3", "A1"), ("H5: A3 - A2", "A3", "A2"), ("H6: A1 - A0", "A1", "A0")):
            r = XM.compare(tabs[x], tabs[y], rng)
            r.pop("replicates")
            rows.append({"comparison": h, **r, "n_seeds_A": len(tabs[x]), "n_seeds_B": len(tabs[y])})
        if m == 1:
            from cosmos.ai.evaluation import holm
            adj = holm({r["comparison"]: r["p_boot"] for r in rows})
            for r in rows:
                r["p_holm"] = adj[r["comparison"]]
                sup = r["p_holm"] < 0.05 and r["diff"] >= 0.05
                ref = r["ci95_high"] < 0.05
                r["verdict"] = "supported" if sup else ("refuted" if ref else "inconclusive")
        res[m] = rows
        for r in rows:
            print(f"M{m} {r['comparison']}: {r['diff']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] "
                  f"p {r['p_boot']:.4f} {r.get('verdict', '')}")
    save_json({**prov, "exactness_passed": passed, "missions": res}, "p3_exact", "confirmatory_exact.json")


if __name__ == "__main__":
    main()
