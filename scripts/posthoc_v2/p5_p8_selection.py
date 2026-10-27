"""P5 (registered anomaly-score arm) and P8 (provenance of the A1 selection files), both with the
frozen ``oof_benchmark.select``, its output redirected to the post-hoc directory.

    python scripts/posthoc_v2/p5_p8_selection.py p8
    python scripts/posthoc_v2/p5_p8_selection.py p5 --mission 1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, ROOT, provenance, save_json  # noqa: E402

from cosmos.ai.telemetry import oof_benchmark as OB  # noqa: E402
from cosmos.ai.telemetry.alarms import Calibrator  # noqa: E402
from cosmos.ai.telemetry.esa_eval import false_alarm_stats, headline, score_alarms  # noqa: E402
from cosmos.ai.telemetry.forecasting import MissionData  # noqa: E402
from cosmos.ai.data.preprocess_telemetry import official_test_labels  # noqa: E402

FROZEN_SEL = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "oof_protocol", "selection")


def redirect(sub):
    base = os.path.join(OUT, sub)

    def rd(*parts):
        d = os.path.join(base, *parts)
        os.makedirs(d, exist_ok=True)
        return d
    OB.result_dir = rd
    return base


def p8():
    prov = provenance(__file__)
    redirect("p8_provenance")
    res = {**prov, "checks": []}
    for m in (1, 2):
        new = OB.select(m, "target", "ar8")
        old = json.load(open(os.path.join(FROZEN_SEL, f"m{m}_target_ar8.json")))
        same_choice = all(new["chosen"][k]["level"] == old["chosen"][k]["level"] and new["chosen"][k]["k"] == old["chosen"][k]["k"]
                          for k in old["chosen"])
        key = lambda r: (r["kind"], float(r["level"]), int(r["k"]))  # noqa: E731
        om = {key(r): r["cv"] for r in old["grid"]}
        diffs = [abs(r["cv"] - om[key(r)]) for r in new["grid"]]
        res["checks"].append({"mission": m, "same_choice": same_choice, "n_grid": len(diffs),
                              "max_abs_cv_diff": float(max(diffs)), "passed": bool(same_choice and max(diffs) < 1e-9)})
        print(res["checks"][-1], flush=True)
    save_json(res, "p8_provenance", "summary.json")


def p5(m):
    prov = provenance(__file__)
    redirect("p5_anomaly_arm")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prereg_analysis as PA
    arms = PA.arms(m, "target")
    sources = {"A1": ["ar8"], "A2": [s.replace("event:", "anomaly:") for s in arms.get("A2", [])],
               "A3": [s.replace("event:", "anomaly:") for s in arms.get("A3", [])]}
    D = MissionData(m, "target")
    ts = D.test.timestamps()
    lab = official_test_labels(m, D.test)
    lab = lab[lab.Channel.isin(D.channels)]
    out = {**prov, "mission": m, "arms": {}}
    for arm, srcs in sources.items():
        rows = []
        for src in srcs:
            sel = OB.select(m, "target", src, objective="anomaly_EW_F0.5")
            S_tr, S_te, tag = OB.load_source(m, "target", src)
            c = sel["chosen"]["conformal_bh"]
            cal = Calibrator("conformal_bh", c["level"], c["k"], c["merge"], rule_s=D.train.rule_s).fit(
                np.asarray(S_tr, np.float32), D.Ltr)
            g, ch = cal.alarms(np.asarray(S_te, np.float32))
            sc = score_alarms(m, ts, ch, D.channels, lab, per_channel=False)
            row = {"source": src, "tag": tag, "level": c["level"], "k": c["k"],
                   "cv_anomaly_EW_F0.5": c["cv_anomaly_EW_F0.5"],
                   **{f"anomaly/{k}": v for k, v in headline(sc, "anomaly").items()},
                   **{f"anomaly_and_rare/{k}": v for k, v in headline(sc, "anomaly_and_rare").items()},
                   **false_alarm_stats(ts, g, lab)}
            rows.append(row)
            print(f"M{m} {arm} {tag}: q {c['level']:g} k {c['k']}: anomaly-only F0.5 {row['anomaly/EW_F0.5']:.4f}",
                  flush=True)
        vals = [r["anomaly/EW_F0.5"] for r in rows]
        out["arms"][arm] = {"runs": rows, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            "min": float(np.min(vals)), "max": float(np.max(vals))}
    save_json(out, "p5_anomaly_arm", f"m{m}_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("p5", "p8"))
    ap.add_argument("--mission", type=int, default=1)
    a = ap.parse_args()
    p8() if a.what == "p8" else p5(a.mission)


if __name__ == "__main__":
    main()
