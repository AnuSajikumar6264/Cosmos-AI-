"""Post hoc (EXPLORATORY, not pre-registered): which labelled test events does each arm detect?

    python scripts/event_class_analysis.py --mission 1

For every arm's saved global test alarms (conformal_bh), each labelled test event (anomaly or
rare event, target channels) counts as detected if an alarm overlaps it.  Recall is broken
down by event class and by whether that class occurs in the training half.  This asks
whether a supervised scorer detects only the classes it was trained on.  Output:
``outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_m{m}.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cosmos.ai.telemetry import oof_benchmark as OB  # noqa: E402
from cosmos.ai.telemetry.forecasting import MissionData  # noqa: E402
from cosmos.ai.data.preprocess_telemetry import processed_dir  # noqa: E402
from scripts.prereg_analysis import arms, tag_of  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--calibrator", default="conformal_bh")
    a = ap.parse_args(argv)
    m = a.mission
    D = MissionData(m)
    t = pd.DatetimeIndex(D.test.timestamps()).as_unit("ns").asi8
    lab = pd.read_csv(os.path.join(processed_dir(m), "labels_with_types.csv"),
                      parse_dates=["StartTime", "EndTime"])
    lab = lab[lab.Channel.isin(D.channels) & lab.Category.isin(["Anomaly", "Rare Event"])]
    split = pd.Timestamp(D.test.timestamps()[0])
    train_classes = set(lab[lab.StartTime < split].Class)
    ev = (lab[lab.StartTime >= split].groupby("ID")
          .agg(s=("StartTime", "min"), e=("EndTime", "max"), cls=("Class", "first"),
               cat=("Category", "first")))
    ev["seen_class"] = ev.cls.isin(train_classes)
    i0 = np.searchsorted(t, ev.s.to_numpy().astype("datetime64[ns]").astype(np.int64))
    i1 = np.searchsorted(t, ev.e.to_numpy().astype("datetime64[ns]").astype(np.int64), "right")
    tags = {k: [tag_of(m, "target", s) for s in v] for k, v in arms(m).items()}
    tags["A0"] = [f"m{m}_target_ar8_calperiod"]
    for arm, ts in tags.items():
        det = []
        for tg in ts:
            p = os.path.join(OB.result_dir("alarms"), f"{tg}_{a.calibrator}_global.npy")
            if not os.path.isfile(p):
                continue
            g = np.unpackbits(np.load(p))[:len(t)].astype(np.int64)
            cs = np.r_[0, np.cumsum(g)]
            det.append((cs[np.minimum(i1, len(t))] - cs[i0]) > 0)
        if det:
            ev[f"{arm}_detect_rate"] = np.mean(det, axis=0)
    cols = [c for c in ev if c.endswith("_detect_rate")]
    by = ev.groupby(["seen_class", "cat"])[cols].mean().round(3)
    by["n_events"] = ev.groupby(["seen_class", "cat"]).size()
    out = OB.result_dir("exploratory")
    ev.to_csv(os.path.join(out, f"event_classes_m{m}.csv"))
    by.to_csv(os.path.join(out, f"event_classes_summary_m{m}.csv"))
    print("EXPLORATORY (post hoc): mean detection rate by seen/unseen class")
    print(by.to_string())


if __name__ == "__main__":
    main()
