"""P4: point-level ACI with (alpha, gamma, k) selected on the EVENT objective by the frozen
cross-fit rule; gamma = 0 is a fixed per-channel conformal threshold.

    python scripts/posthoc_v2/p4_aci_fair.py --mission 1 --source ar8
    python scripts/posthoc_v2/p4_aci_fair.py --mission 1 --source event:<dir>
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import provenance, save_json  # noqa: E402

from cosmos.ai.telemetry import oof_benchmark as OB  # noqa: E402
from cosmos.ai.telemetry.alarms import _nominal_columns, attribute, persist_and_merge  # noqa: E402
from cosmos.ai.telemetry.esa_eval import false_alarm_stats, headline, period_labels, score_alarms  # noqa: E402
from cosmos.ai.telemetry.forecasting import MissionData  # noqa: E402
from cosmos.ai.data.preprocess_telemetry import official_test_labels  # noqa: E402

ALPHAS = [1e-2, 1e-3, 1e-4]
GAMMAS = [0.05, 0.01, 0.001, 1e-4, 0.0]
_G = {}


def _task(args):
    alpha, gamma, i = args
    S, L, halves, ts, lab, m, chans, merge = (_G[k] for k in ("S", "L", "halves", "ts", "lab", "m", "chans", "merge"))
    fit, ev = halves[1 - i], halves[i]
    pools = _nominal_columns(S[fit], L[fit])
    H = OB.aci_hot(np.asarray(S[ev], np.float32), pools, gamma, alpha)
    t = ts[ev]
    lb = lab[(lab.StartTime >= t[0]) & (lab.EndTime <= t[-1])]
    out = []
    for k in OB.KS:
        g = persist_and_merge(H.any(1), k, merge)
        ch = attribute(g, H, back=k - 1)
        s = score_alarms(m, t, ch, chans, lb, per_channel=False, categories=("anomaly_and_rare",))
        out.append({"alpha": alpha, "gamma": gamma, "k": k, "half": i,
                    "EW_F0.5": s.get("anomaly_and_rare/EW_F_0.50", 0.0), "alarm_fraction": float(g.mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--source", default="ar8")
    a = ap.parse_args()
    prov = provenance(__file__)
    m = a.mission
    D = MissionData(m, "target")
    S_tr, S_te, tag = OB.load_source(m, "target", a.source)
    S = np.asarray(S_tr, np.float32)
    n = S.shape[0]
    ts = D.train.timestamps()
    lab = period_labels(m, ts[0], ts[-1])
    lab = lab[lab.Channel.isin(D.channels)]
    merge = int(OB.MERGE_H * 3600 / D.train.rule_s)
    _G.update(S=S, L=D.Ltr, halves=[np.arange(0, n // 2), np.arange(n // 2, n)], ts=ts, lab=lab, m=m,
              chans=D.channels, merge=merge)
    tasks = [(al, ga, i) for al in ALPHAS for ga in GAMMAS for i in (0, 1)]
    with ProcessPoolExecutor(min(16, len(tasks))) as ex:
        rows = [r for part in ex.map(_task, tasks) for r in part]
    grid = (pd.DataFrame(rows).groupby(["alpha", "gamma", "k"], as_index=False)
            .agg(cv=("EW_F0.5", "mean"), alarm_fraction=("alarm_fraction", "mean")))
    ts_te = D.test.timestamps()
    lab_te = official_test_labels(m, D.test)
    lab_te = lab_te[lab_te.Channel.isin(D.channels)]
    pools = _nominal_columns(S, D.Ltr)
    res = {**prov, "mission": m, "source": a.source, "tag": tag, "grid": grid.to_dict("records"), "chosen": {}}
    # best overall, and best with gamma > 0 (adaptive ACI proper)
    for name, sub in (("any_gamma", grid), ("gamma_positive", grid[grid.gamma > 0]), ("gamma_zero", grid[grid.gamma == 0])):
        b = sub.sort_values(["cv", "alarm_fraction"], ascending=[False, True]).iloc[0]
        H = OB.aci_hot(np.asarray(S_te, np.float32), pools, float(b.gamma), float(b.alpha))
        g = persist_and_merge(H.any(1), int(b.k), merge)
        ch = attribute(g, H, back=int(b.k) - 1)
        sc = score_alarms(m, ts_te, ch, D.channels, lab_te, per_channel=False)
        row = {"alpha": float(b.alpha), "gamma": float(b.gamma), "k": int(b.k), "cv": float(b.cv),
               **{f"anomaly_and_rare/{k}": v for k, v in headline(sc, "anomaly_and_rare").items()},
               **{f"anomaly/{k}": v for k, v in headline(sc, "anomaly").items()},
               **false_alarm_stats(ts_te, g, lab_te)}
        res["chosen"][name] = row
        print(f"{tag} {name}: alpha {b.alpha:g} gamma {b.gamma:g} k {int(b.k)} cv {b.cv:.3f} -> test F0.5 "
              f"{row['anomaly_and_rare/EW_F0.5']:.4f}", flush=True)
    save_json(res, "p4_aci", f"{tag}.json")


if __name__ == "__main__":
    main()
