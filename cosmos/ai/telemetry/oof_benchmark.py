"""Alarm-layer selection on the whole TRAINING half (out-of-fold scores), then a gated,
single test pass.

    python -m cosmos.ai.telemetry.oof_benchmark select --mission 1 --source ar8
    python -m cosmos.ai.telemetry.oof_benchmark select --mission 1 --source event:<model dir>
    python -m cosmos.ai.telemetry.oof_benchmark test   --mission 1 --source ...   # after freeze

``select`` (development; reads training-half labels only)
    For each calibrator, cross-fit over the two halves of the training half.  Calibrate on
    the nominal OOF scores of one half, alarm on the other, score with the official corrected
    event-wise F0.5 on the labels of that half, and average both ways.  The best grid point
    is refitted on the whole training half.  On Mission 1 this selects on 105 labelled
    events instead of the 9 that :mod:`esa_benchmark` sees.

``test`` (confirmatory)
    Refuses to run unless ``research_design/preregistration/freeze.json`` exists and the
    SHA-256 of every file it lists is unchanged.  It then applies the selected calibrators
    to the test scores once and writes the official metrics.

``compare`` (confirmatory statistics)
    Paired moving-block bootstrap (30-day blocks, the same blocks for both arms) of the
    difference in corrected event-wise F0.5.  For seeded arms, seeds are resampled too
    (hierarchical bootstrap).  Holm correction is applied over the pre-registered family.

Score sources: ``ar8`` (OOF AR(8) non-conformity, :mod:`oof`), and
``event:<dir>`` / ``anomaly:<dir>`` (event model log-odds, :mod:`event_model`).

Point-level comparator (``aci``): per-channel adaptive conformal inference (Gibbs & Candes
2021), as in TC-CAD level C-3 (Sahin 2026).  The target is alpha = 0.01, and gamma is chosen
on the training half as the value whose empirical nominal false-alarm rate is closest to
alpha.  A sample is hot when it exceeds the channel's current threshold, and the global
alarm is the OR over channels.  The ``aci`` variant uses no persistence and no merging;
``aci_event`` adds this module's persistence and merging (the conformal_bh choice of k).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...common import output_dir
from .alarms import Calibrator, attribute, persist_and_merge
from .esa_eval import (false_alarm_stats, headline, official_test_labels, period_labels,
                       score_alarms)
from .forecasting import MissionData
from ..uncertainty import bh_reject, conformal_pvalues

#: pre-registered grids (research_design/preregistration/PREREG_TELEMETRY_V1.md, section 4)
LEVELS = {"threshold": [1e-3, 1e-4, 1e-5, 0.0],
          "conformal_bh": [1e-2, 1e-3, 1e-4, 1e-5],
          "conformal_bh_rolling": [1e-2, 1e-3, 1e-4, 1e-5]}
KS = [1, 3, 10, 30, 100]
MERGE_H = 1.0
CALIBRATORS = ("threshold", "conformal_bh", "conformal_bh_rolling")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
FREEZE = os.path.join(ROOT, "research_design", "preregistration", "freeze.json")


def result_dir(*parts) -> str:
    return output_dir("benchmarks", "telemetry_esa_adb", "oof_protocol", *parts)


def load_source(mission: int, channels: str, source: str):
    """``(S_train_oof, S_test, tag)`` for a score source."""
    if source == "ar8":
        from . import oof
        S_tr, S_te, _ = oof.load(mission, channels, "ar8")
        return S_tr, S_te, f"m{mission}_{channels}_ar8"
    kind, d = source.split(":", 1)
    S_tr = np.load(os.path.join(d, f"{kind}_train_oof.npy"), mmap_mode="r")
    S_te = np.load(os.path.join(d, f"{kind}_test.npy"), mmap_mode="r")
    return S_tr, S_te, f"{os.path.basename(d.rstrip('/'))}_{kind}"


# ---------------------------------------------------------------------- selection
_G = {}


def _score_task(args):
    """One (calibrator, level, half): hot mask once, then every k (forked worker)."""
    kind, lev, i = args
    S, L, halves, ts, lab, mission, channels, rule, merge = (
        _G[k] for k in ("S", "L", "halves", "ts", "lab", "mission", "channels", "rule",
                        "merge"))
    fit_rows, ev_rows = halves[1 - i], halves[i]
    base = "conformal_bh" if kind == "conformal_bh_rolling" else kind
    c = Calibrator(base, lev, 1, merge, rule_s=rule).fit(S[fit_rows], L[fit_rows])
    H = c.hot(np.asarray(S[ev_rows], np.float32))
    t = ts[ev_rows]
    lab_h = lab[(lab.StartTime >= t[0]) & (lab.EndTime <= t[-1])]
    out = []
    for k in KS:
        g = persist_and_merge(H.any(1), k, merge)
        ch = attribute(g, H, back=k - 1)
        s = score_alarms(mission, t, ch, channels, lab_h, per_channel=False,
                         categories=("anomaly_and_rare", "anomaly"))
        out.append({"kind": kind, "level": lev, "k": k, "half": i,
                    "EW_F0.5": s.get("anomaly_and_rare/EW_F_0.50", 0.0),
                    "anomaly_EW_F0.5": s.get("anomaly/EW_F_0.50", 0.0),
                    "alarm_fraction": float(g.mean())})
    return out


def select(mission: int, channels: str, source: str, objective: str = "EW_F0.5",
           workers: int = 24) -> dict:
    t0 = time.time()
    D = MissionData(mission, channels)
    S_tr, _S_te, tag = load_source(mission, channels, source)
    S = np.asarray(S_tr, np.float32)
    n = S.shape[0]
    ts = D.train.timestamps()
    lab = period_labels(mission, ts[0], ts[-1])
    lab = lab[lab.Channel.isin(D.channels)]
    rule = D.train.rule_s
    merge = int(MERGE_H * 3600 / rule)
    _G.update(S=S, L=D.Ltr, halves=[np.arange(0, n // 2), np.arange(n // 2, n)], ts=ts,
              lab=lab, mission=mission, channels=D.channels, rule=rule, merge=merge)
    tasks = [(kind, lev, i) for kind in CALIBRATORS if kind != "conformal_bh_rolling"
             for lev in LEVELS[kind] for i in (0, 1)]
    with ProcessPoolExecutor(min(workers, len(tasks))) as ex:
        rows = [r for part in ex.map(_score_task, tasks) for r in part]
    # the rolling calibrator differs from conformal_bh only on unlabelled TEST scores, so
    # on the training half it has the same cross-fitted score (it is selected identically)
    rows += [{**r, "kind": "conformal_bh_rolling"} for r in rows if r["kind"] == "conformal_bh"]
    import pandas as pd
    grid = (pd.DataFrame(rows).groupby(["kind", "level", "k"], as_index=False)
            .agg(cv=(objective, "mean"), cv_anomaly=("anomaly_EW_F0.5", "mean"),
                 alarm_fraction=("alarm_fraction", "mean")))
    chosen = {}
    for kind in CALIBRATORS:
        g = grid[grid.kind == kind].sort_values(["cv", "alarm_fraction"],
                                                ascending=[False, True])
        b = g.iloc[0]
        chosen[kind] = {"level": float(b.level), "k": int(b.k), "merge": merge,
                        "cv_EW_F0.5": float(b.cv), "cv_anomaly_EW_F0.5": float(b.cv_anomaly)}
    out = {"mission": mission, "channels": channels, "source": source, "tag": tag,
           "objective": objective, "chosen": chosen, "grid": grid.to_dict("records"),
           "seconds": time.time() - t0, "n_train_events": int(lab.ID.nunique())}
    with open(os.path.join(result_dir("selection"), f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    for kind, c in chosen.items():
        print(f"  {tag} {kind:22s} level {c['level']:g} k {c['k']:3d}: CV EW F0.5 "
              f"{c['cv_EW_F0.5']:.3f} (anomaly-only {c['cv_anomaly_EW_F0.5']:.3f})", flush=True)
    return out


# ---------------------------------------------------------------------- point-level comparator
ACI_ALPHA = 0.01
ACI_GAMMAS = [0.001, 0.005, 0.01, 0.05]


def _aci_kernel():
    import numba

    @numba.njit(cache=True)
    def aci(s, pool, alpha, gamma):
        n, m = s.size, pool.size
        hot = np.zeros(n, np.bool_)
        a = alpha
        for t in range(n):
            if a <= 0.0:
                thr = np.inf
            elif a >= 1.0:
                thr = -np.inf
            else:
                j = int(np.ceil((1.0 - a) * (m + 1))) - 1
                thr = np.inf if j >= m else pool[max(j, 0)]
            x = s[t]
            e = (x == x) and (x > thr)
            hot[t] = e
            a = a + gamma * (alpha - (1.0 if e else 0.0))
        return hot
    return aci


def aci_hot(S: np.ndarray, pools: list, gamma: float, alpha: float = ACI_ALPHA) -> np.ndarray:
    f = _aci_kernel()
    H = np.zeros(S.shape, bool)
    for c in range(S.shape[1]):
        H[:, c] = f(np.ascontiguousarray(S[:, c], np.float64),
                    np.asarray(pools[c], np.float64), alpha, gamma)
    return H


def aci_select_gamma(S_tr: np.ndarray, L_tr: np.ndarray) -> dict:
    """TC-CAD's rule: calibrate on the first half, run on the second, FAR closest to alpha."""
    from .alarms import _nominal_columns
    n = S_tr.shape[0]
    pools = _nominal_columns(S_tr[: n // 2], L_tr[: n // 2])
    S2, L2 = S_tr[n // 2:], L_tr[n // 2:]
    far = {}
    for g in ACI_GAMMAS:
        H = aci_hot(S2, pools, g)
        far[g] = float(H[L2 == 0].mean())
    best = min(ACI_GAMMAS, key=lambda g: abs(far[g] - ACI_ALPHA))
    return {"gamma": best, "far_by_gamma": far}


# ---------------------------------------------------------------------- confirmatory test
def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_freeze() -> dict:
    if not os.path.isfile(FREEZE):
        raise SystemExit("no pre-registration freeze file; the test half stays closed")
    with open(FREEZE) as f:
        fr = json.load(f)
    bad = [p for p, h in fr["sha256"].items() if _sha(os.path.join(ROOT, p)) != h]
    if bad:
        raise SystemExit(f"frozen files changed since the freeze: {bad}")
    return fr


def test(mission: int, channels: str, source: str) -> dict:
    fr = check_freeze()
    D = MissionData(mission, channels)
    S_tr, S_te, tag = load_source(mission, channels, source)
    with open(os.path.join(result_dir("selection"), f"{tag}.json")) as f:
        sel = json.load(f)
    ts = D.test.timestamps()
    lab = official_test_labels(mission, D.test)
    lab = lab[lab.Channel.isin(D.channels)]
    S_tr = np.asarray(S_tr, np.float32)
    S_te = np.asarray(S_te, np.float32)
    rows = []
    for kind, c in sel["chosen"].items():
        cal = Calibrator(kind, c["level"], c["k"], c["merge"],
                         rule_s=D.train.rule_s).fit(S_tr, D.Ltr)
        g, ch = cal.alarms(S_te)
        sc = score_alarms(mission, ts, ch, D.channels, lab, per_channel=True)
        row = {"tag": tag, "source": source, "calibrator": kind, **c,
               "freeze_id": fr["freeze_id"]}
        for key in ("anomaly_and_rare", "anomaly"):
            row.update({f"{key}/{k}": v for k, v in headline(sc, key).items()})
        row.update(false_alarm_stats(ts, g, lab))
        rows.append(row)
        np.save(os.path.join(result_dir("alarms"), f"{tag}_{kind}_global.npy"),
                np.packbits(g))
        print(f"  TEST {tag} {kind:22s}: EW F0.5 {row['anomaly_and_rare/EW_F0.5']:.3f} "
              f"(P {row['anomaly_and_rare/EW_precision']:.3f} R "
              f"{row['anomaly_and_rare/EW_recall']:.3f}); anomaly-only "
              f"{row['anomaly/EW_F0.5']:.3f}; false alarms/yr "
              f"{row['false_alarms_per_year']:.1f}", flush=True)
    from .alarms import _nominal_columns
    sel_g = aci_select_gamma(S_tr, D.Ltr)
    H = aci_hot(S_te, _nominal_columns(S_tr, D.Ltr), sel_g["gamma"])
    kb = sel["chosen"]["conformal_bh"]
    for kind, k, merge in (("aci", 1, 0), ("aci_event", kb["k"], kb["merge"])):
        g = persist_and_merge(H.any(1), k, merge) if merge else H.any(1)
        ch = attribute(g, H, back=k - 1)
        sc = score_alarms(mission, ts, ch, D.channels, lab, per_channel=False)
        row = {"tag": tag, "source": source, "calibrator": kind, "level": ACI_ALPHA,
               "gamma": sel_g["gamma"], "far_by_gamma": sel_g["far_by_gamma"], "k": k,
               "merge": merge, "freeze_id": fr["freeze_id"],
               "test_point_far_nominal": float(H[D.Lte == 0].mean())}
        for key in ("anomaly_and_rare", "anomaly"):
            row.update({f"{key}/{kk}": v for kk, v in headline(sc, key).items()})
        row.update(false_alarm_stats(ts, g, lab))
        rows.append(row)
        np.save(os.path.join(result_dir("alarms"), f"{tag}_{kind}_global.npy"),
                np.packbits(g))
        print(f"  TEST {tag} {kind:22s}: EW F0.5 {row['anomaly_and_rare/EW_F0.5']:.4f}; "
              f"point FAR {row['test_point_far_nominal']:.4f}; alarm runs "
              f"{row['alarm_runs']}", flush=True)
    with open(os.path.join(result_dir("test"), f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return {"rows": rows}


def calperiod_alarms(mission: int, channels: str = "target", calibrator: str = "conformal_bh"):
    """Arm A0: the AR(8) arm of :mod:`esa_benchmark` (selection on the 9-month calibration
    period), its global test alarms regenerated from the saved scores and saved choice."""
    import glob
    from ...common import output_dir as od
    d = sorted(glob.glob(os.path.join(od("experiments"),
                                      f"*_esa_m{mission}_{channels}_ar8_s0_s0")))[-1]
    tag = f"m{mission}_{channels}_ar8_s0"
    with open(os.path.join(od("benchmarks", "telemetry_esa_adb", "arms"), f"{tag}.json")) as f:
        row = [r for r in json.load(f)["rows"] if r["calibrator"] == calibrator][0]
    D = MissionData(mission, channels)
    a, b = D.p["cal"]
    cal = Calibrator(calibrator, row["selected_level"], int(row["selected_k"]),
                     int(row["merge_steps"]), rule_s=D.train.rule_s).fit(
        np.load(os.path.join(d, "scores_cal.npy")), D.Ltr[a:b])
    g, _ = cal.alarms(np.load(os.path.join(d, "scores_test.npy")))
    out = f"m{mission}_{channels}_ar8_calperiod"
    np.save(os.path.join(result_dir("alarms"), f"{out}_{calibrator}_global.npy"), np.packbits(g))
    return out, row["anomaly_and_rare/EW_F0.5"]


# ---------------------------------------------------------------------- statistics
def _arm_tables(mission: int, channels: str, tags: list, calibrator: str):
    import pandas as pd
    from .event_bootstrap import _blockify, _event_table
    D = MissionData(mission, channels)
    ts = D.test.timestamps()
    lab = official_test_labels(mission, D.test)
    lab = lab[lab.Channel.isin(D.channels)]
    out = []
    for t in tags:
        g = np.unpackbits(np.load(os.path.join(result_dir("alarms"),
                                               f"{t}_{calibrator}_global.npy")))[:len(ts)]
        out.append(_blockify(_event_table(mission, g.astype(bool), pd.DatetimeIndex(ts), lab)))
    return out


def compare(mission: int, comparisons: list, channels: str = "target", n_boot: int = 2000,
            seed: int = 20261024) -> list:
    """``comparisons``: ``(name, tags_A, tags_B, calibrator)``; returns one row each."""
    from .event_bootstrap import _ewf, _sample
    rng = np.random.default_rng(seed)
    rows = []
    for name, tags_a, tags_b, cal in comparisons:
        ta = _arm_tables(mission, channels, tags_a, cal)
        tb = _arm_tables(mission, channels, tags_b, cal)
        nb = ta[0]["nb"]
        full = np.arange(nb)
        pa = np.mean([_ewf(t, _sample(t, full)) for t in ta])
        pb = np.mean([_ewf(t, _sample(t, full)) for t in tb])
        d = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, nb, nb)
            sa = rng.integers(0, len(ta), len(ta))
            sb = rng.integers(0, len(tb), len(tb))
            d[i] = (np.mean([_ewf(ta[j], _sample(ta[j], idx)) for j in sa])
                    - np.mean([_ewf(tb[j], _sample(tb[j], idx)) for j in sb]))
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / n_boot)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"comparison": name, "mission": mission, "calibrator": cal,
                     "A": tags_a, "B": tags_b, "A_EW_F0.5": pa, "B_EW_F0.5": pb,
                     "diff": pa - pb, "ci95_low": float(lo), "ci95_high": float(hi),
                     "p_boot": float(p), "n_boot": n_boot, "block_days": 30})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("select", "test"))
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--channels", default="target")
    ap.add_argument("--source", default="ar8")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    if a.action == "select":
        select(a.mission, a.channels, a.source, workers=a.workers)
    else:
        test(a.mission, a.channels, a.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
