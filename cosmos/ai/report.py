"""Aggregate every COSMOS-AI benchmark into tables, statistical comparisons and figures.

    python -m cosmos.ai.report

Writes ``outputs/benchmarks/SUMMARY.md`` (the component table), per-study CSV/JSON tables,
and comparison figures (``outputs/benchmarks/figures/``).  Only numbers present in the
benchmark output files are used; anything missing is reported as "not run".

ESA-ADB confidence intervals: the official metrics are too slow to recompute thousands of
times, so the moving-block bootstrap (30-day blocks over the test half) recomputes the
corrected event-wise F0.5 from an event table regenerated from each arm's saved scores and
selected calibrator.  The full-sample value of that re-implementation is reported next to
the official value so its agreement can be checked.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from ..common import output_dir

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated, fixed order

#: published DMLR 2026 (Table 4) corrected event-wise F0.5, anomalies + rare events
DMLR_BEST = {(1, "target"): ("Telemanom-ESA-Pruned", 0.061), (2, "target"): ("PCC", 0.079),
             (1, "lightweight"): ("Telemanom-ESA-Pruned", 0.786),
             (2, "lightweight"): ("Windowed iForest", 0.949)}


def _bench(*p):
    return os.path.join(output_dir("benchmarks"), *p)


def _json(p):
    return json.load(open(p)) if os.path.isfile(p) else None


# ============================================================================ figures
def _style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def bar_panel(ax, labels, values, errors, families, family_order, xlabel, title,
              ref=None, fmt="{:.3f}", xlim=None):
    """Horizontal thin bars, value at the tip, colour = family (fixed slot order)."""
    y = np.arange(len(labels))[::-1]
    col = [SLOTS[family_order.index(f)] for f in families]
    ax.barh(y, values, height=0.55, color=col, edgecolor=SURFACE, linewidth=2)
    if errors is not None:
        ax.errorbar(values, y, xerr=np.nan_to_num(errors), fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=2)
    top = xlim[1] if xlim else max(1e-9, np.nanmax(np.asarray(values) + np.nan_to_num(errors if errors is not None else 0)))
    for yi, v, e in zip(y, values, errors if errors is not None else [0] * len(values)):
        ax.text(v + (0 if np.isnan(e) else e) + 0.01 * top, yi, fmt.format(v), va="center",
                fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    if ref is not None:
        ax.axvline(ref[1], color=MUTED, linewidth=1)
        ax.text(ref[1], len(labels) - 0.35, f" {ref[0]}", color=INK2, fontsize=8, va="bottom")
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, top * 1.18)
    _style(ax, xlabel)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def _legend(fig, family_order, used):
    from matplotlib.patches import Patch
    hs = [Patch(color=SLOTS[family_order.index(f)], label=f) for f in family_order if f in used]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))


def _save(fig, name):
    d = output_dir("benchmarks", "figures")
    fig.patch.set_facecolor(SURFACE)
    p = os.path.join(d, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


# ============================================================================ ESA-ADB
def esa_table() -> pd.DataFrame:
    from .telemetry.esa_benchmark import collect
    df = collect()
    if df.empty:
        return df
    df["encoder"] = df.encoder.where(~df.telecommands.fillna(False).astype(bool),
                                     df.encoder + "+tc")
    keys = ["mission", "channels", "encoder", "calibrator"]
    met = ["anomaly_and_rare/EW_F0.5", "anomaly_and_rare/EW_precision",
           "anomaly_and_rare/EW_recall", "anomaly_and_rare/alarming_precision",
           "anomaly_and_rare/channel_F0.5", "anomaly_and_rare/subsystem_F0.5",
           "anomaly_and_rare/ADTQC", "anomaly_and_rare/AFF_F0.5", "anomaly/EW_F0.5",
           "anomaly/EW_recall", "false_alarms_per_year", "rare_event_only_runs", "anomaly_runs"]
    met = [m for m in met if m in df]
    g = df.groupby(keys)
    t = g[met].agg(["mean", "std"])
    t.columns = [f"{a}|{b}" for a, b in t.columns]
    t["n_seeds"] = g.size()
    for c in ("params", "train_time_s", "inference_s_per_telemetry_day", "selected_level",
              "selected_k"):
        if c in df:
            t[c] = g[c].mean() if c not in ("selected_level", "selected_k") else g[c].agg(
                lambda v: ",".join(sorted({f"{x:g}" for x in v})))
    t = t.reset_index()
    t.to_csv(_bench("telemetry_esa_adb", "summary_table.csv"), index=False)
    return t


def _arm_dirs(mission, encoder, tc, seed=0, channels="target"):
    tag = f"esa_m{mission}_{channels}_{encoder}{'_tc' if tc else ''}_s{seed}"
    ds = sorted(glob.glob(os.path.join(output_dir("experiments"), f"*_{tag}_s{seed}")))
    return ds[-1] if ds else None


from .telemetry.event_bootstrap import _blockify, _event_table, _ewf, _sample  # noqa: E402


def esa_bootstrap(pairs, n_boot=1000, seed=0) -> list:
    """Paired moving-block bootstrap of EW F0.5 differences for (arm A, arm B) pairs."""
    from .data.preprocess_telemetry import load_esa_split, official_test_labels
    from .telemetry.alarms import Calibrator
    from .telemetry.forecasting import MissionData
    out, cache = [], {}

    def arm_table(mission, encoder, tc, calibrator, s):
        key = (mission, encoder, tc, calibrator, s)
        if key in cache:
            return cache[key]
        d = _arm_dirs(mission, encoder, tc, s)
        if d is None or not os.path.isfile(os.path.join(d, "scores_test.npy")):
            return None
        tag = f"m{mission}_target_{encoder}{'_tc' if tc else ''}_s{s}"
        arm = _json(_bench("telemetry_esa_adb", "arms", f"{tag}.json"))
        row = [r for r in arm["rows"] if r["calibrator"] == calibrator][0]
        if "D" not in cache.get(mission, {}):
            cache[mission] = {"D": MissionData(mission)}
        D = cache[mission]["D"]
        S_cal = np.load(os.path.join(d, "scores_cal.npy"))
        S_te = np.load(os.path.join(d, "scores_test.npy"))
        a, b = D.p["cal"]
        cal = Calibrator(calibrator, row["selected_level"], int(row["selected_k"]),
                         int(row["merge_steps"]), rule_s=D.train.rule_s).fit(
            S_cal, D.Ltr[a:b])
        g, _ = cal.alarms(S_te)
        lab = official_test_labels(mission, D.test)
        lab = lab[lab.Channel.isin(D.channels)]
        tab = _blockify(_event_table(mission, g, D.test.timestamps(), lab))
        tab["official"] = row["anomaly_and_rare/EW_F0.5"]
        cache[key] = tab
        return tab

    rng = np.random.default_rng(seed)
    for name, A, B in pairs:
        ta, tb = arm_table(*A), arm_table(*B)
        if ta is None or tb is None:
            out.append({"comparison": name, "status": "not run"})
            continue
        nb = ta["nb"]
        full = np.arange(nb)
        fa, fb = _ewf(ta, _sample(ta, full)), _ewf(tb, _sample(tb, full))
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, nb, nb)
            diffs.append(_ewf(ta, _sample(ta, idx)) - _ewf(tb, _sample(tb, idx)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.append({"comparison": name, "A": "/".join(map(str, A)), "B": "/".join(map(str, B)),
                    "A_EW_F0.5_official": ta["official"], "B_EW_F0.5_official": tb["official"],
                    "A_EW_F0.5_reimpl": fa, "B_EW_F0.5_reimpl": fb, "diff_reimpl": fa - fb,
                    "ci95_low": float(lo), "ci95_high": float(hi), "block_days": 30,
                    "n_boot": n_boot})
    return out


def esa_figures(t: pd.DataFrame):
    import matplotlib.pyplot as plt
    fam = ["per-channel threshold", "conformal + BH", "conformal + BH, rolling",
           "native (as published / as shipped)"]
    calname = {"threshold": fam[0], "conformal_bh": fam[1], "conformal_bh_rolling": fam[2],
               "native": fam[3]}
    order = ["global_std3", "global_std5", "cosmos_monitor", "persistence", "ar8", "tcn",
             "tcn+tc", "patchtst", "patchtst+tc", "chronos-bolt-small"]
    nice = {"global_std3": "GlobalSTD3 (official)", "global_std5": "GlobalSTD5 (official)",
            "cosmos_monitor": "COSMOS monitor", "persistence": "persistence",
            "ar8": "COSMOS AR(8)", "tcn": "TCN", "tcn+tc": "TCN + telecommands",
            "patchtst": "PatchTST", "patchtst+tc": "PatchTST + telecommands",
            "chronos-bolt-small": "Chronos-Bolt (zero-shot)"}
    for ch in ("target", "lightweight"):
        sub = t[t.channels == ch]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * max(len(sub) // 2, 8) + 1.5),
                                 sharey=False)
        for ax, mission in zip(axes, (1, 2)):
            s = sub[(sub.mission == mission)].copy()
            s = s[s.encoder.isin(order)]
            s["o"] = s.encoder.map(order.index) * 10 + s.calibrator.map(
                {"native": 0, "threshold": 1, "conformal_bh": 2, "conformal_bh_rolling": 3})
            s = s.sort_values("o")
            labels = [f"{nice[e]} · {c.replace('conformal_bh', 'conf-BH').replace('_rolling', ' roll')}"
                      if c != "native" else nice[e] for e, c in zip(s.encoder, s.calibrator)]
            bar_panel(ax, labels, s["anomaly_and_rare/EW_F0.5|mean"].to_numpy(),
                      s["anomaly_and_rare/EW_F0.5|std"].to_numpy(),
                      [calname[c] for c in s.calibrator], fam,
                      "corrected event-wise F0.5 (anomalies + rare events)",
                      f"ESA-ADB Mission {mission} — {'full' if ch == 'target' else 'lightweight'} "
                      "channel set, test half", ref=(f"published best: {DMLR_BEST[(mission, ch)][0]}"
                                                     f" {DMLR_BEST[(mission, ch)][1]}",
                                                     DMLR_BEST[(mission, ch)][1]),
                      xlim=(0, 1.12))
        _legend(fig, fam, set(calname[c] for c in sub.calibrator))
        ns = sorted(set(sub.n_seeds.astype(int)))
        fig.text(0.01, -0.04, f"Bars: mean over seeds; whiskers: SD where >1 seed (seeds per "
                 f"arm: {ns}; see summary_table.csv). Vertical line: best published value "
                 "(Kotowski et al., DMLR 2026, Table 4).", fontsize=8, color=INK2)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        _save(fig, f"esa_adb_{ch}_ew_f05.png")


# ============================================================================ OPSSAT-AD
def opssat_tables():
    rows = []
    for tag in ("official", "time_blocked"):
        p = _bench("telemetry_opssat", f"summary_{tag}.csv")
        if os.path.isfile(p):
            rows.append(pd.read_csv(p))
    if not rows:
        return None, None
    s = pd.concat(rows)
    z = np.load(_bench("telemetry_opssat", "test_scores_official.npz"))
    y = z["official|y"]
    from sklearn.metrics import average_precision_score, f1_score

    def mean_score(model):
        ks = [k for k in z.files if k.startswith(f"official|1.0|{model}|")]
        return np.mean([z[k] for k in ks], 0) if ks else None

    comps = []
    from .evaluation import paired_bootstrap_diff
    for a, b in (("hybrid", "fcnn"), ("hybrid", "xgboost"), ("fcnn", "xgboost"),
                 ("hybrid", "chronos:amazon/chronos-bolt-small"), ("cnn", "fcnn")):
        pa, pb = mean_score(a), mean_score(b)
        if pa is None or pb is None:
            continue
        r = paired_bootstrap_diff(lambda yy, p: average_precision_score(yy, p), y, pa, pb)
        r2 = paired_bootstrap_diff(lambda yy, p: f1_score(yy, (p >= 0.5).astype(int)), y,
                                   pa, pb)
        comps.append({"A": a, "B": b, "metric": "AUC-PR (seed-averaged scores)", **r})
        comps.append({"A": a, "B": b, "metric": "F1 @ 0.5 (seed-averaged scores)", **r2})
    c = pd.DataFrame(comps)
    c.to_csv(_bench("telemetry_opssat", "paired_comparisons.csv"), index=False)
    return s, c


def opssat_figures(s):
    import matplotlib.pyplot as plt
    fam = ["existing COSMOS / unsupervised", "conventional ML", "deep learning",
           "foundation model (frozen)"]
    family = {"cosmos_robust_z": fam[0], "iforest": fam[0], "adaboost": fam[1],
              "logreg": fam[1], "random_forest": fam[1], "xgboost": fam[1], "fcnn": fam[2],
              "cnn": fam[2], "transformer": fam[2], "hybrid": fam[2],
              "chronos:amazon/chronos-bolt-small": fam[3]}
    nice = {"cosmos_robust_z": "COSMOS robust-z", "iforest": "IForest (official example)",
            "adaboost": "AdaBoost (official example)", "logreg": "logistic regression",
            "random_forest": "random forest", "xgboost": "XGBoost", "fcnn": "FCNN (features)",
            "cnn": "1D-CNN (raw)", "transformer": "patch transformer (raw)",
            "hybrid": "hybrid CNN + features", "chronos:amazon/chronos-bolt-small":
            "Chronos-Bolt embedding + probe"}
    for proto in ("official", "time_blocked"):
        d = s[(s.protocol == proto) & (s.train_frac == 1.0)].sort_values("auc_pr_mean",
                                                                         ascending=False)
        if d.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, m, lab in zip(axes, ("f1", "auc_pr"), ("F1 (P >= 0.5)", "AUC-PR")):
            bar_panel(ax, [nice[x] for x in d.model], d[f"{m}_mean"].to_numpy(),
                      d[f"{m}_std"].to_numpy(), [family[x] for x in d.model], fam, lab,
                      f"OPSSAT-AD — {lab}, {proto.replace('_', '-')} split",
                      ref=("published FCNN 0.946" if m == "f1" else "published FCNN 0.979",
                           0.946 if m == "f1" else 0.979) if proto == "official" else None,
                      xlim=(0, 1.12))
        _legend(fig, fam, set(family[x] for x in d.model))
        fig.text(0.01, -0.03, "Mean over 5 seeds (deterministic models: 1 run); whiskers SD. "
                 "Published: Ruszczak et al., Sci. Data 2025, Table 3.", fontsize=8, color=INK2)
        plt.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, f"opssat_{proto}.png")
    # label efficiency (line chart, 5 series, legend + end labels)
    d = s[s.protocol == "official"]
    models = ["xgboost", "fcnn", "cnn", "hybrid", "chronos:amazon/chronos-bolt-small"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, m in enumerate(models):
        r = d[d.model == m].sort_values("train_frac")
        if r.empty:
            continue
        ax.plot(r.train_frac * 100, r.auc_pr_mean, color=SLOTS[k], linewidth=2, marker="o",
                markersize=5, label=nice[m], markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(101.5, r.auc_pr_mean.iloc[-1], f"{r.auc_pr_mean.iloc[-1]:.3f}", fontsize=8,
                color=INK, va="center")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], ["10 %", "25 %", "50 %", "100 %"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("fraction of the official training fragments", color=INK2, fontsize=9)
    ax.set_ylabel("test AUC-PR (mean of 5 seeds)", color=INK2, fontsize=9)
    ax.set_title("OPSSAT-AD label efficiency", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    _save(fig, "opssat_label_efficiency.png")


# ============================================================================ vision
def vision_tables():
    syn = [json.load(open(p)) for p in glob.glob(_bench("vision_cosmos_synthetic", "runs", "*.json"))]
    sg = [json.load(open(p)) for p in glob.glob(_bench("vision_spotgeo", "runs", "*.json"))]
    out = {}
    if syn:
        d = pd.DataFrame(syn)
        cols = ["map50", "map50_95", "small_map50", "map50_class_agnostic", "precision@conf",
                "recall@conf", "f1@conf", "mean_iou@conf", "latency_ms_per_item", "params",
                "train_time_s"]
        cols = [c for c in cols if c in d]
        g = d.groupby("model")[cols].agg(["mean", "std"])
        g.columns = [f"{a}|{b}" for a, b in g.columns]
        g["n_seeds"] = d.groupby("model").size()
        g.reset_index().to_csv(_bench("vision_cosmos_synthetic", "summary.csv"), index=False)
        out["synthetic"] = g.reset_index()
    rows = []
    cl = _json(_bench("vision_spotgeo", "classical_cosmos_frontend.json"))
    if cl:
        rows.append({"model": "classical COSMOS front end + DoG", "inputs": "diff", "seed": 0,
                     "train_frac": 1.0, **{f"test_{k}": v for k, v in cl["test"].items()}})
    rows += sg
    if rows:
        d = pd.DataFrame(rows)
        cols = [c for c in ("test_f1", "test_one_minus_f1", "test_precision", "test_recall",
                            "test_official_mse_mean_per_sequence", "val_f1",
                            "latency_ms_per_sequence", "params", "train_time_s") if c in d]
        g = d.groupby(["model", "inputs", "train_frac"])[cols].agg(["mean", "std"])
        g.columns = [f"{a}|{b}" for a, b in g.columns]
        g["n_seeds"] = d.groupby(["model", "inputs", "train_frac"]).size()
        g.reset_index().to_csv(_bench("vision_spotgeo", "summary.csv"), index=False)
        out["spotgeo"] = g.reset_index()
    return out


def vision_figures(v):
    import matplotlib.pyplot as plt
    if "synthetic" in v:
        d = v["synthetic"].sort_values("map50_95|mean", ascending=False)
        fam = ["existing COSMOS CNN", "YOLO family", "COSMOS-AI heat-map"]
        famof = lambda m: fam[0] if "tinyyolo" in m else fam[1] if "yolo" in m else fam[2]
        fig, axes = plt.subplots(1, 2, figsize=(13, 2.6))
        for ax, m, lab in zip(axes, ("map50", "map50_95"), ("mAP@50", "mAP@50-95")):
            bar_panel(ax, list(d.model), d[f"{m}|mean"].to_numpy(), d[f"{m}|std"].to_numpy(),
                      [famof(x) for x in d.model], fam, f"test {lab} (COCO-style, 4 classes)",
                      f"COSMOS synthetic frame pairs — {lab}", xlim=(0, 1.15))
        _legend(fig, fam, set(famof(x) for x in d.model))
        fig.text(0.01, -0.08, "SYNTHETIC data only. Mean over 3 seeds, whiskers SD; scene-level "
                 "split (200 test scenes).", fontsize=8, color=INK2)
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        _save(fig, "vision_cosmos_synthetic.png")
    if "spotgeo" in v:
        d = v["spotgeo"][v["spotgeo"].train_frac == 1.0].sort_values("test_f1|mean",
                                                                     ascending=False)
        fam = ["existing COSMOS front end", "COSMOS-AI heat-map (scratch)",
               "COSMOS-AI heat-map (pretrained encoder)"]
        famof = lambda m: fam[0] if "classical" in m else fam[2] if "pretrained" in m else fam[1]
        mname = {"unet": "heat-map U-Net", "resnet18_pretrained": "U-Net, ImageNet ResNet-18 encoder",
                 "resnet18_scratch": "U-Net, ResNet-18 encoder (scratch)"}
        iname = {"both": "difference + high-pass", "raw": "high-pass only (no registration)",
                 "diff": "registered difference only"}
        labels = [m if m.startswith("classical") else f"{mname.get(m, m)} · {iname.get(i, i)}"
                  for m, i in zip(d.model, d.inputs)]
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.6))
        bar_panel(ax, labels, d["test_f1|mean"].to_numpy(), d["test_f1|std"].to_numpy(),
                  [famof(x) for x in d.model], fam, "official SpotGEO F1 (test, 5,120 sequences)",
                  "SpotGEO (real, unresolved GEO targets)",
                  ref=("challenge winner 0.948", 1 - 0.0517), xlim=(0, 1.15))
        _legend(fig, fam, set(famof(x) for x in d.model))
        ns = ", ".join(f"{lab.split(' · ')[0]}{' · ' + lab.split(' · ')[1] if ' · ' in lab else ''}: {int(n)}"
                       for lab, n in zip(labels, d.n_seeds))
        fig.text(0.01, -0.06, "Official validation.py scoring (tau 10 px). Seeds per bar — " + ns
                 + ". Whiskers: SD where >1 seed.", fontsize=7, color=INK2, wrap=True)
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        _save(fig, "vision_spotgeo.png")


# ============================================================================ re-entry
def reentry_figures():
    import matplotlib.pyplot as plt
    r = _json(_bench("reentry_landmark", "results.json"))
    if not r:
        return
    fam = ["existing COSMOS physics", "conventional ML", "deep learning"]
    famof = {"physics": fam[0], "logreg": fam[1], "xgboost": fam[1], "xgboost_aft": fam[1],
             "hazard_mlp": fam[2], "hazard_mlp_nophys": fam[2]}
    nice = {"physics": "COSMOS orbital-lifetime model", "logreg": "logistic regression",
            "xgboost": "XGBoost", "xgboost_aft": "XGBoost AFT (survival)",
            "hazard_mlp": "hazard MLP ensemble (+ physics feature)",
            "hazard_mlp_nophys": "hazard MLP ensemble (no physics)"}
    d = pd.DataFrame([{"model": m, **v} for m, v in r["results"].items()]).sort_values(
        "all/h68/auc_pr", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.2))
    for ax, h in zip(axes, (68, 30)):
        bar_panel(ax, [nice[m] for m in d.model], d[f"all/h{h}/auc_pr"].to_numpy(), None,
                  [famof[m] for m in d.model], fam, "test AUC-PR",
                  f"P(re-entry within {h} days of 2026-08-16)", xlim=(0, 1.12))
    _legend(fig, fam, set(famof.values()))
    fig.text(0.01, -0.06, f"Real CelesTrak landmark; test = {r['split_sizes']['test']:,} objects "
             f"({r['split_events']['test']} events), split by launch.", fontsize=8, color=INK2)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "reentry_landmark.png")
    s = _json(_bench("reentry_surrogate", "results.json"))
    if s and s.get("footprints"):
        f = pd.DataFrame(s["footprints"])
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        lim = [0, max(f.cep50_ref_km.max(), f.cep50_sur_km.max()) * 1.1]
        ax.plot(lim, lim, color=AXIS, linewidth=1)
        ax.scatter(f.cep50_ref_km, f.cep50_sur_km, s=40, color=SLOTS[0], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlabel("CEP50, reference propagator [km]", color=INK2, fontsize=9)
        ax.set_ylabel("CEP50, neural surrogate [km]", color=INK2, fontsize=9)
        ax.set_title(f"Footprint scenarios (SIM): median speed-up {f.speedup.median():.0f}x",
                     loc="left", fontsize=10, color=INK)
        _save(fig, "reentry_surrogate_footprints.png")


# ============================================================================ summary
def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ms(t, col, d=3):
    """'mean ± sd (n)' from a summary row."""
    m, s, n = t.get(f"{col}|mean"), t.get(f"{col}|std"), t.get("n_seeds")
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "not run"
    sd = "" if s is None or (isinstance(s, float) and np.isnan(s)) or n == 1 else f" ± {s:.{d}f}"
    return f"{m:.{d}f}{sd} (n={int(n)})"


def summary_markdown() -> str:
    rows = []
    esa = pd.read_csv(_bench("telemetry_esa_adb", "summary_table.csv")) if os.path.isfile(
        _bench("telemetry_esa_adb", "summary_table.csv")) else pd.DataFrame()
    boot = pd.read_csv(_bench("telemetry_esa_adb", "bootstrap_comparisons.csv")) if os.path.isfile(
        _bench("telemetry_esa_adb", "bootstrap_comparisons.csv")) else pd.DataFrame()

    def esa_row(m, enc, cal, ch="target"):
        r = esa[(esa.mission == m) & (esa.encoder == enc) & (esa.calibrator == cal) & (esa.channels == ch)]
        return r.iloc[0].to_dict() if len(r) else {}

    def ci(name):
        b = boot[boot.comparison == name] if len(boot) else boot
        if not len(b) or "ci95_low" not in b or pd.isna(b.iloc[0].get("ci95_low")):
            return "CI not run"
        b = b.iloc[0]
        return f"Δ {b.diff_reimpl:+.3f} [95% CI {b.ci95_low:+.3f}, {b.ci95_high:+.3f}]"

    for m in (1, 2):
        base = esa_row(m, "cosmos_monitor", "native")
        std5 = esa_row(m, "global_std5", "native")
        ar8 = esa_row(m, "ar8", "conformal_bh")
        for enc, label in (("ar8", "COSMOS AR(8) + conformal-BH alarms (proposed calibration)"),
                           ("patchtst", "PatchTST + conformal-BH"), ("tcn", "TCN + conformal-BH"),
                           ("chronos-bolt-small", "Chronos-Bolt-small zero-shot + conformal-BH"),
                           ("patchtst+tc", "PatchTST + telecommand gating + conformal-BH")):
            r = esa_row(m, enc, "conformal_bh")
            if not r:
                rows.append([f"Telemetry (M{m})", "ESA-ADB, full channels, official split", label,
                             "not run", "-", "-", "-", "-"])
                continue
            comp = (f"COSMOS monitor {_fmt(base.get('anomaly_and_rare/EW_F0.5|mean'))}; GlobalSTD5 "
                    f"{_fmt(std5.get('anomaly_and_rare/EW_F0.5|mean'))}; published best "
                    f"{DMLR_BEST[(m, 'target')][1]} ({DMLR_BEST[(m, 'target')][0]})")
            if enc in ("patchtst", "tcn", "chronos-bolt-small"):
                comp += f"; vs AR(8) at equal calibration: {ci(f'H1 M{m}: {enc} vs COSMOS AR(8), conformal-BH')}"
            if enc == "ar8":
                comp += f"; conformal-BH vs threshold: {ci(f'H2 M{m}: ar8 conformal-BH vs threshold')}"
            if enc == "patchtst+tc":
                comp += f"; vs no telecommands: {ci(f'H3 M{m}: patchtst + telecommands vs without, conformal-BH')}"
            met = (f"EW F0.5 {_ms(r, 'anomaly_and_rare/EW_F0.5')}; P {_fmt(r.get('anomaly_and_rare/EW_precision|mean'))}, "
                   f"R {_fmt(r.get('anomaly_and_rare/EW_recall|mean'))}; channel-F0.5 "
                   f"{_fmt(r.get('anomaly_and_rare/channel_F0.5|mean'))}; false alarms/yr "
                   f"{_fmt(r.get('false_alarms_per_year|mean'), 1)}")
            issues = ("M1: 9 calibration events -> noisy selection; low absolute scores" if m == 1 else
                      "test events are 98 % rare nominal events (6 anomalies)")
            rows.append([f"Telemetry (M{m})", "ESA-ADB v1.0, full 58/47 target channels, official split",
                         label, "trained / scored once" if enc not in ("ar8", "chronos-bolt-small")
                         else ("fitted" if enc == "ar8" else "zero-shot"), met, comp,
                         "outputs/benchmarks/telemetry_esa_adb/", issues])
    s = pd.read_csv(_bench("telemetry_opssat", "summary_official.csv")) if os.path.isfile(
        _bench("telemetry_opssat", "summary_official.csv")) else pd.DataFrame()
    if len(s):
        g = lambda mm: s[(s.model == mm) & (s.train_frac == 1.0)].iloc[0]
        for mm, label in (("hybrid", "Hybrid raw-CNN + feature MLP"), ("fcnn", "FCNN on official features"),
                          ("cnn", "1D-CNN on raw fragments"),
                          ("chronos:amazon/chronos-bolt-small", "Chronos-Bolt frozen embedding + probe")):
            r = g(mm)
            rows.append(["Telemetry", "OPSSAT-AD, official split", label, "trained, 5 seeds",
                         f"F1 {r.f1_mean:.3f} ± {r.f1_std:.3f}; AUC-PR {r.auc_pr_mean:.3f} ± {r.auc_pr_std:.3f}; MCC {r.mcc_mean:.3f}",
                         f"XGBoost F1 {g('xgboost').f1_mean:.3f} / AUC-PR {g('xgboost').auc_pr_mean:.3f}; COSMOS robust-z F1 "
                         f"{g('cosmos_robust_z').f1_mean:.3f}; published FCNN 0.946 / 0.979",
                         "outputs/benchmarks/telemetry_opssat/", "random official split; time-blocked split lowers all models; XPU not bit-reproducible"])
    for tag, title in (("heldout", "COSMOS synthetic telemetry (held-out seeds 5-9)"),):
        p = _bench("telemetry_cosmos_synthetic", tag, "summary_by_seed.csv")
        if os.path.isfile(p):
            d = pd.read_csv(p).groupby("monitor").mean(numeric_only=True)
            L, C = d.loc["learned_conformal"], d.loc["classical_ewma"]
            rows.append(["Telemetry (pipeline)", title, "Learned monitor (PatchTST + conformal-BH)",
                         "trained on 200 nominal runs", f"F1 {L.f1:.3f}, P {L.precision:.3f}, R {L.recall:.3f}; localisation {L.localisation_accuracy:.3f}",
                         f"classical EWMA F1 {C.f1:.3f}, localisation {C.localisation_accuracy:.3f}",
                         "outputs/benchmarks/telemetry_cosmos_synthetic/", "SYNTHETIC only; 'stuck' fault crashes the simulator (pre-existing)"])
    v = vision_tables()
    if "spotgeo" in v:
        d = v["spotgeo"]
        cl = d[d.model.str.startswith("classical")]
        for mm, inp, label in (("unet", "both", "Heat-map U-Net (scratch), diff + high-pass"),
                               ("resnet18_pretrained", "both", "Heat-map U-Net, ImageNet ResNet-18 encoder"),
                               ("unet", "raw", "Heat-map U-Net, raw high-pass only (front-end ablation)")):
            r = d[(d.model == mm) & (d.inputs == inp) & (d.train_frac == 1.0)]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append(["Vision", "SpotGEO v2 (real), 5,120 test sequences", label,
                         f"trained, {int(r.n_seeds)} seed(s)", f"official F1 {_ms(r, 'test_f1', 4)}; P {r['test_precision|mean']:.3f}, R {r['test_recall|mean']:.3f}",
                         f"classical COSMOS front end F1 {cl['test_f1|mean'].iloc[0]:.3f}; challenge winner 0.948" if len(cl) else "challenge winner 0.948",
                         "outputs/benchmarks/vision_spotgeo/", "far below the challenge winner; no SNR breakdown (not in labels)"])
    if "synthetic" in v:
        d = v["synthetic"].set_index("model")
        for mm in d.index:
            r = d.loc[mm]
            rows.append(["Vision", "COSMOS synthetic frame pairs (scene split)", mm, f"trained, {int(r.n_seeds)} seeds",
                         f"mAP@50 {r['map50|mean']:.3f} ± {r['map50|std']:.3f}; mAP@50-95 {r['map50_95|mean']:.3f} ± {r['map50_95|std']:.3f}; "
                         f"latency {r['latency_ms_per_item|mean']:.2f} ms/img",
                         "same images, same COCO-style evaluator", "outputs/benchmarks/vision_cosmos_synthetic/",
                         "SYNTHETIC only; Ultralytics' own mAP is higher (different interpolation)"])
    r = _json(_bench("reentry_landmark", "results.json"))
    if r:
        res = r["results"]
        cmp_ = r["comparisons"]
        for mm, label in (("xgboost", "XGBoost (reported model)"), ("hazard_mlp", "Discrete-time hazard MLP ensemble"),
                          ("physics", "COSMOS orbital-lifetime physics")):
            x = res[mm]
            comp = ("MLP vs XGBoost AUC-PR@68 Δ {diff:+.3f} [{ci_low:+.3f}, {ci_high:+.3f}]".format(**cmp_["hazard_mlp_vs_xgboost/h68"])
                    if mm == "hazard_mlp" else f"physics AUC-PR@68 {res['physics']['all/h68/auc_pr']:.3f}")
            rows.append(["Re-entry", "CelesTrak landmark 2026-08-16 → 10-23 (real), split by launch", label,
                         "trained" if mm != "physics" else "physics + Platt",
                         f"AUC-PR@68d {x['all/h68/auc_pr']:.3f}, @30d {x['all/h30/auc_pr']:.3f}; C {x['all/c_index']:.3f}; Brier@68 {x['all/h68/brier']:.4f}",
                         comp, "outputs/benchmarks/reentry_landmark/", "one landmark, 38 test events; no impact-location truth"])
    sres = _json(_bench("reentry_surrogate", "results.json"))
    if sres:
        t, f = sres["test"], pd.DataFrame(sres["footprints"])
        rows.append(["Re-entry (SIM)", "COSMOS propagator outputs (500k trajectories)", "MLP ensemble surrogate",
                     "trained, 5 members", f"median impact error {t['position_error_km_median']:.1f} km (p95 {t['position_error_km_p95']:.0f}); "
                     f"{t['surrogate_samples_per_s']:,.0f} samples/s", f"reference propagator ~220 samples/s; footprint CEP50 "
                     f"rel. error median {np.median(np.abs(f.cep50_sur_km / f.cep50_ref_km - 1)):.1%}",
                     "outputs/benchmarks/reentry_surrogate/", f"SIM only; OOD flag AUROC {sres['ood_auroc_ensemble_std']:.2f}"])
    head = "| Component | Dataset | Model | Training status | Test metrics | Baseline comparison | Evidence/output path | Remaining issues |"
    lines = [head, "|" + "---|" * 8]
    for rr in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in rr) + " |")
    return "\n".join(lines)


def main() -> int:
    t = esa_table()
    if len(t):
        esa_figures(t)
        pairs = []
        for m in (1, 2):
            for enc in ("patchtst", "tcn", "chronos-bolt-small", "persistence"):
                pairs.append((f"H1 M{m}: {enc} vs COSMOS AR(8), conformal-BH",
                              (m, enc, False, "conformal_bh", 0), (m, "ar8", False, "conformal_bh", 0)))
            for enc in ("ar8", "patchtst", "tcn"):
                pairs.append((f"H2 M{m}: {enc} conformal-BH vs threshold",
                              (m, enc, False, "conformal_bh", 0), (m, enc, False, "threshold", 0)))
            for enc in ("patchtst", "tcn"):
                pairs.append((f"H3 M{m}: {enc} + telecommands vs without, conformal-BH",
                              (m, enc, True, "conformal_bh", 0), (m, enc, False, "conformal_bh", 0)))
        boot = esa_bootstrap(pairs)
        with open(_bench("telemetry_esa_adb", "bootstrap_comparisons.json"), "w") as f:
            json.dump(boot, f, indent=2, default=float)
        pd.DataFrame(boot).to_csv(_bench("telemetry_esa_adb", "bootstrap_comparisons.csv"),
                                  index=False)
    s, c = opssat_tables()
    if s is not None:
        opssat_figures(s)
    v = vision_tables()
    vision_figures(v)
    reentry_figures()
    with open(_bench("SUMMARY_TABLE.md"), "w") as f:
        f.write(summary_markdown() + "\n")
    print("figures:", sorted(os.listdir(output_dir("benchmarks", "figures"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
