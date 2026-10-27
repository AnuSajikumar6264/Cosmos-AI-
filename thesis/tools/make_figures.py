"""Result figures for the thesis, drawn only from thesis/evidence/evidence.json (plus, for the
selection landscape, the selection files the evidence cites, and for the monitor grid,
thesis/evidence/original_pipeline_runs.json).

    python thesis/tools/make_figures.py      # -> thesis/figures/*.png (300 dpi)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TH = os.path.dirname(HERE)
ROOT = os.path.dirname(TH)
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

E = json.load(open(os.path.join(TH, "evidence", "evidence.json")))
OUT = os.path.join(TH, "figures")
os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#f2f6fc", "#cfe0f5", "#9cc0ec", "#5f97de", "#2a78d6", "#1d5aa6", "#123c70"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def v(k):
    if k not in E or E[k]["value"] is None:
        raise KeyError(k)
    return E[k]["value"]


def opt(k, d=np.nan):
    return E[k]["value"] if k in E and E[k]["value"] is not None else d


def style(ax, xlabel=None, xgrid=True):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)


def save(fig, name):
    fig.patch.set_facecolor(SURF)
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("wrote", name)


def hbars(ax, labels, vals, errs=None, cols=None, dots=None, xmax=1.0, fmt="{:.3f}"):
    y = np.arange(len(labels))
    ax.barh(y, vals, height=0.6, color=cols, edgecolor=SURF, linewidth=2)
    for i, val in enumerate(vals):
        e = 0 if errs is None or np.isnan(errs[i]) else errs[i]
        if errs is not None and e > 0:
            ax.errorbar(val, i, xerr=e, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        if dots is not None and dots[i] is not None and len(dots[i]) > 1:
            ax.scatter(dots[i], np.full(len(dots[i]), i), s=9, color=INK, zorder=3, linewidths=0)
        right = max([val + e] + (list(dots[i]) if dots is not None and dots[i] else []))
        ax.text(min(right, xmax) + 0.015 * xmax, i, fmt.format(val), va="center", fontsize=7,
                color=INK)
    ax.set_yticks(y, labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.08)


# ----------------------------------------------------------------------------- ESA exploratory
def esa_exploratory():
    rows = [("GlobalSTD5 (official)", "gstd5", "native"), ("Out-of-limit (nominal range)", "ool", "native"),
            ("COSMOS monitor (as shipped)", "cosmon", "native"),
            ("Persistence + threshold", "persistence", "threshold"),
            ("Persistence + conformal-BH", "persistence", "conformal_bh"),
            ("AR(8) + threshold", "ar8", "threshold"), ("AR(8) + conformal-BH", "ar8", "conformal_bh"),
            ("TCN + threshold", "tcn", "threshold"), ("TCN + conformal-BH", "tcn", "conformal_bh"),
            ("PatchTST + threshold", "patchtst", "threshold"),
            ("PatchTST + conformal-BH", "patchtst", "conformal_bh"),
            ("Chronos-Bolt + threshold", "chronos", "threshold"),
            ("Chronos-Bolt + conformal-BH", "chronos", "conformal_bh")]
    fam = {"native": MUTED, "threshold": SLOTS[0], "conformal_bh": SLOTS[1]}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=False)
    for ax, m in zip(axes, (1, 2)):
        lab, val, err, col = [], [], [], []
        for name, enc, cal in rows:
            k = f"explo.m{m}.full.{enc}.{cal}.f05"
            lab.append(name)
            val.append(v(f"{k}.mean"))
            n = v(f"{k}.n")
            err.append(v(f"{k}.std") if n > 1 else np.nan)
            col.append(fam[cal])
        hbars(ax, lab, val, err, col)
        ref = v(f"lit.dmlr.m{m}.full.best")
        ax.axvline(ref, color=INK2, linewidth=0.8, linestyle="--")
        ax.set_ylim(len(lab) - 0.5, -1.4)
        ax.text(ref, -1.0, f" best published ({v(f'lit.dmlr.m{m}.full.best_name')}) {ref:.3f}",
                fontsize=7, color=INK2, va="center")
        style(ax, "corrected event-wise F0.5 (anomalies + rare events)")
        ax.set_title(f"Mission {m}, full target channel set", fontsize=9, color=INK, loc="left")
        if m == 2:
            ax.set_yticklabels([])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=MUTED, label="label-free baseline (native rule)"),
                        Patch(color=SLOTS[0], label="per-channel threshold"),
                        Patch(color=SLOTS[1], label="conformal p-values + BH")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "esa_exploratory.png")


# ----------------------------------------------------------------------------- pre-registered
ARMS = [("A0", "A0: AR(8), calibration-window selection"), ("A1", "A1: AR(8), training-half selection"),
        ("A2", "A2: gradient-boosted trees"), ("A3", "A3: TCN event scorer"),
        ("A3c2", "A3 ablation: 2 classes"), ("A3nr", "A3 ablation: no residual inputs")]


def seeds(m, arm, cal="conformal_bh"):
    pat = {"A2": f"m{m}_target_gbt_ar8_s{{}}_event", "A3": f"m{m}_target_event_ar8_all_c3_s{{}}_event",
           "A3c2": f"m{m}_target_event_ar8_all_c2_s{{}}_event",
           "A3nr": f"m{m}_target_event_ar8_no_residual_c3_s{{}}_event"}.get(arm)
    if pat is None:
        return None
    out = [opt(f"pre.seed.{pat.format(s)}.{cal}.f05", None) for s in range(5)]
    return [x for x in out if x is not None]


def esa_arms():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2), sharey=False,
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, m in zip(axes, (1, 2)):
        lab, val, err, col, dots = [], [], [], [], []
        for i, (a, name) in enumerate(ARMS):
            k = f"pre.m{m}.{a}.conformal_bh.f05"
            if f"{k}.mean" not in E:
                continue
            lab.append(name)
            val.append(v(f"{k}.mean"))
            err.append(opt(f"{k}.std") if opt(f"{k}.n", 1) > 1 else np.nan)
            col.append(SLOTS[min(i, 4)] if i < 5 else MUTED)
            dots.append(seeds(m, a))
        hbars(ax, lab, val, None, col, dots)
        ool = v(f"explo.m{m}.full.ool.native.f05.mean")
        ax.axvline(ool, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_ylim(len(lab) - 0.5, -1.3)
        ax.text(ool, -0.95, f" out-of-limit rule {ool:.3f}", fontsize=7, color=INK2, va="center")
        style(ax, f"Mission {m} test half: corrected event-wise F0.5")
        ax.set_title(f"Mission {m} ({'primary' if m == 1 else 'replication'})", fontsize=9,
                     color=INK, loc="left")
    fig.tight_layout()
    save(fig, "esa_prereg_arms.png")


def esa_forest():
    rows = []
    for m in (1, 2):
        for h, lab in (("h4", "H4: A3 − A1 (TCN vs AR(8))"), ("h5", "H5: A3 − A2 (TCN vs trees)"),
                       ("h6", "H6: A1 − A0 (training-half vs calibration-window)")):
            rows.append((f"M{m} {lab}", v(f"conf.m{m}.{h}.diff_official"),
                         v(f"conf.m{m}.{h}.ci95_low"), v(f"conf.m{m}.{h}.ci95_high"),
                         "pre-registered" if m == 1 else "replication"))
        for x, lab in (("x1", "X1: A2 − A1 (trees vs AR(8))"),):
            rows.append((f"M{m} {lab}", v(f"xcomp.m{m}.{x}.diff_official"), v(f"xcomp.m{m}.{x}.ci95_low"),
                         v(f"xcomp.m{m}.{x}.ci95_high"), "exploratory"))
    col = {"pre-registered": SLOTS[0], "replication": SLOTS[2], "exploratory": MUTED}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (lab, d, lo, hi, kind) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=col[kind], linewidth=1.6)
        ax.scatter([d], [i], s=28, color=col[kind], zorder=3, edgecolor=SURF, linewidth=1)
        ax.text(0.66, i, f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(0.05, color=INK2, linewidth=0.6, linestyle=":")
    ax.text(0.05, -0.9, " minimum effect +0.05", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.65, 1.05)
    ax.axvspan(0.62, 1.05, color=SURF, zorder=0)
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    style(ax, "difference in corrected event-wise F0.5 (95 % bootstrap CI)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=col[k], lw=2, label=k) for k in col],
              frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3)
    fig.tight_layout()
    save(fig, "esa_forest.png")


def esa_landscape():
    import pandas as pd
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    panels = [("m1_target_ar8", "Mission 1, AR(8) scores"), ("m1_target_gbt_ar8_s0_event", "Mission 1, trees (seed 0)"),
              ("m1_target_event_ar8_all_c3_s0_event", "Mission 1, TCN (seed 0)"),
              ("m2_target_ar8", "Mission 2, AR(8) scores")]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), layout="constrained")
    axes = axes.ravel()
    for n, (ax, (t, lbl)) in enumerate(zip(axes, panels)):
        p = os.path.join(ROOT, "outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection", f"{t}.json")
        s = json.load(open(p))
        g = pd.DataFrame(s["grid"])
        g = g[g.kind == "conformal_bh"].pivot(index="level", columns="k", values="cv").sort_index(ascending=False)
        im = ax.imshow(g.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        best = np.unravel_index(np.nanargmax(g.values), g.shape)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                val = g.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color=SURF if val > 0.55 else INK, fontweight="bold" if (i, j) == best else "normal")
        ax.set_xticks(range(g.shape[1]), [str(k) for k in g.columns], fontsize=7)
        ax.set_yticks(range(g.shape[0]), [f"{q:g}" for q in g.index], fontsize=7)
        ax.set_xlabel("persistence k (steps)", fontsize=7)
        if n % 2 == 0:
            ax.set_ylabel("BH level q", fontsize=7)
        ax.set_title(lbl, fontsize=8, color=INK, loc="left")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.85, pad=0.01)
    cb.set_label("training-half CV event-wise F0.5", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    save(fig, "esa_selection_landscape.png")


def esa_shift():
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    keys = [("a1_ar8", "A1 AR(8)")] + [(f"a2_s{s}", f"A2 trees s{s}") for s in range(5)] + \
           [(f"a3_s{s}", f"A3 TCN s{s}") for s in range(5)]
    ratios = [v(f"shift.m1.{k}.ratio") for k, _ in keys]
    cols = [MUTED] + [SLOTS[2]] * 5 + [SLOTS[3]] * 5
    y = np.arange(len(keys))
    ax.scatter(ratios, y, s=30, color=cols, zorder=3, edgecolor=SURF, linewidth=1)
    for i, r in enumerate(ratios):
        ax.text(r * 1.12, i, f"{r:.1f}×", va="center", fontsize=7, color=INK)
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, [l for _, l in keys], fontsize=7, color=INK)
    ax.invert_yaxis()
    style(ax, "nominal exceedance of the 1−1e-4 calibration quantile: test rate / training-half CV rate (log)")
    fig.tight_layout()
    save(fig, "esa_calibration_shift.png")


def esa_classes():
    groups = [("seen", "Anomaly", "anomalies, seen class"), ("unseen", "Anomaly", "anomalies, unseen class"),
              ("seen", "Rare_Event", "rare events, seen class"), ("unseen", "Rare_Event", "rare events, unseen class")]
    arms = [("A0", SLOTS[0]), ("A1", SLOTS[1]), ("A2", SLOTS[2]), ("A3", SLOTS[3])]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    w = 0.2
    for gi, (s, c, lab) in enumerate(groups):
        for ai, (a, col) in enumerate(arms):
            val = v(f"evcls.m1.{s}.{c}.{a}")
            ax.bar(gi + (ai - 1.5) * w, val, width=w * 0.92, color=col, edgecolor=SURF)
    ax.set_xticks(range(len(groups)), [f"{lab}\n(n = {v(f'evcls.m1.{s}.{c}.n')})" for s, c, lab in groups],
                  fontsize=7, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of test events detected (seed mean)", fontsize=7)
    style(ax, xgrid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=a) for a, c in arms], frameon=False, fontsize=7, ncol=4,
              loc="upper left")
    fig.tight_layout()
    save(fig, "esa_event_classes.png")


# ----------------------------------------------------------------------------- other studies
def opssat():
    models = [("hybrid", "hybrid CNN + features"), ("fcnn", "feature MLP (FCNN)"), ("cnn", "1-D CNN"),
              ("xgboost", "XGBoost"), ("chronos", "Chronos-Bolt probe"), ("cosmos_robust_z", "COSMOS robust-z")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.6), sharey=True)
    for ax, (prot, title) in zip(axes, (("official", "official split"), ("time_blocked", "time-blocked split"))):
        lab, val, err, col = [], [], [], []
        for i, (m, name) in enumerate(models):
            k = f"ops.{prot}.{m}.f100.f1"
            if f"{k}.mean" not in E:
                continue
            lab.append(name)
            val.append(v(f"{k}.mean"))
            err.append(opt(f"{k}.std"))
            col.append(SLOTS[i] if i < 5 else MUTED)
        hbars(ax, lab, val, err, col)
        if prot == "official":
            ref = v("lit.opssat.fcnn_f1")
            ax.axvline(ref, color=INK2, linewidth=0.8, linestyle="--")
            ax.text(ref, -0.8, f" published FCNN {ref:.3f}", fontsize=7, color=INK2, ha="right")
        style(ax, "F1 at threshold 0.5 (mean ± sd, 5 seeds)")
        ax.set_title(f"OPSSAT-AD, {title}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "opssat_f1.png")


def vision():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6))
    rows = [("classical COSMOS front end + DoG", "spot.classical.diff.f100"),
            ("U-Net heat map, raw frames only", "spot.unet.raw.f100"),
            ("U-Net heat map, differences only", "spot.unet.diff.f100"),
            ("U-Net heat map, both (10 % data)", "spot.unet.both.f10"),
            ("U-Net heat map, both (25 % data)", "spot.unet.both.f25"),
            ("U-Net heat map, both", "spot.unet.both.f100"),
            ("ResNet-18 encoder (pretrained), both", "spot.resnet18_pretrained.both.f100")]
    lab = [r[0] for r in rows]
    val = [v(f"{k}.test_f1.mean") for _, k in rows]
    err = [opt(f"{k}.test_f1.std") if v(f"{k}.n") > 1 else np.nan for _, k in rows]
    hbars(axes[0], lab, val, err, [MUTED] + [SLOTS[0]] * 5 + [SLOTS[1]])
    style(axes[0], "official SpotGEO F1 on the test set")
    axes[0].set_title("SpotGEO (real GEO imagery)", fontsize=9, color=INK, loc="left")
    rows = [("TinyYOLO (existing COSMOS)", "syn.tinyyolo"), ("YOLOv8n (COCO-pretrained)", "syn.yolov8n"),
            ("heat-map U-Net (COSMOS-AI)", "syn.heatmap")]
    lab = [r[0] for r in rows]
    val = [v(f"{k}.map50.mean") for _, k in rows]
    err = [v(f"{k}.map50.std") for _, k in rows]
    hbars(axes[1], lab, val, err, [MUTED, SLOTS[2], SLOTS[0]])
    style(axes[1], "COCO-style mAP@50 (mean ± sd, 3 seeds)")
    axes[1].set_title("COSMOS synthetic frame pairs (SYNTHETIC)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "vision_results.png")


def reentry():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.5))
    rows = [("physics (COSMOS lifetime)", "physics"), ("logistic regression", "logreg"),
            ("hazard MLP ensemble", "hazard_mlp"), ("XGBoost-AFT", "xgboost_aft"), ("XGBoost", "xgboost")]
    for ax, h in zip(axes, ("h30", "h68")):
        lab = [r[0] for r in rows]
        val = [v(f"lm.{k}.all.{h}.auc_pr") for _, k in rows]
        hbars(ax, lab, val, None, [MUTED, SLOTS[0], SLOTS[3], SLOTS[2], SLOTS[1]])
        style(ax, f"AUC-PR, P(re-entry within {h[1:]} days), test launches")
        ax.set_title(f"Landmark {v('lm.tau')} → horizon {h[1:]} d", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "reentry_landmark.png")


# ----------------------------------------------------------------------------- added analyses
ARM_COL = {"A0": SLOTS[0], "A1": SLOTS[1], "A2": SLOTS[2], "A3": SLOTS[3], "A1lw": SLOTS[4]}
ARM_NAME = {"A0": "A0 AR(8), calibration window", "A1": "A1 AR(8), training half",
            "A2": "A2 trees", "A3": "A3 TCN", "A1lw": "A1, lightweight channels"}


def esa_selection_effect():
    from matplotlib.lines import Line2D
    rows = [("full", "conformal_bh", "full channels, conformal-BH"), ("full", "threshold", "full channels, threshold"),
            ("lw", "conformal_bh", "lightweight, conformal-BH"), ("lw", "threshold", "lightweight, threshold")]
    mcol = {1: SLOTS[0], 2: SLOTS[2]}
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for i, (cs, cal, _) in enumerate(rows):
        for m, off in ((1, -0.13), (2, 0.13)):
            d = v(f"seleff.m{m}.{cs}.{cal}")
            ax.plot([0, d], [i + off, i + off], color=mcol[m], linewidth=1.2, alpha=0.6)
            ax.scatter([d], [i + off], s=36, color=mcol[m], zorder=3, edgecolor=SURF, linewidth=1)
            ax.text(d + (0.02 if d >= 0 else -0.02), i + off, f"{d:+.3f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=7, color=INK)
    ax.scatter([v("seleff.m1.full.conformal_bh")], [-0.13], s=150, facecolor="none", edgecolor=INK,
               linewidth=0.9, zorder=4)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.axvline(0.05, color=INK2, linewidth=0.6, linestyle=":")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=8, color=INK)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(-0.55, 0.55)
    style(ax, "training-half selection minus calibration-window selection (test-half event-wise F0.5)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=mcol[1], lw=0, label="Mission 1"),
                       Line2D([0], [0], marker="o", color=mcol[2], lw=0, label="Mission 2"),
                       Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=INK, lw=0,
                              markersize=9, label="pre-registered H6")],
              frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.tight_layout()
    save(fig, "esa_selection_effect.png")


def esa_alarm_burden():
    from matplotlib.patches import Patch
    arms = ["A0", "A1", "A2", "A3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.4))
    for ax, m in zip(axes, (1, 2)):
        n_ev = v(f"data.esa.m{m}.events_target.test.Anomaly") + v(f"data.esa.m{m}.events_target.test.Rare_Event")
        fa = [v(f"pre.m{m}.{a}.conformal_bh.fa_yr.mean") for a in arms]
        tot = [v(f"pre.m{m}.{a}.conformal_bh.alarm_runs_per_year") for a in arms]
        ev = [t - f for t, f in zip(tot, fa)]
        y = np.arange(len(arms))
        ax.barh(y, ev, height=0.6, color=SLOTS[0], edgecolor=SURF, linewidth=2)
        ax.barh(y, fa, left=ev, height=0.6, color=SLOTS[1], edgecolor=SURF, linewidth=2)
        xmax = max(tot)
        for i, a in enumerate(arms):
            det = v(f"pre.m{m}.{a}.conformal_bh.r.mean") * n_ev
            ax.text(tot[i] + 0.015 * xmax, i, f"{tot[i]:.0f} runs/yr; detects {det:.0f} of {n_ev} events",
                    va="center", fontsize=7, color=INK)
        ax.set_yticks(y, [ARM_NAME[a] for a in arms], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.75)
        style(ax, "alarm runs per year of test half (seed means, conformal-BH)")
        ax.set_title(f"Mission {m}", fontsize=9, color=INK, loc="left")
        if m == 2:
            ax.set_yticklabels([])
    fig.legend(handles=[Patch(color=SLOTS[0], label="runs that touch a target event"),
                        Patch(color=SLOTS[1], label="false-alarm runs")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "esa_alarm_burden.png")


def esa_drift_vs_score():
    pats = {"A2": "m1_target_gbt_ar8_s{}_event", "A3": "m1_target_event_ar8_all_c3_s{}_event"}
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x, y = v("shift.m1.a1_ar8.ratio"), v("pre.m1.A1.conformal_bh.f05.mean")
    ax.scatter([x], [y], s=40, color=ARM_COL["A1"], edgecolor=SURF, linewidth=1, zorder=3)
    ax.text(x * 1.1, y, "A1", fontsize=7, color=INK, va="center")
    for arm, pat in pats.items():
        for s in range(5):
            x = v(f"shift.m1.{arm.lower()}_s{s}.ratio")
            y = v(f"pre.seed.{pat.format(s)}.conformal_bh.f05")
            ax.scatter([x], [y], s=40, color=ARM_COL[arm], edgecolor=SURF, linewidth=1, zorder=3)
            left = (arm, s) in (("A2", 0), ("A3", 3))
            ax.text(x / 1.1 if left else x * 1.1, y, f"{arm} s{s}", fontsize=7, color=INK, va="center",
                    ha="right" if left else "left")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:g}×"))
    ax.set_xlim(0.5, 120)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("test-half event-wise F0.5", fontsize=8, color=INK2)
    style(ax, "nominal exceedance on test / in training-half CV (log scale)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.text(0.02, 0.97, f"Spearman ρ: TCN seeds {v('shift.m1.A3.spearman_f05'):.2f}, "
            f"tree seeds {v('shift.m1.A2.spearman_f05'):.2f}, all ten {v('shift.m1.supervised.spearman_f05'):.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, "esa_drift_vs_score.png")


def esa_primary_vs_anomaly():
    from matplotlib.lines import Line2D
    arms = ["A0", "A1", "A2", "A3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.4), sharey=False)
    for ax, m in zip(axes, (1, 2)):
        for i, a in enumerate(arms):
            p = v(f"pre.m{m}.{a}.conformal_bh.f05.mean")
            q = v(f"pre.m{m}.{a}.conformal_bh.f05_anom.mean")
            ax.plot([q, p], [i, i], color=ARM_COL[a], linewidth=1.4, alpha=0.7)
            ax.scatter([p], [i], s=40, color=ARM_COL[a], zorder=3, edgecolor=SURF, linewidth=1)
            ax.scatter([q], [i], s=40, facecolor=SURF, edgecolor=ARM_COL[a], linewidth=1.4, zorder=3,
                       marker="D")
            ax.text(1.02, i, f"{p:.3f} / {q:.3f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(arms)), [ARM_NAME[a] for a in arms], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(-0.04, 1.25)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        na, nr = v(f"data.esa.m{m}.events_target.test.Anomaly"), v(f"data.esa.m{m}.events_target.test.Rare_Event")
        ax.set_title(f"Mission {m}: {na} anomalies, {nr} rare events in the test half", fontsize=9,
                     color=INK, loc="left")
        style(ax, "event-wise F0.5 (seed means, conformal-BH)")
        if m == 2:
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([0], [0], marker="o", color=INK2, lw=0, label="primary (anomalies + rare events)"),
                        Line2D([0], [0], marker="D", markerfacecolor=SURF, markeredgecolor=INK2, lw=0,
                               label="anomalies only")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "esa_primary_vs_anomaly.png")


def esa_layers():
    from matplotlib.lines import Line2D
    rows = {1: [("A1", "A1"), ("A2", "A2 trees"), ("A3", "A3 TCN"), ("A3nr", "A3, no residual inputs"),
                ("A3c2", "A3, two classes"), ("A1lw", "A1, lightweight")],
            2: [("A1", "A1"), ("A2", "A2 trees"), ("A3", "A3 TCN"), ("A1lw", "A1, lightweight")]}
    lay = [("threshold", "per-channel threshold", SLOTS[0], "o"), ("conformal_bh", "conformal-BH", SLOTS[1], "s"),
           ("conformal_bh_rolling", "rolling conformal-BH", MUTED, "^")]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), gridspec_kw={"height_ratios": [6, 4]})
    for ax, m in zip(axes, (1, 2)):
        for i, (a, lab) in enumerate(rows[m]):
            vals = [v(f"pre.m{m}.{a}.{cal}.f05.mean") for cal, *_ in lay]
            ax.plot([min(vals), max(vals)], [i, i], color=GRID, linewidth=2, zorder=1)
            for (cal, _, col, mk), val in zip(lay, vals):
                ax.scatter([val], [i], s=34, color=col, marker=mk, zorder=3, edgecolor=SURF, linewidth=0.8)
        ax.set_yticks(range(len(rows[m])), [r[1] for r in rows[m]], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(f"Mission {m}", fontsize=9, color=INK, loc="left")
        style(ax, "test-half event-wise F0.5 (seed means)" if m == 2 else None)
    fig.legend(handles=[Line2D([0], [0], marker=mk, color=col, lw=0, label=lab) for _, lab, col, mk in lay],
               frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "esa_layers.png")


def esa_cv_vs_test():
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for m, mk in ((1, "o"), (2, "s")):
        for a in ("A1", "A2", "A3", "A1lw"):
            cv = v(f"sel.m{m}.{a}.conformal_bh.cv.mean")
            te = v(f"pre.m{m}.{a}.conformal_bh.f05.mean")
            ax.scatter([cv], [te], s=46, color=ARM_COL[a], marker=mk, edgecolor=SURF, linewidth=1, zorder=3)
            dx, dy, ha = {(2, "A2"): (-0.015, 0.0, "right"), (2, "A3"): (-0.015, -0.025, "right"),
                          (2, "A1"): (0.015, -0.02, "left")}.get((m, a), (0.015, 0.0, "left"))
            ax.text(cv + dx, te + dy, f"M{m} {a}", fontsize=7, color=INK, va="center", ha=ha)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--")
    ax.text(0.42, 0.46, "test = CV", fontsize=7, color=INK2, ha="center", rotation=38)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("test-half event-wise F0.5", fontsize=8, color=INK2)
    style(ax, "training-half cross-validated F0.5 of the selected setting")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=INK2, lw=0, label="Mission 1"),
                       Line2D([0], [0], marker="s", color=INK2, lw=0, label="Mission 2")],
              frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "esa_cv_vs_test.png")


def opssat_curves():
    mods = [("hybrid", "hybrid CNN + features", SLOTS[0]), ("fcnn", "feature MLP (FCNN)", SLOTS[1]),
            ("cnn", "1-D CNN", SLOTS[2]), ("xgboost", "XGBoost", SLOTS[3]), ("chronos", "Chronos-Bolt probe", SLOTS[4])]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
    ax = axes[0]
    fr = [10, 25, 50, 100]
    for k, lab, col in mods:
        ys = [v(f"ops.official.{k}.f{p}.auc_pr.mean") for p in fr]
        ax.plot(fr, ys, color=col, linewidth=1.6, marker="o", markersize=4, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(fr, [f"{p} %" for p in fr])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("test AUC-PR (mean, 5 seeds)", fontsize=8, color=INK2)
    style(ax, "share of training labels")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Label efficiency (official split)", fontsize=9, color=INK, loc="left")
    rob = [("hybrid", "hybrid", SLOTS[0]), ("fcnn", "FCNN", SLOTS[1]), ("cnn", "1-D CNN", SLOTS[2]),
           ("xgboost", "XGBoost", SLOTS[3]), ("random_forest", "random forest", MUTED)]
    for ax, (pert, levels, xl, title) in zip(axes[1:], (
            ("noise", [0.0, 0.1, 0.25, 0.5, 1.0], "added noise (fraction of fragment sd)", "Robustness to noise"),
            ("dropout", [0.0, 0.1, 0.3, 0.5], "share of samples dropped", "Robustness to sample dropout"))):
        for k, lab, col in rob:
            ys = [v(f"ops.robust.{'noise' if lv == 0 else pert}.{str(lv).replace('.', 'p')}.{k}.f1") for lv in levels]
            ax.plot(levels, ys, color=col, linewidth=1.6, marker="o", markersize=4, label=lab,
                    linestyle="--" if k == "random_forest" else "-")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("test F1 (mean, 5 seeds)", fontsize=8, color=INK2)
        style(ax, xl)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save(fig, "opssat_curves.png")


def reentry_subsets():
    from matplotlib.lines import Line2D
    rows = [("physics (COSMOS lifetime)", "physics", MUTED), ("logistic regression", "logreg", SLOTS[0]),
            ("hazard MLP ensemble", "hazard_mlp", SLOTS[3]), ("XGBoost-AFT", "xgboost_aft", SLOTS[2]),
            ("XGBoost", "xgboost", SLOTS[1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6))
    for ax, h in zip(axes, ("h30", "h68")):
        for i, (lab, k, col) in enumerate(rows):
            a = v(f"lm.{k}.all.{h}.auc_pr")
            u = v(f"lm.{k}.uncontrolled.{h}.auc_pr")
            ax.plot([min(a, u), max(a, u)], [i, i], color=GRID, linewidth=2, zorder=1)
            ax.scatter([a], [i], s=40, color=col, zorder=3, edgecolor=SURF, linewidth=1)
            ax.scatter([u], [i], s=40, facecolor=SURF, edgecolor=col, linewidth=1.4, marker="D", zorder=3)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        pa, pu = v(f"lm.physics.all.{h}.positives"), v(f"lm.physics.uncontrolled.{h}.positives")
        ax.set_title(f"Re-entry within {h[1:]} days (positives: {pa} all, {pu} uncontrolled)", fontsize=9,
                     color=INK, loc="left")
        style(ax, "AUC-PR on the test launch groups")
        if h == "h68":
            ax.set_yticklabels([])
    fig.legend(handles=[Line2D([0], [0], marker="o", color=INK2, lw=0, label=f"all test objects ({v('lm.split.test.n'):,})"),
                        Line2D([0], [0], marker="D", markerfacecolor=SURF, markeredgecolor=INK2, lw=0,
                               label=f"uncontrolled subset ({v('lm.physics.uncontrolled.n'):,})")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "reentry_subsets.png")


def orig_monitor_grid():
    R = json.load(open(os.path.join(TH, "evidence", "original_pipeline_runs.json")))
    g = [r for r in R["monitor_grid"] if r["fault"] != "nominal"]
    subs = list(dict.fromkeys(r["subsystem"] for r in g))
    faults = list(dict.fromkeys(r["fault"] for r in g))
    M = np.full((len(subs), len(faults)), np.nan)
    for r in g:
        if r["detected"]:
            M[subs.index(r["subsystem"]), faults.index(r["fault"])] = r["delay_minutes"]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(M))
    for i in range(len(subs)):
        for j in range(len(faults)):
            val = M[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=SURF, edgecolor=GRID, hatch="///",
                                           linewidth=0))
                ax.text(j, i, "missed", ha="center", va="center", fontsize=7, color=INK2)
            else:
                ax.text(j, i, f"{val:.0f} min", ha="center", va="center", fontsize=7,
                        color=SURF if val > 0.6 * np.nanmax(M) else INK)
    ax.set_xticks(range(len(faults)), faults, fontsize=8)
    ax.set_yticks(range(len(subs)), subs, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(faults)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subs)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)
    ax.tick_params(length=0, which="both")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("detection delay after fault onset (min)", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=6, length=0)
    cb.outline.set_visible(False)
    ax.set_title("Classical monitor on SIMULATED telemetry (seed 7, 24 h runs)", fontsize=9, color=INK, loc="left")
    save(fig, "orig_monitor_grid.png")


def main():
    for fn in (esa_exploratory, esa_arms, esa_forest, esa_landscape, esa_shift, esa_classes, opssat,
               vision, reentry, esa_selection_effect, esa_alarm_burden, esa_drift_vs_score,
               esa_primary_vs_anomaly, esa_layers, esa_cv_vs_test, opssat_curves, reentry_subsets,
               orig_monitor_grid):
        fn()


if __name__ == "__main__":
    main()
