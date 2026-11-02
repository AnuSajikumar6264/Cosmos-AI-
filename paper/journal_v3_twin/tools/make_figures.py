"""Result figures of the twin manuscript, drawn only from the result files (vector PDF + PNG).

    python paper/journal_v3_twin/tools/make_figures.py
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8})
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
B = os.path.join(ROOT, "outputs", "benchmarks")
ART = os.path.join(PAPER, "artwork")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, AMBER, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
LABEL = {"physics_only": "physics expert", "xgboost": "XGBoost (baseline)", "lightgbm": "LightGBM (baseline)",
         "catboost": "CatBoost (baseline)", "mlp": "quantile MLP (baseline)", "tabicl": "TabICLv2 (2026)",
         "tabm": "TabM (2025)", "kumo": "Kumo-Tabular (2026)", "xgboost_aft": "XGBoost AFT (baseline)",
         "equal_average": "physics + learned, equal weights", "stack_constant": "physics + learned, constant weight",
         "cosmos_gated": "reliability-gated (COSMOS)", "cosmos_gated_tuneA": "reliability-gated, gate fitted on tune-A"}


# Figures in order of appearance in the manuscript; files are named Figure_<n> by this order, and
# supplementary figures Figure_S<n>.  make_data_figures.py and make_architecture.py use the same list.
FIGURE_ORDER = ["data_reentry", "data_debris", "data_telemetry", "architecture", "reentry_models", "reentry_outputs",
                "reentry_benchmark", "coverage", "ablation", "cac", "cac_outputs", "c2", "c3", "t1",
                "telemetry_benchmark", "smap_outputs", "twin_state", "twin_forecast", "cost"]
SUPP_ORDER = ["selection", "coverage_width", "class_breakdown"]


def fig_name(key):
    return f"Figure_{FIGURE_ORDER.index(key) + 1}" if key in FIGURE_ORDER else f"Figure_S{SUPP_ORDER.index(key) + 1}"


def style(ax, xlabel=None, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=7.5, colors=INK2)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(ART, f"{name}.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def re_dir(o):
    return os.path.join(B, "reentry_lifetime", f"origin{o}_uncontrolled")


def fig_reentry_models(name=None):
    name = name or fig_name("reentry_models")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    order = ["physics_only", "xgboost", "lightgbm", "catboost", "mlp", "xgboost_aft", "tabm", "tabicl", "kumo",
             "equal_average", "stack_constant", "cosmos_gated_tuneA", "cosmos_gated"]
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021 (origin 2016)", "Test 2011–2015 (origin 2011)")):
        r = pd.read_csv(os.path.join(re_dir(o), "regression_test.csv")).set_index("model")
        aft = json.load(open(os.path.join(re_dir(o), "survival_aft.json")))["test"]
        vals = []
        for m in order:
            if m == "xgboost_aft":
                vals.append(aft["pinball"])
            else:
                vals.append(r.loc[m, "pinball"] if m in r.index else np.nan)
        y = np.arange(len(order))
        col = [ORANGE if m == "cosmos_gated" else (MUTED if m == "physics_only" else
               (GREEN if m in ("tabm", "tabicl", "kumo") else BLUE)) for m in order]
        ax.hlines(y, 0, vals, color=GRID, linewidth=1.2)
        ax.scatter(vals, y, s=34, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        for yy, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + 0.0012, yy, f"{v:.4f}", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(y, [LABEL[m] for m in order], fontsize=7.5, color=INK)
        ax.set_xlim(0, max(v for v in vals if np.isfinite(v)) * 1.18)
        style(ax, "mean pinball loss, log10 days (lower is better)")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, name)


def fig_calibration(name=None):
    name = name or fig_name("coverage")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021", "Test 2011–2015")):
        rules = json.load(open(os.path.join(re_dir(o), "interval_rules.json")))
        r = pd.read_csv(os.path.join(re_dir(o), "regression_test.csv")).set_index("model")
        rows = [("physics ±20 % (operational rule)", rules["physics_only_pm20"]["coverage"], MUTED),
                ("physics ±30 %", rules["physics_only_pm30"]["coverage"], MUTED),
                ("COSMOS raw 80 % quantiles", r.loc["cosmos_gated", "raw_cov80"], BLUE),
                ("COSMOS raw 90 % quantiles", r.loc["cosmos_gated", "raw_cov90"], BLUE),
                ("COSMOS marginal conformal 80 %", r.loc["cosmos_gated_marginal_conformal", "cov80"], GREEN),
                ("COSMOS marginal conformal 90 %", r.loc["cosmos_gated_marginal_conformal", "cov90"], GREEN),
                ("COSMOS Mondrian conformal 80 %", r.loc["cosmos_gated", "cov80"], ORANGE),
                ("COSMOS Mondrian conformal 90 %", r.loc["cosmos_gated", "cov90"], ORANGE)]
        y = np.arange(len(rows))
        ax.barh(y, [v for _, v, _ in rows], color=[c for _, _, c in rows], height=0.6)
        for yy, (_, v, _) in zip(y, rows):
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
        ax.axvline(0.8, color=INK2, linestyle=":", linewidth=0.8)
        ax.axvline(0.9, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_yticks(y, [t for t, _, _ in rows], fontsize=7.2, color=INK)
        ax.set_xlim(0, 1.08)
        style(ax)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].invert_yaxis()
    fig.supxlabel("empirical coverage over all test objects, including those still in orbit after five years "
                  "(dotted 0.8, dashed 0.9 nominal)", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, name)


def fig_ablation(name=None):
    name = name or fig_name("ablation")
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    groups = [("no_physics", "remove physics summaries"), ("no_family", "remove family evidence (C1)"),
              ("no_space_weather", "remove space weather"), ("no_design", "remove design properties"),
              ("plus_rcs_2026", "add 2026 radar cross-section (leak)")]
    for k, (o, col) in enumerate((("2016", BLUE), ("2011", ORANGE))):
        s = json.load(open(os.path.join(re_dir(o), "summary.json")))
        best = s["best_learner_on_tuneA"]
        ab = pd.read_csv(os.path.join(re_dir(o), "ablation_effects_test.csv")).set_index("ablation")
        for i, (g, lab) in enumerate(groups):
            if g not in ab.index:
                continue
            d, lo, hi = ab.loc[g, ["pinball_ablation_minus_full", "ci95_lo", "ci95_hi"]]
            yy = i + (k - 0.5) * 0.3
            ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt="o", color=col, markersize=4, capsize=2, linewidth=1,
                        label=f"origin {o}: {LABEL.get(best, best)}" if i == 0 else None)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(groups)), [lab for _, lab in groups], fontsize=7.5, color=INK)
    ax.invert_yaxis()
    style(ax, "change in test pinball loss (ablation minus full learner); 95 % paired cluster-bootstrap CI")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, name)


def fig_c2(name=None):
    name = name or fig_name("c2")
    df = pd.read_csv(os.path.join(B, "twin_couplings", "c2_cloud_survival.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True)
    for ax, pre, title in zip(axes, ("pred_", "predcal_"), ("(a) raw survival read-off", "(b) calibrated survival (tune-B isotonic maps)")):
        for m, col, lab in (("physics_only", MUTED, "physics expert"), ("cosmos_gated", ORANGE, "reliability-gated (COSMOS)")):
            c = f"{pre}{m}"
            if c not in df:
                continue
            x = df["observed_in_orbit"] / df["catalogued"]
            y = df[c] / df["catalogued"]
            mae = float(np.mean(np.abs(x - y)))
            ax.scatter(x, y, s=12, color=col, alpha=0.75, edgecolor="white", linewidth=0.4, label=f"{lab} (MAE {mae:.3f})")
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        style(ax, "observed fraction of a cloud's fragments still in orbit")
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel("predicted fraction", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, name)


def fig_c3(name=None):
    name = name or fig_name("c3")
    j = json.load(open(os.path.join(B, "twin_couplings", "c3_summary.json")))
    groups = [("non-operational", "non-operational payloads"), ("operational", "operational payloads")]
    pols = [("physics_status_blind", "physics (as of 2026-08-16)", MUTED), ("base_rate_status_blind", "previous-window base rate (as of 2026-08-16)", BLUE),
            ("twin_status_aware_baserate", "status-aware twin (as of 2026-08-16, C3)", ORANGE),
            ("twin_status_aware_recalibrated", "per-status Platt, in-window (not a forecast)", AXIS)]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    for ax, (g, title) in zip(axes, groups):
        e = j["by_status_policy"][g]
        rate = list(e.values())[0]["event_rate"]
        vals = [e[p]["mean_p"] for p, _, _ in pols]
        y = np.arange(len(pols))
        ax.barh(y, vals, color=[c for _, _, c in pols], height=0.55)
        ax.axvline(rate, color=INK, linewidth=1.0, linestyle="--")
        ax.text(rate, 1.0, f" observed {rate:.3f}", fontsize=6.8, color=INK, va="bottom", transform=ax.get_xaxis_transform())
        top = max(max(vals), rate) * 1.6
        ax.set_xlim(0, top)
        for yy, v in zip(y, vals):              # label to the right of both the bar and the observed line
            ax.text(max(v, rate) + 0.02 * top, yy, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(y, [lab for _, lab, _ in pols], fontsize=7.2, color=INK)
        if ax is axes[0]:                       # shared y axis: invert once
            ax.invert_yaxis()
        n = j["by_status"][g]["n"]
        ev = j["by_status"][g]["events"]
        style(ax, "mean predicted P(re-entry within 72 days)")
        ax.set_title(f"{title} (n = {n:,}, {ev} decays)", fontsize=8, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, name)


def fig_t1(name=None):
    name = name or fig_name("t1")
    j = json.load(open(os.path.join(B, "twin_prospective_t1", "t1_summary.json")))
    T = pd.read_csv(os.path.join(B, "twin_prospective_t1", "t1_objects.csv.gz"))
    unc = T["type1"].isin(["D", "R", "C"])
    var = [("physics_history", "physics, history-conditioned (twin)", ORANGE), ("physics_prior", "physics, class prior", MUTED),
           ("lifetime", "lifetime pipeline (gated)", BLUE)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axes[0]
    for i, (v, lab, col) in enumerate(var):
        p, y = T.loc[unc, f"p_{v}_72d"].to_numpy(), T.loc[unc, "y_72d"].to_numpy()
        edges = np.array([0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0])
        k = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        xs, ys, ns = [], [], []
        for b in range(len(edges) - 1):
            s = k == b
            if s.sum() >= 10:
                xs.append(p[s].mean()); ys.append(y[s].mean()); ns.append(s.sum())
        # open markers (no lines across empty bins); area grows with the number of objects in the bin.  Open rather
        # than filled because the three variants' largest bins coincide at the origin and would hide one another
        ax.scatter(xs, ys, s=[12 + 10 * np.log10(n) ** 2 for n in ns], facecolor="none", edgecolor=col, linewidth=1.3,
                   label=lab, zorder=10 - i)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    style(ax, "mean predicted P(re-entry within 72 days)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_ylabel("observed share", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.set_title("(a) reliability, fragments, stages and components", fontsize=8, color=INK, loc="left")
    ax = axes[1]
    e = j["by_horizon"]["72d"]["uncontrolled"]
    obs = e["physics_history"]["events"]
    vals = [e[v]["expected"] for v, _, _ in var]
    y = np.arange(len(var))
    ax.barh(y, vals, color=[c for _, _, c in var], height=0.55)
    ax.axvline(obs, color=INK, linewidth=1.0, linestyle="--")
    ax.text(obs, 1.0, f" observed {obs}", fontsize=6.8, color=INK, va="bottom", transform=ax.get_xaxis_transform())
    for yy, v in zip(y, vals):
        ax.text(v, yy, f" {v:.0f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [lab for _, lab, _ in var], fontsize=7.2, color=INK)
    ax.invert_yaxis()
    style(ax, "expected re-entries in 72 days (sum of probabilities)")
    ax.set_title("(b) expected vs observed count", fontsize=8, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, name)


def fig_cac(name=None):
    name = name or fig_name("cac")
    r = pd.read_csv(os.path.join(B, "conjunction_kelvins_cac", "test_results.csv")).set_index("model")
    g = pd.read_csv(os.path.join(B, "conjunction_kelvins_cac", "gated_persistence_test.csv")).set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
    order = ["persistence", "xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo", "cosmos_gated"]
    lab = {"persistence": "persistence + residual climatology", "xgboost": "XGBoost (baseline)", "lightgbm": "LightGBM (baseline)",
           "catboost": "CatBoost (baseline)", "mlp": "quantile MLP (baseline)", "tabm": "TabM (2025)", "tabicl": "TabICLv2 (2026)",
           "kumo": "Kumo-Tabular (2026)", "cosmos_gated": "reliability-gated (COSMOS)"}
    v = [r.loc[m, "pinball"] for m in order]
    y = np.arange(len(order))
    axes[0].scatter(v, y, s=30, color=[ORANGE if m == "cosmos_gated" else BLUE for m in order], zorder=3)
    axes[0].hlines(y, 0, v, color=GRID)
    for yy, vv in zip(y, v):
        axes[0].text(vv + 0.035 * max(v), yy, f"{vv:.3f}", va="center", fontsize=6.5, color=INK2)
    axes[0].set_xlim(0, max(v) * 1.16)
    axes[0].set_yticks(y, [lab[m] for m in order], fontsize=7.2)
    axes[0].invert_yaxis()
    style(axes[0], "pinball loss of the final-risk distribution")
    axes[0].set_title("(a) probabilistic forecast", fontsize=8, loc="left")
    rows = [("latest_risk_point", "latest risk (challenge baseline)")] + [
        (f"{m}_gated_persistence", f"latest risk gated by {lab[m]}") for m in ("xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo")]
    L = [g.loc[k, "L"] for k, _ in rows]
    y = np.arange(len(rows))
    axes[1].scatter(L, y, s=30, color=[MUTED] + [BLUE] * (len(rows) - 1), zorder=3)
    axes[1].hlines(y, 0, L, color=GRID)
    for yy, vv in zip(y, L):
        axes[1].text(vv + 0.035 * max(L), yy, f"{vv:.3f}", va="center", fontsize=6.5, color=INK2)
    axes[1].set_xlim(0, max(L) * 1.16)
    axes[1].axvline(0.556, color=INK2, linestyle=":", linewidth=0.8)
    axes[1].text(0.556, 1.0, "challenge winner 0.556", fontsize=6.3, color=INK2, ha="center", va="bottom",
                 transform=axes[1].get_xaxis_transform())
    axes[1].set_yticks(y, [t for _, t in rows], fontsize=7.2)
    axes[1].invert_yaxis()
    style(axes[1], "official score L = MSE_HR / F2, clipped (lower is better)")
    axes[1].set_title("(b) official point metric", fontsize=8, loc="left", pad=12)
    fig.tight_layout()
    save(fig, name)


def _pairs(o):
    """Head-to-head test pinball differences A - B (< 0: A better) with paired launch-family CIs."""
    pw = pd.read_csv(os.path.join(re_dir(o), "pairwise_test.csv"))
    d = {(r.a, r.b): (r.pinball_a_minus_b, r.ci95_lo, r.ci95_hi) for r in pw.itertuples()}
    s = json.load(open(os.path.join(re_dir(o), "summary.json")))
    for m, v in s["paired_vs_cosmos_gated"].items():           # stored as other - gated
        d[("cosmos_gated", m)] = (-v["pinball_other_minus_gated"], -v["ci95"][1], -v["ci95"][0])

    def get(a, b):
        if (a, b) in d:
            return d[(a, b)]
        if (b, a) in d:
            v, lo, hi = d[(b, a)]
            return -v, -hi, -lo
        return (np.nan, np.nan, np.nan)
    return get, s["best_learner_on_tuneA"]


def fig_reentry_benchmark(name=None):
    name = name or fig_name("reentry_benchmark")
    rows = [("tabicl", "physics_only"), ("kumo", "physics_only"), ("tabm", "physics_only"), ("cosmos_gated", "physics_only"),
            ("tabicl", "xgboost"), ("kumo", "xgboost"), ("tabicl", "tabm"), ("kumo", "tabm"), ("tabicl", "kumo"),
            ("tabicl", "cosmos_gated"), ("kumo", "cosmos_gated"), ("cosmos_gated", "BEST")]
    short = {"physics_only": "physics", "xgboost": "XGBoost", "tabm": "TabM", "tabicl": "TabICLv2", "kumo": "Kumo-Tabular",
             "cosmos_gated": "gated", "BEST": "its expert"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021 (origin 2016)", "Test 2011–2015 (origin 2011)")):
        get, best = _pairs(o)
        for k, (a, b) in enumerate(rows):
            v, lo, hi = get(a, best if b == "BEST" else b)
            if not np.isfinite(v):
                continue
            sig = lo > 0 or hi < 0
            ax.errorbar(v, k, xerr=[[v - lo], [hi - v]], fmt="o", color=BLUE, mfc=BLUE if sig else "white",
                        markersize=4.5, capsize=2, linewidth=1, zorder=3)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(rows)), [f"{short[a]} minus {short[b]}" for a, b in rows], fontsize=7.4, color=INK)
        style(ax, "difference in test pinball loss (< 0: first model better)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].invert_yaxis()
    axes[1].scatter([], [], color=BLUE, s=18, label="95 % CI excludes 0")
    axes[1].scatter([], [], facecolor="white", edgecolor=BLUE, s=18, label="95 % CI includes 0")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, name)


def fig_telemetry_benchmark(name=None):
    name = name or fig_name("telemetry_benchmark")
    s = pd.read_csv(os.path.join(B, "telemetry_smap_msl", "summary.csv"))
    s["m"] = s["model"].str.split(" ").str[0]
    lab = {"telemanom": "telemanom (published scores)", "chronos2": "Chronos-2, zero-shot (2025)", "ar_ridge": "ridge autoregression",
           "pca": "PCA reconstruction", "knn": "nearest-neighbour window", "random": "random scores"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
    sc_col = (("SMAP", BLUE), ("MSL", ORANGE))
    order = ["telemanom", "chronos2", "ar_ridge", "pca", "knn", "random"]
    ax = axes[0]
    for j, (sc, col) in enumerate(sc_col):
        g = s[s.spacecraft == sc].set_index("m")
        v = [g.loc[m, "pointwise_auc_pr"] if m in g.index else np.nan for m in order]
        ax.scatter(v, np.arange(len(order)) + (j - 0.5) * 0.25, s=24, color=col, zorder=3, label=sc)
    ax.set_yticks(range(len(order)), [lab[m] for m in order], fontsize=7.3, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.65)
    style(ax, "per-channel mean AUC-PR of anomalous steps")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("(a) ranking of anomalous steps", fontsize=8, color=INK, loc="left")
    ax = axes[1]
    order = ["chronos2", "ar_ridge", "pca", "knn"]
    for j, (sc, col) in enumerate(sc_col):
        g = s[s.spacecraft == sc].set_index("m")
        yy = np.arange(len(order)) + (j - 0.5) * 0.25
        ax.scatter([g.loc[m, "esa_ew_f0.5"] for m in order], yy, s=24, color=col, zorder=3, label=f"{sc}: detector")
        ax.scatter([g.loc[m, "esa_ew_f0.5_late"] for m in order], yy, s=24, facecolor="white", edgecolor=col, zorder=3,
                   label=f"{sc}: same alarm time, placed at the end")
    ax.set_yticks(range(len(order)), [lab[m] for m in order], fontsize=7.3, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.5)
    style(ax, "corrected event-wise F0.5 (ESA-ADB metric)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
    ax.set_title("(b) alarms, against a late-alarm reference", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, name)


def fig_cost(name=None):
    """Measured wall-clock of the origin-2016 run.  Panel (a) is the fit-and-predict loop of
    run.fit_learners: (four refits for seeds 1-4, plus predictions on tune-A, tune-B and the test
    cohort for all five seeds) / 5.  The seed-0 fit happened during tuning and is not timed.
    Panel (b) is one call predicting the test cohort, which for the in-context models includes
    encoding the training context (a fixed cost per call, not per object)."""
    name = name or fig_name("cost")
    s = json.load(open(os.path.join(re_dir("2016"), "summary.json")))
    f, n_test = s["fits"], s["sizes"]["test"]
    order = ["xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo"]
    loop = {m: f[m]["fit_seconds_per_seed"] for m in order}
    call = {m: f[m]["predict_ms_per_1000_objects"] * n_test / 1e6 for m in order}
    loop["physics_only"] = s["physics_seconds"]["train"]
    call["physics_only"] = s["physics_seconds"]["test"]
    order = ["physics_only"] + order
    col = [MUTED if m == "physics_only" else (GREEN if m in ("tabm", "tabicl", "kumo") else BLUE) for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    panels = ((axes[0], loop, "seconds per seed (log scale)",
               "(a) fit and predict, per seed\n(physics: its Monte-Carlo training run)"),
              (axes[1], call, "seconds for one call (log scale)",
               f"(b) one call over the {n_test:,} test objects\n(in-context models: with context encoding)"))
    for ax, d, xl, title in panels:
        v = [d[m] for m in order]
        y = np.arange(len(order))
        ax.hlines(y, 1e-3, v, color=GRID, linewidth=1.2)
        ax.scatter(v, y, s=30, color=col, zorder=3)
        for yy, vv in zip(y, v):
            ax.text(vv * 1.3, yy, f"{vv:.3g} s", va="center", fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, max(v) * 60)
        ax.set_yticks(y, [LABEL[m] for m in order], fontsize=7.3, color=INK)
        style(ax, xl)
        ax.set_title(title, fontsize=7.6, color=INK, loc="left")
    axes[0].invert_yaxis()
    fig.text(0.01, -0.06, "Origin 2016. Physics expert and the CPU learners: a two-socket Intel Xeon 6980P node (512 threads); "
             "TabM and Kumo-Tabular: one Intel Arc Pro B60; TabICLv2: CPU.", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, name)


def fig_class_breakdown(name=None):
    name = name or fig_name("class_breakdown")
    models = ["physics_only", "xgboost", "tabm", "tabicl", "kumo", "cosmos_gated"]
    cls = (("D", "fragments", BLUE), ("R", "rocket stages", ORANGE), ("C", "components", GREEN))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021", "Test 2011–2015")):
        d = pd.read_csv(os.path.join(re_dir(o), "class_breakdown_test.csv"))
        ns = []
        for j, (c, lab, col) in enumerate(cls):
            g = d[d.type1 == c].set_index("model")
            v = [g.loc[m, "pinball"] if m in g.index else np.nan for m in models]
            ax.scatter(v, np.arange(len(models)) + (j - 1) * 0.22, s=22, color=col, zorder=3,
                       label=lab if ax is axes[0] else None)
            ns.append(f"{int(g['n'].iloc[0]):,}" if "n" in g else "")
        ax.set_yticks(range(len(models)), [LABEL[m] for m in models], fontsize=7.3, color=INK)
        style(ax, "test pinball loss by object class")
        ax.set_title(f"{title} ({ns[0]}, {ns[1]}, {ns[2]})", fontsize=7.2, color=INK, loc="left")
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, name)


def place_labels(ax, pts, fs=6.2, dx=4, step=8.5):
    """Labels to the right of points, nudged down (in points) when they would overlap a label already placed."""
    ax.margins(x=0.12, y=0.12)
    ax.figure.canvas.draw()
    tr = ax.transData
    placed = []
    k = 72.0 / ax.figure.dpi                                       # display pixels -> points
    y0, y1 = (tr.transform((0, v))[1] * k for v in ax.get_ylim())
    for x, y, text in sorted(pts, key=lambda p: -p[1]):
        px, py = (v * k for v in tr.transform((x, y)))
        up = (py - min(y0, y1)) < 0.3 * abs(y1 - y0)               # near the bottom: stack upwards
        off = 2.0
        while any(abs((py + off) - q) < step and abs(px - r) < 90 for r, q in placed):
            off += step if up else -step
        placed.append((px, py + off))
        moved = abs(off - 2.0) > 1e-9
        ax.annotate(text, (x, y), xytext=(dx + (10 if moved else 0), off), textcoords="offset points", fontsize=fs, color=INK2,
                    va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 3}
                    if moved else None)


def fig_coverage_width(name=None):
    name = name or fig_name("coverage_width")
    models = ["physics_only", "xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo", "cosmos_gated"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021", "Test 2011–2015")):
        r = pd.read_csv(os.path.join(re_dir(o), "regression_test.csv")).set_index("model")
        for m in models:
            col = ORANGE if m == "cosmos_gated" else (MUTED if m == "physics_only" else (GREEN if m in ("tabm", "tabicl", "kumo") else BLUE))
            x, y = r.loc[m, "width90_log10"], r.loc[m, "cov90"]
            ax.scatter(x, y, s=26, color=col, zorder=3)
        ax.axhline(0.9, color=INK2, linestyle="--", linewidth=0.8)
        place_labels(ax, [(r.loc[m, "width90_log10"], r.loc[m, "cov90"], LABEL[m].replace(" (baseline)", "")) for m in models])
        style(ax, "mean width of the 90 % conformal interval (log10 days)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].set_ylabel("empirical 90 % coverage (dashed: nominal)", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, name)


def fig_selection(name=None):
    name = name or fig_name("selection")
    learners = ["xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, o, title in zip(axes, ("2016", "2011"), ("Origin 2016", "Origin 2011")):
        s = json.load(open(os.path.join(re_dir(o), "summary.json")))
        r = pd.read_csv(os.path.join(re_dir(o), "regression_test.csv")).set_index("model")
        for m in learners:
            x, y = s["tuneA_pinball"][m], r.loc[m, "pinball"]
            col = GREEN if m in ("tabm", "tabicl", "kumo") else BLUE
            ax.scatter(x, y, s=28, color=col, zorder=3, edgecolor=INK if m == s["best_learner_on_tuneA"] else "white", linewidth=1.0)
        style(ax, "tune-A pinball loss (selection cohort)")
        place_labels(ax, [(s["tuneA_pinball"][m], r.loc[m, "pinball"], LABEL[m].replace(" (baseline)", "")) for m in learners])
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_ylabel("test pinball loss", fontsize=7.5, color=INK2)
        ax.set_title(f"{title}: selected {LABEL[s['best_learner_on_tuneA']]} (outlined)", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    for fn in (fig_reentry_models, fig_calibration, fig_ablation, fig_c2, fig_c3, fig_cac, fig_t1, fig_reentry_benchmark,
               fig_telemetry_benchmark, fig_cost, fig_class_breakdown, fig_coverage_width, fig_selection):
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print("skip", fn.__name__, type(e).__name__, e)
