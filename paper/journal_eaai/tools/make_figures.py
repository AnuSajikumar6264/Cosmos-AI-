"""Figures specific to the Engineering Applications of Artificial Intelligence manuscript.

    python paper/journal_eaai/tools/make_figures.py

Writes into ``paper/journal_eaai/artwork_eaai/`` (vector PDF + 300 dpi PNG).  Figures reused from
the long-form manuscript are referenced directly from ``paper/journal_v3_twin/artwork/``.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
V3T = os.path.join(ROOT, "paper", "journal_v3_twin", "tools")
sys.path.insert(0, V3T)
from make_figures import (AMBER, AXIS, B, BLUE, GREEN, GRID, INK, INK2, LABEL, MUTED, ORANGE,  # noqa: E402
                          place_labels, plt, style)

ART = os.path.join(PAPER, "artwork_eaai")
CAP = np.log10(1826.25)
DAY = [(np.log10(1.0), "1 d"), (np.log10(30.44), "1 mo"), (np.log10(365.25), "1 yr"), (CAP, "5 yr"),
       (np.log10(36525.0), "100 yr")]


def re_dir(o):
    return os.path.join(B, "reentry_lifetime", f"origin{o}_uncontrolled")


def save(fig, name):
    os.makedirs(ART, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(ART, f"{name}.{ext}"), bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote", name)


def fig_km():
    """Kaplan-Meier survival of the test cohorts: the censoring the learned models must respect."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    groups = (("all", "all uncontrolled", INK), ("D", "fragments", BLUE), ("R", "rocket stages", ORANGE),
              ("C", "components", GREEN))
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021", "Test 2011–2015")):
        C = np.load(os.path.join(re_dir(o), "km_curves_test.npz"))
        tab = pd.read_csv(os.path.join(re_dir(o), "km_descriptive_test.csv")).set_index("group")
        for key, lab, col in groups:
            if f"{key}__t" not in C.files:
                continue
            t, s = C[f"{key}__t"], C[f"{key}__s"]
            x = np.log10(np.maximum(t, 0.5))
            ax.step(x, s, where="post", color=col, linewidth=1.4 if key == "all" else 1.0,
                    label=f"{lab} (n = {int(tab.loc[key, 'n']):,}, {tab.loc[key, 'censored_fraction']:.0%} censored)")
        ax.axvline(CAP, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_xticks([v for v, _ in DAY], [s for _, s in DAY])
        ax.set_xlim(-0.3, 4.6)
        ax.set_ylim(0, 1.02)
        style(ax, "time since the canonical-orbit epoch")
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=6.3, loc="lower left")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].set_ylabel("Kaplan–Meier survival", fontsize=8, color=INK2)
    fig.text(0.5, -0.07, "Dashed line: the five-year cap of the learning target. Right-censored objects are those "
             "still in orbit at the snapshot or whose phase ended without natural decay.",
             ha="center", fontsize=6.4, color=INK2)
    fig.tight_layout()
    save(fig, "eaai_km")


def fig_temporal():
    """Conditional coverage over time: the conformal guarantee is marginal, and it drifts."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021", "Test 2011–2015")):
        d = pd.read_csv(os.path.join(re_dir(o), "temporal_coverage_test.csv"))
        for i, (m, g) in enumerate(d.groupby("model")):
            col = MUTED if m == "physics_only" else (ORANGE if m == "cosmos_gated" else GREEN)
            g = g.sort_values("year")
            ax.plot(g.year, g.cov90, marker="o", markersize=4, linewidth=1.3, color=col,
                    label=LABEL.get(m, m))
        ax.axhline(0.9, color=INK2, linestyle="--", linewidth=0.9)
        ax.set_ylim(0.65, 1.02)
        ax.set_xticks(sorted(d.year.unique()))
        style(ax, "calendar year of the canonical-orbit epoch")
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].set_ylabel("empirical 90 % coverage", fontsize=8, color=INK2)
    fig.text(0.5, -0.07, "Dashed line: the nominal level. Marginal conformal coverage holds over the whole cohort; "
             "its drift within the 2016 period is evidence against exchangeability over time.",
             ha="center", fontsize=6.4, color=INK2)
    fig.tight_layout()
    save(fig, "eaai_temporal")


MOE = [("physics_only", "physics prior only (g = 1)"), ("tabm", "TabM only"), ("tabicl", "TabICLv2 only"),
       ("kumo", "Kumo-Tabular only"), ("xgboost", "XGBoost (gradient-boosted baseline)"),
       ("equal_average", "equal-weight average (g = 1/2)"), ("stack_constant", "learned constant gate (no reliability features)"),
       ("cosmos_gated_tuneA", "reliability gate fitted on tune-A"), ("cosmos_gated", "reliability-gated mixture (proposed)")]


def fig_moe():
    """The eight-arm mixture-of-experts comparison, which is the paper's central AI claim."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, o, title in zip(axes, ("2016", "2011"), ("Test 2016–2021 (origin 2016)", "Test 2011–2015 (origin 2011)")):
        r = pd.read_csv(os.path.join(re_dir(o), "regression_test.csv")).set_index("model")
        v = [r.loc[m, "pinball"] if m in r.index else np.nan for m, _ in MOE]
        y = np.arange(len(MOE))
        col = [ORANGE if m == "cosmos_gated" else (MUTED if m == "physics_only" else
               (GREEN if m in ("tabm", "tabicl", "kumo") else BLUE)) for m, _ in MOE]
        best = np.nanmin(v)
        ax.axvline(best, color=GREEN, linestyle=":", linewidth=0.9)
        ax.hlines(y, 0, v, color=GRID, linewidth=1.2)
        ax.scatter(v, y, s=32, color=col, zorder=3, edgecolor="white", linewidth=0.7)
        for yy, vv in zip(y, v):
            if np.isfinite(vv):
                ax.text(vv + 0.0015, yy, f"{vv:.4f}", va="center", fontsize=6.4, color=INK2)
        ax.set_yticks(y, [lab for _, lab in MOE], fontsize=7.2, color=INK)
        ax.set_xlim(0, np.nanmax(v) * 1.2)
        style(ax)
        ax.set_title(title, fontsize=8.3, color=INK, loc="left")
    axes[0].invert_yaxis()
    fig.supxlabel("mean pinball loss on the test cohort (lower is better)", fontsize=8, color=INK2)
    fig.text(0.5, -0.05, "Dotted line: the best arm at that origin. The proposed gate never attains it, which is the "
             "paper's central negative result.", ha="center", fontsize=6.4, color=INK2)
    fig.tight_layout()
    save(fig, "eaai_moe")


def fig_gate():
    """What the learned gate does: its weight on the physics prior, and its fitted coefficients."""
    names = ["A/m from design", "A/m from class design prior", "physics spread", "log eccentricity",
             "perigee < 300 km", "300–600 km", "fragment", "rocket stage", "payload", "P(at the 5-yr cap)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    w = 0.36
    for k, (o, col) in enumerate((("2016", BLUE), ("2011", ORANGE))):
        s = json.load(open(os.path.join(re_dir(o), "summary.json")))["gates"]["cosmos_gated"]
        by = s["test_weight_by_am_source"]
        keys = ["design", "class_prior_sbm"]
        vals = [by.get(x, np.nan) for x in keys] + [s["test_weight_mean"]]
        ax.bar(np.arange(3) + (k - 0.5) * w, vals, width=w, color=col, label=f"origin {o}")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i + (k - 0.5) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks(range(3), ["design A/m", "breakup-model A/m", "all test objects"], fontsize=7.2)
    ax.set_ylim(0, 0.17)
    style(ax, None, xgrid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_ylabel("mean weight on the physics prior", fontsize=7.6, color=INK2)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.set_title("(a) what the gate decides", fontsize=8.2, color=INK, loc="left")
    ax = axes[1]
    for k, (o, col) in enumerate((("2016", BLUE), ("2011", ORANGE))):
        th = json.load(open(os.path.join(re_dir(o), "summary.json")))["gates"]["cosmos_gated"]["theta"]
        ax.scatter(th[1:], np.arange(len(names)) + (k - 0.5) * 0.22, s=24, color=col, label=f"origin {o}")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(names)), names, fontsize=7, color=INK)
    ax.invert_yaxis()
    style(ax, "fitted gate coefficient (positive: more weight on physics)")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.set_title("(b) fitted reliability coefficients", fontsize=8.2, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "eaai_gate")


if __name__ == "__main__":
    for fn in (fig_km, fig_temporal, fig_moe, fig_gate):
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print("skip", fn.__name__, type(e).__name__, e)
