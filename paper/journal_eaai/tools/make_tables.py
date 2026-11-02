"""Manuscript tables, generated from the result files so that no value is typed by hand.

    python paper/journal_eaai/tools/make_tables.py

Each table is written to ``paper/journal_eaai/build/tables/<key>.md``.  A section asks for one with
the marker ``{{TABLE:key}}``, which ``render.py`` replaces before the placeholder pass; a marker
with no generated table aborts the build, so a section cannot silently lose its evidence.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
B = os.path.join(ROOT, "outputs", "benchmarks")
OUT = os.path.join(PAPER, "build", "tables")
ORIGINS = ("2016", "2011")
T = {}


def re_dir(o):
    return os.path.join(B, "reentry_lifetime", f"origin{o}_uncontrolled")


def md(rows, header, align=None):
    a = align or (["---"] * len(header))
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(a) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(v, spec=".4f"):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


LAB = {"physics_only": "Physics prior only ($g=1$)", "xgboost": "XGBoost", "lightgbm": "LightGBM",
       "catboost": "CatBoost", "mlp": "Quantile multilayer perceptron", "aft": "XGBoost accelerated failure time",
       "tabm": "TabM (2025)", "tabicl": "TabICLv2 (2026)", "kumo": "Kumo-Tabular (2026)",
       "equal_average": "Equal-weight average ($g=1/2$)", "stack_constant": "Learned constant gate",
       "cosmos_gated_tuneA": "Reliability gate fitted on tune-A", "cosmos_gated": "Reliability-gated mixture (proposed)"}


def t_moe():
    """The eight-arm mixture-of-experts comparison: the paper's central claim."""
    order = ["physics_only", "tabm", "tabicl", "kumo", "xgboost", "equal_average", "stack_constant",
             "cosmos_gated_tuneA", "cosmos_gated"]
    kind = {"physics_only": "single expert", "tabm": "single expert", "tabicl": "single expert",
            "kumo": "single expert", "xgboost": "baseline", "equal_average": "fixed fusion",
            "stack_constant": "learned fusion", "cosmos_gated_tuneA": "learned fusion", "cosmos_gated": "learned fusion"}
    r = {o: pd.read_csv(os.path.join(re_dir(o), "regression_test.csv")).set_index("model") for o in ORIGINS}
    rows = []
    for m in order:
        row = [LAB[m], kind[m]]
        for o in ORIGINS:
            d = r[o]
            row += ([f(d.loc[m, "pinball"]), f(d.loc[m, "mae_log10"], ".3f"), f(d.loc[m, "cov90"], ".3f")]
                    if m in d.index else ["", "", ""])
        rows.append(row)
    best = {o: min(r[o].loc[m, "pinball"] for m in order if m in r[o].index) for o in ORIGINS}
    T["moe"] = md(rows, ["Fusion arm", "Type", "Pinball", "MAE", "Cov.", "Pinball", "MAE", "Cov."],
                  ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:"])
    T["moe_note"] = (f"Lowest pinball loss: {f(best['2016'])} at origin 2016 and {f(best['2011'])} at origin 2011; "
                     "the proposed gate attains neither.")


def t_reentry_pairs():
    """Paired launch-family cluster bootstraps of the differences that matter."""
    rows = []
    for o in ORIGINS:
        p = pd.read_csv(os.path.join(re_dir(o), "pairwise_test.csv"))
        for _, x in p.iterrows():
            sig = "yes" if (x.ci95_lo > 0 or x.ci95_hi < 0) else "no"
            rows.append([f"{o}", f"{LAB.get(x.a, x.a)} $-$ {LAB.get(x.b, x.b)}".replace(" ($g=1$)", "").replace(" (proposed)", ""),
                         f(x.pinball_a_minus_b, "+.4f"), f"[{f(x.ci95_lo, '+.4f')}, {f(x.ci95_hi, '+.4f')}]", sig])
    T["reentry_pairs"] = md(rows, ["Origin", "Comparison", "$\\Delta$ pinball", "95 % CI", "Excludes 0"],
                            ["---", "---", "--:", "--:", ":-:"])


def t_km():
    rows = []
    for o in ORIGINS:
        k = pd.read_csv(os.path.join(re_dir(o), "km_descriptive_test.csv"))
        nm = {"all": "all uncontrolled", "D": "fragments", "R": "rocket stages", "C": "components"}
        for _, x in k.iterrows():
            rows.append([o, nm.get(x.group, x.group), f"{int(x.n):,}", f"{int(x.events):,}",
                         f"{x.censored_fraction:.1%}", ("not reached" if not np.isfinite(x.median_days) else f"{x.median_days:,.0f}"),
                         f(x.surv_30d, ".3f"), f(x.surv_1yr, ".3f"), f(x.surv_5yr, ".3f")])
    T["km"] = md(rows, ["Origin", "Group", "$n$", "Decays", "Censored", "Median (d)", "$S$(30 d)", "$S$(1 yr)", "$S$(5 yr)"],
                 ["---", "---", "--:", "--:", "--:", "--:", "--:", "--:", "--:"])


def t_coverage():
    """Empirical coverage, width and efficiency of each interval rule, decayed objects separately."""
    rows = []
    nm = {"physics_only_pm20": "Physics median ±20 % (operational rule)",
          "physics_only_pm30": "Physics median ±30 %",
          "physics_only_conformal80": "Physics prior, conformal 80 %",
          "physics_only_conformal90": "Physics prior, conformal 90 %",
          "cosmos_gated_pm20": "Gated median ±20 %", "cosmos_gated_pm30": "Gated median ±30 %",
          "cosmos_gated_conformal80": "Gated mixture, conformal 80 %",
          "cosmos_gated_conformal90": "Gated mixture, conformal 90 %"}
    j = {o: json.load(open(os.path.join(re_dir(o), "interval_rules.json"))) for o in ORIGINS}
    for k, lab in nm.items():
        row = [lab]
        for o in ORIGINS:
            e = j[o].get(k)
            row += [f(e["coverage"], ".3f"), f(e["coverage_decayed"], ".3f"), f(e["width_log10"], ".2f")] if e else ["", "", ""]
        rows.append(row)
    T["coverage"] = md(rows, ["Interval rule", "Cov. all", "Cov. decayed", "Width", "Cov. all", "Cov. decayed", "Width"],
                       ["---", "--:", "--:", "--:", "--:", "--:", "--:"])


def t_temporal():
    rows = []
    for o in ORIGINS:
        d = pd.read_csv(os.path.join(re_dir(o), "temporal_coverage_test.csv"))
        for m, g in d.groupby("model"):
            g = g.sort_values("year")
            rows.append([o, LAB.get(m, m).replace(" ($g=1$)", "").replace(" (proposed)", ""),
                         " · ".join(f"{v:.3f}" for v in g.cov90), f"{g.cov90.iloc[0] - g.cov90.iloc[-1]:+.3f}"])
    T["temporal"] = md(rows, ["Origin", "Model", "90 % coverage, first to last test year", "First $-$ last"],
                       ["---", "---", "---", "--:"])


def t_cac():
    r = pd.read_csv(os.path.join(B, "conjunction_kelvins_cac", "test_results.csv")).set_index("model")
    order = ["persistence", "xgboost", "lightgbm", "catboost", "mlp", "tabm", "tabicl", "kumo",
             "equal_average", "stack_constant", "cosmos_gated"]
    lab = dict(LAB, persistence="Persistence + residual climatology")
    rows = []
    for m in order:
        if m not in r.index:
            continue
        L = r.loc[m, "L_clip"]
        rows.append([lab.get(m, m), f(r.loc[m, "pinball"], ".3f"), f(r.loc[m, "auc_pr_high"], ".3f"),
                     f(r.loc[m, "auc_roc_high"], ".3f"), f(r.loc[m, "F2"], ".3f"),
                     ("$\\infty$" if not np.isfinite(L) else f(L, ".2f")), f(r.loc[m, "cov90"], ".3f"),
                     f(r.loc[m, "cov90_high"], ".3f")])
    if "latest_risk_point" in r.index:
        rows.append(["Latest-risk point forecast (challenge baseline)", "", "", "", "",
                     f(r.loc["latest_risk_point", "L_clip"], ".2f"), "", ""])
    T["cac"] = md(rows, ["Model", "Pinball", "AUC-PR", "AUC-ROC", "$F_2$", "$L$", "Cov. 90 %", "Cov. high risk"],
                  ["---", "--:", "--:", "--:", "--:", "--:", "--:", "--:"])
    p = pd.read_csv(os.path.join(B, "conjunction_kelvins_cac", "pinball_pairs_test.csv"))
    T["cac_pairs"] = md([[f"{LAB.get(x.a, x.a)} $-$ {lab.get(x.b, x.b)}".replace(" (2026)", ""),
                          f(x.pinball_a_minus_b, "+.3f"), f"[{f(x.ci95_lo, '+.3f')}, {f(x.ci95_hi, '+.3f')}]"]
                         for _, x in p.iterrows()],
                        ["Comparison", "$\\Delta$ pinball", "95 % CI"], ["---", "--:", "--:"])


def t_telemetry():
    s = pd.read_csv(os.path.join(B, "telemetry_smap_msl", "summary.csv"))
    s["m"] = s["model"].str.split(" ").str[0]
    lab = {"telemanom": "telemanom (published scores)", "chronos2": "Chronos-2, zero-shot",
           "ar_ridge": "Ridge autoregression", "pca": "Principal-component reconstruction",
           "knn": "Nearest-neighbour window", "random": "Random scores (floor)",
           "always_on": "Always-on alarm (reference)"}
    rows = []
    for m in ["telemanom", "chronos2", "ar_ridge", "pca", "knn", "random", "always_on"]:
        row = [lab[m]]
        for sc in ("SMAP", "MSL"):
            g = s[(s.spacecraft == sc) & (s.m == m)]
            if len(g) == 0:
                row += ["", "", ""]
                continue
            x = g.iloc[0]
            row += [f(x.get("pointwise_auc_pr"), ".3f"), f(x.get("esa_ew_f0.5"), ".3f"),
                    f(x.get("esa_ew_f0.5_late"), ".3f")]
        rows.append(row)
    T["telemetry"] = md(rows, ["Detector", "AUC-PR", "$F_{0.5}$", "Late ref.", "AUC-PR", "$F_{0.5}$", "Late ref."],
                        ["---", "--:", "--:", "--:", "--:", "--:", "--:"])


def t_couplings():
    rows = []
    ab = {o: pd.read_csv(os.path.join(re_dir(o), "ablation_effects_test.csv")).set_index("ablation") for o in ORIGINS}
    r = []
    for o in ORIGINS:
        d = ab[o]
        if "no_family" in d.index:
            x = d.loc["no_family"]
            r.append(f"{x.pinball_ablation_minus_full:+.4f} [{x.ci95_lo:+.4f}, {x.ci95_hi:+.4f}] at {o}")
    rows.append(["C1", "debris $\\rightarrow$ re-entry", "family features removed from the expert",
                 "; ".join(r), "yes at 2016, not at 2011"])
    c2 = pd.read_csv(os.path.join(B, "twin_couplings", "c2_bootstrap.csv")).set_index("comparison")
    # difference rows are written "a - b"; ablation variants are not candidates for the headline
    cand = [k for k in c2.index if k.startswith(("pred_", "predcal_")) and " - " not in k
            and not any(t in k for t in ("_no_", "_plus_", "_tuneA"))]
    best = min(cand, key=lambda k: c2.loc[k, "mae_fraction"])
    rows.append(["C2", "re-entry $\\rightarrow$ debris", "cloud population against catalogue counts",
                 f"best mean absolute error {c2.loc[best, 'mae_fraction']:.3f} "
                 f"[{c2.loc[best, 'ci95_lo']:.3f}, {c2.loc[best, 'ci95_hi']:.3f}] of a cloud ({best})",
                 "experts not separable"])
    c3 = json.load(open(os.path.join(B, "twin_couplings", "c3_summary.json")))
    d = c3["brier_difference_ci95_launch_bootstrap"]["twin_status_aware_baserate_minus_physics_status_blind"]
    rows.append(["C3", "health $\\rightarrow$ re-entry", "status-aware against status-blind Brier score",
                 f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", "not significant"])
    rows.append(["C4", "health $\\times$ debris", "no public paired dataset", "not evaluated", "demonstration only"])
    T["couplings"] = md(rows, ["Coupling", "Direction", "What is measured", "Measured effect", "Verdict"])


def t_datasets():
    rows = [["Catalogue (re-entry, debris)", "General Catalog of Artificial Space Objects, release 1.8.8; CelesTrak catalogue and space weather",
             "real", "{{data.cohort.rows_out:,}} objects below 2000 km perigee; {{data.cohort.events:,}} natural decays",
             "Creative Commons Attribution 4.0 / CelesTrak policy"],
            ["Conjunctions (debris)", "European Space Agency Kelvins collision-avoidance challenge",
             "real, anonymised", "{{cac.data.train_events_kept:,}} training and {{cac.data.test_events:,}} official test events",
             "Creative Commons Attribution 4.0"],
            ["Telemetry (health)", "National Aeronautics and Space Administration Soil Moisture Active Passive and Mars Science Laboratory",
             "real", "{{smap.config.channels}} channels, {{smap.SMAP.ar_ridge.events:.0f}} and {{smap.MSL.ar_ridge.events:.0f}} labelled sequences",
             "as stated by the authors"],
            ["Thermospheric density", "NRLMSIS 2.1 via pymsis, tabulated", "derived", "density table used by the physics prior",
             "open source"]]
    T["datasets"] = md(rows, ["Role", "Source", "Class", "Extent used", "Licence"])


def t_metrics():
    rows = [["Mean pinball loss", "quality of the whole predictive distribution over seven quantiles", "re-entry, conjunction"],
            ["Mean absolute error of the median", "point accuracy, $\\log_{10}$ days", "re-entry"],
            ["Empirical coverage, width, Winkler score", "honesty and sharpness of the intervals", "re-entry, conjunction"],
            ["Harrell's concordance", "ranking over all objects, censored included", "re-entry"],
            ["AUC-PR, AUC-ROC", "discrimination at a horizon, on the raw probabilities", "all three modules"],
            ["Brier score, expected calibration error", "probability calibration, after isotonic mapping", "re-entry, conjunction"],
            ["Expected over observed count", "absolute bias of a probability forecast", "couplings C3, T1"],
            ["$L=\\mathrm{MSE}_{\\mathrm{HR}}/F_2$", "the challenge's official collision-risk metric", "conjunction"],
            ["Corrected event-wise $F_{0.5}$", "alarm quality, charging alarm time spent on nominal data", "health"],
            ["Kaplan-Meier survival", "censoring-aware description of the cohort", "re-entry"],
            ["Cluster bootstrap, 2,000 replicates", "intervals resampling launch families, launches or clouds", "all"]]
    T["metrics"] = md(rows, ["Metric", "What it measures", "Where used"])


def t_gap():
    """Research gap against prior machine-learning lifetime work.

    Every cell is taken from the verified reading recorded in docs/RELATED_WORK.md; where a study
    could not be read in full the cell says so rather than guessing.
    """
    rows = [["Salmaso et al. [@salmaso2023machine]", "417 bodies; test 15 campaign objects", "decayed only",
             "point epoch", "interleaved in time", "none"],
            ["Lotti et al. [@lotti2025exploring]", "1,783 after filtering; 1,435 train", "decayed only",
             "point lifetime", "random, stratified", "none"],
            ["Jung et al. [@jung2021recurrent]", "not readable (closed access)", "not readable",
             "point epoch", "not readable", "none"],
            ["Shambaugh [@shambaugh2026validation]", "1,283 objects", "decayed only",
             "distribution", "backtest, calibrated on itself", "none"],
            ["This work", "{{data.cohort.rows_out:,}} objects, two periods", "censoring-aware",
             "seven quantiles + intervals", "forward in time, two origins", "three tested"]]
    T["gap"] = md(rows, ["Study", "Scale", "Censoring", "Output", "Evaluation split", "Cross-module couplings"])


def main():
    for fn in (t_moe, t_reentry_pairs, t_km, t_coverage, t_temporal, t_cac, t_telemetry, t_couplings,
               t_datasets, t_metrics, t_gap):
        fn()
    os.makedirs(OUT, exist_ok=True)
    for k, v in T.items():
        open(os.path.join(OUT, f"{k}.md"), "w", encoding="utf-8").write(v + "\n")
    print(f"wrote {len(T)} tables to {OUT}")
    for k in sorted(T):
        print(f"  {k:16s} {len(T[k].splitlines())} lines")


if __name__ == "__main__":
    main()
