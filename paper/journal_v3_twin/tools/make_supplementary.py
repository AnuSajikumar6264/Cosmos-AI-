"""Supplementary tables of the twin manuscript, written directly from the result files.

    python paper/journal_v3_twin/tools/make_supplementary.py

Writes build/supplementary.md and build/supplementary.docx.  Every table states its source file.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
B = os.path.join(ROOT, "outputs", "benchmarks")
RE = os.path.join(B, "reentry_lifetime")
out = []


def md(df: pd.DataFrame, digits=4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def table(label, title, df, src, digits=4, note=""):
    if df is None or len(df) == 0:
        return
    rel = ", ".join(os.path.relpath(s, ROOT) for s in (src if isinstance(src, list) else [src]))
    out.append(f"## Table {label}. {title}\n\n{md(df, digits)}\n\nSource: `{rel}`. {note}\n")


def both(fn):
    frames = []
    for o in ("2016", "2011"):
        p = os.path.join(RE, f"origin{o}_uncontrolled", fn)
        if os.path.isfile(p):
            d = pd.read_csv(p)
            d.insert(0, "origin", o)
            frames.append((d, p))
    return (pd.concat([f for f, _ in frames], ignore_index=True), [p for _, p in frames]) if frames else (None, [])


out.append("# Supplementary material: COSMOS-AI digital twin\n\nAll tables are generated from the result files named "
           "under each table (`paper/journal_v3_twin/tools/make_supplementary.py`).\n")

SUPP_FIGS = [("S1", "Tune-A pinball loss, used to select the expert, against test pinball loss, every learner. The selected "
                    "learner is outlined. A tuning cohort five to seven years older than the test cohort did not rank the "
                    "learners as the test cohort did."),
             ("S2", "Empirical 90 % coverage against mean width of the 90 % Mondrian conformal interval, for the physics "
                    "expert, the six learners and the gated model (data of Table S0; the fixed-weight fusions and the "
                    "ablations of Table S0 are not plotted). The dashed line is the nominal level."),
             ("S3", "Re-entry test pinball loss by object class for the main models (post hoc; data of Table S2). The "
                    "counts in each panel title are the fragments, rocket stages and components of that test cohort.")]
out.append("## Supplementary figures\n")
for k, cap in SUPP_FIGS:
    f = os.path.join(PAPER, "artwork", f"Figure_{k}.png")
    if os.path.isfile(f):
        out.append(f"![**Figure {k}.** {cap}]({f}){{width=100%}}\n")

cols = ["origin", "model", "pinball", "mae_log10", "rmse_log10", "medae_days_decayed", "cov80", "cov90", "width90_log10",
        "winkler90", "raw_cov90", "c_index"]
d, src = both("regression_test.csv")
table("S0", "Re-entry, all candidates (run 2, the reported run)", d[[c for c in cols if c in d]] if d is not None else None, src)
frames, srcs = [], []
for o in ("2016", "2011"):
    p = os.path.join(RE, f"origin{o}_uncontrolled", "run1_learner_set_v1", "regression_test.csv")
    if os.path.isfile(p):
        x = pd.read_csv(p); x.insert(0, "origin", o); frames.append(x); srcs.append(p)
if frames:
    d = pd.concat(frames, ignore_index=True)
    table("S1", "Re-entry, first learner set (run 1; XGBoost, LightGBM, CatBoost, MLP, TabICLv2; deviation record)",
          d[[c for c in cols if c in d]], srcs, note="The expert was selected among all five learners (the MLP at both origins).")
d, src = both("class_breakdown_test.csv")
if d is not None:
    table("S2", "Re-entry test pinball loss by object class (post hoc)",
          d.pivot_table(index=["origin", "model"], columns="type1", values="pinball").reset_index(), src,
          note="D fragments, R rocket stages, C components.")
d, src = both("gate_all_learners.csv")
table("S3", "Post-hoc sensitivity analysis (added after the origin-2016 results were seen): the reliability gate applied to every learner", d, src,
      note="gain > 0: gating helps; nominal 95 % cluster-bootstrap CI, not corrected for multiplicity.")
d, src = both("ablation_effects_test.csv")
table("S4", "Feature-group ablations of the selected learner", d, src, note="> 0: removing the group (or adding the leaky RCS) worsens the test pinball loss.")
frames, srcs = [], []
for o in ("2016", "2011"):
    p = os.path.join(RE, f"origin{o}_uncontrolled", "interval_rules.json")
    if os.path.isfile(p):
        j = json.load(open(p))
        frames.append(pd.DataFrame([{"origin": o, "rule": k, **v} for k, v in j.items()])); srcs.append(p)
if frames:
    table("S5", "Operational +/-20 % and +/-30 % windows against conformal intervals", pd.concat(frames, ignore_index=True), srcs,
          note="coverage_decayed: objects that decayed within five years.")
d, src = both("horizon_test.csv")
if d is not None:
    keep = d["model"].isin(["physics_only", "xgboost", "tabm", "tabicl", "kumo", "cosmos_gated", "stack_constant", "equal_average"])
    table("S6", "Horizon probabilities (AUC-PR, AUC-ROC on raw scores; Brier, ECE after tune-B isotonic calibration)", d[keep], src, digits=3)
d, src = both("subgroups_test.csv")
if d is not None:
    table("S7", "Re-entry subgroups (main models)", d[["origin", "model", "subgroup", "n", "pinball", "mae_log10", "cov90"]], src)
p = os.path.join(RE, "leak_check.json")
if os.path.isfile(p):
    j = json.load(open(p))
    rows = [{"split": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}} for k, v in j.items()]
    table("S8", "Radar-cross-section availability and 30-day decay", pd.DataFrame(rows), p, digits=3)
p = os.path.join(B, "twin_couplings", "c2_summary.csv")
if os.path.isfile(p):
    table("S9", "C2 cloud survival, all candidates", pd.read_csv(p), p, digits=3)
p = os.path.join(B, "twin_couplings", "c3_summary.json")
if os.path.isfile(p):
    j = json.load(open(p))
    rows = [{"status": g, "policy": pol, **e} for g, pols in j["by_status_policy"].items() for pol, e in pols.items()]
    table("S10", "C3 by operational status and policy", pd.DataFrame(rows), p)
p = os.path.join(B, "telemetry_smap_msl", "summary.csv")
if os.path.isfile(p):
    s = pd.read_csv(p)
    c = ["spacecraft", "model", "channels", "events", "event_tp", "event_fp", "event_precision", "event_recall", "event_f1",
         "esa_ew_precision", "esa_ew_f0.5", "nominal_alarm_fraction", "median_delay_steps", "event_f1_swap", "event_f1_swap_p",
         "event_f1_late", "esa_ew_f0.5_late", "pointwise_auc_pr", "pointwise_auc_pr_pooled", "pointwise_auc_pr_pooled_raw", "point_adjusted_f1"]
    table("S11", "NASA SMAP/MSL, all detectors", s[[x for x in c if x in s]], p, digits=3)
p = os.path.join(B, "conjunction_kelvins_cac", "test_results.csv")
if os.path.isfile(p):
    r = pd.read_csv(p)
    c = ["model", "L", "F2", "MSE_HR", "L_clip", "L_decision_clip", "pinball", "auc_pr_high", "auc_roc_high", "brier_high",
         "ece_high", "cov80", "cov90", "cov90_high", "width90"]
    table("S12", "ESA Kelvins collision-risk test set, all models", r[[x for x in c if x in r]], p)
p = os.path.join(B, "conjunction_kelvins_cac", "gated_persistence_test.csv")
if os.path.isfile(p):
    table("S13", "Latest-risk point forecast gated by calibrated P(high risk)", pd.read_csv(p), p)
p = os.path.join(B, "twin_prospective_t1", "t1_summary.json")
if os.path.isfile(p):
    j = json.load(open(p))
    rows = []
    for h, r in j["by_horizon"].items():
        for grp, v in r.items():
            if grp.startswith(("uncontrolled", "class_", "payload_", "all_")):
                for var, e in (v.items() if grp != "all_twin" else [("twin", v)]):
                    rows.append({"horizon": h, "group": grp, "variant": var, **e})
    table("S14", "As-of-date test T1 on a previously analysed window (as of 2026-08-16, decays to 2026-10-27)", pd.DataFrame(rows), p)
d, src = both("pairwise_test.csv")
table("S16", "Re-entry head-to-head comparisons (post hoc; paired launch-family bootstrap; < 0: a is better)", d, src,
      note="Nominal 95 % intervals, not corrected for multiplicity.")
p = os.path.join(B, "twin_couplings", "c2_bootstrap.csv")
if os.path.isfile(p):
    table("S17", "C2 mean absolute fraction errors and paired differences with cloud-cluster bootstrap CIs", pd.read_csv(p), p, digits=3,
          note="pred_: raw read-off; predcal_: calibrated read-off (added after the raw reading was criticised in the first review).")
p = os.path.join(B, "conjunction_kelvins_cac", "pinball_pairs_test.csv")
if os.path.isfile(p):
    table("S18", "Collision-risk pinball-loss comparisons (post hoc; paired bootstrap over test events; < 0: a is better)",
          pd.read_csv(p), p, note="Nominal 95 % intervals, not corrected for multiplicity.")
p = os.path.join(ROOT, "data", "DATASET_MANIFEST.csv")
if os.path.isfile(p):
    m = pd.read_csv(p)
    keep = [c for c in ("dataset", "class", "path", "bytes", "licence", "retrieved_utc") if c in m.columns]
    table("S15", "Dataset manifest (files with checksums)", m[keep] if keep else m, p)

os.makedirs(os.path.join(PAPER, "build"), exist_ok=True)
md_path = os.path.join(PAPER, "build", "supplementary.md")
open(md_path, "w", encoding="utf-8").write("\n".join(out))
sys.path.insert(0, os.path.join(ROOT, "thesis", "tools"))
import render as thesis_render  # noqa: E402
r = subprocess.run([thesis_render.PANDOC, md_path, "--from", "markdown+pipe_tables", "--reference-doc",
                    os.path.join(HERE, "reference.docx"), "-o", os.path.join(PAPER, "build", "supplementary.docx")],
                   capture_output=True, text=True)
print("wrote", md_path, "tables:", sum(1 for x in out if x.startswith("## Table")), "figures:",
      sum(1 for x in out if x.startswith("![")), "pandoc rc", r.returncode, r.stderr[:300])
