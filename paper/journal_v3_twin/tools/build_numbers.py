"""Collect every number the twin manuscript reports from the result files into numbers.json.

    python paper/journal_v3_twin/tools/build_numbers.py

Keys are flat strings ("re.o2016.cosmos_gated.pinball"); each value carries its source file.
The manuscript uses {{key:format}} placeholders only, so no number is typed by hand.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PAPER))
B = os.path.join(ROOT, "outputs", "benchmarks")
N = {}


def put(k, v, src):
    if isinstance(v, dict):                       # nested results: one key per leaf
        for kk, vv in v.items():
            put(f"{k}.{kk}", vv, src)
        return
    if isinstance(v, (list, tuple)) and all(not isinstance(x, (dict, list, tuple)) for x in v):
        for i, vv in enumerate(v):
            put(f"{k}.{i}", vv, src)
        return
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and np.isposinf(v):
        v = "∞"                                   # a real value (e.g. L with F2 = 0), not a missing one
    elif isinstance(v, float) and not np.isfinite(v):
        v = None
    N[k] = {"value": v, "source": os.path.relpath(src, ROOT)}


def reentry():
    for o in ("2016", "2011"):
        d = os.path.join(B, "reentry_lifetime", f"origin{o}_uncontrolled")
        if not os.path.isfile(os.path.join(d, "summary.json")):
            continue
        s = json.load(open(os.path.join(d, "summary.json")))
        p = os.path.join(d, "summary.json")
        for k, v in s["sizes"].items():
            put(f"re.o{o}.n.{k}", v, p)
            put(f"re.o{o}.sizes.{k}", v, p)
        for k, v in s["events_5yr"].items():
            put(f"re.o{o}.events5.{k}", v, p)
        put(f"re.o{o}.best", s["best_learner_on_tuneA"], p)
        for k in ("tuneA_pinball", "selection_pool", "prior_pool_sizes", "known5", "dates"):
            if k in s:
                put(f"re.o{o}.{k}", s[k], p)
        run1 = os.path.join(d, "run1_learner_set_v1")          # first learner set (deviation record)
        if os.path.isfile(os.path.join(run1, "summary.json")):
            s1 = json.load(open(os.path.join(run1, "summary.json")))
            put(f"re1.o{o}.best", s1["best_learner_on_tuneA"], os.path.join(run1, "summary.json"))
            put(f"re1.o{o}.paired", {k: {"diff": v["pinball_other_minus_gated"], "lo": v["ci95"][0], "hi": v["ci95"][1]}
                                     for k, v in s1["paired_vs_cosmos_gated"].items()}, os.path.join(run1, "summary.json"))
            r1 = pd.read_csv(os.path.join(run1, "regression_test.csv"))
            for _, row in r1.iterrows():
                for c in ("pinball", "mae_log10", "cov90", "c_index"):
                    put(f"re1.o{o}.{row['model']}.{c}", row[c], os.path.join(run1, "regression_test.csv"))
        for k, v in s["am_source_test"].items():
            put(f"re.o{o}.am_source_test.{k}", v, p)
        for name, g in s["gates"].items():
            put(f"re.o{o}.gate.{name}.weight_mean", g["test_weight_mean"], p)
            for src, w in g["test_weight_by_am_source"].items():
                put(f"re.o{o}.gate.{name}.weight.{src}", w, p)
        for name, c in s["paired_vs_cosmos_gated"].items():
            put(f"re.o{o}.paired.{name}.diff", c["pinball_other_minus_gated"], p)
            put(f"re.o{o}.paired.{name}.lo", c["ci95"][0], p)
            put(f"re.o{o}.paired.{name}.hi", c["ci95"][1], p)
        for name, c in s["mae_log10_ci"].items():
            put(f"re.o{o}.maeci.{name}.lo", c["ci95"][0], p)
            put(f"re.o{o}.maeci.{name}.hi", c["ci95"][1], p)
        for name, f in s["fits"].items():
            put(f"re.o{o}.fit.{name}.seconds", f["fit_seconds_per_seed"], p)
            ms = f.get("predict_ms_per_1000_objects", f.get("predict_ms_per_1000"))
            put(f"re.o{o}.fit.{name}.ms1000", ms, p)
            put(f"re.o{o}.fit.{name}.ms1000_s", None if ms is None else ms / 1000.0, p)
            # wall-clock of the one call that was timed: predicting the test cohort
            put(f"re.o{o}.fit.{name}.test_call_s", None if ms is None else ms * s["sizes"]["test"] / 1e6, p)
            put(f"re.o{o}.fit.{name}.n_features", f["n_features"], p)
        put(f"re.o{o}.seconds_total", s["seconds_total"], p)
        put(f"re.o{o}.physics_seconds_train", s["physics_seconds"]["train"], p)
        put(f"re.o{o}.physics_seconds_test", s["physics_seconds"]["test"], p)
        q = os.path.join(d, "km_descriptive_test.csv")
        if os.path.isfile(q):
            for _, row in pd.read_csv(q).iterrows():
                for c in ("n", "events", "censored", "censored_fraction", "median_days", "surv_30d", "surv_1yr", "surv_5yr"):
                    put(f"re.o{o}.km.{row['group']}.{c}", row[c], q)
        q = os.path.join(d, "temporal_coverage_test.csv")
        if os.path.isfile(q):
            t = pd.read_csv(q)
            for _, row in t.iterrows():
                put(f"re.o{o}.tcov.{row['model']}.{int(row['year'])}", row["cov90"], q)
            for m, g in t.groupby("model"):
                g = g.sort_values("year")
                put(f"re.o{o}.tcov.{m}.first", float(g.cov90.iloc[0]), q)
                put(f"re.o{o}.tcov.{m}.last", float(g.cov90.iloc[-1]), q)
                put(f"re.o{o}.tcov.{m}.drop", float(g.cov90.iloc[0] - g.cov90.iloc[-1]), q)
        r = pd.read_csv(os.path.join(d, "regression_test.csv"))
        for _, row in r.iterrows():
            for c in r.columns:
                if c != "model":
                    put(f"re.o{o}.{row['model']}.{c}", row[c], os.path.join(d, "regression_test.csv"))
        h = pd.read_csv(os.path.join(d, "horizon_test.csv"))
        for _, row in h.iterrows():
            for c in ("auc_pr", "auc_roc", "brier", "ece", "positives", "n"):
                if c in row:
                    put(f"re.o{o}.h.{row['model']}.{row['horizon']}.{c}", row[c], os.path.join(d, "horizon_test.csv"))
        sd = pd.read_csv(os.path.join(d, "seeds_test.csv"))
        for m, g in sd.groupby("model"):
            put(f"re.o{o}.seedsd.{m}.pinball", g["pinball"].std(ddof=1), os.path.join(d, "seeds_test.csv"))
            put(f"re.o{o}.seedmean.{m}.pinball", g["pinball"].mean(), os.path.join(d, "seeds_test.csv"))
        sub = pd.read_csv(os.path.join(d, "subgroups_test.csv"))
        for _, row in sub.iterrows():
            for c in ("n", "pinball", "mae_log10", "cov90", "medae_days_decayed"):
                if c in row:
                    put(f"re.o{o}.sub.{row['model']}.{row['subgroup']}.{c}", row[c], os.path.join(d, "subgroups_test.csv"))
        for fn, tag in (("diagnostic_oracle_solar.json", "oracle"), ("survival_aft.json", "aft")):
            q = os.path.join(d, fn)
            if os.path.isfile(q):
                j = json.load(open(q))
                if tag == "aft":
                    for k, v in j["test"].items():
                        put(f"re.o{o}.aft.{k}", v, q)
                else:
                    for var, met in j.items():
                        for k, v in met.items():
                            put(f"re.o{o}.oracle.{var}.{k}", v, q)
        q = os.path.join(d, "pairwise_test.csv")
        if os.path.isfile(q):
            for _, row in pd.read_csv(q).iterrows():
                put(f"re.o{o}.pair.{row['a']}.{row['b']}", {"diff": row["pinball_a_minus_b"], "lo": row["ci95_lo"],
                                                           "hi": row["ci95_hi"]}, q)
        q = os.path.join(d, "gate_all_learners.csv")
        if os.path.isfile(q):
            for _, row in pd.read_csv(q).iterrows():
                put(f"re.o{o}.gateall.{row['learner']}", {c: row[c] for c in ("pinball_alone", "pinball_gated", "gain", "gain_ci95_lo",
                                                                              "gain_ci95_hi", "gate_weight_test_mean")}, q)
        q = os.path.join(d, "ablation_effects_test.csv")
        if os.path.isfile(q):
            for _, row in pd.read_csv(q).iterrows():
                put(f"re.o{o}.abl.{row['ablation']}", {"diff": row["pinball_ablation_minus_full"], "lo": row["ci95_lo"],
                                                      "hi": row["ci95_hi"]}, q)
        q = os.path.join(d, "class_breakdown_test.csv")
        if os.path.isfile(q):
            for _, row in pd.read_csv(q).iterrows():
                put(f"re.o{o}.cls.{row['model']}.{row['type1']}.pinball", row["pinball"], q)
                put(f"re.o{o}.cls.{row['model']}.{row['type1']}.n", row["n"], q)
        q = os.path.join(d, "interval_rules.json")
        if os.path.isfile(q):
            for k, v in json.load(open(q)).items():
                for kk, vv in v.items():
                    put(f"re.o{o}.rule.{k}.{kk}", vv, q)


def couplings():
    d = os.path.join(B, "twin_couplings")
    q = os.path.join(d, "c3_summary.json")
    if os.path.isfile(q):
        j = json.load(open(q))
        for k in ("n", "events", "horizon_days", "launches", "launches_with_events", "event_rate",
                  "payload_base_rate_previous_window", "am_source", "prospective", "in_window_cross_fitted",
                  "brier_difference_ci95_launch_bootstrap", "platt_in_window_full_fit"):
            put(f"c3.{k}", j[k], q)
        for g, e in j["by_status"].items():
            put(f"c3.status.{g}", {k: v for k, v in e.items() if k != "gcat_end_status_of_decays"}, q)
        put("c3.bs", j["by_status_policy"], q)
    q = os.path.join(d, "c2_summary.csv")
    if os.path.isfile(q):
        s = pd.read_csv(q)
        for _, row in s.iterrows():
            for c in s.columns:
                if c not in ("model", "reading"):
                    put(f"c2.{row['reading']}.{row['model']}.{c}", row[c], q)
        qb = os.path.join(d, "c2_bootstrap.csv")
        if os.path.isfile(qb):
            for _, row in pd.read_csv(qb).iterrows():
                key = row["comparison"].replace(" - ", "_minus_")
                put(f"c2b.{key}", {"v": row["mae_fraction"], "lo": row["ci95_lo"], "hi": row["ci95_hi"]}, qb)
        c2 = pd.read_csv(os.path.join(d, "c2_cloud_survival.csv"))
        put("c2.n_clouds", c2["cloud"].nunique(), os.path.join(d, "c2_cloud_survival.csv"))
        put("c2.n_cloud_years", len(c2), os.path.join(d, "c2_cloud_survival.csv"))


def improvements():
    """Relative gains of the selected learned expert over the baselines it would replace, so the
    abstract can state them without hard-coding arithmetic."""
    def pct(base, new):
        return None if base in (None, 0) else 100.0 * (base - new) / base
    src = "derived from the regression_test.csv / test_results.csv values above"
    for o in ("2016", "2011"):
        for tag, basekey in (("xgboost", f"re.o{o}.xgboost.pinball"), ("physics", f"re.o{o}.physics_only.pinball")):
            b_, n_ = N.get(basekey), N.get(f"re.o{o}.tabicl.pinball")
            if b_ is not None and n_ is not None:
                put(f"imp.re.o{o}.tabicl_vs_{tag}_pct", pct(b_["value"], n_["value"]), src)
    b_, n_ = N.get("cac.persistence.pinball"), N.get("cac.kumo.pinball")
    if b_ is not None and n_ is not None:
        put("imp.cac.kumo_vs_persistence_pct", pct(b_["value"], n_["value"]), src)


def conjunction():
    d = os.path.join(B, "conjunction_kelvins_cac")
    q = os.path.join(d, "test_results.csv")
    if not os.path.isfile(q):
        return
    r = pd.read_csv(q)
    for _, row in r.iterrows():
        for c in r.columns:
            if c != "model":
                v = row[c]
                put(f"cac.{row['model']}.{c}", v if not isinstance(v, str) else v, q)
    g = pd.read_csv(os.path.join(d, "gated_persistence_test.csv"))
    for _, row in g.iterrows():
        for c in ("L", "F2", "MSE_HR", "tp", "fp", "fn", "threshold_oof", "p_better_than_latest"):
            if c in row:
                put(f"cacg.{row['model']}.{c}", row[c], os.path.join(d, "gated_persistence_test.csv"))
        for c in ("L_minus_latest_ci95", "L_ci95"):
            if c in row and isinstance(row[c], str):
                lo, hi = json.loads(row[c])
                put(f"cacg.{row['model']}.{c}.lo", lo, os.path.join(d, "gated_persistence_test.csv"))
                put(f"cacg.{row['model']}.{c}.hi", hi, os.path.join(d, "gated_persistence_test.csv"))
    qq = os.path.join(d, "test_predictions.npz")
    if os.path.isfile(qq):
        P = np.load(qq)
        hi = P["y"] >= -6.0
        med = P["kumo__q"][hi, 3]
        put("cacq.kumo.n_high_below", int((med < -6).sum()), qq)
        put("cacq.kumo.n_high_in_m10_m6", int(((med > -10) & (med <= -6)).sum()), qq)
        put("cacq.kumo.n_high_near_floor", int((med <= -29).sum()), qq)
    qp = os.path.join(d, "pinball_pairs_test.csv")
    if os.path.isfile(qp):
        for _, row in pd.read_csv(qp).iterrows():
            put(f"cacp.{row['a']}.{row['b']}", {"diff": row["pinball_a_minus_b"], "lo": row["ci95_lo"], "hi": row["ci95_hi"]}, qp)
    s = json.load(open(os.path.join(d, "summary.json")))
    for k, v in s["data"].items():
        put(f"cac.data.{k}", v, os.path.join(d, "summary.json"))
    put("cac.best", s["best_learner_oof"], os.path.join(d, "summary.json"))
    for g in ("cosmos_gated", "stack_constant"):
        if f"gate_{g}" in s:
            put(f"cac.gate_weight_{g}", s[f"gate_{g}"]["test_weight_mean"], os.path.join(d, "summary.json"))
    for k, v in s["fits"].items():
        put(f"cac.fit.{k}.oof_pinball", v["oof_pinball"], os.path.join(d, "summary.json"))
        put(f"cac.fit.{k}.seconds", v["seconds"], os.path.join(d, "summary.json"))


def health():
    q = os.path.join(B, "telemetry_smap_msl", "summary.csv")
    if os.path.isfile(q):
        s = pd.read_csv(q)
        for _, row in s.iterrows():
            m = row["model"].split(" ")[0]
            for c in s.columns:
                if c not in ("spacecraft", "model") and pd.notna(row[c]):
                    put(f"smap.{row['spacecraft']}.{m}.{c}", row[c], q)
        cfg = os.path.join(B, "telemetry_smap_msl", "config.json")
        if os.path.isfile(cfg):
            c = json.load(open(cfg))
            put("smap.config", {k: v for k, v in c.items() if not isinstance(v, list)}, cfg)
    ev = os.path.join(ROOT, "thesis", "evidence", "evidence.json")
    if os.path.isfile(ev):
        E = json.load(open(ev))
        for k in ("pre.m1.A0.conformal_bh.f05.mean", "pre.m1.A1.conformal_bh.f05.mean", "conf.m1.h6.diff_official",
                  "conf.m1.h6.ci95_low", "conf.m1.h6.ci95_high", "pre.m1.A1.conformal_bh.fa_yr.mean",
                  "pre.m1.A2.conformal_bh.f05.mean", "pre.m1.A3.conformal_bh.f05.mean", "lit.dmlr.m1.full.best",
                  "ph.p1.m1.target.conformal_bh.forward.D.delta_official", "ops.official.hybrid.f1.mean",
                  "conf.m2.h6.diff_official", "conf.m2.h6.ci95_low", "conf.m2.h6.ci95_high"):
            if k in E:
                put(f"esa.{k}", E[k]["value"], ev)


def twin():
    for d in sorted(os.listdir(os.path.join(ROOT, "outputs", "twin"))) if os.path.isdir(os.path.join(ROOT, "outputs", "twin")) else []:
        q = os.path.join(ROOT, "outputs", "twin", d, "summary.json")
        if os.path.isfile(q):
            put("twin", json.load(open(q)), q)
            h = os.path.join(ROOT, "outputs", "twin", d, "FORECAST_SHA256.txt")
            if os.path.isfile(h):
                put("twin.sha256", open(h).read().split()[0], h)
                put("twin.sha256_short", open(h).read().split()[0][:16], h)
            c = os.path.join(ROOT, "outputs", "twin", d, "c2_clouds.csv")
            if os.path.isfile(c):
                cl = pd.read_csv(c)
                put("twin.c2.n_clouds", len(cl), c)
                put("twin.c2.fragments", int(cl["fragments_in_orbit"].sum()), c)
                for k in ("30d", "90d", "1yr"):
                    put(f"twin.c2.expected_remaining_{k}", float(cl[f"expected_remaining_{k}"].sum()), c)


LABELS = {"physics_only": "physics expert", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
          "mlp": "quantile MLP", "tabicl": "TabICLv2", "persistence": "persistence"}


def alias_best():
    """Keys of the learner selected on tune-A (re.o<O>.best) and in the CAC study (cac.best) are
    duplicated under the name BEST, so that the text can refer to the selected learner."""
    for prefix in ("re.o2016.", "re.o2011.", "cac."):
        b = N.get(prefix + "best", {}).get("value")
        if not b:
            continue
        put(prefix + "best_label", LABELS.get(b, b), N[prefix + "best"]["source"])
        for k in list(N):
            if not k.startswith(prefix):
                continue
            parts = k.split(".")
            new = ["BEST" + p[len(b):] if (p == b or p.startswith(b + "_")) else p for p in parts]
            if new != parts:
                N[".".join(new)] = dict(N[k])


def leak():
    q = os.path.join(B, "reentry_lifetime", "leak_check.json")
    if os.path.isfile(q):
        put("leak", json.load(open(q)), q)


def t1():
    q = os.path.join(B, "twin_prospective_t1", "t1_summary.json")
    if os.path.isfile(q):
        put("t1", json.load(open(q)), q)


def data():
    p = os.path.join(ROOT, "data", "feature3_reentry", "processed", "lifetime_cohort_2026-10-27_build_log.json")
    if os.path.isfile(p):
        for k, v in json.load(open(p)).items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    put(f"data.cohort.{k}.{kk}", vv, p)
            else:
                put(f"data.cohort.{k}", v, p)


if __name__ == "__main__":
    for fn in (data, reentry, couplings, conjunction, health, twin, leak, t1, improvements, alias_best):
        fn()
    json.dump(N, open(os.path.join(PAPER, "build", "numbers.json"), "w"), indent=1, default=float)
    print(len(N), "numbers ->", os.path.join(PAPER, "build", "numbers.json"))
