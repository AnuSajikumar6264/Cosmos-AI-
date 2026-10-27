"""Compile every number the thesis reports into thesis/evidence/evidence.json, with provenance.

    python thesis/tools/build_evidence.py

Each entry is ``key -> {"value": ..., "source": "<file>[: field]", "kind": ...}`` where kind is

measured     read from a result file written by an experiment or test run
derived      computed here from measured values (means, ranges, counts); the rule is in "note"
config       a value taken from the code or a configuration file (defaults, grids, constants)
literature   a value reported by a cited publication (source names the reference and table)

The thesis sources contain ``{{key}}`` / ``{{key:fmt}}`` placeholders; thesis/tools/render.py
substitutes them and fails on any key that is missing here, so no number is typed by hand.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
EV: dict = {}
B = "outputs/benchmarks"
ESA = f"{B}/telemetry_esa_adb"
PROT = f"{ESA}/oof_protocol"


def put(key, value, source, kind="measured", note=None):
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        value = None
    EV[key] = {"value": value, "source": source, "kind": kind}
    if note:
        EV[key]["note"] = note


def agg(prefix, values, source, note="over seeds"):
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return
    put(f"{prefix}.mean", float(v.mean()), source, "derived", f"mean {note}")
    put(f"{prefix}.std", float(v.std(ddof=1)) if v.size > 1 else 0.0, source, "derived",
        f"sample sd {note}")
    put(f"{prefix}.min", float(v.min()), source, "derived", f"min {note}")
    put(f"{prefix}.max", float(v.max()), source, "derived", f"max {note}")
    put(f"{prefix}.n", int(v.size), source, "derived", f"count {note}")


# ============================================================================ environment
def environment():
    d = sorted(glob.glob("outputs/experiments/*esa_event_m1_target_event_ar8_all_c3_s0*"))[-1]
    env = json.load(open(f"{d}/env.json"))
    for k in ("python", "torch", "numpy", "pandas", "device_name", "cpu_count", "platform"):
        put(f"env.{k}", env[k], f"{d}/env.json")
    rows = [json.loads(l) for l in open("outputs/experiments/index.jsonl")]
    put("runs.total", len(rows), "outputs/experiments/index.jsonl", "derived", "line count")
    st = pd.Series([r.get("status") for r in rows]).value_counts()
    put("runs.ok", int(st.get("ok", 0)), "outputs/experiments/index.jsonl", "derived")
    # (test-half durations are added in cross_cutting)
    put("runs.failed", int(len(rows) - st.get("ok", 0)), "outputs/experiments/index.jsonl", "derived")
    ck = glob.glob("outputs/experiments/*/checkpoints/*")
    put("runs.checkpoints", len(ck), "outputs/experiments/*/checkpoints/", "derived", "file count")
    put("runs.checkpoints_gb", sum(os.path.getsize(p) for p in ck) / 1e9,
        "outputs/experiments/*/checkpoints/", "derived", "total size")
    # code size
    def count(globs):
        files = [f for g in globs for f in glob.glob(g, recursive=True)]
        return len(files), sum(sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
                               for f in files)
    n, l = count(["cosmos/*.py", "cosmos/health/*.py", "cosmos/debris/*.py",
                  "cosmos/reentry/*.py"])
    put("code.original.files", n, "cosmos/{*,health,debris,reentry}/*.py", "derived")
    put("code.original.lines", l, "cosmos/{*,health,debris,reentry}/*.py", "derived")
    files = [f for f in glob.glob("cosmos/ai/**/*.py", recursive=True) if "esa_metrics" not in f]
    put("code.ai.files", len(files), "cosmos/ai/**/*.py (excl. vendored esa_metrics)", "derived")
    put("code.ai.lines", sum(sum(1 for _ in open(f)) for f in files),
        "cosmos/ai/**/*.py (excl. vendored esa_metrics)", "derived")
    for name, log in (("verify", "outputs/tests/test_verify_after.log"),
                      ("ai", "outputs/tests/test_ai_after_prereg.log")):
        txt = open(log).read()
        m = re.search(r"(\d+) / (\d+) checks passed", txt) or re.search(
            r"(\d+) passed, (\d+) failed", txt)
        if name == "verify":
            put("tests.verify.passed", int(m.group(1)), log)
            put("tests.verify.total", int(m.group(2)), log)
        else:
            put("tests.ai.passed", int(m.group(1)), log)
            put("tests.ai.failed", int(m.group(2)), log)
    txt = open("outputs/tests/test_datasets_after_per_test.log").read()
    put("tests.datasets.log_tail", txt.strip().splitlines()[-1][:200],
        "outputs/tests/test_datasets_after_per_test.log")


def verify_checks():
    """Parse the physics verification suite: name, expected, measured."""
    log = "outputs/tests/test_verify_after.log"
    lines = open(log).read().splitlines()
    i = 0
    for k, ln in enumerate(lines):
        m = re.match(r"\s+\[(PASS|FAIL)\] (.+)", ln)
        if m and k + 1 < len(lines):
            m2 = re.match(r"\s+expected (.+?)\s{2,}measured (.+?)(\s{2,}\(.*\))?$", lines[k + 1])
            if m2:
                i += 1
                put(f"verify.{i}.name", m.group(2).strip(), log)
                put(f"verify.{i}.expected", m2.group(1).strip(), log)
                put(f"verify.{i}.measured", m2.group(2).strip(), log)
                put(f"verify.{i}.status", m.group(1), log)
    put("verify.n", i, log, "derived")


# ============================================================================ datasets
def datasets():
    src = "data/DATASET_REPORT.json"
    r = json.load(open(src))
    for k, d in r["datasets"].items():
        put(f"data.{k}.status", d["status"], src)
        put(f"data.{k}.class", d["class"], src)
        put(f"data.{k}.size_mb", d["size_on_disk_mb"], src)
        put(f"data.{k}.licence", d["licence"], src)
    o = r["opssat_ad"]
    put("data.opssat.fragments", o["segments"], src)
    put("data.opssat.channels", o["channels"], src)
    put("data.opssat.n_features", o["n_features"], src)
    put("data.opssat.anomaly_fraction", o["anomaly_fraction"], src)
    for kk, v in o["class_counts"].items():
        put("data.opssat.count." + kk.replace("=", "").replace(",", "_"), v, src)
    put("data.opssat.train", o["class_counts"]["train=1,anomaly=0"] + o["class_counts"]["train=1,anomaly=1"], src, "derived")
    put("data.opssat.test", o["class_counts"]["train=0,anomaly=0"] + o["class_counts"]["train=0,anomaly=1"], src, "derived")
    put("data.opssat.seg_len_median", o["segment_length"]["50%"], src)
    put("data.opssat.seg_len_min", o["segment_length"]["min"], src)
    put("data.opssat.seg_len_max", o["segment_length"]["max"], src)
    e = r["esa_adb"]
    for m in (1, 2):
        mm = e[f"mission{m}"]
        for k in ("grid_seconds", "channels", "target_channels", "categorical_channels",
                  "telecommands_priority_ge_3"):
            put(f"data.esa.m{m}.{k}", mm[k], src)
        for part in ("train", "test"):
            for k in ("n", "start", "end", "telecommand_executions"):
                put(f"data.esa.m{m}.{part}.{k}", mm[part][k], src)
        for per, d in mm["events_target_channels_by_period"].items():
            for k, v in d.items():
                put(f"data.esa.m{m}.events_target.{per}.{k.replace(' ', '_')}", v, src)
    put("data.esa.revision_check", e["revision_check"], src)
    s = r["spotgeo"]
    for part in ("train", "test"):
        for k in ("sequences", "frames_on_disk", "objects", "frames_without_objects",
                  "max_objects_per_frame", "corrupt_png"):
            put(f"data.spotgeo.{part}.{k}", s[part][k], src)
    put("data.spotgeo.dup_train_test", s["exact_duplicate_frames_train_vs_test"], src)
    for k, v in s["validation_split"].items():
        put(f"data.spotgeo.valsplit.{k}", v, src)
    v = r["cosmos_synthetic_vision"]
    for part in ("train", "val", "test"):
        put(f"data.synvis.{part}.images", v[part]["images"], src)
        put(f"data.synvis.{part}.objects", v[part]["objects"], src)
        put(f"data.synvis.{part}.box_median_px", v[part]["box_px"]["median"], src)
    c = r["catalogue_and_reentry"]
    for snap in ("satcat_2026-08-16", "satcat_2026-10-24"):
        s2 = snap.replace("satcat_", "").replace("-", "")
        put(f"data.satcat.{s2}.rows", c[snap]["rows"], src)
        put(f"data.satcat.{s2}.decayed", c[snap]["decayed"], src)
        for t, n in c[snap]["object_types"].items():
            put(f"data.satcat.{s2}.type.{t.replace('/', '')}", n, src)
    lm = c["landmark"]
    for k, n in lm["exclusions"].items():
        put(f"data.landmark.{k}", n, src)
    for k, d in lm["events_by_split"].items():
        put(f"data.landmark.split.{k}.n", d["n"], src)
        put(f"data.landmark.split.{k}.events", d["events"], src)
    for t, n in lm["events_by_type"].items():
        put(f"data.landmark.events_type.{t.replace('/', '')}", n, src)
    put("data.landmark.gp_coverage", lm["gp_coverage"], src)
    rd = c["recovered_debris"]
    for k in ("fragment_rows", "objects", "with_coordinates", "with_mass"):
        put(f"data.recovered.{k}", rd[k], src)
    # labelled events per official half and category (all channels), recomputed from labels
    for m, split in ((1, "2007-01-01"), (2, "2001-10-01")):
        f = f"data/feature1_health/esa_adb/processed/mission{m}/labels_with_types.csv"
        df = pd.read_csv(f, parse_dates=["StartTime", "EndTime"])
        ev = df.groupby("ID").agg(s=("StartTime", "min"), cat=("Category", "first"),
                                  cls=("Class", "first"))
        ev["half"] = np.where(ev.s >= pd.Timestamp(split), "test", "train")
        for (h, cat), n in ev.groupby(["half", "cat"]).size().items():
            put(f"data.esa.m{m}.events_all.{h}.{cat.replace(' ', '_')}", int(n), f, "derived",
                "events (unique ID) by first start time vs the official split date")
        tr, te = set(ev[ev.half == "train"].cls), set(ev[ev.half == "test"].cls)
        put(f"data.esa.m{m}.classes.train", len(tr), f, "derived")
        put(f"data.esa.m{m}.classes.test", len(te), f, "derived")
        put(f"data.esa.m{m}.classes.test_unseen", len(te - tr), f, "derived")


# ============================================================================ ESA exploratory
ENC = {"ar8": "ar8", "persistence": "persistence", "tcn": "tcn", "patchtst": "patchtst",
       "chronos-bolt-small": "chronos", "global_std3": "gstd3", "global_std5": "gstd5",
       "cosmos_monitor": "cosmon", "cosmos_monitor_no_ar": "cosmon_noar",
       "out_of_limits": "ool", "out_of_limits_all": "oolall"}
METRICS = {"f05": "anomaly_and_rare/EW_F0.5", "p": "anomaly_and_rare/EW_precision",
           "r": "anomaly_and_rare/EW_recall", "f05_anom": "anomaly/EW_F0.5",
           "r_anom": "anomaly/EW_recall", "p_anom": "anomaly/EW_precision",
           "fa_yr": "false_alarms_per_year", "rare_only": "rare_event_only_runs",
           "alarm_runs": "alarm_runs", "adtqc": "anomaly_and_rare/ADTQC",
           "aff": "anomaly_and_rare/AFF_F0.5", "chan": "anomaly_and_rare/channel_F0.5",
           "subsys": "anomaly_and_rare/subsystem_F0.5",
           "alarm_prec": "anomaly_and_rare/alarming_precision"}


def esa_exploratory():
    src = f"{ESA}/results.csv"
    df = pd.read_csv(src)
    df["tc"] = df.telecommands.fillna(False).astype(bool)
    for (m, ch, enc, tc, cal), g in df.groupby(["mission", "channels", "encoder", "tc",
                                                "calibrator"]):
        e = ENC.get(enc, enc) + ("_tc" if tc else "")
        chs = "full" if ch == "target" else "lw"
        for short, col in METRICS.items():
            if col in g:
                agg(f"explo.m{m}.{chs}.{e}.{cal}.{short}", g[col].tolist(), src)
        for col, short in (("selected_level", "q"), ("selected_k", "k")):
            if col in g and g[col].notna().all() and len(g) == 1:
                put(f"explo.m{m}.{chs}.{e}.{cal}.{short}", float(g[col].iloc[0]), src)
        for col in ("train_time_s", "params", "inference_s_per_telemetry_day"):
            if col in g and g[col].notna().any():
                agg(f"explo.m{m}.{chs}.{e}.{cal}.{col}", g[col].dropna().tolist(), src)
    src = f"{ESA}/bootstrap_comparisons.csv"
    b = pd.read_csv(src)
    for i, r in b.iterrows():
        h = re.match(r"(H\d) (M\d): (.+)", r.comparison)
        slug = re.sub(r"[^a-z0-9]+", "_", h.group(3).lower()).strip("_")[:40]
        key = f"explo.boot.{h.group(1).lower()}.{h.group(2).lower()}.{slug}"
        put(f"{key}.name", r.comparison, src)
        for c in ("A_EW_F0.5_official", "B_EW_F0.5_official", "diff_reimpl", "ci95_low",
                  "ci95_high", "n_boot"):
            put(f"{key}.{c.replace('.', '')}", r[c], src)
    # telecommand robustness
    for f in sorted(glob.glob(f"{ESA}/robustness/*.json")):
        tag = os.path.basename(f).replace(".json", "")
        rows = json.load(open(f))
        for r in rows:
            sub = (f"mask{r['mask_minutes_after_telecommand']}" if "mask_minutes_after_telecommand"
                   in r else r["calibrator"])
            k = f"explo.robust.{tag}.{sub}"
            for short, col in METRICS.items():
                if col in r:
                    put(f"{k}.{short}", r[col], f)
            if "with_telecommands_EW_F0.5" in r:
                put(f"{k}.f05_with_tc", r["with_telecommands_EW_F0.5"], f)
            if "masked_fraction_of_time" in r:
                put(f"{k}.masked_fraction", r["masked_fraction_of_time"], f)


# ============================================================================ pre-registered
ARM = [(r"^m(\d)_target_ar8$", "A1"), (r"^m(\d)_lightweight_ar8$", "A1lw"),
       (r"^m(\d)_target_gbt_ar8_s\d_event$", "A2"),
       (r"^m(\d)_target_event_ar8_all_c3_s\d_event$", "A3"),
       (r"^m(\d)_target_event_ar8_no_residual_c3_s\d_event$", "A3nr"),
       (r"^m(\d)_target_event_ar8_all_c2_s\d_event$", "A3c2")]


def arm_of(tag):
    for pat, a in ARM:
        m = re.match(pat, tag)
        if m:
            return int(m.group(1)), a
    return None, None


def preregistered():
    vals = defaultdict(list)
    for f in sorted(glob.glob(f"{PROT}/test/*.json")):
        tag = os.path.basename(f)[:-5]
        m, a = arm_of(tag)
        for r in json.load(open(f)):
            cal = r["calibrator"]
            for short, col in METRICS.items():
                if col in r:
                    vals[(m, a, cal, short)].append(r[col])
            for col, short in (("level", "q"), ("k", "k"), ("gamma", "gamma"),
                               ("test_point_far_nominal", "point_far")):
                if col in r:
                    vals[(m, a, cal, short)].append(r[col])
            put(f"pre.seed.{tag}.{cal}.f05", r[METRICS["f05"]], f)
            put(f"pre.seed.{tag}.{cal}.fa_yr", r["false_alarms_per_year"], f)
            put(f"pre.seed.{tag}.{cal}.alarm_runs", r["alarm_runs"], f)
    for (m, a, cal, short), v in vals.items():
        agg(f"pre.m{m}.{a}.{cal}.{short}", v, f"{PROT}/test/*.json")
    # A0 = exploratory AR(8) arm (9-month selection)
    for m in (1, 2):
        f = f"{ESA}/arms/m{m}_target_ar8_s0.json"
        for r in json.load(open(f))["rows"]:
            for short, col in METRICS.items():
                if col in r:
                    put(f"pre.m{m}.A0.{r['calibrator']}.{short}.mean", r[col], f)
            put(f"pre.m{m}.A0.{r['calibrator']}.q", r["selected_level"], f)
            put(f"pre.m{m}.A0.{r['calibrator']}.k", r["selected_k"], f)
    # selection (training-half CV)
    cv = defaultdict(list)
    for f in sorted(glob.glob(f"{PROT}/selection/*.json")):
        s = json.load(open(f))
        m, a = arm_of(s["tag"])
        put(f"sel.{s['tag']}.n_train_events", s["n_train_events"], f)
        for cal, c in s["chosen"].items():
            cv[(m, a, cal, "cv")].append(c["cv_EW_F0.5"])
            cv[(m, a, cal, "cv_anom")].append(c["cv_anomaly_EW_F0.5"])
            cv[(m, a, cal, "q")].append(c["level"])
            cv[(m, a, cal, "k")].append(c["k"])
            put(f"sel.{s['tag']}.{cal}.cv", c["cv_EW_F0.5"], f)
            put(f"sel.{s['tag']}.{cal}.q", c["level"], f)
            put(f"sel.{s['tag']}.{cal}.k", c["k"], f)
    for (m, a, cal, short), v in cv.items():
        agg(f"sel.m{m}.{a}.{cal}.{short}", v, f"{PROT}/selection/*.json")
    for m in (1, 2):
        put(f"sel.m{m}.n_train_events",
            json.load(open(f"{PROT}/selection/m{m}_target_ar8.json"))["n_train_events"],
            f"{PROT}/selection/m{m}_target_ar8.json")
    # confirmatory
    f = f"{PROT}/CONFIRMATORY.json"
    c = json.load(open(f))
    put("conf.freeze_id", c["freeze_id"], f)
    for m, d in c["missions"].items():
        for t in d["tests"]:
            h = t["comparison"].split(":")[0].lower()
            for k in ("A_official", "B_official", "diff_official", "diff", "ci95_low",
                      "ci95_high", "p_boot", "p_holm", "verdict", "n_boot", "block_days"):
                if k in t:
                    put(f"conf.m{m}.{h}.{k}", t[k], f)
    f = f"{PROT}/exploratory/exploratory_comparisons.json"
    x = json.load(open(f))
    for m, rows in x["results"].items():
        for r in rows:
            h = r["comparison"].split(":")[0].lower()
            for k in ("A_EW_F0.5", "B_EW_F0.5", "diff", "ci95_low", "ci95_high", "p_boot"):
                put(f"xcomp.m{m}.{h}.{k.replace('.', '')}", r[k], f)
            put(f"xcomp.m{m}.{h}.name", r["comparison"], f)
    f = f"{PROT}/exploratory/calibration_shift_m1.json"
    cs = json.load(open(f))["results"]
    ratio = {k: v[1] / v[0] for k, v in cs.items()}
    for k, v in cs.items():
        kk = k.replace(" ", "_").lower()
        put(f"shift.m1.{kk}.cv_rate", v[0], f)
        put(f"shift.m1.{kk}.test_rate", v[1], f)
        put(f"shift.m1.{kk}.ratio", ratio[k], f, "derived", "test rate / CV rate")
    for arm in ("A2", "A3"):
        r = [v for k, v in ratio.items() if k.startswith(arm)]
        put(f"shift.m1.{arm}.ratio_min", min(r), f, "derived")
        put(f"shift.m1.{arm}.ratio_max", max(r), f, "derived")
    f = f"{PROT}/exploratory/event_classes_summary_m1.csv"
    ec = pd.read_csv(f)
    for _, r in ec.iterrows():
        k = f"evcls.m1.{'seen' if r.seen_class else 'unseen'}.{r['cat'].replace(' ', '_')}"
        put(f"{k}.n", int(r.n_events), f)
        for col in ec.columns:
            if col.endswith("_detect_rate"):
                put(f"{k}.{col.replace('_detect_rate', '')}", r[col], f)
    # OOF fold setup
    for m in (1, 2):
        f = f"{ESA}/oof/m{m}_target_ar8/folds.json"
        j = json.load(open(f))
        put(f"oof.m{m}.K", j["K"], f)
        put(f"oof.m{m}.purge_steps", j["purge_steps"], f)
        put(f"oof.m{m}.n_train", j["n_train"], f)
        put(f"oof.m{m}.n_test", j["n_test"], f)
        put(f"oof.m{m}.seconds", j["seconds"], f)
    # event-model training cost
    tt = defaultdict(list)
    for f in sorted(glob.glob(f"{ESA}/event_model/*/meta.json")):
        j = json.load(open(f))
        tag = os.path.basename(os.path.dirname(f))
        kind = ("gbt" if "_gbt_" in tag else "tcn_nores" if "no_residual" in tag
                else "tcn_c2" if "_c2_" in tag else "tcn")
        m = int(tag[1])
        secs = [x["seconds"] for x in j["folds"]] + [j["final"]["seconds"]]
        tt[(m, kind)].append(sum(secs))
        if kind == "tcn":
            put(f"evm.m{m}.params", j["final"]["params"], f)
            put(f"evm.receptive_field", j["receptive_field"], f)
    for (m, kind), v in tt.items():
        agg(f"evm.m{m}.{kind}.train_seconds_total", v, f"{ESA}/event_model/*/meta.json",
            "over runs (5 fold models + final model per run)")


def config_values():
    from cosmos.ai.telemetry import event_gbt, event_model, oof, oof_benchmark
    from cosmos.ai.telemetry.alarms import GRIDS, K_GRID
    from cosmos.ai.telemetry.forecasting import PERIODS, ForecastConfig
    import dataclasses
    for f in dataclasses.fields(event_model.EventConfig):
        if f.name not in ("extra", "device", "seed", "mission", "channels"):
            put(f"cfg.evm.{f.name}", f.default if not callable(f.default) else None,
                "cosmos/ai/telemetry/event_model.py: EventConfig", "config")
    put("cfg.evm.features", list(event_model.FEATURES), "cosmos/ai/telemetry/event_model.py", "config")
    for k, v in event_gbt.PARAMS.items():
        put(f"cfg.gbt.{k}", v, "cosmos/ai/telemetry/event_gbt.py: PARAMS", "config")
    put("cfg.gbt.rounds", event_gbt.ROUNDS, "cosmos/ai/telemetry/event_gbt.py", "config")
    put("cfg.gbt.n_rows", event_gbt.N_ROWS, "cosmos/ai/telemetry/event_gbt.py", "config")
    put("cfg.oof.order", oof.ORDER, "cosmos/ai/telemetry/oof.py", "config")
    put("cfg.oof.horizon", oof.HORIZON, "cosmos/ai/telemetry/oof.py", "config")
    put("cfg.oof.max_fit", oof.MAX_FIT, "cosmos/ai/telemetry/oof.py", "config")
    for k, v in oof_benchmark.LEVELS.items():
        put(f"cfg.prereg.levels.{k}", v, "cosmos/ai/telemetry/oof_benchmark.py", "config")
    put("cfg.prereg.ks", oof_benchmark.KS, "cosmos/ai/telemetry/oof_benchmark.py", "config")
    put("cfg.prereg.merge_h", oof_benchmark.MERGE_H, "cosmos/ai/telemetry/oof_benchmark.py", "config")
    put("cfg.prereg.aci_alpha", oof_benchmark.ACI_ALPHA, "cosmos/ai/telemetry/oof_benchmark.py", "config")
    put("cfg.prereg.aci_gammas", oof_benchmark.ACI_GAMMAS, "cosmos/ai/telemetry/oof_benchmark.py", "config")
    for k, v in GRIDS.items():
        put(f"cfg.explo.levels.{k}", v, "cosmos/ai/telemetry/alarms.py", "config")
    put("cfg.explo.ks", K_GRID, "cosmos/ai/telemetry/alarms.py", "config")
    for m, p in PERIODS.items():
        for k, (a, b) in p.items():
            put(f"cfg.periods.m{m}.{k}", f"{a} to {b}", "cosmos/ai/telemetry/forecasting.py", "config")
    for f in dataclasses.fields(ForecastConfig):
        if f.name not in ("extra", "device", "seed", "mission", "channels", "encoder",
                          "telecommands"):
            put(f"cfg.fc.{f.name}", f.default, "cosmos/ai/telemetry/forecasting.py: ForecastConfig", "config")
    fr = json.load(open("research_design/preregistration/freeze_09a98f5e160e.json"))
    put("prereg.freeze_id", fr["freeze_id"], "research_design/preregistration/freeze_09a98f5e160e.json")
    put("prereg.freeze_written", fr["written"], "research_design/preregistration/freeze_09a98f5e160e.json")
    put("prereg.frozen_files", len(fr["sha256"]), "research_design/preregistration/freeze_09a98f5e160e.json", "derived")
    fr2 = json.load(open("research_design/preregistration/freeze.json"))
    put("prereg.refreeze_id", fr2["freeze_id"], "research_design/preregistration/freeze.json")
    dev = [l for l in open("research_design/preregistration/DEVIATIONS.md") if l.startswith("| 2026")]
    put("prereg.n_deviations", len(dev), "research_design/preregistration/DEVIATIONS.md", "derived")


# ============================================================================ OPSSAT, monitor
def opssat():
    for prot in ("official", "time_blocked"):
        src = f"{B}/telemetry_opssat/summary_{prot}.csv"
        d = pd.read_csv(src)
        for _, r in d.iterrows():
            mdl = r.model.replace("chronos:amazon/chronos-bolt-small", "chronos")
            for col in ("f1", "auc_pr", "auc_roc", "mcc", "precision", "recall", "ece", "brier",
                        "false_alarm_rate"):
                put(f"ops.{prot}.{mdl}.f{int(round(r.train_frac * 100))}.{col}.mean",
                    r[f"{col}_mean"], src)
                put(f"ops.{prot}.{mdl}.f{int(round(r.train_frac * 100))}.{col}.std",
                    r[f"{col}_std"], src)
            put(f"ops.{prot}.{mdl}.f{int(round(r.train_frac * 100))}.n", int(r.n_seeds), src)
            for col in ("params", "latency_ms_per_fragment", "train_time_s"):
                put(f"ops.{prot}.{mdl}.f{int(round(r.train_frac * 100))}.{col}", r[col], src)
    src = f"{B}/telemetry_opssat/paired_comparisons.csv"
    for _, r in pd.read_csv(src).iterrows():
        a = r.A.replace("chronos:amazon/chronos-bolt-small", "chronos")
        b = r.B.replace("chronos:amazon/chronos-bolt-small", "chronos")
        met = "aucpr" if r.metric.startswith("AUC") else "f1"
        for c in ("diff", "ci_low", "ci_high", "p_two_sided"):
            put(f"ops.paired.{a}_{b}.{met}.{c}", r[c], src)
    src = f"{B}/telemetry_opssat/robustness_summary.csv"
    for _, r in pd.read_csv(src).iterrows():
        k = f"ops.robust.{r.perturbation}.{str(r.level).replace('.', 'p')}.{r.model}"
        put(f"{k}.f1", r["f1|mean"], src)
        put(f"{k}.f1_std", r["f1|std"], src)
        put(f"{k}.auc_pr", r["auc_pr|mean"], src)
    src = f"{B}/telemetry_opssat/feature_generator_check.json"
    j = json.load(open(src))
    put("ops.featcheck", json.dumps(j)[:400], src)


def learned_monitor():
    for part in ("development", "heldout"):
        src = f"{B}/telemetry_cosmos_synthetic/{part}/summary_by_seed.csv"
        d = pd.read_csv(src)
        for mon, g in d.groupby("monitor"):
            for col in ("precision", "recall", "f1", "localisation_accuracy",
                        "median_delay_samples", "seconds_per_record"):
                agg(f"mon.{part}.{mon}.{col}", g[col].tolist(), src, "over seed sets")
            put(f"mon.{part}.{mon}.false_alarms_total", int(g.false_alarms_on_nominal.sum()),
                src, "derived", "sum over seed sets")
            put(f"mon.{part}.{mon}.seed_sets", ",".join(str(s) for s in g.seed_set), src)
        src2 = f"{B}/telemetry_cosmos_synthetic/{part}/per_record.csv"
        pr = pd.read_csv(src2)
        put(f"mon.{part}.n_records", int(len(pr)), src2, "derived", "rows")
        if "fault" in pr:
            put(f"mon.{part}.n_nominal_records",
                int((pr.fault.astype(str).str.lower().isin(["none", "nominal", "nan"])).sum()),
                src2, "derived")


# ============================================================================ vision, re-entry
def vision():
    src = f"{B}/vision_spotgeo/summary.csv"
    for _, r in pd.read_csv(src).iterrows():
        mdl = {"classical COSMOS front end + DoG": "classical"}.get(r.model, r.model)
        k = f"spot.{mdl}.{r.inputs}.f{int(round(r.train_frac * 100))}"
        for col in ("test_f1", "test_precision", "test_recall", "test_official_mse_mean_per_sequence",
                    "val_f1", "latency_ms_per_sequence", "params", "train_time_s"):
            if f"{col}|mean" in r:
                put(f"{k}.{col}.mean", r[f"{col}|mean"], src)
                put(f"{k}.{col}.std", r[f"{col}|std"], src)
        put(f"{k}.n", int(r.n_seeds), src)
    runs = defaultdict(list)
    for f in sorted(glob.glob(f"{B}/vision_spotgeo/runs/*.json")):
        j = json.load(open(f))
        k = f"spot.run.{j['model']}.{j['inputs']}.f{int(round(j['train_frac'] * 100))}"
        runs[k].append(j)
    for k, js in runs.items():
        agg(f"{k}.mse_sum", [j["test_official_mse_sum"] for j in js], f"{B}/vision_spotgeo/runs/*.json")
        agg(f"{k}.f1", [j["test_f1"] for j in js], f"{B}/vision_spotgeo/runs/*.json")
        agg(f"{k}.threshold", [j["selected_threshold"] for j in js], f"{B}/vision_spotgeo/runs/*.json")
    j = json.load(open(f"{B}/vision_spotgeo/classical_cosmos_frontend.json"))
    for kk in ("test_f1", "test_official_mse_sum", "test_precision", "test_recall"):
        if kk in j:
            put(f"spot.classical.{kk}", j[kk], f"{B}/vision_spotgeo/classical_cosmos_frontend.json")
    src = f"{B}/vision_cosmos_synthetic/summary.csv"
    names = {"heatmap-unet (COSMOS-AI)": "heatmap", "tinyyolo (existing COSMOS)": "tinyyolo",
             "yolov8n (ultralytics, COCO-pretrained)": "yolov8n"}
    for _, r in pd.read_csv(src).iterrows():
        k = f"syn.{names.get(r.model, r.model)}"
        for col in ("map50", "map50_95", "small_map50", "map50_class_agnostic", "precision@conf",
                    "recall@conf", "f1@conf", "mean_iou@conf", "latency_ms_per_item", "params",
                    "train_time_s"):
            put(f"{k}.{col.replace('@', '_at_')}.mean", r[f"{col}|mean"], src)
            put(f"{k}.{col.replace('@', '_at_')}.std", r[f"{col}|std"], src)
        put(f"{k}.n", int(r.n_seeds), src)
    for f in sorted(glob.glob(f"{B}/vision_cosmos_synthetic/runs/*.json")):
        j = json.load(open(f))
        for k in ("legacy_map50", "ultralytics_map50", "ultralytics_map50_95"):
            if k in j:
                put(f"syn.run.{os.path.basename(f)[:-5]}.{k}", j[k], f)


def reentry():
    f = f"{B}/reentry_landmark/results.json"
    r = json.load(open(f))
    put("lm.tau", r["tau"], f)
    put("lm.follow_up", r["follow_up"], f)
    for k, v in r["split_sizes"].items():
        put(f"lm.split.{k}.n", v, f)
        put(f"lm.split.{k}.events", r["split_events"][k], f)
    for k, v in r["leakage_report"].items():
        put(f"lm.leak.{k}", v, f)
    for mdl, res in r["results"].items():
        for k, v in res.items():
            put(f"lm.{mdl}.{k.replace('/', '.')}", v, f)
    for k, v in r["comparisons"].items():
        for kk in ("diff", "ci_low", "ci_high", "n_boot"):
            put(f"lm.cmp.{k.replace('/', '.')}.{kk}", v[kk], f)
    for mdl, d in r["conformal_sets_h68"].items():
        for kind, dd in d.items():
            for kk, v in dd.items():
                put(f"lm.conf.{mdl}.{kind}.{kk}", v, f)
    sel = r["mlp_selection"]["hazard_mlp"]
    put("lm.mlp.selected", json.dumps(sel["selected"]), f)
    put("lm.mlp.grid_size", len(sel["grid"]), f, "derived")
    put("lm.mlp.best_val_aucpr", max(g["val_auc_pr_h68"] for g in sel["grid"]), f, "derived")
    for k, v in r["selective_prediction_h68"].items():
        put(f"lm.selective.{k}.aurc", v["aurc"], f)
    f = f"{B}/reentry_surrogate/results.json"
    s = json.load(open(f))
    for part in ("test", "ood_test"):
        for k, v in s[part].items():
            put(f"sur.{part}.{k}", v, f)
    put("sur.ood_auroc", s["ood_auroc_ensemble_std"], f)
    fp = pd.DataFrame(s["footprints"])
    rel = (fp.cep50_sur_km - fp.cep50_ref_km).abs() / fp.cep50_ref_km
    put("sur.fp.n", len(fp), f, "derived")
    put("sur.fp.cep50_relerr_median", float(rel.median()), f, "derived", "|CEP50 sur - ref| / ref")
    put("sur.fp.cep50_relerr_max", float(rel.max()), f, "derived")
    put("sur.fp.coverage_median", float(fp.reference_coverage_of_surrogate_zone.median()), f, "derived")
    put("sur.fp.coverage_min", float(fp.reference_coverage_of_surrogate_zone.min()), f, "derived")
    put("sur.fp.speedup_median", float(fp.speedup.median()), f, "derived")
    f = f"{B}/reentry_surrogate/pipeline_fallback_modes.csv"
    d = pd.read_csv(f)
    for mode in ("ref", "none", "range", "spread"):
        put(f"sur.mode.{mode}.seconds_mean", float(d[f"{mode}_s"].mean()), f, "derived", "mean over 3 seeds")
        put(f"sur.mode.{mode}.cep50_mean", float(d[f"{mode}_cep50"].mean()), f, "derived")
        if f"{mode}_used" in d:
            put(f"sur.mode.{mode}.used_mean", float(d[f"{mode}_used"].mean()), f, "derived")
    put("sur.mode.n_seeds", len(d), f, "derived")


# ============================================================================ literature
def literature():
    L = "literature"
    dm = "kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)"
    for k, v in {"m1.full.best": 0.061, "m2.full.best": 0.079, "m1.lw.best": 0.786,
                 "m2.lw.best": 0.949, "m1.full.gstd5": 0.003, "m2.full.gstd5": 0.064,
                 "m1.lw.gstd5": 0.253, "m2.lw.gstd5": 0.075, "m1.lw.gstd3": 0.001,
                 "m2.lw.gstd3": 0.007, "m2.full.gstd3": 0.008}.items():
        put(f"lit.dmlr.{k}", v, dm, L)
    put("lit.dmlr.m1.full.best_name", "Telemanom-ESA-Pruned", dm, L)
    put("lit.dmlr.m2.full.best_name", "PCC", dm, L)
    put("lit.dmlr.m1.lw.best_name", "Telemanom-ESA-Pruned", dm, L)
    put("lit.dmlr.m2.lw.best_name", "Windowed iForest", dm, L)
    br = "brancato2027ast, abstract (full text not accessed)"
    for k, v in {"smed_full": 0.902, "qgpr_full": 0.883, "ool_full": 0.754,
                 "qgpr_lw": 0.888, "smed_lw": 0.811, "telemanom_full": 0.061}.items():
        put(f"lit.brancato.{k}", v, br, L)
    al = "alcarria2026, Table 1 (arXiv 2609.10017)"
    for k, v in {"m1_f05": 0.700, "m2_f05": 0.698, "m1_p": 0.852, "m1_r": 0.408,
                 "pitia_m1": 0.424, "pitia_m2": 0.882}.items():
        put(f"lit.alcarria.{k}", v, al, L)
    put("lit.allegrini.public", 0.931, "allegrini2026 (arXiv 2605.06681)", L)
    put("lit.allegrini.private", 0.853, "allegrini2026 (arXiv 2605.06681)", L)
    tc = "sahin2026tccad, figures/table7_official_V0_data.csv and fig5_budget_data.csv (Zenodo 23344000)"
    put("lit.tccad.seeds", 16, tc, L)
    put("lit.tccad.m1_channels", 24, tc, L)
    put("lit.tccad.m2_channels", 19, tc, L)
    put("lit.tccad.c0_anom_ewf05", 0.08951238809421605, tc, L)
    put("lit.tccad.aci_far_overshoot", "1.15–3.4", "sahin2026tccad, preregistration/en/PREREGISTRATION_H4.md section 0", L)
    put("lit.tccad.c3_anom_ewf05_order", "1e-5", tc, L)
    put("lit.tccad.deferral_pp", "29-33", "sahin2026tccad, Zenodo description", L)
    op = "ruszczak2025opssat, Table 3 (research_design/references/opssat_table3.html)"
    put("lit.opssat.fcnn_f1", 0.946, op, L)
    put("lit.opssat.fcnn_aucpr", 0.979, op, L)
    put("lit.opssat.adaboost_f1", 0.836, op, L)
    put("lit.opssat.xgbod_f1", 0.918, op, L)
    put("lit.heydari.shared_channels", 77, "heydari2026, abstract", L)
    sg = "chen2021spotgeo, Table 1 (final ranking, full test set)"
    put("lit.spotgeo.winner_1mf1", 0.0517, sg, L)
    put("lit.spotgeo.winner_f1", 1 - 0.0517, sg, "literature", "1 - (1 - F1) as published")
    put("lit.spotgeo.winner_mse", 33838.99, sg, L)
    put("lit.spotgeo.second_1mf1", 0.0557, sg, L)
    put("lit.spotgeo.second_mse", 30541.73, sg, L)
    put("lit.spotgeo.n_ranked", 14, sg, L)


# ============================================================================ original pipeline
def slug(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def original_pipeline():
    f = "thesis/evidence/original_pipeline_runs.json"
    R = json.load(open(f))
    put("orig.version", R["cosmos_version"], f)
    for row in R["default_summary"]:
        put(f"orig.default.{slug(row['quantity'])}", row["value"], f)
    g = R["monitor_grid"]
    faulted = [r for r in g if r["fault"] != "nominal"]
    put("orig.monitor.n_faulted", len(faulted), f, "derived")
    put("orig.monitor.n_detected", sum(r["detected"] for r in faulted), f, "derived")
    put("orig.monitor.nominal_false_alarm", [r for r in g if r["fault"] == "nominal"][0]["detected"], f)
    for fault in sorted({r["fault"] for r in faulted}):
        rr = [r for r in faulted if r["fault"] == fault]
        put(f"orig.monitor.{fault}.detected", sum(r["detected"] for r in rr), f, "derived")
        put(f"orig.monitor.{fault}.n", len(rr), f, "derived")
        d = [r["delay_minutes"] for r in rr if r["detected"]]
        if d:
            put(f"orig.monitor.{fault}.median_delay_min", float(np.median(d)), f, "derived")
    for sub in sorted({r["subsystem"] for r in faulted}):
        rr = [r for r in faulted if r["subsystem"] == sub]
        put(f"orig.monitor.sub.{slug(sub)}.detected", sum(r["detected"] for r in rr), f, "derived")
    put("orig.monitor.n_subsystems", len({r["subsystem"] for r in faulted}), f, "derived")
    put("orig.monitor.n_faults", len({r["fault"] for r in faulted}), f, "derived")
    put("orig.stuck.raises", R["stuck_fault"]["raises"], f)
    put("orig.stuck.error", R["stuck_fault"].get("error"), f)
    for r in R["scenario_table"]:
        for k, v in r.items():
            if k != "scenario":
                put(f"orig.scen.{slug(r['scenario'])}.{k}", v, f)
    for r in R["detection_single_vs_pair"]:
        for k, v in r.items():
            if k != "detector":
                put(f"orig.det.{slug(r['detector'])}.{k}", v, f)
    c = R["catalogue"]
    for r in c["overview"]:
        for k, v in r.items():
            if k != "OBJECT_TYPE":
                put(f"orig.cat.{slug(r['OBJECT_TYPE'])}.{k}", v, f)
    for r in c["clouds"]:
        for k, v in r.items():
            if k != "event":
                put(f"orig.cloud.{slug(r['event'])}.{k}", v, f)
    for k, v in c["calibration"].items():
        put(f"orig.lifetime_cal.{k}", v, f)
    for r in c["validation"]:
        for k in ("measured_median_years", "modelled_years", "ratio"):
            put(f"orig.lifetime_val.{slug(r['event'])}.{k}", r[k], f)
    for k, v in c["leo_rate"].items():
        put(f"orig.leo.{k}", v, f)
    z = R["zones"]
    for r in z["zones"]:
        n = f"{int(round(r['nominal'] * 100))}"
        for k, v in r.items():
            if k != "nominal":
                put(f"orig.zone.{n}.{k}", v, f)
    for k, v in z["footprint"].items():
        put(f"orig.fp.{k}", v, f)
    for k, v in z["cloud"].items():
        put(f"orig.cloud_demo.{k}", v, f)
    for r in R["failure_modes"]:
        for k, v in r.items():
            if k != "scenario":
                put(f"orig.fm.{slug(r['scenario'])}.{k}", v, f)
    for r in R["corridors"]:
        for k, v in r.items():
            if k not in ("corridor", "population_model"):
                put(f"orig.corr.{slug(r['corridor'])}.{k}", v, f)
    put("orig.population_model", R["corridors"][0]["population_model"], f)


def cross_cutting():
    """Values that summarise several entries (max/min over arms)."""
    # official seed-mean differences for the exploratory trees-vs-AR(8) comparison
    for m in (1, 2):
        a2 = EV[f"pre.m{m}.A2.conformal_bh.f05.mean"]["value"]
        a1 = EV[f"pre.m{m}.A1.conformal_bh.f05.mean"]["value"]
        put(f"xcomp.m{m}.x1.diff_official", a2 - a1, f"{PROT}/test/*.json", "derived",
            "official seed mean A2 minus A1 (conformal_bh)")
        for th in ("threshold",):
            a0t = EV[f"pre.m{m}.A0.{th}.f05.mean"]["value"]
            a1t = EV[f"pre.m{m}.A1.{th}.f05.mean"]["value"]
            put(f"xcomp.m{m}.a1_minus_a0.{th}", a1t - a0t, f"{PROT}/test/*.json + arms", "derived",
                "A1 minus A0 under the threshold layer (not a pre-registered test)")
    # test-half length in years and alarm runs per year
    for m in (1, 2):
        t0 = pd.Timestamp(EV[f"data.esa.m{m}.test.start"]["value"])
        t1 = pd.Timestamp(EV[f"data.esa.m{m}.test.end"]["value"])
        yrs = (t1 - t0).days / 365.25
        put(f"data.esa.m{m}.test.years", yrs, "data/DATASET_REPORT.json", "derived", "test end - start")
        put(f"data.esa.m{m}.test.blocks30", int(np.ceil((t1 - t0).days / 30)), "data/DATASET_REPORT.json",
            "derived", "number of 30-day bootstrap blocks in the test half")
        for arm in ("A0", "A1", "A2", "A3"):
            k = f"pre.m{m}.{arm}.conformal_bh.alarm_runs.mean"
            if k in EV:
                put(f"pre.m{m}.{arm}.conformal_bh.alarm_runs_per_year", EV[k]["value"] / yrs,
                    EV[k]["source"], "derived", "alarm runs / test-half years")
    alpha = EV["cfg.prereg.aci_alpha"]["value"]
    for g in (0.01, 0.05):
        after = alpha + g * (alpha - 1)
        put(f"aci.alpha_after_one_hit.g{g}", after, "update rule of eq. (aci)", "derived",
            "alpha_t + gamma (alpha - 1) starting from alpha")
        if after < 0:
            # the code treats alpha_t <= 0 as an infinite threshold (oof_benchmark.aci), so the channel can
            # fire again only once alpha_t > 0: floor(-after / (gamma alpha)) + 1 non-exceeding steps
            put(f"aci.recovery_steps.g{g}", int(np.floor(-after / (g * alpha) + 1e-9)) + 1, "update rule of eq. (aci)",
                "derived", "non-exceedance steps until alpha_t > 0 again (alpha_t <= 0 means no alarm)")
    put("prereg.frozen_code_files", EV["prereg.frozen_files"]["value"] - 1,
        "research_design/preregistration/freeze_09a98f5e160e.json", "derived", "frozen files minus the document")

    src = "outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json"
    for cal in ("aci", "aci_event"):
        vals = [e["value"] for k, e in EV.items() if re.match(rf"^pre\.seed\..*\.{cal}\.f05$", k)]
        put(f"pre.all.{cal}.f05.max", max(vals), src, "derived", f"max over all test arms and seeds ({len(vals)} runs)")
        put(f"pre.all.{cal}.n", len(vals), src, "derived")
    far = [e["value"] for k, e in EV.items() if re.match(r"^pre\.m\d\..*\.aci\.point_far\.(min|max)$", k)]
    zero = [k for k, e in EV.items() if re.match(r"^pre\.seed\..*\.aci_event\.alarm_runs$", k) and e["value"] == 0]
    tot = [k for k in EV if re.match(r"^pre\.seed\..*\.aci_event\.alarm_runs$", k)]
    put("pre.all.aci_event.zero_alarm_runs", len(zero), src, "derived", "runs where aci_event raised no alarm")
    put("pre.all.aci_event.n_runs", len(tot), src, "derived")
    means = [e["value"] for k, e in EV.items() if re.match(r"^pre\.m\d\.[A-Za-z0-9]+\.aci_event\.f05\.mean$", k)]
    put("pre.all.aci_event.f05.max_seedmean", max(means), src, "derived", "max over arms of the seed-mean")
    put("pre.all.aci.point_far.min", min(far), src, "derived")
    put("pre.all.aci.point_far.max", max(far), src, "derived")
    runs = [e["value"] for k, e in EV.items() if re.match(r"^pre\.m\d\..*\.aci\.alarm_runs\.max$", k)]
    put("pre.all.aci.alarm_runs.max", max(runs), src, "derived")

    # selection effect (A1 minus its calibration-window counterpart) by channel set and layer.
    # A0 is the exploratory AR(8) arm; on the lightweight subsets its counterpart is the
    # exploratory lightweight AR(8) arm. Point estimates only, not pre-registered tests.
    for m in (1, 2):
        for cal in ("threshold", "conformal_bh"):
            a1 = EV[f"pre.m{m}.A1.{cal}.f05.mean"]["value"]
            a0 = EV[f"pre.m{m}.A0.{cal}.f05.mean"]["value"]
            put(f"seleff.m{m}.full.{cal}", a1 - a0, f"{PROT}/test/*.json", "derived", "A1 minus A0")
            a1 = EV[f"pre.m{m}.A1lw.{cal}.f05.mean"]["value"]
            a0 = EV[f"explo.m{m}.lw.ar8.{cal}.f05.mean"]["value"]
            put(f"seleff.m{m}.lw.{cal}", a1 - a0, f"{PROT}/test/*.json + results.csv", "derived",
                "A1 lightweight minus exploratory lightweight AR(8) (calibration-window selection)")

    se = {k: e["value"] for k, e in EV.items() if k.startswith("seleff.")}
    put("seleff.n", len(se), f"{PROT}/test/*.json", "derived")
    put("seleff.n_positive", sum(1 for x in se.values() if x > 0), f"{PROT}/test/*.json", "derived")
    bh = [x for k, x in se.items() if k.endswith(".conformal_bh")]
    put("seleff.bh.n", len(bh), f"{PROT}/test/*.json", "derived")
    put("seleff.bh.n_negative", sum(1 for x in bh if x < 0), f"{PROT}/test/*.json", "derived")

    # where the trees' extra Mission 1 detections come from: (rate A2 - rate A1) x events
    tot = 0.0
    for grp in ("seen.Anomaly", "seen.Rare_Event", "unseen.Anomaly", "unseen.Rare_Event"):
        n = EV[f"evcls.m1.{grp}.n"]["value"]
        g = (EV[f"evcls.m1.{grp}.A2"]["value"] - EV[f"evcls.m1.{grp}.A1"]["value"]) * n
        tot += g
        put(f"evcls.m1.gain_a2_a1.{grp}", g, "event_classes_summary_m1.csv", "derived",
            "(detection rate A2 - A1) x number of events")
    put("evcls.m1.gain_a2_a1.total", tot, "event_classes_summary_m1.csv", "derived")

    # does the calibration shift order the seeds? Spearman rank correlation of shift ratio and
    # test F0.5 within each supervised arm (5 seeds each), and pooled over both arms
    from scipy.stats import spearmanr
    pats = {"A2": "m1_target_gbt_ar8_s{}_event", "A3": "m1_target_event_ar8_all_c3_s{}_event"}
    pool_r, pool_f = [], []
    for arm, pat in pats.items():
        r = [EV[f"shift.m1.{arm.lower()}_s{s}.ratio"]["value"] for s in range(5)]
        f05 = [EV[f"pre.seed.{pat.format(s)}.conformal_bh.f05"]["value"] for s in range(5)]
        put(f"shift.m1.{arm}.spearman_f05", float(spearmanr(r, f05)[0]), "calibration_shift_m1.json + test/*.json",
            "derived", "Spearman rho over 5 seeds; descriptive only")
        pool_r += r
        pool_f += f05
    put("shift.m1.supervised.spearman_f05", float(spearmanr(pool_r, pool_f)[0]),
        "calibration_shift_m1.json + test/*.json", "derived", "Spearman rho over the 10 supervised runs")

    # calibration-window size, test events and events detected per arm (for the journal paper)
    for m in (1, 2):
        a, b = [pd.Timestamp(x.strip()) for x in EV[f"cfg.periods.m{m}.cal"]["value"].split("to")]
        put(f"data.esa.m{m}.cal.n_steps", int((b - a).total_seconds() / EV[f"data.esa.m{m}.grid_seconds"]["value"]),
            "cfg.periods + grid", "derived", "calibration-window length / grid step")
        put(f"data.esa.m{m}.cal.months", round((b - a).days / 30.44, 1), "cfg.periods", "derived")
        n_ev = (EV[f"data.esa.m{m}.events_target.test.Anomaly"]["value"]
                + EV[f"data.esa.m{m}.events_target.test.Rare_Event"]["value"])
        put(f"data.esa.m{m}.events_target.test.total", n_ev, "data/DATASET_REPORT.json", "derived")
        for arm in ("A0", "A1", "A2", "A3"):
            k = f"pre.m{m}.{arm}.conformal_bh.r.mean"
            put(f"pre.m{m}.{arm}.conformal_bh.events_detected", EV[k]["value"] * n_ev, EV[k]["source"], "derived",
                "recall x number of target test events (seed mean for seeded arms)")

    tr = EV["data.esa.m1.events_all.train.Anomaly"]["value"] + EV["data.esa.m1.events_all.train.Rare_Event"]["value"]
    put("data.esa.m1.events_all.train.targets", tr, "data/DATASET_REPORT.json", "derived",
        "anomalies + rare events in the M1 training half (communication gaps excluded)")
    put("evcls.m1.gain_a2_a1.recall_based", EV["pre.m1.A2.conformal_bh.events_detected"]["value"]
        - EV["pre.m1.A1.conformal_bh.events_detected"]["value"], f"{PROT}/test/*.json", "derived",
        "difference of recall-based detected events, A2 seed mean minus A1")

    # gamma selected by the ACI comparator, over all test runs
    gam = [r["gamma"] for fn in glob.glob(os.path.join(ROOT, PROT, "test", "*.json"))
           for r in json.load(open(fn)) if r.get("calibrator") == "aci" and "gamma" in r]
    put("pre.all.aci.gamma005_runs", sum(1 for g in gam if abs(g - 0.05) < 1e-9), f"{PROT}/test/*.json", "derived",
        "runs whose selected gamma is 0.05")
    put("pre.all.aci.gamma_runs", len(gam), f"{PROT}/test/*.json", "derived")

    # rows of the alarm-layer table in which the per-channel threshold equals or beats BH
    rows = [(1, a) for a in ("A1", "A2", "A3", "A3nr", "A3c2", "A1lw")] + [(2, a) for a in ("A1", "A2", "A3", "A1lw")]
    ge = sum(1 for m, a in rows if EV[f"pre.m{m}.{a}.threshold.f05.mean"]["value"]
             >= EV[f"pre.m{m}.{a}.conformal_bh.f05.mean"]["value"] - 1e-12)
    put("layers.n_threshold_ge_bh", ge, f"{PROT}/test/*.json", "derived",
        "rows of the alarm-layer table where threshold F0.5 >= conformal-BH F0.5")
    put("layers.n_rows", len(rows), f"{PROT}/test/*.json", "derived")

    # bootstrap re-implementation versus official metric, and the p-value floor
    gaps = [abs(EV[f"conf.m{m}.{h}.diff"]["value"] - EV[f"conf.m{m}.{h}.diff_official"]["value"])
            for m in (1, 2) for h in ("h4", "h5", "h6")]
    put("conf.reimpl_gap_max", max(gaps), "CONFIRMATORY.json", "derived",
        "max |re-implemented - official difference| over the six comparisons")
    put("conf.p_floor", 1.0 / EV["conf.m1.h6.n_boot"]["value"], "CONFIRMATORY.json", "derived", "1 / replicates")
    put("conf.p_floor_holm", 3.0 / EV["conf.m1.h6.n_boot"]["value"], "CONFIRMATORY.json", "derived",
        "Holm factor 3 x floor")

    # what each verification check compares against (classification from the independent review
    # of tests/test_verify.py; see Appendix C)
    vclass = {1: ("published table", "ext"), 2: ("layer constants (by construction)", "int"),
              3: ("uses g0, not local g (mis-specified)", "mis"), 4: ("round trip (self-consistency)", "int"),
              5: ("same radius on both sides (self-consistency)", "int"),
              6: ("code constant (by construction)", "int"), 7: ("code constant (by construction)", "int"),
              8: ("tuned bump (by construction)", "int"), 9: ("physical limit", "ext"),
              10: ("closed-form bound", "ext"), 11: ("convergence (self-consistency)", "int"),
              12: ("physical expectation", "ext"), 13: ("run shorter than any decay (weak)", "int"),
              14: ("enforced mass closure (by construction)", "int"), 15: ("closed-form sampling law", "ext"),
              16: ("imposed shift (known answer)", "ext"), 17: ("imposed shift (known answer)", "ext"),
              18: ("conformal coverage guarantee", "ext"), 19: ("conformal coverage guarantee", "ext"),
              20: ("known property of the ellipse", "ext")}
    for i, (lab, _) in vclass.items():
        put(f"verify.{i}.reference", lab, "tests/test_verify.py", "derived", "classification of the reference")
    for c, key in (("ext", "external"), ("int", "internal"), ("mis", "misspecified")):
        put(f"verify.n_{key}", sum(1 for _, k in vclass.values() if k == c), "tests/test_verify.py", "derived")


def posthoc_v2():
    """Post-hoc V2 analyses (research_design/preregistration/POSTHOC_PLAN_V2.md); each file is
    optional, so the evidence file can be rebuilt while some analyses are still running."""
    PH = f"{ESA}/posthoc_v2"
    fr = f"{ROOT}/research_design/preregistration/posthoc_v2_freeze.json"
    if os.path.isfile(fr):
        f = json.load(open(fr))
        put("ph.plan_sha256", f["sha256"][:12], "research_design/preregistration/posthoc_v2_freeze.json", "config")
        put("ph.plan_written", f["written"], "research_design/preregistration/posthoc_v2_freeze.json", "config")
    p = f"{ROOT}/{PH}/p3_exact/exactness.json"
    if os.path.isfile(p):
        e = json.load(open(p))
        src = f"{PH}/p3_exact/exactness.json"
        put("ph.p3.n_checks", e["n_checks"], src)
        put("ph.p3.max_abs_diff", e["max_abs_diff"], src)
        put("ph.p3.passed", "passed" if e["passed"] else "failed", src)
    p = f"{ROOT}/{PH}/p3_exact/confirmatory_exact.json"
    if os.path.isfile(p):
        src = f"{PH}/p3_exact/confirmatory_exact.json"
        for m, rows in json.load(open(p))["missions"].items():
            for r in rows:
                h = r["comparison"].split(":")[0].lower()
                for k in ("diff", "ci95_low", "ci95_high", "p_boot", "p_holm", "verdict"):
                    if k in r:
                        put(f"ph.p3.m{m}.{h}.{k}", r[k], src)
    for p in glob.glob(f"{ROOT}/{PH}/p1_decompose/*.json"):
        d = json.load(open(p))
        src = os.path.relpath(p, ROOT)
        pre = f"ph.p1.m{d['mission']}.{d['channels']}.{d['layer']}"
        put(f"{pre}.reproduction", "passed" if d["reproduction"]["passed"] else "failed", src)
        put(f"{pre}.sigma_check", d["sigma_check"]["s0_recompute_max_abs_diff"], src)
        for c, r in d["cells"].items():
            put(f"{pre}.{c}.f05", r["official_EW_F0.5"], src)
            put(f"{pre}.{c}.fa_yr", r["false_alarms_per_year"], src)
            put(f"{pre}.{c}.recall", r["EW_recall"], src)
            put(f"{pre}.{c}.level", r["chosen_level"], src)
            put(f"{pre}.{c}.k", r["chosen_k"], src)
        for s in d["steps"]:
            for k in ("delta_official", "ci95_low", "ci95_high", "p_boot"):
                put(f"{pre}.{s['path']}.{s['factor']}.{k}", s[k], src)
        for f_, v in d["verdict"].items():
            put(f"{pre}.verdict.{f_}", v, src)
    for p in glob.glob(f"{ROOT}/{PH}/p2_forward/m*_summary.json"):
        d = json.load(open(p))
        src = os.path.relpath(p, ROOT)
        for layer, L in d["layers"].items():
            pre = f"ph.p2.m{d['mission']}.{layer}"
            for f_ in L["folds"]:
                for k in ("H", "W", "diff", "H_fa_yr", "W_fa_yr", "H_recall", "W_recall", "n_events"):
                    put(f"{pre}.fold{f_['fold']}.{k}", f_[k], src)
                put(f"{pre}.fold{f_['fold']}.start", f_["start"], src)
            put(f"{pre}.mean_diff", L["mean_diff"], src)
            put(f"{pre}.n_folds", len(L["folds"]), src)
            put(f"{pre}.n_folds_H_better", L["n_folds_H_better"], src)
            for k in ("diff", "ci95_low", "ci95_high", "p_boot", "A", "B"):
                put(f"{pre}.pooled.{k}", L["pooled"][k], src)
            put(f"{pre}.replicates", "yes" if L["replicates_H6"] else "no", src)
    for p in glob.glob(f"{ROOT}/{PH}/p4_aci/*.json"):
        d = json.load(open(p))
        src = os.path.relpath(p, ROOT)
        tag = d["tag"].replace(f"m{d['mission']}_target_", f"m{d['mission']}.")
        for name, r in d["chosen"].items():
            for k in ("alpha", "gamma", "k", "cv"):
                put(f"ph.p4.{tag}.{name}.{k}", r[k], src)
            put(f"ph.p4.{tag}.{name}.f05", r["anomaly_and_rare/EW_F0.5"], src)
            put(f"ph.p4.{tag}.{name}.fa_yr", r["false_alarms_per_year"], src)
    p4 = [json.load(open(p)) for p in glob.glob(f"{ROOT}/{PH}/p4_aci/*.json")]
    if p4:
        src = f"{PH}/p4_aci/*.json"
        put("ph.p4.n_arms", len(p4), src, "derived")
        put("ph.p4.gamma_positive.f05.max", max(d["chosen"]["gamma_positive"]["anomaly_and_rare/EW_F0.5"] for d in p4), src, "derived")
        put("ph.p4.any_gamma.n_gamma_zero", sum(d["chosen"]["any_gamma"]["gamma"] == 0 for d in p4), src, "derived")
        gbt = [d for d in p4 if "gbt" in d["tag"]]
        if gbt:
            for name in ("any_gamma", "gamma_positive", "gamma_zero"):
                agg(f"ph.p4.m1.gbt.{name}.f05", [d["chosen"][name]["anomaly_and_rare/EW_F0.5"] for d in gbt], src)
        # plan rule: "not usable at the event level" only if gamma > 0 stays >= 0.05 below the
        # threshold layer on every arm (same scores, frozen threshold-layer result)
        gaps = []
        for d in p4:
            thr = EV.get(f"pre.seed.{d['tag']}.threshold.f05")
            if thr is not None:
                gaps.append(d["chosen"]["gamma_positive"]["anomaly_and_rare/EW_F0.5"] - thr["value"])
        if gaps:
            put("ph.p4.rule.n_arms", len(gaps), src, "derived")
            put("ph.p4.rule.n_below", sum(g <= -0.05 for g in gaps), src, "derived")
            put("ph.p4.rule.n_at_or_above", sum(g >= 0 for g in gaps), src, "derived")
            put("ph.p4.rule.verdict", "not usable" if all(g <= -0.05 for g in gaps) else "claim not permitted",
                src, "derived")
    for p in glob.glob(f"{ROOT}/{PH}/p5_anomaly_arm/m*_summary.json"):
        d = json.load(open(p))
        src = os.path.relpath(p, ROOT)
        for arm, A in d["arms"].items():
            for k in ("mean", "std", "min", "max"):
                put(f"ph.p5.m{d['mission']}.{arm}.{k}", A[k], src)
            put(f"ph.p5.m{d['mission']}.{arm}.n", len(A["runs"]), src)
            put(f"ph.p5.m{d['mission']}.{arm}.fa_yr", float(np.mean([r["false_alarms_per_year"] for r in A["runs"]])), src)
    p = f"{ROOT}/{PH}/p6_seeds/summary.json"
    if os.path.isfile(p):
        d = json.load(open(p))
        src = f"{PH}/p6_seeds/summary.json"
        for arm, v in d["seed_values"].items():
            agg(f"ph.p6.m1.{arm}.f05", v, src)
        for r in d["comparisons"]:
            h = r["comparison"].split(":")[0].lower()
            for k in ("diff", "ci95_low", "ci95_high", "p_boot", "n_seeds_A", "n_seeds_B"):
                put(f"ph.p6.m1.{h}.{k}", r[k], src)
        for arm, rows in d["new_runs"].items():
            put(f"ph.p6.m1.{arm}.new_runs_ok", sum(1 for r in rows if r.get("calibrator") == "conformal_bh"), src)
    p = f"{ROOT}/{PH}/p7_operational/summary.json"
    if os.path.isfile(p):
        d = json.load(open(p))
        src = f"{PH}/p7_operational/summary.json"
        for m, arms_ in d["missions"].items():
            for arm, A in arms_.items():
                for k, v in A["mean"].items():
                    put(f"ph.p7.m{m}.{arm}.{k}", v, src)
    p = f"{ROOT}/{PH}/p8_provenance/summary.json"
    if os.path.isfile(p):
        d = json.load(open(p))
        src = f"{PH}/p8_provenance/summary.json"
        for c in d["checks"]:
            put(f"ph.p8.m{c['mission']}.passed", "passed" if c["passed"] else "failed", src)
            put(f"ph.p8.m{c['mission']}.max_abs_cv_diff", c["max_abs_cv_diff"], src)


def main():
    for fn in (environment, verify_checks, datasets, esa_exploratory, preregistered,
               config_values, opssat, learned_monitor, vision, reentry, literature,
               original_pipeline, cross_cutting, posthoc_v2):
        try:
            fn()
        except Exception as e:                                       # report and stop
            raise SystemExit(f"{fn.__name__} failed: {e!r}")
    os.makedirs("thesis/evidence", exist_ok=True)
    with open("thesis/evidence/evidence.json", "w") as f:
        json.dump(EV, f, indent=1, default=str, sort_keys=True)
    with open("thesis/evidence/evidence_map.md", "w") as f:
        f.write("# Evidence map (generated by thesis/tools/build_evidence.py)\n\n"
                "| key | value | kind | source | note |\n|---|---|---|---|---|\n")
        for k in sorted(EV):
            e = EV[k]
            v = e["value"]
            vs = f"{v:.6g}" if isinstance(v, float) else str(v)
            f.write(f"| `{k}` | {vs[:80]} | {e['kind']} | {e['source']} | {e.get('note', '')} |\n")
    print(f"{len(EV)} evidence entries -> thesis/evidence/")


if __name__ == "__main__":
    main()
