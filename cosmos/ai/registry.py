"""Model registry: what each COSMOS-AI model takes, returns, was trained on, and has shown.

    from cosmos.ai import registry
    registry.table()                 # one row per model card
    registry.card("esa_conformal_bh_ar8").limitations

A card is built from the files the benchmarks wrote, never typed in by hand, so a card can
only claim what an output file supports.  ``status`` is one of

``validated-real``        scored once on a real held-out test set under a fixed protocol
``validated-synthetic``   scored on generated data only (simulator or synthetic images)
``baseline``              a reference the learned models are compared against
``not-recommended``       measured, and worse than a simpler alternative on the same test
``not-run``               the benchmark output does not exist (yet)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

import pandas as pd

from ..common import output_dir


@dataclass
class ModelCard:
    name: str
    feature: str                          # health | debris | reentry
    task: str
    inputs: str
    outputs: str
    uncertainty: str
    preprocessing: str
    training_data: str
    status: str = "not-run"
    version: str = "n/a"                  # checkpoint SHA-256 prefix, or the config hash
    checkpoint: str | None = None
    evidence: str | None = None           # benchmark output that supports the card
    metrics: dict = field(default_factory=dict)
    limitations: list = field(default_factory=list)


def _sha(path: str | None) -> str:
    if not path or not os.path.isfile(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def _json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _bench(*parts):
    return os.path.join(output_dir("benchmarks"), *parts)


def _esa_rows():
    p = _bench("telemetry_esa_adb", "results.csv")
    return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()


def _esa_card(name, encoder, calibrator, desc, status_if_run, limits):
    df = _esa_rows()
    m = {}
    if len(df):
        for mission in (1, 2):
            sel = df[(df.encoder == encoder) & (df.calibrator == calibrator)
                     & (df.mission == mission) & (df.channels == "target")
                     & ~df.telecommands.fillna(False).astype(bool)]
            if len(sel):
                m[f"M{mission}_EW_F0.5_mean"] = float(sel["anomaly_and_rare/EW_F0.5"].mean())
                m[f"M{mission}_EW_F0.5_n_seeds"] = int(len(sel))
                m[f"M{mission}_false_alarms_per_year_mean"] = float(sel.false_alarms_per_year.mean())
    return ModelCard(
        name=name, feature="health", task="binary event alarms + affected channels on "
        "spacecraft telemetry (ESA-ADB official protocol)",
        inputs="per-channel telemetry on the mission grid (30 s / 18 s), robust-normalised with "
               "training-half statistics; optional telecommand features",
        outputs="per-channel binary alarms; global alarm = OR; channels named in each run",
        uncertainty="conformal p-values per channel, Benjamini-Hochberg across channels"
        if calibrator.startswith("conformal") else "per-channel percentile threshold",
        preprocessing="cosmos.ai.data.preprocess_telemetry.prepare_esa_official (verified "
                      "against a literal port of the official scripts)",
        training_data="ESA-ADB (Zenodo 12528696, v1.0; = rev. 15237121 except channels.csv), "
                      "training half only", status=status_if_run if m else "not-run",
        evidence="outputs/benchmarks/telemetry_esa_adb/results.csv", metrics=m,
        limitations=limits + ["exchangeability needed by conformal p-values is violated by "
                              "serial dependence and drift; realised false-alarm rates are "
                              "measured, not guaranteed"], version=f"{encoder}/{calibrator}")


def cards() -> list:
    out = []
    lim_esa = ["Mission 1 calibration period holds only 9 labelled events: selection is noisy",
               "irregular sampling is resampled (benchmark requirement R8 not met)"]
    for enc, label in (("ar8", "COSMOS AR(8) forecaster"), ("patchtst", "PatchTST"),
                       ("tcn", "causal TCN"), ("chronos-bolt-small", "Chronos-Bolt-small "
                                                                       "(zero-shot)"),
                       ("persistence", "persistence forecast")):
        for cal in ("conformal_bh", "threshold"):
            out.append(_esa_card(f"esa_{cal}_{enc}", enc, cal, label, "validated-real",
                                 list(lim_esa)))
    # ---- OPSSAT-AD
    p = _bench("telemetry_opssat", "summary_official.csv")
    s = pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()
    for model, desc in (("hybrid", "raw-fragment CNN + feature MLP"), ("fcnn", "feature MLP"),
                        ("cnn", "raw-fragment CNN"), ("xgboost", "gradient-boosted trees")):
        r = s[(s.model == model) & (s.train_frac == 1.0)] if len(s) else s
        ck = sorted(glob.glob(os.path.join(output_dir("experiments"),
                                           f"*opssat_{model}_official_f100_s0", "checkpoints",
                                           "best.pt")))
        out.append(ModelCard(
            name=f"opssat_{model}", feature="health",
            task="anomalous / nominal classification of OPS-SAT telemetry fragments",
            inputs="fragment resampled to 256 points (z-score, log time gap, scaled value) "
                   "and/or the 18 official features", outputs="P(anomalous); label at 0.5",
            uncertainty="none beyond the probability (calibration measured: ECE)",
            preprocessing="cosmos.ai.data.preprocess_telemetry.opssat_tensors",
            training_data="OPSSAT-AD (Zenodo 15108715), official train split minus a 20 % "
                          "validation fold", status="validated-real" if len(r) else "not-run",
            version=_sha(ck[-1]) if ck else "n/a", checkpoint=ck[-1] if ck else None,
            evidence=p, metrics={k: float(r.iloc[0][k]) for k in
                                 ("f1_mean", "f1_std", "auc_pr_mean", "mcc_mean")} if len(r) else {},
            limitations=["the official split interleaves train and test fragments in time; "
                         "a time-blocked split lowers every model (see summary_time_blocked)",
                         "one spacecraft, 9 channels"]))
    # ---- vision
    for model, label in (("heatmap_unet", "heat-map U-Net"),):
        rs = [json.load(open(f)) for f in glob.glob(_bench("vision_spotgeo", "runs",
                                                          "spotgeo_unet_both_f100_s*.json"))]
        out.append(ModelCard(
            name="spotgeo_heatmap_unet", feature="debris",
            task="point localisation of unresolved GEO targets in 5-frame sequences",
            inputs="5 frames -> COSMOS registered differences + high-passed frames (10 ch)",
            outputs="per-frame (x, y) points; official SpotGEO scoring",
            uncertainty="heat-map confidence; threshold chosen on validation",
            preprocessing="cosmos.ai.vision.spotgeo.front_end",
            training_data="SpotGEO v2 (Zenodo 4432143) train, 10 % of sequences held out",
            status="validated-real" if rs else "not-run",
            evidence=_bench("vision_spotgeo", "runs"),
            metrics={"test_f1_mean": float(pd.Series([r["test_f1"] for r in rs]).mean()),
                     "n_seeds": len(rs)} if rs else {},
            limitations=["GEO, ground-based, one sensor: no claim about LEO debris imagery"]))
    syn = [json.load(open(f)) for f in glob.glob(_bench("vision_cosmos_synthetic", "runs",
                                                       "heatmap_unet_s*.json"))]
    out.append(ModelCard(
        name="cosmos_synthetic_heatmap", feature="debris",
        task="detection + classification of movers on COSMOS synthetic difference images",
        inputs="640 x 640 registered difference image", outputs="boxes, classes, scores",
        uncertainty="confidence scores", preprocessing="cosmos.debris.register_and_difference",
        training_data="cosmos_synthetic/yolo_v1 (generated, scene-split)",
        status="validated-synthetic" if syn else "not-run",
        evidence=_bench("vision_cosmos_synthetic", "runs"),
        metrics={"map50_95_mean": float(pd.Series([r["map50_95"] for r in syn]).mean()),
                 "n_seeds": len(syn)} if syn else {},
        limitations=["synthetic only", "class labels differ only in PSF width and flux"]))
    # ---- re-entry
    r = _json(_bench("reentry_landmark", "results.json"))
    for model, status_ok in (("xgboost", "validated-real"), ("hazard_mlp", "not-recommended"),
                             ("physics", "baseline")):
        res = (r or {}).get("results", {}).get(model, {})
        out.append(ModelCard(
            name=f"reentry_{model}", feature="reentry",
            task="P(re-entry within 30 / 68 days) from a catalogue landmark",
            inputs="tau-snapshot SATCAT orbit fields, RCS, type, owner, age; tau GP elements "
                   "where available", outputs="horizon probabilities; time-to-decay ranking",
            uncertainty="ensemble spread (MLP); Mondrian conformal sets (val-calibrated)",
            preprocessing="cosmos.ai.data.preprocess_catalogue.build_landmark (leakage report "
                          "must pass)", training_data="CelesTrak SATCAT 2026-08-16 -> "
                                                      "2026-10-23, split by launch",
            status=status_ok if res else "not-run",
            evidence=_bench("reentry_landmark", "results.json"),
            metrics={k: res[k] for k in ("all/h68/auc_pr", "all/h30/auc_pr", "all/c_index")
                     if k in res},
            limitations=["one landmark, 68-day horizon, 38 test events",
                         "payload decays include controlled de-orbits",
                         "no impact-location prediction is made or validated"]))
    sur = _json(_bench("reentry_surrogate", "results.json"))
    out.append(ModelCard(
        name="reentry_propagator_surrogate", feature="reentry",
        task="impact point / fall time of a fragment from its break-up state",
        inputs="break-up altitude, speed, flight-path angle, heading, latitude, BC, mass, "
               "density factor, delta-v", outputs="downrange, crossrange, fall time, speed",
        uncertainty="5-member ensemble spread (also the out-of-range flag)",
        preprocessing="cosmos.ai.surrogate.features",
        training_data="reentry/simulated/propagator_surrogate_v1 (COSMOS propagator output)",
        status="validated-synthetic" if sur else "not-run",
        evidence=_bench("reentry_surrogate", "results.json"),
        metrics={k: sur["test"][k] for k in ("position_error_km_median",
                                             "position_error_km_p95")} if sur else {},
        limitations=["agreement with the COSMOS simulator only — not with real re-entries",
                     "valid inside the sampled input ranges; use the ensemble flag outside"]))
    return out


def table() -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in asdict(c).items() if k not in ("limitations",)}
                         for c in cards()])


def card(name: str) -> ModelCard:
    for c in cards():
        if c.name == name:
            return c
    raise KeyError(name)


def write() -> str:
    p = os.path.join(output_dir("registry"), "model_cards.json")
    with open(p, "w") as f:
        json.dump([asdict(c) for c in cards()], f, indent=2, default=str)
    return p
