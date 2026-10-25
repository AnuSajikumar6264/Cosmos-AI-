#!/usr/bin/env python
"""Build notebooks/COSMOS_AI_Experiments.ipynb (the code lives here so it can be regenerated).

    .venv/bin/python scripts/build_ai_notebook.py
    .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=cosmos-ai notebooks/COSMOS_AI_Experiments.ipynb
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# COSMOS-AI — deep-learning experiments on real and simulated data

This notebook walks through the COSMOS-AI layer on the datasets that were actually downloaded.
Every number it prints is either computed here or read from `outputs/benchmarks/`, where the
benchmark scripts wrote it. Evidence classes, as in the research design:

* **REAL**: ESA-ADB, OPSSAT-AD, SpotGEO, CelesTrak, ESA recovered debris.
* **SYNTHETIC / SIM**: COSMOS telemetry simulator, COSMOS frame pairs, COSMOS re-entry propagator.

Heavy training is not repeated here: the notebook loads the checkpoints the benchmarks saved.
It retrains only the small models where that takes seconds.
""")
code("""
import sys, os, json, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
os.environ.setdefault("COSMOS_DEVICE", "cpu")      # the notebook runs on CPU
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import cosmos
from cosmos.ai.data import download
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

def table(path, **kw):
    # a benchmark table, or a clear note when that benchmark has not been run yet
    if not os.path.isfile(path):
        print(f"not run yet: {path} is missing (run the benchmark, then python -m cosmos.ai.report)")
        return pd.DataFrame()
    return pd.read_csv(path, **kw)

print("cosmos", cosmos.__version__)
download.status()
""")
md("""
## 1. Telemetry — ESA-ADB under the official protocol

The archives (Zenodo 12528696) were MD5-checked. The official benchmark's preprocessing was
re-implemented and checked against a literal port of the official pandas code (`tests/test_ai.py`).
The result is the official 30 s grid for Mission 1: 7 years for training and 7 years for testing.
""")
code("""
from cosmos.ai.data.preprocess_telemetry import load_esa_split, official_test_labels, ESA_PROTOCOL
te = load_esa_split(1, "test")
lab = official_test_labels(1, te)
lab = lab[lab.Channel.isin(ESA_PROTOCOL[1]["target"])]
print(f"test grid {te.n:,} x {len(te.channels)} channels from {te.timestamps()[0]} to {te.timestamps()[-1]}")
lab.groupby("ID").Category.first().value_counts()
""")
code("""
# one labelled event, the channels the operators marked, and the surrounding telemetry
ev = lab.groupby("ID").agg(s=("StartTime", "min"), e=("EndTime", "max"), cat=("Category", "first"),
                           n=("Channel", "nunique")).sort_values("n", ascending=False)
eid = ev[ev.cat == "Anomaly"].index[0]; s, e = ev.loc[eid, "s"], ev.loc[eid, "e"]
chans = sorted(lab[lab.ID == eid].Channel.unique(), key=lambda c: int(c.split("_")[1]))[:4]
i0, i1 = te.index_of(s - (e - s) * 2), te.index_of(e + (e - s) * 2)
ts = te.timestamps()[i0:i1]
fig, axes = plt.subplots(len(chans), 1, figsize=(11, 1.6 * len(chans)), sharex=True)
for ax, c in zip(np.atleast_1d(axes), chans):
    ax.plot(ts, te.X[i0:i1, te.channel_index([c])[0]], lw=0.8, color="#2a78d6")
    ax.axvspan(s, e, color="#e1e0d9"); ax.set_ylabel(c, fontsize=8)
axes[0].set_title(f"ESA-ADB Mission 1, event {eid} (shaded: operator label)", loc="left", fontsize=10)
plt.tight_layout()
""")
md("""
### Results: every encoder through the same calibrators

Each arm uses the same steps:

1. Train on nominal windows of the training half.
2. Select the calibrator on the last 9 months of the training half, by cross-fitting.
3. Score the test half once, with the vendored official metric code.

The vertical line in the figure is the best value published in DMLR 2026.
""")
code("""
t = table("../outputs/benchmarks/telemetry_esa_adb/summary_table.csv")
cols = ["mission", "channels", "encoder", "calibrator", "n_seeds", "anomaly_and_rare/EW_F0.5|mean",
        "anomaly_and_rare/EW_F0.5|std", "anomaly_and_rare/EW_precision|mean", "anomaly_and_rare/EW_recall|mean",
        "anomaly_and_rare/channel_F0.5|mean", "false_alarms_per_year|mean"]
t if t.empty else t[t.channels == "target"][cols].sort_values(["mission", "anomaly_and_rare/EW_F0.5|mean"], ascending=[True, False]).round(3)
""")
code("""
from IPython.display import Image
Image("../outputs/benchmarks/figures/esa_adb_target_ew_f05.png")
""")
md("""
### Uncertainty and explanation: per-channel conformal p-values at an alarm

The alarm rule is the explanation. Each channel's normalised forecast error gets a conformal
p-value against the nominal calibration scores, Benjamini-Hochberg runs across channels, and the
rejected channels are the ones reported. The scores below are the saved test scores of the
PatchTST arm (seed 0).
""")
code("""
import glob
from cosmos.ai.uncertainty import conformal_pvalues
from cosmos.ai.explain import channel_evidence
from cosmos.ai.telemetry.forecasting import MissionData
runs = sorted(glob.glob("../outputs/experiments/*esa_m2_target_patchtst_s0_s0"))
if runs and os.path.isfile(os.path.join(runs[-1], "scores_test.npy")):
    D = MissionData(2)
    S_cal = np.load(os.path.join(runs[-1], "scores_cal.npy")); S_te = np.load(os.path.join(runs[-1], "scores_test.npy"), mmap_mode="r")
    a, b = D.p["cal"]
    cal = [np.sort(S_cal[:, c][(D.Ltr[a:b, c] == 0) & np.isfinite(S_cal[:, c])]) for c in range(S_cal.shape[1])]
    labm2 = official_test_labels(2, D.test); labm2 = labm2[labm2.Channel.isin(D.channels)]
    eid = labm2.ID.iloc[0]; row = labm2[labm2.ID == eid]
    k = D.test.index_of(row.StartTime.min()) + 5
    p = np.array([conformal_pvalues(cal[c], np.nan_to_num(S_te[k:k+1, c], nan=-1))[0] for c in range(len(cal))])
    ev = pd.DataFrame(channel_evidence(p, D.channels, q=1e-4)).head(8)
    ev["labelled_by_operators"] = ev.channel.isin(row.Channel)
    print(f"event {eid} ({row.Category.iloc[0]}), {D.test.timestamps()[k]}")
    display(ev)
else:
    print("PatchTST M2 seed 0 scores not found — run the ESA benchmark first")
""")
md("""
## 2. Telemetry — OPSSAT-AD (official split, 5 seeds)
""")
code("""
from cosmos.ai.data.preprocess_telemetry import load_opssat_ad
d = load_opssat_ad(); f = d["features"]
print(pd.crosstab(f.train.map({1: "train", 0: "test"}), f.anomaly.map({1: "anomalous", 0: "nominal"})))
g = d["segments"]; seg = g[g.segment == f[f.anomaly == 1].index[0]]
plt.figure(figsize=(9, 2)); plt.plot(seg.timestamp, seg.value, lw=1, color="#2a78d6"); plt.title(f"anomalous fragment {seg.segment.iloc[0]} ({seg.channel.iloc[0]})", loc="left", fontsize=10)
""")
code("""
s = table("../outputs/benchmarks/telemetry_opssat/summary_official.csv")
s[s.train_frac == 1.0][["model", "n_seeds", "f1_mean", "f1_std", "auc_pr_mean", "auc_pr_std", "mcc_mean", "ece_mean"]].sort_values("auc_pr_mean", ascending=False).round(3)
""")
code("""
table("../outputs/benchmarks/telemetry_opssat/paired_comparisons.csv").round(4)
""")
md("""
### Explanation: Integrated Gradients on the hybrid model, with a randomisation check

The hybrid checkpoint (seed 0) is loaded and Integrated Gradients are computed over the time axis
of one test fragment. A faithful attribution must change when the weights are re-initialised
(Adebayo et al. 2018), so a low rank correlation passes the check.
""")
code("""
import torch
from cosmos.ai.telemetry.opssat import OpssatData
from cosmos.ai.telemetry.models import build_classifier
from cosmos.ai.explain import integrated_gradients, completeness_gap, randomisation_check
from sklearn.preprocessing import StandardScaler
D = OpssatData(256); tr, va, te_idx = D.split("official")
ck = sorted(glob.glob("../outputs/experiments/*opssat_hybrid_official_f100_s0/checkpoints/best.pt"))
if ck:
    net = build_classifier("hybrid", in_ch=3, n_features=18); net.load_state_dict(torch.load(ck[-1], map_location="cpu")["state_dict"]); net.eval()
    F = torch.tensor(StandardScaler().fit(D.F[tr]).transform(D.F), dtype=torch.float32)
    i = te_idx[D.y[te_idx] == 1][0]
    x = torch.tensor(D.X[i:i+1]); fi = F[i:i+1]
    f = lambda z: torch.sigmoid(net(z, fi.expand(len(z), -1), None))
    att = integrated_gradients(f, x, steps=64)
    rho = randomisation_check(net, lambda m, z: torch.sigmoid(m(z, fi.expand(len(z), -1), None)), x)
    print("P(anomalous) =", round(float(f(x)), 4), "| completeness gap", round(completeness_gap(f, x, att), 4),
          "| randomisation rank corr", round(rho, 3))
    print("sanity check:", "PASS" if rho < 0.5 else "FAIL - the attribution barely depends on the learned weights "
          "(it mostly follows the input's magnitude), so this IG map is NOT evidence of what the model uses")
    fig, ax = plt.subplots(2, 1, figsize=(9, 3.5), sharex=True)
    ax[0].plot(D.X[i, 0], color="#2a78d6", lw=1); ax[0].set_ylabel("z-scored value")
    ax[1].plot(att[0, 0].numpy(), color="#eb6834", lw=1); ax[1].set_ylabel("IG attribution")
    ax[0].set_title(f"test fragment {D.feats.index[i]} (labelled anomalous)", loc="left", fontsize=10); plt.tight_layout()
else:
    print("hybrid checkpoint not found")
""")
md("""
## 3. Telemetry — the learned monitor inside the original COSMOS chain (SYNTHETIC)

`LearnedHealthMonitor` has the same interface as the classical `HealthMonitor`. The benchmark ran
both monitors on 240 faulted and 40 fault-free simulated runs (seed sets 5-9, scored once, after
seed sets 0-4 had been used to fix the design). These are simulator results only.
""")
code("""
table("../outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv").groupby("monitor").mean(numeric_only=True).round(3).drop(columns="seed_set")
""")
code("""
r_classic = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, n_samples=300, seed=3)
r_learned = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, n_samples=300, seed=3, backend="learned")
pd.DataFrame({"classical": r_classic.summary().value.values, "learned backend": r_learned.summary().value.values},
             index=r_classic.summary().quantity).head(10)
""")
md("""
## 4. Vision — SpotGEO (REAL) and COSMOS frame pairs (SYNTHETIC)

SpotGEO frames contain star streaks, clouds and a pointing drift between frames. The unchanged
COSMOS registration front end aligns the star field so that the GEO target stands out in the
difference image.
""")
code("""
from cosmos.ai.vision import spotgeo as S
lab_sg = S.load_annotations("train"); seq = 31
X = S.load_cached("train", seq)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
F0 = S.read_frames("train", seq)[0]
for a, img, title in zip(ax, (F0, X[0, 1], X[0, 0]), ("raw frame 1", "high-passed", "registered difference")):
    lo, hi = np.percentile(img, [1, 99.7]); a.imshow(img, cmap="gray", vmin=lo, vmax=hi); a.set_title(title, fontsize=10); a.axis("off")
    c = lab_sg[seq][1]; a.scatter(c[:, 0], c[:, 1], s=180, facecolors="none", edgecolors="#eb6834")
plt.tight_layout()
""")
code("""
v = table("../outputs/benchmarks/vision_spotgeo/summary.csv")
v if v.empty else v[[c for c in v.columns if c.split("|")[0] in ("model", "inputs", "train_frac", "n_seeds", "test_f1", "test_one_minus_f1", "test_precision", "test_recall", "latency_ms_per_sequence")]].round(4)
""")
code("""
# the trained heat-map detector on one TEST sequence
from cosmos.ai.vision.heatmap import build, decode
from cosmos.ai.vision.train_spotgeo import to_input
ck = sorted(glob.glob("../outputs/experiments/*spotgeo_unet_both_f100_s0/checkpoints/best.pt"))
if ck:
    st = torch.load(ck[-1], map_location="cpu"); m = build("unet", 10, 5); m.load_state_dict(st["state_dict"]); m.eval()
    lab_te = S.load_annotations("test"); seq = 17; Xt = S.load_cached("test", seq)
    with torch.no_grad():
        dets = decode(m(torch.tensor(to_input(Xt, "both")[None])), st["threshold"])[0]
    pred = {seq: {f + 1: dets[f][:, :2] for f in range(5)}}
    print("official score on this sequence:", {k: round(v, 3) for k, v in S.score(pred, "test", [seq], lab_te).items() if k in ("precision", "recall", "f1")})
    plt.figure(figsize=(8, 5.5)); plt.imshow(Xt[0, 0], cmap="gray", vmin=-3, vmax=6)
    g = lab_te[seq][1]; plt.scatter(g[:, 0], g[:, 1], s=220, facecolors="none", edgecolors="#1baf7a", label="truth")
    plt.scatter(dets[0][:, 0], dets[0][:, 1], s=40, marker="x", color="#eb6834", label="detected"); plt.legend(); plt.axis("off")
else:
    print("SpotGEO heat-map checkpoint not found")
""")
code("""
table("../outputs/benchmarks/vision_cosmos_synthetic/summary.csv").round(3)
""")
md("""
## 5. Re-entry — a leakage-free landmark on the real catalogue

The population is every object in orbit on 2026-08-16. Its features are only what the 2026-08-16
snapshot knew; its outcome is the decay date in the 2026-10-24 snapshot, censored on 2026-10-23.
The splits are by launch.
""")
code("""
from cosmos.ai.data.preprocess_catalogue import leakage_report
L = pd.read_csv("../data/feature3_reentry/derived/landmark_2026-08-16_to_2026-10-23.csv")
print(json.load(open("../data/feature3_reentry/derived/landmark_2026-08-16_to_2026-10-23_exclusions.json")))
print(leakage_report(L))
r = json.load(open("../outputs/benchmarks/reentry_landmark/results.json"))
pd.DataFrame({m: {k: v for k, v in x.items() if k in ("all/h68/auc_pr", "all/h30/auc_pr", "all/c_index", "all/h68/brier", "all/h68/ece")}
              for m, x in r["results"].items()}).T.round(4)
""")
code("""
# uncertainty: label-conditional conformal sets cover the rare events, marginal sets do not
pd.DataFrame({m: {"marginal coverage (all)": c["marginal"]["coverage"], "marginal coverage (events)": c["marginal"]["coverage_on_events"],
                  "Mondrian coverage (events)": c["mondrian"]["coverage_events"], "Mondrian coverage (non-events)": c["mondrian"]["coverage_nonevents"]}
              for m, c in r["conformal_sets_h68"].items()}).T.round(3)
""")
code("""
rec = pd.read_csv("../data/feature3_reentry/historical/recovered_debris_esa_with_satcat.csv")
plt.figure(figsize=(9, 4.2)); plt.scatter(rec.lon, rec.lat, s=14, color="#2a78d6")
plt.title(f"ESA list of recovered re-entry debris ({rec.cospar_id.nunique()} objects); "
          f"{int(rec.within_inclination_band.sum())} of {int(rec.INCLINATION.notna().sum())} linked fragments lie inside their orbit's latitude band",
          loc="left", fontsize=9); plt.xlabel("longitude"); plt.ylabel("latitude"); plt.grid(color="#e1e0d9")
""")
md("""
## 6. Re-entry — a neural surrogate of the COSMOS propagator (SIM), inside the pipeline

The surrogate replaces the propagator in the Monte Carlo footprint. Samples that leave its training
ranges, or that its ensemble disagrees on, fall back to the reference code. Agreement is with the
COSMOS simulator only, not with real re-entries.
""")
code("""
sres = json.load(open("../outputs/benchmarks/reentry_surrogate/results.json"))
print({k: round(v, 3) for k, v in sres["test"].items()}); print("OOD AUROC of ensemble spread:", round(sres["ood_auroc_ensemble_std"], 3))
pd.DataFrame(sres["footprints"])[["alt_km", "cep50_ref_km", "cep50_sur_km", "reference_coverage_of_surrogate_zone", "speedup"]].round(2)
""")
code("""
import time
t0 = time.time(); ref = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, n_samples=600, seed=5); t_ref = time.time() - t0
t0 = time.time(); sur = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, n_samples=600, seed=5, footprint_backend="surrogate"); t_sur = time.time() - t0
print(f"reference {t_ref:.1f} s, surrogate {t_sur:.1f} s; surrogate used for {sur.footprint.surrogate_fraction:.0%} of samples "
      f"(inside training ranges {sur.footprint.inside_training_ranges:.0%}, spread threshold {sur.footprint.spread_threshold_km:.1f} km)")
pd.DataFrame({"reference": ref.footprint.summary(), "surrogate": sur.footprint.summary()})
""")
md("""
## 7. The decision layer: combine only compatible, validated evidence

No paired telemetry, imagery and re-entry observations of one object exist, so nothing is trained
jointly across features. The decision layer combines evidence only when the claims are the same,
the object and time window match, the card status is validated, and the input is in-domain.
Otherwise it abstains.
""")
code("""
from cosmos.ai.fusion import Evidence, decide, evidence_from_monitor
from cosmos.ai import registry
ev_t = evidence_from_monitor(r_learned.monitor, "learned_monitor", "validated-synthetic", object_id="SIM-SAT-1")
ev_r = Evidence("reentry", "reentry_xgboost", "re-entry within 68 d", 0.7, "validated-real", object_id="NORAD-12345")
print(decide([ev_t, ev_r], "anomalous telemetry", object_id="SIM-SAT-1"))
print(decide([ev_r], "anomalous telemetry", object_id="SIM-SAT-1"))
registry.table()[["name", "feature", "status", "version", "metrics"]]
""")
md("""
## Summary

Read `outputs/benchmarks/SUMMARY.md` for the component table with every measured number, the
evidence paths and the remaining limitations.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "COSMOS-AI (.venv)", "language": "python",
                                "name": "cosmos-ai"}
p = os.path.join(ROOT, "notebooks", "COSMOS_AI_Experiments.ipynb")
nbf.write(nb, p)
print("wrote", p, len(cells), "cells")
