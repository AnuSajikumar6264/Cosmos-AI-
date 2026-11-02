# COSMOS — Satellite Health, Space Debris and Atmospheric Re-entry

**AI-Enabled Space Environment Digital Twin**

A rocket discards parts at every stage. Some fall straight back through the atmosphere; some
stay in outer space. Once the satellite reaches its orbit, its health has to be monitored.
This project joins those three problems into one system.

| | Feature | Question it answers | Package |
|---|---|---|---|
| **1** | Satellite health monitoring | Is the newly launched satellite healthy? | `cosmos.health` |
| **2** | Debris generation in space | What was left in orbit, and how long will it stay? | `cosmos.debris` |
| **3** | Re-entry to the atmosphere | Where will each discarded part land, and who is at risk? | `cosmos.reentry` |

They are **one chain, not three tools**. The satellite's health decides *how* it breaks up,
which sets the fragment cloud that re-enters — so the same anomaly score produces a different
landing footprint depending on which subsystem failed. That coupling lives in
`cosmos/health/hazard.py` and is the claim the project rests on.

---

## Quick start

```bash
pip install -r requirements.txt

python tests/test_verify.py                 # 20 physics checks, ~2 min
jupyter notebook notebooks/COSMOS_Demo.ipynb # the whole project, end to end
```

Or from Python:

```python
import cosmos

result = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, azimuth_deg=320.0)
result.summary()          # telemetry -> health -> hazard -> break-up -> footprint -> risk
```

---

## Layout

```
Final_Project/
├── cosmos/                       the package — all the work is here
│   ├── common.py                 constants, WGS-84 geodesy, paths
│   ├── pipeline.py               the end-to-end chain, and compare_failure_modes()
│   ├── plots.py                  every figure the notebook draws
│   ├── health/                   ── FEATURE 1 ────────────────────────────────
│   │   ├── telemetry.py          8-subsystem telemetry simulator + fault injection
│   │   ├── monitor.py            anomaly detection (EWMA / IsolationForest), health scoring
│   │   └── hazard.py             THE COUPLING: failure mode -> break-up physics
│   ├── debris/                   ── FEATURE 2 ────────────────────────────────
│   │   ├── catalogue.py          CelesTrak SATCAT, break-up clouds, Kaplan-Meier
│   │   ├── scene.py              synthetic frame pairs + YOLO-format dataset export
│   │   ├── detect.py             registration, differencing, DoG detector, scoring
│   │   ├── yolo.py               TinyYOLO (PyTorch) + Ultralytics adapter, mAP, confusion
│   │   └── lifetime.py           orbital lifetime, censored calibration, collision risk
│   └── reentry/                  ── FEATURE 3 ────────────────────────────────
│       ├── atmosphere.py         US Standard Atmosphere 1976
│       ├── drag.py               Mach- and Knudsen-dependent C_D
│       ├── breakup.py            NASA Standard Breakup Model, with mass closure
│       ├── propagator.py         3-DOF ECEF propagator (adaptive + vectorised ensemble)
│       ├── montecarlo.py         dispersion, Gaussian ellipse vs CONFORMAL zones
│       └── risk.py               population exposure, expected casualties
│   └── ai/                       ── COSMOS-AI (learned models; imports, never edits, the above)
│       ├── data/                 download, validate, preprocess (telemetry/vision/catalogue)
│       ├── telemetry/            ESA-ADB official protocol, OPSSAT-AD, learned monitor
│       ├── vision/               SpotGEO + COSMOS synthetic detection
│       ├── survival/             leakage-free re-entry landmark
│       └── surrogate.py, uncertainty.py, evaluation.py, explain.py, fusion.py, registry.py
├── configs/                      frozen experiment protocols (checked against the code)
├── scripts/                      job queue, notebook builder
├── data/                         datasets, one folder per feature — see data/README.md
├── notebooks/COSMOS_Demo.ipynb   simple demo: every cell is one or two lines
├── notebooks/COSMOS_AI_Experiments.ipynb   the deep-learning layer on the real data
├── tests/test_verify.py          20 checks against independent answers
├── tests/test_ai.py              COSMOS-AI checks (official-code ports, metrics, leakage, models)
└── outputs/                      generated figures and artefacts; benchmarks/ and experiments/
```

---

## Datasets

Organised one folder per feature; see [`data/README.md`](data/README.md) for provenance.

| Feature | Dataset | Real? | Status |
|---|---|---|---|
| 1 Health | simulated telemetry, 8 subsystems | simulated | in use |
| 1 Health | ESA-ADB (3 missions), NASA SMAP/MSL | real | in use (official ESA-ADB protocol in COSMOS-AI) |
| 1 Health | OPSSAT-AD (2,123 labelled fragments) | real | in use (COSMOS-AI) |
| 2 Debris | **CelesTrak SATCAT — 70,288 objects, 35,834 debris** | **real** | **in use** |
| 2 Debris | CelesTrak GP element sets, 4 break-up clouds | real | in use |
| 2 Debris | **SpotGEO v2 — 6,400 real 5-frame sequences, 55,800 labelled GEO targets** | **real** | **in use (COSMOS-AI)** |
| 2 Debris | SPARK | by request | not available (restricted; see data/DATASET_REPORT.md) |
| 2 Debris | SpaceSense-Bench | synthetic, CC BY-NC | inspected, not used (resolved spacecraft) |
| 2 Debris | DUST | — | could not be located (audit A6.15) |
| 2 Debris | synthetic frame pairs + YOLO dataset | simulated | in use |
| 3 Re-entry | **measured orbital lifetimes (decay date − break-up date)** | **real** | **in use** |
| 3 Re-entry | CelesTrak landmark 2026-08-16 → 2026-10-23 (34,158 objects, 235 decays) | real | in use (COSMOS-AI) |
| 3 Re-entry | ESA recovered-debris list (235 fragments with lat/lon) | real | in use (linkage check only) |
| 3 Re-entry | ESA DISCOS / Space-Track history | real | not available (account / token required) |

The catalogue records *where* objects are and *when* they fell. It contains no re-entry
trajectories and no imagery, so those must be simulated. The README says so wherever it
matters, and the notebook marks every dataset **IN USE** or **PLANNED**.

---

## Datasets and Git LFS

The complete local working tree is about **172 GiB**. GitHub cannot hold that: Git LFS allows
**2 GB per file** on Free/Pro/Team plans and the free tier provides **1 GiB of LFS storage and
1 GiB of bandwidth per month**. This repository therefore carries the datasets that are both
**clear for redistribution** and **within that quota**, and documents exactly how to fetch the
rest. Nothing here is a sample or a truncation: every dataset listed as present is the
complete artefact, byte-identical to the checksum recorded in
[`data/DATASET_MANIFEST.csv`](data/DATASET_MANIFEST.csv).

### Carried in this repository (via Git LFS)

| Dataset | Size | Licence | Path |
|---|---|---|---|
| ESA Kelvins Collision Avoidance Challenge | 594 MiB | CC BY 4.0 | `data/feature2_debris/kelvins_cac/raw/` |
| NASA SMAP / MSL telemetry (telemanom) | 179 MiB | Caltech copyright, BSD-style redistribution licence | `data/feature1_health/smap_msl/raw/` |
| General Catalog of Artificial Space Objects (GCAT) | 12 MiB | CC BY 4.0 | `data/feature2_debris/gcat/raw/` |
| ESA Kelvins "Space Debris: the Origin" | 8.4 MiB | CC BY 4.0 | `data/feature2_debris/kelvins_debris_origin/raw/` |
| ESA OPSSAT-AD | 8.3 MiB | CC BY 4.0 | `data/feature1_health/opssat_ad/raw/` |
| CelesTrak SATCAT / GP / space weather snapshots | 17 MiB | CelesTrak usage policy (public, no credentials, no explicit licence) | `data/feature2_debris/celestrak/` |
| Re-entry cohort, NRLMSIS 2.1 density table, derived lifetimes | 7 MiB | derived from the above | `data/feature3_reentry/{processed,derived,historical}/` |

Together with the result arrays of the reported benchmarks, this is **0.664 GiB in LFS** and
**0.141 GiB in ordinary Git** — 1,159 files, 0.805 GiB in total.

One archive is carried as the archive only. The Kelvins CAC loader reads
`kelvins_cac/raw/extracted/`, which is byte-for-byte derivable from the committed
`.zip`; carrying both would spend the LFS quota twice for the same bytes. Recreate it with
`python scripts/extract_lfs_archives.py` (no network access, no manifest rewrite — it calls
the same `safe_extract` the downloader uses).

### Not carried here — fetch locally

These are excluded for size, for redistribution terms, or both. All are reproducible with
`python -m cosmos.ai.data.download` (see [Data](#data-download-verify-prepare)), which verifies
each download against the manifest checksum.

| Dataset | Local size | Why it is not here |
|---|---|---|
| SpotGEO v2 | 45 GiB | CC BY 4.0, but the single archive is 4.0 GiB — above the 2 GB Git LFS per-file limit — and the extracted set alone exceeds the free quota 45× |
| ESA Anomaly Dataset (ESA-ADB), missions 1–3 | 12.6 GiB | each archive is 3.5–3.9 GiB, above the 2 GB per-file limit; the Zenodo record states no licence (the DMLR paper indicates CC BY 3.0 IGO), so redistribution terms are unclear |
| SpaceSense-Bench | 1.0 GiB | CC BY-**NC** 4.0 on the Hugging Face card and no licence file in the source repository; inspected but not used in any reported result |
| SPARK | not held | access restricted, by request only |
| `outputs/benchmarks/telemetry_esa_adb/` | 66 GiB | out-of-fold score dumps for the ESA-ADB companion study; regenerable, and derived from the dataset above |
| `outputs/experiments/` | 37 GiB | per-arm score dumps and checkpoints; regenerable from the recorded configs, and the metrics they summarise **are** committed |

### Working with the LFS content

```bash
# once per machine
git lfs install

git clone https://github.com/AnuSajikumar6264/Cosmos-AI-
cd Cosmos-AI-
git lfs pull                               # downloads the actual dataset bytes
python scripts/extract_lfs_archives.py     # expands the one archive-only dataset
python -m cosmos.ai.data.validate          # re-verifies every checksum in the manifest
```

A plain `git clone` without `git lfs pull` leaves the LFS-tracked files as small pointer
files. If `data/.../raw/*.zip` looks like a ~130-byte text file, run `git lfs pull`.

Because this repository is public and the free tier allows 1 GiB of LFS bandwidth per month,
roughly one full `git lfs pull` per month fits the quota. Heavier use needs a GitHub data pack
or a mirror on object storage (see `docs/` for the DVC option).

---

## What is verified, and what is not

`python tests/test_verify.py` — **20 / 20 pass**. Each expected value comes from outside this
code:

| Check | Expected from | Result |
|---|---|---|
| US76 density, 8 layer boundaries | published US76 tables | < 0.01 % |
| US76 temperature | published tables | 0 K |
| Hydrostatic balance `dp/dz = −ρg` | closed form | < 1 % |
| WGS-84 geodetic round trip | identity | 2 × 10⁻¹⁰ m |
| `C_D` at Kn → ∞ | free-molecular limit 2.10 | exact |
| Terminal velocity | drag = gravity at impact | ratio 1.00 |
| Coriolis deflection, 30 km drop at 45° N | free-fall analytic *upper bound* | 81 m < 82 m |
| J2 altitude oscillation, 800 km orbit | ≈ 19 km peak-to-peak | 18.4 km |
| Step-size convergence, rtol 10⁻⁶ → 10⁻¹² | self-consistency | 0.13 m |
| Break-up mass closure | conservation of mass | 1.0000 |
| Sampled size law | truncated power-law survival function | < 5 % |
| Registration **sign** | known applied shift | 0.23 px |
| Conformal zone coverage | its own nominal target, out-of-sample | ≥ nominal |
| Gaussian ellipse on skewed data | must under-cover | 0.943 < 0.95 |

**Not verified:** the predicted landing *location* against a real recovered fragment. That
needs recorded re-entries with recovery coordinates (ESA DISCOS Re-entry Events Database) and
is the first task of the project plan. Nothing in this repository establishes it.

---

## Bugs worth knowing about

Found while building this, each one silent enough to survive casual testing:

- **Geopotential vs geometric altitude.** US76 layer boundaries are *geopotential*. Feeding a
  geometric altitude straight into a published table gives a 21 % density error at 85 km.
- **Registration sign.** The phase-correlation peak sits at −d. Using +d shifts the frames
  *further* apart. With sub-pixel jitter the error is invisible — alignment is merely twice as
  bad — and it only collapses past about a pixel of jitter. Guarded by
  `debris.check_shift_sign()`.
- **Shot noise is not stationary.** Pixels under a bright star are intrinsically noisier, so a
  global (or block-median) threshold flags star positions as high-significance detections.
  Weighting by a photon-noise variance map removes them: precision went 0.16 → 0.99.
- **Mass conservation is not automatic.** The NASA size and area-to-mass laws are sampled
  independently, so their product over-shot a 1800 kg parent by 15×.
- **Censoring.** A third of catalogued fragments have not re-entered yet. Fitting on the
  decayed ones only moves the fitted area-to-mass by the `bias_factor` the notebook prints.

---

## COSMOS-AI digital twin (Q1 rebuild, 2026-10-27)

**One framework for three functions:** satellite health monitoring, orbital debris analysis and
atmospheric re-entry risk prediction.

**Structure.** Each module is validated on its own real public data. The modules are joined by
explicit couplings, tested where a real test exists (no public dataset follows one object
through telemetry, conjunctions and re-entry).

**Documentation:**
- architecture and claims: `docs/ARCHITECTURE_AND_NOVELTY.md`;
- targets and splits: `docs/TARGET_DEFINITIONS.md`;
- data: `docs/DATASET_AUDIT.md` and `data/DATASET_MANIFEST.csv`;
- models: `docs/SOTA_MODEL_SELECTION.md`;
- review findings, fixes and deviations: `docs/REVIEW_AND_FIXES.md`;
- results: `docs/RESULTS_AND_ANALYSIS.md`;
- implementation and run order: `docs/FINAL_IMPLEMENTATION_REPORT.md`.

The manuscript is in `paper/journal_v3_twin/` (Word and LaTeX built from Markdown; every number
is a placeholder filled from the result files).

| Module | Code | Data | Models |
|---|---|---|---|
| Re-entry risk | `cosmos/ai/lifetime/` | GCAT + CelesTrak (64,516 objects, two temporal test periods) | physics expert (King-Hele and exact orbit averages, NRLMSIS 2.1, Monte-Carlo area-to-mass) fused by a reliability gate with a learned expert. The expert is chosen among TabM (2025), TabICLv2 (2026) and Kumo-Tabular (2026). XGBoost, LightGBM, CatBoost, a quantile MLP and XGBoost-AFT are baselines. Mondrian conformal intervals |
| Debris: collision risk | `cosmos/ai/conjunction/` | ESA Kelvins CAC | persistence expert + the same learners, gated; official metric L |
| Debris: breakup clouds (C2) | `cosmos/ai/twin/couplings.py` | lifetime test folds | per-fragment survival summed per cloud, raw and calibrated |
| Health | `cosmos/ai/telemetry/` | ESA-ADB (companion pre-registered study), NASA SMAP/MSL | conformal alarm layer; on SMAP/MSL, Chronos-2 (2025) zero-shot plus classic baselines and the published telemanom scores |
| Twin engine | `cosmos/ai/twin/engine.py` | CelesTrak snapshot of the as-of date | per-object state; history-conditioned physics for objects in orbit; operational status selects the regime (C3); dated, hashed forecast |
| As-of-date tests | `cosmos/ai/twin/couplings.py` (C3), `cosmos/ai/twin/prospective.py` (T1) | snapshots 2026-08-16 → 2026-10-27 (a window analysed earlier in the project, so not blind) | payload regimes (C3); twin re-entry forecasts for every object class (T1) |

Run order: see `docs/FINAL_IMPLEMENTATION_REPORT.md`, section 3.

**Honest-reporting rules applied:**
- Test sets are read only for scoring. They were scored more than once during development,
  and every reuse, the learner-pool change and every post-hoc analysis are logged in
  `docs/REVIEW_AND_FIXES.md`.
- Results that do not favour the proposed components are reported as they are:
  - At the 2016 origin, two 2026 foundation models are significantly better than the
    reliability-gated fusion and the physics expert.
  - At the 2011 origin, the physics expert, the foundation models and the gated fusion are
    statistically tied (physics has the lowest point estimate).
  - Selection on the tuning cohort chose a significantly worse learner at 2016.
  - No model beats the latest-risk forecast on the Kelvins metric.
- The radar cross-section is excluded because its availability encodes the outcome.

---

## COSMOS-AI — the deep-learning layer

`cosmos/ai/` adds learned models on top of the classical code, which it imports but never
edits. The methodology is `research_design/COSMOS_AI_Research_Design.md` (Candidate 4,
calibrated deep forecasting on the ESA Anomaly Detection Benchmark). Vision and re-entry are
supporting studies. **Every number in this section comes from a file in `outputs/benchmarks/`,
and the full component table is `outputs/benchmarks/SUMMARY_TABLE.md`.**

### Environment

```bash
# Python 3.12 with a PyTorch build for your hardware (here: torch 2.13.0+xpu, Intel Arc Pro B60)
python -m venv --system-site-packages .venv      # on top of an env that already has torch
.venv/bin/pip install -r requirements.txt -r requirements-ai.txt
.venv/bin/python -m ipykernel install --prefix .venv --name cosmos-ai   # notebook kernel
export HF_HOME=$PWD/../.cache/huggingface TORCH_HOME=$PWD/../.cache/torch   # keep caches off /
```

Hardware used: 512 CPU cores, 1 TB RAM, and 7 of 8 Intel Arc Pro B60 GPUs (24 GB each; one
was in use by another user). Minimum practical setup: one 16 GB GPU and 64 GB RAM (the
Mission 1 grids are 7.4 M × 76 per half). Everything except the deep training runs on CPU.
Disk: about 60 GB for data, plus about 2 GB per ESA arm for the saved score arrays.

### Data (download, verify, prepare)

```bash
python -m cosmos.ai.data.download opssat_ad esa_adb spotgeo celestrak   # MD5-verified, resumable
python -m cosmos.ai.data.preprocess_telemetry 1 2    # ESA-ADB official 30 s / 18 s grids
python -m cosmos.ai.vision.spotgeo preprocess        # registered-difference front end
python -m cosmos.ai.vision.cosmos_synthetic generate # 800/200/200 scene-split frame pairs
python -m cosmos.ai.telemetry.learned_monitor generate
python -m cosmos.ai.surrogate generate               # 500k propagator trajectories (SIM)
python -m cosmos.ai.data.validate                    # data/DATASET_REPORT.md (+ .json)
```

See `data/README.md` and `data/DATASET_REPORT.md` for provenance, licences and validation.
Not obtainable here: SPARK (by request; Zenodo record restricted), Space-Track and DISCOSweb
(accounts). SpaceSense-Bench was inspected and judged off-task (resolved spacecraft, CC BY-NC).

### Training and evaluation

```bash
# ESA-ADB, official protocol (one arm = mission x encoder x seed; ~20-60 min per deep arm on 1 GPU)
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --baseline global_std5      # reproduction
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --encoder ar8               # COSMOS AR(8)
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --encoder patchtst --seed 0
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --encoder tcn --telecommands --seed 0
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --encoder chronos-bolt-small
python scripts/job_queue.py outputs/logs/esa_remaining.txt --gpus 0,1,2   # a whole matrix
# OPSSAT-AD (5 seeds, official + time-blocked splits, label efficiency) and robustness
python -m cosmos.ai.telemetry.opssat --config configs/telemetry/opssat.yaml
python -m cosmos.ai.telemetry.robustness
# vision
python -m cosmos.ai.vision.spotgeo classical
python -m cosmos.ai.vision.train_spotgeo --model unet --inputs both --seed 0
python -m cosmos.ai.vision.cosmos_synthetic heatmap --seed 0     # also: tinyyolo, yolov8n
# re-entry
python -m cosmos.ai.survival.landmark
python -m cosmos.ai.surrogate train && python -m cosmos.ai.surrogate evaluate
# learned health monitor (synthetic) and the pipeline back-ends
python -m cosmos.ai.telemetry.learned_monitor train && python -m cosmos.ai.telemetry.learned_monitor heldout
# aggregate everything: tables, bootstrap CIs, figures, SUMMARY_TABLE.md
python -m cosmos.ai.report
```

**Pre-registered telemetry study** (`research_design/preregistration/PREREG_TELEMETRY_V1.md`,
frozen in `freeze.json`). Alarm selection and the supervised event scorers use out-of-fold
scores over the whole training half. The test half can only be scored while every frozen
file is unchanged.

```bash
python -m cosmos.ai.telemetry.oof --mission 1                       # OOF AR(8) scores (A1)
python -m cosmos.ai.telemetry.event_model --mission 1 --seed 0      # TCN event scorer (A3), ~2 h/GPU
python -m cosmos.ai.telemetry.event_gbt --mission 1 --seed 0        # XGBoost event scorer (A2)
python scripts/prereg_analysis.py select   # alarm-layer selection, training labels only
python scripts/prereg_analysis.py test     # one test pass per arm (refuses if the freeze is broken)
python scripts/prereg_analysis.py stats    # H4-H6, Holm, Mission 2 replication -> CONFIRMATORY.md
python scripts/build_paper_tables.py       # paper/tables/*.tex from the result files
```

The pipeline gains two opt-in back-ends. The defaults are unchanged, and the original
behaviour is regression-tested against the untouched baseline code:

```python
cosmos.run_pipeline(faults={"Propulsion System": "leak"}, backend="learned",
                    footprint_backend="surrogate")
```

### Outputs

| Path | Content |
|---|---|
| `outputs/experiments/<run>/` | `config.yaml`, `env.json`, `epochs.csv`, `metrics.json`, `checkpoints/`, saved scores; index in `index.jsonl` |
| `outputs/benchmarks/telemetry_esa_adb/` | `results.csv`, `summary_table.csv`, `bootstrap_comparisons.csv`, per-arm JSON (`arms/`), robustness JSON |
| `outputs/benchmarks/telemetry_opssat/` | summaries, per-run CSV, paired comparisons, robustness, test scores |
| `outputs/benchmarks/vision_spotgeo/`, `vision_cosmos_synthetic/` | per-run JSON, summaries, official test submission files |
| `outputs/benchmarks/reentry_landmark/`, `reentry_surrogate/` | results JSON/CSV, test predictions, fallback-mode timing |
| `outputs/benchmarks/figures/` | comparison figures |
| `outputs/registry/model_cards.json` | one card per model: inputs, outputs, version, validation status, limitations |
| `outputs/baseline/` | pre-change snapshot, test logs, baseline report, environment freeze |

### Headline results (test sets scored once; full table in `outputs/benchmarks/SUMMARY_TABLE.md`)

**Pre-registered study (freeze 09a98f5e160e; `oof_protocol/CONFIRMATORY.md`).** Settings:
ESA-ADB, full channel sets, test half, conformal-BH layer, seed means. Mission 1 is the
pre-registered family; Mission 2 is a replication.

| Mission | A0: AR(8), 9-month selection | A1: AR(8), whole-training-half selection | A2: XGBoost event scorer (5 seeds) | A3: TCN event scorer (5 seeds) | out-of-limit rule |
|---|---|---|---|---|---|
| 1 | 0.074 | **0.428** (1.1 false alarms/yr) | 0.544 ± 0.236 | 0.205 ± 0.114 | 0.0007 |
| 2 | 0.962 | 0.874 | 0.904 ± 0.050 | 0.875 ± 0.058 | 0.894 |

- **Supported:** H6 (selection, A1 > A0) on Mission 1: +0.353 [+0.206, +0.488].
- **Refuted:** H4 (A3 > A1) and H5 (A3 > A2). The deep event scorer does not help.
- **Mission 2:** the selection effect reverses (−0.089).

The exploratory results below predate the pre-registration:

**ESA-ADB, official protocol, full channel sets (58 / 47 target channels), test half scored once.**
Calibrator settings were selected on the last 9 (M1) or 4.5 (M2) months of the training half.
"n" is the number of seeds.

| Mission | Arm | EW F0.5, anomalies + rare events (DMLR primary) | Anomalies only: EW F0.5 / event recall | False alarms / yr |
|---|---|---|---|---|
| 1 | existing COSMOS monitor (as shipped) | 0.004 | 0.002 / 0.76 | 3,278 |
| 1 | official GlobalSTD5 (reproduced) | 0.003 | 0.002 / 0.84 | 3,745 |
| 1 | COSMOS AR(8) + per-channel threshold | **0.088** (n=1) | 0.044 / 0.31 | 64 |
| 1 | PatchTST + conformal-BH | 0.048 ± 0.024 (n=3) | 0.016 / 0.15 | 130 |
| 1 | TCN + telecommands + rolling conformal-BH | 0.318 ± 0.215 (n=3; seed-unstable, recall 0.10) | 0.258 / 0.08 | 0.6 |
| 1 | published best (DMLR Tables 4, 18) | 0.061 (Telemanom-ESA-Pruned) | 0.039 / 0.91 | — |
| 2 | existing COSMOS monitor (as shipped) | 0.047 | 0.001 / 0.50 | 3,576 |
| 2 | official GlobalSTD5 (reproduced) | 0.063 | 0.001 / 0.50 | 3,770 |
| 2 | **COSMOS AR(8) + conformal-BH** | **0.962** (n=1) | 0.147 / 0.17 | 3.4 |
| 2 | PatchTST + conformal-BH | 0.952 ± 0.009 (n=3) | 0.097 / 0.17 | 6.9 |
| 2 | Chronos-Bolt-small zero-shot + conformal-BH | 0.945 (n=1) | 0.093 / 0.17 | 6.3 |
| 2 | published best (DMLR Tables 4, 19) | 0.079 (PCC) | 0.001 / up to 1.0 | — |

Read the Mission 2 rows with care. 356 of its 362 test events are rare nominal events, so the
primary score mostly measures how well those are caught with few false alarms. Every
COSMOS-AI arm finds only **1 of the 6 anomalies**, whereas the published methods find up to 6/6 at
the cost of thousands of false alarms; that is a different operating point, not a uniform win.
Paired block-bootstrap CIs (seed 0, 30-day blocks): deep or pretrained encoders vs AR(8) at
equal calibration — M2 PatchTST −0.021 [−0.051, +0.007], TCN −0.027 [−0.054, +0.002],
Chronos −0.018 [−0.047, +0.014]; conformal-BH vs threshold on M2 — AR(8) +0.080 [+0.041, +0.133],
PatchTST +0.124 [+0.080, +0.177]; on M1 the calibrator effect is ≤ 0.012 in either direction.
Lightweight subsets: M1 PatchTST + threshold 0.496 (published best 0.786); M2 persistence +
conformal-BH 0.964 (published best 0.949, Windowed iForest).

The reproduction of the official GlobalSTD3/5 matches DMLR Tables 4, 16 and 17 on the primary
metric in all 8 configurations (M1 lightweight to every printed digit).


| Study | Best model (selection on validation only) | Test result | Baseline on the same protocol |
|---|---|---|---|
| OPSSAT-AD (official split, 5 seeds) | hybrid raw-CNN + feature MLP | F1 0.949 ± 0.009, AUC-PR 0.986 ± 0.001 | XGBoost 0.911 / 0.961; published FCNN 0.946 / 0.979 |
| SpotGEO v2 (real, 5,120 sequences) | heat-map U-Net (5 frames) | official F1 0.746 ± 0.005 | COSMOS front end + DoG 0.084; challenge winner 0.948 |
| COSMOS frame pairs (SYNTHETIC) | heat-map U-Net | mAP@50-95 0.371 ± 0.022 | YOLOv8n 0.244 ± 0.046; TinyYOLO 0.162 ± 0.004 |
| Re-entry within 68 d (real landmark) | **XGBoost** (the deep model did not earn its place) | AUC-PR 0.821 | hazard-MLP ensemble 0.688 (Δ −0.134, CI [−0.221, −0.024]); COSMOS physics 0.188 |
| Propagator surrogate (SIM) | 5-MLP ensemble | median impact error 16 km (p95 69 km); about 360× faster per sample, 70-200× per footprint without fallback | reference propagator (ground truth for SIM) |
| Learned health monitor (SYNTHETIC, held-out seeds) | PatchTST + conformal BH | F1 0.775, localisation 0.960 | classical EWMA F1 0.834, localisation 0.830 |

### What the results support, and what they do not

- **Calibration does most of the work on ESA-ADB, not the encoder.** Applying the same
  calibrated alarm layer (per-channel conformal p-values, Benjamini-Hochberg across channels,
  persistence, 1 h merging) to every encoder lifts the existing COSMOS AR(8) forecaster far
  above the published full-channel primary scores on Mission 2. It does so by trading
  anomaly recall (1 of 6) for very few false alarms, as the table above explains. Pretrained
  and deep encoders (PatchTST, TCN, Chronos-Bolt) did not beat AR(8) at equal calibration (H1
  not supported; see the bootstrap CIs). Mission 1 stays hard for every method.
- **Telecommand gating (H3) did not suppress rare-nominal-event alarms.** A fixed rule (mask 30
  min after any priority-3 telecommand) cuts Mission 2 rare-event-only alarm runs by 88 % with no
  loss of anomaly recall. So the information is in the telecommand stream, but the learned gate,
  trained only on the forecasting loss, did not use it. Removing telecommands at test time does
  not change the gated models either.
- **Conformal guarantees are approximate on telemetry.** The realised false-alarm rates are
  measured (`false_alarms_per_year`) rather than assumed. Rolling recalibration helps on
  Mission 1 (drift) and hurts on Mission 2 (recurring rare events contaminate the pool).
- **Re-entry:** gradient-boosted trees beat the neural hazard model on this ~34k-row tabular
  landmark, as expected for tabular data. Label-conditional (Mondrian) conformal sets reach their
  coverage on the rare events, while marginal sets ignore them. **No impact-location accuracy is
  claimed.** The ESA recovered-debris list was used only to check record linkage (195 of 197
  linked fragments lie inside their orbit's latitude band).
- **Synthetic results are labelled SYNTHETIC/SIM everywhere** and say nothing about real
  spacecraft.

### Reproducibility

- Seeds are fixed and recorded per run; the configs in `configs/` are tested to equal the code
  defaults.
- Every test set was scored once per frozen configuration. Decisions taken *after* seeing a test
  number are listed in `outputs/benchmarks/SUMMARY.md` (for example, the MLP tuning grid in the
  re-entry study was added after a first, untuned run had been scored).
- XPU training is not bitwise deterministic. Re-running OPSSAT-AD with the same seeds moved
  individual deep-model F1 values by up to 0.025, within the seed-to-seed spread.
- The notebook `notebooks/COSMOS_AI_Experiments.ipynb` was executed end to end on CPU with
  `jupyter nbconvert --execute` in this environment (it loads the saved checkpoints; heavy
  training is not repeated in it).

### Known limitations

- ESA-ADB has one benchmark with two usable missions. Mission 1's calibration period holds only
  9 labelled events, and its test half has 91; Mission 2's test events are 98 % rare nominal
  events (6 anomalies).
- Deep arms have 1-3 seeds, not the 5 the design asked for, because of compute time (each arm
  needs ~2 GB of score arrays and 20-60 GPU-minutes).
- Irregular sampling is resampled (benchmark requirement R8 not met).
- The secondary ESA metrics (channel- and subsystem-aware, affiliation) differ from DMLR by up to
  0.044 in the reproduction, although the primary metric matches. The cause is not identified.
  The 2025-04-17 dataset revision was checked and changes only `channels.csv`, so it is not the cause.
- SpotGEO is GEO-only, ground-based and from one sensor; the learned detector is far below the
  challenge winner.
- The re-entry landmark is a single 68-day window with 38 test events. Space-Track history
  would allow many landmarks and is the next data step.
- Two pre-existing defects are left untouched and documented in `outputs/baseline/BASELINE_REPORT.md`:
  the pandas-3 safe-unpickler test, and the `stuck` fault that crashes the simulator.

---

## Credit

- The AI detection approach — synthetic renders with domain randomisation, YOLO-format
  dataset, nano-scale detector, mAP + confusion matrix, confidence 0.60 — follows
  [jaikr-dev/space-debris-detection-pipeline](https://github.com/jaikr-dev/space-debris-detection-pipeline).
  Its published mAP50-95 of 0.976 (YOLOv8n/v11n) is **that project's result on its own Blender
  dataset**, not ours; `debris.reference_benchmark()` reproduces it for comparison only.
- Re-entry risk is benchmarked conceptually against the
  [ESA Space Debris User Portal](https://sdup.esoc.esa.int) — DRAMA (re-entry survival and
  ground swath), ORIUNDO (on-ground risk from Gridded Population of the World), MASTER
  (debris flux), DISCOS (40,000+ objects and a Re-entry Events Database). Registration is
  required, and DISCOS additionally needs an ESA Member State affiliation.
- Catalogue data: [CelesTrak](https://celestrak.org) (T.S. Kelso).
- Physics: US Standard Atmosphere 1976; Johnson et al. (2001) NASA Standard Breakup Model;
  Dormand & Prince (1980); Sutton & Graves (1971); Vovk et al. (2005) and Lei et al. (2018)
  for conformal prediction.

## Requirements

`numpy`, `pandas`, `matplotlib` are required. `torch` is needed only for the learned detector
and `scikit-learn` only for the IsolationForest option — both degrade gracefully. `ultralytics`
is optional and enables the real YOLOv8n/v11n/v12n backend. COSMOS-AI needs
`requirements-ai.txt` in addition (see the COSMOS-AI section above).
