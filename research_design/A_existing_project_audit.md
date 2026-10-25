# A. Existing project audit (COSMOS → COSMOS-AI)

Audited 2026-10-24. Sources inspected: `cosmos/` package and `tests/` in the working copy
`/data/anupa/COSMOS_work/Final_Project` (moved from /home/anupa/COSMOS_work on 2026-10-24; most complete version: includes the real-data layer),
`notebooks/COSMOS_Demo.ipynb`, the IEEE-format manuscript `paper/main.tex` (in
`.../Satellite-Digital-Twin-for-Anomaly-Detection-and-Heath-Monitoring-System/paper/`), the
re-run evidence in `/home/anupa/COSMOS_Review2/deliverables/evidence/`, and the data folders.
Every number below was read from those files or measured by running code during the audit.
Evidence classes used throughout: **REAL** (externally recorded data), **DERIVED** (computed
from real data), **SYNTHETIC** (generated images/telemetry), **SIM** (numerical-simulator
output), **HAND** (hand-specified coefficients), **VERIFIED-NUM** (numerical check against an
independent answer — not empirical validation).

## A1. Module inventory

| Module | Role | Method class | Notable deps |
|---|---|---|---|
| `common.py` | constants, WGS-84 geodesy, paths, `Table` | numerical | numpy |
| `datasets.py` | registry: source, location, checksum, fetch/prepare | infra | `reentry` |
| `pipeline.py` | `run_pipeline`, `compare_failure_modes`, `corridor_risk` | orchestration | health, reentry |
| `plots.py` | every notebook figure | infra | matplotlib |
| `health/telemetry.py` | 8-subsystem / 16-channel simulator, fault injection | SYNTHETIC generator | numpy |
| `health/monitor.py` | robust z (median/MAD), EWMA, optional IsolationForest; subsystem degradation | conventional statistics / ML | sklearn (optional) |
| `health/realdata.py` | SMAP/MSL + ESA-ADB loaders (safe unpickler), rolling baseline, AR(8) ridge residuals, nominal calibration, event metrics, ablation | conventional statistics | numpy, pandas |
| `health/hazard.py` | health state → break-up type, altitude, p_frag, σ_logB, σ_x | HAND coefficients | monitor |
| `debris/catalogue.py` | SATCAT load, break-up clouds, Kaplan–Meier, derived exports | statistics on REAL | pandas |
| `debris/scene.py` | domain-randomised frame pairs, YOLO export | SYNTHETIC generator | numpy |
| `debris/detect.py` | phase-correlation registration, differencing, DoG + photon-noise map, scoring | classical CV | numpy |
| `debris/yolo.py` | **TinyYOLO** (PyTorch), Ultralytics adapter, AP, confusion matrix | **deep learning** | torch, ultralytics (optional) |
| `debris/lifetime.py` | drag-energy lifetime, censored 1-parameter A/M fit, particle-in-a-box collision rate | numerical + 1-parameter fit | reentry.atmosphere |
| `debris/realimagery.py` | SPARK / DUST label converters to YOLO | infra (no data present) | — |
| `reentry/atmosphere.py` | US Standard Atmosphere 1976 | numerical | — |
| `reentry/drag.py` | Mach/Knudsen-dependent C_D | numerical | — |
| `reentry/breakup.py` | NASA Standard Breakup Model with mass closure | stochastic model | — |
| `reentry/propagator.py` | 3-DOF ECEF propagator, adaptive + vectorised ensemble, wind field | numerical | — |
| `reentry/montecarlo.py` | dispersion (incl. per-sample flight-path angle and wind), Gaussian vs split-conformal zones | SIM + conformal | — |
| `reentry/population.py` | GHS-POP 2020 raster + land mask, 0.25° grid | REAL data prep | — |
| `reentry/risk.py` | casualty area, survival, expected casualties | numerical | population |
| `reentry/discos.py` | ESA DISCOS JSON:API client, fragmentation-cause mapping | infra (no data present) | hazard |
| `tests/test_verify.py` | 20 physics/numerics checks | VERIFIED-NUM | — |
| `tests/test_datasets.py` | event-metric, scoring, safe-unpickler, converter, DISCOS-parser, real-dataset checks | unit tests | — |

## A2. Deep learning vs conventional methods currently implemented

**Deep learning already in the code (one model):**
- `TinyYOLO` (`debris/yolo.py`): 8 conv–BN–SiLU blocks (1→16→32→32→64→64→96→96→128 channels),
  three stride-2 stages (output stride 8), single-scale anchor-free head predicting
  objectness, (dx, dy), log(w, h) and 4 class logits per cell. ≈0.32 M parameters (counted from
  the layer definition). Input: 1-channel **registered difference image** at 320×320
  (`TRAIN_SIZE`; dataset is exported at 640×640). Loss: weighted BCE objectness + 5×L1 box +
  cross-entropy class. Adam, lr 3e-3, batch 8, 40–50 epochs, CPU.
- `UltralyticsYOLO` adapter: optional wrapper for YOLOv8n/v11n/v12n. **No result produced by
  it exists in the evidence**; the YOLO numbers in the notebook/paper are the *reference
  repository's* published numbers, not COSMOS runs.

**Everything else is conventional:** robust z-score, EWMA, IsolationForest (200 trees,
contamination 0.02), 30-day rolling baseline, AR(8) ridge forecaster, DoG blob detector,
phase correlation, Kaplan–Meier, grid search for one area-to-mass parameter, US76, NASA SBM,
3-DOF integration, Gaussian ellipse and split-conformal regions.

## A3. Dataset inventory

| Dataset | Class | Location / size (measured) | Labels | Status |
|---|---|---|---|---|
| ESA Anomaly Dataset (ESA-ADB), Zenodo 10.5281/zenodo.12528696 (creators De Canio, Kotowski, Haskamp; 2024-06-25; `access_right: open`, licence field absent from the local record — must be confirmed) | REAL | 3 zips, ≈11.6 GB raw; prepared `.npz`: M1 1,472,832 × 76 ch @5 min; M2 919,440 × 100 ch @2 min; M3 932,243 × 48 ch @5 min | per-channel event rows: 3,589 / 11,167 / 2,586, with Class, Subclass, Category (Anomaly, Rare Event, Communication Gap, Invalid Segment), Dimensionality, Locality, Length; channel→subsystem map | in use |
| NASA SMAP/MSL (Hundman et al. 2018) | REAL | 82 channels (55 SMAP, 27 MSL), 166 MB; e.g. `A-1` test = 8,640 × 25 | labelled anomaly windows per channel | in use |
| COSMOS simulated telemetry | SYNTHETIC | generated on demand: 8 subsystems, 16 channels, 30 s cadence | injected fault type + onset | in use |
| CelesTrak SATCAT | REAL | 70,288 objects × 17 fields; launches 1957-10-04 → 2026-08-16; 35,470 with DECAY_DATE | DECAY_DATE (event time), OBJECT_TYPE | in use |
| CelesTrak GP element sets (4 clouds + active) | REAL | current epoch only, **in-orbit objects only** — no element-set history | — | in use |
| `measured_lifetimes.csv` | DERIVED (from REAL) | 7,711 fragments, 4 events; 4,652 observed, 3,059 right-censored | lifetime_years + observed flag | in use |
| `spatial_density.csv` | DERIVED | 14 shells | — | in use |
| Synthetic frame pairs + YOLO dataset | SYNTHETIC | 48 images (36 train / 12 val), 640×640, 381 labels, 4 classes | boxes + classes | in use |
| SPARK, DUST | REAL (planned) | folders **empty**; converter code only | — | not downloaded |
| ESA DISCOS | REAL (planned) | folder **empty**; client needs an access token | — | not available |
| GHS-POP E2020 R2023A 30ss (CC BY 4.0) + Natural Earth land | REAL | 482 MB raster + derived 0.25° grid | — | in use |
| Monte Carlo footprints, fragment clouds | SIM | generated per run | — | in use |
| Health→break-up coupling coefficients | HAND | `health/hazard.py` | — | in use |

## A4. Training / validation / testing procedures as implemented

| Experiment | Fit / train | Selection / validation | Test | Problem |
|---|---|---|---|---|
| Simulated health | first 40 % of each record (notebook: 1,152 of 2,880 samples) | none | rest of the same record; 3 seed sets, 288 records each (TP+FN = 240 faulted, FP+TN = 48 fault-free) | same generator for design and test |
| SMAP/MSL | each channel's nominal train split | **none — variants were compared on the test split** | labelled test split, event-wise | variant choice is test-set selection |
| ESA-ADB | first 20 % of each mission (COSMOS's own resampling: median per 5/2-min bin) | none | remaining 80 %, COSMOS event metric | **not the official ESA-ADB split, preprocessing or metrics → not comparable with published ESA-ADB results** |
| TinyYOLO | 36 synthetic images | none | 12 synthetic images (95 objects), single seed | validation set = test set; tiny |
| Lifetime A/M | all 7,711 fragments | none | the same 4 clouds | in-sample |
| Conformal zones | half of each simulated impact set | — | other half, 12 random splits | SIM only; exchangeable by construction |

## A5. Existing results and their conditions

| Result | Value | Condition | Class |
|---|---|---|---|
| Simulated health F1 | 0.824–0.829 (P 1.0, R ≈0.70); median delay ≈315 samples | 3 seed sets, 288 records | SYNTHETIC |
| — per fault | recall: degradation 0.104, leak 0.792, dropout/overheat/saturation 0.875; Thermal Control 0.000 | same | SYNTHETIC |
| ESA-ADB, "simulator monitor" | F1 1.0 / 1.0 / 0.056 | **alarm duty 1.0 on M1 and M2 — the monitor alarms all the time, so F1 = 1.0 is degenerate**; time-precision 0.103 / 0.056 | REAL |
| ESA-ADB, AR-residual variant | F1 0.344 / 0.069 / 0.528; false alarms/yr 0 / 4.29 / 0.28 | COSMOS protocol | REAL |
| ESA-ADB subsystem attribution (AR variant) | 0.889 / 0.500 / 1.000 | **"always blame the largest subsystem" baseline scores 1.000 / 0.944 / 1.000 — attribution does not beat the majority baseline on any mission** | REAL |
| SMAP/MSL event F1 | 0.472 (base) → 0.572 (AR residuals) | event-wise, no point-adjust | REAL |
| Single-frame vs registered-difference detection | P 0.110 → 0.987, R 0.963 → 0.563 | 16 synthetic scenes, 135 objects | SYNTHETIC |
| TinyYOLO | P 0.884, R 0.958, F1 0.919, "mAP@50" 0.947, mean IoU 0.611 | **default `match="centre"`: AP is class-agnostic, matches within 6 px of centre, and uses only detections above conf 0.60 — it is not COCO-style IoU≥0.5 mAP.** 12 val images, 1 seed | SYNTHETIC |
| Lifetime model vs measured medians | ratio 0.297 (Cosmos 1408), 1.017, 0.878 | in-sample, 1 parameter | DERIVED |
| Censoring bias on A/M | 1.049× | same | DERIVED |
| Conformal vs Gaussian zones | coverage held; 61–65 % smaller area | MC impacts | SIM |
| Ground risk, GHS-POP vs coarse grid | E_c 2.1e-5 → 4.2e-4 (India NW corridor) | per-fragment 1 m² casualty area in headline | SIM × REAL |
| Physics checks | 20 / 20 pass | `tests/test_verify.py` | VERIFIED-NUM |

## A6. Missing functionality and limitations

1. **No deep temporal model** for telemetry; nothing learned exploits cross-channel structure.
2. **No official-protocol evaluation** on ESA-ADB; COSMOS numbers cannot be placed next to
   published baselines.
3. **No validation split anywhere**; every model/variant choice was made on test data.
4. **Attribution is not better than a trivial baseline** on real data (A5).
5. **Vision has no real data at all**; SPARK/DUST folders are empty and, as public datasets of
   *resolved* spacecraft, may not match the unresolved-point-source problem (see F).
6. **Detector metric mislabelled** as mAP@50; class discrimination is poor.
7. **No element-set history**: GP files are current-epoch, in-orbit only, so no time-to-decay
   model *as of a prediction date* can be trained from what is on disk.
8. **Target leakage hazard in the catalogue:** for decayed objects SATCAT's PERIGEE/APOGEE are
   last-known values (median perigee 199 km for decayed vs 623 km for in-orbit objects, measured
   during this audit). `measured_lifetimes.csv` carries these columns; using them as features
   for lifetime prediction would leak the label.
9. **Detection → ballistic coefficient** link is claimed in the manuscript but not implemented.
10. Coupling coefficients are HAND; no learned or data-fitted link between health and break-up.
11. Single seeds, no confidence intervals, no significance tests (except the health re-runs).
12. Paper code and real-data code live in two different copies.
13. **Telecommands are on disk but unused.** The ESA-ADB archives contain `telecommands.csv` plus
    698 (Mission 1) and 123 (Mission 2) telecommand files — 821 in total, matching the benchmark
    paper — but `prepare_esa_mission` reads channels only.
14. **Mission 3 is reported, but the benchmark authors exclude it** from algorithm comparison
    (Kotowski et al., DMLR 2026). COSMOS should stop reporting comparative numbers on it.
15. **"DUST" could not be located** by the literature search (arXiv, OpenAlex, Crossref, GitHub,
    Kaggle); the README lists it as a planned dataset and should be corrected.

## A7. Retain / replace / extend / remove

| Component | Decision | Reason |
|---|---|---|
| US76, drag, propagator, break-up, Monte Carlo, verification suite | **Retain** as reference implementation and simulation-data generator | verified; not the DL contribution |
| Split-conformal machinery (`montecarlo.py`) | **Retain and generalise** into a shared `ai/uncertainty.py` | reused by every learned model |
| Robust-z/EWMA/AR(8) monitor | **Retain as baseline** | the honest prior; must be beaten |
| Event metrics (`realdata.event_metrics`) | **Retain** + add official ESA-ADB metrics | needed for both comparability and continuity |
| COSMOS ESA-ADB preprocessing | **Extend**: add the official ESA-ADB split/preprocessing path | comparability |
| Registration + differencing front end | **Retain** as input stage and as baseline | it solves the star/fragment ambiguity; learned pair models must beat it, not bypass it |
| TinyYOLO | **Retain as baseline**; fix the metric label; add a COCO-style evaluator | — |
| Lifetime 1-parameter fit | **Retain as physics baseline** | — |
| Hand-specified coupling | **Keep but demote** to a documented scenario layer; not a contribution | cannot be learned from available data |
| `measured_lifetimes.csv` orbit columns | **Remove from any feature set** | label leakage |
| "mAP@50" label for centre-AP | **Remove / rename** | misleading |

## A8. Minimal-change migration plan

1. Merge the paper copy and `COSMOS_work` into one repository; tag the current state as the
   *baseline release* so every later comparison runs against frozen baseline code.
2. Add a new sub-package `cosmos/ai/` that imports from, but never modifies, `health/`,
   `debris/`, `reentry/`. Learned models expose the same return types the pipeline already
   consumes (`MonitorResult`, `Detections`, lifetime arrays), so `pipeline.py` gains a
   `backend="classical" | "learned"` switch instead of being rewritten.
3. Add dataset loaders that implement **official splits** first, then models.
4. Every new model must ship with (a) a frozen config, (b) a validation split used for all
   selection, (c) a held-out test touched once, (d) ≥5 seeds, (e) a unit test on a toy input.
5. The existing verification suite and dataset tests must keep passing at every phase.
