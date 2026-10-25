# COSMOS-AI — implementation summary (2026-10-24)

Every number below was read from a file in `outputs/benchmarks/`, `outputs/experiments/` or
`data/DATASET_REPORT.json`. Each test set was scored once per frozen configuration. Synthetic
and simulated results are labelled as such and say nothing about real spacecraft.

## 1. Existing functionality preserved

- **Existing source changed in one file only.** `cosmos/pipeline.py` gained two opt-in
  keyword arguments: `backend="classical"|"learned"` and
  `footprint_backend="reference"|"surrogate"`. The defaults are unchanged.
- **Original behaviour regression-tested.** `tests/test_ai.py::test_pipeline_backward_compatible`
  extracts the untouched original package from `outputs/baseline/` and runs it in a subprocess.
  The default `run_pipeline()` summary is identical row for row.
- **Documentation extended, not rewritten.** `README.md` and `data/README.md` were extended; their
  original text is kept, except that out-of-date "planned" dataset rows were corrected.
- **Tests before and after** (`outputs/baseline/*.log`, `outputs/tests/*.log`):

| Suite | Before | After |
|---|---|---|
| `tests/test_verify.py` | 20 / 20 | 20 / 20 |
| `tests/test_datasets.py` (per test) | 20 pass, 1 error (pre-existing, pandas 3) | 20 pass, the same 1 error |
| `tests/test_ai.py` (new) | — | 38 / 38 (also under pytest) |

## 2. Datasets actually downloaded (manifests: `data/**/MANIFEST.json`; report: `data/DATASET_REPORT.md`)

| Dataset | Source | Licence | Size | Verification |
|---|---|---|---|---|
| OPSSAT-AD | Zenodo 15108715 (the record behind 12588358) + GitHub kplabs-pl/OPS-SAT-AD @dafd0d3 | CC BY 4.0 / MIT | 18.6 MB | MD5 vs Zenodo; 2,123 fragments, 0 missing values, features re-derived exactly |
| ESA-ADB | Zenodo 12528696 v1.0 (3 archives, already on disk, linked) + GitHub kplabs-pl/ESA-ADB @67194d3 | CC BY 3.0 IGO per the paper / MIT | 11.6 GB raw, 8.5 GB official grids | MD5 vs Zenodo; revision 15237121 compared member by member by CRC-32, and only `channels.csv` differs (fetched) |
| SpotGEO v2 + starter kit | Zenodo 4432143, 3874368 | CC BY 4.0 | 4.23 GB | MD5; 32,000 frames, 0 corrupt, 0 train/test duplicates |
| CelesTrak snapshot 2026-10-24 | celestrak.org (SATCAT, 6 GP groups, SW-All) | usage policy | 12 MB | SHA-256 recorded; 2026-08-16 snapshot kept separately |
| ESA recovered debris | reentry.esoc.esa.int public `/api/home` | ESA web content | 0.2 MB | parsed into 235 fragments / 87 objects; 195 of 197 SATCAT-linked fragments lie inside their orbit's latitude band |
| SpaceSense-Bench (sample) | HF Alvin16/SpaceSense-Bench @90bf1da, 1 of 136 archives | CC BY-NC 4.0 | 0.5 GB | inspected only (resolved spacecraft, off-task) |

Generated, kept separate from real data: `feature1_health/cosmos_synthetic` (200 nominal runs),
`feature2_debris/cosmos_synthetic/yolo_v1` (800/200/200 scene-split frame pairs), and
`feature3_reentry/simulated` (500k propagator trajectories). Derived from real data:
`feature3_reentry/derived/landmark_2026-08-16_to_2026-10-23.csv` (leakage report all True).

## 3. Downloads that failed or were not attempted, with the reason

- **SPARK (all releases):** by request through a contact form; the Zenodo record 6599762 is
  `restricted` with no public files. No request was made on your behalf.
- **Space-Track (GP history, TIP), ESA DISCOSweb, ESA re-entry predictions:** account or token
  required. The ESA prediction API answers HTTP 401. Not attempted further, and no access
  control was bypassed.
- **Full ESA-AD revision 15237121 (7.9 GB):** the download was started, then stopped once the
  member-level CRC check showed that only `channels.csv` differs.
- **SpaceSense-Bench (remaining 135 archives, ~72 GB):** not downloaded; the task does not match.
- **DUST:** still not locatable (audit A6.15).

## 4. New and modified source files

Modified: `cosmos/pipeline.py`, `README.md`, `data/README.md`. Everything else is new:

- `cosmos/ai/data/`: `download.py` (resumable, checksum-verified, manifests, revision check),
  `validate.py` (dataset report), `preprocess_telemetry.py` (official ESA-ADB grids, verified
  against a literal port of the official pandas code; OPSSAT tensors), and
  `preprocess_catalogue.py` (landmark, leakage report, recovered debris).
- `cosmos/ai/telemetry/`:
  - official metric code vendored unmodified (`esa_metrics/`);
  - `esa_eval.py`, `baselines.py` (exact GlobalSTD port, COSMOS monitor);
  - `models.py` (TCN, PatchTST, classifiers, gated fusion);
  - `forecasting.py`, `alarms.py` (conformal-BH, thresholds, rolling, event logic), `esa_benchmark.py`;
  - `opssat.py`, `robustness.py`, `learned_monitor.py`.
- `cosmos/ai/vision/`: `spotgeo.py` (front end, official scorer `spotgeo_validation.py`
  vendored), `heatmap.py`, `train_spotgeo.py`, `cosmos_synthetic.py`, `metrics.py` (COCO mAP).
- `cosmos/ai/survival/landmark.py`, `cosmos/ai/surrogate.py`.
- `cosmos/ai/uncertainty.py`, `evaluation.py`, `explain.py`, `fusion.py`, `registry.py`,
  `experiment.py`, `report.py`.
- `tests/test_ai.py`, `scripts/job_queue.py`, `scripts/build_ai_notebook.py`, `configs/**.yaml`,
  `requirements-ai.txt`, `.gitignore`, `notebooks/COSMOS_AI_Experiments.ipynb`.

## 5-8. Training, configs, checkpoints, evaluation

- **Training and evaluation entry points:** listed in the README section "Training and
  evaluation". Frozen protocols are in `configs/` (tested equal to the code defaults).
- **Run records:** 152 runs in `outputs/experiments/index.jsonl` (149 ok, 3 failed). The failures
  are the first TCN smoke test and two SpotGEO runs that hit a scorer division-by-zero; all were
  re-run successfully. Each run folder holds `config.yaml`, `env.json`, the epoch log,
  `metrics.json` and checkpoints.
- **Checkpoints:** 166 (487 MB), under `outputs/experiments/*/checkpoints/`, plus the surrogate
  in `data/feature3_reentry/simulated/propagator_surrogate_v1/checkpoints/` and the learned
  monitor in `data/feature1_health/cosmos_synthetic/`.
- **Model cards:** `outputs/registry/model_cards.json`, built from the benchmark files.

## 9-11. Results, figures, ablations, robustness

The full component table is `SUMMARY_TABLE.md`, and the figures are in `figures/`.

### Telemetry — ESA-ADB (primary study, official protocol)

- **Reproduction.** The ported official GlobalSTD3/5 reproduces DMLR Tables 4/16/17 on the
  primary metric in 8/8 configurations, and to every printed digit on Mission 1 lightweight.
  Secondary metrics are within 0.044.
- **Main result.** See the README table. In brief:
  - Mission 2: AR(8) + conformal-BH 0.962; PatchTST 0.952 ± 0.009; Chronos-Bolt zero-shot 0.945
    (published best 0.079). These arms catch only 1 of the 6 anomalies.
  - Mission 1: best stable arm AR(8) + threshold 0.088 (published best 0.061). A TCN with rolling
    calibration reaches 0.318 ± 0.215 at recall 0.10.
- **H1 (pretrained / deep encoder beats calibrated classical): not supported.**
  - Mission 2 Δ vs AR(8): PatchTST −0.021 [−0.051, +0.007], TCN −0.027 [−0.054, +0.002],
    Chronos −0.018 [−0.047, +0.014].
  - Mission 1: Chronos −0.052 [−0.075, −0.032], significantly worse.
- **H2 (conformal FDR alarms): supported on Mission 2, not on Mission 1.**
  - Mission 2 Δ conformal-BH − threshold: AR(8) +0.080 [+0.041, +0.133], PatchTST +0.124
    [+0.080, +0.177], TCN +0.077 [+0.037, +0.130].
  - Mission 1: −0.012 to −0.001.
  - Realised false alarms (conformal-BH arms): 3.4–22 per year on Mission 2, 92–304 per year on
    Mission 1.
  - Rolling recalibration helps Mission 1 (drift) and hurts Mission 2.
- **H3 (telecommand gating suppresses rare-nominal-event alarms): not supported.**
  - Rare-event-only alarm runs: M2 PatchTST 356 with telecommands vs 353 without; TCN 393 vs 399.
  - Primary-metric Δ (seed 0): M2 +0.010 [−0.011, +0.033] PatchTST, +0.013 [+0.000, +0.025] TCN;
    M1 −0.045 and −0.033 (significant).
  - Removing telecommands at test time changes nothing (0.951 → 0.951, 0.948 → 0.948), so the
    gate learned to ignore them.
  - Yet a fixed rule (mask 30 min after any priority-3 telecommand) cuts M2 rare-event-only runs
    351 → 42 with unchanged anomaly recall. The information exists; the forecasting objective
    does not make the gate use it.
- **Ablations run:** encoder swap; calibrator swap (3 calibrators); telecommands on/off;
  missing-modality at test; full vs lightweight channel sets; rule-mask baseline.

### Telemetry — OPSSAT-AD (5 seeds)

- **Official split:**
  - hybrid F1 0.949 ± 0.009, AUC-PR 0.986;
  - FCNN 0.937 / 0.980;
  - 1D-CNN 0.939 / 0.987;
  - XGBoost 0.911 / 0.961;
  - Chronos-Bolt probe 0.889 / 0.955;
  - COSMOS robust-z 0.436 / 0.471.
  Published: FCNN 0.946 / 0.979, XGBOD 0.918 / 0.975, AdaBoost 0.836 (ours 0.838).
- **Paired bootstrap:** hybrid − FCNN AUC-PR +0.006 [+0.0004, +0.013]; hybrid − XGBoost +0.025
  [+0.008, +0.046].
- **Time-blocked split:** hybrid 0.910, FCNN 0.907, XGBoost 0.894. Every model drops.
- **Label efficiency, AUC-PR at 10 %:** FCNN 0.936, CNN 0.912, hybrid 0.910, XGBoost 0.891,
  Chronos probe 0.823.
- **Robustness, F1:**
  - noise 0.1σ: hybrid 0.856, FCNN 0.807, XGBoost 0.766, CNN 0.568;
  - 30 % sample dropout: XGBoost 0.850, deep models about 0.67–0.77.
- **Explanation check:** Integrated Gradients on the hybrid fails the randomisation check
  (rank ρ ≈ 0.91), so its maps are not used as evidence.

### Telemetry — learned monitor in the COSMOS pipeline (SYNTHETIC)

- Held-out seeds 5–9: learned F1 0.775 vs classical 0.834; localisation 0.960 vs 0.830; false
  alarms on 3 of 40 fault-free runs vs 0 of 40.
- The design was iterated on seeds 0–4.

### Vision — SpotGEO (real) and COSMOS frame pairs (SYNTHETIC)

- **SpotGEO, official F1:**
  - heat-map U-Net 0.746 ± 0.005 (n=3);
  - pretrained ResNet-18 encoder 0.765 ± 0.034 (n=2; within noise of scratch);
  - classical COSMOS front end + DoG 0.084;
  - challenge winner 0.948.
- **SpotGEO ablations:**
  - front end: raw high-pass only 0.751 vs registered difference only 0.644 (the classical front
    end is not needed once all 5 frames are input);
  - label efficiency: 10 % → 0.535, 25 % → 0.663.
- **Synthetic, COCO mAP@50 / mAP@50-95:**
  - heat-map 0.780 / 0.370;
  - YOLOv8n 0.431 / 0.244 (Ultralytics' own evaluator reports 0.55 / 0.31 on average, from its
    interpolation);
  - TinyYOLO 0.379 / 0.162 (its legacy centre-distance "mAP@50" is 0.935).

### Re-entry — real landmark, and simulator surrogate (SIM)

- **Landmark, AUC-PR within 68 / 30 days:**
  - XGBoost 0.821 / 0.761;
  - XGBoost-AFT 0.814 / 0.658;
  - hazard-MLP ensemble 0.688 / 0.610, after a validation-selected grid; Δ vs XGBoost −0.134
    [−0.221, −0.024];
  - COSMOS physics 0.188 / 0.254.
  - The deep model did not earn its place, so XGBoost is the reported model.
- **Conformal sets at 68 days:** marginal sets cover 0–13 % of events. Label-conditional sets
  cover 87–97 % of events and 88–93 % of non-events (nominal 90 %).
- **Surrogate:**
  - median impact error 16 km (p95 69 km); 80,000 vs about 220 samples/s;
  - over 12 footprint scenarios, median CEP50 relative error 1.7 % (max 5.2 %) and coverage of
    reference impacts by the surrogate's 95 % conformal zone 0.955 (min 0.927);
  - out-of-range flag AUROC 0.73.
  - In-pipeline modes: all-surrogate is 70–200× faster with +2–3.5 % CEP50 bias; any reference
    fallback forfeits most of the speed-up.

## 12-13. Notebook and README

- `notebooks/COSMOS_AI_Experiments.ipynb` (built by `scripts/build_ai_notebook.py`) was executed
  end to end in this environment on CPU with `jupyter nbconvert --execute`. It loads real data,
  inspects labels, runs saved models, evaluates against baselines, shows uncertainty and
  explanation checks, and runs the integrated pipeline.
- The original `COSMOS_Demo.ipynb` is unchanged.
- README: environment, data commands, training and evaluation commands, outputs, hardware,
  headline results, limitations, reproducibility.

## Decisions taken after a test number had been seen (for the paper's threats-to-validity section)

1. **Re-entry MLP.** The tuning grid was added after the first, untuned MLP had been scored on
   test (0.528). Selection within the grid used validation only, and the conclusion (trees win)
   is unchanged.
2. **Re-entry conformal sets.** Mondrian sets were added after marginal sets were seen to miss
   the events. This is a methodological addition, not a tuned quantity.
3. **Learned monitor.** The calibration design was changed twice on seed sets 0–4. The reported
   numbers are from seeds 5–9, scored once afterwards.
4. **OPSSAT-AD.** The study was re-run to save per-fragment scores, with no other change.
   Results moved by up to 0.025 F1, because XPU training is not deterministic.
5. **Surrogate pipeline.** The fallback threshold was recalibrated on validation, and the sampler
   was made to replicate the reference's draws, after the first in-pipeline run.
6. **ESA-ADB bug fixes after first test scores.** These were fixes to timestamp units in
   false-alarm counting and to output paths, and changed no alarm or score. The ESA calibrator
   grid hit its edge on Mission 2 (k = 30) and was not widened after seeing test numbers.

## 14. Remaining limitations and next steps for the paper

- **Seeds.** Deep ESA arms have 3 seeds (AR(8), persistence and Chronos have 1, being
  deterministic). The design asked for ≥ 5, which was compute-bound at 20–60 GPU-min per arm.
  Mission 1 seed variance is large, so add seeds before any Mission 1 claim. Bootstrap CIs cover
  test-time variability only.
- **Benchmark coverage.** Mission 1's calibration period holds 9 events, so use a longer
  calibration window or pooled selection across phases. Mission 2 has 6 test anomalies, so its
  anomaly-only results are fragile.
- **H3.** Train the gate against an alarm-level objective (e.g. penalise alarms on
  rare-nominal-labelled training spans), or add the rule mask as a calibrated component and
  test it pre-registered on fresh missions/phases. Note that rare events are *targets* in the
  primary DMLR metric, so suppression must be scored on the anomaly-only metric and on
  rare-event-only run counts.
- **Foundation models.** Only Chronos-Bolt-small zero-shot was run. Chronos-2 and fine-tuning
  (LoRA) are implemented in the library but were not run here.
- **Secondary metrics.** The reproduction gap on channel- and subsystem-aware scores (up to
  0.044) is unexplained.
- **Vision.** Temporal post-processing of the heat-map detections, a SpotGEO-like synthetic
  generator for sim-to-real, and an SNR breakdown remain to do.
- **Re-entry.** Obtain Space-Track access for multi-landmark training with element-set history
  (CREATION_DATE ≤ τ). No impact-location accuracy can be claimed until re-entry state vectors
  are available.
- **Read before submission** (as listed in the research design): Heydari et al. 2026 (possible
  H2 overlap), Fejjari et al. 2026, Dai et al. 2026.


## 15. Pre-registered telemetry study (added 2026-10-24, after the Q1 request)

- **Design.** `research_design/preregistration/PREREG_TELEMETRY_V1.md`, frozen with SHA-256
  hashes (`freeze_09a98f5e160e.json`), test scoring gated by the hash, and deviations logged
  in its section 9 (re-frozen afterwards as `681cc7ddec86`).
  - Out-of-fold scores over the whole training half: `telemetry/oof.py`.
  - Supervised event scorers, TCN and XGBoost: `event_model.py`, `event_gbt.py`.
  - Alarm selection on the training half and a single gated test pass:
    `oof_benchmark.py`, `scripts/prereg_analysis.py`.
- **Results.** `oof_protocol/CONFIRMATORY.md`. Mission 1 is primary (Holm family); Mission 2
  is a replication. Values are test half, conformal-BH, seed means:
  - **Mission 1:** A0 (AR(8), 9-month selection) 0.074; A1 (AR(8), whole-half selection)
    0.428; A2 (XGBoost event scorer) 0.544 ± 0.236; A3 (TCN event scorer) 0.205 ± 0.114.
  - **Hypotheses:**
    - H6 (A1 > A0) is **supported**: +0.353 [+0.206, +0.488], Holm p 0.0015.
    - H4 (A3 > A1) is **refuted**: −0.222 [−0.395, +0.042].
    - H5 (A3 > A2) is **refuted**: −0.338 [−0.540, −0.033].
  - **Mission 2:** A0 0.962, A1 0.874, A2 0.904, A3 0.875. H6 reverses: −0.089
    [−0.143, −0.049].
  - **Baselines on the same code:** an out-of-limit rule scores 0.894 on Mission 2 and 0.0007
    on Mission 1. Point-level ACI (TC-CAD-style) scores ≤ 0.002 everywhere.
- **Exploratory** (`oof_protocol/exploratory/`):
  - trees − AR(8) on Mission 1: +0.108 [−0.133, +0.344];
  - nominal score drift between halves: TCN 3–54×, trees 5–9×, AR(8) 0.8×;
  - no arm detects the 3 Mission 1 test anomalies whose class is unseen in training.
- **Consequence for the paper.** Deep learning is a pre-registered negative result. The
  manuscript (`paper/manuscript.tex`) is framed accordingly: alarm selection and score
  stability matter more than model capacity.

## Component table

See `SUMMARY_TABLE.md` (generated by `python -m cosmos.ai.report`).
