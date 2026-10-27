# Deliverable B: Claim-to-evidence matrix (Stage 1)

**Strength scale.**

- **Demonstrated (confirmatory):** a pre-registered test whose verdict follows the frozen rule.
- **Demonstrated (measured):** a measured result reproduced from its result file, but not confirmatory.
- **Partially supported:** the evidence exists but the claim goes further than it.
- **Exploratory:** post hoc or not pre-registered; descriptive only.
- **Proposed:** not done.
- **Unsupported:** contradicted or without evidence.

**Path abbreviations.**

- `T/` = `outputs/benchmarks/telemetry_esa_adb/`
- `P/` = `T/oof_protocol/`

Every number below is the value in `thesis/evidence/evidence.json`. Test-half exposure is described in [STAGE1_AUDIT.md](STAGE1_AUDIT.md) §3.

## Primary telemetry study (ESA-ADB)

| # | Exact claim | Supporting result | Evidence location | Strength | Relevant limitation | Required action |
|---|---|---|---|---|---|---|
| T1 | The ported official GlobalSTD rule reproduces the published values to within 0.001 in all 8 configurations | Table 10 | `T/results.csv`; DMLR Table 4 | Demonstrated (measured) | Checks the pipeline, not a method | None |
| T2 | The parameter-free out-of-limit rule scores 0.894 on Mission 2 full channels, so Mission 2 is near-saturated | Table 10 | `outputs/logs/ool_baselines.log`, `T/results.csv` | Demonstrated (measured) | Exploratory; seen before the pre-registration | None |
| T3 | **H6:** on Mission 1 the A1 protocol beats A0 by +0.353 (95 % CI [+0.206, +0.488], Holm p ≤ 0.0015) | Table 13, Fig. 10 | `P/CONFIRMATORY.json`, `P/test/m1_target_ar8.json`; A0 from `T/results.csv` | Demonstrated (confirmatory) **as a comparison of two protocols** | (i) A0 and A1 differ in **five** respects: fit data, residual scale σ, calibration pool, selection data and grid (STAGE1_AUDIT §5.1; the thesis says four); (ii) A0's test value (0.074) and A1's training-half CV (0.624) were known before the freeze; (iii) the test half had been used in the exploratory study; (iv) a single deterministic run per arm, so the CI reflects test-period resampling only | Keep the wording "protocol"; run the factorial decomposition (Plan F1) before any attribution |
| T4 | The H6 gain is not specific to BH: +0.340 under the per-channel threshold | Table 15, Fig. 15 | `seleff.m1.full.threshold` ← `P/test/*.json` | Exploratory | No interval; same four-way confound | Add a bootstrap interval (cheap; Plan F5) |
| T5 | The selection effect depends on the setting: −0.383 (lightweight M1, BH), −0.056 (lightweight M1, threshold), −0.089 (M2, BH), +0.094 (M2, threshold), +0.070 / +0.157 (lightweight M2) | Table 15 | `seleff.*` | Exploratory | Lightweight "A0" is the exploratory lightweight arm; single runs, no intervals | Intervals (F5); report in the paper as a limitation of H6's generality |
| T6 | **H4 and H5 refuted:** the TCN did not improve on AR(8) (−0.222, CI [−0.395, +0.042]) or on trees (−0.338, CI [−0.540, −0.033]) by the minimum effect | Table 13 | `P/CONFIRMATORY.json` | Demonstrated (confirmatory) | "Refuted" means no benefit ≥ 0.05, not "worse". One architecture, one configuration, 5 seeds, no tuning budget | Keep; never generalise to "deep learning" |
| T7 | Supervised arms vary strongly between seeds: trees 0.157–0.769, TCN 0.085–0.358 | Fig. 9 | `pre.seed.*` | Demonstrated (measured) | 5 seeds | Report per-seed table; more seeds (F4) |
| T8 | The supervised scorers' nominal scores shift between halves (test/CV exceedance 2.8–54× for TCN, 5.1–9.3× for trees, 0.8× for AR(8)) | Fig. 17 | `P/exploratory/calibration_shift_m1.json` | Exploratory | Mission 1 only; every 4th channel; post hoc | None for the thesis; pre-register if used in the paper |
| T9 | The shift does not explain differences between runs (Spearman ρ −0.12 over 10 runs; 0.00 within TCN seeds) | Fig. 16 | `shift.m1.*.spearman_f05` | Exploratory | n = 5 or 10; descriptive | None |
| T10 | The trees' Mission 1 lead is on rare events: anomaly-only F0.5 0.327 (trees) against 0.336 (AR(8)); about 18 of their ~21 extra detections are rare events | Figs. 18–19 | `pre.m1.A2.conformal_bh.f05_anom`, `evcls.m1.gain_a2_a1.*` | Exploratory | Seed means; 52 + 3 test anomalies | None |
| T11 | Point-level ACI (α = 0.01) fails at the event level (F0.5 ≤ 0.0017 over 30 runs); adding event logic gives at most 0.046 (seed mean), with 10/30 runs raising no alarm | Table 14 | `pre.all.aci*` | Demonstrated (measured) for this configuration | γ tuned to a point rate, not the event metric; γ = 0.05 in 29/30 runs silences each channel for 80 steps after an exceedance, so persistence is rare (assembled across channels), not impossible; one α | Fair ACI comparator (F3) before any claim about the method family |
| T12 | Aggregation to events, not cross-channel BH, makes alarms usable: the threshold ≥ BH in 8 of 10 rows | Table 14, Fig. 14 | `layers.n_threshold_ge_bh` | Exploratory | Seed means, no intervals | Paired intervals (F5) |
| T13 | The A1 alarm burden on Mission 1: 1.1 false alarms/yr, 17.6 alarm runs/yr, recall 0.176 (16 of 91 events), alarming precision 0.139 | Fig. 11 | `P/test/m1_target_ar8.json` | Demonstrated (measured) | No detection delay or alarm-duration distribution reported yet | Add delay and duration from the alarm files (F6; no new run needed) |
| T14 | H3 (telecommand gating reduces rare-only alarm runs) not supported | Table 11 | `T/results.csv`, `T/bootstrap_comparisons.json` | Exploratory | 3 seeds; exploratory phase | None |
| T15 | Cross-validation over-estimates test performance for the supervised arms and lightweight M1 (0.528 CV → 0.025 test) | Fig. 12 | `P/selection/*.json`, `P/test/*.json` | Demonstrated (measured) | Cross-fitting, not forward chaining | Forward-chaining evaluation (F2) |
| T16 | Our Mission 1 results exceed the published full-channel best (0.061) | Discussion | DMLR Table 4 | Partially supported | Published baselines did not tune on training labels; on the lightweight subset we are far below the best published value (0.786) | State the label-use difference wherever compared |
| T17 | No prior pre-registered full-channel comparison of unsupervised, tree and deep event scorers under the official split | Ch. 2 | literature review | Partially supported | Brancato et al. (AST), Fejjari et al. and Dai et al. are not read in full | Literature audit (Stage 4) before any novelty claim |
| T18 | The vendored metric is the official one | Ch. 3 | byte comparison of all 5 metric files with upstream commit 67194d3, re-run 2026-10-25 | Demonstrated (measured) | — | Unit tests of the adapter (F7) |
| T19 | The confirmatory intervals and p-values are valid for the official metric | Table 13 | `event_bootstrap.py` | Partially supported | The bootstrap uses span-based events, cannot detect off-grid point events, and uses sample-count TNR (exact for A1, A3 and all M2 arms; H5 −0.330 vs −0.338 official) | Recompute with an official-equivalent bootstrap in a new, non-frozen module; report both (F5) |
| T20 | Every confirmatory input was produced by the frozen code | Appendix B | `freeze_09a98f5e160e.json`, file times | Partially supported | The gate hashes code only; the A1 selection files predate the last edit of `oof_benchmark.py`; the parallel test driver is unfrozen | Re-run the A1 selection with the frozen code and compare byte for byte (training half only; cheap) |
| T21 | The test inputs are free of label information | Ch. 5 | `preprocess_telemetry.py:143–157` | Partially supported | The official "restore" step moves annotated samples onto grid points (M1: 51 points, zero effect on detections; M2: 470 points) | State it as a benchmark property in the thesis and paper |

## Supporting studies

| # | Exact claim | Supporting result | Evidence location | Strength | Relevant limitation | Required action |
|---|---|---|---|---|---|---|
| S1 | OPSSAT-AD: the hybrid reaches F1 0.949 (official split); the re-implemented FCNN reproduces the published 0.946 (0.937) | Table 16 | `outputs/benchmarks/telemetry_opssat/summary_official.csv` | Demonstrated (measured) | The official split interleaves fragments in time | None |
| S2 | On OPSSAT's tabular features the FCNN beats XGBoost (F1 +0.032, CI [+0.004, +0.063]) | Results | `paired_comparisons.csv` | Demonstrated (measured) | Paired bootstrap on seed means; not multiplicity-corrected | None |
| S3 | Deep models' lead shrinks under the time-blocked split; AdaBoost and logistic regression rise | Table 16 | `summary_time_blocked.csv` | Demonstrated (measured) | One time split | None |
| S4 | SpotGEO: heat-map U-Net F1 0.746 (3 seeds) ≫ classical front end 0.084; far below the winner 0.948 | Table 17 | `outputs/benchmarks/vision_spotgeo/summary.csv` | Demonstrated (measured) | Input ablations single-seed | Label ablations as weak |
| S5 | Re-entry landmark: XGBoost beats the hazard MLP at 68 days (AUC-PR −0.134, CI [−0.221, −0.024]) | Table 18 | `outputs/benchmarks/reentry_landmark/results.json` | Demonstrated (measured) | One landmark date; the MLP grid was added after a first test score | None |
| S6 | On uncontrolled objects at 30 days the physics lifetime has the highest AUC-PR (0.512) | Fig. 24 | as above | Exploratory | 4 positives | Do not use as a claim beyond "suggests" |
| S7 | The surrogate reproduces the COSMOS propagator (median error 16.2 km) | Results | `outputs/benchmarks/reentry_surrogate/results.json` | Demonstrated (measured), simulator only | No real re-entries | None |
| S8 | Learned monitor: F1 0.775 against classical 0.834 on held-out simulated seeds | Results | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` | Demonstrated (measured), SIMULATED | Calibration design changed twice on seeds 0–4 | None |

## Original pipeline and physics

| # | Exact claim | Supporting result | Evidence location | Strength | Relevant limitation | Required action |
|---|---|---|---|---|---|---|
| P1 | All 20 verification checks pass | Table 7 | `outputs/tests/test_verify_after.log` | Demonstrated (measured) | Only 10 use external references; 9 hold by construction or are self-consistency checks; the hydrostatic check is mis-specified; `check()` never raises | Fix the hydrostatic check and the non-raising `check()` (Stage 3, code change with a test) |
| P2 | The chain couples health to break-up physics | Table 9 | `thesis/evidence/original_pipeline_runs.json` | Partially supported | Tie-break defect; P_frag unused; undetected faults mis-attributed; hand-set weights | Fix the tie-break if intended behaviour is confirmed with you |
| P3 | Conformal density zones reach near-nominal coverage (0.944 at 0.95) | Results | as above | Demonstrated (measured), SIMULATED, one footprint | One footprint | None |
| P4 | The lifetime model under-predicts Cosmos 1408 by about 3× (ratio 0.292) | Results | as above | Demonstrated (measured), real catalogue | Static atmosphere, single area-to-mass ratio | None |
| P5 | Impact-location / re-entry prediction accuracy | — | — | **Unsupported** (no ground truth) | No recorded trajectories with recovery coordinates | Never claim; future work needs DISCOS / Space-Track access |
| P6 | COSMOS is an integrated, validated digital twin | — | — | **Unsupported** | Components are offline; debris module not connected; no real-world validation | The thesis must call it a simulation framework |
