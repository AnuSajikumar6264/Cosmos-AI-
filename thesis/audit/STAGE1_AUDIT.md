# Stage 1: Scientific audit of the COSMOS thesis (Deliverable A)

**Date:** 2026-10-25.

**Scope.** This audit covers the thesis build of 2026-10-25 (`thesis/build/COSMOS_Thesis.pdf`, 187 pages), its sources and its supporting files. It also includes a read-only code audit of the pre-registered pipeline, done independently of the thesis text (§5).

**What it does not change.** No result file, frozen file or experiment was changed or run for this audit. Companion files:

- [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md) (Deliverable B);
- [REVISION_PLAN.md](REVISION_PLAN.md) (Deliverable C and the experiment designs of F);
- [CHANGELOG.md](CHANGELOG.md).

## 1. Evidence inventory

**Central research question (as pre-registered).** On ESA-ADB, under the official split, with the full target channel sets and the corrected event-wise F0.5, two questions:

- Does a supervised deep event scorer improve on an unsupervised forecaster when both feed the same event-level alarm layer, selected on the whole training half?
- Does that selection protocol by itself fix the Mission 1 failure of the exploratory study?

**Hypotheses and outcomes (Mission 1, conformal-BH layer, frozen rule).**

| ID | Comparison | Point estimate | 95 % CI | Raw p | Holm p | Verdict |
|---|---|---|---|---|---|---|
| H4 | A3 − A1 | −0.222 | [−0.395, +0.042] | 0.101 | 0.101 | refuted: no gain ≥ 0.05; not shown worse |
| H5 | A3 − A2 | −0.338 | [−0.540, −0.033] | 0.035 | 0.070 | refuted: no gain ≥ 0.05 |
| H6 | A1 − A0 | +0.353 | [+0.206, +0.488] | 0.0005 (floor) | 0.0015 | supported, as a comparison of two protocols |

The Holm arithmetic is correct (0.0005 × 3; max(0.0015, 0.035 × 2); max(0.07, 0.101)). The point estimates use the official metric. The intervals and p-values come from the block-decomposable re-implementation, which differs from the official metric by at most 0.0081.

**Mission 2 replication (no inference).**

- H4: +0.001 [−0.055, +0.054].
- H5: −0.029 [−0.103, +0.037].
- H6: −0.089 [−0.143, −0.049].

**Data and splits.**

- **Data.** ESA Anomalies Dataset v1.0 (Zenodo 12528696), Missions 1 and 2, official grids (30 s and 18 s).
- **Split.** Official split at 2007-01-01 (M1) and 2001-10-01 (M2).
- **Channels.** Target channel sets of 58 and 47 channels.
- **Training labels.** M1 holds 109 annotated events on the target channels: 105 targets (63 anomalies, 42 rare events) and 4 communication gaps.
- **Test labels.** M1 holds 55 anomalies and 36 rare events. M2 holds 6 anomalies and 356 rare events.
- **Mission 3** is on disk but has no official split in the code. Its authors advise against benchmarking on it, and it holds 8 anomalies and 25 rare events in total.

**Models.**

- AR(8), ridge 1e-3, horizon 16, out of fold with K = 5 contiguous blocks and a 1-day purge.
- XGBoost: 3 classes, 400 trees of depth 8.
- EventTCN: 8 levels, receptive field 1,021, 220,739 parameters, 1,500 steps.
- Comparators: point-level ACI (α = 0.01, γ from {0.001, 0.005, 0.01, 0.05}), a threshold layer, and a rolling layer.

**Metric and statistics.**

- **Metric.** Corrected event-wise F0.5 over anomalies and rare events, computed by the vendored official code.
- **Seeds.** A2 and A3 have 5 seeds each; the ablations have 3; A0 and A1 are deterministic.
- **Statistics.** Paired bootstrap with fixed 30-day blocks (86 blocks on M1, 22 on M2), hierarchical over seeds, B = 2,000. Holm correction over {H4, H5, H6}, with a minimum effect of +0.05.

## 2. Central contribution, as the evidence stands

The study is a pre-registered, hash-frozen comparison under the official ESA-ADB protocol at full channel scale. Its findings:

1. On Mission 1, a protocol that selects an event-level alarm layer on out-of-fold scores of the whole training half outperformed the calibration-window protocol by a large margin.
2. A supervised TCN event scorer showed no benefit over AR(8) or gradient-boosted trees.
3. Point-level ACI, as configured, cannot produce usable event-level alarms, and the reason is a mechanism of its update rule.

The defensible claim is narrower than "alarm selection matters more than model architecture". The selection protocol's gain is:

- confounded with four other protocol differences (§5.1);
- specific to Mission 1 at full channel scale;
- reversed on the lightweight Mission 1 subset and, under BH, on Mission 2.

The provisional title "Beyond Model Complexity: Evaluating Event-Level Alarm Selection for Spacecraft Telemetry Anomaly Detection" is **compatible with the evidence only if** the paper states these conditions. It also needs a decomposition of H6 (F1), without which it cannot say *which* part of the protocol matters.

## 3. Timeline of test-half exposure (reconstructed from run IDs, file times and the pre-registration)

| Time (2026-10-24 unless stated) | Event | Source |
|---|---|---|
| 09:11 – 12:11 | Exploratory study scores both official test halves in 128 configuration–layer combinations of persistence, AR(8), TCN, PatchTST and Chronos-Bolt, with and without telecommands, under the threshold, conformal-BH and rolling layers, on full and lightweight channels | `run_id` timestamps in `T/results.csv` |
| by 13:22 | Exploratory bootstrap comparisons (H1–H3) on the test halves | `T/bootstrap_comparisons.json` |
| by 15:24 | Out-of-limit and GlobalSTD baselines scored on the test halves | `outputs/logs/ool_baselines.log` |
| from 15:29 | Training-half CV of the A1 AR(8) arm computed and seen (M1 0.624, M2 0.919) | `P/selection/m*_target_ar8.json`; PREREG §0 |
| 15:44 – 15:53 | Seed-0 TCN and XGBoost training started, with the designs fixed | PREREG §0 |
| **15:56:29** | **Freeze `09a98f5e160e`**: the document plus 27 code files | `freeze_09a98f5e160e.json` |
| 18:25 | First confirmatory test scoring (A1, M1) | `P/test/m1_target_ar8.json` |
| up to 20:11 | Training-half selection of late arms, after other arms had been test-scored (deviation 3) | `P/selection/*` file times |
| 20:37 | Last arm test-scored | `P/test/*` file times |
| 20:48 | `CONFIRMATORY.json` | file time |
| 21:01 | Freeze files rewritten: the re-freeze `681cc7ddec86` after the deviation log was filled in. The log gives "~22:00"; the file time is 21:01:59 | file times; PREREG §9 |
| 2026-10-25 01:04 | Commit `2525367` | git |

**What was known before the freeze.**

- The **test value of A0** (M1 0.074, M2 0.962) was known, because A0 *is* the exploratory arm.
- It was known that Mission 1 failed under the calibration window.
- The training-half CV of A1 and the out-of-limit test results were known.

**What was not known.**

- The test values of A1, A2 and A3.
- Any training-half CV of A2 and A3.

**What can be called pre-registered.**

- H4–H6 on Mission 1 (conformal-BH, primary metric), and the Mission 2 replication.
- The list of secondary metrics and comparators, as "reported without inference".

**What cannot be called pre-registered.**

- Table 15 (all cells except M1 full / BH).
- The drift diagnostic, the event-class decomposition and the alarm-burden analysis.
- The layer comparison as an inference.
- Everything in the exploratory study.

**Consequences.**

- (i) The test halves are **not** untouched. The pre-registration controls which comparisons are confirmatory. It does not undo adaptive design based on earlier test results, namely the decision to study Mission 1 selection and the extension of the grid to q = 1e-5 and k = 100.
- (ii) H6's comparator value was known in advance, so H6 tests whether a newly designed protocol beats a known baseline. The outcome (A1's test value) was genuinely unknown.
- (iii) A1 selected q = 1e-4, k = 10, a setting that is in **both** grids. The grid difference may therefore be immaterial for A1. That is not shown, and F1 would show it at low cost.

**No genuinely independent evaluation set exists in the available data.** Proposed substitutes are:

- (a) a forward-chaining, pseudo-prospective evaluation inside the training halves (F2). These periods have never been used as evaluation periods, but their labels entered the A1 cross-validation objective.
- (b) Mission 3 as an underpowered external sanity check, which its authors do not recommend.

Neither substitute would be "fresh data", and the thesis and paper must say so.

## 4. Strengths

- The official metric code is vendored byte-identically, and the official GlobalSTD baseline is reproduced to within 0.001 in eight configurations.
- Hash-frozen pre-registration, gated test scoring, a logged deviation table, and every outcome reported, including refuted hypotheses.
- Strong simple baselines: a parameter-free out-of-limit rule, trees on the same features, and the official GlobalSTD.
- The hierarchical block bootstrap with Holm correction matches the dependence structure better than i.i.d. tests.
- Every number in the thesis is traceable to a result file through `evidence.json`.
- Stored alarm arrays for every arm and layer allow new analyses (intervals, delay, duration) without refitting.

## 5. Code audit of the pre-registered pipeline (independent, read-only)

An independent read-only review of `cosmos/ai/telemetry/*`, `cosmos/ai/data/preprocess_telemetry.py` and the drivers was carried out (scratch scripts in `/tmp/leak_audit/`). I re-checked its two most consequential statements myself: the residual-scale difference in `forecasting.py:391–393` against `oof.py:115`, and the metric files. The vendored metric files are **byte-identical to upstream commit 67194d3**: `ESA_ADB_metrics.py`, `latency_metrics.py`, `metric.py`, `ranking_metrics.py` and `utils.py`, compared with `data/feature1_health/esa_adb/raw/kplabs-pl__ESA-ADB-67194d389c4a.tar.gz`.

**5.1 A0 and A1 differ in five respects, not four.** The thesis and deviation 6 say four. The full list:

| # | Difference | A0 | A1 |
|---|---|---|---|
| 1 | Forecaster fit data | `fit` period only, nominal check on the target and lag 8 (`forecasting.py:288`) | all nominal training samples, all 8 lags nominal (`oof.py:52–55, 114`) |
| 2 | **Residual scale σ** (newly identified) | out-of-sample residuals on the 3-month `es` period (`forecasting.py:391–393`; the docstring at l.16 wrongly says `fit`) | in-sample residuals over the whole training half; per-fold in-sample σ for the OOF scores (`oof.py:112–115`). On the M1 test half the per-channel ratio of median nominal scores (A0/A1) has median 1.007 but ranges 0.52–4.70; 37 of 58 M1 channels sit at the σ floor |
| 3 | Calibration pool | 792,000 rows of the 9 months before test, from the model that scores test (`oof_benchmark.py:293–296`) | 7,364,161 OOF rows from five fold models with different σ, while test scores come from the full-fit model (`oof_benchmark.py:238–240`) |
| 4 | Selection data | 9 M1 events (7 anomalies, 2 rare); on M2, 43 rare events and 0 anomalies | 105 M1 events (+ 4 gaps); 282 on M2 |
| 5 | Grid | {1e-2, 1e-3, 3e-4, 1e-4} × {1, 3, 10, 30} (`alarms.py:139–142`) | {1e-2 … 1e-5} × {1, 3, 10, 30, 100} (`oof_benchmark.py:54–57`) |

Identical in both: normalisation (median and IQR/1.349 of the `fit`-period nominal samples), score formula, forecast-origin schedule on test, the alarm layer, the 1 h merge, the objective and the refit-after-selection rule. The frozen code reproduces the saved A0 and A1 alarms bit for bit.

**5.2 Leakage.** The review found **no path by which test-half labels reach any model, score, calibration pool or selection.** `select()` loads the test scores but does not use them (`oof_benchmark.py:114`). Minor flows, all quantified:

- **Official "restore" step** (`preprocess_telemetry.py:143–157`). It moves annotated samples onto grid points, so test labels shape test inputs. This is inherent to the benchmark's official preprocessing.
  - M1: 51 grid points changed; 13 points (events id_11, id_118, id_121) annotated only because of restore. The effect on M1 detections is zero.
  - M2: 470 points and 1 run that exists only through restore.
  - The pre-registration does not mention the restore step. The thesis and paper must.
- **Inputs carried across held-out boundaries.** Fold normalisation statistics include the held-out block's nominal samples; XGBoost/TCN `log_s` inputs come from AR models fitted with the held-out block. Slow EWM features carry inputs (not labels) past the 1-day purge (up to 72 % of `ewm_3d` at the first post-purge row). All affect inputs only; judged negligible.
- **Shared grid point at the split.** One shared grid point lets the first test sample enter the last training row's features (`event_model.py:105–109`): 1 row of 7.36 M.
- **Other operations.** The forward differencing of monotonic channels touches no target channel. The merge is non-causal, but recomputing without it leaves A0 and A1 true and false positives unchanged.
- **Boundaries.** No anomaly or rare event crosses any fold, selection-half, calibration-half or split boundary.

**5.3 Metric.** Every reported test metric goes through the vendored official code (`esa_eval.py:87–99`), including A0's exploratory value.

**Confirmed defect:** the bootstrap re-implementation (`event_bootstrap.py`), which produces every confirmatory CI and p-value, differs from the official metric in three ways:

- (i) Each event is reduced to one [min start, max end] span, although 81 of 91 M1 test events have several intervals.
- (ii) Off-grid point events can never be detected (id_42, lost for all A2 seeds).
- (iii) TNR is computed from sample counts rather than seconds.

It is exact for A1, all A3 seeds and all M2 arms. Its differences from the official metric are H5 −0.330 vs −0.338 and H6 +0.352 vs +0.353. **No verdict changes.** The intervals and p-values must still be described as coming from an approximation.

**5.4 The freeze gate covers code only.** `check_freeze` re-hashes the listed files. It does not cover:

- the freeze record itself;
- the selection JSONs and the OOF / event-model outputs;
- `esa_benchmark.py`, `cosmos/common.py`, or `scripts/prereg_test_parallel.py`, which ran the test pass;
- library versions.

"Scored once" is enforced by the drivers skipping existing results, not by `test()`.

The A1 selection files (m2 at 15:29:43, m1 at 15:33:05, file-server time) predate the last edit of `oof_benchmark.py` (15:37:52) and were not re-run. Their grids, fields and argmax match the frozen rule, but the gate cannot attest to their provenance.

The PREREG document no longer matches the `09a98f5e160e` hash, as expected after §9 was filled in. There is no copy that proves only §9 changed.

**Severity:** a provenance concern; no effect was found.

**5.5 Clock offset.** The compute host's clock runs about 17 minutes ahead of the file server. `run_id`s and `written` fields use compute time; file modification times use server time. The order of the timeline in §3 is unaffected; the exploratory runs end hours before the freeze on either clock.

**5.6 Review ranking.**

1. H6 compares two whole protocols: documented limitation, now five differences.
2. The bootstrap is not the official metric: confirmed defect, small.
3. The gate covers code only, and the A1 selection files predate the frozen code: plausible provenance concern.
4. Official restore puts test labels into test inputs: benchmark limitation, zero effect on M1.
5. Input-only cross-fold flows: not an issue in practice.
6. Cross-fitting instead of forward chaining: documented limitation.
7. Differencing, back-fill, the shared grid row and the merge: not an issue.

## 6. Critical methodological risks (ranked)

1. **Confounded primary comparison (H6).** Five simultaneous differences separate the arms (§5.1; the thesis says four), so no attribution is possible yet (F1).
2. **Test-half exposure.** Both test halves were scored in 128 configuration–layer combinations during the exploratory phase, before the pre-registered protocol was designed. The pre-registration was motivated by a test result.
3. **Generality of the selection effect.** The sign changes across channel sets and layers, and only one of the eight cells is confirmatory.
4. **Comparator fairness.** The ACI rate γ was tuned to a point rate, not the event objective. With γ = 0.05 (29 of 30 runs) each channel is silenced for 80 steps after an exceedance, so persistence must come from different channels in turn and is rare (not impossible: 20 of 30 event-logic runs raised alarms). Neither supervised scorer had a tuning budget. One deep architecture.
5. **Seed variance.** Trees range 0.157–0.769 and the TCN 0.085–0.358 with five seeds, so arm means are uncertain to about ±0.1.
6. **Cross-fitting optimism.** CV over-estimates test performance strongly for the supervised arms and for lightweight M1 (0.528 → 0.025). Forward chaining is not done.
7. **Bootstrap granularity and approximation.** Mission 2 has 22 blocks, and H6's raw p sits at the 1/2000 floor. The bootstrap uses an approximate re-implementation of the metric (§5.3).
8. **Freeze coverage.** The gate covers code only. The A1 selection files predate the frozen code; the test driver is unfrozen (§5.4). This is a provenance concern; no effect was found.

## 7. Missing evidence

- Decomposition of H6 (F1).
- A temporal or forward-chaining check (F2).
- Intervals for the exploratory contrasts (F5).
- Detection-delay and alarm-duration distributions (F6).
- Unit tests of the metric adapter on edge cases (F7).
- A fair point-level comparator (F3).
- Full texts of Brancato et al., Fejjari et al. and Dai et al.
- Verified journal scope.
- Administrative information: degree, institution, supervisor, declaration and acknowledgements.

## 8. Novelty risks

- **TC-CAD** (Şahin 2026, Zenodo) is pre-registered, works on ESA-ADB, uses ACI and triage, and reports a negative telecommand result. It overlaps the calibration-and-triage part closely, so the novelty must rest on full-channel scale, event-level selection on all training labels, and the supervised-versus-unsupervised comparison.
- **Brancato et al.** report M1 full 0.902 under a protocol we cannot verify. A reviewer will ask about it.
- **Heydari 2026** (workload-constrained alerting) and Alcarria 2026 (retraining) address operator alarms and adaptation.
- The "first" or "no prior work" claim is **not verifiable** until the literature audit (Stage 4) is done. The thesis currently makes no "first" claim, and it should keep it that way.

## 9. Required experiments

They are listed, with designs and costs, in [REVISION_PLAN.md](REVISION_PLAN.md):

- **Must-have:** F1 and F2.
- **Strongly recommended:** F3 and F5.
- **Recommended:** F4, F6 and F7.

All except the extra TCN seeds and the TCN tuning budget run on CPU in hours.

## 10. Issues fixable by editing alone

- Exposure terminology (§3).
- Title and abstract narrowed to the telemetry question.
- Alignment of `paper/manuscript.tex`, which still carries the uncorrected claims.
- Consistent naming of arms and layers.
- Physics material condensed in the journal version.
- Administrative fields, once you provide the information.

## 11. Readiness verdict

**The thesis is substantially improved and internally consistent, but it is not yet submission-ready as a Q1 journal paper.** The telemetry study is a credible basis for a focused empirical paper. Two things are needed first:

- (i) the H6 decomposition and a temporal check, or else a paper explicitly limited to a protocol-level comparison;
- (ii) the literature audit of the closest work.

The confirmatory intervals should also be recomputed with an official-equivalent bootstrap; no verdict is expected to change. The thesis must correct "four differences" to five.

At present the strongest honest framing is: "a pre-registered, protocol-level comparison showing that alarm-layer selection data and protocol, not model capacity, determined event-level performance on the one mission where calibration data were scarce — with explicit reversals elsewhere".

## 12. Decisions that need your approval before Stage 3

See the questions at the end of the Stage-1 report in the conversation. In short:

- the publication scope;
- whether new analyses may score the already-seen test halves;
- whether shared GPU time may be used;
- whether original-package code may be fixed (the verification suite and the hazard tie-break).
