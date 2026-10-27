# Post-hoc analysis plan V2: decomposition, temporal validation and comparator fairness

**Written:** 2026-10-25, after the confirmatory analysis of `PREREG_TELEMETRY_V1.md` (freeze `09a98f5e160e`, results in `oof_protocol/CONFIRMATORY.json`). The plan was written before any result of the analyses below existed.

**Status:** every analysis here is **post hoc and exploratory**. Where an analysis scores the official test halves, those halves have been used before: in the exploratory study and in the confirmatory pass. This plan fixes the analyses, rules and outputs in advance, so that the post-hoc work cannot be tuned to its own results. It does not make the test halves untouched.

**Frozen code is not modified.** New code lives in `scripts/posthoc_v2/` and imports the frozen modules. Outputs go to `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/`. Large intermediate arrays go to `/tmp` and are deleted afterwards. The SHA-256 of this document is recorded in `posthoc_v2_freeze.json` before any analysis starts. The SHA-256 of each analysis script is recorded in its own result file.

## P1. Decomposition of the H6 contrast (A1 − A0)

A0 and A1 differ in five respects:

- **G**, the grid;
- **F**, the AR(8) fit data;
- **σ**, the residual scale;
- **P**, the calibration pool;
- **D**, the selection data.

The F and σ score levels:

| Level | AR(8) fit | Residual scale |
|---|---|---|
| S0 | A0's model | A0's out-of-sample `es` σ (A0 itself) |
| S½ | A1's model | A0's σ (per-channel rescaling) |
| S1 | A1's model | A1's σ (A1 itself) |

**Cells.** Each cell is a full protocol (selection, refit, one test scoring) evaluated with the official metric. The forward path changes one factor at a time:

| Step | Cell | Factors |
|---|---|---|
| 0 | c0 = A0 | (S0, P_cal, D_cal, G0) |
| 1 (G) | c1 | (S0, P_cal, D_cal, G1) |
| 2 (F) | c2 | (S½, P_cal, D_cal, G1) |
| 3 (σ) | c3 | (S1, P_cal, D_cal, G1) |
| 4 (P) | c4 | (S1, P_half, D_cal, G1) |
| 5 (D) | c5 = A1 | (S1, P_half, D_half, G1) |

The reverse path changes the factors in the opposite order, from A1 back to A0, through r1–r4: G, D, P, σ, F.

**Definitions.**

- **P_half with D_cal.** The pool is the nominal out-of-fold scores of the whole training half, excluding the calibration-window half being evaluated during cross-fitting. The objective uses calibration-window labels only.
- **Selection.** The frozen rule: cross-fitted mean of the official event-wise F0.5, with ties going to the smaller alarm fraction.

**Scope.**

- **Primary:** Mission 1, full target channels, conformal-BH layer.
- **Secondary:** Mission 2, the threshold layer, and the lightweight channel sets.

**Statistics.** Each step reports the test-half difference and a paired block-bootstrap 95 % interval (P3 bootstrap, 2,000 replicates, seed 20261024).

**Validation of the code.** c0 must reproduce A0's official test value, and c5 must reproduce A1's, to within 1e-9. If either fails, P1 is reported as failed.

**Interpretation rule.** A factor is called *material* if its step is ≥ +0.05 on both paths. It is called *order-dependent* if the two paths disagree by more than 0.05. Otherwise it is *not material*.

## P2. Forward-chaining (pseudo-prospective) replication of H6, inside the training halves

**Folds.**

- **Mission 1:** evaluation years 2003, 2004, 2005 and 2006.
- **Mission 2:** evaluation quarters starting 2001-01-01, 2001-04-01 and 2001-07-01.

For each fold, only data before the fold start are used. Normalisation statistics are recomputed from the nominal samples of the fitting part of that history.

**Two protocols per fold,** mirroring A0 and A1:

- **W (window).**
  - Fit AR(8) on the history except its last 12 months (M1) or last 6 months (M2).
  - Estimate the out-of-sample σ on the next 3 months (M1) or 1.5 months (M2).
  - Use the last 9 months (M1) or 4.5 months (M2) before the fold as the pool and the selection data, on grid G0.
- **H (history).**
  - Compute out-of-fold AR(8) scores over the whole history (K = 5, 1-day purge).
  - Fit the final model on all nominal history rows, with in-sample σ.
  - Select by cross-fitting the two halves of the history on grid G1, with the pool being all out-of-fold nominal history scores.

**Evaluation.** Each protocol scores the fold period once, using the official metric on events lying entirely inside the fold period.

**Outputs.**

- per-fold F0.5, recall, false alarms per year and alarm runs per year for H and W;
- the per-fold difference H − W;
- the mean difference, and a paired block bootstrap over the pooled fold periods;
- the number of folds with H > W.

**Interpretation rule.** H6 *replicates prospectively* on a mission if the mean difference is ≥ +0.05 and at least three of four (M1), or two of three (M2), folds favour H. Note: these fold periods are not new data. Their labels entered A1's training-half cross-validation, but they were never used as an evaluation period, and no protocol decision was based on them.

## P3. Exact block bootstrap

A block-decomposable version of the official corrected event-wise F0.5 with:

- per-event interval lists (closed label intervals, closed-open alarm runs);
- matching in continuous time;
- false positives defined as alarm runs that touch no label of any category;
- the true-negative rate in seconds, clipped to 30-day blocks.

**Exactness criterion.** On the full test half, the value must equal the official value to within 1e-9 for every stored conformal-BH and threshold arm in `oof_protocol/test/`.

If the criterion holds, the H4–H6 comparisons are recomputed with the same blocks, the same seed and the same hierarchical resampling. They are reported beside the frozen intervals, and the frozen verdicts remain the confirmatory ones.

## P4. Event-tuned point-level comparator (fair ACI)

ACI with its parameters selected on the **event** objective by the frozen cross-fit rule:

- α ∈ {1e-2, 1e-3, 1e-4};
- γ ∈ {0.05, 0.01, 0.001, 1e-4, 0}, where γ = 0 is a fixed per-channel conformal threshold;
- k ∈ {1, 3, 10, 30, 100};
- 1 h merge.

**Arms:** A1 on Missions 1 and 2, and the A2 seeds on Mission 1. Each is reported beside the BH and threshold layers.

**Interpretation rule.** "ACI is not usable at the event level" may be stated only if event-tuned ACI with γ > 0 stays below the threshold layer by at least 0.05 on every arm.

## P5. The registered secondary anomaly-score arm (PREREG §3)

This is the arm that was registered but not run.

- **Scores:** the anomaly log-odds of A2 and A3.
- **Selection:** the frozen rule with the anomaly-only cross-validated objective.
- **Evaluation:** the anomaly-only official metric on the test half.
- **Comparator:** the same metric for A1's AR(8) layer selected on the anomaly-only objective.

Results are descriptive (no inference) and are labelled as run after the confirmatory analysis.

## P6. Seed extension (Mission 1)

- **New runs:** A2 seeds 5–9 (CPU) and A3 seeds 5–9 (one GPU each), with the frozen code and defaults; selection and test as for the frozen arms.
- **Reporting:** the original 5-seed confirmatory result is unchanged. The 10-seed result is reported beside it, with the same comparisons and the P3 bootstrap.
- **Stopping rule:** if a run fails, it is rerun once with the same seed. A run that fails again is reported as failed. No further seeds are added after results are seen.

## P7. Operational metrics

From the stored test alarm arrays of every arm (conformal-BH):

- detection delay (first alarm time minus event start, clipped at zero) for detected events;
- alarm-run durations;
- alarm runs per detected event;
- false-alarm runs per 1,000 nominal hours;
- detected and missed events by class (anomaly or rare event).

**Statistics:** medians and interquartile ranges. No inference.

## P8. Provenance check

Re-run the frozen `oof_benchmark.select` for `m1_target_ar8` and `m2_target_ar8`, with output redirected to the post-hoc directory, and compare the chosen settings and the full grid with the saved selection files. The check passes if the chosen settings are identical and every grid value agrees to within 1e-9.

## Reporting

Every result of P1–P8, favourable or not, goes into the manuscript or its supplementary material, labelled post hoc. Failed validations (P1 reproduction, P3 exactness, P8 provenance) are reported as failures.
