# Pre-registration: supervised event scoring and training-half alarm selection on ESA-ADB

**Written:** 2026-10-24, before any test-half number of the arms below existed.
**Binding:** an analysis is confirmatory only if it runs after `freeze.json` is written
and the file hashes recorded there still match. Everything earlier is exploratory,
including every result in `outputs/benchmarks/SUMMARY.md` and the training-half
cross-validation numbers that the selection step prints.
**Deviations:** each departure from this document is logged in section 9. The log records
the date, the reason, and whether test results had been seen when the change was made.

## 0. What is already known, and was seen before this document

Every item below is **exploratory** and will be labelled so in the paper.

| Record | Content |
|---|---|
| `outputs/benchmarks/telemetry_esa_adb/results.csv` | the H1–H3 study. The alarm layer is selected on the last 9 months of the training half (9 labelled events on Mission 1). M1 AR(8)+conformal-BH 0.074 (threshold 0.088); M2 AR(8)+conformal-BH 0.962 |
| `outputs/logs/ool_baselines.log` | out-of-limit baselines on the test half: M1 full 0.0007 / 0.0051, M2 full 0.894 / 0.261 |
| `oof_protocol/selection/m{1,2}_target_ar8.json` | training-half CV of the AR(8) arm (M1 0.624, M2 0.919). Training labels only |
| literature | Brancato et al. AST 2026 (M1 full 0.902, protocol not verifiable here), TC-CAD (Şahin 2026), Alcarria et al. 2026 (non-official protocol), DMLR baselines |

The event-model and GBT designs (section 3) were fixed before any of their outputs, including
training-half CV, had been looked at. Their seed-0 runs were started on 2026-10-24 at
15:44–15:53, and this document was written while they were still training.

## 1. Question

On spacecraft telemetry with operator-grade evaluation (ESA-ADB corrected event-wise
F0.5, official split), does a supervised deep event scorer improve on an unsupervised
forecaster when both feed the same event-level conformal alarm layer, selected on the
whole training half? Separately, does that selection protocol by itself fix the Mission 1
failure seen in the exploratory study?

## 2. Data, split and metrics (unchanged from the official benchmark)

- ESA Anomalies Dataset v1.0 (Zenodo 12528696), official grids (30 s / 18 s). We use the
  official train/test split (M1 2007-01-01, M2 2001-10-01) and the full target channel sets
  (58 / 47). Labels are restricted to the target channels and to the test half, as in
  `official_test_labels`.
- **Primary metric:** corrected event-wise F0.5 over anomalies and rare nominal events
  (`anomaly_and_rare/EW_F_0.50`), computed by the vendored official ESA-ADB code.
- **Secondary metrics** (reported, with no confirmatory inference): anomaly-only EW F0.5,
  EW precision and recall, alarming precision, ADTQC, affiliation F0.5, false alarms per
  year, rare-event-only alarm runs, and channel-aware F0.5 on Mission 1.
- **Mission roles:** **Mission 1 is primary.** All confirmatory tests and the Holm family
  are on Mission 1. **Mission 2 is a replication**, reported separately and outside the
  family. The reason is that Mission 2 is close to saturated by a parameter-free
  out-of-limit rule (0.894), so differences there say little. This role assignment was
  made before any arm of section 3 was scored.

## 3. Arms

All arms use the same score-to-alarm layer (`alarms.Calibrator`) with the same grids
(`oof_benchmark.LEVELS`, `KS`) and a fixed 1 h merge.

| Arm | Scores | Selection of level and k | Seeds |
|---|---|---|---|
| **A0** | AR(8) non-conformity, `esa_benchmark` arm `m{m}_target_ar8_s0` | the original rule: cross-fit on the 9-month calibration period | deterministic |
| **A1** | AR(8) non-conformity, out-of-fold over the training half (`oof.py`, K = 5 contiguous blocks, 1-day purge); test scores from an AR(8) fitted on every nominal training sample | cross-fit on the two halves of the whole training half (`oof_benchmark.select`) | deterministic |
| **A2** | XGBoost event model on the same features (`event_gbt.py`), out of fold, same blocks | as A1 | 5 (0–4) |
| **A3** | `EventTCN` (`event_model.py`, defaults of `EventConfig`: all 7 inputs, 3 classes, 1,500 steps), out of fold, same blocks; the test model is trained on all blocks | as A1 | 5 (0–4) |

- **Primary calibrator:** `conformal_bh` (per-channel conformal p-values, Benjamini–Hochberg
  across channels, persistence k, 1 h merge). `threshold` and `conformal_bh_rolling` are
  secondary.
- **Score used:** the `event` log-odds for the primary metric. The `anomaly` log-odds of A2
  and A3 is a secondary arm, scored on the anomaly-only metric and selected on the
  anomaly-only training CV. It is run only if time allows, and is otherwise reported as not
  run.

**Comparators and ablations** (reported, outside the family):

- the point-level ACI comparator, TC-CAD-like (`aci`, and `aci_event` with our event logic),
  on A1 and A3 scores;
- A3 with `--inputs no_residual` and with `--classes 2` (3 seeds each where compute allows);
- lightweight channel sets for A1;
- out-of-limit and GlobalSTD baselines, already scored (exploratory, reported).

## 4. Selection rule (frozen code: `oof_benchmark.select`)

- **Grids:**
  - conformal levels {1e-2, 1e-3, 1e-4, 1e-5};
  - threshold tail fractions {1e-3, 1e-4, 1e-5, 0 (= max)};
  - k ∈ {1, 3, 10, 30, 100};
  - merge = 1 h.
- **Objective:** mean over the two cross-fitted halves of the official EW F0.5 on
  training-half labels. Ties are broken by the smaller alarm fraction.
- The chosen calibrator is refitted on all training-half OOF scores (its nominal samples
  form the conformal calibration pool) and applied once to the test scores.
- No test-half quantity enters any choice.

## 5. Hypotheses (confirmatory, Mission 1, primary metric, primary calibrator)

| ID | Comparison | Claim |
|---|---|---|
| **H4** | A3 − A1 | the supervised deep event scorer beats the unsupervised forecaster under the same alarm layer and selection |
| **H5** | A3 − A2 | the deep event scorer beats gradient-boosted trees on the same features |
| **H6** | A1 − A0 | selecting the alarm layer on the whole training half beats selecting it on the 9-month calibration period (same AR(8) encoder) |

- **Test:** a paired moving-block bootstrap with 30-day blocks; both arms are resampled on
  the same blocks. Seeded arms (A2, A3) are bootstrapped hierarchically: seeds are resampled
  with replacement and the arm value is the mean over the resampled seeds. There are
  B = 2,000 replicates with bootstrap seed 20261024 (`oof_benchmark.compare`). The p-value
  is two-sided, `2 min(P(d ≤ 0), P(d ≥ 0))`, floored at 1/B. The point estimate is the
  difference of the seed-mean **official** metric values. The bootstrap uses the
  block-decomposable re-implementation of the same metric
  (`telemetry/event_bootstrap.py`). It was validated against the official code in the
  exploratory study, and both values are reported.
- **Family:** {H4, H5, H6}, with Holm–Bonferroni at α = 0.05.
- **Minimum effect size of interest:** 0.05 EW F0.5. A hypothesis is **supported** only if
  its Holm-adjusted p < 0.05 **and** the point estimate ≥ 0.05. It is **refuted** if the
  95 % interval lies below +0.05. Otherwise it is **inconclusive**. Every outcome is
  reported.
- **Mission 2 replication:** the same three comparisons, the same test, no correction.
  These are reported as replications. They neither strengthen nor weaken the Mission 1
  conclusions.

## 6. Seeds and stopping rule

There are 5 seeds (0–4) for A2 and A3. This number is fixed and will not be raised or
lowered after results are seen. A run that crashes is rerun with the same seed and code;
the rerun is logged. If compute runs out before 5 seeds finish, the analysis is run on the
seeds that finished, and the result is labelled "underpowered (n = …)".

## 7. What would change the paper's claims

- **H4 refuted or inconclusive:** the paper reports that supervision does not help once the
  alarm layer is selected properly, and that the deep model is not needed.
- **H6 supported and H4 refuted:** the contribution is the selection protocol, not the
  model.
- **H5 refuted:** trees are reported as the recommended event scorer.

## 8. Known threats, stated in advance

- The test half of both missions was used in the exploratory study for other
  configurations. The arms here are new, but the decision to study Mission 1 selection was
  taken after seeing Mission 1 fail.
- OOF forecasters and event models are trained on later blocks as well as earlier ones
  (cross-fitting, not forward chaining). Their training-half scores may therefore be more
  favourable than deployed test scores. The test models are trained on the whole training
  half and are not updated during the test half.
- Conformal exchangeability is violated by serial dependence and drift. Conformal levels
  are tuning knobs here, not guarantees.
- Mission 1 has 4 test event classes that never occur in its training half. A supervised
  scorer cannot learn them.
- Brancato et al.'s Mission 1 number (0.902) could not be reproduced or checked. Their
  out-of-limit reference (0.754) is not what an out-of-limit rule gives under our
  evaluation (0.005). The paper must state this and should not claim to beat or trail
  their number.

## 9. Deviation log

| Date | Deviation | Reason | Test results seen? |
|---|---|---|---|
| 2026-10-24 ~18:40 | The test pass is driven by `scripts/prereg_test_parallel.py` (several processes, one atomic lock per arm) rather than by the sequential loop in `prereg_analysis.py test`. | A sequential pass over about 20 arms would take 5–6 h. Effect on results: None. The same frozen `oof_benchmark.test` scores each arm once. | Yes: A1 on Mission 1 had been scored by the sequential loop |
| 2026-10-24 ~20:05 | Training-half selection for M2 A3 seeds 3–4 and the 6 Mission 1 ablation arms ran after other arms had been test-scored. Four test workers had crashed on the missing selection files, before scoring anything, and their lock files were removed. | These arms finished training after the last selection pass. Effect on results: None. Selection reads training-half labels only, and the frozen code is unchanged (freeze check passes). | Yes, for other arms; selection cannot see test data |
| 2026-10-24 ~19:05 | The sequential `prereg_analysis.py test` process was stopped after it had written A1 (Mission 1). From then on, arms were scored by the parallel driver. | wall time Effect on results: None | Yes (A1 Mission 1) |
| 2026-10-24 ~21:30 | The A0 global alarms for Mission 2 were generated by calling the frozen `oof_benchmark.calperiod_alarms(2, ...)` directly. The first `stats` run crashed on the missing file and was rerun. | `prereg_analysis.py test` creates A0 alarms at the start of each mission's loop, and the sequential loop had been stopped after Mission 1 (row above). Effect on results: None. The function is deterministic, and the bootstrap seed is fixed, so the rerun recomputes identical numbers. | Yes |
| 2026-10-24 ~22:00 | Section 9 filled in from `DEVIATIONS.md`, and the freeze re-issued with `--force` (the previous ID was 09a98f5e160e). | Section 9 sits inside a frozen file and could only be filled in after every confirmatory number existed. Effect on results: none; only this section changed. | Yes |
