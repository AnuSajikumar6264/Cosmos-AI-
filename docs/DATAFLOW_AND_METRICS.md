<!-- Generated from docs/DATAFLOW_AND_METRICS.src.md by paper/journal_v3_twin/tools/render_doc.py; edit the source, not this file. -->

# COSMOS-AI: per-feature architecture, data flow, outputs and metrics

Every number below is a named placeholder filled from the result files by
`paper/journal_v3_twin/tools/render_doc.py`, so this document cannot drift from the runs.
Companion documents: `ARCHITECTURE_AND_NOVELTY.md` (design rationale and novelty claims),
`TARGET_DEFINITIONS.md` (targets, splits, censoring), `RESULTS_AND_ANALYSIS.md` (full results),
`REVIEW_AND_FIXES.md` (review history and disclosures).

---

## 0. The shared state

All three features read and write one record per tracked object (`cosmos/ai/twin/state.py`):

    s_i(t) = ( o_i(t), pi_i, h_i(t), F_i(t), e(t), C_i(t) )

| Component | Contents | Source |
|---|---|---|
| `o_i(t)` | perigee, apogee, inclination at the latest epoch <= t | GCAT canonical orbit; CelesTrak GP for the current orbit |
| `pi_i` | mass, length, diameter, span, shape, mass-uncertainty flag | GCAT |
| `h_i(t)` | operational status (payloads); alarm state where telemetry exists | CelesTrak SATCAT; feature H |
| `F_i(t)` | launch family, breakup parent, sibling statistics known at `t` | GCAT |
| `e(t)` | trailing F10.7 and Ap, observed days only | CelesTrak SW-All |
| `C_i(t)` | conjunction-message sequence when one exists | ESA Kelvins |

The radar cross-section is stored in the record but is **never a model input**: its availability in
the catalogue depends on the outcome (adjusted odds ratio
3.59 for decay within 30 days).

Every module output attached to the state carries its model name, its validation status and its
own uncertainty.

---

## 1. Feature R — atmospheric re-entry risk

Code: `cosmos/ai/lifetime/` (`physics.py`, `models.py`, `fusion.py`, `run.py`).

### 1.1 Input

| Input | Detail |
|---|---|
| Orbit | one dated orbit per object: the GCAT canonical orbit, epoch `t0` |
| Design | mass, length, diameter, span, shape, mass-uncertainty flag |
| Space weather | trailing F10.7 (81 d, 365 d), Ap (81 d), F10.7 trend; observed days before `t0` only |
| Family | sibling statistics restricted to events before `t0` |
| Target | `z = log10 min(T, 5 yr)`, where `T` is the time from `t0` to natural decay |

Cohort: 64,516 objects with perigee below 2000 km, of which
32,970 decayed naturally. Primary population: fragments, rocket stages and
components (payloads excluded, many being operator-controlled).

Splits per origin `O` (objects with a known five-year outcome):

| Split | Rule | Origin 2016 | Origin 2011 | Used for |
|---|---|---|---|---|
| train | `t0 < O - 7 yr` | 25,334 | 20,777 | fitting |
| tune-A | `O - 7 <= t0 < O - 5 yr`, half the launch families | 1,740 | 248 | early stopping, hyperparameters, expert selection |
| tune-B | same cohort, other half | 1,560 | 332 | conformal and probability calibration |
| test | `O <= t0 <` test end | 2,978 | 2,667 | scoring only |

### 1.2 Processing — path A, the physics expert

    orbit + A/m sample + NRLMSIS 2.1 density at perigee
      -> orbit-averaged drag equations for da/dt and de/dt
      -> step-limited integration until perigee < 120 km (or the 5-yr cap)
      -> 32 Monte-Carlo trajectories
      -> predictive distribution of the capped time to re-entry

- `e <= 0.05`: King-Hele Bessel-function series.
- `e > 0.05`: the same orbit averages integrated exactly over the eccentric anomaly (257-node
  trapezoid). The truncated series raises the perigee above about `e = 0.3`, which is unphysical,
  so the exact form is required.
- Step rule: a step may not drop the perigee by more than 2 % of its height above 120 km (floor
  1 km) and may not exceed 10 days.
- Area-to-mass, recorded per object as `am_source`:
  - `design` — the GCAT design value, log-normal with 0.25 dex;
  - `class_prior_design` — for intact payloads and rocket stages without design values, the
    training distribution of design values of their class;
  - `class_prior_sbm` — for fragments and components, the NASA Standard Breakup Model with a
    characteristic length drawn from the training distribution of radar-derived sizes of their class.
- Each object has its own RNG stream seeded from its catalogue identifier.

### 1.3 Processing — path B, the learned expert

Quantile regression of `z` on 53 features in seven groups (`models.NUMERIC`, `models.CATEGORICAL`):

| Group | Features |
|---|---|
| `orbit` | perigee, apogee, inclination, eccentricity, semi-major axis, mean altitude, uncertainty flag |
| `age` | age at `t0` |
| `design` | log mass, length, diameter, span, log area, log A/m, mass-uncertainty flag |
| `space_weather` | F10.7 (81 d, 365 d), Ap (81 d), F10.7 trend |
| `family` | launch- and breakup-family sibling counts, decayed counts, decayed fractions, mean log-lifetime (8 features) |
| `physics` | five summaries of path A: q10, q50, q90, spread, P(at the cap) |
| `class` | object type, parent type, shape (one-hot) |

Candidates are learners published 2022-2026: TabM (ICLR 2025), TabICLv2 (2026), Kumo-Tabular
(2026). The candidate with the lowest tune-A pinball loss becomes the expert
(selected: tabm at 2016, kumo at 2011). XGBoost, LightGBM, CatBoost, a
plain quantile MLP and XGBoost-AFT are reference baselines and are not eligible. Each tuned
learner gets 12 random configurations and five seeds; the in-context models use their defaults.

### 1.4 Processing — fusion and calibration

Reliability-gated quantile averaging:

    q_tau(i) = g(r_i) q_tau^phys(i) + (1 - g(r_i)) q_tau^ml(i),    g(r) = sigmoid(w'r + b)

The gate input `r_i` describes **the physics expert's own reliability**: the A/m source, its
Monte-Carlo spread, eccentricity, perigee band, object class and its survival probability.
`(w, b)` minimise the pinball loss on out-of-fold predictions of the training period. A convex
combination of two monotone quantile functions is a monotone quantile function, so the result is
always a valid predictive distribution. Mean physics weight on test:
0.11 at 2016, 0.12
at 2011.

Mondrian conformal calibration on tune-B (disjoint from fitting and gating): score
`S = max(q_lo - z, z - q_hi)`; in group `k` (object class x perigee band, falling back to class,
then pooled) the correction `Q_k` is the `ceil((1-alpha)(n_k+1))`-th smallest score, giving
`[q_lo - Q_k, q_hi + Q_k]`. Horizon probabilities `P(T <= H)` are read off the fused quantile
function and calibrated by isotonic regression on the same split.

### 1.5 Output

| Output | Form |
|---|---|
| Predictive distribution | seven quantiles (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) of `log10` days |
| Intervals | Mondrian conformal 80 % and 90 % |
| Horizon probabilities | isotonic-calibrated `P(T <= H)` for H = 30 d, 6 mo, 1 yr, 5 yr |
| Provenance | A/m source, gate weight, calibration group |

### 1.6 The extra step for objects already in orbit

The module is fitted at canonical-orbit epochs, but a twin forecasts objects that have survived
for years, which selects low area-to-mass ratios. `physics.history_posterior` therefore conditions
A/m on each object's catalogue history:

1. 128 prior samples are propagated from the canonical orbit to the as-of date with the realised
   trailing space weather;
2. samples that would already have re-entered get weight zero;
3. the rest are weighted by a Gaussian likelihood of the current perigee and apogee, with
   `sigma = 10 km + 0.3 |delta|`;
4. 32 posterior samples are drawn by systematic resampling (fallbacks: survival only if the
   effective sample size is below 4; the eight longest survivors if none survives).

The physics expert then forecasts from the **current** orbit with these samples.

---

## 2. Feature D — orbital debris analysis

### 2.1 Family evidence (feeds R)

For object `i` at time `t`, from `cosmos/ai/lifetime/catalogue.py`: the number of siblings
catalogued by `t`, the number and fraction that decayed naturally before `t`, and their mean
log-lifetime, for the launch family and (for fragments) the breakup family. Only events strictly
before `t` enter, so the feature is causal. This is coupling C1.

### 2.2 Conjunction risk

Code: `cosmos/ai/conjunction/` (`data.py`, `run.py`).

**Input.** ESA Kelvins conjunction data messages (CDMs), anonymised, 2015-2019. Only messages
released at least two days before closest approach are inputs; the target is the risk
(`log10` collision probability) of the final message. Training events are filtered to mirror the
test inputs, leaving 8,293 training events
(66 high-risk, i.e. final risk >= -6) and the official
2,167 test events (150 high-risk).

**Processing.**

    CDM sequence -> one feature row per event
      (latest risk; risk min/max/mean/first; risk change; 3-point slope; last step;
       miss-distance log-ratio; covariance sigma log-ratios; message count; span)
      -> persistence expert + learned experts
      -> reliability gate (same form as R)
      -> Mondrian conformal calibration (groups: latest-risk bin)
      -> isotonic-calibrated P(final risk >= -6)
      -> decision rule for the official point metric

The **persistence expert** is the latest risk plus the training distribution of
(final - latest) changes within the same latest-risk bin, i.e. an honest persistence forecast with
uncertainty. Learner selection, the gate, both calibrations and the decision threshold are all
driven by nested out-of-fold predictions over five event-hash folds. Selected expert:
kumo; mean gate weight on persistence 0.0002, so the gated
model is in effect the learned expert.

Quantiles within `1e-4` of the risk floor (-30) are snapped to it exactly, because most final
risks sit at the floor and float32 spacing there is about `2e-6`.

**Output.** Quantiles of the final risk, conformal 80/90 % intervals, calibrated
`P(final risk >= -6)`, and a point forecast for the official metric: predict the latest risk when
`P(high)` exceeds an out-of-fold-chosen threshold, otherwise -6.001.

---

## 3. Feature H — satellite health

Code: `cosmos/ai/telemetry/smap_msl.py` (public cross-check) and the companion ESA-ADB study.

**Input.** Per-channel spacecraft telemetry. NASA SMAP/MSL: 81 channels with
68 SMAP and 36 MSL labelled anomaly
sequences (the label file lists channel P-2 twice; the rows are merged). ESA-ADB Missions 1 and 2
are used in the companion pre-registered study.

**Processing.**

    forecaster fitted on nominal data -> non-conformity score s_c(t)
      -> conformal p-value against that channel's own calibration pool:
         p_c(t) = (1 + #{s' in C_c : s' >= s_c(t)}) / (|C_c| + 1)
      -> Benjamini-Hochberg across channels at each step
      -> persistence: alarm from the K-th consecutive step with p <= alpha_eff
      -> merge into alarm runs

Settings fixed in advance: window 30 steps, level
`alpha = 0.001`, persistence 3 consecutive steps, and
`alpha_eff = max(alpha, 1/(n_cal + 1))` per channel. Calibration scores are out-of-fold in five
contiguous blocks with a 30-step purge on each side, so no step is scored by a model that saw it.

Detectors compared: Chronos-2 (`amazon/chronos-2`, zero-shot, context
512, horizon 8, scored by the absolute
residual of the forecast median), ridge autoregression, PCA reconstruction, nearest-neighbour
window distance, the published telemanom scores, a random-score floor and an always-on reference.

**Output.** A causal alarm state per channel written to the twin, plus the operational status of
catalogue objects.

---

## 4. How the features connect

| Coupling | Flow | Definition | Tested on real data |
|---|---|---|---|
| C1 | D -> R | family statistics enter R's feature matrix | yes: ablation at both origins |
| C2 | R -> D | `E[N(t)] = sum_{i: t0_i <= t} S_i(t - t0_i)`, `S_i` the fused survival function, read raw and isotonic-calibrated | yes: 160 cloud-years in 27 clouds with >= 30 fragments |
| C3 | H -> R | operational status sets the regime; in the operator-controlled regime the drag forecast is withheld and the previous-window payload base rate used | yes: 13,794 payloads, 182 decays in 72 days |
| T1 | state -> R | history-conditioned physics for fragments, stages and components; C3 policies for payloads | yes: 13,186 uncontrolled objects plus payloads, 30,826 in total |
| C4 | H x D | a high-risk conjunction for a payload with an active alarm or non-operational status is escalated | **no**: no public dataset pairs the signals — demonstration only |

---

## 5. Metrics and why each is used

| Metric | What it measures | Applies to |
|---|---|---|
| Mean pinball loss over seven quantiles | the quality of the whole predictive distribution; the primary metric, because a single predicted date is not decision-useful | R, D |
| MAE and RMSE of the median (`log10` days) | point accuracy on a scale where errors are multiplicative | R |
| Median absolute error in days | point accuracy in operational units, for objects that decayed within five years | R |
| Coverage of the 80 % and 90 % intervals | whether the stated uncertainty is honest | R, D |
| Mean interval width, Winkler score | sharpness, and sharpness and coverage together so that narrow-and-wrong is penalised | R |
| Harrell's C | ranking ability over all test objects, including censored ones | R |
| AUC-PR and AUC-ROC | discrimination at a horizon; AUC-PR because decays are rare. Computed on the **raw** probabilities, because isotonic calibration ties scores | R, D, H |
| Brier score, ECE | probability calibration, computed on the isotonic-calibrated values | R, D |
| Expected/observed ratio | absolute bias of a probability forecast on real objects | T1, C3 |
| `L = MSE_HR / F2` | the ESA Kelvins official metric: squared error on true high-risk events divided by the F2 of the high-risk classification, low-risk predictions clipped at -6.001 | D |
| Corrected event-wise F0.5 (ESA-ADB) | alarm quality that charges alarm time spent on nominal data, so an always-on alarm scores 0 | H |
| Event precision, recall, F1; detection delay | operational alarm behaviour | H |
| Swap and late surrogates | label-free nulls: keep a detector's alarm positions but reassign them across channels (swap), or place the same amount of alarm time at the end (late). These separate channel-specific detection from benchmark artefacts | H |
| Mean absolute error as a fraction of the cloud | C2 accuracy in units an operator cares about | C2 |
| Cluster bootstrap, 2,000 replicates | confidence intervals resampling launch families (or launches, or clouds), paired for model differences, because objects from one launch are not independent | all |

Fixed quantile grid: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95.

---

## 6. Results

### 6.1 Re-entry, two forward-in-time test periods

| Model | 2016-2021 pinball | MAE | cov90 | C | 2011-2015 pinball | MAE | cov90 | C |
|---|---|---|---|---|---|---|---|---|
| Physics expert | 0.0892 | 0.230 | 0.876 | 0.739 | 0.0527 | 0.148 | 0.974 | 0.766 |
| XGBoost (baseline) | 0.0576 | 0.190 | 0.921 | 0.888 | 0.0648 | 0.231 | 0.894 | 0.807 |
| TabM (2025) | 0.0593 | 0.192 | 0.877 | 0.861 | 0.0994 | 0.354 | 0.918 | 0.783 |
| TabICLv2 (2026) | 0.0499 | 0.166 | 0.962 | 0.894 | 0.0596 | 0.160 | 0.919 | 0.844 |
| Kumo-Tabular (2026) | 0.0508 | 0.162 | 0.974 | 0.899 | 0.0616 | 0.163 | 0.966 | 0.845 |
| Reliability-gated | 0.0604 | 0.193 | 0.874 | 0.860 | 0.0583 | 0.157 | 0.959 | 0.848 |

Reading, with paired launch-family cluster bootstraps:

- **2016-2021:** the two foundation models are significantly better than the physics expert, the
  gated model, the selected TabM and the best tree baseline. TabICLv2 minus XGBoost is
  -0.0076 (-0.0121 to
  -0.0036).
- **2011-2015:** the physics expert has the lowest point estimate and is statistically tied with
  TabICLv2 and the gated model. The one significant difference among them: Kumo-Tabular minus the
  gated model is +0.0033
  (+0.0007 to +0.0055).
- Selection on the older tuning cohort chose a significantly worse learner at 2016.
- The gate never beat the best single expert.

Re-entry within 30 days, origin 2016 (AUC-PR on raw scores, Brier after calibration):

| Model | AUC-PR | Brier |
|---|---|---|
| Kumo-Tabular | 0.883 | 0.036 |
| TabICLv2 | 0.874 | 0.035 |
| Gated model | 0.829 | 0.051 |
| Physics expert | 0.719 | 0.051 |

### 6.2 The leaking covariate

Availability of the radar cross-section is strongly associated with early decay (adjusted odds
ratio 3.59, cluster-robust 95 % CI
2.09-6.17).
Adding it changes the test pinball loss by amounts within noise at both origins, but at origin
2016 it drops 90 % coverage from 87.7 % to
76.8 %. It is excluded as a predictor and kept only as a
labelled demonstration.

### 6.3 T1 — the twin on objects in orbit (72 days, 64 observed decays)

| Variant | Expected | Expected/observed | 95 % CI | Brier | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|
| History-conditioned physics | 87 | 1.37 | 1.20-1.63 | 0.00217 | 0.737 | 0.991 |
| Physics, class priors | 233 | 3.64 | 2.90-4.68 | 0.00625 | 0.601 | 0.988 |
| Lifetime pipeline | 610 | 9.5 | 6.9-13.0 | 0.01344 | 0.616 | 0.995 |

All three variants rank decaying objects almost equally well (AUC-ROC within 0.01). What differs
is calibration, which is the point of the state-estimation step. This window had been analysed
earlier in the project, so T1 is an as-of-date test, not a blind one.

### 6.4 Conjunction risk, official test set

| Model | Pinball | AUC-PR of P(high) | F2 | L |
|---|---|---|---|---|
| Persistence | 1.045 | 0.664 | 0.000 | ∞ |
| XGBoost (baseline) | 0.855 | 0.606 | 0.000 | ∞ |
| TabICLv2 | 0.692 | 0.665 | 0.057 | 30.059 |
| Kumo-Tabular | 0.653 | 0.733 | 0.168 | 9.892 |
| Latest-risk baseline | — | — | — | 0.694 |

`L` is infinite whenever `F2 = 0`: every median forecast falls below the high-risk threshold, so
the official metric divides by zero. Kumo-Tabular is significantly the best probabilistic
forecast (minus TabICLv2 -0.039, -0.053 to
-0.025), and still no model beats the latest-risk point forecast on the
official metric. Its conformal 90 % interval covers 94.2 % of all test events but
only 48.0 % of the high-risk ones.

### 6.5 Health, NASA SMAP/MSL

| Detector | SMAP AUC-PR | SMAP F0.5 | SMAP late ref. | MSL AUC-PR | MSL F0.5 | MSL late ref. |
|---|---|---|---|---|---|---|
| telemanom (published) | 0.576 | — | — | 0.473 | — | — |
| Chronos-2, zero-shot | 0.343 | 0.170 | 0.197 | 0.367 | 0.035 | 0.224 |
| ridge autoregression | 0.355 | 0.240 | 0.413 | 0.329 | 0.252 | 0.297 |
| PCA reconstruction | 0.322 | 0.249 | 0.306 | 0.328 | 0.212 | 0.314 |
| nearest-neighbour window | 0.341 | 0.114 | 0.272 | 0.364 | 0.334 | 0.281 |
| random scores | 0.129 | 0.000 | — | 0.141 | 0.000 | — |

The always-on reference reaches an uncorrected event F1 of 0.548
(SMAP) and 0.537 (MSL), higher than every detector, while its
corrected F0.5 is 0. The "late ref." column is the key caution: a label-free reference that keeps
each detector's amount of alarm time but places it as one block at the end of the series beats the
detector itself in seven of the eight detector-spacecraft pairs, because
93.7 % of labelled steps lie in the second half of their
series. The zero-shot foundation model does not improve on simple detectors here.

### 6.6 Couplings

| Coupling | Result |
|---|---|
| C1 family evidence | helps the selected learner at 2016 (+0.0056, +0.0015 to +0.0101); at 2011 the interval includes zero |
| C2 cloud survival | best readings are accurate to about a tenth of a cloud, but with 27 clouds the experts cannot be ranked: calibrated TabICLv2 0.095 (0.068-0.123) against raw physics 0.129 (0.078-0.191) |
| C3 operational status | drag physics over-predicts for operational payloads (mean predicted 0.039 against observed 0.012) and under-predicts for non-operational ones (0.076 against 0.106). The status-aware policy lowers the Brier score by -0.0054 (-0.0120 to +0.0006): not significant |
| T1 state estimation | the largest effect in the study, section 6.3 |
| C4 | demonstration only, untested |

### 6.7 The dated forecast

As of 2026-10-27 the twin holds 31,396 objects with perigee below 2000 km:
13,787 in natural decay,
16,386 operator-controlled payloads and
1,223 payloads without a reported status. Expected natural-decay
re-entries in the next 30 days: 24.5. The record is
fixed by its SHA-256 hash `19a1fede1fd4871f380ff802f285d2ad03732f6cd3179790e67336c7d202a6b5` so that a later catalogue snapshot can score it
without the possibility of revision. This is the only fully blind test in the project.

---

## 7. Honest limits on the above

- The test sets were scored more than once during development, and the eligible learner pool was
  changed after the first run's test results were known.
- T1 and C3 use a window that had been analysed earlier in the project, so neither is blind.
- Several comparisons were added after results were seen; their intervals are nominal 95 %
  intervals without correction for multiplicity.
- C4 has no test.
- The health module is the least validated of the three: its alarm protocol helped on one ESA-ADB
  mission and reversed on the other.

All of this is recorded in the manuscript's "Deviations and prior use of test data" paragraph and
in `REVIEW_AND_FIXES.md`.
