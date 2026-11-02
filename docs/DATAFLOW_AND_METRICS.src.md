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
{{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.2f}} for decay within 30 days).

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

Cohort: {{data.cohort.rows_out:,}} objects with perigee below 2000 km, of which
{{data.cohort.events:,}} decayed naturally. Primary population: fragments, rocket stages and
components (payloads excluded, many being operator-controlled).

Splits per origin `O` (objects with a known five-year outcome):

| Split | Rule | Origin 2016 | Origin 2011 | Used for |
|---|---|---|---|---|
| train | `t0 < O - 7 yr` | {{re.o2016.known5.train:,}} | {{re.o2011.known5.train:,}} | fitting |
| tune-A | `O - 7 <= t0 < O - 5 yr`, half the launch families | {{re.o2016.known5.tuneA:,}} | {{re.o2011.known5.tuneA:,}} | early stopping, hyperparameters, expert selection |
| tune-B | same cohort, other half | {{re.o2016.known5.tuneB:,}} | {{re.o2011.known5.tuneB:,}} | conformal and probability calibration |
| test | `O <= t0 <` test end | {{re.o2016.known5.test:,}} | {{re.o2011.known5.test:,}} | scoring only |

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
(selected: {{re.o2016.best}} at 2016, {{re.o2011.best}} at 2011). XGBoost, LightGBM, CatBoost, a
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
{{re.o2016.gate.cosmos_gated.weight_mean:.2f}} at 2016, {{re.o2011.gate.cosmos_gated.weight_mean:.2f}}
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
test inputs, leaving {{cac.data.train_events_kept:,}} training events
({{cac.data.train_high_risk}} high-risk, i.e. final risk >= -6) and the official
{{cac.data.test_events:,}} test events ({{cac.data.test_high_risk}} high-risk).

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
{{cac.best}}; mean gate weight on persistence {{cac.gate_weight_cosmos_gated:.4f}}, so the gated
model is in effect the learned expert.

Quantiles within `1e-4` of the risk floor (-30) are snapped to it exactly, because most final
risks sit at the floor and float32 spacing there is about `2e-6`.

**Output.** Quantiles of the final risk, conformal 80/90 % intervals, calibrated
`P(final risk >= -6)`, and a point forecast for the official metric: predict the latest risk when
`P(high)` exceeds an out-of-fold-chosen threshold, otherwise -6.001.

---

## 3. Feature H — satellite health

Code: `cosmos/ai/telemetry/smap_msl.py` (public cross-check) and the companion ESA-ADB study.

**Input.** Per-channel spacecraft telemetry. NASA SMAP/MSL: {{smap.config.channels}} channels with
{{smap.SMAP.ar_ridge.events:.0f}} SMAP and {{smap.MSL.ar_ridge.events:.0f}} MSL labelled anomaly
sequences (the label file lists channel P-2 twice; the rows are merged). ESA-ADB Missions 1 and 2
are used in the companion pre-registered study.

**Processing.**

    forecaster fitted on nominal data -> non-conformity score s_c(t)
      -> conformal p-value against that channel's own calibration pool:
         p_c(t) = (1 + #{s' in C_c : s' >= s_c(t)}) / (|C_c| + 1)
      -> Benjamini-Hochberg across channels at each step
      -> persistence: alarm from the K-th consecutive step with p <= alpha_eff
      -> merge into alarm runs

Settings fixed in advance: window {{smap.config.window}} steps, level
`alpha = {{smap.config.alpha}}`, persistence {{smap.config.persistence}} consecutive steps, and
`alpha_eff = max(alpha, 1/(n_cal + 1))` per channel. Calibration scores are out-of-fold in five
contiguous blocks with a 30-step purge on each side, so no step is scored by a model that saw it.

Detectors compared: Chronos-2 (`{{smap.config.chronos2.model}}`, zero-shot, context
{{smap.config.chronos2.context}}, horizon {{smap.config.chronos2.horizon}}, scored by the absolute
residual of the forecast median), ridge autoregression, PCA reconstruction, nearest-neighbour
window distance, the published telemanom scores, a random-score floor and an always-on reference.

**Output.** A causal alarm state per channel written to the twin, plus the operational status of
catalogue objects.

---

## 4. How the features connect

| Coupling | Flow | Definition | Tested on real data |
|---|---|---|---|
| C1 | D -> R | family statistics enter R's feature matrix | yes: ablation at both origins |
| C2 | R -> D | `E[N(t)] = sum_{i: t0_i <= t} S_i(t - t0_i)`, `S_i` the fused survival function, read raw and isotonic-calibrated | yes: {{c2.n_cloud_years}} cloud-years in {{c2.n_clouds}} clouds with >= 30 fragments |
| C3 | H -> R | operational status sets the regime; in the operator-controlled regime the drag forecast is withheld and the previous-window payload base rate used | yes: {{c3.n:,}} payloads, {{c3.events}} decays in {{c3.horizon_days:.0f}} days |
| T1 | state -> R | history-conditioned physics for fragments, stages and components; C3 policies for payloads | yes: {{t1.uncontrolled:,}} uncontrolled objects plus payloads, {{t1.objects:,}} in total |
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
| Physics expert | {{re.o2016.physics_only.pinball:.4f}} | {{re.o2016.physics_only.mae_log10:.3f}} | {{re.o2016.physics_only.cov90:.3f}} | {{re.o2016.physics_only.c_index:.3f}} | {{re.o2011.physics_only.pinball:.4f}} | {{re.o2011.physics_only.mae_log10:.3f}} | {{re.o2011.physics_only.cov90:.3f}} | {{re.o2011.physics_only.c_index:.3f}} |
| XGBoost (baseline) | {{re.o2016.xgboost.pinball:.4f}} | {{re.o2016.xgboost.mae_log10:.3f}} | {{re.o2016.xgboost.cov90:.3f}} | {{re.o2016.xgboost.c_index:.3f}} | {{re.o2011.xgboost.pinball:.4f}} | {{re.o2011.xgboost.mae_log10:.3f}} | {{re.o2011.xgboost.cov90:.3f}} | {{re.o2011.xgboost.c_index:.3f}} |
| TabM (2025) | {{re.o2016.tabm.pinball:.4f}} | {{re.o2016.tabm.mae_log10:.3f}} | {{re.o2016.tabm.cov90:.3f}} | {{re.o2016.tabm.c_index:.3f}} | {{re.o2011.tabm.pinball:.4f}} | {{re.o2011.tabm.mae_log10:.3f}} | {{re.o2011.tabm.cov90:.3f}} | {{re.o2011.tabm.c_index:.3f}} |
| TabICLv2 (2026) | {{re.o2016.tabicl.pinball:.4f}} | {{re.o2016.tabicl.mae_log10:.3f}} | {{re.o2016.tabicl.cov90:.3f}} | {{re.o2016.tabicl.c_index:.3f}} | {{re.o2011.tabicl.pinball:.4f}} | {{re.o2011.tabicl.mae_log10:.3f}} | {{re.o2011.tabicl.cov90:.3f}} | {{re.o2011.tabicl.c_index:.3f}} |
| Kumo-Tabular (2026) | {{re.o2016.kumo.pinball:.4f}} | {{re.o2016.kumo.mae_log10:.3f}} | {{re.o2016.kumo.cov90:.3f}} | {{re.o2016.kumo.c_index:.3f}} | {{re.o2011.kumo.pinball:.4f}} | {{re.o2011.kumo.mae_log10:.3f}} | {{re.o2011.kumo.cov90:.3f}} | {{re.o2011.kumo.c_index:.3f}} |
| Reliability-gated | {{re.o2016.cosmos_gated.pinball:.4f}} | {{re.o2016.cosmos_gated.mae_log10:.3f}} | {{re.o2016.cosmos_gated.cov90:.3f}} | {{re.o2016.cosmos_gated.c_index:.3f}} | {{re.o2011.cosmos_gated.pinball:.4f}} | {{re.o2011.cosmos_gated.mae_log10:.3f}} | {{re.o2011.cosmos_gated.cov90:.3f}} | {{re.o2011.cosmos_gated.c_index:.3f}} |

Reading, with paired launch-family cluster bootstraps:

- **2016-2021:** the two foundation models are significantly better than the physics expert, the
  gated model, the selected TabM and the best tree baseline. TabICLv2 minus XGBoost is
  {{re.o2016.pair.tabicl.xgboost.diff:+.4f}} ({{re.o2016.pair.tabicl.xgboost.lo:+.4f}} to
  {{re.o2016.pair.tabicl.xgboost.hi:+.4f}}).
- **2011-2015:** the physics expert has the lowest point estimate and is statistically tied with
  TabICLv2 and the gated model. The one significant difference among them: Kumo-Tabular minus the
  gated model is {{re.o2011.pair.kumo.cosmos_gated.diff:+.4f}}
  ({{re.o2011.pair.kumo.cosmos_gated.lo:+.4f}} to {{re.o2011.pair.kumo.cosmos_gated.hi:+.4f}}).
- Selection on the older tuning cohort chose a significantly worse learner at 2016.
- The gate never beat the best single expert.

Re-entry within 30 days, origin 2016 (AUC-PR on raw scores, Brier after calibration):

| Model | AUC-PR | Brier |
|---|---|---|
| Kumo-Tabular | {{re.o2016.h.kumo.30d.auc_pr:.3f}} | {{re.o2016.h.kumo.30d.brier:.3f}} |
| TabICLv2 | {{re.o2016.h.tabicl.30d.auc_pr:.3f}} | {{re.o2016.h.tabicl.30d.brier:.3f}} |
| Gated model | {{re.o2016.h.cosmos_gated.30d.auc_pr:.3f}} | {{re.o2016.h.cosmos_gated.30d.brier:.3f}} |
| Physics expert | {{re.o2016.h.physics_only.30d.auc_pr:.3f}} | {{re.o2016.h.physics_only.30d.brier:.3f}} |

### 6.2 The leaking covariate

Availability of the radar cross-section is strongly associated with early decay (adjusted odds
ratio {{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.2f}}, cluster-robust 95 % CI
{{leak.o2016.train.adjusted_or_rcs_missing.ci95.0:.2f}}-{{leak.o2016.train.adjusted_or_rcs_missing.ci95.1:.2f}}).
Adding it changes the test pinball loss by amounts within noise at both origins, but at origin
2016 it drops 90 % coverage from {{re.o2016.BEST.cov90:pct}} to
{{re.o2016.BEST_plus_rcs_2026.cov90:pct}}. It is excluded as a predictor and kept only as a
labelled demonstration.

### 6.3 T1 — the twin on objects in orbit (72 days, {{t1.by_horizon.72d.uncontrolled.physics_history.events}} observed decays)

| Variant | Expected | Expected/observed | 95 % CI | Brier | AUC-PR | AUC-ROC |
|---|---|---|---|---|---|---|
| History-conditioned physics | {{t1.by_horizon.72d.uncontrolled.physics_history.expected:.0f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ratio:.2f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ci95.0:.2f}}-{{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ci95.1:.2f}} | {{t1.by_horizon.72d.uncontrolled.physics_history.brier:.5f}} | {{t1.by_horizon.72d.uncontrolled.physics_history.auc_pr:.3f}} | {{t1.by_horizon.72d.uncontrolled.physics_history.auc_roc:.3f}} |
| Physics, class priors | {{t1.by_horizon.72d.uncontrolled.physics_prior.expected:.0f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ratio:.2f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ci95.0:.2f}}-{{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ci95.1:.2f}} | {{t1.by_horizon.72d.uncontrolled.physics_prior.brier:.5f}} | {{t1.by_horizon.72d.uncontrolled.physics_prior.auc_pr:.3f}} | {{t1.by_horizon.72d.uncontrolled.physics_prior.auc_roc:.3f}} |
| Lifetime pipeline | {{t1.by_horizon.72d.uncontrolled.lifetime.expected:.0f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ci95.0:.1f}}-{{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ci95.1:.1f}} | {{t1.by_horizon.72d.uncontrolled.lifetime.brier:.5f}} | {{t1.by_horizon.72d.uncontrolled.lifetime.auc_pr:.3f}} | {{t1.by_horizon.72d.uncontrolled.lifetime.auc_roc:.3f}} |

All three variants rank decaying objects almost equally well (AUC-ROC within 0.01). What differs
is calibration, which is the point of the state-estimation step. This window had been analysed
earlier in the project, so T1 is an as-of-date test, not a blind one.

### 6.4 Conjunction risk, official test set

| Model | Pinball | AUC-PR of P(high) | F2 | L |
|---|---|---|---|---|
| Persistence | {{cac.persistence.pinball:.3f}} | {{cac.persistence.auc_pr_high:.3f}} | {{cac.persistence.F2:.3f}} | {{cac.persistence.L_clip:.3f}} |
| XGBoost (baseline) | {{cac.xgboost.pinball:.3f}} | {{cac.xgboost.auc_pr_high:.3f}} | {{cac.xgboost.F2:.3f}} | {{cac.xgboost.L_clip:.3f}} |
| TabICLv2 | {{cac.tabicl.pinball:.3f}} | {{cac.tabicl.auc_pr_high:.3f}} | {{cac.tabicl.F2:.3f}} | {{cac.tabicl.L_clip:.3f}} |
| Kumo-Tabular | {{cac.kumo.pinball:.3f}} | {{cac.kumo.auc_pr_high:.3f}} | {{cac.kumo.F2:.3f}} | {{cac.kumo.L_clip:.3f}} |
| Latest-risk baseline | — | — | — | {{cac.latest_risk_point.L_clip:.3f}} |

`L` is infinite whenever `F2 = 0`: every median forecast falls below the high-risk threshold, so
the official metric divides by zero. Kumo-Tabular is significantly the best probabilistic
forecast (minus TabICLv2 {{cacp.kumo.tabicl.diff:+.3f}}, {{cacp.kumo.tabicl.lo:+.3f}} to
{{cacp.kumo.tabicl.hi:+.3f}}), and still no model beats the latest-risk point forecast on the
official metric. Its conformal 90 % interval covers {{cac.kumo.cov90:pct}} of all test events but
only {{cac.kumo.cov90_high:pct}} of the high-risk ones.

### 6.5 Health, NASA SMAP/MSL

| Detector | SMAP AUC-PR | SMAP F0.5 | SMAP late ref. | MSL AUC-PR | MSL F0.5 | MSL late ref. |
|---|---|---|---|---|---|---|
| telemanom (published) | {{smap.SMAP.telemanom.pointwise_auc_pr:.3f}} | — | — | {{smap.MSL.telemanom.pointwise_auc_pr:.3f}} | — | — |
| Chronos-2, zero-shot | {{smap.SMAP.chronos2.pointwise_auc_pr:.3f}} | {{smap.SMAP.chronos2.esa_ew_f0.5:.3f}} | {{smap.SMAP.chronos2.esa_ew_f0.5_late:.3f}} | {{smap.MSL.chronos2.pointwise_auc_pr:.3f}} | {{smap.MSL.chronos2.esa_ew_f0.5:.3f}} | {{smap.MSL.chronos2.esa_ew_f0.5_late:.3f}} |
| ridge autoregression | {{smap.SMAP.ar_ridge.pointwise_auc_pr:.3f}} | {{smap.SMAP.ar_ridge.esa_ew_f0.5:.3f}} | {{smap.SMAP.ar_ridge.esa_ew_f0.5_late:.3f}} | {{smap.MSL.ar_ridge.pointwise_auc_pr:.3f}} | {{smap.MSL.ar_ridge.esa_ew_f0.5:.3f}} | {{smap.MSL.ar_ridge.esa_ew_f0.5_late:.3f}} |
| PCA reconstruction | {{smap.SMAP.pca.pointwise_auc_pr:.3f}} | {{smap.SMAP.pca.esa_ew_f0.5:.3f}} | {{smap.SMAP.pca.esa_ew_f0.5_late:.3f}} | {{smap.MSL.pca.pointwise_auc_pr:.3f}} | {{smap.MSL.pca.esa_ew_f0.5:.3f}} | {{smap.MSL.pca.esa_ew_f0.5_late:.3f}} |
| nearest-neighbour window | {{smap.SMAP.knn.pointwise_auc_pr:.3f}} | {{smap.SMAP.knn.esa_ew_f0.5:.3f}} | {{smap.SMAP.knn.esa_ew_f0.5_late:.3f}} | {{smap.MSL.knn.pointwise_auc_pr:.3f}} | {{smap.MSL.knn.esa_ew_f0.5:.3f}} | {{smap.MSL.knn.esa_ew_f0.5_late:.3f}} |
| random scores | {{smap.SMAP.random.pointwise_auc_pr:.3f}} | {{smap.SMAP.random.esa_ew_f0.5:.3f}} | — | {{smap.MSL.random.pointwise_auc_pr:.3f}} | {{smap.MSL.random.esa_ew_f0.5:.3f}} | — |

The always-on reference reaches an uncorrected event F1 of {{smap.SMAP.always_on.event_f1:.3f}}
(SMAP) and {{smap.MSL.always_on.event_f1:.3f}} (MSL), higher than every detector, while its
corrected F0.5 is 0. The "late ref." column is the key caution: a label-free reference that keeps
each detector's amount of alarm time but places it as one block at the end of the series beats the
detector itself in seven of the eight detector-spacecraft pairs, because
{{smap.config.labelled_steps_in_second_half:pct}} of labelled steps lie in the second half of their
series. The zero-shot foundation model does not improve on simple detectors here.

### 6.6 Couplings

| Coupling | Result |
|---|---|
| C1 family evidence | helps the selected learner at 2016 ({{re.o2016.abl.no_family.diff:+.4f}}, {{re.o2016.abl.no_family.lo:+.4f}} to {{re.o2016.abl.no_family.hi:+.4f}}); at 2011 the interval includes zero |
| C2 cloud survival | best readings are accurate to about a tenth of a cloud, but with {{c2.n_clouds}} clouds the experts cannot be ranked: calibrated TabICLv2 {{c2b.predcal_tabicl.v:.3f}} ({{c2b.predcal_tabicl.lo:.3f}}-{{c2b.predcal_tabicl.hi:.3f}}) against raw physics {{c2b.pred_physics_only.v:.3f}} ({{c2b.pred_physics_only.lo:.3f}}-{{c2b.pred_physics_only.hi:.3f}}) |
| C3 operational status | drag physics over-predicts for operational payloads (mean predicted {{c3.status.operational.mean_p_physics:.3f}} against observed {{c3.status.operational.event_rate:.3f}}) and under-predicts for non-operational ones ({{c3.status.non-operational.mean_p_physics:.3f}} against {{c3.status.non-operational.event_rate:.3f}}). The status-aware policy lowers the Brier score by {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.mean:+.4f}} ({{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.ci95.0:+.4f}} to {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.ci95.1:+.4f}}): not significant |
| T1 state estimation | the largest effect in the study, section 6.3 |
| C4 | demonstration only, untested |

### 6.7 The dated forecast

As of {{twin.as_of}} the twin holds {{twin.objects:,}} objects with perigee below 2000 km:
{{twin.regime_counts.natural_decay:,}} in natural decay,
{{twin.regime_counts.operator_controlled:,}} operator-controlled payloads and
{{twin.regime_counts.unknown:,}} payloads without a reported status. Expected natural-decay
re-entries in the next 30 days: {{twin.expected_reentries_natural_decay.30d:.1f}}. The record is
fixed by its SHA-256 hash `{{twin.sha256}}` so that a later catalogue snapshot can score it
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
