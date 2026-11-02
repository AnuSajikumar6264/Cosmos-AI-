# Prediction targets, inputs and leakage-resistant splits

## R. Re-entry of catalogued objects (central task)

**Unit.** One catalogued object whose GCAT canonical orbit (an orbit "representative of the
spacecraft's early operations, after initial positioning" — J. McDowell, GCAT documentation)
has perigee below 2000 km. Source: GCAT `satcat.tsv` release of 2026-10-26, retrieved
2026-10-27 (`data/feature2_debris/gcat/raw/2026-10-27/`).

**Prediction time.** `t0` = the canonical-orbit epoch (GCAT `ODate`), required to be day-precise.

**Event.** Destruction in the atmosphere after natural decay (GCAT status `R`) at `DDate`
(day-precise). Time to event `T = DDate − t0` in days.

**Censoring.**
- Still in orbit (`O`, `AO`): censored at the snapshot date 2026-10-27.
- Controlled or other endings (deorbit `D`, landing `L`, docking, attachment, explosion `E`, renaming, failure): censored at their `DDate` (cause-specific view: a controlled end prevents observing natural decay).
- Excluded: lost objects (`OX`), deep-space phases, objects whose dates are not day-precise, and 499 rows whose end date precedes `t0` (catalogue inconsistencies). Counts: `data/feature3_reentry/processed/lifetime_cohort_2026-10-27_build_log.json`.

**Targets.**
- `z = log10(min(T, 5 yr) / 1 day)`, floored at 0.5 day: the capped time to re-entry. It is known when the object decayed within 5 years or was followed for at least 5 years.
- Horizon events `T ≤ H` for `H` ∈ {30 d, 182.6 d, 1 yr, 5 yr}: known when the event happened before `H` or follow-up reached `H`.
- For error in days, the subset that decayed within 5 years is also reported.

**Inputs (all known at `t0`).**
| Group | Features |
|---|---|
| orbit | perigee, apogee, inclination, eccentricity, semi-major axis, mean altitude |
| age | `t0` minus launch date |
| design | mass, length, diameter, span, design cross-section and area-to-mass (from GCAT design values; unknown for fragments), GCAT mass-uncertainty flag |
| space weather | trailing 81-day and 365-day F10.7, trailing 81-day Ap, 1-year F10.7 trend; only observed or interpolated CelesTrak `SW-All.csv` rows of days that ended before `t0` (predicted rows are dropped; `t0` is day-precise, and a day's F10.7 and daily Ap are complete only at its end) |
| family | for the same launch and, for fragmentation debris, the same breakup parent: siblings catalogued by `t0`, siblings that decayed naturally before `t0`, their fraction, and their mean log-lifetime |
| physics | quantiles (10, 50, 90 %), spread and 5-year survival probability of the physics expert's Monte-Carlo predictive distribution |
| class | object class (payload, rocket stage, component, fragment), parent class, shape |

Not used: decay dates of the object itself, any sibling decay after `t0`, future solar activity
(only in a labelled oracle diagnostic), GCAT status (it encodes the outcome), and the object's
**radar cross-section**. The RCS comes from the 2026 SATCAT snapshot, and whether a value
exists depends on the outcome. Objects that decayed within weeks of `t0` mostly never got one
published, and no values are published for objects catalogued after about 2018. In the
origin-2016 training split, P(decay within 30 d) is 0.76 without an RCS value against 0.17
with one, and the gap persists within decades and perigee bands. Using the value, or its
absence, would leak the outcome (adversarial review of 2026-10-27, finding 0). RCS enters only:
- the class-level size prior of the physics expert, built from training objects only;
- a labelled leak demonstration (`<learner>_plus_rcs_2026`), never a candidate for selection.

Design-data availability for components (class C) is also associated with the outcome
(adjusted odds ratio 0.29 for 30-day decay, given orbit and decade). GCAT design values come
from launch documentation, so they are treated as known at `t0`; the `no_design` ablation
shows how much the results depend on them.

**Physics expert inputs.** The area-to-mass ratio comes from one of three sources:
- the design value;
- for intact payloads and rocket stages without design values, a class prior of design values;
- for fragments and components without design values, the NASA breakup-model law with a class
  prior of characteristic length.

The orbit averages use King-Hele's series for e ≤ 0.05 and exact quadrature over the eccentric
anomaly above that. See `cosmos/ai/lifetime/physics.py`.

**Populations.**
- *Uncontrolled* (primary): fragments, rocket stages and components, which cannot manoeuvre.
- *All* (secondary): adds payloads. Many payloads manoeuvre or are deorbited by their operators, so their lifetimes are not drag-determined. This population is implemented (`--population all`) but was **not run** for the results reported. Payloads are covered by the prospective tests C3 and T1 instead, which condition on operational status.

**Protocol "as of origin O"** (implemented in `cosmos/ai/lifetime/run.py`):
| Split | Cohort (`t0`) | Outcomes known up to | Use |
|---|---|---|---|
| train | `t0 < O − 7 yr` | censored at `O` | fitting |
| tune-A | `O − 7 yr ≤ t0 < O − 5 yr`, half of the launch families | censored at `O` | early stopping, hyperparameter and model selection, gate fitting check |
| tune-B | same cohort, other half of the launch families | censored at `O` | conformal and probability calibration |
| test | `O ≤ t0 <` test end (2016-01-01 at origin 2011; 2021-10-27 at origin 2016) | 2026-10-27 (5-year follow-up complete) | scoring only (scored more than once during development; see `REVIEW_AND_FIXES.md`) |

Two origins give two independent test periods: O = 2016-01-01 (test 2016–2021) and
O = 2011-01-01 (test 2011–2015). Splitting by `t0` keeps every launch family within one split
(siblings share `t0` up to a few weeks; tune-A and tune-B are split by launch family). No
outcome after `O` and no space weather after `t0` enters fitting or selection.

**Metrics.**
- Mean pinball loss over the quantiles {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}.
- MAE and RMSE of the median in log10 days.
- MAE, median and 90th-percentile absolute error in days, and median relative error. These are
  computed on all test objects with a known capped outcome, and on those that decayed within
  5 years.
- Coverage, mean width and Winkler interval score of 80 % and 90 % intervals. Quantiles within
  1e-6 of the 5-year cap are set exactly to it, because the cap is a point mass of the target.
- Harrell's C on all test objects, censored included.
- For each horizon: AUC-PR and AUC-ROC on the raw probability read off the quantile function, and
  the Brier score and ECE on the tune-B isotonic-calibrated probability. Isotonic calibration
  ties scores, which would change the rank metrics.

Uncertainty is given as 95 % percentile intervals from a cluster bootstrap over launch families
(2000 replicates), paired for model differences. Every learned expert is run with five seeds.
Learner caches are keyed by a fingerprint of the split, the features and the settings.

## D. Collision risk from Conjunction Data Messages (debris analysis)

**Unit.** One conjunction event (ESA Kelvins Collision Avoidance Challenge, Zenodo 4463683),
a time series of anonymised CDMs.

**Inputs.** The CDMs of the event released at least 2 days before the time of closest approach
(TCA): every numeric field of the latest such CDM, the chaser object type and mission, and
sequence summaries (number of CDMs, first, minimum, maximum and mean risk, change and slope of
the risk over the last three CDMs, last step, time span, log-ratios of miss distance and
position sigmas between the first and latest CDM). Fields spanning many decades (covariance
determinants up to 1e46, sigmas) get a signed log10 transform chosen on the training data.

**Target.** The risk `r` (log10 collision probability, floored at −30) of the event's final CDM.
Training events must have a CDM at ≥ 2 days and a final CDM at < 1 day before TCA (8,293 of
13,154 events; 66 high-risk). Test: the official 2,167 events (150 high-risk) with the targets
released after the challenge.

**Official metric** (Uriot et al. 2022, Sec. 4.3): high-risk events are `r ≥ −6`;
`F2` is the F-score with β = 2 for the high-risk classification over all test events;
`MSE_HR` is the mean squared error over the true high-risk events; `L = MSE_HR / F2` (lower is
better). Low-risk predictions are clipped at −6.001 as in the paper. Check of the
implementation: the latest-risk baseline gives L = 0.6940 (F2 0.7391, MSE_HR 0.5129), the
published value (0.694).

**Further metrics.** Pinball loss of the predicted final-risk quantiles; AUC-PR, Brier score
and ECE of P(r ≥ −6); coverage and width of conformal 80 % and 90 % intervals, overall and on the
high-risk events.

**Splits.** Hyperparameters are chosen on a 20 % hold-out of training events; gating and
calibration use 5-fold out-of-fold predictions of the training events; the official test set
is used only for scoring (it was scored more than once during development; see
`REVIEW_AND_FIXES.md`).

## H. Telemetry anomaly detection (supporting)

ESA-ADB: as in the pre-registered study (corrected event-wise F0.5, official split). OPSSAT-AD:
official split. NASA SMAP/MSL: official telemanom labels; point-adjusted metrics are not used as
headline metrics (they inflate random detectors).
