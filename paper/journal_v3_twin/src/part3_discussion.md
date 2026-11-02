# Discussion {#sec:discussion}

**Which re-entry expert is best depends on the test period.**
- For 2016–2021, two in-context tabular foundation models released in 2026, TabICLv2 and
  Kumo-Tabular, were significantly more accurate than the physics expert, the gated model, the
  selected TabM and the best tree ensemble. They also had the best concordance and near-nominal
  raw coverage.
- For 2011–2015, the physics expert alone had the lowest point estimate, but it could not be
  separated from the foundation models or from the gated model. It had the lowest loss for
  fragments in that period. The one significant difference among them is that Kumo-Tabular, the
  learner selected there, is worse than the gated model that fuses it.

TabICLv2 was thus best or tied in both periods. One change between the periods is measured:
replaying the realised F10.7 lowers the physics expert's error much more in the rising phase of
cycle 25 than in cycle 24. Other changes in the test cohorts were not isolated here. Two test
periods, of which several comparisons were added after the first results were seen, are not
enough to say whether the foundation models will keep their lead.

**Selecting a learner on a tuning cohort five to seven years older than the test cohort did not
reliably pick the model that generalised best.**
- At 2016, TabM led on tune-A by a wide margin and was significantly worse than both foundation
  models on test.
- At 2011, the small tune-A set ({{re.o2011.known5.tuneA}} objects) chose Kumo-Tabular, which
  is tied with TabICLv2 on test.
- The first learner set, recorded as a deviation, selected its MLP at both origins, and the MLP
  and TabM failed on 2011 fragments.

The temporal protocol is what exposes this. A random split would have hidden it.

**The reliability gate helped when the physics expert was reliable, and it did not beat the best
single expert.**
- At 2011 it significantly improved every learner to which it gave non-negligible weight
  (post-hoc sensitivity analysis). The reported 2011 gated model depends on the change of the
  candidate pool: with the first learner set, its gated model was much worse.
- At 2016, when solar activity degraded the physics expert, it improved none and cost a little.
  The cost was not statistically significant.

The gate is fitted on out-of-fold predictions of the training period, so it cannot anticipate
that a learner will degrade on a future cohort. Its descriptors (area-to-mass source, the physics
expert's spread, orbit, class) describe the physics expert's reliability, not the learner's
reliability under shift. A gate that also sees descriptors of covariate shift is a natural next
step, but it would have to be validated on a third test period.

**The radar-cross-section leak is a warning for catalogue-based learning.**
- A covariate that "is a size measurement" encodes the outcome through its availability
  (adjusted odds ratio {{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.1f}}, cluster-robust
  95 % CI {{leak.o2016.train.adjusted_or_rcs_missing.ci95.0:.1f}}–{{leak.o2016.train.adjusted_or_rcs_missing.ci95.1:.1f}}).
- Adding it changed the test pinball loss by amounts within noise at both origins, so point
  accuracy would not have revealed the leak.
- At 2016, where its missingness pattern differs from training, it broke interval coverage.

Any study that joins catalogue snapshots taken long after the prediction time should test the
missingness of each covariate against the outcome. Our code includes such a check.

**What the couplings show.**
- **Family evidence (C1)** improved the learner in one of two periods.
- **Cloud survival (C2)** is accurate to about a tenth of a cloud's size for the best readings,
  but with {{c2.calibrated.cosmos_gated.n_clouds}} clouds the experts cannot be ranked.
- **Operational status (C3)** marks a regime in which drag physics over-predicts re-entry by a
  factor of about three. A status-aware policy lowered the Brier score in one as-of-date window,
  but not significantly once outcomes are clustered by launch.
- **The twin's current-state forecasts (T1)** are where the coupling of history and physics
  matters most. Conditioning on the catalogue history reduced the over-prediction of 72-day
  re-entries from {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}}-fold (lifetime
  pipeline) and {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ratio:.1f}}-fold (class
  priors) to {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ratio:.2f}}-fold, in one
  window that had been analysed earlier in the project.

C1, C2 and C3 are small or uncertain effects. T1 is a large effect, but it comes from one
window that had been analysed before. Together they are what a digital twin that only joins
separate tools cannot provide: a measured account of which links between the modules carry
information.

## Limitations and threats to validity {#sec:limits}

**One orbit per object.**
- Orbital-element histories at catalogue scale are not legitimately public: Space-Track
  requires an account and forbids redistribution.
- The re-entry module therefore predicts from the canonical orbit, and the twin uses two orbits:
  canonical and current.
- Operational tools that fit the ballistic coefficient to weeks of TLE data
  [@pardini2008impact; @gondelach2017ballistic] use far more information near re-entry. Our
  months-to-years forecasts are not comparable with their final-days accuracy.

**Validation of current-state forecasts.**
- The lifetime study validates forecasts at canonical-orbit epochs.
- Forecasts for objects already in orbit are validated only by T1. T1 covers one 72-day window,
  and it is not blind: an earlier version of the project fitted and tested re-entry classifiers
  on the same window, and its aggregate decay count was seen before the history-conditioned
  variant was designed.
- The hashed forecast of 2026-10-27 is a blind test that a later catalogue snapshot will
  score.

**Priors built from catalogue values.**
- The class prior of characteristic length comes from training objects that received a
  radar-cross-section value, so it is biased towards long-lived objects.
- Among components, the availability of design data is associated with early decay (adjusted
  odds ratio {{leak.o2016.train.components_adjusted_or_design_missing.odds_ratio:.1f}}). We
  treat it as known at launch, because GCAT design values come from launch documentation. The
  `no_design` ablation bounds its influence.

**Resolution of the decay integrator.** A step may not drop the perigee by more than 2 % of its
height above 120 km, with a floor of 1 km, and may not exceed 10 days. For an object whose perigee
is already within 1 km of the 120 km threshold, the first step can therefore be a full 10 days, and
the forecast is reported at the end of that step. Two of the {{twin.regime_counts.natural_decay:,}}
natural-decay objects of the dated forecast have a degenerate 90 % interval at exactly 10 days for
this reason. The affected objects are within days of re-entry in any case, and no reported
comparison depends on them.

**Physics simplifications.**
- The density is a global NRLMSIS 2.1 average without latitude, local-time or seasonal
  variation.
- The atmosphere is exponential about perigee within an orbit.
- Lunisolar and J2 effects on perigee are not modelled. They matter for transfer-orbit stages.
- Solar activity is held at its value at the prediction time. The oracle diagnostic shows how
  much of the error this causes (Section [-@sec:results]).

**Small calibration sets at the 2011 origin.** The 2004–2005 tuning cohort is small, so model
selection and calibration at that origin are noisy.

**Test-set reuse and post-hoc analyses.**
- The re-entry and collision-risk test sets were scored more than once during development:
  before an internal review, with a first learner set, and in smoke tests of the new learners.
- The pool of eligible learners was changed after the first set's test results were known.
- The head-to-head comparisons, the gate applied to every learner and the calibrated C2 reading
  were added after results were seen. Their intervals are nominal and uncorrected for
  multiplicity.

All of this is recorded in Section [-@sec:protocol] and in the released logs. It weakens the
confirmatory value of the comparisons, and a fresh test period is needed to confirm them.

**Bit-level reproducibility.** Seeded training of TabM on the GPU is not bit-reproducible; on CPU
it is. The re-entry study reuses cached predictions, so its numbers are fixed, but a refit gives
slightly different results: two runs of the T1 lifetime-pipeline variant gave expected/observed
ratios of 8.9 and {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}}. Chronos-2
inference on the GPU varies at the same level; repeating the SMAP/MSL run moved its per-channel
AUC-PR by at most $2 \times 10^{-5}$ and left every reported value unchanged at three decimals.

**Health evidence is a proxy.**
- Catalogue operational status is the only health evidence available for every payload. It is
  not a measurement of manoeuvre capability.
- No public dataset links telemetry, conjunction messages and catalogue objects, so the
  health-to-debris coupling (C4) is a demonstration only.

**Event metrics for telemetry.** Event-wise metrics can be satisfied by trivial alarm rules, so
they are reported next to trivial references and with the corrected precision of ESA-ADB. Even
so, a reference that only places alarms late in each series beats most detectors on SMAP/MSL.
The alarm protocol of the health module helped on one ESA-ADB mission and hurt on the other, so
the health module is the least validated part of the twin.

# Conclusions {#sec:conclusions}

COSMOS-AI joins satellite health monitoring, orbital debris analysis and atmospheric re-entry
prediction in one digital twin. Each module is validated on public data, and the couplings are
tested wherever a real test exists. Five findings stand out.

1. **Re-entry prediction across solar-cycle phases.**
   - Two 2026 tabular foundation models were significantly the most accurate in 2016–2021.
   - In 2011–2015 TabICLv2 was statistically tied with the physics expert and the gated model;
     Kumo-Tabular was significantly worse than the gated model that fuses it.
   - Selection on an older tuning cohort chose a significantly worse learner in 2016.

   The reliability gate improved the learners it weighted at the 2011 origin and was neutral to
   slightly harmful at 2016. It did not beat the best single expert.
2. **Catalogue covariates can carry the outcome in their availability.** Whether an object has a
   radar cross-section is strongly associated with early decay. Using it broke interval coverage
   in the period whose missingness pattern differed from training, while its effect on point
   accuracy was within noise.
3. **For objects already in orbit, the twin's state estimate matters more than the choice of
   learner.** Conditioning each object's area-to-mass ratio on its catalogue history reduced the
   over-prediction of 72-day re-entries from
   {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}}-fold (the lifetime
   pipeline) and {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ratio:.1f}}-fold
   (physics with class priors) to
   {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ratio:.1f}}-fold.
4. **Operational status separates payloads whose re-entry drag physics cannot predict.** Using
   it lowered the Brier score in one 72-day window, but not significantly.
5. **Conjunction and telemetry: the newest models did not beat the established baselines on the
   operational metrics.**
   - On the ESA collision-risk benchmark, a 2026 tabular foundation model gave significantly the
     best probabilistic forecasts. Neither it nor any other model beat the latest-risk forecast
     on the official point metric.
   - A zero-shot time-series foundation model did not beat simple detectors on SMAP/MSL.

**Limits.**
- The twin rests on single catalogue orbits rather than orbital histories.
- Findings 3 and 4 come from one as-of-date window that had been analysed earlier in the
  project, so they are not blind tests.
- The test sets were scored more than once during development, and several analyses were added
  after results were seen (Section [-@sec:protocol]).

**Verification.** The hashed forecast of 2026-10-27 is the first blind test of the twin and
will be scored against later catalogue snapshots. The code and derived data needed to repeat
every number are released with the paper.

