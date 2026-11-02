# Ablations and uncertainty evaluation {#sec:ablations}

## Censoring

Any evaluation restricted to objects that have already decayed discards most of the cohort. Table [-@tbl:km] and Figure [-@fig:km] give the Kaplan–Meier description of both test cohorts, which is model-free and uses the censoring indicator directly.

{{TABLE:km}}

: **Kaplan–Meier description of the test cohorts.** $S(\cdot)$ is the estimated survival probability. A median that is "not reached" means more than half the group is still in orbit, so no median lifetime exists for it. {#tbl:km}

![**Kaplan–Meier survival of the test cohorts**, overall and by object class, with the censored fraction of each group.](artwork_eaai/eaai_km.png){#fig:km width=100%}

{{re.o2016.km.all.censored_fraction:pct}} of the 2016 cohort and {{re.o2011.km.all.censored_fraction:pct}} of the 2011 cohort are right-censored. At origin 2011 the median lifetime of the cohort is never reached. Rocket stages and components decay substantially faster than fragments in both periods. This is why the target is capped and scored only where the five-year outcome is known, why the accelerated-failure-time baseline is trained on censored observations as well, and why the comparison in Table [-@tbl:gap] against studies that train on decayed objects only is not like-for-like in their favour.

## Feature groups

Removing each feature group from the selected expert and refitting with identical settings and seeds gives the paired effects in Figure [-@fig:ablation]. The family features are the coupling C1 result reported in Table [-@tbl:couplings]. Design properties help at 2016 ({{re.o2016.abl.no_design.diff:+.4f}}, {{re.o2016.abl.no_design.lo:+.4f}} to {{re.o2016.abl.no_design.hi:+.4f}}); the physics summaries help the selected expert at 2011 ({{re.o2011.abl.no_physics.diff:+.4f}}) but hurt at 2016 ({{re.o2016.abl.no_physics.diff:+.4f}}), which is consistent with the physics prior's own error growing in the rising phase of the solar cycle. All intervals are nominal and uncorrected for the eight comparisons.

![**Feature-group ablations of the selected expert.** In the first four rows a group is removed, so a positive change means the group helps; the last row instead adds the leaking covariate, where a positive change means it makes the model worse.](artwork/Figure_9.png){#fig:ablation width=78%}

## Leakage through covariate availability

Whether the catalogue publishes a radar cross-section depends on how quickly the object decayed. In the origin-2016 training cohort {{leak.o2016.train.p_decay30_given_missing:pct}} of objects without a value decayed within 30 days of their canonical-orbit epoch, against {{leak.o2016.train.p_decay30_given_present:pct}} of those with one; adjusted for orbit, class and decade, and with standard errors clustered on launch families, the odds ratio is {{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.2f}} ({{leak.o2016.train.adjusted_or_rcs_missing.ci95.0:.2f}} to {{leak.o2016.train.adjusted_or_rcs_missing.ci95.1:.2f}}).

The consequence for a modeller is instructive. Adding the covariate changes the test pinball loss by amounts whose intervals include zero at both origins, so point accuracy would not reveal the problem. What it does break is calibration: at origin 2016, where the missingness pattern differs from training, the 90 % coverage of the selected expert falls from {{re.o2016.BEST.cov90:pct}} to {{re.o2016.BEST_plus_rcs_2026.cov90:pct}}. A covariate that looks like a size measurement is in fact partly an outcome indicator, and the symptom appears in the interval rather than in the mean. We therefore exclude it, and release the availability test with the code.

## Uncertainty quantification

{{TABLE:coverage}}

: **Interval rules on both test periods.** "Cov. all" is over every test object including those surviving past the five-year cap; "Cov. decayed" is restricted to objects that decayed within five years; width is in $\log_{10}$ days. The first three numeric columns are origin 2016, the last three origin 2011. {#tbl:coverage}

Against the operational practice it would replace, the calibrated interval is a clear improvement on the objects that matter. The ±20 % residual-lifetime window around the physics median — the rule assessed at roughly 90 % coverage in the final days before re-entry [@pardini2018assessing] — covers {{re.o2016.rule.physics_only_pm20.coverage_decayed:pct}} and {{re.o2011.rule.physics_only_pm20.coverage_decayed:pct}} of objects that decayed within five years when it is applied at canonical-orbit epochs, while the conformal 90 % interval covers {{re.o2016.rule.physics_only_conformal90.coverage_decayed:pct}} and {{re.o2011.rule.physics_only_conformal90.coverage_decayed:pct}}. The comparison is not a criticism of the operational rule, which is designed for a different regime with continuous tracking; it quantifies how far that regime is from forecasting years ahead from one catalogue orbit. Figure [-@fig:coverage] places every rule side by side.

![**Interval coverage** over all test objects, including those still in orbit after five years.](artwork/Figure_8.png){#fig:coverage width=92%}

**Conditional coverage.** Mondrian grouping is the mechanism intended to make coverage conditional, and Table [-@tbl:coverage] shows it reaching the nominal level marginally at origin 2011 and falling short at origin 2016 for the arms with the narrowest intervals.

**Temporal calibration.** Conformal prediction guarantees *marginal* coverage under exchangeability, and a forward-in-time test period is precisely where exchangeability is doubtful. Table [-@tbl:temporal] and Figure [-@fig:temporal] break the 90 % coverage down by calendar year of the prediction time.

{{TABLE:temporal}}

: **Conditional coverage over time.** Years with fewer than 25 scored objects are omitted. {#tbl:temporal}

![**Coverage by calendar year.** The dashed line is the nominal level. Coverage drifts downward across the 2016 test period and is stable across the 2011 period.](artwork_eaai/eaai_temporal.png){#fig:temporal width=100%}

At origin 2016 the physics prior's coverage falls from {{re.o2016.tcov.physics_only.first:.3f}} in the first test year to {{re.o2016.tcov.physics_only.last:.3f}} in the last, a drop of {{re.o2016.tcov.physics_only.drop:.3f}}, while at origin 2011 it stays between {{re.o2011.tcov.physics_only.last:.3f}} and {{re.o2011.tcov.physics_only.first:.3f}}. The marginal guarantee is kept; what degrades is coverage conditional on time, as the rising phase of solar cycle 25 moves the covariate distribution away from the calibration cohort. To our knowledge this is the first demonstration of the effect on an operational space-surveillance catalogue, and it is a caution for any deployment that calibrates once and then runs for years.

## State estimation for objects in orbit

The lifetime pipeline is fitted at canonical-orbit epochs; applied to objects that have survived for years it over-predicts re-entry by {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}}-fold over a 72-day horizon ({{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ci95.0:.1f}} to {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ci95.1:.1f}}), because survival selects low area-to-mass ratios that the fitted model cannot know about. Class priors reduce this to {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ratio:.2f}}-fold and conditioning on the catalogue history to {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ratio:.2f}}-fold ({{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ci95.0:.2f}} to {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ci95.1:.2f}}), still significantly above one. The three variants rank decaying objects almost identically, with AUC-ROC within 0.01 of each other; what differs is calibration, which is the quantity the state-estimation step targets. Figure [-@fig:t1] shows the reliability and the counts.

![**As-of-date test of the three variants** for objects in orbit on 2026-08-16, scored against decays to 2026-10-27. (a) Reliability in six fixed probability bins, marker area growing with the number of objects. (b) Expected against observed re-entries.](artwork/Figure_14.png){#fig:t1 width=92%}

This window had been analysed earlier in the project, so this comparison is an as-of-date test rather than a blind one; a dated, hashed forecast released with the code is the blind test, and a later catalogue snapshot will score it.
