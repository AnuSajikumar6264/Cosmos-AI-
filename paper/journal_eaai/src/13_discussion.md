# Discussion, limitations and threats to validity {#sec:discussion}

## What the evidence supports

Three claims survive the measurements. First, **in-context tabular foundation models transfer to an operational space catalogue**: at origin 2016 they reduce pinball loss by {{imp.re.o2016.tabicl_vs_xgboost_pct:.0f}} % against the best gradient-boosted baseline and by {{imp.re.o2016.tabicl_vs_physics_pct:.0f}} % against the physics prior, with intervals excluding zero, and the same family of models gives significantly the best collision-risk distribution. This is a positive result for in-context learning outside its usual benchmark setting. Second, **distribution-free calibration is what makes the output usable**: the conformal interval covers a far larger share of decayed objects than the operational window applied at the same horizon, and the Mondrian grouping is what keeps coverage near nominal at the earlier origin. Third, **the state estimate matters more than the learner** for objects already in orbit, where a Bayesian conditioning step removes most of a ninefold bias that no choice of learner addressed.

## What the evidence does not support

The reliability gate, which is the architectural novelty, **does not beat the best single expert at either origin**. Table [-@tbl:moe] makes this explicit, and we draw three conclusions rather than one. The gate is fitted on out-of-fold predictions of the training period, so it can learn where the physics prior is unreliable but cannot anticipate that a *learner* will degrade on a future cohort; its descriptors are all properties of the prior. The equal-weight and constant-gate arms show that the benefit of learning the weight at all is small and inconsistent across origins. And the tuning-split variant shows that the fitting protocol dominates the parameterisation. A gate that also saw descriptors of covariate shift is the obvious next step, and it would have to be validated on a third test period rather than on these two.

The honest reading is therefore that reliability gating is a safety mechanism rather than an accuracy mechanism: it prevented the catastrophic arm at each origin (TabM at 2011, the physics prior at 2016) without matching the best one. For an operator who must fix an architecture before knowing which regime the next five years bring, that property has value; for a benchmark table it is a loss.

Two further negative results are reported. Zero-shot time-series forecasting did not beat ridge autoregression on spacecraft telemetry, and on the official collision-risk metric no model beat a baseline that simply repeats the latest risk.

## Threats to validity

**Test-set reuse.** The test sets were scored more than once during development, the candidate pool was narrowed after an earlier run's test results were known, and several analyses were added after results had been seen. Section [-@sec:setup] states this and the supplement records it in full. This bounds the comparisons' confirmatory value; the significance statements should be read as nominal.

**Two periods only.** Every statement about generalisation rests on two test periods from one catalogue. The direction of the solar-cycle effect is measured, not assumed, but a third period would be needed to establish that the foundation models keep their advantage.

**Single orbit per object.** The re-entry module predicts from one dated orbit because orbital histories at catalogue scale cannot be redistributed. Operational tools that fit a ballistic coefficient to weeks of tracking data use far more information near re-entry, and our months-to-years forecasts are not comparable with their final-days accuracy.

**Priors estimated from catalogue values.** The physics prior's class pools are built from objects that received a radar-derived size, which biases them towards longer-lived objects; the design-data ablation bounds the influence of this.

**Telemetry domain.** The telemetry benchmarks are real but belong to spacecraft outside the catalogue population, so the health module transfers a method and not a fitted model, and its coupling into the debris modules runs through operational status rather than telemetry. The benchmark's own metric artefacts, documented in Section [-@sec:results], mean its absolute scores should not be read as operational detection rates.

**Integrator resolution.** A step of the decay integrator may not exceed ten days, so for an object whose perigee is already within one kilometre of the re-entry threshold the forecast is reported at the end of that step; two objects in the released forecast have a degenerate interval for this reason. They are within days of re-entry regardless, and no reported comparison depends on them.

**Reproducibility.** Seeded training on the graphics processor is not bit-reproducible, while on the central processor it is. Repeating the telemetry run moved per-channel AUC-PR by at most $2 \times 10^{-5}$ and changed no reported value at three decimals; two runs of the lifetime variant in the as-of-date test gave ratios of 8.9 and {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}}.
