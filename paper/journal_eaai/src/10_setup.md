# Experimental setup and metrics {#sec:setup}

## Temporal protocol

The prediction time $t_0$ is the canonical-orbit epoch. Under the protocol *as of origin $O$*, the training cohort has $t_0 < O - 7$ yr, the tuning cohort $O - 7 \leq t_0 < O - 5$ yr, and all their outcomes are censored administratively at $O$ so that nothing later than the origin can influence a fitted quantity. The tuning cohort is split by a hash of the launch family into tune-A, used for early stopping, the hyperparameter search, learner selection and gate checks, and tune-B, used for conformal and probability calibration; splitting by family rather than at random prevents siblings of one launch from appearing on both sides. The test cohort starts at $O$ and ends where the five-year follow-up is complete at the snapshot, or at the next origin: $t_0 <$ 2016-01-01 at origin 2011 and $t_0 <$ 2021-10-27 at origin 2016. The two origins, 2011-01-01 and 2016-01-01, therefore give two disjoint test periods, which span different phases of the solar cycle. The primary population is objects that cannot manoeuvre; payloads are excluded from it because many are operator-controlled.

For the collision-risk module the official challenge split is used unchanged. For telemetry the official channel split is used, with thresholds fixed in advance.

## Metrics

Table [-@tbl:metrics] lists every metric and what it is for. Pinball loss over the seven quantiles is primary throughout, because a single predicted date is not a decision-useful object for either re-entry planning or conjunction screening.

{{TABLE:metrics}}

: **Metrics.** Ranking metrics use the raw probabilities, because isotonic calibration ties scores; calibration metrics use the mapped ones. {#tbl:metrics}

Confidence intervals come from 2,000 cluster-bootstrap replicates, paired for model differences. The resampling unit is the launch family for the catalogue cohorts, the launch for the payload-status test and the cloud for the cloud-population test, because objects sharing an origin do not provide independent evidence. Learner predictions are cached under a fingerprint of the data, features and settings, so a stale result cannot enter a comparison.

## Deviations and prior use of test data

The test sets were not untouched when the reported results were produced, and the full record is in the supplementary material. In summary: an earlier complete run used a different candidate pool and its test results were known before the pool was narrowed to models published 2022–2026; two engineering smoke tests printed test scores before the reported run; every benchmark was scored once before an internal review and corrected afterwards; the window used for the as-of-date tests of C3 and the state-estimation comparison had been analysed earlier in the project; and the head-to-head comparisons, the gate-applied-to-every-learner analysis and the calibrated cloud reading were added after results had been seen. All intervals are nominal 95 % intervals without correction for multiplicity. We report this because it bounds how confirmatory the comparisons can be, and a fresh test period would be needed to confirm them.
