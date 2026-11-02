<!-- Generated from docs/RESULTS_AND_ANALYSIS.src.md by paper/journal_v3_twin/tools/render_doc.py; edit the source, not this file. -->

# Results and analysis (COSMOS-AI digital twin, 2026-10-27)

All numbers below are placeholders filled from the result files by `paper/journal_v3_twin/tools/render_doc.py`.
The text is assembled from the same drafts as the manuscript. Full tables are in
`paper/journal_v3_twin/build/supplementary.md`; the review findings and fixes are in `docs/REVIEW_AND_FIXES.md`.

**Deviations and prior use of test data.** The test sets of this study were not untouched when the reported results were produced, and we list what had been seen.
- *Learner pool.* The first complete re-entry run (run 1; Supplementary Table S1) chose the expert among XGBoost, LightGBM, CatBoost, the plain MLP and TabICLv2. After its test results were known at both origins, and after the collision-risk test set had been scored with the same pool, we restricted the eligible pool of both the re-entry and the collision-risk experts to learners published 2022–2026 (TabM, TabICLv2, Kumo-Tabular), so that the expert would be chosen among current methods; the classic learners became reference baselines. The selection rule, budget, seeds, features, gate and calibration were not changed. The reported run is run 2. Run 1 selected the MLP at both origins, and its 2011 gated model was much worse than run 2's, so the 2011 gated result depends on this change.
- *Smoke tests.* Before run 2, engineering tests of the new learners printed two test-set scores: Kumo-Tabular's origin-2016 pinball loss (0.051) and TabM's origin-2011 pinball loss (0.0729). No setting was chosen with them.
- *Repeated scoring.* An earlier version of every benchmark was scored on its test set and then corrected after an internal review; the superseded results are archived with the code. The re-entry and collision-risk test sets were therefore scored more than once during development.
- *The T1 and C3 window.* An earlier version of the project fitted and tested re-entry classifiers (physics, logistic regression, XGBoost, XGBoost-AFT and a hazard network) on the window from 2026-08-16 to 2026-10-23, and the aggregate decay count of the window was seen while the lifetime pipeline's bias on current orbits was diagnosed, before the history-conditioned variant was designed. A defect in the twin's payload prior (payloads without design values fell back to the fragment breakup law) was found and fixed after T1 had first been scored; the fix changes payload probabilities only.
- *Post-hoc analyses.* The class breakdown, the gate applied to every learner, the head-to-head comparisons, the calibrated C2 reading and the C2 bootstrap were added after test results had been seen. They are labelled as post hoc, and all intervals are nominal 95 % intervals without correction for multiplicity.

## Summary

| Question | Answer (measured) |
|---|---|
| Best re-entry model, 2016–2021 | TabICLv2 (0.0499) and Kumo-Tabular (0.0508), tied with each other and significantly better than the physics expert (0.0892), the gated model (TabM, 0.0604) and XGBoost (0.0576) |
| Best re-entry model, 2011–2015 | physics alone has the lowest point estimate (0.0527) but is tied with TabICLv2 (0.0596), Kumo-Tabular (0.0616) and the gated model (0.0583) |
| Does the reliability gate help? | 2011: yes, for every learner it weights (post hoc). 2016: no (neutral to slightly harmful). Never better than the best single expert |
| Did tune-A selection find the best learner? | 2016: no, TabM was chosen and is significantly worse than both foundation models. 2011: Kumo-Tabular, tied with TabICLv2 |
| RCS leak | Adjusted OR 3.59 (cluster-robust 95 % CI 2.09–6.17). Pinball changes within noise at both origins; 2016 90 % coverage falls from 87.7 % to 76.8 % |
| Twin forecasts for objects in orbit (T1, 72 d) | expected/observed: history-conditioned 1.37, class prior 3.64, lifetime pipeline 9.52 (as-of-date test on a previously analysed window; not blind) |
| Health → re-entry regime (C3) | Brier difference, status-aware minus physics-only: -0.0054 (95 % CI -0.0120 to +0.0006), not significant |
| Collision risk (Kelvins) | best probabilistic: Kumo-Tabular (pinball 0.653; minus TabICLv2 -0.039, 95 % CI -0.053 to -0.025; persistence 1.045); official metric: no model beats the latest risk (0.694) |
| Telemetry (SMAP/MSL) | published telemanom ranks best (AUC-PR 0.576 / 0.473); Chronos-2 zero-shot ≈ classic baselines; a late-alarm reference beats most detectors on corrected F0.5 |
| Telemetry (ESA-ADB, companion study) | alarm-protocol gain on Mission 1 +0.353 (95 % CI +0.206 to +0.488); reversed on Mission 2 (-0.089) |

## Re-entry

| Model | Pinball | MAE | Cov. 90 % | Width 90 % | C | Pinball | MAE | Cov. 90 % | Width 90 % | C |
|---|---|---|---|---|---|---|---|---|---|---|
| Physics expert alone | 0.0892 | 0.230 | 0.876 | 0.67 | 0.739 | 0.0527 | 0.148 | 0.974 | 1.03 | 0.766 |
| XGBoost (baseline) | 0.0576 | 0.190 | 0.921 | 0.80 | 0.888 | 0.0648 | 0.231 | 0.894 | 1.01 | 0.807 |
| LightGBM (baseline) | 0.0662 | 0.191 | 0.971 | 1.00 | 0.887 | 0.0683 | 0.221 | 0.981 | 1.18 | 0.811 |
| CatBoost (baseline) | 0.0611 | 0.207 | 0.911 | 0.93 | 0.878 | 0.0627 | 0.201 | 0.788 | 0.98 | 0.821 |
| Quantile MLP (baseline) | 0.0578 | 0.193 | 0.892 | 0.84 | 0.851 | 0.1147 | 0.390 | 0.852 | 1.40 | 0.747 |
| XGBoost AFT, censoring-aware (baseline) | 0.0657 | 0.185 | 0.901 | 0.62 | 0.866 | 0.0531 | 0.156 | 0.961 | 0.87 | 0.813 |
| TabM (2025) | 0.0593 | 0.192 | 0.877 | 0.68 | 0.861 | 0.0994 | 0.354 | 0.918 | 1.38 | 0.783 |
| TabICLv2 (2026) | 0.0499 | 0.166 | 0.962 | 1.00 | 0.894 | 0.0596 | 0.160 | 0.919 | 1.29 | 0.844 |
| Kumo-Tabular (2026) | 0.0508 | 0.162 | 0.974 | 1.07 | 0.899 | 0.0616 | 0.163 | 0.966 | 1.34 | 0.845 |
| Physics + expert, equal weights | 0.0691 | 0.205 | 0.838 | 0.64 | 0.859 | 0.0527 | 0.148 | 0.970 | 1.04 | 0.851 |
| Physics + expert, constant weight | 0.0601 | 0.193 | 0.877 | 0.67 | 0.861 | 0.0608 | 0.162 | 0.960 | 1.31 | 0.847 |
| Reliability-gated, gate fitted on tune-A | 0.0695 | 0.202 | 0.832 | 0.62 | 0.863 | 0.0616 | 0.163 | 0.959 | 1.34 | 0.845 |
| **Reliability-gated (COSMOS)** | 0.0604 | 0.193 | 0.874 | 0.67 | 0.860 | 0.0583 | 0.157 | 0.959 | 1.21 | 0.848 |

### Test period 2016–2021 (origin 2016)

**Selection.** On tune-A, TabM had the lowest loss among the recent learners (Supplementary Fig. S1 plots tune-A against
test loss for every learner):

| Learner | Tune-A pinball |
|---|---|
| TabM | 0.0283 |
| TabICLv2 | 0.0369 |
| Kumo-Tabular | 0.0433 |

On the test cohort the ranking is different, and TabM falls behind both foundation models:

| Model | Test pinball |
|---|---|
| TabICLv2 | 0.0499 |
| Kumo-Tabular | 0.0508 |
| TabM | 0.0593 |
| Classic baselines (best: XGBoost) | 0.0576 |
| Physics expert alone | 0.0892 |

Both foundation models are significantly better than the selected TabM (post-hoc paired
comparisons, Supplementary Table S16; a negative difference favours the first model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| TabICLv2 minus TabM | -0.0093 | -0.0162 to -0.0036 |
| Kumo-Tabular minus TabM | -0.0085 | -0.0167 to -0.0017 |
| TabICLv2 minus Kumo-Tabular | -0.0008 | -0.0036 to +0.0024 |

The two foundation models cannot be separated from each other.

**Coverage and concordance.** The raw 90 % quantile intervals of the two in-context foundation
models already cover 95.8 % (TabICLv2) and
97.3 % (Kumo-Tabular) of test objects. TabM's cover
80.8 %.

After Mondrian calibration on tune-B, TabM and the gated model remain below nominal coverage
(87.7 % and 87.4 %). This is consistent with a
loss of exchangeability between the 2009–2010 calibration cohort and the test cohort, although
the shortfall is small relative to the launch-clustered sampling error.
Their intervals are narrower, and their Winkler scores slightly lower:

| Model | Winkler score (90 %) |
|---|---|
| TabM | 1.099 |
| TabICLv2 | 1.129 |

Supplementary Fig. S2 plots coverage against interval width for every model, and
the figure shows individual forecasts and, for every test object with a known
five-year outcome, predicted against observed outcomes.

The foundation models also have the best concordance (C = 0.894 and
0.899). A tuning cohort five to seven years older than the test cohort
did not identify the model that generalises best.

**Gated model.** The reliability-gated model fuses the physics expert with TabM; its mean gate
weight on physics on the test cohort is 0.11. Its
pinball loss of 0.0604 compares as follows (paired cluster
bootstrap; a positive difference favours the gated model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| Physics expert alone | +0.0289 | +0.0148 to +0.0469 |
| TabM alone | -0.0011 | -0.0028 to +0.0002 |
| TabICLv2 | -0.0104 | -0.0179 to -0.0043 |
| Kumo-Tabular | -0.0096 | -0.0181 to -0.0023 |

The gated model is clearly better than physics alone and no different from TabM alone. It is
significantly worse than both foundation models.

**Horizon probabilities.** The foundation models also lead for re-entry within 30 days:

| Model | AUC-PR | Brier score |
|---|---|---|
| Kumo-Tabular | 0.883 | 0.036 |
| TabICLv2 | 0.874 | 0.035 |
| Gated model | 0.829 | 0.051 |
| Physics expert alone | 0.719 | 0.051 |

**Solar activity.** Solar cycle 25 rose faster than the persistence assumption allows. Replaying
the realised F10.7 lowers the physics expert's pinball loss from
0.0892 to 0.0783.
This is a diagnostic only: solar activity after the prediction time accounts for a substantial
part of the physics expert's error in this period.

### Test period 2011–2015 (origin 2011)

**Selection.** At this origin the tuning cohort is small: 248 objects in
tune-A and 332 in tune-B. Kumo-Tabular had the lowest tune-A loss:

| Learner | Tune-A pinball |
|---|---|
| Kumo-Tabular | 0.0547 |
| TabM | 0.0561 |
| TabICLv2 | 0.0581 |

**Physics expert against the foundation models.** The physics expert alone has the lowest point
estimate of the test pinball loss (0.0527), but it cannot be
separated from the foundation models or from the gated model (post-hoc paired comparisons,
Supplementary Table S16; a negative difference favours the physics expert):

| Physics expert minus | Pinball difference | 95 % CI |
|---|---|---|
| TabICLv2 | -0.0069 | -0.0217 to +0.0122 |
| Kumo-Tabular | -0.0089 | -0.0253 to +0.0133 |
| Gated model | -0.0056 | -0.0199 to +0.0139 |

The physics expert's 90 % conformal coverage is 97.4 %, but its
concordance is low (C = 0.766).

**TabM.** TabM, the expert selected at 2016, has a much higher loss here
(0.0994), as does the plain MLP (0.1147).

**By object class** (post hoc, Supplementary Table S2 and Fig. S3):

| Model | Fragments | Rocket stages | Components |
|---|---|---|---|
| TabM | 0.1122 | 0.0473 | 0.0597 |
| Physics expert | 0.0432 | 0.0733 | 0.0959 |

TabM's error is concentrated in fragments. The physics expert has the lowest loss for fragments,
while the learned experts have lower losses for rocket stages and components (e.g. TabICLv2:
0.0469 and 0.0703).

**Gated model.** The gated model fuses the physics expert with Kumo-Tabular (mean physics weight
0.12) and scores 0.0583.
Paired differences (a positive difference favours the gated model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| Kumo-Tabular alone | +0.0033 | +0.0007 to +0.0055 |
| XGBoost | +0.0066 | +0.0017 to +0.0136 |
| LightGBM | +0.0101 | +0.0023 to +0.0215 |
| CatBoost | +0.0044 | -0.0011 to +0.0122 |
| TabM | +0.0412 | +0.0011 to +0.0905 |
| TabICLv2 | +0.0013 | -0.0016 to +0.0044 |
| Physics expert alone | -0.0056 | -0.0199 to +0.0139 |

The 95 % intervals exclude zero for Kumo-Tabular alone, XGBoost, LightGBM and TabM, but not for
CatBoost, TabICLv2 or the physics expert. The censoring-aware AFT baseline scores
0.0531.

**Solar activity.** Replaying the realised F10.7 changes the physics expert's pinball loss only
from 0.0527 to
0.0516.

### Across both origins

All intervals in this section are nominal 95 % intervals without correction for multiplicity.
Apart from the comparisons of the gated model, the analyses below were added after the test
results were seen, so we read them as exploratory. the figure collects the
head-to-head comparisons.

1. **TabICLv2 was best or tied in both periods.** At 2016 the two foundation models are
   significantly better than the physics expert, the gated model and the best tree baseline;
   for example, TabICLv2 minus XGBoost is
   -0.0076 (-0.0121 to -0.0036).
   At 2011 TabICLv2, the physics expert and the gated model are statistically tied; Kumo-Tabular
   is tied with TabICLv2 and the physics expert but significantly worse than the gated model that
   fuses it (+0.0033,
   +0.0007 to +0.0055).
2. **Selection on the tuning cohort chose a significantly worse expert at 2016.** There, TabM was
   selected over the two foundation models. At 2011 the selected Kumo-Tabular is tied with
   TabICLv2.
3. **Gating improved the expert it fused at 2011 and was neutral at 2016.** In a post-hoc
   sensitivity analysis that gates every learner (Supplementary Table S3):
   - at 2011 the intervals exclude zero for every learner to which the gate gave non-negligible
     weight (TabICLv2: +0.0027);
   - at 2016 no interval excludes zero, and the point estimates are mostly slightly negative
     (TabICLv2 -0.0046, -0.0112 to
     +0.0007). This is when the physics expert's own error
     was largest.

**First learner set.** The deviation record (the deviations paragraph above, Supplementary Table S1)
used XGBoost, LightGBM, CatBoost, the plain MLP and TabICLv2 as candidates. It selected the MLP
at both origins. Its gated model scored 0.1074 at 2011 and
0.0592 at 2016. The reported 2011 gated model therefore depends on
the change of the candidate pool.

**Operational windows** (Supplementary Table S5). The ±20 % residual-lifetime window around the
physics median, the rule used for the final days before re-entry [@pardini2018assessing], covers
12.2 % (2016) and
17.1 % (2011) of the test objects that decayed within
five years, against 72.6 % and
90.6 % for the physics expert's 90 % conformal
interval. The shortfall is a property of forecasts made from a canonical orbit years earlier, not of
the horizon: the window covers a similar share of the objects that decayed within a week of their
canonical-orbit epoch. The rule's reported ~90 % coverage refers to the final days before re-entry,
when the orbit is tracked continuously. the figure shows the same rules over all test
objects, including those still in orbit after five years, where the ±20 % window scores higher
because it is wide enough to contain the five-year cap.

### An outcome leak through catalogue covariates

Whether an object has a radar cross-section in the 2026 catalogue depends on its fate.

**Evidence of the leak.** In the origin-2016 training cohort:
- 75.9 % of objects without a value decayed within
  30 days of their canonical-orbit epoch, against 17.2 %
  of objects with one.
- Adjusted for orbit, class and decade, the odds ratio is
  3.59. Its 95 % CI,
  2.09–6.17,
  uses cluster-robust standard errors over launch families.

The test periods differ in how this availability relates to the outcome:
- In the 2011–2015 test cohort, only 1.9 % of objects lack a
  value, and 84.0 % of them decayed within 30 days.
- In the 2016–2021 cohort, the value is missing for 99.9 %
  of objects catalogued from 2018 on, whatever their fate.

**Effect on the selected learner** (refitted with the 2026 RCS added, a labelled demonstration
and never a candidate):

| Origin | Pinball with RCS | Pinball without | Difference (95 % CI) | 90 % coverage with RCS | 90 % coverage without |
|---|---|---|---|---|---|
| 2011 | 0.0578 | 0.0616 | -0.0038 (-0.0072 to +0.0011) | 96.1 % | 96.6 % |
| 2016 | 0.0664 | 0.0593 | +0.0071 (-0.0010 to +0.0151) | 76.8 % | 87.7 % |

**Pinball loss.** The change from adding the RCS is within noise at both origins. It is also not
consistent across learner sets: with the first set, adding it worsened the selected MLP at 2011
(0.1221 against 0.1147).

**Interval coverage.** The consistent effect is on coverage in the period whose missingness
pattern differs from training. At 2016, 90 % coverage falls from 87.7 % to
76.8 %. With the first learner set it falls from
89.2 % to 72.1 %.

**Conclusion.** The association between availability and outcome is clear, and using the
covariate breaks calibration when that association changes. Its effect on point accuracy was not
detectable here.

### Feature groups and coupling C1 (family evidence)

Each feature group was removed from the selected learner and the learner refitted with the same
settings and seeds. The table below gives the paired launch-family bootstrap of the change
in test pinball loss; a positive change means the group helps. The intervals are nominal 95 %
intervals without correction for the eight comparisons.

| Group removed | Origin 2016 (TabM) | 95 % CI | Origin 2011 (Kumo-Tabular) | 95 % CI |
|---|---|---|---|---|
| Family evidence (C1) | +0.0056 | +0.0015 to +0.0101 | +0.0024 | -0.0058 to +0.0103 |
| Physics summaries | -0.0050 | -0.0086 to -0.0022 | +0.0022 | +0.0002 to +0.0039 |
| Design properties | +0.0071 | +0.0035 to +0.0121 | +0.0008 | -0.0001 to +0.0017 |
| Space weather | +0.0038 | -0.0057 to +0.0129 | +0.0009 | -0.0016 to +0.0038 |

**Family evidence (C1).** Siblings from the same launch and breakup that were catalogued, and
had decayed, before the prediction time improve TabM at 2016. At 2011 the point estimate is
positive but the interval includes zero.

**Physics summaries.** They help Kumo-Tabular at 2011, by a margin that would not survive a
multiplicity correction, and hurt TabM at 2016. A plausible reason, not tested here: in 2016–2021 the
physics expert's own error grew with solar cycle 25, and a learner that leans on its summaries
inherits that error.

**Design properties.** They help at 2016.

## Collision risk

**Setup.** The official Kelvins test set has 2,167 events, of which
150 are high-risk. Kumo-Tabular
had the lowest out-of-fold pinball loss among the recent learners and became the expert. The
gate then put almost no weight on the persistence expert (mean
0.0002), so the gated model is in effect Kumo-Tabular.

**Probabilistic forecast.** Kumo-Tabular's final-risk distribution has the lowest pinball loss.
Paired bootstrap over test events (post hoc; a negative difference favours Kumo-Tabular):
against TabICLv2 -0.039 (-0.053 to -0.025),
against XGBoost -0.202 (-0.233 to -0.171)
and against persistence -0.392 (-0.451 to -0.337).

| Model | Pinball loss | AUC-PR of P(high risk) | AUC-ROC of P(high risk) |
|---|---|---|---|
| Kumo-Tabular | 0.653 | 0.733 | 0.970 |
| TabICLv2 | 0.692 | 0.665 | 0.961 |
| Best tree baseline (XGBoost) | 0.855 | 0.606 | 0.945 |
| Persistence | 1.045 | 0.664 | 0.944 |

Its conformal 90 % intervals cover 94.2 % of all test events but only
48.0 % of the high-risk ones. With
66 high-risk training events, the group-conditional calibration is
coarse exactly where it matters.

**Official point metric.** It does not reward these gains.
- Every model's median forecast falls below the high-risk threshold for most high-risk events, so
  its $F_2$ is low: 129 of the 150 high-risk
  events receive a median below −6 from Kumo-Tabular, giving $F_2$ = 0.17. Most of
  those medians are not at the −30 floor but just below the threshold
  (41 lie between −10 and −6), so the floor is not the main cause.
- The latest-risk baseline scores $L$ = 0.694, the published
  value.
- Gating the latest-risk point with each model's calibrated P(high risk) brings the best models
  close to the baseline:
  - TabM: 0.698 (difference from the baseline, 95 % CI
    -0.005 to +0.019);
  - the gated model: 0.704
    (-0.055 to +0.071).
- No model beats the baseline.

This agrees with the challenge, where few teams beat the latest risk, and with a pre-registered
re-evaluation [@uriot2022spacecraft; @vazquez2026latest].

| Model | L (clipped) | F2 | MSE_HR | Pinball | AUC-PR P(high) | Brier P(high) | Cov. 90 % | Cov. 90 %, high risk |
|---|---|---|---|---|---|---|---|---|
| Latest risk (challenge baseline) | 0.694 | 0.739 | 0.513 | — | — | — | — | — |
| Persistence + residual climatology | ∞ | 0.000 | 1.752 | 1.045 | 0.664 | 0.0523 | 0.910 | 0.593 |
| XGBoost (baseline) | ∞ | 0.000 | 1.752 | 0.855 | 0.606 | 0.0552 | 0.917 | 0.433 |
| LightGBM (baseline) | ∞ | 0.000 | 1.752 | 0.861 | 0.541 | 0.0601 | 0.905 | 0.253 |
| CatBoost (baseline) | ∞ | 0.000 | 1.752 | 0.869 | 0.542 | 0.0564 | 0.912 | 0.467 |
| Quantile MLP (baseline) | ∞ | 0.000 | 1.752 | 1.035 | 0.434 | 0.0653 | 0.936 | 0.853 |
| TabM (2025) | ∞ | 0.000 | 1.752 | 0.885 | 0.512 | 0.0591 | 0.919 | 0.727 |
| TabICLv2 (2026) | 30.059 | 0.057 | 1.728 | 0.692 | 0.665 | 0.0489 | 0.918 | 0.327 |
| Kumo-Tabular (2026) | 9.892 | 0.168 | 1.662 | 0.653 | 0.733 | 0.0453 | 0.942 | 0.480 |
| Persistence + expert, constant weight | 9.898 | 0.168 | 1.663 | 0.653 | 0.733 | 0.0453 | 0.935 | 0.480 |
| **Reliability-gated (COSMOS)** | 9.896 | 0.168 | 1.662 | 0.653 | 0.733 | 0.0453 | 0.940 | 0.480 |

## Couplings and twin

### C2: from per-fragment re-entry to breakup-cloud survival

**Setup.** C2 covers the test folds of both origins: 160
cloud-years in 27 breakup clouds, each with at least 30
catalogued fragments. For each cloud we predict the number of its fragments still in orbit one
to five years after its first catalogued fragment, by summing per-fragment survival
probabilities, and compare it with the catalogue count.

**Results** (mean absolute error as a fraction of the cloud, with 95 % intervals from a
bootstrap over clouds; Supplementary Table S17):

| Model | Raw reading | 95 % CI | Calibrated reading | 95 % CI |
|---|---|---|---|---|
| Physics expert alone | 0.129 | 0.078–0.191 | 0.145 | 0.111–0.187 |
| TabICLv2 | 0.153 | 0.119–0.188 | 0.095 | 0.068–0.123 |
| Gated model | 0.155 | 0.121–0.192 | 0.122 | 0.085–0.161 |

- **Raw reading.** The physics expert has the lowest point error. Its mean bias is
  +0.007, and 73.1 % of
  cloud-years are within 10 %. The learned experts under-predict survival on this reading, but
  the physics expert cannot be separated from the gated model (difference
  -0.025, 95 % CI
  -0.084 to
  +0.033).
- **Calibrated reading** (isotonic maps of $P(T \le a)$ fitted on tune-B fragments). Calibration
  significantly lowers the error of TabICLv2
  (+0.058,
  +0.034 to +0.082)
  and of the gated model (+0.033,
  +0.007 to
  +0.055). Calibrated TabICLv2 has the lowest
  point error, with 63.1 % of cloud-years within 10 %, but it
  cannot be separated from the raw physics expert
  (+0.035,
  -0.007 to
  +0.078). Calibration does not help the physics
  expert.

The calibrated reading and the cloud bootstrap were added after the raw reading had been seen
and criticised in an internal review; they are post hoc.

**What C2 shows.** With 27 clouds, the errors of the best
readings are about a tenth of a cloud's size, and the uncertainty is too large to rank the
experts. How survival is read off the predictive distribution matters for the learned experts.
In the twin, cloud forecasts are sums of the history-conditioned physics probabilities. T1 tests
the calibration of those probabilities on objects in orbit.

### C3: operational status as health evidence for the re-entry regime

**Cohort.** On 2026-08-16, 13,794 Earth-orbiting payloads had perigee below 600 km. In the
following 72 days, 182 of them decayed; the decays came from
86 of their 1,070 launches.

**Physics by status.** The drag physics over-predicts re-entry for payloads reported
operational:

| Group | n | Decays | Mean predicted P | Observed rate |
|---|---|---|---|---|
| Operational | 13,478 | 164 | 0.039 | 0.012 |
| Non-operational | 161 | 17 | 0.076 | 0.106 |

- **Operational payloads.** The physics probabilities nevertheless rank them well (AUC-ROC
  0.959), consistent with operators letting some low
  payloads decay while keeping others in orbit.
- **Non-operational payloads.** The physics expert under-predicts somewhat, with AUC-ROC
  0.935.

**As-of-date policies** (information at 2026-08-16 only):

| Policy | Brier score | ECE |
|---|---|---|
| Drag physics for every payload | 0.0179 | 0.0267 |
| Previous-window base rate (0.0127) for every payload | 0.0130 | 0.0005 |
| Status-aware twin (base rate for operational payloads, physics otherwise) | 0.0125 | 0.0008 |

The status-aware twin has the lowest Brier score, but neither difference is significant once
launches are resampled:

| Status-aware twin minus | Brier difference | 95 % CI |
|---|---|---|
| Physics for every payload | -0.0054 | -0.0120 to 0.0006 |
| Base rate for every payload | -0.0005 | -0.0012 to 0.0001 |

With 182 decays in one window, the test can show the direction of the effect but
cannot establish it. The window is also not fresh: an earlier version of the project had fitted
and tested re-entry classifiers on it.

**In-window diagnostic (not a forecast).** We also fitted Platt recalibrations on the follow-up
window itself, cross-fitted by launch. Recalibrating per status rather than for all payloads
changes the Brier score by
-0.0004
(95 % CI
-0.0010
to
0.0001).
Even with look-ahead, then, status adds little once the physics probabilities are rescaled. The
regime split is what keeps the twin from issuing drag forecasts for payloads whose operators
control their orbits.

### T1: the twin's re-entry forecasts for objects in orbit

**Cohort.** As of 2026-08-16, the twin held 13,186 fragments, rocket stages and
components with perigee below 2000 km. Within 72 days, 64
of them decayed, 29 within 30 days.

**Results at 72 days** (expected re-entries, and their ratio to the observed number):

| Variant | Expected | Exp./obs. | 95 % CI | Brier score |
|---|---|---|---|---|
| History-conditioned physics | 87 | 1.37 | 1.20–1.63 | 0.00217 |
| Physics with class priors | 233 | 3.64 | 2.90–4.68 | 0.00625 |
| Lifetime pipeline (gated TabM) | 610 | 9.5 | 6.9–13.0 | 0.01344 |

- **Lifetime pipeline.** Applied to current orbits, it over-predicts re-entry about ninefold,
  although it was the validated model at canonical-orbit epochs. Objects that have stayed in
  orbit for years are the survivors, with low area-to-mass ratios, and a model fitted at
  canonical-orbit epochs does not know this. Its TabM expert is refitted on the GPU, whose
  training is not bit-reproducible: an earlier run of the same configuration gave a ratio of 8.9.
- **Physics with class priors.** It over-predicts about fourfold.
- **History-conditioned physics.** Conditioning the area-to-mass ratio on the object's catalogue
  history cuts the excess to a factor of about 1.4. This is still significantly above one.

**Brier differences** (launch-cluster bootstrap):

| History-conditioned physics minus | Brier difference | 95 % CI |
|---|---|---|
| Physics with class priors | -0.00408 | -0.00598 to -0.00266 |
| Lifetime pipeline | -0.01127 | -0.01812 to -0.00632 |

**Ranking.** Rank metrics use the raw probabilities. By AUC-PR the history-conditioned variant
ranks best at 72 days: 0.737, against
0.601 (class priors) and
0.616 (lifetime pipeline). By AUC-ROC the three
are within 0.01 of each other (0.991,
0.988 and
0.995), so all three separate decaying objects
from the rest about equally well. What differs is mainly calibration. At 30 days the AUC-PR
order is the same, and by AUC-ROC the history-conditioned variant is slightly lower
(0.982, against
0.999 and
0.998).

**Payloads.**
- *Operational payloads* (operator-controlled regime). The C3 policy expected
  170 decays against
  164 observed. Drag physics would
  have expected 498. The
  base-rate policy cannot rank payloads, though: its AUC-ROC is
  0.58, against
  0.96 for drag physics.
- *Non-operational payloads.* The physics expert expected
  11 against
  17 observed, the same under-prediction as
  in C3.

**The whole twin.** Over all 30,826 objects, the twin's per-regime choice expected
272 re-entries against
246 observed (Brier 0.0065).

**Caveats.**
- T1 was specified before it was run, but it is not a blind test. An earlier version of the
  project had fitted and tested re-entry classifiers on the same window, and the window's
  aggregate decay count was seen while the lifetime pipeline's bias on current orbits was being
  diagnosed, before the history-conditioned variant was designed.
- A defect in the payload prior was corrected after T1 had first been scored. Payload numbers changed; those of fragments, rocket stages and components
  did not.

## The dated twin forecast of 2026-10-27

**Scope.** As of 2026-10-27 the twin holds 31,396 objects in orbit with perigee below
2000 km:
- 13,787 in the natural-decay regime;
- 16,386 operator-controlled payloads;
- 1,223 payloads without a reported status.

**History conditioning** (the figure shows three examples). Among fragments, rocket stages and components:
- for 12,606, the catalogue history determined the area-to-mass
  posterior (effective sample size of at least 4);
- for 410, survival alone was used;
- for 56, no prior sample survived, and the longest
  survivors were used;
- 166 objects without a usable canonical orbit kept the class prior.

**Expected re-entries in the next 30 days** (natural-decay regime): 24.5.
For fragments, rocket stages and components, the three variants differ by an order of
magnitude:

| Variant | Expected re-entries, 30 days |
|---|---|
| History-conditioned physics | 21.6 |
| Physics with class priors | 76.3 |
| Gated lifetime pipeline, fitted at canonical-orbit epochs | 229.1 |

T1 indicates which of these to believe. the figure maps the forecast and
lists the natural-decay objects with the highest 30-day probability.

**Clouds and verification.** The forecast includes breakup-cloud forecasts for the
61 clouds with at least 30 fragments in orbit. Its record file is fixed by its
SHA-256 hash, `19a1fede1fd4871f380ff802f285d2ad03732f6cd3179790e67336c7d202a6b5`, so that later catalogue snapshots can score it without the
possibility of revision. An earlier forecast of the same date (hash beginning `fad1c619`)
contained the payload-prior defect described in the deviations paragraph above. It was replaced
before any later snapshot existed, and it is archived with the code.

## Health

### Health module: NASA SMAP/MSL telemetry and the companion ESA-ADB study

**Setup.** On the 81 channels (68 SMAP and
36 MSL anomaly sequences), detectors are compared at a level and
persistence fixed in advance.

**Ranking** (the figurea). The published telemanom scores [@hundman2018] rank anomalous steps best. Their
per-channel mean AUC-PR is 0.576 (SMAP) and
0.473 (MSL). Used zero-shot, the Chronos-2 foundation
model reaches 0.343 and
0.367. That is level with the ridge, PCA and
nearest-neighbour baselines, and above the random-score floor of
0.129 and 0.141.

**Alarms** (the figureb). On the ESA-ADB corrected event-wise $F_{0.5}$, which charges alarm time on nominal
data, the highest detector scores are PCA on SMAP (0.249) and the
nearest-neighbour detector on MSL (0.334). Chronos-2's residual
alarms are too frequent on MSL (508 false-positive segments),
giving a corrected $F_{0.5}$ of 0.035 there, against
0.170 on SMAP. That contrast is not a property of the spacecraft:
one MSL channel contributes 414 of those segments, and three contribute 95 % of them. They are
channels whose training series is constant, so every calibration score sits at the score floor and
any non-zero test residual raises an alarm.

**Why these alarm scores are weak evidence.**
- An always-on alarm reaches an uncorrected event $F_1$ of
  0.548 (SMAP) and 0.537 (MSL),
  higher than every detector. Its corrected $F_{0.5}$ is 0.
- 93.7 % of labelled steps lie in the second half of
  their channel's test series. A label-free reference that keeps each detector's amount of alarm
  time but places it as one block at the end of the series scores higher on the corrected
  $F_{0.5}$ than the detector itself in seven of the eight detector–spacecraft pairs; for
  example, 0.306 against 0.249 for
  PCA on SMAP. The exception is the nearest-neighbour detector on MSL
  (0.334 against 0.281).

We therefore read skill against a null that keeps each detector's alarm positions but
reassigns them across channels. Every fitted detector exceeds it on both spacecraft. For
example, ridge's event $F_1$ is 0.295 against
0.136 under the null (permutation $p$ =
0.005). This shows channel-specific detection, but a
modest one: on this benchmark, alarms that are merely late are rewarded about as much.
the figure shows the scores and alarms on one typical channel of each spacecraft.

**Companion ESA-ADB study.** The health module's alarm layer (conformal p-values,
Benjamini–Hochberg aggregation across channels and persistence) was developed and tested in a separate,
pre-registered study on ESA-ADB [@kotowski2026esaadb] by the same authors. On Mission 1, fitting
an AR(8) forecaster and selecting the alarm settings on the whole training half, rather than on a
short calibration window, raised the corrected event-wise $F_{0.5}$ from
0.074 to 0.428
(difference +0.353, 95 % CI
+0.206 to +0.488), at about
1 false alarm per year. On Mission 2 the same
contrast reversed (-0.089,
-0.143 to -0.049). Validation analyses of
that study attributed the Mission 1 gain to the labels used to select the alarm settings, not to
the fitted scores.

**Summary.** The recent forecasting foundation model does not improve on simple detectors on
SMAP/MSL, and the alarm protocol that the twin's health module uses helped on one ESA-ADB
mission and hurt on the other. The health module is therefore the least validated of the three.

### Computational cost

Times are measured on the origin-2016 run (25,334 training objects,
2,993 test objects). Hardware:
- CPU learners, TabICLv2 and the physics expert: a two-socket Intel Xeon 6980P node (512 hardware
  threads).
- TabM and Kumo-Tabular: an Intel Arc Pro B60 GPU.

| Model | Fit and predict per seed (s) | One call over the test cohort (s) |
|---|---|---|
| Physics expert (32 Monte-Carlo samples) | 128 | 34 |
| XGBoost | 26 | 0.029 |
| TabM | 60 | 0.063 |
| Kumo-Tabular | 94 | 95 |
| TabICLv2 | 915 | 617 |

**What is timed.** The first column is the wall-clock of the fit-and-predict loop divided by the
five seeds: it contains four refits (the seed-0 fit happens during tuning and is not timed) and
the predictions of all five seeds on the tuning and test cohorts. It is therefore an upper bound
on fitting alone, and for the in-context models it is dominated by prediction. The second column
is one call predicting the test cohort.

**In-context foundation models.** Their prediction cost includes encoding the training context,
up to 16,000 rows for Kumo-Tabular and 25,334 for TabICLv2 on CPU. Predicting
the same cohort takes them three to five orders of magnitude longer than a tree ensemble
(Kumo-Tabular 95 s and TabICLv2
617 s, against 0.029 s for
XGBoost and 0.005 s for CatBoost). Because the context encoding
is a fixed cost per call rather than per object, these times cannot be rescaled to a different
number of objects, and we do not extrapolate them to the full catalogue; both models predicted the
whole test cohort in one call.

**The physics features are part of every learner's cost.** All learners take five summaries of the
physics expert's Monte-Carlo distribution as inputs, so a deployed pipeline also pays that run:
34 s for the 2,993 test objects, which
dominates the tree ensembles' own prediction time by three orders of magnitude. The second column
above times the learner alone.

## Limitations and threats to validity

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
  odds ratio 3.4). We
  treat it as known at launch, because GCAT design values come from launch documentation. The
  `no_design` ablation bounds its influence.

**Resolution of the decay integrator.** A step may not drop the perigee by more than 2 % of its
height above 120 km, with a floor of 1 km, and may not exceed 10 days. For an object whose perigee
is already within 1 km of the 120 km threshold, the first step can therefore be a full 10 days, and
the forecast is reported at the end of that step. Two of the 13,787
natural-decay objects of the dated forecast have a degenerate 90 % interval at exactly 10 days for
this reason. The affected objects are within days of re-entry in any case, and no reported
comparison depends on them.

**Physics simplifications.**
- The density is a global NRLMSIS 2.1 average without latitude, local-time or seasonal
  variation.
- The atmosphere is exponential about perigee within an orbit.
- Lunisolar and J2 effects on perigee are not modelled. They matter for transfer-orbit stages.
- Solar activity is held at its value at the prediction time. The oracle diagnostic shows how
  much of the error this causes.

**Small calibration sets at the 2011 origin.** The 2004–2005 tuning cohort is small, so model
selection and calibration at that origin are noisy.

**Test-set reuse and post-hoc analyses.**
- The re-entry and collision-risk test sets were scored more than once during development:
  before an internal review, with a first learner set, and in smoke tests of the new learners.
- The pool of eligible learners was changed after the first set's test results were known.
- The head-to-head comparisons, the gate applied to every learner and the calibrated C2 reading
  were added after results were seen. Their intervals are nominal and uncorrected for
  multiplicity.

All of this is recorded in the deviations paragraph above and in the released logs. It weakens the
confirmatory value of the comparisons, and a fresh test period is needed to confirm them.

**Bit-level reproducibility.** Seeded training of TabM on the GPU is not bit-reproducible; on CPU
it is. The re-entry study reuses cached predictions, so its numbers are fixed, but a refit gives
slightly different results: two runs of the T1 lifetime-pipeline variant gave expected/observed
ratios of 8.9 and 9.5. Chronos-2
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

