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
| Best re-entry model, 2016–2021 | TabICLv2 ({{re.o2016.tabicl.pinball:.4f}}) and Kumo-Tabular ({{re.o2016.kumo.pinball:.4f}}), tied with each other and significantly better than the physics expert ({{re.o2016.physics_only.pinball:.4f}}), the gated model (TabM, {{re.o2016.cosmos_gated.pinball:.4f}}) and XGBoost ({{re.o2016.xgboost.pinball:.4f}}) |
| Best re-entry model, 2011–2015 | physics alone has the lowest point estimate ({{re.o2011.physics_only.pinball:.4f}}) but is tied with TabICLv2 ({{re.o2011.tabicl.pinball:.4f}}), Kumo-Tabular ({{re.o2011.kumo.pinball:.4f}}) and the gated model ({{re.o2011.cosmos_gated.pinball:.4f}}) |
| Does the reliability gate help? | 2011: yes, for every learner it weights (post hoc). 2016: no (neutral to slightly harmful). Never better than the best single expert |
| Did tune-A selection find the best learner? | 2016: no, TabM was chosen and is significantly worse than both foundation models. 2011: Kumo-Tabular, tied with TabICLv2 |
| RCS leak | Adjusted OR {{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.2f}} (cluster-robust 95 % CI {{leak.o2016.train.adjusted_or_rcs_missing.ci95.0:.2f}}–{{leak.o2016.train.adjusted_or_rcs_missing.ci95.1:.2f}}). Pinball changes within noise at both origins; 2016 90 % coverage falls from {{re.o2016.BEST.cov90:pct}} to {{re.o2016.BEST_plus_rcs_2026.cov90:pct}} |
| Twin forecasts for objects in orbit (T1, 72 d) | expected/observed: history-conditioned {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ratio:.2f}}, class prior {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ratio:.2f}}, lifetime pipeline {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.2f}} (as-of-date test on a previously analysed window; not blind) |
| Health → re-entry regime (C3) | Brier difference, status-aware minus physics-only: {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.mean:+.4f}} (95 % CI {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.ci95.0:+.4f}} to {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.ci95.1:+.4f}}), not significant |
| Collision risk (Kelvins) | best probabilistic: Kumo-Tabular (pinball {{cac.kumo.pinball:.3f}}; minus TabICLv2 {{cacp.kumo.tabicl.diff:+.3f}}, 95 % CI {{cacp.kumo.tabicl.lo:+.3f}} to {{cacp.kumo.tabicl.hi:+.3f}}; persistence {{cac.persistence.pinball:.3f}}); official metric: no model beats the latest risk ({{cac.latest_risk_point.L_clip:.3f}}) |
| Telemetry (SMAP/MSL) | published telemanom ranks best (AUC-PR {{smap.SMAP.telemanom.pointwise_auc_pr:.3f}} / {{smap.MSL.telemanom.pointwise_auc_pr:.3f}}); Chronos-2 zero-shot ≈ classic baselines; a late-alarm reference beats most detectors on corrected F0.5 |
| Telemetry (ESA-ADB, companion study) | alarm-protocol gain on Mission 1 {{esa.conf.m1.h6.diff_official:+.3f}} (95 % CI {{esa.conf.m1.h6.ci95_low:+.3f}} to {{esa.conf.m1.h6.ci95_high:+.3f}}); reversed on Mission 2 ({{esa.conf.m2.h6.diff_official:+.3f}}) |

## Re-entry

| Model | Pinball | MAE | Cov. 90 % | Width 90 % | C | Pinball | MAE | Cov. 90 % | Width 90 % | C |
|---|---|---|---|---|---|---|---|---|---|---|
| Physics expert alone | {{re.o2016.physics_only.pinball:.4f}} | {{re.o2016.physics_only.mae_log10:.3f}} | {{re.o2016.physics_only.cov90:.3f}} | {{re.o2016.physics_only.width90_log10:.2f}} | {{re.o2016.physics_only.c_index:.3f}} | {{re.o2011.physics_only.pinball:.4f}} | {{re.o2011.physics_only.mae_log10:.3f}} | {{re.o2011.physics_only.cov90:.3f}} | {{re.o2011.physics_only.width90_log10:.2f}} | {{re.o2011.physics_only.c_index:.3f}} |
| XGBoost (baseline) | {{re.o2016.xgboost.pinball:.4f}} | {{re.o2016.xgboost.mae_log10:.3f}} | {{re.o2016.xgboost.cov90:.3f}} | {{re.o2016.xgboost.width90_log10:.2f}} | {{re.o2016.xgboost.c_index:.3f}} | {{re.o2011.xgboost.pinball:.4f}} | {{re.o2011.xgboost.mae_log10:.3f}} | {{re.o2011.xgboost.cov90:.3f}} | {{re.o2011.xgboost.width90_log10:.2f}} | {{re.o2011.xgboost.c_index:.3f}} |
| LightGBM (baseline) | {{re.o2016.lightgbm.pinball:.4f}} | {{re.o2016.lightgbm.mae_log10:.3f}} | {{re.o2016.lightgbm.cov90:.3f}} | {{re.o2016.lightgbm.width90_log10:.2f}} | {{re.o2016.lightgbm.c_index:.3f}} | {{re.o2011.lightgbm.pinball:.4f}} | {{re.o2011.lightgbm.mae_log10:.3f}} | {{re.o2011.lightgbm.cov90:.3f}} | {{re.o2011.lightgbm.width90_log10:.2f}} | {{re.o2011.lightgbm.c_index:.3f}} |
| CatBoost (baseline) | {{re.o2016.catboost.pinball:.4f}} | {{re.o2016.catboost.mae_log10:.3f}} | {{re.o2016.catboost.cov90:.3f}} | {{re.o2016.catboost.width90_log10:.2f}} | {{re.o2016.catboost.c_index:.3f}} | {{re.o2011.catboost.pinball:.4f}} | {{re.o2011.catboost.mae_log10:.3f}} | {{re.o2011.catboost.cov90:.3f}} | {{re.o2011.catboost.width90_log10:.2f}} | {{re.o2011.catboost.c_index:.3f}} |
| Quantile MLP (baseline) | {{re.o2016.mlp.pinball:.4f}} | {{re.o2016.mlp.mae_log10:.3f}} | {{re.o2016.mlp.cov90:.3f}} | {{re.o2016.mlp.width90_log10:.2f}} | {{re.o2016.mlp.c_index:.3f}} | {{re.o2011.mlp.pinball:.4f}} | {{re.o2011.mlp.mae_log10:.3f}} | {{re.o2011.mlp.cov90:.3f}} | {{re.o2011.mlp.width90_log10:.2f}} | {{re.o2011.mlp.c_index:.3f}} |
| XGBoost AFT, censoring-aware (baseline) | {{re.o2016.aft.pinball:.4f}} | {{re.o2016.aft.mae_log10:.3f}} | {{re.o2016.aft.cov90:.3f}} | {{re.o2016.aft.width90_log10:.2f}} | {{re.o2016.aft.c_index:.3f}} | {{re.o2011.aft.pinball:.4f}} | {{re.o2011.aft.mae_log10:.3f}} | {{re.o2011.aft.cov90:.3f}} | {{re.o2011.aft.width90_log10:.2f}} | {{re.o2011.aft.c_index:.3f}} |
| TabM (2025) | {{re.o2016.tabm.pinball:.4f}} | {{re.o2016.tabm.mae_log10:.3f}} | {{re.o2016.tabm.cov90:.3f}} | {{re.o2016.tabm.width90_log10:.2f}} | {{re.o2016.tabm.c_index:.3f}} | {{re.o2011.tabm.pinball:.4f}} | {{re.o2011.tabm.mae_log10:.3f}} | {{re.o2011.tabm.cov90:.3f}} | {{re.o2011.tabm.width90_log10:.2f}} | {{re.o2011.tabm.c_index:.3f}} |
| TabICLv2 (2026) | {{re.o2016.tabicl.pinball:.4f}} | {{re.o2016.tabicl.mae_log10:.3f}} | {{re.o2016.tabicl.cov90:.3f}} | {{re.o2016.tabicl.width90_log10:.2f}} | {{re.o2016.tabicl.c_index:.3f}} | {{re.o2011.tabicl.pinball:.4f}} | {{re.o2011.tabicl.mae_log10:.3f}} | {{re.o2011.tabicl.cov90:.3f}} | {{re.o2011.tabicl.width90_log10:.2f}} | {{re.o2011.tabicl.c_index:.3f}} |
| Kumo-Tabular (2026) | {{re.o2016.kumo.pinball:.4f}} | {{re.o2016.kumo.mae_log10:.3f}} | {{re.o2016.kumo.cov90:.3f}} | {{re.o2016.kumo.width90_log10:.2f}} | {{re.o2016.kumo.c_index:.3f}} | {{re.o2011.kumo.pinball:.4f}} | {{re.o2011.kumo.mae_log10:.3f}} | {{re.o2011.kumo.cov90:.3f}} | {{re.o2011.kumo.width90_log10:.2f}} | {{re.o2011.kumo.c_index:.3f}} |
| Physics + expert, equal weights | {{re.o2016.equal_average.pinball:.4f}} | {{re.o2016.equal_average.mae_log10:.3f}} | {{re.o2016.equal_average.cov90:.3f}} | {{re.o2016.equal_average.width90_log10:.2f}} | {{re.o2016.equal_average.c_index:.3f}} | {{re.o2011.equal_average.pinball:.4f}} | {{re.o2011.equal_average.mae_log10:.3f}} | {{re.o2011.equal_average.cov90:.3f}} | {{re.o2011.equal_average.width90_log10:.2f}} | {{re.o2011.equal_average.c_index:.3f}} |
| Physics + expert, constant weight | {{re.o2016.stack_constant.pinball:.4f}} | {{re.o2016.stack_constant.mae_log10:.3f}} | {{re.o2016.stack_constant.cov90:.3f}} | {{re.o2016.stack_constant.width90_log10:.2f}} | {{re.o2016.stack_constant.c_index:.3f}} | {{re.o2011.stack_constant.pinball:.4f}} | {{re.o2011.stack_constant.mae_log10:.3f}} | {{re.o2011.stack_constant.cov90:.3f}} | {{re.o2011.stack_constant.width90_log10:.2f}} | {{re.o2011.stack_constant.c_index:.3f}} |
| Reliability-gated, gate fitted on tune-A | {{re.o2016.cosmos_gated_tuneA.pinball:.4f}} | {{re.o2016.cosmos_gated_tuneA.mae_log10:.3f}} | {{re.o2016.cosmos_gated_tuneA.cov90:.3f}} | {{re.o2016.cosmos_gated_tuneA.width90_log10:.2f}} | {{re.o2016.cosmos_gated_tuneA.c_index:.3f}} | {{re.o2011.cosmos_gated_tuneA.pinball:.4f}} | {{re.o2011.cosmos_gated_tuneA.mae_log10:.3f}} | {{re.o2011.cosmos_gated_tuneA.cov90:.3f}} | {{re.o2011.cosmos_gated_tuneA.width90_log10:.2f}} | {{re.o2011.cosmos_gated_tuneA.c_index:.3f}} |
| **Reliability-gated (COSMOS)** | {{re.o2016.cosmos_gated.pinball:.4f}} | {{re.o2016.cosmos_gated.mae_log10:.3f}} | {{re.o2016.cosmos_gated.cov90:.3f}} | {{re.o2016.cosmos_gated.width90_log10:.2f}} | {{re.o2016.cosmos_gated.c_index:.3f}} | {{re.o2011.cosmos_gated.pinball:.4f}} | {{re.o2011.cosmos_gated.mae_log10:.3f}} | {{re.o2011.cosmos_gated.cov90:.3f}} | {{re.o2011.cosmos_gated.width90_log10:.2f}} | {{re.o2011.cosmos_gated.c_index:.3f}} |

### Test period 2016–2021 (origin 2016)

**Selection.** On tune-A, TabM had the lowest loss among the recent learners (Supplementary Fig. S1 plots tune-A against
test loss for every learner):

| Learner | Tune-A pinball |
|---|---|
| TabM | {{re.o2016.tuneA_pinball.tabm:.4f}} |
| TabICLv2 | {{re.o2016.tuneA_pinball.tabicl:.4f}} |
| Kumo-Tabular | {{re.o2016.tuneA_pinball.kumo:.4f}} |

On the test cohort the ranking is different, and TabM falls behind both foundation models:

| Model | Test pinball |
|---|---|
| TabICLv2 | {{re.o2016.tabicl.pinball:.4f}} |
| Kumo-Tabular | {{re.o2016.kumo.pinball:.4f}} |
| TabM | {{re.o2016.tabm.pinball:.4f}} |
| Classic baselines (best: XGBoost) | {{re.o2016.xgboost.pinball:.4f}} |
| Physics expert alone | {{re.o2016.physics_only.pinball:.4f}} |

Both foundation models are significantly better than the selected TabM (post-hoc paired
comparisons, Supplementary Table S16; a negative difference favours the first model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| TabICLv2 minus TabM | {{re.o2016.pair.tabicl.tabm.diff:+.4f}} | {{re.o2016.pair.tabicl.tabm.lo:+.4f}} to {{re.o2016.pair.tabicl.tabm.hi:+.4f}} |
| Kumo-Tabular minus TabM | {{re.o2016.pair.kumo.tabm.diff:+.4f}} | {{re.o2016.pair.kumo.tabm.lo:+.4f}} to {{re.o2016.pair.kumo.tabm.hi:+.4f}} |
| TabICLv2 minus Kumo-Tabular | {{re.o2016.pair.tabicl.kumo.diff:+.4f}} | {{re.o2016.pair.tabicl.kumo.lo:+.4f}} to {{re.o2016.pair.tabicl.kumo.hi:+.4f}} |

The two foundation models cannot be separated from each other.

**Coverage and concordance.** The raw 90 % quantile intervals of the two in-context foundation
models already cover {{re.o2016.tabicl.raw_cov90:pct}} (TabICLv2) and
{{re.o2016.kumo.raw_cov90:pct}} (Kumo-Tabular) of test objects. TabM's cover
{{re.o2016.tabm.raw_cov90:pct}}.

After Mondrian calibration on tune-B, TabM and the gated model remain below nominal coverage
({{re.o2016.tabm.cov90:pct}} and {{re.o2016.cosmos_gated.cov90:pct}}). This is consistent with a
loss of exchangeability between the 2009–2010 calibration cohort and the test cohort, although
the shortfall is small relative to the launch-clustered sampling error.
Their intervals are narrower, and their Winkler scores slightly lower:

| Model | Winkler score (90 %) |
|---|---|
| TabM | {{re.o2016.tabm.winkler90:.3f}} |
| TabICLv2 | {{re.o2016.tabicl.winkler90:.3f}} |

Supplementary Fig. S2 plots coverage against interval width for every model, and
the figure shows individual forecasts and, for every test object with a known
five-year outcome, predicted against observed outcomes.

The foundation models also have the best concordance (C = {{re.o2016.tabicl.c_index:.3f}} and
{{re.o2016.kumo.c_index:.3f}}). A tuning cohort five to seven years older than the test cohort
did not identify the model that generalises best.

**Gated model.** The reliability-gated model fuses the physics expert with TabM; its mean gate
weight on physics on the test cohort is {{re.o2016.gate.cosmos_gated.weight_mean:.2f}}. Its
pinball loss of {{re.o2016.cosmos_gated.pinball:.4f}} compares as follows (paired cluster
bootstrap; a positive difference favours the gated model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| Physics expert alone | {{re.o2016.paired.physics_only.diff:+.4f}} | {{re.o2016.paired.physics_only.lo:+.4f}} to {{re.o2016.paired.physics_only.hi:+.4f}} |
| TabM alone | {{re.o2016.paired.tabm.diff:+.4f}} | {{re.o2016.paired.tabm.lo:+.4f}} to {{re.o2016.paired.tabm.hi:+.4f}} |
| TabICLv2 | {{re.o2016.paired.tabicl.diff:+.4f}} | {{re.o2016.paired.tabicl.lo:+.4f}} to {{re.o2016.paired.tabicl.hi:+.4f}} |
| Kumo-Tabular | {{re.o2016.paired.kumo.diff:+.4f}} | {{re.o2016.paired.kumo.lo:+.4f}} to {{re.o2016.paired.kumo.hi:+.4f}} |

The gated model is clearly better than physics alone and no different from TabM alone. It is
significantly worse than both foundation models.

**Horizon probabilities.** The foundation models also lead for re-entry within 30 days:

| Model | AUC-PR | Brier score |
|---|---|---|
| Kumo-Tabular | {{re.o2016.h.kumo.30d.auc_pr:.3f}} | {{re.o2016.h.kumo.30d.brier:.3f}} |
| TabICLv2 | {{re.o2016.h.tabicl.30d.auc_pr:.3f}} | {{re.o2016.h.tabicl.30d.brier:.3f}} |
| Gated model | {{re.o2016.h.cosmos_gated.30d.auc_pr:.3f}} | {{re.o2016.h.cosmos_gated.30d.brier:.3f}} |
| Physics expert alone | {{re.o2016.h.physics_only.30d.auc_pr:.3f}} | {{re.o2016.h.physics_only.30d.brier:.3f}} |

**Solar activity.** Solar cycle 25 rose faster than the persistence assumption allows. Replaying
the realised F10.7 lowers the physics expert's pinball loss from
{{re.o2016.oracle.persistence.pinball:.4f}} to {{re.o2016.oracle.oracle_realised_f107.pinball:.4f}}.
This is a diagnostic only: solar activity after the prediction time accounts for a substantial
part of the physics expert's error in this period.

### Test period 2011–2015 (origin 2011)

**Selection.** At this origin the tuning cohort is small: {{re.o2011.known5.tuneA}} objects in
tune-A and {{re.o2011.known5.tuneB}} in tune-B. Kumo-Tabular had the lowest tune-A loss:

| Learner | Tune-A pinball |
|---|---|
| Kumo-Tabular | {{re.o2011.tuneA_pinball.kumo:.4f}} |
| TabM | {{re.o2011.tuneA_pinball.tabm:.4f}} |
| TabICLv2 | {{re.o2011.tuneA_pinball.tabicl:.4f}} |

**Physics expert against the foundation models.** The physics expert alone has the lowest point
estimate of the test pinball loss ({{re.o2011.physics_only.pinball:.4f}}), but it cannot be
separated from the foundation models or from the gated model (post-hoc paired comparisons,
Supplementary Table S16; a negative difference favours the physics expert):

| Physics expert minus | Pinball difference | 95 % CI |
|---|---|---|
| TabICLv2 | {{re.o2011.pair.physics_only.tabicl.diff:+.4f}} | {{re.o2011.pair.physics_only.tabicl.lo:+.4f}} to {{re.o2011.pair.physics_only.tabicl.hi:+.4f}} |
| Kumo-Tabular | {{re.o2011.pair.physics_only.kumo.diff:+.4f}} | {{re.o2011.pair.physics_only.kumo.lo:+.4f}} to {{re.o2011.pair.physics_only.kumo.hi:+.4f}} |
| Gated model | {{re.o2011.pair.physics_only.cosmos_gated.diff:+.4f}} | {{re.o2011.pair.physics_only.cosmos_gated.lo:+.4f}} to {{re.o2011.pair.physics_only.cosmos_gated.hi:+.4f}} |

The physics expert's 90 % conformal coverage is {{re.o2011.physics_only.cov90:pct}}, but its
concordance is low (C = {{re.o2011.physics_only.c_index:.3f}}).

**TabM.** TabM, the expert selected at 2016, has a much higher loss here
({{re.o2011.tabm.pinball:.4f}}), as does the plain MLP ({{re.o2011.mlp.pinball:.4f}}).

**By object class** (post hoc, Supplementary Table S2 and Fig. S3):

| Model | Fragments | Rocket stages | Components |
|---|---|---|---|
| TabM | {{re.o2011.cls.tabm.D.pinball:.4f}} | {{re.o2011.cls.tabm.R.pinball:.4f}} | {{re.o2011.cls.tabm.C.pinball:.4f}} |
| Physics expert | {{re.o2011.cls.physics_only.D.pinball:.4f}} | {{re.o2011.cls.physics_only.R.pinball:.4f}} | {{re.o2011.cls.physics_only.C.pinball:.4f}} |

TabM's error is concentrated in fragments. The physics expert has the lowest loss for fragments,
while the learned experts have lower losses for rocket stages and components (e.g. TabICLv2:
{{re.o2011.cls.tabicl.R.pinball:.4f}} and {{re.o2011.cls.tabicl.C.pinball:.4f}}).

**Gated model.** The gated model fuses the physics expert with Kumo-Tabular (mean physics weight
{{re.o2011.gate.cosmos_gated.weight_mean:.2f}}) and scores {{re.o2011.cosmos_gated.pinball:.4f}}.
Paired differences (a positive difference favours the gated model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| Kumo-Tabular alone | {{re.o2011.paired.kumo.diff:+.4f}} | {{re.o2011.paired.kumo.lo:+.4f}} to {{re.o2011.paired.kumo.hi:+.4f}} |
| XGBoost | {{re.o2011.paired.xgboost.diff:+.4f}} | {{re.o2011.paired.xgboost.lo:+.4f}} to {{re.o2011.paired.xgboost.hi:+.4f}} |
| LightGBM | {{re.o2011.paired.lightgbm.diff:+.4f}} | {{re.o2011.paired.lightgbm.lo:+.4f}} to {{re.o2011.paired.lightgbm.hi:+.4f}} |
| CatBoost | {{re.o2011.paired.catboost.diff:+.4f}} | {{re.o2011.paired.catboost.lo:+.4f}} to {{re.o2011.paired.catboost.hi:+.4f}} |
| TabM | {{re.o2011.paired.tabm.diff:+.4f}} | {{re.o2011.paired.tabm.lo:+.4f}} to {{re.o2011.paired.tabm.hi:+.4f}} |
| TabICLv2 | {{re.o2011.paired.tabicl.diff:+.4f}} | {{re.o2011.paired.tabicl.lo:+.4f}} to {{re.o2011.paired.tabicl.hi:+.4f}} |
| Physics expert alone | {{re.o2011.paired.physics_only.diff:+.4f}} | {{re.o2011.paired.physics_only.lo:+.4f}} to {{re.o2011.paired.physics_only.hi:+.4f}} |

The 95 % intervals exclude zero for Kumo-Tabular alone, XGBoost, LightGBM and TabM, but not for
CatBoost, TabICLv2 or the physics expert. The censoring-aware AFT baseline scores
{{re.o2011.aft.pinball:.4f}}.

**Solar activity.** Replaying the realised F10.7 changes the physics expert's pinball loss only
from {{re.o2011.oracle.persistence.pinball:.4f}} to
{{re.o2011.oracle.oracle_realised_f107.pinball:.4f}}.

### Across both origins

All intervals in this section are nominal 95 % intervals without correction for multiplicity.
Apart from the comparisons of the gated model, the analyses below were added after the test
results were seen, so we read them as exploratory. the figure collects the
head-to-head comparisons.

1. **TabICLv2 was best or tied in both periods.** At 2016 the two foundation models are
   significantly better than the physics expert, the gated model and the best tree baseline;
   for example, TabICLv2 minus XGBoost is
   {{re.o2016.pair.tabicl.xgboost.diff:+.4f}} ({{re.o2016.pair.tabicl.xgboost.lo:+.4f}} to {{re.o2016.pair.tabicl.xgboost.hi:+.4f}}).
   At 2011 TabICLv2, the physics expert and the gated model are statistically tied; Kumo-Tabular
   is tied with TabICLv2 and the physics expert but significantly worse than the gated model that
   fuses it ({{re.o2011.pair.kumo.cosmos_gated.diff:+.4f}},
   {{re.o2011.pair.kumo.cosmos_gated.lo:+.4f}} to {{re.o2011.pair.kumo.cosmos_gated.hi:+.4f}}).
2. **Selection on the tuning cohort chose a significantly worse expert at 2016.** There, TabM was
   selected over the two foundation models. At 2011 the selected Kumo-Tabular is tied with
   TabICLv2.
3. **Gating improved the expert it fused at 2011 and was neutral at 2016.** In a post-hoc
   sensitivity analysis that gates every learner (Supplementary Table S3):
   - at 2011 the intervals exclude zero for every learner to which the gate gave non-negligible
     weight (TabICLv2: {{re.o2011.gateall.tabicl.gain:+.4f}});
   - at 2016 no interval excludes zero, and the point estimates are mostly slightly negative
     (TabICLv2 {{re.o2016.gateall.tabicl.gain:+.4f}}, {{re.o2016.gateall.tabicl.gain_ci95_lo:+.4f}} to
     {{re.o2016.gateall.tabicl.gain_ci95_hi:+.4f}}). This is when the physics expert's own error
     was largest.

**First learner set.** The deviation record (the deviations paragraph above, Supplementary Table S1)
used XGBoost, LightGBM, CatBoost, the plain MLP and TabICLv2 as candidates. It selected the MLP
at both origins. Its gated model scored {{re1.o2011.cosmos_gated.pinball:.4f}} at 2011 and
{{re1.o2016.cosmos_gated.pinball:.4f}} at 2016. The reported 2011 gated model therefore depends on
the change of the candidate pool.

**Operational windows** (Supplementary Table S5). The ±20 % residual-lifetime window around the
physics median, the rule used for the final days before re-entry [@pardini2018assessing], covers
{{re.o2016.rule.physics_only_pm20.coverage_decayed:pct}} (2016) and
{{re.o2011.rule.physics_only_pm20.coverage_decayed:pct}} (2011) of the test objects that decayed within
five years, against {{re.o2016.rule.physics_only_conformal90.coverage_decayed:pct}} and
{{re.o2011.rule.physics_only_conformal90.coverage_decayed:pct}} for the physics expert's 90 % conformal
interval. The shortfall is a property of forecasts made from a canonical orbit years earlier, not of
the horizon: the window covers a similar share of the objects that decayed within a week of their
canonical-orbit epoch. The rule's reported ~90 % coverage refers to the final days before re-entry,
when the orbit is tracked continuously. the figure shows the same rules over all test
objects, including those still in orbit after five years, where the ±20 % window scores higher
because it is wide enough to contain the five-year cap.

### An outcome leak through catalogue covariates

Whether an object has a radar cross-section in the 2026 catalogue depends on its fate.

**Evidence of the leak.** In the origin-2016 training cohort:
- {{leak.o2016.train.p_decay30_given_missing:pct}} of objects without a value decayed within
  30 days of their canonical-orbit epoch, against {{leak.o2016.train.p_decay30_given_present:pct}}
  of objects with one.
- Adjusted for orbit, class and decade, the odds ratio is
  {{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.2f}}. Its 95 % CI,
  {{leak.o2016.train.adjusted_or_rcs_missing.ci95.0:.2f}}–{{leak.o2016.train.adjusted_or_rcs_missing.ci95.1:.2f}},
  uses cluster-robust standard errors over launch families.

The test periods differ in how this availability relates to the outcome:
- In the 2011–2015 test cohort, only {{leak.o2011.test.share_rcs_missing:pct}} of objects lack a
  value, and {{leak.o2011.test.p_decay30_given_missing:pct}} of them decayed within 30 days.
- In the 2016–2021 cohort, the value is missing for {{leak.o2016.test.share_rcs_missing_t0_from_2018:pct}}
  of objects catalogued from 2018 on, whatever their fate.

**Effect on the selected learner** (refitted with the 2026 RCS added, a labelled demonstration
and never a candidate):

| Origin | Pinball with RCS | Pinball without | Difference (95 % CI) | 90 % coverage with RCS | 90 % coverage without |
|---|---|---|---|---|---|
| 2011 | {{re.o2011.BEST_plus_rcs_2026.pinball:.4f}} | {{re.o2011.BEST.pinball:.4f}} | {{re.o2011.abl.plus_rcs_2026.diff:+.4f}} ({{re.o2011.abl.plus_rcs_2026.lo:+.4f}} to {{re.o2011.abl.plus_rcs_2026.hi:+.4f}}) | {{re.o2011.BEST_plus_rcs_2026.cov90:pct}} | {{re.o2011.BEST.cov90:pct}} |
| 2016 | {{re.o2016.BEST_plus_rcs_2026.pinball:.4f}} | {{re.o2016.BEST.pinball:.4f}} | {{re.o2016.abl.plus_rcs_2026.diff:+.4f}} ({{re.o2016.abl.plus_rcs_2026.lo:+.4f}} to {{re.o2016.abl.plus_rcs_2026.hi:+.4f}}) | {{re.o2016.BEST_plus_rcs_2026.cov90:pct}} | {{re.o2016.BEST.cov90:pct}} |

**Pinball loss.** The change from adding the RCS is within noise at both origins. It is also not
consistent across learner sets: with the first set, adding it worsened the selected MLP at 2011
({{re1.o2011.mlp_plus_rcs_2026.pinball:.4f}} against {{re1.o2011.mlp.pinball:.4f}}).

**Interval coverage.** The consistent effect is on coverage in the period whose missingness
pattern differs from training. At 2016, 90 % coverage falls from {{re.o2016.BEST.cov90:pct}} to
{{re.o2016.BEST_plus_rcs_2026.cov90:pct}}. With the first learner set it falls from
{{re1.o2016.mlp.cov90:pct}} to {{re1.o2016.mlp_plus_rcs_2026.cov90:pct}}.

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
| Family evidence (C1) | {{re.o2016.abl.no_family.diff:+.4f}} | {{re.o2016.abl.no_family.lo:+.4f}} to {{re.o2016.abl.no_family.hi:+.4f}} | {{re.o2011.abl.no_family.diff:+.4f}} | {{re.o2011.abl.no_family.lo:+.4f}} to {{re.o2011.abl.no_family.hi:+.4f}} |
| Physics summaries | {{re.o2016.abl.no_physics.diff:+.4f}} | {{re.o2016.abl.no_physics.lo:+.4f}} to {{re.o2016.abl.no_physics.hi:+.4f}} | {{re.o2011.abl.no_physics.diff:+.4f}} | {{re.o2011.abl.no_physics.lo:+.4f}} to {{re.o2011.abl.no_physics.hi:+.4f}} |
| Design properties | {{re.o2016.abl.no_design.diff:+.4f}} | {{re.o2016.abl.no_design.lo:+.4f}} to {{re.o2016.abl.no_design.hi:+.4f}} | {{re.o2011.abl.no_design.diff:+.4f}} | {{re.o2011.abl.no_design.lo:+.4f}} to {{re.o2011.abl.no_design.hi:+.4f}} |
| Space weather | {{re.o2016.abl.no_space_weather.diff:+.4f}} | {{re.o2016.abl.no_space_weather.lo:+.4f}} to {{re.o2016.abl.no_space_weather.hi:+.4f}} | {{re.o2011.abl.no_space_weather.diff:+.4f}} | {{re.o2011.abl.no_space_weather.lo:+.4f}} to {{re.o2011.abl.no_space_weather.hi:+.4f}} |

**Family evidence (C1).** Siblings from the same launch and breakup that were catalogued, and
had decayed, before the prediction time improve TabM at 2016. At 2011 the point estimate is
positive but the interval includes zero.

**Physics summaries.** They help Kumo-Tabular at 2011, by a margin that would not survive a
multiplicity correction, and hurt TabM at 2016. A plausible reason, not tested here: in 2016–2021 the
physics expert's own error grew with solar cycle 25, and a learner that leans on its summaries
inherits that error.

**Design properties.** They help at 2016.

## Collision risk

**Setup.** The official Kelvins test set has {{cac.data.test_events:,}} events, of which
{{cac.data.test_high_risk}} are high-risk. Kumo-Tabular
had the lowest out-of-fold pinball loss among the recent learners and became the expert. The
gate then put almost no weight on the persistence expert (mean
{{cac.gate_weight_cosmos_gated:.4f}}), so the gated model is in effect Kumo-Tabular.

**Probabilistic forecast.** Kumo-Tabular's final-risk distribution has the lowest pinball loss.
Paired bootstrap over test events (post hoc; a negative difference favours Kumo-Tabular):
against TabICLv2 {{cacp.kumo.tabicl.diff:+.3f}} ({{cacp.kumo.tabicl.lo:+.3f}} to {{cacp.kumo.tabicl.hi:+.3f}}),
against XGBoost {{cacp.kumo.xgboost.diff:+.3f}} ({{cacp.kumo.xgboost.lo:+.3f}} to {{cacp.kumo.xgboost.hi:+.3f}})
and against persistence {{cacp.kumo.persistence.diff:+.3f}} ({{cacp.kumo.persistence.lo:+.3f}} to {{cacp.kumo.persistence.hi:+.3f}}).

| Model | Pinball loss | AUC-PR of P(high risk) | AUC-ROC of P(high risk) |
|---|---|---|---|
| Kumo-Tabular | {{cac.kumo.pinball:.3f}} | {{cac.kumo.auc_pr_high:.3f}} | {{cac.kumo.auc_roc_high:.3f}} |
| TabICLv2 | {{cac.tabicl.pinball:.3f}} | {{cac.tabicl.auc_pr_high:.3f}} | {{cac.tabicl.auc_roc_high:.3f}} |
| Best tree baseline (XGBoost) | {{cac.xgboost.pinball:.3f}} | {{cac.xgboost.auc_pr_high:.3f}} | {{cac.xgboost.auc_roc_high:.3f}} |
| Persistence | {{cac.persistence.pinball:.3f}} | {{cac.persistence.auc_pr_high:.3f}} | {{cac.persistence.auc_roc_high:.3f}} |

Its conformal 90 % intervals cover {{cac.kumo.cov90:pct}} of all test events but only
{{cac.kumo.cov90_high:pct}} of the high-risk ones. With
{{cac.data.train_high_risk}} high-risk training events, the group-conditional calibration is
coarse exactly where it matters.

**Official point metric.** It does not reward these gains.
- Every model's median forecast falls below the high-risk threshold for most high-risk events, so
  its $F_2$ is low: {{cacq.kumo.n_high_below:.0f}} of the {{cac.data.test_high_risk}} high-risk
  events receive a median below −6 from Kumo-Tabular, giving $F_2$ = {{cac.kumo.F2:.2f}}. Most of
  those medians are not at the −30 floor but just below the threshold
  ({{cacq.kumo.n_high_in_m10_m6:.0f}} lie between −10 and −6), so the floor is not the main cause.
- The latest-risk baseline scores $L$ = {{cac.latest_risk_point.L_clip:.3f}}, the published
  value.
- Gating the latest-risk point with each model's calibrated P(high risk) brings the best models
  close to the baseline:
  - TabM: {{cacg.tabm_gated_persistence.L:.3f}} (difference from the baseline, 95 % CI
    {{cacg.tabm_gated_persistence.L_minus_latest_ci95.lo:+.3f}} to {{cacg.tabm_gated_persistence.L_minus_latest_ci95.hi:+.3f}});
  - the gated model: {{cacg.cosmos_gated_gated_persistence.L:.3f}}
    ({{cacg.cosmos_gated_gated_persistence.L_minus_latest_ci95.lo:+.3f}} to {{cacg.cosmos_gated_gated_persistence.L_minus_latest_ci95.hi:+.3f}}).
- No model beats the baseline.

This agrees with the challenge, where few teams beat the latest risk, and with a pre-registered
re-evaluation [@uriot2022spacecraft; @vazquez2026latest].

| Model | L (clipped) | F2 | MSE_HR | Pinball | AUC-PR P(high) | Brier P(high) | Cov. 90 % | Cov. 90 %, high risk |
|---|---|---|---|---|---|---|---|---|
| Latest risk (challenge baseline) | {{cac.latest_risk_point.L_clip:.3f}} | {{cac.latest_risk_point.F2_clip:.3f}} | {{cac.latest_risk_point.MSE_HR_clip:.3f}} | — | — | — | — | — |
| Persistence + residual climatology | {{cac.persistence.L_clip:.3f}} | {{cac.persistence.F2_clip:.3f}} | {{cac.persistence.MSE_HR_clip:.3f}} | {{cac.persistence.pinball:.3f}} | {{cac.persistence.auc_pr_high:.3f}} | {{cac.persistence.brier_high:.4f}} | {{cac.persistence.cov90:.3f}} | {{cac.persistence.cov90_high:.3f}} |
| XGBoost (baseline) | {{cac.xgboost.L_clip:.3f}} | {{cac.xgboost.F2_clip:.3f}} | {{cac.xgboost.MSE_HR_clip:.3f}} | {{cac.xgboost.pinball:.3f}} | {{cac.xgboost.auc_pr_high:.3f}} | {{cac.xgboost.brier_high:.4f}} | {{cac.xgboost.cov90:.3f}} | {{cac.xgboost.cov90_high:.3f}} |
| LightGBM (baseline) | {{cac.lightgbm.L_clip:.3f}} | {{cac.lightgbm.F2_clip:.3f}} | {{cac.lightgbm.MSE_HR_clip:.3f}} | {{cac.lightgbm.pinball:.3f}} | {{cac.lightgbm.auc_pr_high:.3f}} | {{cac.lightgbm.brier_high:.4f}} | {{cac.lightgbm.cov90:.3f}} | {{cac.lightgbm.cov90_high:.3f}} |
| CatBoost (baseline) | {{cac.catboost.L_clip:.3f}} | {{cac.catboost.F2_clip:.3f}} | {{cac.catboost.MSE_HR_clip:.3f}} | {{cac.catboost.pinball:.3f}} | {{cac.catboost.auc_pr_high:.3f}} | {{cac.catboost.brier_high:.4f}} | {{cac.catboost.cov90:.3f}} | {{cac.catboost.cov90_high:.3f}} |
| Quantile MLP (baseline) | {{cac.mlp.L_clip:.3f}} | {{cac.mlp.F2_clip:.3f}} | {{cac.mlp.MSE_HR_clip:.3f}} | {{cac.mlp.pinball:.3f}} | {{cac.mlp.auc_pr_high:.3f}} | {{cac.mlp.brier_high:.4f}} | {{cac.mlp.cov90:.3f}} | {{cac.mlp.cov90_high:.3f}} |
| TabM (2025) | {{cac.tabm.L_clip:.3f}} | {{cac.tabm.F2_clip:.3f}} | {{cac.tabm.MSE_HR_clip:.3f}} | {{cac.tabm.pinball:.3f}} | {{cac.tabm.auc_pr_high:.3f}} | {{cac.tabm.brier_high:.4f}} | {{cac.tabm.cov90:.3f}} | {{cac.tabm.cov90_high:.3f}} |
| TabICLv2 (2026) | {{cac.tabicl.L_clip:.3f}} | {{cac.tabicl.F2_clip:.3f}} | {{cac.tabicl.MSE_HR_clip:.3f}} | {{cac.tabicl.pinball:.3f}} | {{cac.tabicl.auc_pr_high:.3f}} | {{cac.tabicl.brier_high:.4f}} | {{cac.tabicl.cov90:.3f}} | {{cac.tabicl.cov90_high:.3f}} |
| Kumo-Tabular (2026) | {{cac.kumo.L_clip:.3f}} | {{cac.kumo.F2_clip:.3f}} | {{cac.kumo.MSE_HR_clip:.3f}} | {{cac.kumo.pinball:.3f}} | {{cac.kumo.auc_pr_high:.3f}} | {{cac.kumo.brier_high:.4f}} | {{cac.kumo.cov90:.3f}} | {{cac.kumo.cov90_high:.3f}} |
| Persistence + expert, constant weight | {{cac.stack_constant.L_clip:.3f}} | {{cac.stack_constant.F2_clip:.3f}} | {{cac.stack_constant.MSE_HR_clip:.3f}} | {{cac.stack_constant.pinball:.3f}} | {{cac.stack_constant.auc_pr_high:.3f}} | {{cac.stack_constant.brier_high:.4f}} | {{cac.stack_constant.cov90:.3f}} | {{cac.stack_constant.cov90_high:.3f}} |
| **Reliability-gated (COSMOS)** | {{cac.cosmos_gated.L_clip:.3f}} | {{cac.cosmos_gated.F2_clip:.3f}} | {{cac.cosmos_gated.MSE_HR_clip:.3f}} | {{cac.cosmos_gated.pinball:.3f}} | {{cac.cosmos_gated.auc_pr_high:.3f}} | {{cac.cosmos_gated.brier_high:.4f}} | {{cac.cosmos_gated.cov90:.3f}} | {{cac.cosmos_gated.cov90_high:.3f}} |

## Couplings and twin

### C2: from per-fragment re-entry to breakup-cloud survival

**Setup.** C2 covers the test folds of both origins: {{c2.calibrated.cosmos_gated.n_cloud_years}}
cloud-years in {{c2.calibrated.cosmos_gated.n_clouds}} breakup clouds, each with at least 30
catalogued fragments. For each cloud we predict the number of its fragments still in orbit one
to five years after its first catalogued fragment, by summing per-fragment survival
probabilities, and compare it with the catalogue count.

**Results** (mean absolute error as a fraction of the cloud, with 95 % intervals from a
bootstrap over clouds; Supplementary Table S17):

| Model | Raw reading | 95 % CI | Calibrated reading | 95 % CI |
|---|---|---|---|---|
| Physics expert alone | {{c2b.pred_physics_only.v:.3f}} | {{c2b.pred_physics_only.lo:.3f}}–{{c2b.pred_physics_only.hi:.3f}} | {{c2b.predcal_physics_only.v:.3f}} | {{c2b.predcal_physics_only.lo:.3f}}–{{c2b.predcal_physics_only.hi:.3f}} |
| TabICLv2 | {{c2b.pred_tabicl.v:.3f}} | {{c2b.pred_tabicl.lo:.3f}}–{{c2b.pred_tabicl.hi:.3f}} | {{c2b.predcal_tabicl.v:.3f}} | {{c2b.predcal_tabicl.lo:.3f}}–{{c2b.predcal_tabicl.hi:.3f}} |
| Gated model | {{c2b.pred_cosmos_gated.v:.3f}} | {{c2b.pred_cosmos_gated.lo:.3f}}–{{c2b.pred_cosmos_gated.hi:.3f}} | {{c2b.predcal_cosmos_gated.v:.3f}} | {{c2b.predcal_cosmos_gated.lo:.3f}}–{{c2b.predcal_cosmos_gated.hi:.3f}} |

- **Raw reading.** The physics expert has the lowest point error. Its mean bias is
  {{c2.raw.physics_only.bias_fraction:+.3f}}, and {{c2.raw.physics_only.within_10pct:pct}} of
  cloud-years are within 10 %. The learned experts under-predict survival on this reading, but
  the physics expert cannot be separated from the gated model (difference
  {{c2b.pred_physics_only_minus_pred_cosmos_gated.v:+.3f}}, 95 % CI
  {{c2b.pred_physics_only_minus_pred_cosmos_gated.lo:+.3f}} to
  {{c2b.pred_physics_only_minus_pred_cosmos_gated.hi:+.3f}}).
- **Calibrated reading** (isotonic maps of $P(T \le a)$ fitted on tune-B fragments). Calibration
  significantly lowers the error of TabICLv2
  ({{c2b.pred_tabicl_minus_predcal_tabicl.v:+.3f}},
  {{c2b.pred_tabicl_minus_predcal_tabicl.lo:+.3f}} to {{c2b.pred_tabicl_minus_predcal_tabicl.hi:+.3f}})
  and of the gated model ({{c2b.pred_cosmos_gated_minus_predcal_cosmos_gated.v:+.3f}},
  {{c2b.pred_cosmos_gated_minus_predcal_cosmos_gated.lo:+.3f}} to
  {{c2b.pred_cosmos_gated_minus_predcal_cosmos_gated.hi:+.3f}}). Calibrated TabICLv2 has the lowest
  point error, with {{c2.calibrated.tabicl.within_10pct:pct}} of cloud-years within 10 %, but it
  cannot be separated from the raw physics expert
  ({{c2b.pred_physics_only_minus_predcal_tabicl.v:+.3f}},
  {{c2b.pred_physics_only_minus_predcal_tabicl.lo:+.3f}} to
  {{c2b.pred_physics_only_minus_predcal_tabicl.hi:+.3f}}). Calibration does not help the physics
  expert.

The calibrated reading and the cloud bootstrap were added after the raw reading had been seen
and criticised in an internal review; they are post hoc.

**What C2 shows.** With {{c2.calibrated.cosmos_gated.n_clouds}} clouds, the errors of the best
readings are about a tenth of a cloud's size, and the uncertainty is too large to rank the
experts. How survival is read off the predictive distribution matters for the learned experts.
In the twin, cloud forecasts are sums of the history-conditioned physics probabilities. T1 tests
the calibration of those probabilities on objects in orbit.

### C3: operational status as health evidence for the re-entry regime

**Cohort.** On 2026-08-16, {{c3.n:,}} Earth-orbiting payloads had perigee below 600 km. In the
following {{c3.horizon_days:.0f}} days, {{c3.events}} of them decayed; the decays came from
{{c3.launches_with_events}} of their {{c3.launches:,}} launches.

**Physics by status.** The drag physics over-predicts re-entry for payloads reported
operational:

| Group | n | Decays | Mean predicted P | Observed rate |
|---|---|---|---|---|
| Operational | {{c3.status.operational.n:,}} | {{c3.status.operational.events}} | {{c3.status.operational.mean_p_physics:.3f}} | {{c3.status.operational.event_rate:.3f}} |
| Non-operational | {{c3.status.non-operational.n}} | {{c3.status.non-operational.events}} | {{c3.status.non-operational.mean_p_physics:.3f}} | {{c3.status.non-operational.event_rate:.3f}} |

- **Operational payloads.** The physics probabilities nevertheless rank them well (AUC-ROC
  {{c3.status.operational.auc_roc_physics:.3f}}), consistent with operators letting some low
  payloads decay while keeping others in orbit.
- **Non-operational payloads.** The physics expert under-predicts somewhat, with AUC-ROC
  {{c3.status.non-operational.auc_roc_physics:.3f}}.

**As-of-date policies** (information at 2026-08-16 only):

| Policy | Brier score | ECE |
|---|---|---|
| Drag physics for every payload | {{c3.prospective.physics_status_blind.brier:.4f}} | {{c3.prospective.physics_status_blind.ece:.4f}} |
| Previous-window base rate ({{c3.payload_base_rate_previous_window:.4f}}) for every payload | {{c3.prospective.base_rate_status_blind.brier:.4f}} | {{c3.prospective.base_rate_status_blind.ece:.4f}} |
| Status-aware twin (base rate for operational payloads, physics otherwise) | {{c3.prospective.twin_status_aware_baserate.brier:.4f}} | {{c3.prospective.twin_status_aware_baserate.ece:.4f}} |

The status-aware twin has the lowest Brier score, but neither difference is significant once
launches are resampled:

| Status-aware twin minus | Brier difference | 95 % CI |
|---|---|---|
| Physics for every payload | {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.mean:.4f}} | {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.ci95.0:.4f}} to {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_physics_status_blind.ci95.1:.4f}} |
| Base rate for every payload | {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_base_rate_status_blind.mean:.4f}} | {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_base_rate_status_blind.ci95.0:.4f}} to {{c3.brier_difference_ci95_launch_bootstrap.twin_status_aware_baserate_minus_base_rate_status_blind.ci95.1:.4f}} |

With {{c3.events}} decays in one window, the test can show the direction of the effect but
cannot establish it. The window is also not fresh: an earlier version of the project had fitted
and tested re-entry classifiers on it.

**In-window diagnostic (not a forecast).** We also fitted Platt recalibrations on the follow-up
window itself, cross-fitted by launch. Recalibrating per status rather than for all payloads
changes the Brier score by
{{c3.brier_difference_ci95_launch_bootstrap.in_window_recal_status_aware_minus_recal_blind.mean:.4f}}
(95 % CI
{{c3.brier_difference_ci95_launch_bootstrap.in_window_recal_status_aware_minus_recal_blind.ci95.0:.4f}}
to
{{c3.brier_difference_ci95_launch_bootstrap.in_window_recal_status_aware_minus_recal_blind.ci95.1:.4f}}).
Even with look-ahead, then, status adds little once the physics probabilities are rescaled. The
regime split is what keeps the twin from issuing drag forecasts for payloads whose operators
control their orbits.

### T1: the twin's re-entry forecasts for objects in orbit

**Cohort.** As of 2026-08-16, the twin held {{t1.uncontrolled:,}} fragments, rocket stages and
components with perigee below 2000 km. Within 72 days, {{t1.by_horizon.72d.uncontrolled.physics_history.events}}
of them decayed, {{t1.by_horizon.30d.uncontrolled.physics_history.events}} within 30 days.

**Results at 72 days** (expected re-entries, and their ratio to the observed number):

| Variant | Expected | Exp./obs. | 95 % CI | Brier score |
|---|---|---|---|---|
| History-conditioned physics | {{t1.by_horizon.72d.uncontrolled.physics_history.expected:.0f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ratio:.2f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ci95.0:.2f}}–{{t1.by_horizon.72d.expected_over_observed_ci95.physics_history.ci95.1:.2f}} | {{t1.by_horizon.72d.uncontrolled.physics_history.brier:.5f}} |
| Physics with class priors | {{t1.by_horizon.72d.uncontrolled.physics_prior.expected:.0f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ratio:.2f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ci95.0:.2f}}–{{t1.by_horizon.72d.expected_over_observed_ci95.physics_prior.ci95.1:.2f}} | {{t1.by_horizon.72d.uncontrolled.physics_prior.brier:.5f}} |
| Lifetime pipeline (gated TabM) | {{t1.by_horizon.72d.uncontrolled.lifetime.expected:.0f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ratio:.1f}} | {{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ci95.0:.1f}}–{{t1.by_horizon.72d.expected_over_observed_ci95.lifetime.ci95.1:.1f}} | {{t1.by_horizon.72d.uncontrolled.lifetime.brier:.5f}} |

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
| Physics with class priors | {{t1.by_horizon.72d.brier_difference_ci95_launch_bootstrap.physics_history_minus_physics_prior.mean:+.5f}} | {{t1.by_horizon.72d.brier_difference_ci95_launch_bootstrap.physics_history_minus_physics_prior.ci95.0:+.5f}} to {{t1.by_horizon.72d.brier_difference_ci95_launch_bootstrap.physics_history_minus_physics_prior.ci95.1:+.5f}} |
| Lifetime pipeline | {{t1.by_horizon.72d.brier_difference_ci95_launch_bootstrap.physics_history_minus_lifetime.mean:+.5f}} | {{t1.by_horizon.72d.brier_difference_ci95_launch_bootstrap.physics_history_minus_lifetime.ci95.0:+.5f}} to {{t1.by_horizon.72d.brier_difference_ci95_launch_bootstrap.physics_history_minus_lifetime.ci95.1:+.5f}} |

**Ranking.** Rank metrics use the raw probabilities. By AUC-PR the history-conditioned variant
ranks best at 72 days: {{t1.by_horizon.72d.uncontrolled.physics_history.auc_pr:.3f}}, against
{{t1.by_horizon.72d.uncontrolled.physics_prior.auc_pr:.3f}} (class priors) and
{{t1.by_horizon.72d.uncontrolled.lifetime.auc_pr:.3f}} (lifetime pipeline). By AUC-ROC the three
are within 0.01 of each other ({{t1.by_horizon.72d.uncontrolled.physics_history.auc_roc:.3f}},
{{t1.by_horizon.72d.uncontrolled.physics_prior.auc_roc:.3f}} and
{{t1.by_horizon.72d.uncontrolled.lifetime.auc_roc:.3f}}), so all three separate decaying objects
from the rest about equally well. What differs is mainly calibration. At 30 days the AUC-PR
order is the same, and by AUC-ROC the history-conditioned variant is slightly lower
({{t1.by_horizon.30d.uncontrolled.physics_history.auc_roc:.3f}}, against
{{t1.by_horizon.30d.uncontrolled.physics_prior.auc_roc:.3f}} and
{{t1.by_horizon.30d.uncontrolled.lifetime.auc_roc:.3f}}).

**Payloads.**
- *Operational payloads* (operator-controlled regime). The C3 policy expected
  {{t1.by_horizon.72d.payload_operator_controlled.twin.expected:.0f}} decays against
  {{t1.by_horizon.72d.payload_operator_controlled.twin.events}} observed. Drag physics would
  have expected {{t1.by_horizon.72d.payload_operator_controlled.physics_prior.expected:.0f}}. The
  base-rate policy cannot rank payloads, though: its AUC-ROC is
  {{t1.by_horizon.72d.payload_operator_controlled.twin.auc_roc:.2f}}, against
  {{t1.by_horizon.72d.payload_operator_controlled.physics_prior.auc_roc:.2f}} for drag physics.
- *Non-operational payloads.* The physics expert expected
  {{t1.by_horizon.72d.payload_natural_decay.twin.expected:.0f}} against
  {{t1.by_horizon.72d.payload_natural_decay.twin.events}} observed, the same under-prediction as
  in C3.

**The whole twin.** Over all {{t1.objects:,}} objects, the twin's per-regime choice expected
{{t1.by_horizon.72d.all_twin.expected:.0f}} re-entries against
{{t1.by_horizon.72d.all_twin.events}} observed (Brier {{t1.by_horizon.72d.all_twin.brier:.4f}}).

**Caveats.**
- T1 was specified before it was run, but it is not a blind test. An earlier version of the
  project had fitted and tested re-entry classifiers on the same window, and the window's
  aggregate decay count was seen while the lifetime pipeline's bias on current orbits was being
  diagnosed, before the history-conditioned variant was designed.
- A defect in the payload prior was corrected after T1 had first been scored. Payload numbers changed; those of fragments, rocket stages and components
  did not.

## The dated twin forecast of 2026-10-27

**Scope.** As of 2026-10-27 the twin holds {{twin.objects:,}} objects in orbit with perigee below
2000 km:
- {{twin.regime_counts.natural_decay:,}} in the natural-decay regime;
- {{twin.regime_counts.operator_controlled:,}} operator-controlled payloads;
- {{twin.regime_counts.unknown:,}} payloads without a reported status.

**History conditioning** (the figure shows three examples). Among fragments, rocket stages and components:
- for {{twin.history_kinds.history:,}}, the catalogue history determined the area-to-mass
  posterior (effective sample size of at least 4);
- for {{twin.history_kinds.survival_only}}, survival alone was used;
- for {{twin.history_kinds.longest_survivors}}, no prior sample survived, and the longest
  survivors were used;
- {{twin.history_kinds.prior}} objects without a usable canonical orbit kept the class prior.

**Expected re-entries in the next 30 days** (natural-decay regime): {{twin.expected_reentries_natural_decay.30d:.1f}}.
For fragments, rocket stages and components, the three variants differ by an order of
magnitude:

| Variant | Expected re-entries, 30 days |
|---|---|
| History-conditioned physics | {{twin.expected_reentries_uncontrolled_by_variant.physics_history.30d:.1f}} |
| Physics with class priors | {{twin.expected_reentries_uncontrolled_by_variant.physics_prior.30d:.1f}} |
| Gated lifetime pipeline, fitted at canonical-orbit epochs | {{twin.expected_reentries_uncontrolled_by_variant.lifetime.30d:.1f}} |

T1 indicates which of these to believe. the figure maps the forecast and
lists the natural-decay objects with the highest 30-day probability.

**Clouds and verification.** The forecast includes breakup-cloud forecasts for the
{{twin.c2.n_clouds}} clouds with at least 30 fragments in orbit. Its record file is fixed by its
SHA-256 hash, `{{twin.sha256}}`, so that later catalogue snapshots can score it without the
possibility of revision. An earlier forecast of the same date (hash beginning `fad1c619`)
contained the payload-prior defect described in the deviations paragraph above. It was replaced
before any later snapshot existed, and it is archived with the code.

## Health

### Health module: NASA SMAP/MSL telemetry and the companion ESA-ADB study

**Setup.** On the {{smap.config.channels}} channels ({{smap.SMAP.ar_ridge.events:.0f}} SMAP and
{{smap.MSL.ar_ridge.events:.0f}} MSL anomaly sequences), detectors are compared at a level and
persistence fixed in advance.

**Ranking** (the figurea). The published telemanom scores [@hundman2018] rank anomalous steps best. Their
per-channel mean AUC-PR is {{smap.SMAP.telemanom.pointwise_auc_pr:.3f}} (SMAP) and
{{smap.MSL.telemanom.pointwise_auc_pr:.3f}} (MSL). Used zero-shot, the Chronos-2 foundation
model reaches {{smap.SMAP.chronos2.pointwise_auc_pr:.3f}} and
{{smap.MSL.chronos2.pointwise_auc_pr:.3f}}. That is level with the ridge, PCA and
nearest-neighbour baselines, and above the random-score floor of
{{smap.SMAP.random.pointwise_auc_pr:.3f}} and {{smap.MSL.random.pointwise_auc_pr:.3f}}.

**Alarms** (the figureb). On the ESA-ADB corrected event-wise $F_{0.5}$, which charges alarm time on nominal
data, the highest detector scores are PCA on SMAP ({{smap.SMAP.pca.esa_ew_f0.5:.3f}}) and the
nearest-neighbour detector on MSL ({{smap.MSL.knn.esa_ew_f0.5:.3f}}). Chronos-2's residual
alarms are too frequent on MSL ({{smap.MSL.chronos2.event_fp:.0f}} false-positive segments),
giving a corrected $F_{0.5}$ of {{smap.MSL.chronos2.esa_ew_f0.5:.3f}} there, against
{{smap.SMAP.chronos2.esa_ew_f0.5:.3f}} on SMAP. That contrast is not a property of the spacecraft:
one MSL channel contributes 414 of those segments, and three contribute 95 % of them. They are
channels whose training series is constant, so every calibration score sits at the score floor and
any non-zero test residual raises an alarm.

**Why these alarm scores are weak evidence.**
- An always-on alarm reaches an uncorrected event $F_1$ of
  {{smap.SMAP.always_on.event_f1:.3f}} (SMAP) and {{smap.MSL.always_on.event_f1:.3f}} (MSL),
  higher than every detector. Its corrected $F_{0.5}$ is 0.
- {{smap.config.labelled_steps_in_second_half:pct}} of labelled steps lie in the second half of
  their channel's test series. A label-free reference that keeps each detector's amount of alarm
  time but places it as one block at the end of the series scores higher on the corrected
  $F_{0.5}$ than the detector itself in seven of the eight detector–spacecraft pairs; for
  example, {{smap.SMAP.pca.esa_ew_f0.5_late:.3f}} against {{smap.SMAP.pca.esa_ew_f0.5:.3f}} for
  PCA on SMAP. The exception is the nearest-neighbour detector on MSL
  ({{smap.MSL.knn.esa_ew_f0.5:.3f}} against {{smap.MSL.knn.esa_ew_f0.5_late:.3f}}).

We therefore read skill against a null that keeps each detector's alarm positions but
reassigns them across channels. Every fitted detector exceeds it on both spacecraft. For
example, ridge's event $F_1$ is {{smap.SMAP.ar_ridge.event_f1:.3f}} against
{{smap.SMAP.ar_ridge.event_f1_swap:.3f}} under the null (permutation $p$ =
{{smap.SMAP.ar_ridge.event_f1_swap_p:.3f}}). This shows channel-specific detection, but a
modest one: on this benchmark, alarms that are merely late are rewarded about as much.
the figure shows the scores and alarms on one typical channel of each spacecraft.

**Companion ESA-ADB study.** The health module's alarm layer (conformal p-values,
Benjamini–Hochberg aggregation across channels and persistence) was developed and tested in a separate,
pre-registered study on ESA-ADB [@kotowski2026esaadb] by the same authors. On Mission 1, fitting
an AR(8) forecaster and selecting the alarm settings on the whole training half, rather than on a
short calibration window, raised the corrected event-wise $F_{0.5}$ from
{{esa.pre.m1.A0.conformal_bh.f05.mean:.3f}} to {{esa.pre.m1.A1.conformal_bh.f05.mean:.3f}}
(difference {{esa.conf.m1.h6.diff_official:+.3f}}, 95 % CI
{{esa.conf.m1.h6.ci95_low:+.3f}} to {{esa.conf.m1.h6.ci95_high:+.3f}}), at about
{{esa.pre.m1.A1.conformal_bh.fa_yr.mean:.0f}} false alarm per year. On Mission 2 the same
contrast reversed ({{esa.conf.m2.h6.diff_official:+.3f}},
{{esa.conf.m2.h6.ci95_low:+.3f}} to {{esa.conf.m2.h6.ci95_high:+.3f}}). Validation analyses of
that study attributed the Mission 1 gain to the labels used to select the alarm settings, not to
the fitted scores.

**Summary.** The recent forecasting foundation model does not improve on simple detectors on
SMAP/MSL, and the alarm protocol that the twin's health module uses helped on one ESA-ADB
mission and hurt on the other. The health module is therefore the least validated of the three.

### Computational cost

Times are measured on the origin-2016 run ({{re.o2016.known5.train:,}} training objects,
{{re.o2016.sizes.test:,}} test objects). Hardware:
- CPU learners, TabICLv2 and the physics expert: a two-socket Intel Xeon 6980P node (512 hardware
  threads).
- TabM and Kumo-Tabular: an Intel Arc Pro B60 GPU.

| Model | Fit and predict per seed (s) | One call over the test cohort (s) |
|---|---|---|
| Physics expert (32 Monte-Carlo samples) | {{re.o2016.physics_seconds_train:.0f}} | {{re.o2016.physics_seconds_test:.0f}} |
| XGBoost | {{re.o2016.fit.xgboost.seconds:.0f}} | {{re.o2016.fit.xgboost.test_call_s:.3f}} |
| TabM | {{re.o2016.fit.tabm.seconds:.0f}} | {{re.o2016.fit.tabm.test_call_s:.3f}} |
| Kumo-Tabular | {{re.o2016.fit.kumo.seconds:.0f}} | {{re.o2016.fit.kumo.test_call_s:.0f}} |
| TabICLv2 | {{re.o2016.fit.tabicl.seconds:.0f}} | {{re.o2016.fit.tabicl.test_call_s:.0f}} |

**What is timed.** The first column is the wall-clock of the fit-and-predict loop divided by the
five seeds: it contains four refits (the seed-0 fit happens during tuning and is not timed) and
the predictions of all five seeds on the tuning and test cohorts. It is therefore an upper bound
on fitting alone, and for the in-context models it is dominated by prediction. The second column
is one call predicting the test cohort.

**In-context foundation models.** Their prediction cost includes encoding the training context,
up to 16,000 rows for Kumo-Tabular and {{re.o2016.known5.train:,}} for TabICLv2 on CPU. Predicting
the same cohort takes them three to five orders of magnitude longer than a tree ensemble
(Kumo-Tabular {{re.o2016.fit.kumo.test_call_s:.0f}} s and TabICLv2
{{re.o2016.fit.tabicl.test_call_s:.0f}} s, against {{re.o2016.fit.xgboost.test_call_s:.3f}} s for
XGBoost and {{re.o2016.fit.catboost.test_call_s:.3f}} s for CatBoost). Because the context encoding
is a fixed cost per call rather than per object, these times cannot be rescaled to a different
number of objects, and we do not extrapolate them to the full catalogue; both models predicted the
whole test cohort in one call.

**The physics features are part of every learner's cost.** All learners take five summaries of the
physics expert's Monte-Carlo distribution as inputs, so a deployed pipeline also pays that run:
{{re.o2016.physics_seconds_test:.0f}} s for the {{re.o2016.sizes.test:,}} test objects, which
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

