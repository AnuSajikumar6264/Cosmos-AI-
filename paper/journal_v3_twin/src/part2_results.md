# Results {#sec:results}

## Re-entry prediction at canonical-orbit epochs {#sec:res-reentry}

Table [-@tbl:reentry] lists every re-entry model on both test periods and Fig. [-@fig:reentry] shows their pinball losses.

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

: **Re-entry prediction on the two test periods** (left: origin 2016, test 2016–2021; right: origin 2011, test 2011–2015). Pinball loss over seven quantiles and MAE of the median, both in log10 days; coverage and mean width (log10 days) of the Mondrian conformal 90 % interval; Harrell's C. Models marked "baseline" are not eligible as the gated expert. {#tbl:reentry}

![**Test pinball loss of every re-entry model** on the two test periods. Grey: physics expert; blue: reference baselines, fixed-weight fusions and the variant whose gate is fitted on tune-A; green: the 2025–2026 learned experts; orange: the reliability-gated model.](artwork/Figure_5.png){#fig:reentry width=100%}

### Test period 2016–2021 (origin 2016)

**Selection.** On tune-A, TabM had the lowest loss among the recent learners (Supplementary Fig. S1 plots tune-A against
test loss for every learner):

| Learner | Tune-A pinball |
|---|---|
| TabM | {{re.o2016.tuneA_pinball.tabm:.4f}} |
| TabICLv2 | {{re.o2016.tuneA_pinball.tabicl:.4f}} |
| Kumo-Tabular | {{re.o2016.tuneA_pinball.kumo:.4f}} |

On the test cohort the ranking is different, and TabM falls behind both foundation models
(Table [-@tbl:reentry]):

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
Fig. [-@fig:reentry-outputs] shows individual forecasts and, for every test object with a known
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

![**What the re-entry module outputs** (origin 2016, test 2016–2021). (a) Forty test objects drawn at random (fixed seed) among those with a known five-year outcome, sorted by outcome: median and 90 % Mondrian conformal interval of the physics expert and of the reliability-gated model, against the observed time to decay (diamonds) or survival beyond the five-year cap (triangles). (b–d) Predicted median against observed outcome for every such test object, for the physics expert, TabICLv2 and the gated model.](artwork/Figure_6.png){#fig:reentry-outputs width=100%}

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
results were seen, so we read them as exploratory. Fig. [-@fig:reentry-benchmark] collects the
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

**First learner set.** The deviation record (Section [-@sec:protocol], Supplementary Table S1)
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
when the orbit is tracked continuously. Fig. [-@fig:coverage] shows the same rules over all test
objects, including those still in orbit after five years, where the ±20 % window scores higher
because it is wide enough to contain the five-year cap.

![**Head-to-head re-entry comparisons** (post hoc): differences in test pinball loss with paired launch-family cluster-bootstrap 95 % intervals; a negative difference favours the first model. Filled markers: the nominal interval, uncorrected for multiplicity, excludes zero. "Its expert" is TabM at origin 2016 and Kumo-Tabular at origin 2011.](artwork/Figure_7.png){#fig:reentry-benchmark width=100%}

![**Interval coverage on the test periods**, over all test objects, including those still in orbit after five years: the operational ±20 % and ±30 % residual-lifetime windows around the physics median, the raw quantile intervals of the gated model, and its marginal and Mondrian conformal intervals. Supplementary Table S5 gives the same rules restricted to the objects that decayed within five years.](artwork/Figure_8.png){#fig:coverage width=100%}

### An outcome leak through catalogue covariates {#sec:res-leak}

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

### Feature groups and coupling C1 (family evidence) {#sec:res-c1}

Each feature group was removed from the selected learner and the learner refitted with the same
settings and seeds. Fig. [-@fig:ablation] gives the paired launch-family bootstrap of the change
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

![**Feature-group ablations of the selected learner** (paired cluster bootstrap over launch families). In the first four rows a group is removed, so a positive change means the group helps. The last row instead *adds* the leaking 2026 radar cross-section, so there a positive change means the covariate makes the model worse.](artwork/Figure_9.png){#fig:ablation width=100%}

## Debris module: collision risk {#sec:res-cac}

**Setup.** The official Kelvins test set has {{cac.data.test_events:,}} events, of which
{{cac.data.test_high_risk}} are high-risk (Table [-@tbl:cac], Fig. [-@fig:cac]). Kumo-Tabular
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
  ({{cacq.kumo.n_high_in_m10_m6:.0f}} lie between −10 and −6), so the floor is not the main cause
  (Fig. [-@fig:cac-outputs]).
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

: **ESA Kelvins Collision Avoidance Challenge, official test set.** $L = \text{MSE}_\text{HR}/F_2$ of the median (predictions clipped at -6.001 as in the challenge); pinball loss of the final-risk quantiles; AUC-PR and Brier score of P(final risk >= -6); coverage of the conformal 90 % interval over all events and over the high-risk events. {#tbl:cac}

![**ESA Kelvins collision-risk test set.** (a) Pinball loss of the final-risk distribution; (b) the official score of the latest-risk point forecast and of the same forecast gated by each model's calibrated P(high risk).](artwork/Figure_10.png){#fig:cac width=100%}

![**What the collision-risk module outputs** (official test set). (a) Final risk against the latest risk available at least two days before closest approach, the input of the persistence expert. (b) Final risk against Kumo-Tabular's median forecast. {{cacq.kumo.n_high_below:.0f}} of the {{cac.data.test_high_risk}} high-risk events (orange) receive a median below the −6 threshold, {{cacq.kumo.n_high_in_m10_m6:.0f}} of them between −10 and −6 and {{cacq.kumo.n_high_near_floor:.0f}} within one decade of the −30 floor, so the median forecast reaches $F_2$ = {{cac.kumo.F2:.2f}} on the official metric. Dashed lines mark the −6 threshold and the dotted line is equality. (c–e) Three high-risk test events drawn at random (fixed seed): the input messages, Kumo-Tabular's predictive quantiles (open circle: median) and the final risk, both drawn to the right of the input window.](artwork/Figure_11.png){#fig:cac-outputs width=100%}

## Couplings {#sec:res-couplings}

### C2: from per-fragment re-entry to breakup-cloud survival {#sec:res-c2}

**Setup.** C2 covers the test folds of both origins: {{c2.calibrated.cosmos_gated.n_cloud_years}}
cloud-years in {{c2.calibrated.cosmos_gated.n_clouds}} breakup clouds, each with at least 30
catalogued fragments. For each cloud we predict the number of its fragments still in orbit one
to five years after its first catalogued fragment, by summing per-fragment survival
probabilities, and compare it with the catalogue count (Fig. [-@fig:c2]).

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
the calibration of those probabilities on objects in orbit (Section [-@sec:res-t1]).

![**C2: predicted against observed fraction of each breakup cloud's fragments still in orbit**, one point per cloud-year, (a) raw and (b) calibrated survival read-off.](artwork/Figure_12.png){#fig:c2 width=100%}

### C3: operational status as health evidence for the re-entry regime {#sec:res-c3}

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

**As-of-date policies** (information at 2026-08-16 only; Fig. [-@fig:c3]):

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
and tested re-entry classifiers on it (Section [-@sec:protocol]).

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

![**C3: mean predicted probability of re-entry within 72 days** by operational status, against the observed rate (dashed). The in-window Platt map is fitted on the follow-up outcomes and is shown only as a reference.](artwork/Figure_13.png){#fig:c3 width=100%}

### T1: the twin's re-entry forecasts for objects in orbit {#sec:res-t1}

**Cohort.** As of 2026-08-16, the twin held {{t1.uncontrolled:,}} fragments, rocket stages and
components with perigee below 2000 km. Within 72 days, {{t1.by_horizon.72d.uncontrolled.physics_history.events}}
of them decayed, {{t1.by_horizon.30d.uncontrolled.physics_history.events}} within 30 days
(Fig. [-@fig:t1]).

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
- A defect in the payload prior was corrected after T1 had first been scored (Section
  [-@sec:protocol]). Payload numbers changed; those of fragments, rocket stages and components
  did not.

![**T1: as-of-date test of the twin's re-entry forecasts** for fragments, rocket stages and components in orbit on 2026-08-16. (a) Reliability at 72 days, in six fixed probability bins; marker area grows with the number of objects in the bin, bins with fewer than ten objects are dropped, and both axes are symmetric-log so that bins with no observed decay sit on the zero line. (b) Expected against observed re-entries.](artwork/Figure_14.png){#fig:t1 width=100%}

## Health module {#sec:res-health-wrap}

### NASA SMAP/MSL telemetry and the companion ESA-ADB study {#sec:res-health}

**Setup.** On the {{smap.config.channels}} channels ({{smap.SMAP.ar_ridge.events:.0f}} SMAP and
{{smap.MSL.ar_ridge.events:.0f}} MSL anomaly sequences), detectors are compared at a level and
persistence fixed in advance.

**Ranking** (Fig. [-@fig:telemetry-benchmark]a). The published telemanom scores [@hundman2018] rank anomalous steps best. Their
per-channel mean AUC-PR is {{smap.SMAP.telemanom.pointwise_auc_pr:.3f}} (SMAP) and
{{smap.MSL.telemanom.pointwise_auc_pr:.3f}} (MSL). Used zero-shot, the Chronos-2 foundation
model reaches {{smap.SMAP.chronos2.pointwise_auc_pr:.3f}} and
{{smap.MSL.chronos2.pointwise_auc_pr:.3f}}. That is level with the ridge, PCA and
nearest-neighbour baselines, and above the random-score floor of
{{smap.SMAP.random.pointwise_auc_pr:.3f}} and {{smap.MSL.random.pointwise_auc_pr:.3f}}.

**Alarms** (Fig. [-@fig:telemetry-benchmark]b). On the ESA-ADB corrected event-wise $F_{0.5}$, which charges alarm time on nominal
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
Fig. [-@fig:smap-outputs] shows the scores and alarms on one typical channel of each spacecraft.

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

![**Telemetry benchmark on SMAP/MSL.** (a) Per-channel mean AUC-PR of anomalous steps. (b) Corrected event-wise F0.5 of each detector's alarms (filled) and of a label-free reference that keeps the detector's amount of alarm time but places it as one block at the end of each series (open).](artwork/Figure_15.png){#fig:telemetry-benchmark width=100%}

![**What the health module outputs** on one channel per spacecraft, chosen by rule as the median channel by Chronos-2 per-channel AUC-PR, taking the lower of the two middle channels where that number is even (SMAP E-6, MSL C-2). Top: test series and labelled anomalies. Middle: conformal p-values of PCA reconstruction and Chronos-2 against the channel's own training scores, with the alarm level. Bottom: labelled sequences and each detector's causal alarms. The training series of C-2 is constant, so every Chronos-2 calibration score sits at the score floor and any non-zero test residual alarms; Chronos-2 is in alarm from step 210 on, the most extreme Chronos-2 alarm behaviour of the 27 MSL channels. The recomputed alarms reproduce the benchmark's event counts for both channels.](artwork/Figure_16.png){#fig:smap-outputs width=100%}

## The dated twin forecast of 2026-10-27 {#sec:res-twin}

**Scope.** As of 2026-10-27 the twin holds {{twin.objects:,}} objects in orbit with perigee below
2000 km:
- {{twin.regime_counts.natural_decay:,}} in the natural-decay regime;
- {{twin.regime_counts.operator_controlled:,}} operator-controlled payloads;
- {{twin.regime_counts.unknown:,}} payloads without a reported status.

**History conditioning** (Fig. [-@fig:twin-state] shows three examples). Among fragments, rocket stages and components:
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

T1 indicates which of these to believe (Section [-@sec:res-t1]). Fig. [-@fig:twin-forecast] maps the forecast and
lists the natural-decay objects with the highest 30-day probability.

**Clouds and verification.** The forecast includes breakup-cloud forecasts for the
{{twin.c2.n_clouds}} clouds with at least 30 fragments in orbit. Its record file is fixed by its
SHA-256 hash, `{{twin.sha256}}`, so that later catalogue snapshots can score it without the
possibility of revision. An earlier forecast of the same date (hash beginning `fad1c619`)
contained the payload-prior defect described in Section [-@sec:protocol]. It was replaced
before any later snapshot existed, and it is archived with the code.

![**Twin state estimation**, for three objects drawn at random (one per class; fixed seed) among natural-decay objects whose catalogue history determined the posterior (effective sample size at least 4) and whose history-conditioned median forecast is within five years. (a) Prior area-to-mass samples (grey: the object's design value where GCAT has one, otherwise a draw from its class prior) and the posterior after conditioning on the change from the canonical to the current orbit (orange). (b) The re-entry forecasts the same samples produce. The recomputed posteriors reproduce the stored forecasts.](artwork/Figure_17.png){#fig:twin-state width=100%}

![**The dated twin forecast of 2026-10-27.** (a) Current orbits of the objects in the twin, by regime; the regimes overlap, and the largest is drawn first. (b) The twin's probability of re-entry within 30 days against current perigee, natural-decay regime. (c) The 15 natural-decay objects with the highest probability, ranked by that probability and then by the median forecast: median and 90 % Monte-Carlo interval of the forecast re-entry time (history-conditioned physics for fragments, rocket stages and components; class-prior physics for non-operational payloads). Two rows have a degenerate interval at exactly 10 days: their perigee is already within 1 km of the 120 km re-entry threshold, where the integrator's maximum step of 10 days sets the resolution.](artwork/Figure_18.png){#fig:twin-forecast width=100%}

### Computational cost {#sec:res-cost}

Times are measured on the origin-2016 run ({{re.o2016.known5.train:,}} training objects,
{{re.o2016.sizes.test:,}} test objects; Fig. [-@fig:cost]). Hardware:
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

![**Computational cost**, origin 2016, on log scales. (a) Wall-clock of the fit-and-predict loop per seed: four refits (the seed-0 fit happens during tuning and is not timed) plus the predictions of all five seeds on the tuning and test cohorts, so it bounds fitting alone from above and for the in-context models is dominated by prediction; for the physics expert, its Monte-Carlo run over the training set. (b) One call predicting the test cohort, which for the in-context models includes encoding the training context, a fixed cost per call rather than per object.](artwork/Figure_19.png){#fig:cost width=100%}

