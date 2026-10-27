# Discussion {#sec:discussion}

## RQ1: what the integrated chain establishes

The verification suite shows that the physics components implement their equations ([@tbl:verify]). The checks differ in strength:

- **{{verify.n_external}} compare with an answer from outside the code.** For example, the US76 density matches the published tables to {{verify.1.measured}}, impact speed reaches the terminal-velocity limit, the registration recovers imposed shifts with the correct sign, and the conformal zones reach their nominal coverage.
- **{{verify.n_internal}} are self-consistency checks or hold by construction.** These include the convergence of the impact point to {{verify.11.measured}} as the tolerance tightens, and the mass balance of the break-up model.
- **The hydrostatic check is mis-specified** (Appendix C).

These results establish the components as implementations of their equations. They do not establish that the equations are adequate for real re-entries. The atmosphere is static, the break-up model is a simplified and partly non-standard variant, and no recorded re-entry with recovery coordinates was available to test predicted landing locations.

The chain does connect health to re-entry physics. The subsystem that fails selects the break-up type, altitude and dispersion ([@tbl:failure-modes]). Running the chain, rather than reading it, reveals five limitations of the coupling.

1. **It depends on detection.** When the monitor misses a fault, the subsystem with the largest residual degradation becomes the driver instead of the faulted one. Slow degradation is missed in all but {{orig.monitor.degradation.detected}} of {{orig.monitor.degradation.n}} subsystems, and every fault on Thermal Control is missed. The resulting footprint is indistinguishable from a nominal one.
2. **Ties are broken by dictionary order.** In the all-nominal scenario every degradation is equal. The rule then names {{orig.scen.all_nominal.driver}} as the driver and labels the case an {{orig.scen.all_nominal.breakup_type}}.
3. **The probabilities are not propagated.** $P_{\text{frag}}$ and $\sigma_{\text{state}}$ are computed but not used. A break-up is propagated for every scenario, and the risk is not weighted by its probability.
4. **The debris feature is not connected.** It stands outside the chain, so the claim that "the objects catalogued in orbit are the same objects that later come down" is not implemented.
5. **The rules are hand-set.** The coupling weights are expert choices. They have not been fitted to, or checked against, recorded fragmentation events. The `discos.py` module was written for that check, but the DISCOS data it needs require an account.

The answer to RQ1 is therefore partial. The chain exists and its parts implement their equations. The coupling, however, is a hand-set rule whose outputs are only partly used. Its scientific value would rest on validation against recorded fragmentations and re-entries.

## RQ2: what determines event-wise performance on ESA-ADB

The pre-registered study gives a clear answer for Mission 1 at full channel scale. Elsewhere the answer is conditional.

### The selection result

On Mission 1 the A1 protocol raised AR(8) from {{pre.m1.A0.conformal_bh.f05.mean:.3f}} to {{pre.m1.A1.conformal_bh.f05.mean:.3f}}, and H6 is supported. Three observations qualify what this shows.

1. **It is a protocol effect.** A0 and A1 differ in five respects ([@sec:results-prereg]): the forecaster's fitting data, the residual scale, the calibration pool, the selection data ({{data.esa.m1.events_target.cal.events}} against {{data.esa.m1.events_all.train.targets}} target events) and the grid. Two of them change the test scores themselves. The post-hoc decomposition ([@sec:results-posthoc]) apportions the gain: the selection data carry it ({{ph.p1.m1.target.conformal_bh.forward.D.delta_official:+.3f}} and {{ph.p1.m1.target.conformal_bh.reverse.D.delta_official:+.3f}} on the two paths), the pool adds {{ph.p1.m1.target.conformal_bh.forward.P.delta_official:+.3f}}, and neither score-changing difference is material.
2. **It does not depend on BH.** Under the per-channel threshold the same contrast is {{seleff.m1.full.threshold:+.3f}} ([@tbl:seleff]).
3. **It does not transfer to every setting.** On the lightweight Mission 1 subset the contrast is negative under both layers. On Mission 2 it is negative under BH and positive under the threshold. Of the {{seleff.n}} contrasts in [@tbl:seleff], {{seleff.n_positive}} favour training-half selection. The cross-validated value over-estimated the lightweight BH result by more than an order of magnitude ({{sel.m1.A1lw.conformal_bh.cv.mean:.3f}} against {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} on test). Its selected level lies at the edge of the grid.

The selection effect is therefore real and large for Mission 1 at full scale, where the calibration window was too small to select from. It is not a general rule. A post-hoc reading is that more labelled history helps when the recent window is uninformative, and can hurt when a setting tuned to older data does not transfer to the test period. The post-hoc analyses fit that reading without testing it: the lightweight and Mission 2 reversals come from the same factor (the selection data), and in forward-chaining folds inside the training halves the history protocol won only {{ph.p2.m1.conformal_bh.n_folds_H_better}} of {{ph.p2.m1.conformal_bh.n_folds}} Mission 1 folds ([@sec:results-posthoc]).

**What the selected layer does.** The selected layer is strict ($q$ = {{sel.m1_target_ar8.conformal_bh.q:g}}, $k$ = {{sel.m1_target_ar8.conformal_bh.k:g}}). It trades recall ({{pre.m1.A1.conformal_bh.r.mean:.3f}}) for precision ({{pre.m1.A1.conformal_bh.p.mean:.3f}}), which is what $F_{0.5}$ with a true-negative-rate correction rewards. The benchmark authors credit the relative success of their best Mission 1 baseline to its dynamic thresholding and pruning, a comparable emphasis on suppressing false detections [@kotowski2026esaadb].

For an operator, the strict layer means three things:

- about one false alarm per year;
- about one target event in six detected;
- about {{pre.m1.A1.conformal_bh.alarm_runs_per_year:.0f}} alarm runs per year, most of them repeat alarms inside long events.

How many runs reach the operator therefore depends as much on the merging rule as on the detector.

### The deep supervised scorer

The TCN event scorer had full access to the training labels, a receptive field of {{evm.receptive_field}} steps and cross-channel mixing. It scored {{pre.m1.A3.conformal_bh.f05.mean:.3f}}, below the forecaster ({{conf.m1.h4.diff_official:+.3f}}) and below the trees ({{conf.m1.h5.diff_official:+.3f}}). Neither difference reached the minimum effect, so H4 and H5 are refuted. Neither is significant in the other direction either: the H4 interval includes zero, and the Holm-adjusted $p$ for H5 is {{conf.m1.h5.p_holm:.2f}}. The supported conclusion is that the deep scorer showed no benefit. It is not that the deep scorer is shown to be worse.

Four explanations are compatible with the data. None was pre-registered, and the study cannot separate them.

- **Distribution shift.** The supervised scorers' nominal test exceedances were {{shift.m1.A3.ratio_min:.1f}}–{{shift.m1.A3.ratio_max:.0f}}× (TCN) and {{shift.m1.A2.ratio_min:.1f}}–{{shift.m1.A2.ratio_max:.1f}}× (trees) their cross-validated rate, against {{shift.m1.a1_ar8.ratio:.1f}}× for AR(8). This can explain why cross-validation over-estimates the supervised arms. It does not explain the differences between runs: the rank correlation of shift with test score is {{shift.m1.supervised.spearman_f05:.2f}} over the ten supervised runs. The trees shift as much as three of the five TCN seeds ({{shift.m1.A2.ratio_min:.1f}}–{{shift.m1.A2.ratio_max:.1f}}× against {{shift.m1.a3_s2.ratio:.1f}}–{{shift.m1.a3_s4.ratio:.1f}}×), yet score far higher. Two plausible sources of shift remain: drift of Mission 1 over its record ({{data.esa.m1.train.start:date}} to {{data.esa.m1.test.end:date}}), and the difference between the final model and the fold models that produced the calibration pool.
- **Capacity relative to labels.** The TCN has {{evm.m1.params:int}} parameters and learns from {{sel.m1.n_train_events}} annotated events, {{data.esa.m1.events_all.train.Anomaly}} of them anomalies. It detects fewer unseen-class rare events ({{evcls.m1.unseen.Rare_Event.A3:.3f}}) than the trees or the unsupervised A0 ({{evcls.m1.unseen.Rare_Event.A2:.3f}} each). That pattern is consistent with a model that has fitted the recurring classes. The two ablations, without residual inputs and with two classes, changed little, so the result is not tied to those design choices.
- **Seed instability.** The TCN seeds range from {{pre.m1.A3.conformal_bh.f05.min:.3f}} to {{pre.m1.A3.conformal_bh.f05.max:.3f}} (sd {{pre.m1.A3.conformal_bh.f05.std:.3f}}). With five seeds, the arm mean is uncertain by roughly ±0.1 or more at 95 %.
- **The layer.** Under the per-channel threshold the TCN scored {{pre.m1.A3.threshold.f05.mean:.3f}}. The layer choice changes the size of the gap to the trees but not its direction.

### Trees, AR(8) and what the primary metric rewards

The trees' mean of {{pre.m1.A2.conformal_bh.f05.mean:.3f}} is the best Mission 1 result of the study. Their seed range ({{pre.m1.A2.conformal_bh.f05.min:.3f}}–{{pre.m1.A2.conformal_bh.f05.max:.3f}}) makes the difference from AR(8) statistically indistinguishable from zero. The event-class analysis locates the advantage. Of the trees' roughly {{evcls.m1.gain_a2_a1.total:.0f}} additional detections per seed over A1:

- about {{evcls.m1.gain_a2_a1.seen.Rare_Event:.0f}} are seen-class rare events;
- {{evcls.m1.gain_a2_a1.unseen.Rare_Event:.0f}} are unseen-class rare events, which the unsupervised A0 also detects;
- {{evcls.m1.gain_a2_a1.seen.Anomaly:.1f}} are anomalies.

On anomalies alone the two arms are equal: anomaly-only $F_{0.5}$ is {{pre.m1.A2.conformal_bh.f05_anom.mean:.3f}} for the trees and {{pre.m1.A1.conformal_bh.f05_anom.mean:.3f}} for AR(8). The trees are therefore better at the benchmark's combined target, not better at finding anomalies. No arm detected any of the {{evcls.m1.unseen.Anomaly.n}} anomalies of unseen classes.

### Mission 2

On Mission 2, whole-half selection was worse than the calibration window under BH ({{conf.m2.h6.diff_official:+.3f}}) but better under the threshold ({{seleff.m2.full.threshold:+.3f}}). The reversal is therefore a property of the BH selection on this mission, not of the selection data as such.

The mission leaves little room for any effect. Its calibration window holds {{data.esa.m2.events_target.cal.events}} rare events and immediately precedes the test half. The parameter-free out-of-limit rule already scores {{explo.m2.full.ool.native.f05.mean:.3f}}. The four A0 and A1 values lie between {{pre.m2.A1.conformal_bh.f05.mean:.3f}} and {{pre.m2.A1.threshold.f05.mean:.3f}}. Its cross-validated landscape is a broad plateau ([@fig:landscape]), on which near-equivalent settings are separated by small differences that need not transfer. Mission 2 provides no evidence either way on the selection-data question.

## RQ3: event-level versus point-level calibration

On identical scores, point-level ACI held its point-wise false-alarm rate at the target. Its primary metric never exceeded {{pre.all.aci.f05.max:.4f}} ([@tbl:layers]), because it produced up to {{pre.all.aci.alarm_runs.max:int}} alarm runs on a test half. This agrees with the TC-CAD study, which found the same collapse with Telemanom scores on a smaller channel set [@sahin2026tccad]. In that study ACI also exceeded its target false-alarm rate, by {{lit.tccad.aci_far_overshoot}}× according to its deposit; here it did not. The present study adds three points.

- **The collapse occurs with other scores.** It occurs with forecaster and supervised scores, at full channel scale.
- **Event logic does not rescue ACI.** With persistence and merging, the best seed mean was {{pre.all.aci_event.f05.max_seedmean:.3f}}, and {{pre.all.aci_event.zero_alarm_runs}} of {{pre.all.aci_event.n_runs}} runs raised no alarm.
- **The reason is mechanical.** With the $\gamma$ selected in {{pre.all.aci.gamma005_runs}} of {{pre.all.aci.gamma_runs}} runs, one exceedance silences the channel for {{aci.recovery_steps.g0.05}} steps. A channel therefore cannot stay hot through an event, and persistence must be assembled from different channels firing in turn, which is rare ([@sec:results-ablations]).

A fair event-level ACI would select $\gamma$ on the event metric, or apply persistence before the ACI update. Neither was tested.

**What makes the alarms usable.** In this study, usable alarms come from aggregation to events (persistence and merging), together with a strict operating point selected on the event metric. Multiplicity control across channels is not shown to be necessary. The per-channel threshold, which has none, equalled or beat conformal-BH in {{layers.n_threshold_ge_bh}} of {{layers.n_rows}} rows, and it gave the same Mission 1 selection effect. The robust conclusion concerns the level of aggregation and the selection, not BH as such.

**No formal guarantee applies here.** BH is run across channels whose p-values come from separate per-channel pools and whose scores are correlated. The guarantee of Bates et al. covers test points that share one calibration pool [@bates2023], and exchangeability over time fails. In this setting, the conformal-BH layer is a scale-free family of thresholds indexed by $(q, k)$, not a procedure with false-discovery-rate control.

**The rolling layer is weak evidence.** The rolling layer recalibrates on trailing unlabelled test scores. It lowered AR(8) and the trees sharply. It raised only the lightweight Mission 1 arm, whose conformal-BH result had collapsed, and the TCN, whose seeds then ranged from {{pre.m1.A3.conformal_bh_rolling.f05.min:.3f}} to {{pre.m1.A3.conformal_bh_rolling.f05.max:.3f}}. Its $(q, k)$ were copied from the conformal-BH selection rather than selected for it. It is therefore weak evidence about drift in either direction.

## RQ4: deep versus classical models on the supporting real datasets

The three supporting real-data studies do not show a single pattern by model family.

- **OPSSAT-AD.**
  - On the official split, the deep models lead. The hybrid has the highest F1 and the 1-D CNN the highest AUC-PR. Even on the {{data.opssat.n_features}} tabular features, the feature MLP beats XGBoost (F1 {{ops.paired.fcnn_xgboost.f1.diff:+.3f}}, interval above zero).
  - Under the time-blocked split, which is the more realistic one because the official split interleaves training and test fragments in time, the deep models drop and two classical models rise. The ranking compresses.
  - Gradient-boosted trees are the most robust model to sample dropout.
- **SpotGEO.** The heat-map networks are an order of magnitude better than the classical COSMOS front end. Without temporal post-processing, they remain far from the challenge winner ({{spot.unet.both.f100.test_f1.mean:.3f}} against {{lit.spotgeo.winner_f1:.3f}}).
- **Re-entry landmark.**
  - Trees beat the neural hazard model ({{lm.cmp.hazard_mlp_vs_xgboost.h68.diff:+.3f}} at 68 days), consistent with the general finding for medium-sized tabular data [@grinsztajn2022tree].
  - The physics lifetime, far behind on the full test set, has the highest 30-day AUC-PR on uncontrolled objects, whose re-entry drag governs. That comparison rests on {{lm.physics.uncontrolled.h30.positives}} positive objects.

The common thread with RQ2 is that model capacity was rarely the binding constraint. The evaluation protocol changed the rankings more than the choice between model families did: the split on OPSSAT-AD, and the alarm layer and selection data on ESA-ADB.

## RQ5: the propagator surrogate

The surrogate reproduces the COSMOS propagator closely inside its training ranges:

- median position error {{sur.test.position_error_km_median:.1f}} km;
- median CEP50 error {{sur.fp.cep50_relerr_median:pct1}} over {{sur.fp.n}} footprint scenarios;
- about {{sur.fp.speedup_median:.0f}}× faster.

Its ensemble spread is only a moderate out-of-range signal (AUROC {{sur.ood_auroc:.3f}}). Inside the pipeline, any fallback to the reference propagator removes most of the speed-up, because a few reference samples dominate the run time. All of these statements concern agreement with the simulator. None concerns real re-entries.

## Relation to the literature

**Comparison with the published baselines.** The official baselines [@kotowski2026esaadb] set the reference on the full channel sets. Our Mission 1 numbers under the same protocol are higher, at {{pre.m1.A1.conformal_bh.f05.mean:.3f}} (AR(8)) and {{pre.m1.A2.conformal_bh.f05.mean:.3f}} (trees), against a published best of {{lit.dmlr.m1.full.best:.3f}}. Three facts limit what this comparison shows.

- **The use of labels differs.** The published baselines did not tune their thresholds on training labels. A1 selects its layer on those labels, and A2 and A3 are supervised. The comparison shows what label-based selection can achieve under the official protocol. It does not show that these scorers are better detectors.
- **Mission 2 is saturated.** The published best full-channel value is {{lit.dmlr.m2.full.best:.3f}}, but the parameter-free out-of-limit rule scores {{explo.m2.full.ool.native.f05.mean:.3f}}. The large margin over the published values therefore says more about those values than about our scorers.
- **The advantage does not extend to the lightweight subset.** On the lightweight Mission 1 subset the best published value is {{lit.dmlr.m1.lw.best:.3f}} ({{lit.dmlr.m1.lw.best_name}}). The training-half-selected AR(8) reached only {{pre.m1.A1lw.threshold.f05.mean:.3f}} (threshold) and {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} (BH), and the calibration-window arm {{explo.m1.lw.ar8.conformal_bh.f05.mean:.3f}}.

**Other protocols.** Our numbers are lower than those reported under other protocols:

- {{lit.alcarria.m1_f05:.3f}} with monthly retraining over the whole mission [@alcarria2026];
- {{lit.allegrini.private:.3f}} on the Kaggle split [@allegrini2026];
- {{lit.brancato.smed_full:.3f}} with SMED, a lightweight similarity-search detector [@brancato2027ast].

The protocols differ in retraining, splits and channel selection, so these numbers do not establish a ranking. At least one of these methods, monthly retraining, adapts to new data over time. Adaptation is a plausible direction for future work, although the drift diagnostic here does not by itself show that drift limits the present scorers.

**Calibration and triage.** The TC-CAD study's negative result on telecommand conditioning is replicated here with different detectors [@sahin2026tccad]. Its finding on point-level conformal calibration is extended to the full channel sets, with a mechanical explanation for why event logic does not repair it.

## Practical implications

For an operator choosing an anomaly monitor on telemetry like ESA-ADB's, the evidence supports five practical steps.

1. **Select at the event level.** Evaluate and select alarm settings at the event level, on as much labelled history as can be scored out of sample. Then check the selected setting on a later held-out period, because under BH the selection did not transfer on {{seleff.bh.n_negative}} of the {{seleff.bh.n}} mission and channel-set combinations.
2. **Report the alarm burden.** Report false alarms per year, alarm runs per year and alarming precision next to any $F$ score. With events that last days, the merging rule determines much of the alarm count.
3. **Monitor supervised scorers.** Treat the calibration of supervised scorers as perishable, and monitor their nominal exceedance rate after deployment.
4. **Do not rely on depth.** Do not expect a deeper model to compensate for poor alarm selection.
5. **Report anomaly-only metrics.** The primary metric counts rare nominal events as targets, so a score near 0.9 on Mission 2 can coexist with detecting only a few of its {{data.esa.m2.events_target.test.Anomaly}} anomalies. On Mission 1, the trees' lead disappears on anomalies alone. Anomaly-only metrics should always be reported alongside the primary score.
