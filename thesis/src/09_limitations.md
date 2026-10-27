# Limitations and Future Work {#sec:limitations}

## Threats to validity {#sec:threats}

**Internal validity of the pre-registered study.**

- **Prior exposure of the test halves.** Both missions' test halves had been scored in the exploratory study for other configurations. The new arms had never been scored. However, the decision to study alarm selection on Mission 1 was taken after Mission 1 had failed. This is a researcher degree of freedom that the pre-registration documents but cannot remove.
- **Cross-fitting optimism.** Out-of-fold models are trained on later as well as earlier blocks, so training-half scores are more favourable than deployed scores. The gap between cross-validation and test in [@tbl:prereg-arms] is consistent with this.
- **Selections on grid edges.** Several selections lie on an edge of the pre-registered grid, at $q = 10^{-5}$ or $10^{-2}$, or at $k = 100$. The grid was not widened after the freeze, so better settings may lie outside it.
- **Too few seeds for the supervised arms.** Five seeds are few for arms whose Mission 1 results span {{pre.m1.A2.conformal_bh.f05.min:.3f}}–{{pre.m1.A2.conformal_bh.f05.max:.3f}} (trees). The hierarchical bootstrap accounts for this, but the intervals are wide.
- **Procedural deviations.** {{prereg.n_deviations}} deviations are logged (Appendix B). Five are procedural and changed no computation. The sixth concerns the design: arm A0 used the exploratory selection grid, not the pre-registered one, so H6 compares two complete protocols.
- **Confounded H6 contrast.** A0 and A1 differ in fitting period, residual scale, calibration pool, selection data and grid; the first two change the test scores. The confirmatory result cannot be attributed to any one of them.
- **Small bootstrap block counts.** The test halves contain {{data.esa.m1.test.blocks30}} (Mission 1) and {{data.esa.m2.test.blocks30}} (Mission 2) 30-day blocks. With so few blocks, the Mission 2 intervals are themselves uncertain.

**Construct validity.**

- **Rare events count as targets.** The primary metric treats rare nominal events as targets. A detector can therefore score well on Mission 2 while missing most of its {{data.esa.m2.events_target.test.Anomaly}} anomalies. Anomaly-only metrics are reported for this reason.
- **Exchangeability is violated, and the pools are per channel.** Conformal levels are used as tuning parameters, not as guarantees.
- **The pre-registered point-level comparator is not like-for-like.** ACI's $\gamma$ was tuned to a point-wise rate, the BH layer's $(q, k)$ to the event metric. Tuned post hoc on the event metric (P4), ACI reached {{ph.p4.m1.ar8.gamma_positive.f05:.3f}} on A1 (Mission 1), so the pre-registered collapse reflects the tuning target; it still trailed the threshold layer on {{ph.p4.rule.n_below}} of {{ph.p4.rule.n_arms}} arms.

**External validity.**

- **One benchmark.** All telemetry conclusions rest on one benchmark with two usable missions, and Mission 2 is close to saturated. The selection effect also changes sign between the full and lightweight Mission 1 channel sets.
- **Unseen event classes.** {{data.esa.m1.classes.test_unseen}} Mission 1 test event classes are absent from training.
- **Simulator-only and synthetic results.** These (the learned monitor, synthetic scenes, the surrogate and the original pipeline's footprints) say nothing about real spacecraft.

**Comparability with published numbers.** Published ESA-ADB results use different protocols ([@tbl:lit-esa]). The out-of-limit reference of Brancato et al. could not be reproduced, and their full text was not available.

## Limitations of the implementation

Appendix C lists every defect found during this work. The ones that limit the conclusions most are:

- **Hazard rule.** It uses the unweighted argmax as the driver and does not propagate $P_{\text{frag}}$.
- **Monte Carlo footprint.** It shares one flight-path-angle draw across all samples and ignores the configured ballistic-coefficient and wind dispersions.
- **Risk.** A per-fragment casualty expectation is compared with a per-event threshold. The event-level debris area is 1.35× too large. The casualty area, which adds the person's dimension, grows by less.
- **Atmosphere and break-up physics.** The atmosphere is static. The break-up $\Delta v$ law applies the collision slope to every break-up type, which gives explosion fragments far smaller velocity increments than the Standard Breakup Model.
- **XGBoost-AFT.** Its landmark predictions use all boosting rounds.
- **Telemetry preprocessing.** It inherits a one-sample look-ahead in differenced monotonic channels from the official scripts.

## Future work

The following experiments would most directly strengthen the evidence. Each is stated as a new study, because the test halves used here have now been seen.

1. **Drift-aware supervised scoring.** Recalibrate supervised scorers on trailing unlabelled data with a layer selected for that purpose, or retrain them causally (forward chaining). Pre-register the comparison against A1 and A2. Test it on data not yet used: a later ESA release or another operator's telemetry.
2. **A factorial, pre-registered decomposition on new data.** The post-hoc one-factor-at-a-time decomposition ([@sec:results-posthoc]) attributes H6 to the selection data along two orders; a full factorial on unused data would also identify interactions.
3. **Forward chaining on new data.** The post-hoc forward-chaining folds did not replicate the gain on Mission 1 ({{ph.p2.m1.conformal_bh.n_folds_H_better}} of {{ph.p2.m1.conformal_bh.n_folds}} folds); a prospective test on a later data release is the decisive check.
4. **A wider, pre-registered selection grid with more seeds.** At least ten seeds for the supervised arms, and anomaly-only selection as a separate pre-registered objective.
5. **A reproduction of Brancato et al.'s protocol** through the same evaluation code, if their implementation becomes available.
6. **Validation of the re-entry chain against recorded events.** This needs ESA DISCOS or Space-Track re-entry records with recovery coordinates. It would be the first test of predicted landing *locations*.
7. **Fixes to the physics chain** listed in Appendix C. Each fix should be followed by re-verification: per-sample flight-path-angle draws, the event-level risk threshold, the use of $P_{\text{frag}}$, a tie-break for the hazard driver, solar-activity-dependent density, and a standard SBM.
8. **Temporal post-processing for SpotGEO**, such as trajectory linking across the five frames, which the challenge winners used.
9. **A multi-landmark re-entry study** with element-set histories, which would allow more events and a time-split test.

The following items are possibilities, not commitments:

- foundation-model fine-tuning, which is not implemented;
- learned fusion of evidence across features;
- an operator-facing interface.
