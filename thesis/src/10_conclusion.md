# Conclusion {#sec:conclusion}

This thesis documented COSMOS as it is currently implemented. COSMOS is a framework that chains simulated satellite health, break-up and re-entry physics, and adds a machine-learning layer evaluated on real public data. The thesis reported every result through an evidence file that traces each number to its source.

**Methodology.**

- The physics components were checked by an automated suite: {{verify.n_external}} of its checks compare with external references, and the rest are self-consistency checks.
- The behaviour of the chain was characterised with fixed seeds.
- The learned components were evaluated on ESA-ADB, OPSSAT-AD, SpotGEO and a leakage-checked CelesTrak landmark, under official or leakage-controlled protocols and against strong classical baselines.
- The principal study was pre-registered and hash-frozen before scoring.

**Principal findings.**

1. **Whole-training-half selection was supported on Mission 1.** On ESA-ADB under the official protocol at full channel scale, the protocol that selects an event-level alarm layer on out-of-fold scores of the whole training half raised Mission 1 from {{pre.m1.A0.conformal_bh.f05.mean:.3f}} to {{pre.m1.A1.conformal_bh.f05.mean:.3f}} (H6 supported, Holm-adjusted $p$ ≤ {{conf.p_floor_holm:g}}). The gain held under a per-channel threshold as well. The two arms differ in five respects, two of which change the scores, so the gain belongs to the protocol as a whole; a post-hoc decomposition traced it to the selection data, and forward-chaining folds inside the training half did not replicate it. The resulting layer raises about one false alarm per year, but about {{pre.m1.A1.conformal_bh.alarm_runs_per_year:.0f}} alarm runs in total, and it detects about one target event in six.
2. **The supervised deep event scorer showed no benefit.** Its point estimates were below both the AR(8) forecaster and gradient-boosted trees on the same features (H4 and H5 refuted: neither reached the minimum effect). An exploratory diagnostic found that the supervised scorers' nominal score distributions shift between training and test, but the size of the shift did not track the test score across runs.
3. **The selection effect is conditional.** It reversed on the near-saturated Mission 2 under the BH layer, but not under the threshold. It also reversed on the lightweight Mission 1 subset under both layers. Mission 2 is already handled well by a parameter-free out-of-limit rule ({{explo.m2.full.ool.native.f05.mean:.3f}}).
4. **Point-level adaptive conformal calibration failed at the event level** on both missions, and adding event logic did not repair it, for a mechanical reason. What made alarms usable was aggregation to events with a strict operating point. A per-channel threshold did as well as cross-channel BH in most comparisons.
5. **The supporting studies show no single pattern by model family.** Deep models led on OPSSAT-AD's official split, even on its tabular features, but the advantage shrank under a time-blocked split. Trees beat the neural hazard model on the re-entry landmark, and the physics lifetime was competitive on uncontrolled objects at 30 days. Deep heat-map detectors improved on the classical optical front end but remained far from the SpotGEO winner.
6. **The physics pipeline implements its equations component by component.** Its coupling of health to re-entry is a hand-set rule whose outputs are only partly used, and it has not been validated against real events.

**Implications.** For spacecraft telemetry evaluated with operator-grade metrics, the alarm layer, the data used to select it and the stability of the score distribution mattered more than model capacity. A selection that works on one channel set may fail on another, so the selected setting should be checked on a held-out period. Claims that deep models solve such benchmarks should be tested under the official protocol, against equally well-calibrated simple scorers, and with anomaly-only metrics reported.

**Remaining challenges.**

- A factorial test of what drives the selection effect, and drift-aware scoring, both on unseen data.
- Validation of predicted re-entry locations against recorded events.
- Correction of the documented defects in the physics chain.

The roadmap in Appendix D describes how the telemetry study can be turned into a focused journal manuscript.

\newpage

# References {.unnumbered}

::: {#refs}
:::
