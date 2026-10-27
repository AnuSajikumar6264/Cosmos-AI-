# Results and Evaluation {#sec:results}

The results are presented in the order in which their evidence is strongest:

1. verification of the physics components;
2. the behaviour of the original pipeline;
3. the ESA-ADB studies;
4. the supporting studies.

Every value in this chapter is inserted from the evidence file. Values computed from simulated or synthetic data are labelled.

## Verification of the physics components {#sec:results-verify}

All {{tests.verify.total}} checks of the verification suite pass ([@tbl:verify]). They differ in strength, and the table states what each one compares against:

- {{verify.n_external}} compare with an answer that does not come from the COSMOS code: a published table, a closed form, a physical limit, an imposed shift or a statistical guarantee;
- {{verify.n_internal}} are self-consistency checks or hold by construction, for example a constant compared with itself or a mass budget that the code enforces;
- {{verify.n_misspecified}} check, the hydrostatic one, is mis-specified. It compares $dp/dz$ with $-\rho g_0$ instead of $-\rho g(z)$, so its {{verify.3.measured}} error is the ratio $g(30\,\text{km})/g_0$, not a numerical error, and the check would fail its own 1 % limit above about 33 km.

The checks print PASS or FAIL but do not raise an exception, so under pytest a failing check would still pass (Appendix C).

: Verification checks of the original pipeline (`tests/test_verify.py`; log `outputs/tests/test_verify_after.log`). "Reference" states what the expected value is based on. {#tbl:verify}

| Check | Reference | Expected | Measured |
|---|---|---|---|
| {{verify.1.name}} | {{verify.1.reference}} | {{verify.1.expected}} | {{verify.1.measured}} |
| {{verify.2.name}} | {{verify.2.reference}} | {{verify.2.expected}} | {{verify.2.measured}} |
| {{verify.3.name}} | {{verify.3.reference}} | {{verify.3.expected}} | {{verify.3.measured}} |
| {{verify.4.name}} | {{verify.4.reference}} | {{verify.4.expected}} | {{verify.4.measured}} |
| {{verify.5.name}} | {{verify.5.reference}} | {{verify.5.expected}} | {{verify.5.measured}} |
| {{verify.6.name}} | {{verify.6.reference}} | {{verify.6.expected}} | {{verify.6.measured}} |
| {{verify.7.name}} | {{verify.7.reference}} | {{verify.7.expected}} | {{verify.7.measured}} |
| {{verify.8.name}} | {{verify.8.reference}} | {{verify.8.expected}} | {{verify.8.measured}} |
| {{verify.9.name}} | {{verify.9.reference}} | {{verify.9.expected}} | {{verify.9.measured}} |
| {{verify.10.name}} | {{verify.10.reference}} | {{verify.10.expected}} | {{verify.10.measured}} |
| {{verify.11.name}} | {{verify.11.reference}} | {{verify.11.expected}} | {{verify.11.measured}} |
| {{verify.12.name}} | {{verify.12.reference}} | {{verify.12.expected}} | {{verify.12.measured}} |
| {{verify.13.name}} | {{verify.13.reference}} | {{verify.13.expected}} | {{verify.13.measured}} |
| {{verify.14.name}} | {{verify.14.reference}} | {{verify.14.expected}} | {{verify.14.measured}} |
| {{verify.15.name}} | {{verify.15.reference}} | {{verify.15.expected}} | {{verify.15.measured}} |
| {{verify.16.name}} | {{verify.16.reference}} | {{verify.16.expected}} | {{verify.16.measured}} |
| {{verify.17.name}} | {{verify.17.reference}} | {{verify.17.expected}} | {{verify.17.measured}} |
| {{verify.18.name}} | {{verify.18.reference}} | {{verify.18.expected}} | {{verify.18.measured}} |
| {{verify.19.name}} | {{verify.19.reference}} | {{verify.19.expected}} | {{verify.19.measured}} |
| {{verify.20.name}} | {{verify.20.reference}} | {{verify.20.expected}} | {{verify.20.measured}} |

These checks cover how individual components behave. They do not test the hazard rule, the Monte Carlo footprint, or the risk estimate as a whole. Nor do they test anything against a real re-entry.

The machine-learning layer has {{tests.ai.passed}} checks in `tests/test_ai.py`, all passing with {{tests.ai.failed}} failures. Among other things they compare:

- the preprocessing with a literal port of the official code;
- the official metric with a hand-solved case;
- the conformal p-values and BH procedure with their theoretical properties;
- the TCN and the event model for causality;
- the original pipeline's output with the untouched baseline.

The original dataset tests have one pre-existing error under pandas 3: a safe-unpickler test fails on a pandas `ArrowStringArray`. The last line of that log is "{{tests.datasets.log_tail}}".

## Behaviour of the original pipeline (simulated) {#sec:results-orig}

All results in this section were regenerated with the current code (package version {{orig.version}}) by `thesis/tools/run_original_pipeline.py`. They use simulated telemetry and simulated re-entries.

**Health monitor.** Over all {{orig.monitor.n_subsystems}} subsystems and {{orig.monitor.n_faults}} fault types (excluding `stuck`, which crashes the simulator), the classical monitor raised a sustained alarm in {{orig.monitor.n_detected}} of {{orig.monitor.n_faulted}} faulted 24-hour runs. It raised no alarm on the fault-free run (false alarm: {{orig.monitor.nominal_false_alarm}}). Detection depended strongly on the fault type ([@tbl:monitor-grid]):

- slow *degradation* was detected in {{orig.monitor.degradation.detected}} of {{orig.monitor.degradation.n}} subsystems;
- step-like *dropouts* were detected in {{orig.monitor.dropout.detected}} of {{orig.monitor.dropout.n}}, with a median delay of {{orig.monitor.dropout.median_delay_min:.0f}} minutes.

Detection also depends on the subsystem. Thermal Control was detected in {{orig.monitor.sub.thermal_control.detected}} of its {{orig.monitor.n_faults}} faulted runs, the data storage unit in {{orig.monitor.sub.data_storage_unit.detected}} and every other subsystem in {{orig.monitor.sub.power_system.detected}}. Apart from slow degradation, which is missed almost everywhere, every miss is on Thermal Control. The likely reason lies in the simulator's constants (`CHANNELS`, l.25–42). A fault's deviation scales with the channel's noise level $\sigma$, while the robust scale of the fitting window is dominated by the orbital sinusoid of amplitude $A$. Thermal Control is the only subsystem in which both channels have $\sigma/A \le 0.11$ (0.04 and 0.11), so its faults stay small in $z$. Every other subsystem has a channel with $\sigma/A \ge 0.13$. [@fig:monitor-grid] shows every run.

![Detection delay of the classical health monitor for every subsystem and fault type (SIMULATED telemetry, seed 7, 24-hour runs). Hatched cells were not detected within the run. Source: `thesis/evidence/original_pipeline_runs.json`.](figures/orig_monitor_grid.png){#fig:monitor-grid width=85%}

: Classical health monitor on simulated telemetry, seed 7: number of subsystems in which each fault type was detected within 24 h, and median detection delay. {#tbl:monitor-grid}

| Fault | Detected | Median delay (min) |
|---|---|---|
| degradation | {{orig.monitor.degradation.detected}} / {{orig.monitor.degradation.n}} | {{orig.monitor.degradation.median_delay_min:.0f}} |
| leak | {{orig.monitor.leak.detected}} / {{orig.monitor.leak.n}} | {{orig.monitor.leak.median_delay_min:.0f}} |
| overheat | {{orig.monitor.overheat.detected}} / {{orig.monitor.overheat.n}} | {{orig.monitor.overheat.median_delay_min:.0f}} |
| dropout | {{orig.monitor.dropout.detected}} / {{orig.monitor.dropout.n}} | {{orig.monitor.dropout.median_delay_min:.0f}} |
| saturation | {{orig.monitor.saturation.detected}} / {{orig.monitor.saturation.n}} | {{orig.monitor.saturation.median_delay_min:.0f}} |

**Coupling.** The hazard rule maps the same anomaly score to different break-up physics depending on the driving subsystem. In the scenario table, at anomaly score 0.85:

- a thermal-driven scenario gives an aerodynamic break-up at {{orig.scen.thermal_critical.breakup_altitude_km}} km, with $P_{\text{frag}}$ = {{orig.scen.thermal_critical.p_fragmentation}};
- a power-driven scenario gives an explosion at {{orig.scen.power_critical.breakup_altitude_km}} km.

The all-nominal scenario is run at anomaly score {{orig.scen.all_nominal.anomaly_score}}, with every degradation at the same value 0.05. Its $P_{\text{frag}}$ is {{orig.scen.all_nominal.p_fragmentation}}. Because all degradations tie, `max()` returns the first subsystem in the dictionary, {{orig.scen.all_nominal.driver}}. The nominal case is therefore labelled an {{orig.scen.all_nominal.breakup_type}} at {{orig.scen.all_nominal.breakup_altitude_km}} km. This tie-break is a defect of the rule (Appendix C).

In the end-to-end comparison of failure modes ([@tbl:failure-modes]), the propulsion scenario produces an explosion at {{orig.fm.propulsion.breakup_alt_km}} km with CEP50 = {{orig.fm.propulsion.cep50_km}} km. The thermal scenario produces an aerodynamic break-up at {{orig.fm.thermal.breakup_alt_km}} km with CEP50 = {{orig.fm.thermal.cep50_km}} km.

The table also exposes a limitation of the coupling. The *structure* scenario is driven by "{{orig.fm.structure.driver}}", because the structural fault is not detected and the subsystem with the largest residual degradation becomes the driver. Its footprint (CEP50 {{orig.fm.structure.cep50_km}} km) is nearly identical to that of the *nominal* scenario ({{orig.fm.nominal.cep50_km}} km).

: Failure-mode comparison through the whole chain (simulated, 600 Monte Carlo samples per scenario). {#tbl:failure-modes}

| Scenario | Driver | Break-up | Altitude (km) | $P_{\text{frag}}$ | $\sigma_{\log B}$ | Mean downrange (km) | CEP50 (km) |
|---|---|---|---|---|---|---|---|
| nominal | {{orig.fm.nominal.driver}} | {{orig.fm.nominal.breakup_type}} | {{orig.fm.nominal.breakup_alt_km}} | {{orig.fm.nominal.P_fragmentation}} | {{orig.fm.nominal.sigma_logB}} | {{orig.fm.nominal.downrange_mean_km}} | {{orig.fm.nominal.cep50_km}} |
| thermal | {{orig.fm.thermal.driver}} | {{orig.fm.thermal.breakup_type}} | {{orig.fm.thermal.breakup_alt_km}} | {{orig.fm.thermal.P_fragmentation}} | {{orig.fm.thermal.sigma_logB}} | {{orig.fm.thermal.downrange_mean_km}} | {{orig.fm.thermal.cep50_km}} |
| structure | {{orig.fm.structure.driver}} | {{orig.fm.structure.breakup_type}} | {{orig.fm.structure.breakup_alt_km}} | {{orig.fm.structure.P_fragmentation}} | {{orig.fm.structure.sigma_logB}} | {{orig.fm.structure.downrange_mean_km}} | {{orig.fm.structure.cep50_km}} |
| propulsion | {{orig.fm.propulsion.driver}} | {{orig.fm.propulsion.breakup_type}} | {{orig.fm.propulsion.breakup_alt_km}} | {{orig.fm.propulsion.P_fragmentation}} | {{orig.fm.propulsion.sigma_logB}} | {{orig.fm.propulsion.downrange_mean_km}} | {{orig.fm.propulsion.cep50_km}} |
| comms lost | {{orig.fm.comms_lost.driver}} | {{orig.fm.comms_lost.breakup_type}} | {{orig.fm.comms_lost.breakup_alt_km}} | {{orig.fm.comms_lost.P_fragmentation}} | {{orig.fm.comms_lost.sigma_logB}} | {{orig.fm.comms_lost.downrange_mean_km}} | {{orig.fm.comms_lost.cep50_km}} |

**Impact zones.** For one simulated footprint ({{orig.fp.n_impacts}} impacts, CEP50 {{orig.fp.cep50_km}} km), the split-conformal density zone achieved near-nominal held-out coverage:

| Nominal coverage | Conformal density zone | Gaussian ellipse | Area saving of the density zone |
|---|---|---|---|
| 0.90 | {{orig.zone.90.conformal_density_coverage}} | {{orig.zone.90.gaussian_ellipse_coverage}} | {{orig.zone.90.density_area_saving_pct}} % |
| 0.95 | {{orig.zone.95.conformal_density_coverage}} | {{orig.zone.95.gaussian_ellipse_coverage}} | {{orig.zone.95.density_area_saving_pct}} % |
| 0.99 | {{orig.zone.99.conformal_density_coverage}} | {{orig.zone.99.gaussian_ellipse_coverage}} | {{orig.zone.99.density_area_saving_pct}} % |

At nominal 0.95 the density zone needs {{orig.zone.95.density_area_saving_pct}} % less area than the Gaussian ellipse. On this sample the Gaussian ellipse is itself close to nominal: its coverage shortfall is {{orig.zone.90.gaussian_shortfall_pts}}, {{orig.zone.95.gaussian_shortfall_pts}} and {{orig.zone.99.gaussian_shortfall_pts}} percentage points at the three levels, where a negative value means over-coverage. The gain from the conformal zone is therefore a smaller area, not better coverage. The earlier review's statement of "0.9927 versus 0.9750 coverage in 17 % less area" is therefore not reproduced by the current code.

**Two-frame detection (synthetic).** On 16 synthetic scenes, the registered difference with the photon-noise map gave precision {{orig.det.registered_difference.precision}}, recall {{orig.det.registered_difference.recall}} and F1 {{orig.det.registered_difference.f1}}. The single-frame baseline gave precision {{orig.det.single_frame.precision}} and F1 {{orig.det.single_frame.f1}}.

**Catalogue and lifetimes (real catalogue).** The SATCAT snapshot of 2026-08-16 contains {{data.satcat.20260816.rows:int}} objects, {{data.satcat.20260816.type.DEB:int}} of them debris. The censoring-aware area-to-mass fit used {{orig.lifetime_cal.n_fragments:int}} fragments from four break-up clouds: {{orig.lifetime_cal.n_observed:int}} decayed and {{orig.lifetime_cal.n_censored:int}} censored. It gave an area-to-mass ratio of {{orig.lifetime_cal.area_to_mass_censored}} m² kg⁻¹, which is {{orig.lifetime_cal.bias_factor}}× smaller than a fit to decayed fragments only. The ratio of modelled to measured median lifetime was:

- {{orig.lifetime_val.iridium_33.ratio}} for Iridium 33 and {{orig.lifetime_val.cosmos_2251.ratio}} for Cosmos 2251, both at 789 km;
- {{orig.lifetime_val.cosmos_1408.ratio}} for Cosmos 1408 (480 km), so the model under-predicts the lifetime of that low cloud by a factor of about three.

With a single area-to-mass ratio and a static atmosphere, the model cannot fit clouds at different altitudes and solar epochs. The earlier review's figure of "1.03× on 1,806 fragments" is not reproduced.

**Risk.** For the four predefined corridors, the per-fragment expected casualties ranged from {{orig.corr.mid_pacific_e.expected_casualties}} over the open ocean to {{orig.corr.peninsular_india_nw.expected_casualties:.2e}} over peninsular India. The corresponding event-level values computed by `corridor_risk` were {{orig.corr.peninsular_india_nw.expected_casualties_event:.3f}} for India and {{orig.corr.central_europe_e.expected_casualties_event:.3f}} for central Europe. These values assume that every fragment survives to the ground, and they rest on 300 Monte Carlo samples per corridor. They are artefacts of the simulated break-up and of the risk defects listed in Appendix C, not risk assessments of any real object.

## ESA-ADB: reproduction and label-free baselines {#sec:results-baselines}

[@tbl:baselines] compares the label-free baselines with the published values. The ported official GlobalSTD rule reproduces the published primary metric to within 0.001 in all eight configurations. Two Mission 2 cells differ in the third decimal because of rounding. This checks the whole chain: download, preprocessing, scoring adapter and official metric code. The parameter-free out-of-limit rule fails on Mission 1. On Mission 2 it scores {{explo.m2.full.ool.native.f05.mean:.3f}} (full channels) and {{explo.m2.lw.ool.native.f05.mean:.3f}} (lightweight subset), above every value published with the benchmark.

: Corrected event-wise $F_{0.5}$ of label-free baselines on the official test halves. Published values (DMLR Table 4 [@kotowski2026esaadb]) are in parentheses. {#tbl:baselines}

| Detector | M1 full | M1 lightweight | M2 full | M2 lightweight |
|---|---|---|---|---|
| best published | {{lit.dmlr.m1.full.best:.3f}} | {{lit.dmlr.m1.lw.best:.3f}} | {{lit.dmlr.m2.full.best:.3f}} | {{lit.dmlr.m2.lw.best:.3f}} |
| GlobalSTD3 (official rule) | {{explo.m1.full.gstd3.native.f05.mean:.4f}} (<0.001) | {{explo.m1.lw.gstd3.native.f05.mean:.4f}} ({{lit.dmlr.m1.lw.gstd3:.3f}}) | {{explo.m2.full.gstd3.native.f05.mean:.4f}} ({{lit.dmlr.m2.full.gstd3:.3f}}) | {{explo.m2.lw.gstd3.native.f05.mean:.4f}} ({{lit.dmlr.m2.lw.gstd3:.3f}}) |
| GlobalSTD5 (official rule) | {{explo.m1.full.gstd5.native.f05.mean:.4f}} ({{lit.dmlr.m1.full.gstd5:.3f}}) | {{explo.m1.lw.gstd5.native.f05.mean:.3f}} ({{lit.dmlr.m1.lw.gstd5:.3f}}) | {{explo.m2.full.gstd5.native.f05.mean:.3f}} ({{lit.dmlr.m2.full.gstd5:.3f}}) | {{explo.m2.lw.gstd5.native.f05.mean:.3f}} ({{lit.dmlr.m2.lw.gstd5:.3f}}) |
| out-of-limit, nominal range | {{explo.m1.full.ool.native.f05.mean:.4f}} | {{explo.m1.lw.ool.native.f05.mean:.3f}} | {{explo.m2.full.ool.native.f05.mean:.3f}} | {{explo.m2.lw.ool.native.f05.mean:.3f}} |
| out-of-limit, full range | {{explo.m1.full.oolall.native.f05.mean:.4f}} | {{explo.m1.lw.oolall.native.f05.mean:.3f}} | {{explo.m2.full.oolall.native.f05.mean:.3f}} | {{explo.m2.lw.oolall.native.f05.mean:.3f}} |
| COSMOS monitor (as shipped) | {{explo.m1.full.cosmon.native.f05.mean:.4f}} | {{explo.m1.lw.cosmon.native.f05.mean:.3f}} | {{explo.m2.full.cosmon.native.f05.mean:.3f}} | {{explo.m2.lw.cosmon.native.f05.mean:.3f}} |

On Mission 1 the out-of-limit rule raises {{explo.m1.full.ool.native.fa_yr.mean:int}} false alarms per year. Brancato et al. report {{lit.brancato.ool_full:.3f}} for an out-of-limit reference on Mission 1 [@brancato2027ast]. That value is not reproduced under the official protocol as implemented here. The likely reason is a difference in protocol or in the definition of the limits, which cannot be checked without their implementation.

## ESA-ADB: exploratory study (H1–H3) {#sec:results-explo}

[@fig:esa-explo] shows the exploratory results on the full channel sets.

![Exploratory study: corrected event-wise $F_{0.5}$ on the official test halves (full channel sets). Bars show seed means; whiskers show the standard deviation where more than one seed exists. The dashed line marks the best published value.](figures/esa_exploratory.png){#fig:esa-explo width=100%}

On **Mission 2**, every forecaster combined with conformal-BH scored far above the published baselines:

- AR(8): {{explo.m2.full.ar8.conformal_bh.f05.mean:.3f}};
- PatchTST: {{explo.m2.full.patchtst.conformal_bh.f05.mean:.3f}} ± {{explo.m2.full.patchtst.conformal_bh.f05.std:.3f}};
- zero-shot Chronos-Bolt: {{explo.m2.full.chronos.conformal_bh.f05.mean:.3f}}.

These arms detect only {{explo.m2.full.ar8.conformal_bh.r_anom.mean:pct1}} of the anomalies, however. Their score comes from the rare nominal events, which dominate the Mission 2 test half.

On **Mission 1**, no arm was usable:

- the best deterministic arm (AR(8) with a threshold) reached {{explo.m1.full.ar8.threshold.f05.mean:.3f}};
- AR(8) with conformal-BH reached {{explo.m1.full.ar8.conformal_bh.f05.mean:.3f}}, at {{explo.m1.full.ar8.conformal_bh.fa_yr.mean:.0f}} false alarms per year;
- the only higher mean, {{explo.m1.full.tcn_tc.conformal_bh_rolling.f05.mean:.3f}} ± {{explo.m1.full.tcn_tc.conformal_bh_rolling.f05.std:.3f}}, came from the TCN with telecommands and the rolling layer. Its three seeds ranged from {{explo.m1.full.tcn_tc.conformal_bh_rolling.f05.min:.3f}} to {{explo.m1.full.tcn_tc.conformal_bh_rolling.f05.max:.3f}}, so it was not a stable arm.

On the lightweight Mission 1 subset, by contrast, AR(8) with conformal-BH reached {{explo.m1.lw.ar8.conformal_bh.f05.mean:.3f}}. The failure on Mission 1 was a failure at full channel scale.

[@tbl:explo-boot] gives the paired bootstrap comparisons.

: Exploratory paired block-bootstrap comparisons (seed 0, 30-day blocks). "Difference" is the re-implemented $F_{0.5}$ difference A − B with its 95 % interval. {#tbl:explo-boot}

| Comparison | A (official) | B (official) | Difference | 95 % CI |
|---|---|---|---|---|
| {{explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.name}} | {{explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official:.3f}} | {{explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official:.3f}} | {{explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.diff_reimpl:+.3f}} | [{{explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_low:+.3f}}, {{explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_high:+.3f}}] |
| {{explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.name}} | {{explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official:.3f}} | {{explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official:.3f}} | {{explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.diff_reimpl:+.3f}} | [{{explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.ci95_low:+.3f}}, {{explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.ci95_high:+.3f}}] |
| {{explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.name}} | {{explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.A_EW_F05_official:.3f}} | {{explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.B_EW_F05_official:.3f}} | {{explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.diff_reimpl:+.3f}} | [{{explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_low:+.3f}}, {{explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_high:+.3f}}] |
| {{explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.name}} | {{explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official:.3f}} | {{explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official:.3f}} | {{explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.diff_reimpl:+.3f}} | [{{explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_low:+.3f}}, {{explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_high:+.3f}}] |
| {{explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.name}} | {{explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official:.3f}} | {{explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official:.3f}} | {{explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.diff_reimpl:+.3f}} | [{{explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.ci95_low:+.3f}}, {{explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.ci95_high:+.3f}}] |
| {{explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.name}} | {{explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.A_EW_F05_official:.3f}} | {{explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.B_EW_F05_official:.3f}} | {{explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.diff_reimpl:+.3f}} | [{{explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_low:+.3f}}, {{explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_high:+.3f}}] |
| {{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.name}} | {{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.A_EW_F05_official:.3f}} | {{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.B_EW_F05_official:.3f}} | {{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.diff_reimpl:+.3f}} | [{{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.ci95_low:+.3f}}, {{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.ci95_high:+.3f}}] |
| {{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.name}} | {{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.A_EW_F05_official:.3f}} | {{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.B_EW_F05_official:.3f}} | {{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.diff_reimpl:+.3f}} | [{{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.ci95_low:+.3f}}, {{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.ci95_high:+.3f}}] |
| {{explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.name}} | {{explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.A_EW_F05_official:.3f}} | {{explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.B_EW_F05_official:.3f}} | {{explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.diff_reimpl:+.3f}} | [{{explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.ci95_low:+.3f}}, {{explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.ci95_high:+.3f}}] |
| {{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.name}} | {{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.A_EW_F05_official:.3f}} | {{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.B_EW_F05_official:.3f}} | {{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.diff_reimpl:+.3f}} | [{{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.ci95_low:+.3f}}, {{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.ci95_high:+.3f}}] |
| {{explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.name}} | {{explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.A_EW_F05_official:.3f}} | {{explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.B_EW_F05_official:.3f}} | {{explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.diff_reimpl:+.3f}} | [{{explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.ci95_low:+.3f}}, {{explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.ci95_high:+.3f}}] |
| {{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.name}} | {{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.A_EW_F05_official:.3f}} | {{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.B_EW_F05_official:.3f}} | {{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.diff_reimpl:+.3f}} | [{{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.ci95_low:+.3f}}, {{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.ci95_high:+.3f}}] |
| {{explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.name}} | {{explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.A_EW_F05_official:.3f}} | {{explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.B_EW_F05_official:.3f}} | {{explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.diff_reimpl:+.3f}} | [{{explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.ci95_low:+.3f}}, {{explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.ci95_high:+.3f}}] |

**Summary of the exploratory hypotheses.**

- **H1 is not supported.** No deep or pretrained encoder beat AR(8) under the same calibration. On Mission 1, Chronos-Bolt was significantly worse.
- **H2 holds on Mission 2 only.** Conformal-BH beat the per-channel threshold on Mission 2 by {{explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.diff_reimpl:+.3f}} for AR(8). On Mission 1 the difference was {{explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.diff_reimpl:+.3f}}.
- **H3 is not supported.** H3 predicted fewer alarm runs touching only rare events when the encoder receives telecommands. Seed means of those runs (conformal-BH):
  - TCN: {{explo.m1.full.tcn.conformal_bh.rare_only.mean:.0f}} without and {{explo.m1.full.tcn_tc.conformal_bh.rare_only.mean:.0f}} with telecommands on Mission 1, and {{explo.m2.full.tcn.conformal_bh.rare_only.mean:.0f}} and {{explo.m2.full.tcn_tc.conformal_bh.rare_only.mean:.0f}} on Mission 2;
  - PatchTST: {{explo.m1.full.patchtst.conformal_bh.rare_only.mean:.0f}} and {{explo.m1.full.patchtst_tc.conformal_bh.rare_only.mean:.0f}} on Mission 1, and {{explo.m2.full.patchtst.conformal_bh.rare_only.mean:.0f}} and {{explo.m2.full.patchtst_tc.conformal_bh.rare_only.mean:.0f}} on Mission 2.

  The reductions are small, and PatchTST doubles these runs on Mission 1. The primary metric agrees: telecommands lowered both encoders on Mission 1 ({{explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.diff_reimpl:+.3f}} and {{explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.diff_reimpl:+.3f}}, both intervals below zero), and changed them by about +0.01 on Mission 2. A trained gated model is also insensitive to its telecommand input: removing the stream at test time left PatchTST on Mission 2 at {{explo.robust.m2_target_patchtst_tc_s0_telecommands_removed.conformal_bh.f05:.3f}} (with the stream: {{explo.robust.m2_target_patchtst_tc_s0_telecommands_removed.conformal_bh.f05_with_tc:.3f}}). This mirrors the pre-registered negative result of TC-CAD [@sahin2026tccad].

**A fixed telecommand rule.** A simple rule was also tested: suppress alarms for 30 minutes after any priority-3 telecommand. On Mission 2 it reduced alarm runs that touch only rare events from {{explo.robust.m2_target_ar8_s0_telecommand_mask.mask0.rare_only}} to {{explo.robust.m2_target_ar8_s0_telecommand_mask.mask30.rare_only}}, and anomaly recall stayed at {{explo.robust.m2_target_ar8_s0_telecommand_mask.mask30.r_anom:.3f}}. Because the primary metric counts rare events as targets, the primary score fell at the same time from {{explo.robust.m2_target_ar8_s0_telecommand_mask.mask0.f05:.3f}} to {{explo.robust.m2_target_ar8_s0_telecommand_mask.mask30.f05:.3f}}.

## ESA-ADB: pre-registered confirmatory study {#sec:results-prereg}

### Test-half results of the arms

[@tbl:prereg-arms] and [@fig:prereg-arms] give the test-half results under the primary conformal-BH layer.

: Pre-registered arms on the official test halves, conformal-BH layer (mean ± sd over seeds; A0 and A1 are deterministic). "CV" is the selection objective, i.e. the cross-fitted training-half $F_{0.5}$ of the chosen setting. {#tbl:prereg-arms}

| Mission | Arm | Test $F_{0.5}$ | Precision | Recall | Anomaly-only $F_{0.5}$ | False alarms / yr | CV (training half) |
|---|---|---|---|---|---|---|---|
| 1 | A0 | {{pre.m1.A0.conformal_bh.f05.mean:.3f}} | {{pre.m1.A0.conformal_bh.p.mean:.3f}} | {{pre.m1.A0.conformal_bh.r.mean:.3f}} | {{pre.m1.A0.conformal_bh.f05_anom.mean:.3f}} | {{pre.m1.A0.conformal_bh.fa_yr.mean:.1f}} | — |
| 1 | A1 | {{pre.m1.A1.conformal_bh.f05.mean:.3f}} | {{pre.m1.A1.conformal_bh.p.mean:.3f}} | {{pre.m1.A1.conformal_bh.r.mean:.3f}} | {{pre.m1.A1.conformal_bh.f05_anom.mean:.3f}} | {{pre.m1.A1.conformal_bh.fa_yr.mean:.1f}} | {{sel.m1.A1.conformal_bh.cv.mean:.3f}} |
| 1 | A2 | {{pre.m1.A2.conformal_bh.f05.mean:.3f}} ± {{pre.m1.A2.conformal_bh.f05.std:.3f}} | {{pre.m1.A2.conformal_bh.p.mean:.3f}} | {{pre.m1.A2.conformal_bh.r.mean:.3f}} | {{pre.m1.A2.conformal_bh.f05_anom.mean:.3f}} | {{pre.m1.A2.conformal_bh.fa_yr.mean:.1f}} | {{sel.m1.A2.conformal_bh.cv.mean:.3f}} |
| 1 | A3 | {{pre.m1.A3.conformal_bh.f05.mean:.3f}} ± {{pre.m1.A3.conformal_bh.f05.std:.3f}} | {{pre.m1.A3.conformal_bh.p.mean:.3f}} | {{pre.m1.A3.conformal_bh.r.mean:.3f}} | {{pre.m1.A3.conformal_bh.f05_anom.mean:.3f}} | {{pre.m1.A3.conformal_bh.fa_yr.mean:.1f}} | {{sel.m1.A3.conformal_bh.cv.mean:.3f}} |
| 2 | A0 | {{pre.m2.A0.conformal_bh.f05.mean:.3f}} | {{pre.m2.A0.conformal_bh.p.mean:.3f}} | {{pre.m2.A0.conformal_bh.r.mean:.3f}} | {{pre.m2.A0.conformal_bh.f05_anom.mean:.3f}} | {{pre.m2.A0.conformal_bh.fa_yr.mean:.1f}} | — |
| 2 | A1 | {{pre.m2.A1.conformal_bh.f05.mean:.3f}} | {{pre.m2.A1.conformal_bh.p.mean:.3f}} | {{pre.m2.A1.conformal_bh.r.mean:.3f}} | {{pre.m2.A1.conformal_bh.f05_anom.mean:.3f}} | {{pre.m2.A1.conformal_bh.fa_yr.mean:.1f}} | {{sel.m2.A1.conformal_bh.cv.mean:.3f}} |
| 2 | A2 | {{pre.m2.A2.conformal_bh.f05.mean:.3f}} ± {{pre.m2.A2.conformal_bh.f05.std:.3f}} | {{pre.m2.A2.conformal_bh.p.mean:.3f}} | {{pre.m2.A2.conformal_bh.r.mean:.3f}} | {{pre.m2.A2.conformal_bh.f05_anom.mean:.3f}} | {{pre.m2.A2.conformal_bh.fa_yr.mean:.1f}} | {{sel.m2.A2.conformal_bh.cv.mean:.3f}} |
| 2 | A3 | {{pre.m2.A3.conformal_bh.f05.mean:.3f}} ± {{pre.m2.A3.conformal_bh.f05.std:.3f}} | {{pre.m2.A3.conformal_bh.p.mean:.3f}} | {{pre.m2.A3.conformal_bh.r.mean:.3f}} | {{pre.m2.A3.conformal_bh.f05_anom.mean:.3f}} | {{pre.m2.A3.conformal_bh.fa_yr.mean:.1f}} | {{sel.m2.A3.conformal_bh.cv.mean:.3f}} |

![Pre-registered arms, conformal-BH layer. Bars show seed means; dots show individual seeds. The dotted line marks the out-of-limit rule.](figures/esa_prereg_arms.png){#fig:prereg-arms width=100%}

The seed spread is large for the supervised arms on Mission 1:

- trees range from {{pre.m1.A2.conformal_bh.f05.min:.3f}} to {{pre.m1.A2.conformal_bh.f05.max:.3f}};
- the TCN ranges from {{pre.m1.A3.conformal_bh.f05.min:.3f}} to {{pre.m1.A3.conformal_bh.f05.max:.3f}}.

The best tree seed (seed 4, {{pre.seed.m1_target_gbt_ar8_s4_event.conformal_bh.f05:.3f}}) raises {{pre.seed.m1_target_gbt_ar8_s4_event.conformal_bh.fa_yr:.1f}} false alarms per year. Seed 1 ({{pre.seed.m1_target_gbt_ar8_s1_event.conformal_bh.f05:.3f}}) raises {{pre.seed.m1_target_gbt_ar8_s1_event.conformal_bh.fa_yr:.1f}}. No single seed is representative; the confirmatory tests use all of them.

### Confirmatory tests

[@tbl:confirmatory] and [@fig:forest] report the confirmatory tests.

: Confirmatory comparisons (official seed-mean difference; 95 % interval and p from the paired hierarchical block bootstrap with {{conf.m1.h6.n_boot:int}} replicates). H4: A3 > A1; H5: A3 > A2; H6: A1 > A0. Mission 1 is the pre-registered family (Holm); Mission 2 is a replication outside the family, with unadjusted p-values. Freeze `{{conf.freeze_id}}`. {#tbl:confirmatory}

| Mission | Test | A | B | Δ (official) | Δ (re-impl.) | 95 % CI | $p_{\text{Holm}}$ | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | H4 | {{conf.m1.h4.A_official:.3f}} | {{conf.m1.h4.B_official:.3f}} | {{conf.m1.h4.diff_official:+.3f}} | {{conf.m1.h4.diff:+.3f}} | [{{conf.m1.h4.ci95_low:+.3f}}, {{conf.m1.h4.ci95_high:+.3f}}] | {{conf.m1.h4.p_holm:.4f}} | **{{conf.m1.h4.verdict}}** |
| 1 | H5 | {{conf.m1.h5.A_official:.3f}} | {{conf.m1.h5.B_official:.3f}} | {{conf.m1.h5.diff_official:+.3f}} | {{conf.m1.h5.diff:+.3f}} | [{{conf.m1.h5.ci95_low:+.3f}}, {{conf.m1.h5.ci95_high:+.3f}}] | {{conf.m1.h5.p_holm:.4f}} | **{{conf.m1.h5.verdict}}** |
| 1 | H6 | {{conf.m1.h6.A_official:.3f}} | {{conf.m1.h6.B_official:.3f}} | {{conf.m1.h6.diff_official:+.3f}} | {{conf.m1.h6.diff:+.3f}} | [{{conf.m1.h6.ci95_low:+.3f}}, {{conf.m1.h6.ci95_high:+.3f}}] | {{conf.m1.h6.p_holm:.4f}} | **{{conf.m1.h6.verdict}}** |
| 2 | H4 | {{conf.m2.h4.A_official:.3f}} | {{conf.m2.h4.B_official:.3f}} | {{conf.m2.h4.diff_official:+.3f}} | {{conf.m2.h4.diff:+.3f}} | [{{conf.m2.h4.ci95_low:+.3f}}, {{conf.m2.h4.ci95_high:+.3f}}] | — | replication |
| 2 | H5 | {{conf.m2.h5.A_official:.3f}} | {{conf.m2.h5.B_official:.3f}} | {{conf.m2.h5.diff_official:+.3f}} | {{conf.m2.h5.diff:+.3f}} | [{{conf.m2.h5.ci95_low:+.3f}}, {{conf.m2.h5.ci95_high:+.3f}}] | — | replication |
| 2 | H6 | {{conf.m2.h6.A_official:.3f}} | {{conf.m2.h6.B_official:.3f}} | {{conf.m2.h6.diff_official:+.3f}} | {{conf.m2.h6.diff:+.3f}} | [{{conf.m2.h6.ci95_low:+.3f}}, {{conf.m2.h6.ci95_high:+.3f}}] | — | replication |

![Confirmatory and exploratory differences with 95 % bootstrap intervals. The dotted line marks the minimum effect size of interest (+0.05).](figures/esa_forest.png){#fig:forest width=90%}

The verdicts follow the pre-registered rules.

- **H6 is supported on Mission 1.** The A1 protocol scored {{conf.m1.h6.diff_official:+.3f}} above A0. Every bootstrap replicate favoured A1, so the raw $p$ sits at the floor of {{conf.p_floor:g}} and the Holm-adjusted value is $p$ ≤ {{conf.p_floor_holm:g}}. The selected layer is strict: $q$ = {{sel.m1_target_ar8.conformal_bh.q:g}} and $k$ = {{sel.m1_target_ar8.conformal_bh.k:g}}.
- **H4 and H5 are refuted.** Under the pre-registered rule, *refuted* means that the upper end of the 95 % interval lies below the minimum effect of +0.05. It does not mean that the TCN is shown to be worse. Its point estimates are below both comparators, but the H4 interval includes zero ([{{conf.m1.h4.ci95_low:+.3f}}, {{conf.m1.h4.ci95_high:+.3f}}]) and the Holm-adjusted $p$ for H5 is {{conf.m1.h5.p_holm:.2f}}.
- **The Mission 2 replication reverses H6.** Selection on the whole training half was worse than selection on the 4.5-month calibration window, by {{conf.m2.h6.diff_official:+.3f}}.

**What H6 compares.** A0 and A1 differ in more than the selection data, so H6 is a comparison of two complete protocols:

| | A0 | A1 |
|---|---|---|
| AR(8) fitted on | the fitting period ({{cfg.periods.m1.fit}}) | the whole training half, out of fold |
| Calibration pool | scores of the calibration period | out-of-fold scores of the whole training half |
| Residual scale | out-of-sample residuals of the early-stopping period | in-sample residuals over the training half |
| Selection data | the calibration period ({{data.esa.m1.events_target.cal.events}} target events) | the whole training half ({{data.esa.m1.events_all.train.targets}} target events) |
| Selection grid | the exploratory grid: $q \in$ {{cfg.explo.levels.conformal_bh}}, $k \in$ {{cfg.explo.ks}} | the pre-registered grid: $q \in$ {{cfg.prereg.levels.conformal_bh}}, $k \in$ {{cfg.prereg.ks}} |

The pre-registration states that all arms use the same grids. A0 was instead taken from the exploratory study, unchanged. This inconsistency is logged as a deviation (Appendix B). A later code review found a fifth difference, the residual scale; it and the fitting data change the test scores themselves. The study cannot attribute the gain to any one of the five differences.

**Alarm burden.** False alarms fell from {{pre.m1.A0.conformal_bh.fa_yr.mean:.1f}} to {{pre.m1.A1.conformal_bh.fa_yr.mean:.1f}} per year, and alarm runs of any kind from {{pre.m1.A0.conformal_bh.alarm_runs_per_year:.0f}} to {{pre.m1.A1.conformal_bh.alarm_runs_per_year:.0f}} per year. The false-alarm count understates the operator's load. Most A1 alarm runs fall inside an event that has already been alarmed. Alarming precision is {{pre.m1.A1.conformal_bh.alarm_prec.mean:.3f}}: of the alarm runs that touch an event, only about one in seven is the first on that event. Mission 1 events often last days, longer than the {{cfg.prereg.merge_h}} h merge gap, so one event can produce many runs. Recall is {{pre.m1.A1.conformal_bh.r.mean:.3f}}: A1 detects about one target event in six. [@fig:alarm-burden] sets these quantities side by side for every arm.

![Alarm burden of the pre-registered arms (conformal-BH layer, seed means). Each bar splits the alarm runs per year of test half into runs that touch a target event and false-alarm runs; the label gives the total and the number of target events detected (recall times the number of test events). Source: `oof_protocol/test/*.json`.](figures/esa_alarm_burden.png){#fig:alarm-burden width=100%}

### Cross-validation versus test

The selection objective over-estimated test performance for every Mission 1 arm, but by very different amounts:

| Arm | CV (training half) | Test |
|---|---|---|
| AR(8), A1 | {{sel.m1.A1.conformal_bh.cv.mean:.3f}} | {{pre.m1.A1.conformal_bh.f05.mean:.3f}} |
| trees, A2 | {{sel.m1.A2.conformal_bh.cv.mean:.3f}} | {{pre.m1.A2.conformal_bh.f05.mean:.3f}} |
| TCN, A3 | {{sel.m1.A3.conformal_bh.cv.mean:.3f}} | {{pre.m1.A3.conformal_bh.f05.mean:.3f}} |

[@fig:cv-test] shows the same comparison for both missions and for the lightweight Mission 1 arm, whose cross-validated value fell from {{sel.m1.A1lw.conformal_bh.cv.mean:.3f}} to {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} on test. On Mission 2 every arm lies close to the diagonal.

![Selection objective against test result for every arm with training-half selection (conformal-BH layer, seed means). Points below the dashed line did worse on test than in cross-validation. Sources: `oof_protocol/selection/*.json`, `oof_protocol/test/*.json`.](figures/esa_cv_vs_test.png){#fig:cv-test width=65%}

[@fig:landscape] shows the cross-validated landscape over the selection grid. For AR(8) on Mission 1 the optimum is a narrow ridge at $k = 10$. The trees' landscape is broad, with many settings within a few hundredths of the best. The TCN's (seed 0) is flat and low. On Mission 2 the AR(8) landscape is a broad plateau.

![Training-half cross-validated $F_{0.5}$ over the selection grid of the conformal-BH layer (training labels only).](figures/esa_selection_landscape.png){#fig:landscape width=100%}

## Alarm layers, ablations and the point-level comparator {#sec:results-ablations}

[@tbl:layers] compares the alarm layers on identical scores.

: Test-half $F_{0.5}$ by alarm layer (seed means). ACI is the point-level adaptive conformal comparator; "ACI + events" adds the persistence and merging of the conformal-BH layer. {#tbl:layers}

| Mission | Arm | threshold | conformal-BH | rolling BH | ACI | ACI + events |
|---|---|---|---|---|---|---|
| 1 | A1 | {{pre.m1.A1.threshold.f05.mean:.3f}} | {{pre.m1.A1.conformal_bh.f05.mean:.3f}} | {{pre.m1.A1.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A1.aci.f05.mean:.4f}} | {{pre.m1.A1.aci_event.f05.mean:.3f}} |
| 1 | A2 | {{pre.m1.A2.threshold.f05.mean:.3f}} | {{pre.m1.A2.conformal_bh.f05.mean:.3f}} | {{pre.m1.A2.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A2.aci.f05.mean:.4f}} | {{pre.m1.A2.aci_event.f05.mean:.3f}} |
| 1 | A3 | {{pre.m1.A3.threshold.f05.mean:.3f}} | {{pre.m1.A3.conformal_bh.f05.mean:.3f}} | {{pre.m1.A3.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A3.aci.f05.mean:.4f}} | {{pre.m1.A3.aci_event.f05.mean:.3f}} |
| 1 | A3, no residual inputs | {{pre.m1.A3nr.threshold.f05.mean:.3f}} | {{pre.m1.A3nr.conformal_bh.f05.mean:.3f}} | {{pre.m1.A3nr.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A3nr.aci.f05.mean:.4f}} | {{pre.m1.A3nr.aci_event.f05.mean:.3f}} |
| 1 | A3, two classes | {{pre.m1.A3c2.threshold.f05.mean:.3f}} | {{pre.m1.A3c2.conformal_bh.f05.mean:.3f}} | {{pre.m1.A3c2.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A3c2.aci.f05.mean:.4f}} | {{pre.m1.A3c2.aci_event.f05.mean:.3f}} |
| 1 | A1, lightweight channels | {{pre.m1.A1lw.threshold.f05.mean:.3f}} | {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} | {{pre.m1.A1lw.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A1lw.aci.f05.mean:.4f}} | {{pre.m1.A1lw.aci_event.f05.mean:.3f}} |
| 2 | A1 | {{pre.m2.A1.threshold.f05.mean:.3f}} | {{pre.m2.A1.conformal_bh.f05.mean:.3f}} | {{pre.m2.A1.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A1.aci.f05.mean:.4f}} | {{pre.m2.A1.aci_event.f05.mean:.3f}} |
| 2 | A2 | {{pre.m2.A2.threshold.f05.mean:.3f}} | {{pre.m2.A2.conformal_bh.f05.mean:.3f}} | {{pre.m2.A2.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A2.aci.f05.mean:.4f}} | {{pre.m2.A2.aci_event.f05.mean:.3f}} |
| 2 | A3 | {{pre.m2.A3.threshold.f05.mean:.3f}} | {{pre.m2.A3.conformal_bh.f05.mean:.3f}} | {{pre.m2.A3.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A3.aci.f05.mean:.4f}} | {{pre.m2.A3.aci_event.f05.mean:.3f}} |
| 2 | A1, lightweight channels | {{pre.m2.A1lw.threshold.f05.mean:.3f}} | {{pre.m2.A1lw.conformal_bh.f05.mean:.3f}} | {{pre.m2.A1lw.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A1lw.aci.f05.mean:.4f}} | {{pre.m2.A1lw.aci_event.f05.mean:.3f}} |

![Test-half $F_{0.5}$ of the same scores under three alarm layers (seed means; values of [@tbl:layers]). Where two markers coincide, the layers gave the same value (A1 on Mission 1; A1 lightweight on Mission 2). ACI is omitted because all its values are below 0.002.](figures/esa_layers.png){#fig:layers width=85%}

Four findings stand out.

1. **Point-level ACI fails at the event level.** Point-level ACI kept its point-wise false-alarm rate on nominal test data between {{pre.all.aci.point_far.min:.4f}} and {{pre.all.aci.point_far.max:.4f}} for the target $\alpha$ = {{cfg.prereg.aci_alpha}}. Its primary metric never exceeded {{pre.all.aci.f05.max:.4f}} over {{pre.all.aci.n}} runs, because it produced up to {{pre.all.aci.alarm_runs.max:int}} alarm runs. With event logic it reached {{pre.all.aci_event.f05.max:.3f}} in a single tree seed; the best seed mean of any arm was {{pre.all.aci_event.f05.max_seedmean:.3f}}, and {{pre.all.aci_event.zero_alarm_runs}} of {{pre.all.aci_event.n_runs}} runs raised no alarm at all. The reason is mechanical. The selection chose $\gamma$ = 0.05 in {{pre.all.aci.gamma005_runs}} of {{pre.all.aci.gamma_runs}} runs. With that value, one exceedance moves $\alpha_t$ from {{cfg.prereg.aci_alpha}} to {{aci.alpha_after_one_hit.g0.05:.4f}} ([@eq:aci]). The quantile then becomes infinite, and the channel cannot fire again for {{aci.recovery_steps.g0.05}} steps. In the remaining run (A1, Mission 1, $\gamma$ = 0.01) one exceedance lowers $\alpha_t$ to {{aci.alpha_after_one_hit.g0.01:g}}, so the next one needs a far more extreme score. A channel therefore cannot stay hot through an event. Because the global alarm is an OR over channels, persistence can still be met by different channels firing in successive steps, but that happens rarely. The comparison is also not like-for-like: ACI's $\gamma$ was tuned to a point-wise false-alarm rate, while the BH layer's $(q, k)$ were tuned on the event metric. The result concerns ACI at $\alpha$ = {{cfg.prereg.aci_alpha}}, not point-level calibration in general; tuned on the event metric post hoc, ACI is competitive on some arms ([@sec:results-posthoc]).
2. **No layer dominates.** The per-channel threshold, which has no cross-channel multiplicity control, equalled or beat conformal-BH in {{layers.n_threshold_ge_bh}} of the {{layers.n_rows}} rows. Conformal-BH was better only for the trees and the TCN on Mission 2. For A1 on Mission 1 the two layers give the same value. The rolling layer copies the $(q, k)$ selected for conformal-BH rather than being selected itself. It cut AR(8) on Mission 1 from {{pre.m1.A1.conformal_bh.f05.mean:.3f}} to {{pre.m1.A1.conformal_bh_rolling.f05.mean:.3f}} and the trees from {{pre.m1.A2.conformal_bh.f05.mean:.3f}} to {{pre.m1.A2.conformal_bh_rolling.f05.mean:.3f}}. It raised only the TCN, whose seeds spread widely under it, and the lightweight Mission 1 arm, whose conformal-BH result had collapsed ({{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} to {{pre.m1.A1lw.conformal_bh_rolling.f05.mean:.3f}}).
3. **The TCN ablations change little.** Neither removing the residual inputs nor merging the two event classes changed the TCN by more than its seed spread (exploratory comparisons below).
4. **Mission 1 lightweight channels.** On the lightweight Mission 1 subset, the conformal-BH layer selected on the training half scored only {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}}, with recall {{pre.m1.A1lw.conformal_bh.r.mean:.3f}}. Its cross-validated value had been {{sel.m1.A1lw.conformal_bh.cv.mean:.3f}}. The selection chose $q$ = {{sel.m1.A1lw.conformal_bh.q.mean:g}}, at the edge of the grid, with $k$ = {{sel.m1.A1lw.conformal_bh.k.mean:g}}. The threshold layer scored {{pre.m1.A1lw.threshold.f05.mean:.3f}}. Both are far below the best published lightweight value ({{lit.dmlr.m1.lw.best:.3f}}) and below the exploratory calibration-window arm ({{explo.m1.lw.ar8.conformal_bh.f05.mean:.3f}}).

**The selection effect across channel sets and layers.** [@tbl:seleff] sets the confirmatory H6 result beside the same contrast on the other channel sets and layers. These are point estimates on single deterministic runs, not tests.

: Training-half selection minus calibration-window selection for AR(8) (A1 − A0; on the lightweight subsets, A1 minus the exploratory lightweight AR(8) arm). Only the Mission 1 full-channel conformal-BH cell is a pre-registered test. {#tbl:seleff}

| Channel set | Layer | Mission 1 | Mission 2 |
|---|---|---|---|
| full | conformal-BH | {{seleff.m1.full.conformal_bh:+.3f}} | {{seleff.m2.full.conformal_bh:+.3f}} |
| full | threshold | {{seleff.m1.full.threshold:+.3f}} | {{seleff.m2.full.threshold:+.3f}} |
| lightweight | conformal-BH | {{seleff.m1.lw.conformal_bh:+.3f}} | {{seleff.m2.lw.conformal_bh:+.3f}} |
| lightweight | threshold | {{seleff.m1.lw.threshold:+.3f}} | {{seleff.m2.lw.threshold:+.3f}} |

![The selection contrast of [@tbl:seleff]. The ringed point is the pre-registered H6 comparison; the dotted line marks the minimum effect of interest (+0.05). All other points are exploratory point estimates without intervals.](figures/esa_selection_effect.png){#fig:seleff width=95%}

The gain on Mission 1 at full scale is large under both layers, so it does not depend on BH. Elsewhere the sign changes. On Mission 2 it is negative only under BH. On the lightweight Mission 1 subset it is negative under both layers, because the selection settled on a setting that did not transfer to the test half.

## Exploratory diagnostics (not pre-registered)

**Trees against AR(8).** The trees outperformed AR(8) on Mission 1 in the point estimate, by {{xcomp.m1.x1.diff_official:+.3f}} with the official metric. With seeds resampled, the re-implemented difference is {{xcomp.m1.x1.diff:+.3f}} (95 % CI [{{xcomp.m1.x1.ci95_low:+.3f}}, {{xcomp.m1.x1.ci95_high:+.3f}}]), which is not distinguishable from zero. On Mission 2 the official difference is {{xcomp.m2.x1.diff_official:+.3f}} and the re-implemented one {{xcomp.m2.x1.diff:+.3f}} [{{xcomp.m2.x1.ci95_low:+.3f}}, {{xcomp.m2.x1.ci95_high:+.3f}}].

**Score drift between halves.** [@fig:shift] compares, on nominal samples, how often each scorer exceeds the $1-10^{-4}$ quantile of its own out-of-fold pool on the test half against how often it does so in cross-validation on the training half:

- AR(8): ratio {{shift.m1.a1_ar8.ratio:.1f}};
- trees: ratios {{shift.m1.A2.ratio_min:.1f}}–{{shift.m1.A2.ratio_max:.1f}};
- TCN: ratios {{shift.m1.A3.ratio_min:.1f}}–{{shift.m1.A3.ratio_max:.1f}}.

The supervised scorers, and above all two TCN seeds, shift their nominal score distribution between training and test: a calibrated layer then raises more alarms on the test half than the selection expected. The shift does not explain the differences in test score, however:

- TCN seed 1 has the second-largest shift ({{shift.m1.a3_s1.ratio:.1f}}×) and the best TCN score ({{pre.seed.m1_target_event_ar8_all_c3_s1_event.conformal_bh.f05:.3f}});
- TCN seed 2 has the smallest shift ({{shift.m1.a3_s2.ratio:.1f}}×) and scores {{pre.seed.m1_target_event_ar8_all_c3_s2_event.conformal_bh.f05:.3f}};
- the trees shift as much as TCN seeds 2–4 ({{shift.m1.A2.ratio_min:.1f}}–{{shift.m1.A2.ratio_max:.1f}}× against {{shift.m1.a3_s2.ratio:.1f}}–{{shift.m1.a3_s4.ratio:.1f}}×), yet score far higher.

The rank correlation between shift and test $F_{0.5}$ is {{shift.m1.A3.spearman_f05:.2f}} over the TCN seeds, {{shift.m1.A2.spearman_f05:.2f}} over the tree seeds and {{shift.m1.supervised.spearman_f05:.2f}} over all ten supervised runs. With five seeds per arm these correlations are descriptive only, but they give no support to shift as the cause of the TCN's low scores. The shift is a property of the supervised scorers that an operator would have to monitor. It is not shown to be what separates good from bad runs ([@fig:drift-score]).

![Exploratory: calibration shift against test-half $F_{0.5}$ for every supervised run on Mission 1 (conformal-BH layer). Each point is one seed. Sources: `oof_protocol/exploratory/calibration_shift_m1.json`, `oof_protocol/test/*.json`.](figures/esa_drift_vs_score.png){#fig:drift-score width=85%}

![Exploratory: nominal test exceedance rate of the $1-10^{-4}$ out-of-fold calibration quantile, relative to the training-half cross-validation rate (Mission 1, every fourth channel).](figures/esa_calibration_shift.png){#fig:shift width=85%}

**Event classes.** [@fig:classes] breaks detection down by event class.

- No arm detected any of the {{evcls.m1.unseen.Anomaly.n}} Mission 1 test anomalies whose class never occurs in training.
- For seen-class anomalies, recall was {{evcls.m1.seen.Anomaly.A1:.2f}} (AR(8)), {{evcls.m1.seen.Anomaly.A2:.2f}} (trees) and {{evcls.m1.seen.Anomaly.A3:.2f}} (TCN).
- The trees' advantage over AR(8) came mainly from rare nominal events. They detected {{evcls.m1.seen.Rare_Event.A2:.2f}} of seen-class and {{evcls.m1.unseen.Rare_Event.A2:.2f}} of unseen-class rare events, against {{evcls.m1.seen.Rare_Event.A1:.2f}} and {{evcls.m1.unseen.Rare_Event.A1:.2f}} for A1. Of their roughly {{evcls.m1.gain_a2_a1.total:.0f}} additional detections per seed, about {{evcls.m1.gain_a2_a1.seen.Rare_Event:.0f}} are seen-class rare events, {{evcls.m1.gain_a2_a1.unseen.Rare_Event:.0f}} unseen-class rare events and {{evcls.m1.gain_a2_a1.seen.Anomaly:.1f}} anomalies.
- The unseen-class rare events are not detected through learned classes. The unsupervised A0 detects the same share of them ({{evcls.m1.unseen.Rare_Event.A0:.3f}}) as the trees, while A1 detects none. They are large deviations that a lenient layer catches and A1's strict layer misses. The trees combine recall of this kind with a better false-alarm record.
- On anomalies alone, the trees' lead over A1 disappears ([@fig:primary-anom]).

![Primary $F_{0.5}$ (anomalies and rare events) against anomaly-only $F_{0.5}$ for the pre-registered arms (conformal-BH layer, seed means). The labels give primary / anomaly-only. Mission 2's anomaly-only values rest on six anomalies.](figures/esa_primary_vs_anomaly.png){#fig:primary-anom width=100%}

![Exploratory: share of Mission 1 test events detected, by category and by whether the event class occurs in the training half (seed means, conformal-BH layer).](figures/esa_event_classes.png){#fig:classes width=85%}

**Ablations of A3.** Compared with the full A3 (three seeds each):

- dropping the residual inputs changed the score by {{xcomp.m1.x4.diff:+.3f}} [{{xcomp.m1.x4.ci95_low:+.3f}}, {{xcomp.m1.x4.ci95_high:+.3f}}];
- using two classes changed it by {{xcomp.m1.x5.diff:+.3f}} [{{xcomp.m1.x5.ci95_low:+.3f}}, {{xcomp.m1.x5.ci95_high:+.3f}}].

## ESA-ADB: post-hoc validation analyses (P1–P8) {#sec:results-posthoc}

After the confirmatory analysis, a second plan (`research_design/preregistration/POSTHOC_PLAN_V2.md`) fixed eight further analyses, their code checks and their interpretation rules. It was frozen by hash ({{ph.plan_sha256}}, {{ph.plan_written}}) before any of them was run. All eight use test halves that had been scored before, so every result in this section is exploratory. What the plan controls is which analyses were run and how they are read.

**P3, exact bootstrap.** The frozen bootstrap used an approximate block-wise metric. P3 replaces it with an exact decomposition that accumulates, per 30-day block, detected and total events, false-positive runs, nominal time and alarmed nominal time (`scripts/posthoc_v2/exact_metric.py`):

- it matched the official code on {{ph.p3.n_checks}} stored arm–metric pairs, with a largest difference of {{ph.p3.max_abs_diff:.1e}};
- with it, the H6 interval is [{{ph.p3.m1.h6.ci95_low:+.3f}}, {{ph.p3.m1.h6.ci95_high:+.3f}}] and H5's Holm-adjusted p-value {{ph.p3.m1.h5.p_holm:.3f}};
- all three confirmatory verdicts are unchanged.

**P8, provenance.** Re-running the frozen selection for A1 reproduced the saved choice and every grid value on both missions (largest difference {{ph.p8.m1.max_abs_cv_diff:g}}).

**P1, decomposition of H6.** Ten cells, each a full protocol scored once, switch the five differences between A0 and A1 one at a time, in the order G, F, σ, P, D (forward) and in the reverse order. In all six settings the first and last cells reproduced A0's and A1's official values exactly. [@fig:ph-decomp] shows four of the settings.

- Mission 1, BH: the selection data carry the contrast, {{ph.p1.m1.target.conformal_bh.forward.D.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.D.ci95_high:+.3f}}] forward and {{ph.p1.m1.target.conformal_bh.reverse.D.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.D.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.D.ci95_high:+.3f}}] reverse, the only factor the rule calls material. The pool added {{ph.p1.m1.target.conformal_bh.forward.P.delta_official:+.3f}} on both paths and the fit data {{ph.p1.m1.target.conformal_bh.forward.F.delta_official:+.3f}}.
- The residual scale had no effect in any setting. Both layers compare a test score with nominal scores of the same channel, so a per-channel rescaling shared by pool and test cancels.
- The grid was order-dependent: under whole-half selection the extended grid allowed a stricter level and cost {{ph.p1.m1.target.conformal_bh.reverse.G.delta_official:.3f}}; with the original grid, whole-half selection would have scored {{ph.p1.m1.target.conformal_bh.r1.f05:.3f}}.
- Mission 1, threshold: pool and selection data were both material ({{ph.p1.m1.target.threshold.forward.P.delta_official:+.3f}} and {{ph.p1.m1.target.threshold.forward.D.delta_official:+.3f}}).
- Mission 1, lightweight channels: the reversal comes from the same two factors, the pool ({{ph.p1.m1.lightweight.conformal_bh.forward.P.delta_official:.3f}} and {{ph.p1.m1.lightweight.conformal_bh.reverse.P.delta_official:.3f}}) and the selection data ({{ph.p1.m1.lightweight.conformal_bh.forward.D.delta_official:.3f}}).
- Mission 2, BH: the reversal also comes from the selection data, {{ph.p1.m2.target.conformal_bh.forward.D.delta_official:.3f}} [{{ph.p1.m2.target.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.D.ci95_high:+.3f}}] on both paths; every other factor changed the score by less than 0.005.

Neither of the two differences that change the scores (fit data and residual scale) explains the gain. The H6 contrast is, in effect, a contrast of selection data.

![Post hoc (P1): change in test-half event-wise $F_{0.5}$ when one of the five differences between A0 and A1 is switched, forward (A0 to A1) and reverse (A1 to A0), with exact block-bootstrap 95 % intervals. Dotted line: minimum effect of interest. Source: `posthoc_v2/p1_decompose/*.json`.](figures/esa_posthoc_decomposition.png){#fig:ph-decomp width=100%}

**P2, forward chaining.** Folds inside the training halves (Mission 1: the years 2003–2006; Mission 2: three quarters of 2001) compare a window protocol W, which mirrors A0, with a history protocol H, which mirrors A1, using only data before each fold ([@fig:ph-forward]).

- Mission 1: mean H − W {{ph.p2.m1.conformal_bh.mean_diff:+.3f}}, but H won {{ph.p2.m1.conformal_bh.n_folds_H_better}} of {{ph.p2.m1.conformal_bh.n_folds}} folds, so by the pre-set rule (at least 3 of 4) the gain does not replicate. The mean comes from 2005, when W raised {{ph.p2.m1.conformal_bh.fold2.W_fa_yr:,.0f}} false alarms per year. In 2004 H raised no false alarm but also detected no event.
- Mission 2: H won all {{ph.p2.m2.conformal_bh.n_folds}} folds, by a mean of {{ph.p2.m2.conformal_bh.mean_diff:+.3f}}, below the minimum effect.
- Each Mission 1 fold holds 9–10 events, so single detections move a fold's score a long way.

![Post hoc (P2): forward-chaining folds, window protocol W against history protocol H (conformal-BH layer). Labels: fold $F_{0.5}$ and false alarms per year. Source: `posthoc_v2/p2_forward/m*_summary.json`.](figures/esa_posthoc_forward.png){#fig:ph-forward width=100%}

**P4, event-tuned ACI.** With $(\alpha, \gamma, k)$ selected on the event objective, ACI on A1 (Mission 1) chose $\gamma$ = {{ph.p4.m1.ar8.gamma_positive.gamma:g}} and scored {{ph.p4.m1.ar8.gamma_positive.f05:.3f}}, above BH ({{pre.m1.A1.conformal_bh.f05.mean:.3f}}). Over the {{ph.p4.rule.n_arms}} arms, ACI with $\gamma > 0$ reached or exceeded the threshold layer on {{ph.p4.rule.n_at_or_above}} and stayed at least 0.05 below it on {{ph.p4.rule.n_below}}. The plan's rule therefore does not allow the claim that ACI is unusable at the event level. The pre-registered collapse ([@sec:results-ablations]) came from tuning $\gamma$ to a point-wise rate.

**P5, the registered anomaly-score arm.** Selected and scored on anomalies only (source: `posthoc_v2/p5_anomaly_arm/m*_summary.json`):

- Mission 1: AR(8) {{ph.p5.m1.A1.mean:.3f}}; the trees' anomaly log-odds {{ph.p5.m1.A2.mean:.3f}} (range {{ph.p5.m1.A2.min:.3f}}–{{ph.p5.m1.A2.max:.3f}}, {{ph.p5.m1.A2.fa_yr:.1f}} false alarms per year); the TCN's {{ph.p5.m1.A3.mean:.3f}} ({{ph.p5.m1.A3.min:.3f}}–{{ph.p5.m1.A3.max:.3f}});
- Mission 2, with six test anomalies: AR(8) {{ph.p5.m2.A1.mean:.3f}}, trees {{ph.p5.m2.A2.mean:.3f}}, TCN {{ph.p5.m2.A3.mean:.3f}}.

Supervised anomaly scores therefore did not improve anomaly detection on Mission 1, where the training half holds {{data.esa.m1.events_all.train.Anomaly}} labelled anomalies.

**P6, seed extension.** Seeds 5–9 of A2 and A3 were trained with the frozen code (source: `posthoc_v2/p6_seeds/summary.json`):

- trees: {{ph.p6.m1.A2.f05.mean:.3f}} ± {{ph.p6.m1.A2.f05.std:.3f}} over {{ph.p6.m1.A2.f05.n}} seeds (range {{ph.p6.m1.A2.f05.min:.3f}}–{{ph.p6.m1.A2.f05.max:.3f}});
- TCN: {{ph.p6.m1.A3.f05.mean:.3f}} ± {{ph.p6.m1.A3.f05.std:.3f}} over {{ph.p6.m1.A3.f05.n}} seeds (range {{ph.p6.m1.A3.f05.min:.3f}}–{{ph.p6.m1.A3.f05.max:.3f}});
- with all seeds (exact bootstrap, unadjusted): TCN − AR(8) {{ph.p6.m1.h4.diff:+.3f}} [{{ph.p6.m1.h4.ci95_low:+.3f}}, {{ph.p6.m1.h4.ci95_high:+.3f}}]; TCN − trees {{ph.p6.m1.h5.diff:+.3f}} [{{ph.p6.m1.h5.ci95_low:+.3f}}, {{ph.p6.m1.h5.ci95_high:+.3f}}]; trees − AR(8) {{ph.p6.m1.x1.diff:+.3f}} [{{ph.p6.m1.x1.ci95_low:+.3f}}, {{ph.p6.m1.x1.ci95_high:+.3f}}].

The confirmatory five-seed verdicts are unchanged; the extension narrows the intervals around the same conclusion.

**P7, operational metrics.** On Mission 1, A1 detected {{ph.p7.m1.A1.detected:.0f}} of {{ph.p7.m1.A1.n_events:.0f}} target events with a median delay of {{ph.p7.m1.A1.delay_h_median:.2f}} h, and raised {{ph.p7.m1.A1.false_alarms_per_1000_nominal_h:.2f}} false alarms per 1000 nominal hours. A0 detected {{ph.p7.m1.A0.detected:.0f}} at {{ph.p7.m1.A0.false_alarms_per_1000_nominal_h:.1f}} per 1000 h. The TCN's runs were the longest (median {{ph.p7.m1.A3.run_duration_h_median:.1f}} h) and its detections the latest (median {{ph.p7.m1.A3.delay_h_median:.1f}} h).


## Supporting studies {#sec:results-support}

### OPSSAT-AD (real)

[@tbl:opssat] and [@fig:opssat] give the OPSSAT-AD results.

: OPSSAT-AD test results (mean ± sd over 5 seeds where applicable; threshold 0.5). {#tbl:opssat}

| Model | F1 (official) | AUC-PR (official) | F1 (time-blocked) | AUC-PR (time-blocked) |
|---|---|---|---|---|
| hybrid CNN + features | {{ops.official.hybrid.f100.f1.mean:.3f}} ± {{ops.official.hybrid.f100.f1.std:.3f}} | {{ops.official.hybrid.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.hybrid.f100.f1.mean:.3f}} | {{ops.time_blocked.hybrid.f100.auc_pr.mean:.3f}} |
| feature MLP (FCNN) | {{ops.official.fcnn.f100.f1.mean:.3f}} ± {{ops.official.fcnn.f100.f1.std:.3f}} | {{ops.official.fcnn.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.fcnn.f100.f1.mean:.3f}} | {{ops.time_blocked.fcnn.f100.auc_pr.mean:.3f}} |
| 1-D CNN | {{ops.official.cnn.f100.f1.mean:.3f}} ± {{ops.official.cnn.f100.f1.std:.3f}} | {{ops.official.cnn.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.cnn.f100.f1.mean:.3f}} | {{ops.time_blocked.cnn.f100.auc_pr.mean:.3f}} |
| patch transformer | {{ops.official.transformer.f100.f1.mean:.3f}} | {{ops.official.transformer.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.transformer.f100.f1.mean:.3f}} | {{ops.time_blocked.transformer.f100.auc_pr.mean:.3f}} |
| XGBoost | {{ops.official.xgboost.f100.f1.mean:.3f}} | {{ops.official.xgboost.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.xgboost.f100.f1.mean:.3f}} | {{ops.time_blocked.xgboost.f100.auc_pr.mean:.3f}} |
| random forest | {{ops.official.random_forest.f100.f1.mean:.3f}} | {{ops.official.random_forest.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.random_forest.f100.f1.mean:.3f}} | {{ops.time_blocked.random_forest.f100.auc_pr.mean:.3f}} |
| AdaBoost | {{ops.official.adaboost.f100.f1.mean:.3f}} | {{ops.official.adaboost.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.adaboost.f100.f1.mean:.3f}} | {{ops.time_blocked.adaboost.f100.auc_pr.mean:.3f}} |
| logistic regression | {{ops.official.logreg.f100.f1.mean:.3f}} | {{ops.official.logreg.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.logreg.f100.f1.mean:.3f}} | {{ops.time_blocked.logreg.f100.auc_pr.mean:.3f}} |
| Chronos-Bolt probe | {{ops.official.chronos.f100.f1.mean:.3f}} | {{ops.official.chronos.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.chronos.f100.f1.mean:.3f}} | {{ops.time_blocked.chronos.f100.auc_pr.mean:.3f}} |
| COSMOS robust-z | {{ops.official.cosmos_robust_z.f100.f1.mean:.3f}} | {{ops.official.cosmos_robust_z.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.cosmos_robust_z.f100.f1.mean:.3f}} | {{ops.time_blocked.cosmos_robust_z.f100.auc_pr.mean:.3f}} |
| Isolation Forest | {{ops.official.iforest.f100.f1.mean:.3f}} | {{ops.official.iforest.f100.auc_pr.mean:.3f}} | {{ops.time_blocked.iforest.f100.f1.mean:.3f}} | {{ops.time_blocked.iforest.f100.auc_pr.mean:.3f}} |
| published FCNN [@ruszczak2025opssat] | {{lit.opssat.fcnn_f1:.3f}} | {{lit.opssat.fcnn_aucpr:.3f}} | — | — |

![OPSSAT-AD F1 at threshold 0.5 under the official and time-blocked splits (5 seeds).](figures/opssat_f1.png){#fig:opssat width=100%}

**Main comparisons.**

- The re-implemented FCNN reproduces the published model: F1 {{ops.official.fcnn.f100.f1.mean:.3f}} against {{lit.opssat.fcnn_f1:.3f}} published. AdaBoost reaches {{ops.official.adaboost.f100.f1.mean:.3f}} against {{lit.opssat.adaboost_f1:.3f}} published.
- The hybrid network has the highest F1 on the official split; the 1-D CNN has a marginally higher AUC-PR ({{ops.official.cnn.f100.auc_pr.mean:.3f}} against {{ops.official.hybrid.f100.auc_pr.mean:.3f}}). In a paired bootstrap on seed-averaged scores the hybrid exceeds the FCNN in AUC-PR by {{ops.paired.hybrid_fcnn.aucpr.diff:+.4f}} [{{ops.paired.hybrid_fcnn.aucpr.ci_low:+.4f}}, {{ops.paired.hybrid_fcnn.aucpr.ci_high:+.4f}}]. In F1 the difference is not significant ({{ops.paired.hybrid_fcnn.f1.diff:+.3f}} [{{ops.paired.hybrid_fcnn.f1.ci_low:+.3f}}, {{ops.paired.hybrid_fcnn.f1.ci_high:+.3f}}]).
- The hybrid exceeds XGBoost by {{ops.paired.hybrid_xgboost.aucpr.diff:+.3f}} in AUC-PR [{{ops.paired.hybrid_xgboost.aucpr.ci_low:+.3f}}, {{ops.paired.hybrid_xgboost.aucpr.ci_high:+.3f}}].
- On the {{data.opssat.n_features}} tabular features alone, the feature MLP beats XGBoost: F1 {{ops.paired.fcnn_xgboost.f1.diff:+.3f}} [{{ops.paired.fcnn_xgboost.f1.ci_low:+.3f}}, {{ops.paired.fcnn_xgboost.f1.ci_high:+.3f}}] and AUC-PR {{ops.paired.fcnn_xgboost.aucpr.diff:+.3f}} [{{ops.paired.fcnn_xgboost.aucpr.ci_low:+.3f}}, {{ops.paired.fcnn_xgboost.aucpr.ci_high:+.3f}}]. On this dataset, trees do not dominate tabular data.

**Time-blocked split.** Under this split the deep models drop and the ranking compresses: the hybrid scores {{ops.time_blocked.hybrid.f100.f1.mean:.3f}}, the FCNN {{ops.time_blocked.fcnn.f100.f1.mean:.3f}} and XGBoost {{ops.time_blocked.xgboost.f100.f1.mean:.3f}}. Not every model drops: AdaBoost rises from {{ops.official.adaboost.f100.f1.mean:.3f}} to {{ops.time_blocked.adaboost.f100.f1.mean:.3f}} and logistic regression from {{ops.official.logreg.f100.f1.mean:.3f}} to {{ops.time_blocked.logreg.f100.f1.mean:.3f}}. The official split's advantage for the deep models is partly an advantage of interleaved training and test fragments.

**Label efficiency.** At 10 % of the training labels, the AUC-PR values are:

- FCNN {{ops.official.fcnn.f10.auc_pr.mean:.3f}};
- CNN {{ops.official.cnn.f10.auc_pr.mean:.3f}};
- hybrid {{ops.official.hybrid.f10.auc_pr.mean:.3f}};
- XGBoost {{ops.official.xgboost.f10.auc_pr.mean:.3f}};
- Chronos-Bolt probe {{ops.official.chronos.f10.auc_pr.mean:.3f}}.

[@fig:opssat-curves] shows the label-efficiency curves together with the robustness results described next.

![OPSSAT-AD label efficiency and robustness. Left: test AUC-PR against the share of training labels (official split; note the y-axis starts at 0.8). Centre and right: test F1 under added noise and under sample dropout; the 0 level is the unperturbed run of the robustness study. Means over 5 seeds. Sources: `summary_official.csv`, `robustness_summary.csv`.](figures/opssat_curves.png){#fig:opssat-curves width=100%}

**Robustness.** The models were perturbed at test time with noise (σ as a fraction of each fragment's standard deviation) and with sample dropout:

- **Noise at 0.1σ.** F1 fell to {{ops.robust.noise.0p1.hybrid.f1:.3f}} for the hybrid, {{ops.robust.noise.0p1.fcnn.f1:.3f}} for the FCNN, {{ops.robust.noise.0p1.xgboost.f1:.3f}} for XGBoost and {{ops.robust.noise.0p1.cnn.f1:.3f}} for the CNN.
- **Larger noise.** At 0.25σ and above, all models fell to about 0.3–0.4.
- **30 % sample dropout.** XGBoost was the most robust ({{ops.robust.dropout.0p3.xgboost.f1:.3f}}); the deep models dropped to {{ops.robust.dropout.0p3.cnn.f1:.3f}}–{{ops.robust.dropout.0p3.hybrid.f1:.3f}}.

### Learned health monitor (SIMULATED)

The learned monitor was compared with the classical monitor on the held-out seed sets {{mon.heldout.learned_conformal.seed_sets}}, using simulated telemetry:

| Monitor | F1 | Precision | Recall | Localisation accuracy | False alarms (fault-free runs) |
|---|---|---|---|---|---|
| classical EWMA | {{mon.heldout.classical_ewma.f1.mean:.3f}} ± {{mon.heldout.classical_ewma.f1.std:.3f}} | {{mon.heldout.classical_ewma.precision.mean:.3f}} | {{mon.heldout.classical_ewma.recall.mean:.3f}} | {{mon.heldout.classical_ewma.localisation_accuracy.mean:.3f}} | {{mon.heldout.classical_ewma.false_alarms_total}} |
| learned (PatchTST + conformal) | {{mon.heldout.learned_conformal.f1.mean:.3f}} ± {{mon.heldout.learned_conformal.f1.std:.3f}} | {{mon.heldout.learned_conformal.precision.mean:.3f}} | {{mon.heldout.learned_conformal.recall.mean:.3f}} | {{mon.heldout.learned_conformal.localisation_accuracy.mean:.3f}} | {{mon.heldout.learned_conformal.false_alarms_total}} |

The learned monitor localises the faulted subsystem more often. It detects fewer faults and raises more false alarms. Both monitors would cap their healthy window at the true fault onset if the onset came before the end of the window. With the simulator's default onset at 55 % of the record and the window at 40 %, the cap never binds, so the results do not use ground truth (Appendix C).

### SpotGEO (real) and synthetic scenes (SYNTHETIC)

[@fig:vision] and [@tbl:spotgeo] give the vision results.

![Vision results on SpotGEO and on the synthetic scenes. Left: official SpotGEO F1 on the {{data.spotgeo.test.sequences:int}} test sequences. Right: COCO mAP@50 on the synthetic test scenes (SYNTHETIC).](figures/vision_results.png){#fig:vision width=100%}

: SpotGEO results on the full test set with the official scorer (mean ± sd over seeds where several exist). The challenge winner is from [@chen2021spotgeo]. {#tbl:spotgeo}

| Model | Inputs | Training data | F1 | Precision | Recall | Official MSE (sum) |
|---|---|---|---|---|---|---|
| COSMOS front end + DoG (classical) | differences | — | {{spot.classical.diff.f100.test_f1.mean:.3f}} | {{spot.classical.diff.f100.test_precision.mean:.3f}} | {{spot.classical.diff.f100.test_recall.mean:.3f}} | — |
| heat-map U-Net | both | 100 % | {{spot.unet.both.f100.test_f1.mean:.3f}} ± {{spot.unet.both.f100.test_f1.std:.3f}} | {{spot.unet.both.f100.test_precision.mean:.3f}} | {{spot.unet.both.f100.test_recall.mean:.3f}} | {{spot.run.unet.both.f100.mse_sum.mean:int}} |
| heat-map U-Net | high-pass only | 100 % | {{spot.unet.raw.f100.test_f1.mean:.3f}} | {{spot.unet.raw.f100.test_precision.mean:.3f}} | {{spot.unet.raw.f100.test_recall.mean:.3f}} | {{spot.run.unet.raw.f100.mse_sum.mean:int}} |
| heat-map U-Net | differences only | 100 % | {{spot.unet.diff.f100.test_f1.mean:.3f}} | {{spot.unet.diff.f100.test_precision.mean:.3f}} | {{spot.unet.diff.f100.test_recall.mean:.3f}} | {{spot.run.unet.diff.f100.mse_sum.mean:int}} |
| heat-map U-Net | both | 25 % | {{spot.unet.both.f25.test_f1.mean:.3f}} | {{spot.unet.both.f25.test_precision.mean:.3f}} | {{spot.unet.both.f25.test_recall.mean:.3f}} | {{spot.run.unet.both.f25.mse_sum.mean:int}} |
| heat-map U-Net | both | 10 % | {{spot.unet.both.f10.test_f1.mean:.3f}} | {{spot.unet.both.f10.test_precision.mean:.3f}} | {{spot.unet.both.f10.test_recall.mean:.3f}} | {{spot.run.unet.both.f10.mse_sum.mean:int}} |
| ResNet-18 encoder (pretrained) | both | 100 % | {{spot.resnet18_pretrained.both.f100.test_f1.mean:.3f}} ± {{spot.resnet18_pretrained.both.f100.test_f1.std:.3f}} | {{spot.resnet18_pretrained.both.f100.test_precision.mean:.3f}} | {{spot.resnet18_pretrained.both.f100.test_recall.mean:.3f}} | {{spot.run.resnet18_pretrained.both.f100.mse_sum.mean:int}} |
| challenge winner | — | — | {{lit.spotgeo.winner_f1:.3f}} | — | — | {{lit.spotgeo.winner_mse:.0f}} |

**SpotGEO findings.**

- The learned detectors improve on the classical COSMOS front end by an order of magnitude in F1. They remain well short of the challenge winner, whose solution added temporal post-processing [@chen2021spotgeo].
- The "differences only" input is the weakest learned variant, and the high-pass-only input comes close to the combined one. This suggests that, once all five frames are given to the network, the COSMOS registration-and-differencing front end adds little. The input ablations were run with one seed each, so the support is weak.
- With 10 % or 25 % of the training sequences, F1 falls to {{spot.unet.both.f10.test_f1.mean:.3f}} and {{spot.unet.both.f25.test_f1.mean:.3f}}.

**Synthetic scenes.** On the synthetic scenes, which carry no evidence about real sensors, the COCO-style scores (mAP@50 / mAP@50–95) are:

- heat-map U-Net: {{syn.heatmap.map50.mean:.3f}} / {{syn.heatmap.map50_95.mean:.3f}};
- YOLOv8n: {{syn.yolov8n.map50.mean:.3f}} / {{syn.yolov8n.map50_95.mean:.3f}};
- existing TinyYOLO: {{syn.tinyyolo.map50.mean:.3f}} / {{syn.tinyyolo.map50_95.mean:.3f}}.

### Re-entry landmark (real catalogue)

[@tbl:landmark] and [@fig:landmark] give the landmark results.

: Re-entry landmark on the test launch groups ({{lm.split.test.n:int}} objects, {{lm.split.test.events}} events). {#tbl:landmark}

| Model | AUC-PR 30 d | AUC-PR 68 d | AUC-ROC 68 d | C-index | Brier 68 d | ECE 68 d |
|---|---|---|---|---|---|---|
| XGBoost | {{lm.xgboost.all.h30.auc_pr:.3f}} | {{lm.xgboost.all.h68.auc_pr:.3f}} | {{lm.xgboost.all.h68.auc_roc:.3f}} | {{lm.xgboost.all.c_index:.3f}} | {{lm.xgboost.all.h68.brier:.4f}} | {{lm.xgboost.all.h68.ece:.4f}} |
| XGBoost-AFT$^\dagger$ | {{lm.xgboost_aft.all.h30.auc_pr:.3f}} | {{lm.xgboost_aft.all.h68.auc_pr:.3f}} | {{lm.xgboost_aft.all.h68.auc_roc:.3f}} | {{lm.xgboost_aft.all.c_index:.3f}} | {{lm.xgboost_aft.all.h68.brier:.4f}} | {{lm.xgboost_aft.all.h68.ece:.4f}} |
| hazard MLP ensemble | {{lm.hazard_mlp.all.h30.auc_pr:.3f}} | {{lm.hazard_mlp.all.h68.auc_pr:.3f}} | {{lm.hazard_mlp.all.h68.auc_roc:.3f}} | {{lm.hazard_mlp.all.c_index:.3f}} | {{lm.hazard_mlp.all.h68.brier:.4f}} | {{lm.hazard_mlp.all.h68.ece:.4f}} |
| hazard MLP without physics feature | {{lm.hazard_mlp_nophys.all.h30.auc_pr:.3f}} | {{lm.hazard_mlp_nophys.all.h68.auc_pr:.3f}} | {{lm.hazard_mlp_nophys.all.h68.auc_roc:.3f}} | {{lm.hazard_mlp_nophys.all.c_index:.3f}} | {{lm.hazard_mlp_nophys.all.h68.brier:.4f}} | {{lm.hazard_mlp_nophys.all.h68.ece:.4f}} |
| logistic regression | {{lm.logreg.all.h30.auc_pr:.3f}} | {{lm.logreg.all.h68.auc_pr:.3f}} | {{lm.logreg.all.h68.auc_roc:.3f}} | {{lm.logreg.all.c_index:.3f}} | {{lm.logreg.all.h68.brier:.4f}} | {{lm.logreg.all.h68.ece:.4f}} |
| COSMOS physics lifetime | {{lm.physics.all.h30.auc_pr:.3f}} | {{lm.physics.all.h68.auc_pr:.3f}} | {{lm.physics.all.h68.auc_roc:.3f}} | {{lm.physics.all.c_index:.3f}} | {{lm.physics.all.h68.brier:.4f}} | {{lm.physics.all.h68.ece:.4f}} |

$^\dagger$ The AFT predictions use all boosting rounds rather than the early-stopped iteration, because of a defect (Appendix C). The AFT row therefore does not represent the intended configuration.

![Re-entry landmark: AUC-PR for re-entry within 30 and 68 days on the test launch groups.](figures/reentry_landmark.png){#fig:landmark width=100%}

**Model comparisons.**

- Trees outperform the neural discrete-time hazard model. At 68 days the difference is {{lm.cmp.hazard_mlp_vs_xgboost.h68.diff:+.3f}} [{{lm.cmp.hazard_mlp_vs_xgboost.h68.ci_low:+.3f}}, {{lm.cmp.hazard_mlp_vs_xgboost.h68.ci_high:+.3f}}], from a launch-group bootstrap.
- On the full test set, the learned models greatly outperform the physics lifetime: XGBoost exceeds physics by {{lm.cmp.xgboost_vs_physics.h68.diff:+.3f}} [{{lm.cmp.xgboost_vs_physics.h68.ci_low:+.3f}}, {{lm.cmp.xgboost_vs_physics.h68.ci_high:+.3f}}]. Most re-entries in the population are payloads ({{data.landmark.events_type.PAY}} of {{data.landmark.events}}), many of them controlled, which a drag-only lifetime cannot represent.
- The physics feature adds a little to the MLP: {{lm.cmp.hazard_mlp_vs_hazard_mlp_nophys.h68.diff:+.3f}} [{{lm.cmp.hazard_mlp_vs_hazard_mlp_nophys.h68.ci_low:+.3f}}, {{lm.cmp.hazard_mlp_vs_hazard_mlp_nophys.h68.ci_high:+.3f}}].
- On the *uncontrolled* subset ({{lm.physics.uncontrolled.n:int}} objects, {{lm.physics.uncontrolled.events}} events), the order changes at the short horizon. At 30 days physics has the highest AUC-PR ({{lm.physics.uncontrolled.h30.auc_pr:.3f}}), against {{lm.logreg.uncontrolled.h30.auc_pr:.3f}} for logistic regression, {{lm.xgboost.uncontrolled.h30.auc_pr:.3f}} for XGBoost and {{lm.hazard_mlp.uncontrolled.h30.auc_pr:.3f}} for the MLP. At 68 days XGBoost ({{lm.xgboost.uncontrolled.h68.auc_pr:.3f}}) is ahead of physics ({{lm.physics.uncontrolled.h68.auc_pr:.3f}}), and the defective AFT model is highest ({{lm.xgboost_aft.uncontrolled.h68.auc_pr:.3f}}). The 30-day comparison rests on {{lm.physics.uncontrolled.h30.positives}} positive objects, so it is not conclusive. It suggests that the learned models' advantage comes mainly from objects whose re-entry is not governed by drag ([@fig:landmark-subsets]).

![Re-entry landmark: AUC-PR on all test objects and on the uncontrolled subset, at 30 and 68 days. The positives in each set are given in the panel titles; with four positives, the 30-day uncontrolled values are highly uncertain. Source: `reentry_landmark/results.json`.](figures/reentry_subsets.png){#fig:landmark-subsets width=100%}

- The AUC-ROC and C-index values are close to 1 for every model. Most objects are far from re-entry, so these measures say little. AUC-PR is the informative measure here.

**Conformal sets at 68 days ($\alpha$ = 0.1).**

- *Marginal sets* reach the nominal overall coverage (XGBoost {{lm.conf.xgboost.marginal.coverage:.3f}}). They cover the events themselves only {{lm.conf.xgboost.marginal.coverage_on_events:.3f}} of the time (MLP: {{lm.conf.hazard_mlp.marginal.coverage_on_events:.3f}}), because events are rare.
- *Label-conditional (Mondrian) sets* cover events with probability {{lm.conf.xgboost.mondrian.coverage_events:.3f}} and non-events with probability {{lm.conf.xgboost.mondrian.coverage_nonevents:.3f}} for XGBoost.

### Propagator surrogate (SIMULATED)

All errors in this section are measured against the COSMOS propagator, not against real re-entries.

**In-range test set.** On {{sur.test.n:int}} test cases the surrogate ensemble reproduced the reference propagator with:

- median impact-position error {{sur.test.position_error_km_median:.1f}} km (95th percentile {{sur.test.position_error_km_p95:.1f}} km);
- median relative downrange error {{sur.test.downrange_relative_error_median:pct1}};
- fall-time mean absolute error {{sur.test.fall_time_mae_s:.1f}} s;
- impact classification accuracy {{sur.test.impact_classification_accuracy:.3f}}.

Its throughput was {{sur.test.surrogate_samples_per_s:int}} samples per second.

**Out-of-range test set.** With break-up altitudes of 100–120 km, the median error rose to {{sur.ood_test.position_error_km_median:.1f}} km and the 95th percentile to {{sur.ood_test.position_error_km_p95:.0f}} km. The ensemble spread separates out-of-range from in-range cases with AUROC {{sur.ood_auroc:.3f}}.

**Footprints.** Over {{sur.fp.n}} simulated footprint scenarios:

- the median relative error of CEP50 was {{sur.fp.cep50_relerr_median:pct1}} (maximum {{sur.fp.cep50_relerr_max:pct1}});
- the surrogate's 95 % conformal zone covered a median {{sur.fp.coverage_median:.3f}} (minimum {{sur.fp.coverage_min:.3f}}) of the reference impacts;
- the median speed-up was {{sur.fp.speedup_median:.0f}}×.

These scenarios clip their inputs into the training ranges, so they do not test the fallback.

**Inside the pipeline.** Mean over {{sur.mode.n_seeds}} seeds:

| Fallback mode | Seconds per footprint | CEP50 (km) |
|---|---|---|
| reference only | {{sur.mode.ref.seconds_mean:.2f}} | {{sur.mode.ref.cep50_mean:.0f}} |
| `none` (surrogate only) | {{sur.mode.none.seconds_mean:.2f}} | {{sur.mode.none.cep50_mean:.0f}} |
| `range` | {{sur.mode.range.seconds_mean:.2f}} | {{sur.mode.range.cep50_mean:.0f}} |
| `spread` | {{sur.mode.spread.seconds_mean:.2f}} | {{sur.mode.spread.cep50_mean:.0f}} |

In `range` mode the surrogate handles {{sur.mode.range.used_mean:pct1}} of samples. The remainder still dominate the run time, so any fallback gives up most of the speed-up.
