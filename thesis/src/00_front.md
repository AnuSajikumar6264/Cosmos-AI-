---
title: "COSMOS: An Evidence-Graded Machine-Learning Framework for Satellite Health Monitoring, Space-Debris Detection and Re-entry Risk"
subtitle: "With a Pre-registered Study of Alarm Selection on the ESA Anomaly Detection Benchmark"
author: "Anupa Sajikumar"
lang: en-GB
---

::: {custom-style="Subtitle"}
Master's thesis, Project Phase. Submitted in partial fulfilment of the requirements for the degree of [degree title to be confirmed by the author].
:::

| | |
|---|---|
| Student | Anupa Sajikumar |
| Roll number | CB.SC.P2AIE25003 |
| Supervisor | Dr. Shanmuga Priya S |
| Institution | [institution, school and department to be confirmed by the author] |
| Industry placement | Intel India Private Ltd., AI Software Engineer Intern (11 months) |
| Date | October 2026 |

*The student, roll-number, supervisor and placement details are taken from the author's Project Phase Review-1 presentation in the project folder. The degree title and institution must be completed by the author before submission.*

\newpage

# Declaration {.unnumbered}

I declare that this thesis is my own work. It reports the software, experiments and results contained in the COSMOS project repository (`https://github.com/AnuSajikumar6264/Cosmos-AI-`, commit `2525367`, together with the result files listed in the evidence map in Appendix E). Where other work is used, it is cited. Datasets released by third parties are acknowledged in [@sec:datasets].

Signature: ____________________ Date: ____________________

# Acknowledgements {.unnumbered}

[To be written by the author.]

# Abstract {.unnumbered}

Satellite operators face three linked problems. They must recognise a failing spacecraft from its telemetry, detect and catalogue the debris that missions leave in orbit, and estimate when and where discarded objects re-enter the atmosphere. COSMOS is a Python framework that addresses all three. It is organised as an original physics-based pipeline and a machine-learning layer, COSMOS-AI, built on top of it. This thesis documents what the current implementation contains and evaluates each component on the data it can honestly be evaluated on. It reports the outcome of a pre-registered confirmatory study.

**The physics pipeline.** The pipeline comprises the following components:

- a 16-channel telemetry simulator with fault injection;
- a robust-statistics health monitor;
- a rule that couples the failing subsystem to break-up parameters;
- a simplified NASA-style break-up model;
- a US Standard Atmosphere 1976 model;
- a three-degree-of-freedom re-entry propagator;
- Monte Carlo footprints with split-conformal impact zones;
- a casualty-expectation estimate.

All {{tests.verify.total}} of its automated verification checks pass. {{verify.n_external}} compare with external references (published tables, closed forms, physical limits, known answers); {{verify.n_internal}} are self-consistency checks; and one is mis-specified. Several of its coupling outputs are not used downstream, and the debris feature is not connected to the end-to-end chain; both are documented.

**The COSMOS-AI layer.** The layer adds checksum-verified data acquisition and models evaluated on five real public sources: the ESA Anomalies Dataset, OPSSAT-AD, SpotGEO, CelesTrak catalogue snapshots and ESA's list of recovered debris. Simulator-only results are labelled as such.

**The pre-registered study.** The principal study uses the ESA Anomaly Detection Benchmark (ESA-ADB) under its official split, on the full sets of {{data.esa.m1.target_channels}} and {{data.esa.m2.target_channels}} target channels. Its primary metric is the corrected event-wise $F_{0.5}$.

An exploratory phase established three findings:

- Calibrating per-channel conformal p-values with a Benjamini–Hochberg (BH) step across channels gave {{explo.m2.full.ar8.conformal_bh.f05.mean:.3f}} on Mission 2.
- The best full-channel value published with the benchmark on Mission 2 is {{lit.dmlr.m2.full.best:.3f}}. However, a parameter-free out-of-limit rule already reaches {{explo.m2.full.ool.native.f05.mean:.3f}} there, so Mission 2 is close to saturated.
- On Mission 1 the same layer reached only {{explo.m1.full.ar8.conformal_bh.f05.mean:.3f}}. Its settings had been chosen on a nine-month window containing {{data.esa.m1.events_target.cal.events}} labelled events.

We then pre-registered three hypotheses and froze {{prereg.frozen_files}} code and protocol files by SHA-256 hash (freeze `{{prereg.freeze_id}}`). Each arm was scored once on the test half. The arms use out-of-fold scores over the whole training half, which holds {{sel.m1.n_train_events}} annotated events on Mission 1. Either an unsupervised AR(8) forecaster or a supervised event scorer, trained out of fold, feeds the same event-level conformal alarm layer.

**Confirmatory outcomes on Mission 1:**

- The protocol that fits AR(8) and selects the alarm layer on the whole training half raised AR(8) from {{pre.m1.A0.conformal_bh.f05.mean:.3f}} to {{pre.m1.A1.conformal_bh.f05.mean:.3f}}. The difference was {{conf.m1.h6.diff_official:+.3f}} (95 % CI [{{conf.m1.h6.ci95_low:+.3f}}, {{conf.m1.h6.ci95_high:+.3f}}], Holm-adjusted *p* ≤ {{conf.p_floor_holm:g}}). The arm raised {{pre.m1.A1.conformal_bh.fa_yr.mean:.1f}} false alarms and {{pre.m1.A1.conformal_bh.alarm_runs_per_year:.0f}} alarm runs in total per year, and it detected about one target event in six. The hypothesis was **supported**. The two arms differ in selection data, calibration pool, forecaster fitting period, residual scale and grid, so this is a comparison of two complete protocols. The gain was of similar size under a per-channel threshold layer ({{seleff.m1.full.threshold:+.3f}}).
- A causal temporal convolutional event scorer reached {{pre.m1.A3.conformal_bh.f05.mean:.3f}} ± {{pre.m1.A3.conformal_bh.f05.std:.3f}} over five seeds. Its point estimates were below the forecaster ({{conf.m1.h4.diff_official:+.3f}}) and below gradient-boosted trees on the same features ({{pre.m1.A2.conformal_bh.f05.mean:.3f}} ± {{pre.m1.A2.conformal_bh.f05.std:.3f}}; {{conf.m1.h5.diff_official:+.3f}}). Both hypotheses were **refuted**: neither improvement reached the pre-registered minimum effect.
- On the near-saturated Mission 2 replication, whole-half selection was worse than the calibration window under the BH layer ({{conf.m2.h6.diff_official:+.3f}}). Under a per-channel threshold layer it was better ({{xcomp.m2.a1_minus_a0.threshold:+.3f}}), so the reversal depends on the layer. On the lightweight Mission 1 channel subset, whole-half selection was worse under both layers.
- A point-level adaptive conformal comparator (ACI at $\alpha$ = {{cfg.prereg.aci_alpha}}) scored at most {{pre.all.aci.f05.max:.3f}} on either mission. With the selected adaptation rate, its update rule silences a channel for many steps after each exceedance, so persistent alarms are rare and adding event logic did not repair it.
- An exploratory diagnostic found that the supervised scorers' nominal score distributions shift between training and test. Nominal test samples exceeded the TCN's calibration quantile {{shift.m1.A3.ratio_min:.0f}}–{{shift.m1.A3.ratio_max:.0f}} times more often than in cross-validation, against {{shift.m1.a1_ar8.ratio:.1f}} times for AR(8). The shift is consistent with the deep scorer's weak result, but it does not explain the differences between seeds or models (rank correlation with test score {{shift.m1.supervised.spearman_f05:.2f}} over ten supervised runs).

**Post-hoc validation.** Eight further analyses, fixed and frozen before they were run (on the same test halves), qualify these results. A one-factor-at-a-time decomposition attributed the Mission 1 gain to the labels used to select the alarm settings ({{ph.p1.m1.target.conformal_bh.forward.D.delta_official:+.3f}} and {{ph.p1.m1.target.conformal_bh.reverse.D.delta_official:+.3f}} on the two paths), not to the fitted scores. In forward-chaining folds inside the training halves the protocol won only {{ph.p2.m1.conformal_bh.n_folds_H_better}} of {{ph.p2.m1.conformal_bh.n_folds}} Mission 1 folds, so the gain did not replicate by the pre-set rule. ACI tuned on the event metric reached {{ph.p4.m1.ar8.gamma_positive.f05:.3f}}, so its pre-registered collapse reflects the tuning target. More seeds did not change the H4 and H5 picture.

**Supporting studies.** These give the following results:

- *OPSSAT-AD:* a hybrid convolutional network reached F1 {{ops.official.hybrid.f100.f1.mean:.3f}} on the official split. The deep models' lead shrank under a time-blocked split.
- *SpotGEO:* a heat-map U-Net reached the official F1 {{spot.unet.both.f100.test_f1.mean:.3f}}.
- *Re-entry within 68 days, from a leakage-checked catalogue landmark:* gradient-boosted trees outperformed a neural discrete-time hazard model, with AUC-PR {{lm.xgboost.all.h68.auc_pr:.3f}} against {{lm.hazard_mlp.all.h68.auc_pr:.3f}}. On uncontrolled objects at 30 days, the physics lifetime model scored highest.

Under operator-grade metrics, the alarm layer and the data used to select it mattered more than model capacity, but the selection effect was conditional on the mission and channel set. The thesis closes with the threats to validity, a register of known defects, and a roadmap towards a journal publication.

**Keywords:** spacecraft telemetry; anomaly detection; conformal prediction; false discovery rate; ESA-ADB; pre-registration; temporal convolutional networks; space debris; re-entry prediction; digital twin.

\newpage

::: {custom-style="TOC Heading"}
Contents
:::

<!-- TOC -->

# List of Figures {.unnumbered}

<!-- LOF -->

# List of Tables {.unnumbered}

<!-- LOT -->

\newpage

# Abbreviations and Symbols {.unnumbered}

| Abbreviation | Meaning |
|---|---|
| ACI | adaptive conformal inference |
| ADTQC | anomaly detection timing quality curve (ESA-ADB timing metric) |
| AFF | affiliation-based metric |
| AFT | accelerated failure time |
| AR(8) | autoregressive model of order 8 |
| AUC-PR / AUC-ROC | area under the precision–recall / receiver-operating-characteristic curve |
| BH | Benjamini–Hochberg procedure |
| CEP50 | circular error probable (median radial distance) |
| CI | confidence interval |
| CV | cross-validation |
| DoG | difference of Gaussians |
| ESA-ADB | European Space Agency Anomaly Detection Benchmark |
| EW | event-wise |
| EWMA | exponentially weighted moving average |
| FDR | false discovery rate |
| GBT | gradient-boosted trees |
| GEO / LEO | geostationary / low Earth orbit |
| MESI | minimum effect size of interest |
| OOF | out of fold |
| OOL | out-of-limit (rule) |
| SATCAT | satellite catalogue (CelesTrak) |
| SBM | NASA Standard Breakup Model |
| TCN | temporal convolutional network |
| TNR | true-negative rate |
| US76 | US Standard Atmosphere 1976 |

| Symbol | Meaning |
|---|---|
| $z_{t,c}$ | robust-normalised value of channel $c$ at grid step $t$ |
| $s_{t,c}$ | non-conformity (anomaly) score |
| $p_{t,c}$ | conformal p-value |
| $q$ | Benjamini–Hochberg level |
| $k$ | persistence (minimum consecutive alarm steps) |
| $P_e, R_e$ | corrected event-wise precision and event-wise recall |
| $F_{0.5}$ | $F_\beta$ score with $\beta = 0.5$ |
| $\tau$ | landmark date (re-entry study) |
| $h$ | prediction horizon (days) |
| $B$ | ballistic coefficient $m/(C_D A)$ |
| $\rho$ | atmospheric density |
| $E_c$ | expected number of casualties |

\newpage
