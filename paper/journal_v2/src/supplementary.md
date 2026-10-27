---
title: "Supplementary Material"
subtitle: "Beyond model complexity: a pre-registered evaluation of event-level alarm protocols for spacecraft telemetry anomaly detection"
---

This file accompanies the manuscript. Every number is inserted from the study's evidence file (`thesis/evidence/evidence.json`), which records the result file each value comes from. Section S4 reports the validation analyses of the second frozen plan (`POSTHOC_PLAN_V2.md`, SHA-256 prefix `{{ph.plan_sha256}}`, frozen {{ph.plan_written}}). Abbreviations: A0–A3, study arms; BH, Benjamini–Hochberg; ACI, adaptive conformal inference; M1, M2, Missions 1 and 2; FA, false alarms.

# S1. Pre-registration, deviation log and validation plan

The following documents are provided as separate files in the repository and in the archived release:

- `research_design/preregistration/PREREG_TELEMETRY_V1.md`: the pre-registration, frozen as `{{prereg.freeze_id}}`, with {{prereg.frozen_files}} files hashed;
- `research_design/preregistration/DEVIATIONS.md`: every departure from the pre-registration, with its timing;
- `research_design/preregistration/freeze_09a98f5e160e.json` and `freeze.json`: the original freeze and the re-issued one (`{{prereg.refreeze_id}}`);
- `research_design/preregistration/POSTHOC_PLAN_V2.md` and `posthoc_v2_freeze.json`: the validation plan (P1–P8), frozen before any of its analyses was run.

# S2. Results reported on ESA-ADB

: **Table S1.** **Results reported on ESA-ADB.** Primary scores as published; the protocols differ, so the numbers are not comparable.

| Study | Protocol | Method | Reported score |
|---|---|---|---|
| @kotowski2026esaadb | official split; full and lightweight channels | TimeEval baselines, detector-specific thresholds | M1 full {{lit.dmlr.m1.full.best:.3f}}; M2 full {{lit.dmlr.m2.full.best:.3f}}; M1 lightweight {{lit.dmlr.m1.lw.best:.3f}} |
| @sahin2026tccad | official code; {{lit.tccad.m1_channels}} / {{lit.tccad.m2_channels}} channels; {{lit.tccad.seeds}} seeds; pre-registered | telecommand conditioning; split, sliding and adaptive conformal thresholds | conditioning negative; point-level ACI collapses the event score |
| @nguyen2026 | own preprocessing; uncorrected event-wise metrics | supervised CNN and graph networks against unsupervised detectors | not comparable |
| @brancato2027ast | Mission 1, hierarchical protocol (as stated in the abstract) | similarity search (SMED); quantile GP regression with retraining | M1 full {{lit.brancato.smed_full:.3f}} (SMED) |
| @alcarria2026 | monthly incremental retraining | LSTM autoencoder; PCA, k-means and LOF; adaptive thresholds | M1 {{lit.alcarria.m1_f05:.3f}}; M2 {{lit.alcarria.m2_f05:.3f}} |
| @allegrini2026 | Kaggle split of Mission 1, hidden test | supervised hierarchical ensemble | {{lit.allegrini.private:.3f}} (private leaderboard) |
| @heydari2026 | two missions, {{lit.heydari.shared_channels}} shared channels | threshold transfer under alert budgets | policy diagnostics |
| This study | official split; full target channels; pre-registered | AR(8), XGBoost and TCN under one event-level layer | M1 {{pre.m1.A1.conformal_bh.f05.mean:.3f}} (AR(8)), {{pre.m1.A2.conformal_bh.f05.mean:.3f}} (trees) |

::: {custom-style="Table Note"}
Note: GP, Gaussian process; LOF, local outlier factor; SMED, the similarity-search detector of Brancato et al.
:::

# S3. Models and pre-registered arms

: **Table S2.** **Model settings.** No setting was tuned on test data; the supervised scorers received no hyperparameter search.

| Model | Setting | Value |
|---|---|---|
| AR(8) | order; forecast horizon; maximum fitted rows | {{cfg.oof.order}}; {{cfg.oof.horizon}} steps; {{cfg.oof.max_fit:int}} |
| AR(8) | out-of-fold blocks; purge | 5 contiguous blocks; {{cfg.evm.purge_days:g}} d on either side |
| AR(8) | score | absolute forecast error / max(residual scale, 0.05) |
| XGBoost (A2) | trees; depth; learning rate | {{cfg.gbt.rounds}}; {{cfg.gbt.max_depth}}; {{cfg.gbt.eta}} |
| XGBoost (A2) | row and column subsampling; minimum child weight | {{cfg.gbt.subsample}}; {{cfg.gbt.colsample_bytree}}; {{cfg.gbt.min_child_weight}} |
| XGBoost (A2) | training rows; classes; class weights | {{cfg.gbt.n_rows:int}} (half labelled); {{cfg.gbt.num_class}}; {{cfg.evm.class_weights}} |
| TCN (A3) | levels; kernel; width; receptive field | {{cfg.evm.levels}}; {{cfg.evm.kernel}}; {{cfg.evm.d_model}}; {{evm.receptive_field}} steps |
| TCN (A3) | cross-channel mixing; parameters (M1) | every {{cfg.evm.mix_every}} levels; {{evm.m1.params:int}} |
| TCN (A3) | optimiser; steps; learning rate; weight decay; dropout | AdamW, one-cycle; {{cfg.evm.steps:int}}; {{cfg.evm.lr}}; {{cfg.evm.weight_decay}}; {{cfg.evm.dropout}} |
| TCN (A3) | batch; window; share of windows with a labelled sample | {{cfg.evm.batch}}; {{cfg.evm.window:int}} steps; {{cfg.evm.event_fraction}} |
| A2, A3 | per-step features | asinh z; log(1 + s); its EWMA over 1 h, 12 h, 3 d; deviation from trailing 1 d mean; forecast indicator |

: **Table S3.** **Secondary metrics, Mission 1.** Conformal-BH layer, test half, seed means.

| Arm | Precision | Recall | Alarming precision | ADTQC | Affiliation | Rare-only runs | Channel | Subsystem |
|---|---|---|---|---|---|---|---|---|
| A0 | {{pre.m1.A0.conformal_bh.p.mean:.3f}} | {{pre.m1.A0.conformal_bh.r.mean:.3f}} | {{pre.m1.A0.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m1.A0.conformal_bh.adtqc.mean:.3f}} | {{pre.m1.A0.conformal_bh.aff.mean:.3f}} | {{pre.m1.A0.conformal_bh.rare_only.mean:.0f}} | {{pre.m1.A0.conformal_bh.chan.mean:.3f}} | {{pre.m1.A0.conformal_bh.subsys.mean:.3f}} |
| A1 | {{pre.m1.A1.conformal_bh.p.mean:.3f}} | {{pre.m1.A1.conformal_bh.r.mean:.3f}} | {{pre.m1.A1.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m1.A1.conformal_bh.adtqc.mean:.3f}} | {{pre.m1.A1.conformal_bh.aff.mean:.3f}} | {{pre.m1.A1.conformal_bh.rare_only.mean:.0f}} | {{pre.m1.A1.conformal_bh.chan.mean:.3f}} | {{pre.m1.A1.conformal_bh.subsys.mean:.3f}} |
| A2 | {{pre.m1.A2.conformal_bh.p.mean:.3f}} | {{pre.m1.A2.conformal_bh.r.mean:.3f}} | {{pre.m1.A2.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m1.A2.conformal_bh.adtqc.mean:.3f}} | {{pre.m1.A2.conformal_bh.aff.mean:.3f}} | {{pre.m1.A2.conformal_bh.rare_only.mean:.0f}} | {{pre.m1.A2.conformal_bh.chan.mean:.3f}} | {{pre.m1.A2.conformal_bh.subsys.mean:.3f}} |
| A3 | {{pre.m1.A3.conformal_bh.p.mean:.3f}} | {{pre.m1.A3.conformal_bh.r.mean:.3f}} | {{pre.m1.A3.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m1.A3.conformal_bh.adtqc.mean:.3f}} | {{pre.m1.A3.conformal_bh.aff.mean:.3f}} | {{pre.m1.A3.conformal_bh.rare_only.mean:.0f}} | {{pre.m1.A3.conformal_bh.chan.mean:.3f}} | {{pre.m1.A3.conformal_bh.subsys.mean:.3f}} |

::: {custom-style="Table Note"}
Note: ADTQC, alarm detection timing quality; affiliation, affiliation-based $F_{0.5}$; rare-only runs, alarm runs touching only rare events; channel and subsystem, channel- and subsystem-aware $F_{0.5}$.
:::

: **Table S4.** **Secondary metrics, Mission 2.** Conformal-BH layer, test half, seed means.

| Arm | Precision | Recall | Alarming precision | ADTQC | Affiliation | Rare-only runs |
|---|---|---|---|---|---|---|
| A0 | {{pre.m2.A0.conformal_bh.p.mean:.3f}} | {{pre.m2.A0.conformal_bh.r.mean:.3f}} | {{pre.m2.A0.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m2.A0.conformal_bh.adtqc.mean:.3f}} | {{pre.m2.A0.conformal_bh.aff.mean:.3f}} | {{pre.m2.A0.conformal_bh.rare_only.mean:.0f}} |
| A1 | {{pre.m2.A1.conformal_bh.p.mean:.3f}} | {{pre.m2.A1.conformal_bh.r.mean:.3f}} | {{pre.m2.A1.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m2.A1.conformal_bh.adtqc.mean:.3f}} | {{pre.m2.A1.conformal_bh.aff.mean:.3f}} | {{pre.m2.A1.conformal_bh.rare_only.mean:.0f}} |
| A2 | {{pre.m2.A2.conformal_bh.p.mean:.3f}} | {{pre.m2.A2.conformal_bh.r.mean:.3f}} | {{pre.m2.A2.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m2.A2.conformal_bh.adtqc.mean:.3f}} | {{pre.m2.A2.conformal_bh.aff.mean:.3f}} | {{pre.m2.A2.conformal_bh.rare_only.mean:.0f}} |
| A3 | {{pre.m2.A3.conformal_bh.p.mean:.3f}} | {{pre.m2.A3.conformal_bh.r.mean:.3f}} | {{pre.m2.A3.conformal_bh.alarm_prec.mean:.3f}} | {{pre.m2.A3.conformal_bh.adtqc.mean:.3f}} | {{pre.m2.A3.conformal_bh.aff.mean:.3f}} | {{pre.m2.A3.conformal_bh.rare_only.mean:.0f}} |

::: {custom-style="Table Note"}
Note: ADTQC, alarm detection timing quality; affiliation, affiliation-based $F_{0.5}$; rare-only runs, alarm runs touching only rare events.
:::

: **Table S5.** **Alarm layers on identical scores.** Test-half event-wise $F_{0.5}$, seed means.

| Mission | Arm | Threshold | Conformal-BH | Rolling BH | ACI | ACI + events |
|---|---|---|---|---|---|---|
| 1 | A1 | {{pre.m1.A1.threshold.f05.mean:.3f}} | {{pre.m1.A1.conformal_bh.f05.mean:.3f}} | {{pre.m1.A1.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A1.aci.f05.mean:.4f}} | {{pre.m1.A1.aci_event.f05.mean:.3f}} |
| 1 | A2 | {{pre.m1.A2.threshold.f05.mean:.3f}} | {{pre.m1.A2.conformal_bh.f05.mean:.3f}} | {{pre.m1.A2.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A2.aci.f05.mean:.4f}} | {{pre.m1.A2.aci_event.f05.mean:.3f}} |
| 1 | A3 | {{pre.m1.A3.threshold.f05.mean:.3f}} | {{pre.m1.A3.conformal_bh.f05.mean:.3f}} | {{pre.m1.A3.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A3.aci.f05.mean:.4f}} | {{pre.m1.A3.aci_event.f05.mean:.3f}} |
| 1 | A3, no residual inputs | {{pre.m1.A3nr.threshold.f05.mean:.3f}} | {{pre.m1.A3nr.conformal_bh.f05.mean:.3f}} | {{pre.m1.A3nr.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A3nr.aci.f05.mean:.4f}} | {{pre.m1.A3nr.aci_event.f05.mean:.3f}} |
| 1 | A3, two classes | {{pre.m1.A3c2.threshold.f05.mean:.3f}} | {{pre.m1.A3c2.conformal_bh.f05.mean:.3f}} | {{pre.m1.A3c2.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A3c2.aci.f05.mean:.4f}} | {{pre.m1.A3c2.aci_event.f05.mean:.3f}} |
| 1 | A1, lightweight | {{pre.m1.A1lw.threshold.f05.mean:.3f}} | {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} | {{pre.m1.A1lw.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m1.A1lw.aci.f05.mean:.4f}} | {{pre.m1.A1lw.aci_event.f05.mean:.3f}} |
| 2 | A1 | {{pre.m2.A1.threshold.f05.mean:.3f}} | {{pre.m2.A1.conformal_bh.f05.mean:.3f}} | {{pre.m2.A1.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A1.aci.f05.mean:.4f}} | {{pre.m2.A1.aci_event.f05.mean:.3f}} |
| 2 | A2 | {{pre.m2.A2.threshold.f05.mean:.3f}} | {{pre.m2.A2.conformal_bh.f05.mean:.3f}} | {{pre.m2.A2.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A2.aci.f05.mean:.4f}} | {{pre.m2.A2.aci_event.f05.mean:.3f}} |
| 2 | A3 | {{pre.m2.A3.threshold.f05.mean:.3f}} | {{pre.m2.A3.conformal_bh.f05.mean:.3f}} | {{pre.m2.A3.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A3.aci.f05.mean:.4f}} | {{pre.m2.A3.aci_event.f05.mean:.3f}} |
| 2 | A1, lightweight | {{pre.m2.A1lw.threshold.f05.mean:.3f}} | {{pre.m2.A1lw.conformal_bh.f05.mean:.3f}} | {{pre.m2.A1lw.conformal_bh_rolling.f05.mean:.3f}} | {{pre.m2.A1lw.aci.f05.mean:.4f}} | {{pre.m2.A1lw.aci_event.f05.mean:.3f}} |

::: {custom-style="Table Note"}
Note: ACI and ACI + events are the point-level comparator as pre-registered ($\gamma$ tuned to a point-wise rate).
:::

![**Fig. S1.** **Generality of the training-half protocol's effect.** Training-half protocol minus calibration-window protocol for AR(8), test-half event-wise $F_{0.5}$, by mission, channel set and alarm layer. The ringed point is the pre-registered H6 comparison; all other points are exploratory single runs.](artwork/Figure_S1.png){width=70%}

![**Fig. S2.** **Alarm layers on identical scores.** Test-half event-wise $F_{0.5}$ of the same scores under the threshold, conformal-BH and rolling layers (seed means). Coinciding markers mean equal values; ACI is omitted because all its values are below 0.002.](artwork/Figure_S2.png){width=60%}

: **Table S6.** **Per-seed results of the supervised arms.** Mission 1, conformal-BH layer, test half (pre-registered seeds 0–4).

| Seed | A2 $F_{0.5}$ | A2 FA / yr | A3 $F_{0.5}$ | A3 FA / yr |
|---|---|---|---|---|
| 0 | {{pre.seed.m1_target_gbt_ar8_s0_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_gbt_ar8_s0_event.conformal_bh.fa_yr:.1f}} | {{pre.seed.m1_target_event_ar8_all_c3_s0_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_event_ar8_all_c3_s0_event.conformal_bh.fa_yr:.1f}} |
| 1 | {{pre.seed.m1_target_gbt_ar8_s1_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_gbt_ar8_s1_event.conformal_bh.fa_yr:.1f}} | {{pre.seed.m1_target_event_ar8_all_c3_s1_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_event_ar8_all_c3_s1_event.conformal_bh.fa_yr:.1f}} |
| 2 | {{pre.seed.m1_target_gbt_ar8_s2_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_gbt_ar8_s2_event.conformal_bh.fa_yr:.1f}} | {{pre.seed.m1_target_event_ar8_all_c3_s2_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_event_ar8_all_c3_s2_event.conformal_bh.fa_yr:.1f}} |
| 3 | {{pre.seed.m1_target_gbt_ar8_s3_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_gbt_ar8_s3_event.conformal_bh.fa_yr:.1f}} | {{pre.seed.m1_target_event_ar8_all_c3_s3_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_event_ar8_all_c3_s3_event.conformal_bh.fa_yr:.1f}} |
| 4 | {{pre.seed.m1_target_gbt_ar8_s4_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_gbt_ar8_s4_event.conformal_bh.fa_yr:.1f}} | {{pre.seed.m1_target_event_ar8_all_c3_s4_event.conformal_bh.f05:.3f}} | {{pre.seed.m1_target_event_ar8_all_c3_s4_event.conformal_bh.fa_yr:.1f}} |

: **Table S7.** **Exploratory phase.** AR(8) with the layer selected on the calibration window, and the label-free baselines; test-half $F_{0.5}$ (before the pre-registration).

| Detector | M1 full | M1 lightweight | M2 full | M2 lightweight |
|---|---|---|---|---|
| AR(8), conformal-BH | {{explo.m1.full.ar8.conformal_bh.f05.mean:.4f}} | {{explo.m1.lw.ar8.conformal_bh.f05.mean:.3f}} | {{explo.m2.full.ar8.conformal_bh.f05.mean:.3f}} | {{explo.m2.lw.ar8.conformal_bh.f05.mean:.3f}} |
| AR(8), threshold | {{explo.m1.full.ar8.threshold.f05.mean:.4f}} | {{explo.m1.lw.ar8.threshold.f05.mean:.3f}} | {{explo.m2.full.ar8.threshold.f05.mean:.3f}} | {{explo.m2.lw.ar8.threshold.f05.mean:.3f}} |
| out-of-limit rule | {{explo.m1.full.ool.native.f05.mean:.4f}} | {{explo.m1.lw.ool.native.f05.mean:.3f}} | {{explo.m2.full.ool.native.f05.mean:.3f}} | {{explo.m2.lw.ool.native.f05.mean:.3f}} |
| GlobalSTD5 | {{explo.m1.full.gstd5.native.f05.mean:.4f}} | {{explo.m1.lw.gstd5.native.f05.mean:.3f}} | {{explo.m2.full.gstd5.native.f05.mean:.3f}} | {{explo.m2.lw.gstd5.native.f05.mean:.3f}} |

![**Fig. S3.** **Calibration shift against test score (exploratory).** For each supervised run on Mission 1 (BH layer), the ratio of the nominal test exceedance rate to the cross-validation rate at the $1-10^{-4}$ quantile, against test-half event-wise $F_{0.5}$.](artwork/Figure_S3.png){width=60%}

# S4. Validation analyses (P1–P8)

All analyses in this section were specified and frozen before they were run, after the confirmatory analysis. They use test halves that had been used before.

**P3, exact bootstrap.** The block-decomposable metric of Appendix A was compared with the official value on {{ph.p3.n_checks}} stored arm–metric pairs; the largest absolute difference was {{ph.p3.max_abs_diff:.2g}} (criterion {{ph.p3.passed}}).

: **Table S8.** **P1 decomposition cells, Mission 1, target channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{ph.p1.m1.target.conformal_bh.reproduction}}.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | {{ph.p1.m1.target.conformal_bh.c0.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.c0.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.c0.level:g}} | {{ph.p1.m1.target.conformal_bh.c0.k}} |
| c1 | S0 | cal | cal | G1 | {{ph.p1.m1.target.conformal_bh.c1.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.c1.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.c1.level:g}} | {{ph.p1.m1.target.conformal_bh.c1.k}} |
| c2 | S½ | cal | cal | G1 | {{ph.p1.m1.target.conformal_bh.c2.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.c2.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.c2.level:g}} | {{ph.p1.m1.target.conformal_bh.c2.k}} |
| c3 | S1 | cal | cal | G1 | {{ph.p1.m1.target.conformal_bh.c3.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.c3.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.c3.level:g}} | {{ph.p1.m1.target.conformal_bh.c3.k}} |
| c4 | S1 | half | cal | G1 | {{ph.p1.m1.target.conformal_bh.c4.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.c4.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.c4.level:g}} | {{ph.p1.m1.target.conformal_bh.c4.k}} |
| c5 | S1 | half | half | G1 | {{ph.p1.m1.target.conformal_bh.c5.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.c5.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.c5.level:g}} | {{ph.p1.m1.target.conformal_bh.c5.k}} |
| r1 | S1 | half | half | G0 | {{ph.p1.m1.target.conformal_bh.r1.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.r1.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.r1.level:g}} | {{ph.p1.m1.target.conformal_bh.r1.k}} |
| r2 | S1 | half | cal | G0 | {{ph.p1.m1.target.conformal_bh.r2.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.r2.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.r2.level:g}} | {{ph.p1.m1.target.conformal_bh.r2.k}} |
| r3 | S1 | cal | cal | G0 | {{ph.p1.m1.target.conformal_bh.r3.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.r3.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.r3.level:g}} | {{ph.p1.m1.target.conformal_bh.r3.k}} |
| r4 | S½ | cal | cal | G0 | {{ph.p1.m1.target.conformal_bh.r4.f05:.3f}} | {{ph.p1.m1.target.conformal_bh.r4.fa_yr:.1f}} | {{ph.p1.m1.target.conformal_bh.r4.level:g}} | {{ph.p1.m1.target.conformal_bh.r4.k}} |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S9.** **P1 steps, Mission 1, target channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | {{ph.p1.m1.target.conformal_bh.forward.G.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.G.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.G.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.reverse.G.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.G.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.G.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.verdict.G}} |
| AR(8) fit data | {{ph.p1.m1.target.conformal_bh.forward.F.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.F.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.F.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.reverse.F.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.F.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.F.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.verdict.F}} |
| residual scale | {{ph.p1.m1.target.conformal_bh.forward.sigma.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.sigma.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.sigma.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.reverse.sigma.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.sigma.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.sigma.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.verdict.sigma}} |
| calibration pool | {{ph.p1.m1.target.conformal_bh.forward.P.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.P.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.P.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.reverse.P.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.P.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.P.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.verdict.P}} |
| selection data | {{ph.p1.m1.target.conformal_bh.forward.D.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.D.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.reverse.D.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.D.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.D.ci95_high:+.3f}}] | {{ph.p1.m1.target.conformal_bh.verdict.D}} |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S10.** **P1 decomposition cells, Mission 1, target channels, threshold layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{ph.p1.m1.target.threshold.reproduction}}.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | {{ph.p1.m1.target.threshold.c0.f05:.3f}} | {{ph.p1.m1.target.threshold.c0.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.c0.level:g}} | {{ph.p1.m1.target.threshold.c0.k}} |
| c1 | S0 | cal | cal | G1 | {{ph.p1.m1.target.threshold.c1.f05:.3f}} | {{ph.p1.m1.target.threshold.c1.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.c1.level:g}} | {{ph.p1.m1.target.threshold.c1.k}} |
| c2 | S½ | cal | cal | G1 | {{ph.p1.m1.target.threshold.c2.f05:.3f}} | {{ph.p1.m1.target.threshold.c2.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.c2.level:g}} | {{ph.p1.m1.target.threshold.c2.k}} |
| c3 | S1 | cal | cal | G1 | {{ph.p1.m1.target.threshold.c3.f05:.3f}} | {{ph.p1.m1.target.threshold.c3.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.c3.level:g}} | {{ph.p1.m1.target.threshold.c3.k}} |
| c4 | S1 | half | cal | G1 | {{ph.p1.m1.target.threshold.c4.f05:.3f}} | {{ph.p1.m1.target.threshold.c4.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.c4.level:g}} | {{ph.p1.m1.target.threshold.c4.k}} |
| c5 | S1 | half | half | G1 | {{ph.p1.m1.target.threshold.c5.f05:.3f}} | {{ph.p1.m1.target.threshold.c5.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.c5.level:g}} | {{ph.p1.m1.target.threshold.c5.k}} |
| r1 | S1 | half | half | G0 | {{ph.p1.m1.target.threshold.r1.f05:.3f}} | {{ph.p1.m1.target.threshold.r1.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.r1.level:g}} | {{ph.p1.m1.target.threshold.r1.k}} |
| r2 | S1 | half | cal | G0 | {{ph.p1.m1.target.threshold.r2.f05:.3f}} | {{ph.p1.m1.target.threshold.r2.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.r2.level:g}} | {{ph.p1.m1.target.threshold.r2.k}} |
| r3 | S1 | cal | cal | G0 | {{ph.p1.m1.target.threshold.r3.f05:.3f}} | {{ph.p1.m1.target.threshold.r3.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.r3.level:g}} | {{ph.p1.m1.target.threshold.r3.k}} |
| r4 | S½ | cal | cal | G0 | {{ph.p1.m1.target.threshold.r4.f05:.3f}} | {{ph.p1.m1.target.threshold.r4.fa_yr:.1f}} | {{ph.p1.m1.target.threshold.r4.level:g}} | {{ph.p1.m1.target.threshold.r4.k}} |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S11.** **P1 steps, Mission 1, target channels, threshold layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | {{ph.p1.m1.target.threshold.forward.G.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.forward.G.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.forward.G.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.reverse.G.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.reverse.G.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.reverse.G.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.verdict.G}} |
| AR(8) fit data | {{ph.p1.m1.target.threshold.forward.F.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.forward.F.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.forward.F.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.reverse.F.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.reverse.F.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.reverse.F.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.verdict.F}} |
| residual scale | {{ph.p1.m1.target.threshold.forward.sigma.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.forward.sigma.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.forward.sigma.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.reverse.sigma.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.reverse.sigma.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.reverse.sigma.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.verdict.sigma}} |
| calibration pool | {{ph.p1.m1.target.threshold.forward.P.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.forward.P.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.forward.P.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.reverse.P.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.reverse.P.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.reverse.P.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.verdict.P}} |
| selection data | {{ph.p1.m1.target.threshold.forward.D.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.forward.D.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.forward.D.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.reverse.D.delta_official:+.3f}} [{{ph.p1.m1.target.threshold.reverse.D.ci95_low:+.3f}}, {{ph.p1.m1.target.threshold.reverse.D.ci95_high:+.3f}}] | {{ph.p1.m1.target.threshold.verdict.D}} |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S12.** **P1 decomposition cells, Mission 1, lightweight channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{ph.p1.m1.lightweight.conformal_bh.reproduction}}.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | {{ph.p1.m1.lightweight.conformal_bh.c0.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.c0.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.c0.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.c0.k}} |
| c1 | S0 | cal | cal | G1 | {{ph.p1.m1.lightweight.conformal_bh.c1.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.c1.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.c1.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.c1.k}} |
| c2 | S½ | cal | cal | G1 | {{ph.p1.m1.lightweight.conformal_bh.c2.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.c2.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.c2.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.c2.k}} |
| c3 | S1 | cal | cal | G1 | {{ph.p1.m1.lightweight.conformal_bh.c3.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.c3.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.c3.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.c3.k}} |
| c4 | S1 | half | cal | G1 | {{ph.p1.m1.lightweight.conformal_bh.c4.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.c4.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.c4.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.c4.k}} |
| c5 | S1 | half | half | G1 | {{ph.p1.m1.lightweight.conformal_bh.c5.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.c5.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.c5.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.c5.k}} |
| r1 | S1 | half | half | G0 | {{ph.p1.m1.lightweight.conformal_bh.r1.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.r1.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.r1.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.r1.k}} |
| r2 | S1 | half | cal | G0 | {{ph.p1.m1.lightweight.conformal_bh.r2.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.r2.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.r2.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.r2.k}} |
| r3 | S1 | cal | cal | G0 | {{ph.p1.m1.lightweight.conformal_bh.r3.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.r3.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.r3.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.r3.k}} |
| r4 | S½ | cal | cal | G0 | {{ph.p1.m1.lightweight.conformal_bh.r4.f05:.3f}} | {{ph.p1.m1.lightweight.conformal_bh.r4.fa_yr:.1f}} | {{ph.p1.m1.lightweight.conformal_bh.r4.level:g}} | {{ph.p1.m1.lightweight.conformal_bh.r4.k}} |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S13.** **P1 steps, Mission 1, lightweight channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | {{ph.p1.m1.lightweight.conformal_bh.forward.G.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.forward.G.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.forward.G.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.reverse.G.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.reverse.G.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.reverse.G.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.verdict.G}} |
| AR(8) fit data | {{ph.p1.m1.lightweight.conformal_bh.forward.F.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.forward.F.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.forward.F.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.reverse.F.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.reverse.F.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.reverse.F.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.verdict.F}} |
| residual scale | {{ph.p1.m1.lightweight.conformal_bh.forward.sigma.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.forward.sigma.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.forward.sigma.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.reverse.sigma.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.reverse.sigma.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.reverse.sigma.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.verdict.sigma}} |
| calibration pool | {{ph.p1.m1.lightweight.conformal_bh.forward.P.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.forward.P.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.forward.P.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.reverse.P.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.reverse.P.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.reverse.P.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.verdict.P}} |
| selection data | {{ph.p1.m1.lightweight.conformal_bh.forward.D.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.forward.D.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.reverse.D.delta_official:+.3f}} [{{ph.p1.m1.lightweight.conformal_bh.reverse.D.ci95_low:+.3f}}, {{ph.p1.m1.lightweight.conformal_bh.reverse.D.ci95_high:+.3f}}] | {{ph.p1.m1.lightweight.conformal_bh.verdict.D}} |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S14.** **P1 decomposition cells, Mission 2, target channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{ph.p1.m2.target.conformal_bh.reproduction}}.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | {{ph.p1.m2.target.conformal_bh.c0.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.c0.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.c0.level:g}} | {{ph.p1.m2.target.conformal_bh.c0.k}} |
| c1 | S0 | cal | cal | G1 | {{ph.p1.m2.target.conformal_bh.c1.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.c1.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.c1.level:g}} | {{ph.p1.m2.target.conformal_bh.c1.k}} |
| c2 | S½ | cal | cal | G1 | {{ph.p1.m2.target.conformal_bh.c2.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.c2.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.c2.level:g}} | {{ph.p1.m2.target.conformal_bh.c2.k}} |
| c3 | S1 | cal | cal | G1 | {{ph.p1.m2.target.conformal_bh.c3.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.c3.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.c3.level:g}} | {{ph.p1.m2.target.conformal_bh.c3.k}} |
| c4 | S1 | half | cal | G1 | {{ph.p1.m2.target.conformal_bh.c4.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.c4.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.c4.level:g}} | {{ph.p1.m2.target.conformal_bh.c4.k}} |
| c5 | S1 | half | half | G1 | {{ph.p1.m2.target.conformal_bh.c5.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.c5.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.c5.level:g}} | {{ph.p1.m2.target.conformal_bh.c5.k}} |
| r1 | S1 | half | half | G0 | {{ph.p1.m2.target.conformal_bh.r1.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.r1.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.r1.level:g}} | {{ph.p1.m2.target.conformal_bh.r1.k}} |
| r2 | S1 | half | cal | G0 | {{ph.p1.m2.target.conformal_bh.r2.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.r2.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.r2.level:g}} | {{ph.p1.m2.target.conformal_bh.r2.k}} |
| r3 | S1 | cal | cal | G0 | {{ph.p1.m2.target.conformal_bh.r3.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.r3.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.r3.level:g}} | {{ph.p1.m2.target.conformal_bh.r3.k}} |
| r4 | S½ | cal | cal | G0 | {{ph.p1.m2.target.conformal_bh.r4.f05:.3f}} | {{ph.p1.m2.target.conformal_bh.r4.fa_yr:.1f}} | {{ph.p1.m2.target.conformal_bh.r4.level:g}} | {{ph.p1.m2.target.conformal_bh.r4.k}} |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S15.** **P1 steps, Mission 2, target channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | {{ph.p1.m2.target.conformal_bh.forward.G.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.forward.G.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.G.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.reverse.G.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.reverse.G.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.reverse.G.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.verdict.G}} |
| AR(8) fit data | {{ph.p1.m2.target.conformal_bh.forward.F.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.forward.F.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.F.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.reverse.F.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.reverse.F.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.reverse.F.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.verdict.F}} |
| residual scale | {{ph.p1.m2.target.conformal_bh.forward.sigma.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.forward.sigma.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.sigma.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.reverse.sigma.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.reverse.sigma.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.reverse.sigma.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.verdict.sigma}} |
| calibration pool | {{ph.p1.m2.target.conformal_bh.forward.P.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.forward.P.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.P.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.reverse.P.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.reverse.P.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.reverse.P.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.verdict.P}} |
| selection data | {{ph.p1.m2.target.conformal_bh.forward.D.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.D.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.reverse.D.delta_official:+.3f}} [{{ph.p1.m2.target.conformal_bh.reverse.D.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.reverse.D.ci95_high:+.3f}}] | {{ph.p1.m2.target.conformal_bh.verdict.D}} |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S16.** **P1 decomposition cells, Mission 2, target channels, threshold layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{ph.p1.m2.target.threshold.reproduction}}.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | {{ph.p1.m2.target.threshold.c0.f05:.3f}} | {{ph.p1.m2.target.threshold.c0.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.c0.level:g}} | {{ph.p1.m2.target.threshold.c0.k}} |
| c1 | S0 | cal | cal | G1 | {{ph.p1.m2.target.threshold.c1.f05:.3f}} | {{ph.p1.m2.target.threshold.c1.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.c1.level:g}} | {{ph.p1.m2.target.threshold.c1.k}} |
| c2 | S½ | cal | cal | G1 | {{ph.p1.m2.target.threshold.c2.f05:.3f}} | {{ph.p1.m2.target.threshold.c2.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.c2.level:g}} | {{ph.p1.m2.target.threshold.c2.k}} |
| c3 | S1 | cal | cal | G1 | {{ph.p1.m2.target.threshold.c3.f05:.3f}} | {{ph.p1.m2.target.threshold.c3.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.c3.level:g}} | {{ph.p1.m2.target.threshold.c3.k}} |
| c4 | S1 | half | cal | G1 | {{ph.p1.m2.target.threshold.c4.f05:.3f}} | {{ph.p1.m2.target.threshold.c4.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.c4.level:g}} | {{ph.p1.m2.target.threshold.c4.k}} |
| c5 | S1 | half | half | G1 | {{ph.p1.m2.target.threshold.c5.f05:.3f}} | {{ph.p1.m2.target.threshold.c5.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.c5.level:g}} | {{ph.p1.m2.target.threshold.c5.k}} |
| r1 | S1 | half | half | G0 | {{ph.p1.m2.target.threshold.r1.f05:.3f}} | {{ph.p1.m2.target.threshold.r1.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.r1.level:g}} | {{ph.p1.m2.target.threshold.r1.k}} |
| r2 | S1 | half | cal | G0 | {{ph.p1.m2.target.threshold.r2.f05:.3f}} | {{ph.p1.m2.target.threshold.r2.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.r2.level:g}} | {{ph.p1.m2.target.threshold.r2.k}} |
| r3 | S1 | cal | cal | G0 | {{ph.p1.m2.target.threshold.r3.f05:.3f}} | {{ph.p1.m2.target.threshold.r3.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.r3.level:g}} | {{ph.p1.m2.target.threshold.r3.k}} |
| r4 | S½ | cal | cal | G0 | {{ph.p1.m2.target.threshold.r4.f05:.3f}} | {{ph.p1.m2.target.threshold.r4.fa_yr:.1f}} | {{ph.p1.m2.target.threshold.r4.level:g}} | {{ph.p1.m2.target.threshold.r4.k}} |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S17.** **P1 steps, Mission 2, target channels, threshold layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | {{ph.p1.m2.target.threshold.forward.G.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.forward.G.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.forward.G.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.reverse.G.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.reverse.G.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.reverse.G.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.verdict.G}} |
| AR(8) fit data | {{ph.p1.m2.target.threshold.forward.F.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.forward.F.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.forward.F.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.reverse.F.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.reverse.F.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.reverse.F.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.verdict.F}} |
| residual scale | {{ph.p1.m2.target.threshold.forward.sigma.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.forward.sigma.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.forward.sigma.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.reverse.sigma.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.reverse.sigma.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.reverse.sigma.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.verdict.sigma}} |
| calibration pool | {{ph.p1.m2.target.threshold.forward.P.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.forward.P.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.forward.P.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.reverse.P.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.reverse.P.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.reverse.P.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.verdict.P}} |
| selection data | {{ph.p1.m2.target.threshold.forward.D.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.forward.D.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.forward.D.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.reverse.D.delta_official:+.3f}} [{{ph.p1.m2.target.threshold.reverse.D.ci95_low:+.3f}}, {{ph.p1.m2.target.threshold.reverse.D.ci95_high:+.3f}}] | {{ph.p1.m2.target.threshold.verdict.D}} |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S18.** **P1 decomposition cells, Mission 2, lightweight channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: {{ph.p1.m2.lightweight.conformal_bh.reproduction}}.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | {{ph.p1.m2.lightweight.conformal_bh.c0.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.c0.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.c0.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.c0.k}} |
| c1 | S0 | cal | cal | G1 | {{ph.p1.m2.lightweight.conformal_bh.c1.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.c1.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.c1.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.c1.k}} |
| c2 | S½ | cal | cal | G1 | {{ph.p1.m2.lightweight.conformal_bh.c2.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.c2.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.c2.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.c2.k}} |
| c3 | S1 | cal | cal | G1 | {{ph.p1.m2.lightweight.conformal_bh.c3.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.c3.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.c3.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.c3.k}} |
| c4 | S1 | half | cal | G1 | {{ph.p1.m2.lightweight.conformal_bh.c4.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.c4.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.c4.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.c4.k}} |
| c5 | S1 | half | half | G1 | {{ph.p1.m2.lightweight.conformal_bh.c5.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.c5.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.c5.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.c5.k}} |
| r1 | S1 | half | half | G0 | {{ph.p1.m2.lightweight.conformal_bh.r1.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.r1.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.r1.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.r1.k}} |
| r2 | S1 | half | cal | G0 | {{ph.p1.m2.lightweight.conformal_bh.r2.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.r2.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.r2.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.r2.k}} |
| r3 | S1 | cal | cal | G0 | {{ph.p1.m2.lightweight.conformal_bh.r3.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.r3.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.r3.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.r3.k}} |
| r4 | S½ | cal | cal | G0 | {{ph.p1.m2.lightweight.conformal_bh.r4.f05:.3f}} | {{ph.p1.m2.lightweight.conformal_bh.r4.fa_yr:.1f}} | {{ph.p1.m2.lightweight.conformal_bh.r4.level:g}} | {{ph.p1.m2.lightweight.conformal_bh.r4.k}} |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S19.** **P1 steps, Mission 2, lightweight channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | {{ph.p1.m2.lightweight.conformal_bh.forward.G.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.forward.G.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.forward.G.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.reverse.G.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.reverse.G.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.reverse.G.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.verdict.G}} |
| AR(8) fit data | {{ph.p1.m2.lightweight.conformal_bh.forward.F.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.forward.F.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.forward.F.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.reverse.F.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.reverse.F.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.reverse.F.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.verdict.F}} |
| residual scale | {{ph.p1.m2.lightweight.conformal_bh.forward.sigma.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.forward.sigma.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.forward.sigma.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.reverse.sigma.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.reverse.sigma.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.reverse.sigma.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.verdict.sigma}} |
| calibration pool | {{ph.p1.m2.lightweight.conformal_bh.forward.P.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.forward.P.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.forward.P.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.reverse.P.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.reverse.P.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.reverse.P.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.verdict.P}} |
| selection data | {{ph.p1.m2.lightweight.conformal_bh.forward.D.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.forward.D.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.reverse.D.delta_official:+.3f}} [{{ph.p1.m2.lightweight.conformal_bh.reverse.D.ci95_low:+.3f}}, {{ph.p1.m2.lightweight.conformal_bh.reverse.D.ci95_high:+.3f}}] | {{ph.p1.m2.lightweight.conformal_bh.verdict.D}} |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S20.** **P2 forward chaining, Mission 1.** Per fold: events of the fold, $F_{0.5}$ and false alarms per year for the calibration-window protocol W and the training-history protocol H.

| Layer | Fold start | Events | W $F_{0.5}$ | H $F_{0.5}$ | H − W | W FA / yr | H FA / yr |
|---|---|---|---|---|---|---|---|
| conformal-BH | {{ph.p2.m1.conformal_bh.fold0.start}} | {{ph.p2.m1.conformal_bh.fold0.n_events}} | {{ph.p2.m1.conformal_bh.fold0.W:.3f}} | {{ph.p2.m1.conformal_bh.fold0.H:.3f}} | {{ph.p2.m1.conformal_bh.fold0.diff:+.3f}} | {{ph.p2.m1.conformal_bh.fold0.W_fa_yr:.1f}} | {{ph.p2.m1.conformal_bh.fold0.H_fa_yr:.1f}} |
| conformal-BH | {{ph.p2.m1.conformal_bh.fold1.start}} | {{ph.p2.m1.conformal_bh.fold1.n_events}} | {{ph.p2.m1.conformal_bh.fold1.W:.3f}} | {{ph.p2.m1.conformal_bh.fold1.H:.3f}} | {{ph.p2.m1.conformal_bh.fold1.diff:+.3f}} | {{ph.p2.m1.conformal_bh.fold1.W_fa_yr:.1f}} | {{ph.p2.m1.conformal_bh.fold1.H_fa_yr:.1f}} |
| conformal-BH | {{ph.p2.m1.conformal_bh.fold2.start}} | {{ph.p2.m1.conformal_bh.fold2.n_events}} | {{ph.p2.m1.conformal_bh.fold2.W:.3f}} | {{ph.p2.m1.conformal_bh.fold2.H:.3f}} | {{ph.p2.m1.conformal_bh.fold2.diff:+.3f}} | {{ph.p2.m1.conformal_bh.fold2.W_fa_yr:.1f}} | {{ph.p2.m1.conformal_bh.fold2.H_fa_yr:.1f}} |
| conformal-BH | {{ph.p2.m1.conformal_bh.fold3.start}} | {{ph.p2.m1.conformal_bh.fold3.n_events}} | {{ph.p2.m1.conformal_bh.fold3.W:.3f}} | {{ph.p2.m1.conformal_bh.fold3.H:.3f}} | {{ph.p2.m1.conformal_bh.fold3.diff:+.3f}} | {{ph.p2.m1.conformal_bh.fold3.W_fa_yr:.1f}} | {{ph.p2.m1.conformal_bh.fold3.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m1.threshold.fold0.start}} | {{ph.p2.m1.threshold.fold0.n_events}} | {{ph.p2.m1.threshold.fold0.W:.3f}} | {{ph.p2.m1.threshold.fold0.H:.3f}} | {{ph.p2.m1.threshold.fold0.diff:+.3f}} | {{ph.p2.m1.threshold.fold0.W_fa_yr:.1f}} | {{ph.p2.m1.threshold.fold0.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m1.threshold.fold1.start}} | {{ph.p2.m1.threshold.fold1.n_events}} | {{ph.p2.m1.threshold.fold1.W:.3f}} | {{ph.p2.m1.threshold.fold1.H:.3f}} | {{ph.p2.m1.threshold.fold1.diff:+.3f}} | {{ph.p2.m1.threshold.fold1.W_fa_yr:.1f}} | {{ph.p2.m1.threshold.fold1.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m1.threshold.fold2.start}} | {{ph.p2.m1.threshold.fold2.n_events}} | {{ph.p2.m1.threshold.fold2.W:.3f}} | {{ph.p2.m1.threshold.fold2.H:.3f}} | {{ph.p2.m1.threshold.fold2.diff:+.3f}} | {{ph.p2.m1.threshold.fold2.W_fa_yr:.1f}} | {{ph.p2.m1.threshold.fold2.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m1.threshold.fold3.start}} | {{ph.p2.m1.threshold.fold3.n_events}} | {{ph.p2.m1.threshold.fold3.W:.3f}} | {{ph.p2.m1.threshold.fold3.H:.3f}} | {{ph.p2.m1.threshold.fold3.diff:+.3f}} | {{ph.p2.m1.threshold.fold3.W_fa_yr:.1f}} | {{ph.p2.m1.threshold.fold3.H_fa_yr:.1f}} |

::: {custom-style="Table Note"}
Note: Pooled over folds (BH): H − W = {{ph.p2.m1.conformal_bh.pooled.diff:+.3f}} [{{ph.p2.m1.conformal_bh.pooled.ci95_low:+.3f}}, {{ph.p2.m1.conformal_bh.pooled.ci95_high:+.3f}}]. Replication criterion met: {{ph.p2.m1.conformal_bh.replicates}}.
:::

: **Table S21.** **P2 forward chaining, Mission 2.** Per fold: events of the fold, $F_{0.5}$ and false alarms per year for the calibration-window protocol W and the training-history protocol H.

| Layer | Fold start | Events | W $F_{0.5}$ | H $F_{0.5}$ | H − W | W FA / yr | H FA / yr |
|---|---|---|---|---|---|---|---|
| conformal-BH | {{ph.p2.m2.conformal_bh.fold0.start}} | {{ph.p2.m2.conformal_bh.fold0.n_events}} | {{ph.p2.m2.conformal_bh.fold0.W:.3f}} | {{ph.p2.m2.conformal_bh.fold0.H:.3f}} | {{ph.p2.m2.conformal_bh.fold0.diff:+.3f}} | {{ph.p2.m2.conformal_bh.fold0.W_fa_yr:.1f}} | {{ph.p2.m2.conformal_bh.fold0.H_fa_yr:.1f}} |
| conformal-BH | {{ph.p2.m2.conformal_bh.fold1.start}} | {{ph.p2.m2.conformal_bh.fold1.n_events}} | {{ph.p2.m2.conformal_bh.fold1.W:.3f}} | {{ph.p2.m2.conformal_bh.fold1.H:.3f}} | {{ph.p2.m2.conformal_bh.fold1.diff:+.3f}} | {{ph.p2.m2.conformal_bh.fold1.W_fa_yr:.1f}} | {{ph.p2.m2.conformal_bh.fold1.H_fa_yr:.1f}} |
| conformal-BH | {{ph.p2.m2.conformal_bh.fold2.start}} | {{ph.p2.m2.conformal_bh.fold2.n_events}} | {{ph.p2.m2.conformal_bh.fold2.W:.3f}} | {{ph.p2.m2.conformal_bh.fold2.H:.3f}} | {{ph.p2.m2.conformal_bh.fold2.diff:+.3f}} | {{ph.p2.m2.conformal_bh.fold2.W_fa_yr:.1f}} | {{ph.p2.m2.conformal_bh.fold2.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m2.threshold.fold0.start}} | {{ph.p2.m2.threshold.fold0.n_events}} | {{ph.p2.m2.threshold.fold0.W:.3f}} | {{ph.p2.m2.threshold.fold0.H:.3f}} | {{ph.p2.m2.threshold.fold0.diff:+.3f}} | {{ph.p2.m2.threshold.fold0.W_fa_yr:.1f}} | {{ph.p2.m2.threshold.fold0.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m2.threshold.fold1.start}} | {{ph.p2.m2.threshold.fold1.n_events}} | {{ph.p2.m2.threshold.fold1.W:.3f}} | {{ph.p2.m2.threshold.fold1.H:.3f}} | {{ph.p2.m2.threshold.fold1.diff:+.3f}} | {{ph.p2.m2.threshold.fold1.W_fa_yr:.1f}} | {{ph.p2.m2.threshold.fold1.H_fa_yr:.1f}} |
| threshold | {{ph.p2.m2.threshold.fold2.start}} | {{ph.p2.m2.threshold.fold2.n_events}} | {{ph.p2.m2.threshold.fold2.W:.3f}} | {{ph.p2.m2.threshold.fold2.H:.3f}} | {{ph.p2.m2.threshold.fold2.diff:+.3f}} | {{ph.p2.m2.threshold.fold2.W_fa_yr:.1f}} | {{ph.p2.m2.threshold.fold2.H_fa_yr:.1f}} |

::: {custom-style="Table Note"}
Note: Pooled over folds (BH): H − W = {{ph.p2.m2.conformal_bh.pooled.diff:+.3f}} [{{ph.p2.m2.conformal_bh.pooled.ci95_low:+.3f}}, {{ph.p2.m2.conformal_bh.pooled.ci95_high:+.3f}}]. Replication criterion met: {{ph.p2.m2.conformal_bh.replicates}}.
:::

: **Table S22.** **P4, ACI tuned on the event objective.** Chosen $(\alpha, \gamma, k)$ and test-half $F_{0.5}$ for the best setting overall, the best with $\gamma > 0$, and $\gamma = 0$ (a fixed per-channel conformal threshold).

| Arm | Best: $\alpha$, $\gamma$, $k$ | $F_{0.5}$ | Best with $\gamma>0$ | $F_{0.5}$ | $\gamma=0$ $F_{0.5}$ |
|---|---|---|---|---|---|
| M1 A1 | {{ph.p4.m1.ar8.any_gamma.alpha:g}}, {{ph.p4.m1.ar8.any_gamma.gamma:g}}, {{ph.p4.m1.ar8.any_gamma.k}} | {{ph.p4.m1.ar8.any_gamma.f05:.3f}} | {{ph.p4.m1.ar8.gamma_positive.alpha:g}}, {{ph.p4.m1.ar8.gamma_positive.gamma:g}}, {{ph.p4.m1.ar8.gamma_positive.k}} | {{ph.p4.m1.ar8.gamma_positive.f05:.3f}} | {{ph.p4.m1.ar8.gamma_zero.f05:.3f}} |
| M1 A2 seed 0 | {{ph.p4.m1.gbt_ar8_s0_event.any_gamma.alpha:g}}, {{ph.p4.m1.gbt_ar8_s0_event.any_gamma.gamma:g}}, {{ph.p4.m1.gbt_ar8_s0_event.any_gamma.k}} | {{ph.p4.m1.gbt_ar8_s0_event.any_gamma.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s0_event.gamma_positive.alpha:g}}, {{ph.p4.m1.gbt_ar8_s0_event.gamma_positive.gamma:g}}, {{ph.p4.m1.gbt_ar8_s0_event.gamma_positive.k}} | {{ph.p4.m1.gbt_ar8_s0_event.gamma_positive.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s0_event.gamma_zero.f05:.3f}} |
| M1 A2 seed 1 | {{ph.p4.m1.gbt_ar8_s1_event.any_gamma.alpha:g}}, {{ph.p4.m1.gbt_ar8_s1_event.any_gamma.gamma:g}}, {{ph.p4.m1.gbt_ar8_s1_event.any_gamma.k}} | {{ph.p4.m1.gbt_ar8_s1_event.any_gamma.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s1_event.gamma_positive.alpha:g}}, {{ph.p4.m1.gbt_ar8_s1_event.gamma_positive.gamma:g}}, {{ph.p4.m1.gbt_ar8_s1_event.gamma_positive.k}} | {{ph.p4.m1.gbt_ar8_s1_event.gamma_positive.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s1_event.gamma_zero.f05:.3f}} |
| M1 A2 seed 2 | {{ph.p4.m1.gbt_ar8_s2_event.any_gamma.alpha:g}}, {{ph.p4.m1.gbt_ar8_s2_event.any_gamma.gamma:g}}, {{ph.p4.m1.gbt_ar8_s2_event.any_gamma.k}} | {{ph.p4.m1.gbt_ar8_s2_event.any_gamma.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s2_event.gamma_positive.alpha:g}}, {{ph.p4.m1.gbt_ar8_s2_event.gamma_positive.gamma:g}}, {{ph.p4.m1.gbt_ar8_s2_event.gamma_positive.k}} | {{ph.p4.m1.gbt_ar8_s2_event.gamma_positive.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s2_event.gamma_zero.f05:.3f}} |
| M1 A2 seed 3 | {{ph.p4.m1.gbt_ar8_s3_event.any_gamma.alpha:g}}, {{ph.p4.m1.gbt_ar8_s3_event.any_gamma.gamma:g}}, {{ph.p4.m1.gbt_ar8_s3_event.any_gamma.k}} | {{ph.p4.m1.gbt_ar8_s3_event.any_gamma.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s3_event.gamma_positive.alpha:g}}, {{ph.p4.m1.gbt_ar8_s3_event.gamma_positive.gamma:g}}, {{ph.p4.m1.gbt_ar8_s3_event.gamma_positive.k}} | {{ph.p4.m1.gbt_ar8_s3_event.gamma_positive.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s3_event.gamma_zero.f05:.3f}} |
| M1 A2 seed 4 | {{ph.p4.m1.gbt_ar8_s4_event.any_gamma.alpha:g}}, {{ph.p4.m1.gbt_ar8_s4_event.any_gamma.gamma:g}}, {{ph.p4.m1.gbt_ar8_s4_event.any_gamma.k}} | {{ph.p4.m1.gbt_ar8_s4_event.any_gamma.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s4_event.gamma_positive.alpha:g}}, {{ph.p4.m1.gbt_ar8_s4_event.gamma_positive.gamma:g}}, {{ph.p4.m1.gbt_ar8_s4_event.gamma_positive.k}} | {{ph.p4.m1.gbt_ar8_s4_event.gamma_positive.f05:.3f}} | {{ph.p4.m1.gbt_ar8_s4_event.gamma_zero.f05:.3f}} |
| M2 A1 | {{ph.p4.m2.ar8.any_gamma.alpha:g}}, {{ph.p4.m2.ar8.any_gamma.gamma:g}}, {{ph.p4.m2.ar8.any_gamma.k}} | {{ph.p4.m2.ar8.any_gamma.f05:.3f}} | {{ph.p4.m2.ar8.gamma_positive.alpha:g}}, {{ph.p4.m2.ar8.gamma_positive.gamma:g}}, {{ph.p4.m2.ar8.gamma_positive.k}} | {{ph.p4.m2.ar8.gamma_positive.f05:.3f}} | {{ph.p4.m2.ar8.gamma_zero.f05:.3f}} |

: **Table S23.** **P5, registered anomaly-score arm, Mission 1.** Anomaly-only test-half $F_{0.5}$ with the layer selected on the anomaly-only objective.

| Arm | Runs | Mean | sd | Min | Max | FA / yr |
|---|---|---|---|---|---|---|
| A1 | {{ph.p5.m1.A1.n}} | {{ph.p5.m1.A1.mean:.3f}} | {{ph.p5.m1.A1.std:.3f}} | {{ph.p5.m1.A1.min:.3f}} | {{ph.p5.m1.A1.max:.3f}} | {{ph.p5.m1.A1.fa_yr:.1f}} |
| A2 | {{ph.p5.m1.A2.n}} | {{ph.p5.m1.A2.mean:.3f}} | {{ph.p5.m1.A2.std:.3f}} | {{ph.p5.m1.A2.min:.3f}} | {{ph.p5.m1.A2.max:.3f}} | {{ph.p5.m1.A2.fa_yr:.1f}} |
| A3 | {{ph.p5.m1.A3.n}} | {{ph.p5.m1.A3.mean:.3f}} | {{ph.p5.m1.A3.std:.3f}} | {{ph.p5.m1.A3.min:.3f}} | {{ph.p5.m1.A3.max:.3f}} | {{ph.p5.m1.A3.fa_yr:.1f}} |

::: {custom-style="Table Note"}
Note: A1 is deterministic; A2 and A3 use their anomaly log-odds.
:::

: **Table S24.** **P5, registered anomaly-score arm, Mission 2.** Anomaly-only test-half $F_{0.5}$ with the layer selected on the anomaly-only objective.

| Arm | Runs | Mean | sd | Min | Max | FA / yr |
|---|---|---|---|---|---|---|
| A1 | {{ph.p5.m2.A1.n}} | {{ph.p5.m2.A1.mean:.3f}} | {{ph.p5.m2.A1.std:.3f}} | {{ph.p5.m2.A1.min:.3f}} | {{ph.p5.m2.A1.max:.3f}} | {{ph.p5.m2.A1.fa_yr:.1f}} |
| A2 | {{ph.p5.m2.A2.n}} | {{ph.p5.m2.A2.mean:.3f}} | {{ph.p5.m2.A2.std:.3f}} | {{ph.p5.m2.A2.min:.3f}} | {{ph.p5.m2.A2.max:.3f}} | {{ph.p5.m2.A2.fa_yr:.1f}} |
| A3 | {{ph.p5.m2.A3.n}} | {{ph.p5.m2.A3.mean:.3f}} | {{ph.p5.m2.A3.std:.3f}} | {{ph.p5.m2.A3.min:.3f}} | {{ph.p5.m2.A3.max:.3f}} | {{ph.p5.m2.A3.fa_yr:.1f}} |

::: {custom-style="Table Note"}
Note: A1 is deterministic; A2 and A3 use their anomaly log-odds.
:::

: **Table S25.** **P6, seed extension.** Mission 1, conformal-BH layer: seed statistics with all available seeds, and the comparisons with the exact bootstrap (unadjusted, exploratory).

| Arm or comparison | Seeds | Mean $F_{0.5}$ / difference | sd | Min / 95 % CI | Max / $p$ |
|---|---|---|---|---|---|
| A2 | {{ph.p6.m1.A2.f05.n}} | {{ph.p6.m1.A2.f05.mean:.3f}} | {{ph.p6.m1.A2.f05.std:.3f}} | {{ph.p6.m1.A2.f05.min:.3f}} | {{ph.p6.m1.A2.f05.max:.3f}} |
| A3 | {{ph.p6.m1.A3.f05.n}} | {{ph.p6.m1.A3.f05.mean:.3f}} | {{ph.p6.m1.A3.f05.std:.3f}} | {{ph.p6.m1.A3.f05.min:.3f}} | {{ph.p6.m1.A3.f05.max:.3f}} |
| H4: A3 − A1 |  | {{ph.p6.m1.h4.diff:+.3f}} |  | [{{ph.p6.m1.h4.ci95_low:+.3f}}, {{ph.p6.m1.h4.ci95_high:+.3f}}] | p = {{ph.p6.m1.h4.p_boot:.3f}} |
| H5: A3 − A2 |  | {{ph.p6.m1.h5.diff:+.3f}} |  | [{{ph.p6.m1.h5.ci95_low:+.3f}}, {{ph.p6.m1.h5.ci95_high:+.3f}}] | p = {{ph.p6.m1.h5.p_boot:.3f}} |
| X1: A2 − A1 |  | {{ph.p6.m1.x1.diff:+.3f}} |  | [{{ph.p6.m1.x1.ci95_low:+.3f}}, {{ph.p6.m1.x1.ci95_high:+.3f}}] | p = {{ph.p6.m1.x1.p_boot:.3f}} |

: **Table S26.** **P7, operational metrics.** Test halves, conformal-BH layer; seed means for A2 and A3.

| Mission | Arm | Detected / events | Anomalies | Rare events | Delay, median [IQR] (h) | Within 1 h | Within 24 h | Runs per detected event | FA per 1000 nominal h |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A0 | {{ph.p7.m1.A0.detected:.1f}} / {{ph.p7.m1.A0.n_events:.0f}} | {{ph.p7.m1.A0.detected_anomalies:.1f}} | {{ph.p7.m1.A0.detected_rare:.1f}} | {{ph.p7.m1.A0.delay_h_median:.2f}} [{{ph.p7.m1.A0.delay_h_q25:.2f}}, {{ph.p7.m1.A0.delay_h_q75:.2f}}] | {{ph.p7.m1.A0.detected_within_1h:.2f}} | {{ph.p7.m1.A0.detected_within_24h:.2f}} | {{ph.p7.m1.A0.runs_per_detected_event_median:.1f}} | {{ph.p7.m1.A0.false_alarms_per_1000_nominal_h:.3f}} |
| 1 | A1 | {{ph.p7.m1.A1.detected:.1f}} / {{ph.p7.m1.A1.n_events:.0f}} | {{ph.p7.m1.A1.detected_anomalies:.1f}} | {{ph.p7.m1.A1.detected_rare:.1f}} | {{ph.p7.m1.A1.delay_h_median:.2f}} [{{ph.p7.m1.A1.delay_h_q25:.2f}}, {{ph.p7.m1.A1.delay_h_q75:.2f}}] | {{ph.p7.m1.A1.detected_within_1h:.2f}} | {{ph.p7.m1.A1.detected_within_24h:.2f}} | {{ph.p7.m1.A1.runs_per_detected_event_median:.1f}} | {{ph.p7.m1.A1.false_alarms_per_1000_nominal_h:.3f}} |
| 1 | A2 | {{ph.p7.m1.A2.detected:.1f}} / {{ph.p7.m1.A2.n_events:.0f}} | {{ph.p7.m1.A2.detected_anomalies:.1f}} | {{ph.p7.m1.A2.detected_rare:.1f}} | {{ph.p7.m1.A2.delay_h_median:.2f}} [{{ph.p7.m1.A2.delay_h_q25:.2f}}, {{ph.p7.m1.A2.delay_h_q75:.2f}}] | {{ph.p7.m1.A2.detected_within_1h:.2f}} | {{ph.p7.m1.A2.detected_within_24h:.2f}} | {{ph.p7.m1.A2.runs_per_detected_event_median:.1f}} | {{ph.p7.m1.A2.false_alarms_per_1000_nominal_h:.3f}} |
| 1 | A3 | {{ph.p7.m1.A3.detected:.1f}} / {{ph.p7.m1.A3.n_events:.0f}} | {{ph.p7.m1.A3.detected_anomalies:.1f}} | {{ph.p7.m1.A3.detected_rare:.1f}} | {{ph.p7.m1.A3.delay_h_median:.2f}} [{{ph.p7.m1.A3.delay_h_q25:.2f}}, {{ph.p7.m1.A3.delay_h_q75:.2f}}] | {{ph.p7.m1.A3.detected_within_1h:.2f}} | {{ph.p7.m1.A3.detected_within_24h:.2f}} | {{ph.p7.m1.A3.runs_per_detected_event_median:.1f}} | {{ph.p7.m1.A3.false_alarms_per_1000_nominal_h:.3f}} |
| 2 | A0 | {{ph.p7.m2.A0.detected:.1f}} / {{ph.p7.m2.A0.n_events:.0f}} | {{ph.p7.m2.A0.detected_anomalies:.1f}} | {{ph.p7.m2.A0.detected_rare:.1f}} | {{ph.p7.m2.A0.delay_h_median:.2f}} [{{ph.p7.m2.A0.delay_h_q25:.2f}}, {{ph.p7.m2.A0.delay_h_q75:.2f}}] | {{ph.p7.m2.A0.detected_within_1h:.2f}} | {{ph.p7.m2.A0.detected_within_24h:.2f}} | {{ph.p7.m2.A0.runs_per_detected_event_median:.1f}} | {{ph.p7.m2.A0.false_alarms_per_1000_nominal_h:.3f}} |
| 2 | A1 | {{ph.p7.m2.A1.detected:.1f}} / {{ph.p7.m2.A1.n_events:.0f}} | {{ph.p7.m2.A1.detected_anomalies:.1f}} | {{ph.p7.m2.A1.detected_rare:.1f}} | {{ph.p7.m2.A1.delay_h_median:.2f}} [{{ph.p7.m2.A1.delay_h_q25:.2f}}, {{ph.p7.m2.A1.delay_h_q75:.2f}}] | {{ph.p7.m2.A1.detected_within_1h:.2f}} | {{ph.p7.m2.A1.detected_within_24h:.2f}} | {{ph.p7.m2.A1.runs_per_detected_event_median:.1f}} | {{ph.p7.m2.A1.false_alarms_per_1000_nominal_h:.3f}} |
| 2 | A2 | {{ph.p7.m2.A2.detected:.1f}} / {{ph.p7.m2.A2.n_events:.0f}} | {{ph.p7.m2.A2.detected_anomalies:.1f}} | {{ph.p7.m2.A2.detected_rare:.1f}} | {{ph.p7.m2.A2.delay_h_median:.2f}} [{{ph.p7.m2.A2.delay_h_q25:.2f}}, {{ph.p7.m2.A2.delay_h_q75:.2f}}] | {{ph.p7.m2.A2.detected_within_1h:.2f}} | {{ph.p7.m2.A2.detected_within_24h:.2f}} | {{ph.p7.m2.A2.runs_per_detected_event_median:.1f}} | {{ph.p7.m2.A2.false_alarms_per_1000_nominal_h:.3f}} |
| 2 | A3 | {{ph.p7.m2.A3.detected:.1f}} / {{ph.p7.m2.A3.n_events:.0f}} | {{ph.p7.m2.A3.detected_anomalies:.1f}} | {{ph.p7.m2.A3.detected_rare:.1f}} | {{ph.p7.m2.A3.delay_h_median:.2f}} [{{ph.p7.m2.A3.delay_h_q25:.2f}}, {{ph.p7.m2.A3.delay_h_q75:.2f}}] | {{ph.p7.m2.A3.detected_within_1h:.2f}} | {{ph.p7.m2.A3.detected_within_24h:.2f}} | {{ph.p7.m2.A3.runs_per_detected_event_median:.1f}} | {{ph.p7.m2.A3.false_alarms_per_1000_nominal_h:.3f}} |

::: {custom-style="Table Note"}
Note: Delay is first-alarm time minus event start for detected events; IQR, interquartile range.
:::

**P8, provenance.** Re-running the frozen selection for the A1 arms reproduced the saved choice and every grid value: Mission 1 {{ph.p8.m1.passed}} (largest difference {{ph.p8.m1.max_abs_cv_diff:.2g}}), Mission 2 {{ph.p8.m2.passed}} (largest difference {{ph.p8.m2.max_abs_cv_diff:.2g}}).

# S5. Code, data and reproduction

The repository contains the frozen code of the pre-registered study, the validation scripts (`scripts/posthoc_v2/`), the evidence builder (`thesis/tools/build_evidence.py`) and the manuscript build (`paper/journal_v2/tools/`). The ESA Anomaly Dataset is obtained from Zenodo (DOI 10.5281/zenodo.12528696). Derived score and alarm arrays will be deposited with the archived release.

# References

::: {#refs}
:::

