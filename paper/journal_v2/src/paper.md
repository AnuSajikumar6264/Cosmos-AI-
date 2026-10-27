---
title: "Beyond model complexity: a pre-registered evaluation of event-level alarm protocols for spacecraft telemetry anomaly detection"
author:
  - "[Author name(s)]^a^ · ^a^[Affiliation, postal address, country] · Corresponding author: [name, e-mail]"
abstract: |
  Event-level evaluation has made alarm handling central to anomaly detection in spacecraft telemetry. This study asked whether event-level performance on the ESA Anomaly Detection Benchmark (ESA-ADB) depends more on the protocol that fits a scoring model and turns its scores into alarms than on the model. In a pre-registered, hash-frozen study on the official split at full channel scale, an order-8 autoregressive forecaster (AR(8)), gradient-boosted trees and a temporal convolutional network (TCN) fed one event-level alarm layer of conformal p-values with Benjamini–Hochberg (BH) aggregation. On Mission 1, fitting AR(8) and selecting the alarm layer on the whole training half raised the corrected event-wise $F_{0.5}$ from {{pre.m1.A0.conformal_bh.f05.mean:.3f}} to {{pre.m1.A1.conformal_bh.f05.mean:.3f}} (difference {{conf.m1.h6.diff_official:+.3f}}; 95 % confidence interval {{conf.m1.h6.ci95_low:+.3f}} to {{conf.m1.h6.ci95_high:+.3f}}; Holm-adjusted $p$ ≤ {{conf.p_floor_holm:g}}), at about one false alarm per year. The TCN improved on neither AR(8) nor the trees. Validation analyses planned in advance attributed the gain to the labels used to select the alarm settings, not to the fitted scores. In forward-chaining folds inside the training halves it won only {{ph.p2.m1.conformal_bh.n_folds_H_better}} of {{ph.p2.m1.conformal_bh.n_folds}} Mission 1 folds, short of the replication criterion; the gain also reversed on a lightweight channel subset and, under BH, on Mission 2. Adaptive conformal inference failed when tuned to a point-wise rate but reached {{ph.p4.m1.ar8.gamma_positive.f05:.3f}} when tuned on events. Both test halves had been used in an earlier exploratory phase. Alarm-protocol choices moved event-level scores more than model capacity did, but the gain depended on the setting.

  **Keywords:** Spacecraft telemetry; Anomaly detection; Event-wise evaluation; Conformal prediction; Alarm calibration; Pre-registration

  **Classification codes (Inspec):** C6170K; C1140Z; C5290; B0170N; E3650C
---

# Introduction {#sec:intro}

A flight operations team watches thousands of telemetry channels, and an automated detector earns its place only if its alarms are few and point at something real: a detector that raises hundreds of false alarms a year is soon ignored.

For this reason the scoring of anomaly detectors has moved from point-wise to event-wise metrics. Point-adjusted scores can make even random detectors look strong [@kim2022rigorous], several popular benchmarks are close to trivial or mislabelled [@wu2021flawed], and @sehili2023flawed proposed correcting event-wise precision with a time-based true-negative rate so that long alarms on nominal data are penalised. The ESA Anomaly Detection Benchmark (ESA-ADB) applies this view to real spacecraft telemetry annotated by operations engineers, with a chronological split and a corrected event-wise $F_{0.5}$ as its primary metric [@kotowski2026esaadb]. Its baselines score at most {{lit.dmlr.m1.full.best:.3f}} on Mission 1 and {{lit.dmlr.m2.full.best:.3f}} on Mission 2 on the full channel sets, and its authors trace the Mission 1 failures to false detections and stress that thresholding and post-processing matter.

Much of the method literature proposes stronger scorers instead: recurrent forecasters with dynamic thresholds [@hundman2018], variational models [@garcia2022dcvae], and transformer and pretrained forecasters used as anomaly scorers [@nie2023patchtst; @ansari2024chronos]. That raises a practical question. How much of event-level performance is decided by the protocol around the scorer (how it is fitted, which data form the calibration pool, on which labels the alarm settings are chosen), and how much by its capacity?

The objectives of this study were (i) to compare an unsupervised forecaster, gradient-boosted trees and a deep temporal convolutional network under one event-level alarm layer, on the official ESA-ADB protocol at full channel scale; (ii) to test whether a protocol that fits and selects on the whole training half improves on one that uses a short calibration window; and (iii) to establish, with analyses whose rules were fixed before they were run, which part of that protocol produces any gain and whether the gain holds forward in time. The confirmatory part was pre-registered and frozen by SHA-256 hashes before any new arm was scored on test data, and each new arm was scored on the test half once. The contributions are:

1. A confirmatory comparison of AR(8), XGBoost and a causal TCN under one event-level conformal alarm layer, with the minimum effect of interest and the family-wise correction fixed in advance. All three hypotheses are reported, including the two that were refuted.
2. Evidence that on Mission 1 the largest improvement came from the alarm protocol rather than from model capacity, a one-factor-at-a-time decomposition that locates the improvement, and a forward-chaining check of whether it carries over to later periods.
3. An account of point-level adaptive conformal inference at the event level: why it fails when tuned to a point-wise rate, and how it performs when tuned on events.
4. An exact block-decomposable form of the official metric that makes the block bootstrap exact, and an evidence file that traces every number in this paper to its source result file.

Two features of the study limit how its results may be read, and we report them as findings. The protocols compared in the main hypothesis differ in five respects, and both test halves had been scored in an exploratory phase before the pre-registration ([@sec:limits]).

# Related work {#sec:related}

**Evaluation and prior results.** Benchmark design for time-series anomaly detection has been criticised for labelling and triviality problems [@wu2021flawed], inflated scores under point adjustment [@kim2022rigorous] and protocols that overstate performance [@sehili2023flawed]. Range-based [@tatbul2018precision] and affiliation metrics [@huet2022local] address the fragmentation of events, and large evaluations find that no algorithm family dominates [@schmidl2022anomaly; @paparrizos2022tsbuad; @liu2024tsbad]. The studies that report results on ESA-ADB since the benchmark paper ([S:s-lit]) differ in split, retraining, channel selection, preprocessing or metric. TC-CAD, a pre-registered study on {{lit.tccad.m1_channels}} and {{lit.tccad.m2_channels}} channels, found telecommand conditioning unhelpful [@sahin2026tccad]; @nguyen2026 compared supervised and unsupervised detectors with uncorrected metrics; @brancato2027ast, @alcarria2026 and @allegrini2026 report higher Mission 1 scores ({{lit.brancato.smed_full:.3f}}, {{lit.alcarria.m1_f05:.3f}} and {{lit.allegrini.private:.3f}}) under a hierarchical protocol, monthly retraining and a Kaggle split, respectively; and @heydari2026 study threshold transfer under alert budgets. None of these numbers is comparable with ours.

**From scores to alarms.** Telemanom made dynamic thresholds with pruning practical on spacecraft data [@hundman2018], and extreme-value thresholds adapt to streaming data [@siffer2017]. Conformal anomaly detection turns a non-conformity score into a p-value against a calibration set [@laxhammar2010conformal; @laxhammar2015inductive]. Benjamini–Hochberg (BH) controls the false-discovery rate for independent [@benjamini1995] and positively dependent tests [@benjamini2001control], and @bates2023 showed that conformal p-values sharing one calibration set satisfy the required condition. Adaptive conformal inference (ACI) adjusts the miscoverage level online to keep long-run coverage under drift [@gibbs2021aci], and weighted variants relax exchangeability [@barber2023]. On ESA-ADB, TC-CAD found ACI the best calibrated of its point-level layers, yet it exceeded its point-wise target by {{lit.tccad.aci_far_overshoot}}× and collapsed the event score [@sahin2026tccad]. The present work moves calibration to the event level, selects the layer on the event metric over all training labels and compares scorer families under that one layer. On tabular problems, which our engineered per-step features resemble, gradient-boosted trees often remain strong against deep models [@grinsztajn2022tree].

# Materials and methods {#sec:methods}

## Data, split and metric {#sec:data}

**Data.** We use the ESA Anomaly Dataset, version 1.0 [@esaadb_dataset], Missions 1 and 2, on the official sampling grids of {{data.esa.m1.grid_seconds}} s and {{data.esa.m2.grid_seconds}} s. The official chronological split divides each mission into a training half and a test half. We use the full target channel sets of {{data.esa.m1.target_channels}} and {{data.esa.m2.target_channels}} channels; the lightweight subsets appear only in secondary analyses. On the target channels the Mission 1 training half holds {{data.esa.m1.events_all.train.targets}} target events ({{data.esa.m1.events_all.train.Anomaly}} anomalies and {{data.esa.m1.events_all.train.Rare_Event}} rare nominal events). The Mission 1 test half, {{data.esa.m1.test.years:.1f}} years long, holds {{data.esa.m1.events_target.test.Anomaly}} anomalies and {{data.esa.m1.events_target.test.Rare_Event}} rare events; the Mission 2 test half holds {{data.esa.m2.events_target.test.Anomaly}} anomalies and {{data.esa.m2.events_target.test.Rare_Event}} rare events. Mission 3 is not used, because its authors advise against it for algorithm comparison [@kotowski2026esaadb].

**Preprocessing.** The official preprocessing was re-implemented in vectorised form and checked against a literal port of the official code. It includes a "restore" step that moves annotated samples onto grid points, so test labels influence test inputs at a few grid points. The step is part of the benchmark protocol and applies equally to every arm.

**Metric.** The primary metric is the corrected event-wise $F_{0.5}$ over anomalies and rare events, computed by the official code, which we vendor unchanged (byte-identical to the upstream repository at commit 67194d3). With true-positive events $\text{TP}_e$, false-positive alarm runs $\text{FP}_e$ and missed events $\text{FN}_e$,
$$
P_e = \frac{\text{TP}_e}{\text{TP}_e+\text{FP}_e}\,\text{TNR},
$$ {#eq:prec}
$$
R_e = \frac{\text{TP}_e}{\text{TP}_e+\text{FN}_e},
$$ {#eq:rec}
$$
F_{0.5} = \frac{1.25\,P_e R_e}{0.25\,P_e+R_e},
$$ {#eq:metric}
where $\text{TNR}$ is one minus the share of nominal time covered by alarms. We also report anomaly-only $F_{0.5}$, false-alarm runs per year, all alarm runs per year and alarming precision, that is, true-positive events divided by true-positive events plus redundant detections (further alarm runs on an event already detected). The other pre-registered secondary metrics are in [S:s-secondary-m1] and [S:s-secondary-m2].

## Scoring models {#sec:models}

**AR(8).** Each channel is robust-normalised with the median and interquartile range of the nominal samples of a fitting period at the start of the training half. A ridge autoregressive model of order 8, fitted on at most 500,000 randomly sampled nominal rows, forecasts up to 16 steps ahead. The non-conformity score is the absolute forecast error divided by a robust residual scale $\hat\sigma_c$ of channel $c$, floored at $\delta$ = 0.05.

**Out-of-fold scores.** The training half is cut into five contiguous blocks. Each block is scored by a model fitted on the nominal samples of the other blocks, with a one-day purge on either side ({{oof.m1.purge_steps:int}} steps on Mission 1). Test scores come from a model fitted on all nominal training rows.

**Supervised scorers.** Every channel and step receives seven causal features built from the normalised value and the out-of-fold AR(8) score: $\text{asinh}\,z$, $\log(1+s)$, its exponential moving averages over 1 h, 12 h and 3 d, the deviation from the trailing one-day mean and a forecast indicator. Arm A2 is a three-class XGBoost model [@chen2016xgboost] (nominal, anomaly, rare event) on these features plus trailing maxima of $\log(1+s)$ and the channel index. Arm A3 is a causal temporal convolutional network [@bai2018tcn] with dilated convolutions, a receptive field of {{evm.receptive_field}} steps, a learned channel embedding and cross-channel mixing ({{evm.m1.params:int}} parameters on Mission 1), trained with AdamW [@loshchilov2019adamw], a one-cycle schedule [@smith2019superconvergence] and class-weighted cross-entropy. Both were trained out of fold on the same blocks as AR(8), without a hyperparameter search ([S:s-models]); their event score is $\log P(\text{anomaly}\cup\text{rare}) - \log P(\text{nominal})$.

## The event-level alarm layer {#sec:layer}

The pool $\mathcal C_c$ of channel $c$ holds nominal training scores of that channel. A test score $s$ receives the conformal p-value
$$
p_c(s) = \frac{1+\#\{s'\in\mathcal C_c: s'\ge s\}}{|\mathcal C_c|+1}.
$$ {#eq:p}
At every step, BH at level $q$ is applied across channels, and a channel whose p-value is rejected is *hot*. A global alarm starts once at least $k$ consecutive steps contain a hot channel. Alarm runs closer than 1 h are merged. The BH guarantee and the result of @bates2023 do not apply here, because the pools are per channel, the scores are correlated across channels, and telemetry is serially dependent and drifts. We therefore treat $q$ as a tuning parameter and report realised false-alarm rates.

**Selection.** $(q, k)$ is chosen by cross-fitting: calibrate on the nominal scores of one half of the training half, raise alarms on the other half and score them with the official metric on that half's labels, then swap the halves and average. Ties go to the smaller alarm fraction. The chosen layer is refitted on the whole training half and applied once to the test scores. The pre-registered grid is $q \in$ {{cfg.prereg.levels.conformal_bh}} and $k \in$ {{cfg.prereg.ks}}.

**Comparators.** Three comparators were scored on identical scores. The *per-channel threshold* selects its tail fraction from {{cfg.prereg.levels.threshold}} by the same rule (0 means the maximum nominal score). The *rolling layer* refreshes its pools daily from the trailing 30 days of unlabelled test scores and reuses the $(q,k)$ selected for BH. *Point-level ACI* [@gibbs2021aci] updates its level as
$$
\alpha_{t+1} = \alpha_t + \gamma\left(\alpha - \mathbb 1[s_t > \hat Q_{1-\alpha_t}]\right),
$$ {#eq:aci}
with $\alpha$ = {{cfg.prereg.aci_alpha}}; the threshold $\hat Q_{1-\alpha_t}$ is infinite while $\alpha_t \le 0$. In the pre-registration $\gamma \in$ {{cfg.prereg.aci_gammas}} was chosen so that the point-wise false-alarm rate on the second training half came closest to $\alpha$, a rule adapted from TC-CAD [@sahin2026tccad]. ACI was scored without event logic and with the persistence and merging of the BH layer.

## Arms and the five differences between them {#sec:arms}

A0 is the AR(8) arm of the exploratory phase, with its alarm layer selected on a short calibration window just before the test half. A1 applies the protocol of [@sec:layer] to out-of-fold AR(8) scores; A2 and A3 apply it to the supervised scorers, with five seeds each. An independent review of the code found five differences between A0 and A1 ([@tbl:arms]). The pre-registration named four of them and missed the residual scale.

: The arms, and the five respects in which A0 and A1 differ. {#tbl:arms}

| Factor | A0 (calibration window) | A1 (training half) |
|---|---|---|
| F, AR(8) fit data | fitting period only | all nominal training rows (out of fold for training scores) |
| σ, residual scale | out-of-sample residuals of an early-stopping period (3 months on M1) | in-sample residuals over the training half |
| P, calibration pool | the {{data.esa.m1.cal.n_steps:int}} steps ({{data.esa.m1.cal.months:g}} months) before the test half (M1) | out-of-fold nominal scores of the whole training half ({{data.esa.m1.train.n:int}} steps, M1) |
| D, selection data | the calibration window: {{data.esa.m1.events_target.cal.events}} target events (M1); {{data.esa.m2.events_target.cal.events}} rare events and no anomaly (M2) | both halves of the training half: {{data.esa.m1.events_all.train.targets}} target events (M1) |
| G, grid | $q \in$ {{cfg.explo.levels.conformal_bh}}, $k \in$ {{cfg.explo.ks}} | $q \in$ {{cfg.prereg.levels.conformal_bh}}, $k \in$ {{cfg.prereg.ks}} |

::: {custom-style="Table Note"}
Note: A2 and A3 share A1's protocol and use five seeds each; A0 and A1 are deterministic. F and σ change the test scores themselves; P, D and G change only the alarm layer. M1, M2: Missions 1 and 2.
:::

## Pre-registration, timeline and test exposure {#sec:prereg}

[@fig:timeline] shows the order of the study; all runs took place on 2026-10-24 and 2026-10-25. An exploratory phase produced 128 test-half result rows across missions, channel sets, encoders, alarm layers and seeds, with alarm layers selected on the calibration window. On Mission 1 that approach failed: AR(8) with the BH layer reached {{pre.m1.A0.conformal_bh.f05.mean:.3f}}, no deep or pretrained forecaster beat AR(8) under the same calibration, and telecommand inputs did not reduce rare-event alarms.

The pre-registration, written next, recorded what had been seen and fixed the arms, grids, seeds, stopping rule, minimum effect of interest, verdict rules and statistics. The document and {{prereg.frozen_code_files}} code files were hashed into a freeze record ({{prereg.freeze_id}}); the test function refuses to run if a frozen file has changed. At the freeze, A0's test value was known, A1's training-half selection had been computed and seed-0 training of A2 and A3 had started, but no test-half output of A1, A2 or A3 existed. After the confirmatory numbers existed, the deviation log was completed and the freeze re-issued ({{prereg.refreeze_id}}).

![**Order of the study.** The exploratory phase scored the official test halves before the pre-registered protocol existed. The confirmatory pass scored each new arm once on those halves, and the validation analyses were fixed by a second frozen plan before they were run.](artwork/Figure_1.png){#fig:timeline width=72%}

## Hypotheses and statistics {#sec:stats}

The hypotheses keep the identifiers of the registered document, in which H1–H3 belonged to the exploratory phase: H4 (A3 > A1), H5 (A3 > A2) and H6 (A1 > A0), all on Mission 1, the primary metric and the BH layer. With a minimum effect of interest of +0.05, a hypothesis is *supported* if its Holm-adjusted $p < 0.05$ [@holm1979] and its point estimate is at least +0.05; it is *refuted* if the upper end of its unadjusted 95 % interval lies below +0.05; otherwise it is *inconclusive*. Mission 2 repeats the comparisons without multiplicity correction and outside the confirmatory family. It was given that role before any arm was scored, because a parameter-free out-of-limit rule already scores {{explo.m2.full.ool.native.f05.mean:.3f}} there.

Intervals are percentile intervals from a paired bootstrap over fixed, non-overlapping 30-day blocks of the test half [@carlstein1986], {{data.esa.m1.test.blocks30}} on Mission 1 and {{data.esa.m2.test.blocks30}} on Mission 2, shared by both arms of a comparison; seeds are resampled hierarchically. We drew $B$ = {{conf.m1.h6.n_boot:int}} replicates with seed 20261024, and two-sided p-values are floored at $1/B$. The pre-registration calls the scheme a moving-block bootstrap [@kunsch1989]; the frozen code uses fixed blocks, and we report it as implemented. Because event-wise $F_{0.5}$ does not decompose over blocks, the frozen bootstrap used an approximate block-wise re-implementation, which agreed with the official point estimates to within {{conf.reimpl_gap_max:.4f}}. The validation analyses replace it with an exact decomposition (Appendix A).

## Validation analyses {#sec:posthoc}

After the confirmatory results existed, a second plan fixed eight further analyses, their code checks and their interpretation rules, and was frozen by hash ({{ph.plan_sha256}}, {{ph.plan_written}}) before any of them was run ([@tbl:posthoc]). They use the same test halves, so they are exploratory: the plan controls which analyses were run and how they are read, not what had been seen.

: Validation analyses of the second frozen plan. {#tbl:posthoc}

| Analysis | Question | Design | Decision rule |
|---|---|---|---|
| P1 decomposition | Which of the five differences produces the H6 contrast? | Ten cells, each a full protocol scored once; G, F, σ, P, D switched in turn from A0 to A1 (forward) and back (reverse) | *Material* if the step is ≥ +0.05 on both paths; *order-dependent* if the paths differ by > 0.05 |
| P2 forward chaining | Does the training-half protocol win in later periods? | Folds inside the training halves (M1: 2003–2006; M2: three quarters of 2001); W mirrors A0 and H mirrors A1, on earlier data only | Replicates if mean H − W ≥ +0.05 and H wins ≥ 3 of 4 (M1) or 2 of 3 (M2) folds |
| P3 exact bootstrap | Do verdicts depend on the approximate metric? | Exact block decomposition (Appendix A) | Agreement with official code < 10^−9^ |
| P4 event-tuned ACI | Is ACI's failure a tuning artefact? | $(\alpha, \gamma, k)$ selected on the event objective; $\gamma = 0$ allowed | "Not usable" only if $\gamma > 0$ is ≥ 0.05 below the threshold layer on every arm |
| P5 anomaly arm | Registered arm not run before | Anomaly log-odds of A2 and A3, anomaly-only objective | Descriptive |
| P6 seed extension | Are five seeds enough? | Five more seeds of A2 and A3 (M1) | Descriptive |
| P7 operations | What would an operator see? | Delay, run length, runs per detected event | Descriptive |
| P8 provenance | Do saved selections match the frozen rule? | Re-run the frozen selection for A1 | Identical choice and grid values |

::: {custom-style="Table Note"}
Note: W, calibration-window protocol (fit on the history minus its last 12 months on M1 or 6 months on M2; out-of-sample σ; pool and selection on the last 9 or 4.5 months; grid G0). H, training-history protocol (out-of-fold scores over the whole history; in-sample σ; cross-fitted selection; grid G1). G0 and G1 are A0's and A1's grids. The full plan is listed in Supplementary Section S1.
:::

In P1, the score levels are $S_0$ (A0's scores), $S_1$ (A1's scores) and an intermediate level that applies A0's residual scale to A1's forecast errors,
$$
s^{(1/2)}_{t,c} = s^{(1)}_{t,c}\,\frac{\max(\hat\sigma^{(1)}_c,\delta)}{\max(\hat\sigma^{(0)}_c,\delta)},
$$ {#eq:half}
where $\hat\sigma^{(1)}_c$ is the scale of the fold or final model that produced the score. The step from $S_0$ to $S_{1/2}$ isolates the fit data F, and the step from $S_{1/2}$ to $S_1$ isolates the residual scale σ. The code was accepted only if the first cell reproduced A0's official test value and the last reproduced A1's to within 10^−9^.

## Implementation {#sec:impl}

All runs used one server with {{env.cpu_count}} logical CPUs and Intel GPUs (PyTorch {{env.torch}} XPU back end, Python {{env.python}}); one Mission 1 TCN run (five fold models and a final model) took {{evm.m1.tcn.train_seconds_total.mean:.0f}} s, one XGBoost run {{evm.m1.gbt.train_seconds_total.mean:.0f}} s. Every number in this paper is inserted at build time from an evidence file that names its source result file. GPU kernels are not bit-deterministic, one reason seeds are reported as distributions.

# Results {#sec:results}

## Confirmatory tests {#sec:conf}

[@tbl:conf] gives the pre-registered comparisons. H6 is supported on Mission 1: every bootstrap replicate favoured A1, so the raw p-value sits at its floor. H4 and H5 are refuted, because neither improvement reaches the minimum effect. The family-wise analysis does not show that the TCN is worse: the H4 interval includes zero, and the H5 interval excludes zero but its raw p-value of {{conf.m1.h5.p_boot:.3f}} becomes {{conf.m1.h5.p_holm:.2f}} after Holm correction. The pre-registration states that a refuted H5 makes the trees the recommended event scorer; we follow that rule, adding that the trees did not significantly beat AR(8) ([@sec:supervised]). On Mission 2 the direction of H6 reverses ({{conf.m2.h6.diff_official:+.3f}}, unadjusted p-value at the bootstrap floor).

: Pre-registered comparisons on the official test halves (BH layer). {#tbl:conf}

| Mission | Test | A | B | Difference | 95 % CI | $p$ | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | H4: A3 − A1 | {{conf.m1.h4.A_official:.3f}} | {{conf.m1.h4.B_official:.3f}} | {{conf.m1.h4.diff_official:+.3f}} | [{{conf.m1.h4.ci95_low:+.3f}}, {{conf.m1.h4.ci95_high:+.3f}}] | {{conf.m1.h4.p_holm:.3f}} | {{conf.m1.h4.verdict}} |
| 1 | H5: A3 − A2 | {{conf.m1.h5.A_official:.3f}} | {{conf.m1.h5.B_official:.3f}} | {{conf.m1.h5.diff_official:+.3f}} | [{{conf.m1.h5.ci95_low:+.3f}}, {{conf.m1.h5.ci95_high:+.3f}}] | {{conf.m1.h5.p_holm:.3f}} | {{conf.m1.h5.verdict}} |
| 1 | H6: A1 − A0 | {{conf.m1.h6.A_official:.3f}} | {{conf.m1.h6.B_official:.3f}} | {{conf.m1.h6.diff_official:+.3f}} | [{{conf.m1.h6.ci95_low:+.3f}}, {{conf.m1.h6.ci95_high:+.3f}}] | ≤ {{conf.p_floor_holm:g}} | {{conf.m1.h6.verdict}} |
| 2 | H4: A3 − A1 | {{conf.m2.h4.A_official:.3f}} | {{conf.m2.h4.B_official:.3f}} | {{conf.m2.h4.diff_official:+.3f}} | [{{conf.m2.h4.ci95_low:+.3f}}, {{conf.m2.h4.ci95_high:+.3f}}] | {{conf.m2.h4.p_boot:.3f}} | replication |
| 2 | H5: A3 − A2 | {{conf.m2.h5.A_official:.3f}} | {{conf.m2.h5.B_official:.3f}} | {{conf.m2.h5.diff_official:+.3f}} | [{{conf.m2.h5.ci95_low:+.3f}}, {{conf.m2.h5.ci95_high:+.3f}}] | {{conf.m2.h5.p_boot:.3f}} | replication |
| 2 | H6: A1 − A0 | {{conf.m2.h6.A_official:.3f}} | {{conf.m2.h6.B_official:.3f}} | {{conf.m2.h6.diff_official:+.3f}} | [{{conf.m2.h6.ci95_low:+.3f}}, {{conf.m2.h6.ci95_high:+.3f}}] | ≤ {{conf.p_floor:g}} | replication |

::: {custom-style="Table Note"}
Note: A and B are official seed-mean $F_{0.5}$; intervals and p-values come from the frozen paired hierarchical block bootstrap. Mission 1 p-values are Holm-adjusted; Mission 2 p-values are unadjusted and outside the confirmatory family.
:::

**Exactness and provenance.** The exact decomposition (P3) matched the official code on {{ph.p3.n_checks}} stored arm–metric pairs, with a largest difference of {{ph.p3.max_abs_diff:.1e}}. With it, the H6 interval is [{{ph.p3.m1.h6.ci95_low:+.3f}}, {{ph.p3.m1.h6.ci95_high:+.3f}}], H5's Holm-adjusted p-value is {{ph.p3.m1.h5.p_holm:.3f}}, and all three verdicts are unchanged. Re-running the frozen selection for A1 (P8) reproduced the saved choice and every grid value on both missions (largest difference {{ph.p8.m1.max_abs_cv_diff:g}}).

## What produces the Mission 1 gain {#sec:decomp}

The first and last cells reproduced A0's and A1's official values exactly; [@fig:decomp] shows the steps. On Mission 1 with BH one factor carries the contrast, the data on which the alarm settings are selected. Selecting $(q,k)$ on the {{data.esa.m1.events_all.train.targets}} target events of the whole training half instead of the {{data.esa.m1.events_target.cal.events}} of the calibration window raised the score by {{ph.p1.m1.target.conformal_bh.forward.D.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.forward.D.ci95_high:+.3f}}] on the forward path and {{ph.p1.m1.target.conformal_bh.reverse.D.delta_official:+.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.D.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.D.ci95_high:+.3f}}] on the reverse path, and it is the only factor the pre-set rule calls material. The calibration pool added {{ph.p1.m1.target.conformal_bh.forward.P.delta_official:+.3f}} on both paths, an interval that excludes zero but an effect below the minimum of interest, and the AR(8) fit data added {{ph.p1.m1.target.conformal_bh.forward.F.delta_official:+.3f}}.

The residual scale had no effect on either path. This is expected: both layers compare a test score with nominal scores of the same channel, so a per-channel rescaling shared by pool and test scores cancels, and the scales of the fold and final models differ little. Neither of the two differences that change the scores ([@tbl:arms]) therefore explains the gain. The grid was order-dependent. It did nothing on the forward path, where the calibration-window selection chose a level present in both grids, but under whole-half selection the extended grid allowed a stricter level, and the score fell by {{ph.p1.m1.target.conformal_bh.reverse.G.delta_official:.3f}} [{{ph.p1.m1.target.conformal_bh.reverse.G.ci95_low:+.3f}}, {{ph.p1.m1.target.conformal_bh.reverse.G.ci95_high:+.3f}}]; with the original grid, whole-half selection would have scored {{ph.p1.m1.target.conformal_bh.r1.f05:.3f}}. Under the per-channel threshold, the pool and the selection data were both material ({{ph.p1.m1.target.threshold.forward.P.delta_official:+.3f}} and {{ph.p1.m1.target.threshold.forward.D.delta_official:+.3f}} on both paths), the fit data changed it by {{ph.p1.m1.target.threshold.forward.F.delta_official:+.3f}}, and grid and residual scale not at all ([S:p1-steps-m1-target-threshold]). The Mission 2 reversal has the same source: selecting on the training half rather than on the calibration window, whose {{data.esa.m2.events_target.cal.events}} events include no anomaly, lowered the score by {{ph.p1.m2.target.conformal_bh.forward.D.delta_official:.3f}} [{{ph.p1.m2.target.conformal_bh.forward.D.ci95_low:+.3f}}, {{ph.p1.m2.target.conformal_bh.forward.D.ci95_high:+.3f}}] on both paths, while the pool changed it by {{ph.p1.m2.target.conformal_bh.forward.P.delta_official:+.3f}} and the fit data by {{ph.p1.m2.target.conformal_bh.forward.F.delta_official:+.3f}}.

![**Decomposition of the training-half protocol's effect.** Change in test-half event-wise $F_{0.5}$ when one of the five differences between A0 and A1 is switched, on the forward path (A0 to A1) and the reverse path (A1 to A0), with exact block-bootstrap 95 % intervals. The dotted line marks the minimum effect of interest (+0.05).](artwork/Figure_2.png){#fig:decomp width=90%}

## Does the gain hold forward in time? {#sec:forward}

Forward chaining asks whether, at a given date and with only earlier data, the history protocol H would have beaten the window protocol W ([@fig:forward]). On Mission 1 the mean fold difference was {{ph.p2.m1.conformal_bh.mean_diff:+.3f}}, but H won only {{ph.p2.m1.conformal_bh.n_folds_H_better}} of {{ph.p2.m1.conformal_bh.n_folds}} folds, so by the pre-set rule the gain does not replicate. The mean is driven by 2005, when W chose a permissive setting that raised {{ph.p2.m1.conformal_bh.fold2.W_fa_yr:,.0f}} false alarms per year while H raised none (H − W = {{ph.p2.m1.conformal_bh.fold2.diff:+.3f}}). In 2004 W raised {{ph.p2.m1.conformal_bh.fold1.W_fa_yr:.0f}} false alarms per year and H none, but H also detected no event, so neither produced a useful score; in 2006 the two tied at {{ph.p2.m1.conformal_bh.fold3.H:.3f}}. Under the per-channel threshold the pattern was the same (mean {{ph.p2.m1.threshold.mean_diff:+.3f}}; H won {{ph.p2.m1.threshold.n_folds_H_better}} of {{ph.p2.m1.threshold.n_folds}}). On Mission 2, H won all {{ph.p2.m2.conformal_bh.n_folds}} folds but by a mean of only {{ph.p2.m2.conformal_bh.mean_diff:+.3f}}, below the minimum effect; inside the training half, then, the reversal seen on the Mission 2 test half did not appear.

![**Forward-chaining folds inside the training halves (conformal-BH layer).** Each fold is scored by the window protocol W and the history protocol H, using only data before the fold. Labels give the fold's event-wise $F_{0.5}$ and false alarms per year.](artwork/Figure_3.png){#fig:forward width=90%}

## How general is the protocol's gain? {#sec:general}

Across missions, channel sets and layers, {{seleff.n_positive}} of {{seleff.n}} contrasts favour the training-half protocol ([S:s-seleff]; on the lightweight subsets A0's counterpart is the exploratory lightweight AR(8) arm). The Mission 1 gain does not depend on BH: under the per-channel threshold it is {{seleff.m1.full.threshold:+.3f}}. On the lightweight Mission 1 subset the contrast is {{seleff.m1.lw.conformal_bh:+.3f}} under BH: the selection chose $q$ = {{sel.m1.A1lw.conformal_bh.q.mean:g}}, at the edge of the grid, and a cross-validated score of {{sel.m1.A1lw.conformal_bh.cv.mean:.3f}} became {{pre.m1.A1lw.conformal_bh.f05.mean:.3f}} on test. Its decomposition ([@fig:decomp]) mirrors the full-channel gain: the pool lowered the score by {{ph.p1.m1.lightweight.conformal_bh.forward.P.delta_official:.3f}} and {{ph.p1.m1.lightweight.conformal_bh.reverse.P.delta_official:.3f}} on the two paths and the selection data by {{ph.p1.m1.lightweight.conformal_bh.forward.D.delta_official:.3f}} ([S:p1-steps-m1-lightweight-conformal_bh]). On Mission 2 the contrast is negative only under BH ({{seleff.m2.full.conformal_bh:+.3f}}, against {{seleff.m2.full.threshold:+.3f}} under the threshold).

The per-channel threshold, which has no cross-channel multiplicity control, equalled or beat BH in {{layers.n_threshold_ge_bh}} of {{layers.n_rows}} arm–mission rows; BH was better only for the two supervised scorers on Mission 2. The rolling layer, whose $(q, k)$ were copied from the BH selection, is reported in [S:s-layers] and [S:s-layersfig].

## Operating points and operational behaviour {#sec:ops}

[@tbl:ops] reports what each arm would mean for an operator. A1's Mission 1 layer is strict ($q$ = {{sel.m1_target_ar8.conformal_bh.q:g}}, $k$ = {{sel.m1_target_ar8.conformal_bh.k:g}}): compared with A0 it cuts false alarms from {{pre.m1.A0.conformal_bh.fa_yr.mean:.1f}} to {{pre.m1.A1.conformal_bh.fa_yr.mean:.1f}} per year ({{ph.p7.m1.A1.false_alarms_per_1000_nominal_h:.2f}} per 1000 nominal hours) but detects fewer events, {{ph.p7.m1.A1.detected:.0f}} of {{ph.p7.m1.A1.n_events:.0f}} against {{ph.p7.m1.A0.detected:.0f}}, though quickly ({{ph.p7.m1.A1.detected_within_24h:.0%}} within 24 h). Its alarming precision of {{pre.m1.A1.conformal_bh.alarm_prec.mean:.3f}} means that most runs touching an event repeat an alarm on an event already detected, so the false-alarm count understates the alarms an operator must handle. The trees detect more than twice as many events as A1 at about seven times the false-alarm rate; the TCN raises the longest runs and the latest detections.

: Test-half operating points and operational behaviour (BH layer). {#tbl:ops}

| Arm | $F_{0.5}$ (sd) | Detected (anomalies) | FA / yr | Runs / yr | Delay (h) | Run (h) | Alarm prec. |
|---|---|---|---|---|---|---|---|
| M1 A0 | {{pre.m1.A0.conformal_bh.f05.mean:.3f}} | {{ph.p7.m1.A0.detected:.0f}} ({{ph.p7.m1.A0.detected_anomalies:.0f}}) | {{pre.m1.A0.conformal_bh.fa_yr.mean:.1f}} | {{pre.m1.A0.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m1.A0.delay_h_median:.2f}} | {{ph.p7.m1.A0.run_duration_h_median:.2f}} | {{pre.m1.A0.conformal_bh.alarm_prec.mean:.3f}} |
| M1 A1 | {{pre.m1.A1.conformal_bh.f05.mean:.3f}} | {{ph.p7.m1.A1.detected:.0f}} ({{ph.p7.m1.A1.detected_anomalies:.0f}}) | {{pre.m1.A1.conformal_bh.fa_yr.mean:.1f}} | {{pre.m1.A1.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m1.A1.delay_h_median:.2f}} | {{ph.p7.m1.A1.run_duration_h_median:.2f}} | {{pre.m1.A1.conformal_bh.alarm_prec.mean:.3f}} |
| M1 A2 | {{pre.m1.A2.conformal_bh.f05.mean:.3f}} ({{pre.m1.A2.conformal_bh.f05.std:.3f}}) | {{ph.p7.m1.A2.detected:.1f}} ({{ph.p7.m1.A2.detected_anomalies:.1f}}) | {{pre.m1.A2.conformal_bh.fa_yr.mean:.1f}} | {{pre.m1.A2.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m1.A2.delay_h_median:.2f}} | {{ph.p7.m1.A2.run_duration_h_median:.2f}} | {{pre.m1.A2.conformal_bh.alarm_prec.mean:.3f}} |
| M1 A3 | {{pre.m1.A3.conformal_bh.f05.mean:.3f}} ({{pre.m1.A3.conformal_bh.f05.std:.3f}}) | {{ph.p7.m1.A3.detected:.1f}} ({{ph.p7.m1.A3.detected_anomalies:.1f}}) | {{pre.m1.A3.conformal_bh.fa_yr.mean:.1f}} | {{pre.m1.A3.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m1.A3.delay_h_median:.2f}} | {{ph.p7.m1.A3.run_duration_h_median:.2f}} | {{pre.m1.A3.conformal_bh.alarm_prec.mean:.3f}} |
| M2 A0 | {{pre.m2.A0.conformal_bh.f05.mean:.3f}} | {{ph.p7.m2.A0.detected:.0f}} ({{ph.p7.m2.A0.detected_anomalies:.0f}}) | {{pre.m2.A0.conformal_bh.fa_yr.mean:.1f}} | {{pre.m2.A0.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m2.A0.delay_h_median:.2f}} | {{ph.p7.m2.A0.run_duration_h_median:.2f}} | {{pre.m2.A0.conformal_bh.alarm_prec.mean:.3f}} |
| M2 A1 | {{pre.m2.A1.conformal_bh.f05.mean:.3f}} | {{ph.p7.m2.A1.detected:.0f}} ({{ph.p7.m2.A1.detected_anomalies:.0f}}) | {{pre.m2.A1.conformal_bh.fa_yr.mean:.1f}} | {{pre.m2.A1.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m2.A1.delay_h_median:.2f}} | {{ph.p7.m2.A1.run_duration_h_median:.2f}} | {{pre.m2.A1.conformal_bh.alarm_prec.mean:.3f}} |
| M2 A2 | {{pre.m2.A2.conformal_bh.f05.mean:.3f}} ({{pre.m2.A2.conformal_bh.f05.std:.3f}}) | {{ph.p7.m2.A2.detected:.1f}} ({{ph.p7.m2.A2.detected_anomalies:.1f}}) | {{pre.m2.A2.conformal_bh.fa_yr.mean:.1f}} | {{pre.m2.A2.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m2.A2.delay_h_median:.2f}} | {{ph.p7.m2.A2.run_duration_h_median:.2f}} | {{pre.m2.A2.conformal_bh.alarm_prec.mean:.3f}} |
| M2 A3 | {{pre.m2.A3.conformal_bh.f05.mean:.3f}} ({{pre.m2.A3.conformal_bh.f05.std:.3f}}) | {{ph.p7.m2.A3.detected:.1f}} ({{ph.p7.m2.A3.detected_anomalies:.1f}}) | {{pre.m2.A3.conformal_bh.fa_yr.mean:.1f}} | {{pre.m2.A3.conformal_bh.alarm_runs_per_year:.0f}} | {{ph.p7.m2.A3.delay_h_median:.2f}} | {{ph.p7.m2.A3.run_duration_h_median:.2f}} | {{pre.m2.A3.conformal_bh.alarm_prec.mean:.3f}} |

::: {custom-style="Table Note"}
Note: Seed means for A2 and A3 (sd over five seeds). Detected: target events touched by an alarm run, of {{ph.p7.m1.A1.n_events:.0f}} (M1) or {{ph.p7.m2.A1.n_events:.0f}} (M2); anomalies in parentheses. FA, false alarms; runs, all alarm runs; delay and run, medians of first-alarm delay and run length (post hoc, P7; more in [S:s-p7]); alarm prec., alarming precision.
:::

## Point-level and event-tuned adaptive conformal inference {#sec:aci}

Tuned to a point-wise rate as pre-registered, ACI held its point-wise false-alarm rate on nominal test data between {{pre.all.aci.point_far.min:.4f}} and {{pre.all.aci.point_far.max:.4f}}, yet its event-wise $F_{0.5}$ never exceeded {{pre.all.aci.f05.max:.4f}} over {{pre.all.aci.n}} runs ([@tbl:aci]): the global alarm is an OR over channels that each fire on about one per cent of steps, giving up to {{pre.all.aci.alarm_runs.max:int}} alarm runs on a test half. With persistence and merging it reached a best seed mean of {{pre.all.aci_event.f05.max_seedmean:.3f}}, and {{pre.all.aci_event.zero_alarm_runs}} of {{pre.all.aci_event.n_runs}} runs raised no alarm at all. The update rule ([@eq:aci]) explains why. The pre-registered selection chose $\gamma$ = 0.05 in {{pre.all.aci.gamma005_runs}} of {{pre.all.aci.gamma_runs}} runs; with that value one exceedance moves $\alpha_t$ from {{cfg.prereg.aci_alpha}} to {{aci.alpha_after_one_hit.g0.05:.4f}}, and the channel cannot fire again for {{aci.recovery_steps.g0.05}} steps (Appendix B). A channel therefore cannot stay hot through an event, and persistence must be assembled from different channels firing in successive steps.

Tuned on the event objective (P4), ACI behaved differently. For A1 on Mission 1 the selection chose $\alpha$ = {{ph.p4.m1.ar8.gamma_positive.alpha:g}}, $\gamma$ = {{ph.p4.m1.ar8.gamma_positive.gamma:g}} and $k$ = {{ph.p4.m1.ar8.gamma_positive.k}}, and the test score was {{ph.p4.m1.ar8.gamma_positive.f05:.3f}}, above both the BH and the threshold layers on the same scores. With $\gamma$ = 0, which turns ACI into a fixed per-channel conformal threshold, the score was {{ph.p4.m1.ar8.gamma_zero.f05:.3f}}. Across the arms, event-tuned ACI with $\gamma > 0$ stayed 0.05 or more below the threshold layer on {{ph.p4.rule.n_below}} of {{ph.p4.rule.n_arms}} and reached or exceeded it on {{ph.p4.rule.n_at_or_above}}. Under the pre-set rule, therefore, the claim that ACI is unusable at the event level is not permitted. The collapse in the pre-registered comparison came from tuning $\gamma$ to a point-wise rate, although event-tuned ACI still trailed the simpler per-channel threshold on most arms.

: ACI against the event-level layers on identical scores (test-half event-wise $F_{0.5}$). {#tbl:aci}

| Arm | ACI, point-tuned | ACI + events, point-tuned | ACI, event-tuned, $\gamma>0$ | ACI, event-tuned, $\gamma=0$ | Threshold | Conformal-BH |
|---|---|---|---|---|---|---|
| M1 A1 | {{pre.m1.A1.aci.f05.mean:.4f}} | {{pre.m1.A1.aci_event.f05.mean:.3f}} | {{ph.p4.m1.ar8.gamma_positive.f05:.3f}} | {{ph.p4.m1.ar8.gamma_zero.f05:.3f}} | {{pre.m1.A1.threshold.f05.mean:.3f}} | {{pre.m1.A1.conformal_bh.f05.mean:.3f}} |
| M1 A2 | {{pre.m1.A2.aci.f05.mean:.4f}} | {{pre.m1.A2.aci_event.f05.mean:.3f}} | {{ph.p4.m1.gbt.gamma_positive.f05.mean:.3f}} | {{ph.p4.m1.gbt.gamma_zero.f05.mean:.3f}} | {{pre.m1.A2.threshold.f05.mean:.3f}} | {{pre.m1.A2.conformal_bh.f05.mean:.3f}} |
| M2 A1 | {{pre.m2.A1.aci.f05.mean:.4f}} | {{pre.m2.A1.aci_event.f05.mean:.3f}} | {{ph.p4.m2.ar8.gamma_positive.f05:.3f}} | {{ph.p4.m2.ar8.gamma_zero.f05:.3f}} | {{pre.m2.A1.threshold.f05.mean:.3f}} | {{pre.m2.A1.conformal_bh.f05.mean:.3f}} |

::: {custom-style="Table Note"}
Note: Point-tuned columns are pre-registered; event-tuned columns are post-hoc (P4), with $(\alpha,\gamma,k)$ selected by the frozen cross-fit rule on the event objective. A2 values are means over its {{ph.p4.m1.gbt.gamma_positive.f05.n}} seeds. Chosen settings per run are in [S:s-p4].
:::

## The supervised scorers {#sec:supervised}

**Seeds.** The supervised arms vary widely between seeds: with the five pre-registered seeds the trees range from {{pre.m1.A2.conformal_bh.f05.min:.3f}} to {{pre.m1.A2.conformal_bh.f05.max:.3f}} and the TCN from {{pre.m1.A3.conformal_bh.f05.min:.3f}} to {{pre.m1.A3.conformal_bh.f05.max:.3f}}. With the seed extension (P6) the trees averaged {{ph.p6.m1.A2.f05.mean:.3f}} (sd {{ph.p6.m1.A2.f05.std:.3f}}) over {{ph.p6.m1.A2.f05.n}} seeds and the TCN {{ph.p6.m1.A3.f05.mean:.3f}} (sd {{ph.p6.m1.A3.f05.std:.3f}}) over {{ph.p6.m1.A3.f05.n}} (the tenth TCN run was lost before scoring and not completed). With all seeds the TCN trailed AR(8) by {{ph.p6.m1.h4.diff:+.3f}} [{{ph.p6.m1.h4.ci95_low:+.3f}}, {{ph.p6.m1.h4.ci95_high:+.3f}}] and the trees by {{ph.p6.m1.h5.diff:+.3f}} [{{ph.p6.m1.h5.ci95_low:+.3f}}, {{ph.p6.m1.h5.ci95_high:+.3f}}], while the trees' lead over AR(8) was {{ph.p6.m1.x1.diff:+.3f}} [{{ph.p6.m1.x1.ci95_low:+.3f}}, {{ph.p6.m1.x1.ci95_high:+.3f}}] (exact bootstrap, unadjusted; [S:s-p6]).

**Optimistic cross-validation and shift.** Cross-validation over-estimated every Mission 1 arm (AR(8) {{sel.m1.A1.conformal_bh.cv.mean:.3f}} against {{pre.m1.A1.conformal_bh.f05.mean:.3f}} on test, trees {{sel.m1.A2.conformal_bh.cv.mean:.3f}} against {{pre.m1.A2.conformal_bh.f05.mean:.3f}}, TCN {{sel.m1.A3.conformal_bh.cv.mean:.3f}} against {{pre.m1.A3.conformal_bh.f05.mean:.3f}}). Nominal test samples exceeded the $1-10^{-4}$ quantile of their own out-of-fold pool {{shift.m1.a1_ar8.ratio:.1f}} times as often as in cross-validation for AR(8), {{shift.m1.A2.ratio_min:.1f}}–{{shift.m1.A2.ratio_max:.1f}} times for the trees and {{shift.m1.A3.ratio_min:.1f}}–{{shift.m1.A3.ratio_max:.0f}} times for the TCN (every fourth channel). The shift does not order the runs by test score (Spearman correlation {{shift.m1.supervised.spearman_f05:.2f}} over ten runs; [S:s-drift]).

**Where the trees' lead comes from.** The lead lies in rare nominal events, not anomalies. By recall the trees detect about {{evcls.m1.gain_a2_a1.recall_based:.0f}} more events per seed than AR(8): about {{evcls.m1.gain_a2_a1.seen.Rare_Event:.0f}} rare events of classes seen in training, {{evcls.m1.gain_a2_a1.unseen.Rare_Event:.0f}} of unseen classes and {{evcls.m1.gain_a2_a1.seen.Anomaly:.1f}} anomalies. No arm detected any of the {{evcls.m1.unseen.Anomaly.n}} Mission 1 test anomalies whose class is absent from training. The registered anomaly-score arm (P5) points the same way: selected and scored on anomalies only, the anomaly log-odds of the trees reached a mean anomaly-only $F_{0.5}$ of {{ph.p5.m1.A2.mean:.3f}} (range {{ph.p5.m1.A2.min:.3f}}–{{ph.p5.m1.A2.max:.3f}}) and those of the TCN {{ph.p5.m1.A3.mean:.3f}} on Mission 1, against {{ph.p5.m1.A1.mean:.3f}} for AR(8) selected on the same objective ([S:s-p5-m1]; on Mission 2, with six test anomalies, the trees reached {{ph.p5.m2.A2.mean:.3f}} and AR(8) {{ph.p5.m2.A1.mean:.3f}}). Two TCN ablations with three seeds each changed the score by less than its seed spread: dropping the forecast-residual inputs by {{xcomp.m1.x4.diff:+.3f}} [{{xcomp.m1.x4.ci95_low:+.3f}}, {{xcomp.m1.x4.ci95_high:+.3f}}], and using two classes instead of three by {{xcomp.m1.x5.diff:+.3f}} [{{xcomp.m1.x5.ci95_low:+.3f}}, {{xcomp.m1.x5.ci95_high:+.3f}}].

# Discussion {#sec:discussion}

## Interpretation {#sec:interp}

**What the evidence shows.** No more complex scoring model produced a gain of the kind the training-half protocol produced on Mission 1, and the decomposition narrows that protocol-level result to one mechanism, the labels on which the alarm settings are chosen. The fitted scores contributed nothing material, so the contrast is not an effect of better forecasting. A plausible reading is that nine events barely discriminate between settings: the calibration-window selection settled on a permissive level that raised {{ph.p1.m1.target.conformal_bh.c0.fa_yr:.0f}} false alarms per year, while the cross-fitted objective over {{data.esa.m1.events_all.train.targets}} events picked the strict setting that the test half rewarded.

**Why the gain does not travel.** The mechanism that helped on Mission 1 can also hurt. On the lightweight Mission 1 subset the training-half labels pushed the selection to the edge of the grid, and pool and selection data together cost more than 0.3. In forward chaining the history protocol chose strict settings in every fold; that removed the window protocol's false alarms, but in 2004 it also removed every detection. More labels make the choice of setting more stable. They do not make it right for a period whose events differ from the older ones. This reading is consistent with the results but was not tested directly.

**The deep scorer.** With all training labels and a receptive field of {{evm.receptive_field}} steps, the TCN showed no benefit over a linear forecaster or over trees on the same features. Distribution shift, too little labelled data for its capacity ({{evm.m1.params:int}} parameters against {{data.esa.m1.events_all.train.Anomaly}} training anomalies), seed instability and the absence of a tuning budget remain open explanations; the shift diagnostic alone does not separate good from bad runs. Doubling the seeds did not change the picture: the TCN stayed below both AR(8) and the trees, and the trees' lead over AR(8) stayed within seed noise.

## Relation to prior work {#sec:prior}

Our Mission 1 results exceed the benchmark's full-channel baselines [@kotowski2026esaadb], but those baselines did not tune their thresholds on training labels, whereas A1 selects its layer on them and A2 and A3 are supervised; on the lightweight subset the best published baseline ({{lit.dmlr.m1.lw.best:.3f}}) is far above anything we obtained. That labels used for threshold selection can matter more than the detector agrees with the benchmark authors' emphasis on thresholding. The pre-registered ACI result agrees with TC-CAD [@sahin2026tccad], and the event-tuned result shows that the collapse belongs to the tuning target rather than to adaptive conformal inference as such. Higher numbers reported elsewhere come from other protocols ([S:s-lit]) and do not establish a ranking against ours.

## Implications for practice {#sec:practice}

For teams working with telemetry like that of ESA-ADB Mission 1, the evidence supports four working rules. Select alarm settings on the event objective, using out-of-sample labelled history, and confirm the chosen setting on a later held-out period, since in our forward-chaining folds the advantage of more history was erratic. Report false alarms per year, alarm runs per year, alarming precision and detected events next to any $F$ score: A1's higher $F_{0.5}$ came with fewer detected events than A0, {{ph.p7.m1.A1.detected:.0f}} against {{ph.p7.m1.A0.detected:.0f}}. Tune any adaptive calibrator on the quantity the operator cares about, not on a point-wise rate. And do not assume that a deeper scorer will make up for a poor alarm protocol; in our study it did not, although a single untuned TCN cannot settle the question in general.

## Limitations {#sec:limits}

**Test exposure.** Both test halves had been scored in the exploratory phase, and A0's test value was known when the study was designed; the decision to study alarm selection on Mission 1 and the extended grid came after the exploratory results. The pre-registration controls which comparisons are confirmatory. It does not make the test halves untouched, and the validation analyses use the same halves.

**Confounding and statistics.** A0 and A1 differ in five respects. The decomposition attributes the contrast to factors along two orders only, and interactions between factors are not identified. Supervised-arm seeds are few relative to their spread, Mission 2 offers only {{data.esa.m2.test.blocks30}} blocks, and H6's raw p-value sits at the bootstrap's resolution floor. Forward-chaining folds hold 9–10 events on Mission 1, too few to estimate a fold's score precisely.

**Comparators and protocol.** Neither supervised scorer received a tuning budget, and only one deep architecture was tested. Out-of-fold models see later as well as earlier blocks, which makes cross-validated values optimistic. The freeze gate covers code, not intermediate outputs, and the parallel test driver was not frozen. The official "restore" step carries label information into test inputs. Two usable missions, one close to saturated, limit generalisation, and no result concerns real-time operation.

# Conclusions {#sec:conclusion}

Under the official ESA-ADB protocol at full channel scale, the largest improvement on Mission 1 came from the protocol for fitting AR(8) and selecting its event-level alarm layer, not from a more complex scoring model. A pre-specified decomposition traced that gain to the labels used to select the alarm settings, not to the fitted scores. A causal TCN trained on the labels showed no benefit over AR(8) or over gradient-boosted trees. Adaptive conformal calibration failed when tuned to a point-wise rate and competed when tuned on events. The gain was obtained on a test half used earlier for exploration, reversed on the lightweight Mission 1 subset and on Mission 2, and did not replicate reliably in forward-chaining folds. Alarm protocols deserve the same scrutiny as models, and the next test of this one must be on data this project has not used.

# Appendix A. Exact block-decomposable event-wise metric {.unnumbered}

The official metric treats each predicted alarm run as a half-open interval (closed at the end of the record) and each labelled event as a set of closed intervals, one per affected channel. An event is a true positive if any alarm run intersects any of its intervals, and an alarm run is a false positive if it touches no labelled interval of any category. Nominal time is the record span minus the union of all labelled intervals. For a partition of the test half into 30-day blocks $b = 1, \dots, B$, each event is assigned to the block containing its earliest labelled start and each false-positive run to the block containing its start, while nominal time and alarmed nominal time are clipped to the block boundaries. Each block then carries its true-positive count $t_b$, event count $n_b$, false-positive count $f_b$, nominal time $\nu_b$ and alarmed nominal time $a_b$. For a bootstrap resample $\mathcal B$ of blocks (with repetition),
$$
P_e(\mathcal B) = \frac{\sum_{b\in\mathcal B} t_b}{\sum_{b\in\mathcal B} (t_b + f_b)}\left(1 - \frac{\sum_{b\in\mathcal B} a_b}{\sum_{b\in\mathcal B} \nu_b}\right),
$$ {#eq:A1}
$$
R_e(\mathcal B) = \frac{\sum_{b\in\mathcal B} t_b}{\sum_{b\in\mathcal B} n_b},
$$ {#eq:A2}
and $F_{0.5}(\mathcal B)$ follows from [@eq:metric]. Times are measured in continuous time, as in the official code. With the identity resample, Eqs. (A.1) and (A.2) reproduce the official value, which we verified to within {{ph.p3.max_abs_diff:.1e}} on {{ph.p3.n_checks}} arm–metric pairs. The approximation used by the frozen bootstrap differed from Eqs. (A.1)–(A.2) in treating each event as one time span and computing the true-negative rate from sample counts.

# Appendix B. Recovery time of point-level ACI {.unnumbered}

Under [@eq:aci], a step with an exceedance changes the level by $\gamma(\alpha - 1)$ and a step without one by $\gamma\alpha$. Starting from $\alpha_t = \alpha$, one exceedance gives $\alpha_{t+1} = \alpha - \gamma(1-\alpha)$, which is negative whenever $\gamma > \alpha/(1-\alpha)$. While $\alpha_t \le 0$ the threshold is infinite and the channel cannot fire, so the number of steps until it can fire again is
$$
n_{\mathrm{rec}} = \left\lfloor \frac{\gamma(1-\alpha) - \alpha}{\gamma\alpha} \right\rfloor + 1.
$$ {#eq:B1}
With $\alpha$ = {{cfg.prereg.aci_alpha}} and $\gamma$ = 0.05, $n_{\mathrm{rec}}$ = {{aci.recovery_steps.g0.05}}. Persistence of $k \ge 3$ steps then requires different channels to fire in successive steps.

# Supplementary material {.unnumbered}

Supplementary_Material.docx lists the pre-registration documents (S1), results reported on ESA-ADB (S2), model settings and further results of the pre-registered arms (S3), all cells and steps of the validation analyses (S4), and code and data details (S5).

# Declarations {.unnumbered}

**CRediT authorship contribution statement.** [Author 1]: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization. [Author 2]: Supervision, Writing – review & editing. [To be completed by the authors.]

**Data availability.** The ESA Anomaly Dataset is openly available from Zenodo [@esaadb_dataset]. The derived data supporting this study, namely the out-of-fold and test scores, the alarm arrays, the result files and the evidence file behind every number in this paper, will be deposited in Zenodo with a DOI before publication [DOI to be added]. The code is available on GitHub [@cosmos_software]; the archived release will add the validation scripts, the latest deviation-log entries and the manuscript build.

**Funding.** [To be completed by the authors.]

**Declaration of competing interest.** [To be completed by the authors.]

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work the author(s) used Claude (Anthropic) in order to assist with writing analysis and plotting code and with drafting and editing the text. After using this tool, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article. All figures were rendered with Matplotlib from the study's result files; no figure was generated or altered by a generative AI model. [The authors must review this statement and adjust it to their actual use.]

**Acknowledgements.** [Optional.]

# References {.unnumbered}

::: {#refs}
:::
