---
title: "Beyond model complexity: a pre-registered evaluation of event-level alarm protocols for spacecraft telemetry anomaly detection"
author:
  - "[Author name(s)]^a^ · ^a^[Affiliation, postal address, country] · Corresponding author: [name, e-mail]"
abstract: |
  Event-level evaluation has made alarm handling central to anomaly detection in spacecraft telemetry. This study asked whether event-level performance on the ESA Anomaly Detection Benchmark (ESA-ADB) depends more on the protocol that fits a scoring model and turns its scores into alarms than on the model. In a pre-registered, hash-frozen study on the official split at full channel scale, an order-8 autoregressive forecaster (AR(8)), gradient-boosted trees and a temporal convolutional network (TCN) fed one event-level alarm layer of conformal p-values with Benjamini–Hochberg (BH) aggregation. On Mission 1, fitting AR(8) and selecting the alarm layer on the whole training half raised the corrected event-wise $F_{0.5}$ from 0.074 to 0.428 (difference +0.353; 95 % confidence interval +0.206 to +0.488; Holm-adjusted $p$ ≤ 0.0015), at about one false alarm per year. The TCN improved on neither AR(8) nor the trees. Validation analyses planned in advance attributed the gain to the labels used to select the alarm settings, not to the fitted scores. In forward-chaining folds inside the training halves it won only 2 of 4 Mission 1 folds, short of the replication criterion; the gain also reversed on a lightweight channel subset and, under BH, on Mission 2. Adaptive conformal inference failed when tuned to a point-wise rate but reached 0.472 when tuned on events. Both test halves had been used in an earlier exploratory phase. Alarm-protocol choices moved event-level scores more than model capacity did, but the gain depended on the setting.

  **Keywords:** Spacecraft telemetry; Anomaly detection; Event-wise evaluation; Conformal prediction; Alarm calibration; Pre-registration

  **Classification codes (Inspec):** C6170K; C1140Z; C5290; B0170N; E3650C
---

# Introduction {#sec:intro}

A flight operations team watches thousands of telemetry channels, and an automated detector earns its place only if its alarms are few and point at something real: a detector that raises hundreds of false alarms a year is soon ignored.

For this reason the scoring of anomaly detectors has moved from point-wise to event-wise metrics. Point-adjusted scores can make even random detectors look strong [@kim2022rigorous], several popular benchmarks are close to trivial or mislabelled [@wu2021flawed], and @sehili2023flawed proposed correcting event-wise precision with a time-based true-negative rate so that long alarms on nominal data are penalised. The ESA Anomaly Detection Benchmark (ESA-ADB) applies this view to real spacecraft telemetry annotated by operations engineers, with a chronological split and a corrected event-wise $F_{0.5}$ as its primary metric [@kotowski2026esaadb]. Its baselines score at most 0.061 on Mission 1 and 0.079 on Mission 2 on the full channel sets, and its authors trace the Mission 1 failures to false detections and stress that thresholding and post-processing matter.

Much of the method literature proposes stronger scorers instead: recurrent forecasters with dynamic thresholds [@hundman2018], variational models [@garcia2022dcvae], and transformer and pretrained forecasters used as anomaly scorers [@nie2023patchtst; @ansari2024chronos]. That raises a practical question. How much of event-level performance is decided by the protocol around the scorer (how it is fitted, which data form the calibration pool, on which labels the alarm settings are chosen), and how much by its capacity?

The objectives of this study were (i) to compare an unsupervised forecaster, gradient-boosted trees and a deep temporal convolutional network under one event-level alarm layer, on the official ESA-ADB protocol at full channel scale; (ii) to test whether a protocol that fits and selects on the whole training half improves on one that uses a short calibration window; and (iii) to establish, with analyses whose rules were fixed before they were run, which part of that protocol produces any gain and whether the gain holds forward in time. The confirmatory part was pre-registered and frozen by SHA-256 hashes before any new arm was scored on test data, and each new arm was scored on the test half once. The contributions are:

1. A confirmatory comparison of AR(8), XGBoost and a causal TCN under one event-level conformal alarm layer, with the minimum effect of interest and the family-wise correction fixed in advance. All three hypotheses are reported, including the two that were refuted.
2. Evidence that on Mission 1 the largest improvement came from the alarm protocol rather than from model capacity, a one-factor-at-a-time decomposition that locates the improvement, and a forward-chaining check of whether it carries over to later periods.
3. An account of point-level adaptive conformal inference at the event level: why it fails when tuned to a point-wise rate, and how it performs when tuned on events.
4. An exact block-decomposable form of the official metric that makes the block bootstrap exact, and an evidence file that traces every number in this paper to its source result file.

Two features of the study limit how its results may be read, and we report them as findings. The protocols compared in the main hypothesis differ in five respects, and both test halves had been scored in an exploratory phase before the pre-registration ([@sec:limits]).

# Related work {#sec:related}

**Evaluation and prior results.** Benchmark design for time-series anomaly detection has been criticised for labelling and triviality problems [@wu2021flawed], inflated scores under point adjustment [@kim2022rigorous] and protocols that overstate performance [@sehili2023flawed]. Range-based [@tatbul2018precision] and affiliation metrics [@huet2022local] address the fragmentation of events, and large evaluations find that no algorithm family dominates [@schmidl2022anomaly; @paparrizos2022tsbuad; @liu2024tsbad]. The studies that report results on ESA-ADB since the benchmark paper (Supplementary Table S1) differ in split, retraining, channel selection, preprocessing or metric. TC-CAD, a pre-registered study on 24 and 19 channels, found telecommand conditioning unhelpful [@sahin2026tccad]; @nguyen2026 compared supervised and unsupervised detectors with uncorrected metrics; @brancato2027ast, @alcarria2026 and @allegrini2026 report higher Mission 1 scores (0.902, 0.700 and 0.853) under a hierarchical protocol, monthly retraining and a Kaggle split, respectively; and @heydari2026 study threshold transfer under alert budgets. None of these numbers is comparable with ours.

**From scores to alarms.** Telemanom made dynamic thresholds with pruning practical on spacecraft data [@hundman2018], and extreme-value thresholds adapt to streaming data [@siffer2017]. Conformal anomaly detection turns a non-conformity score into a p-value against a calibration set [@laxhammar2010conformal; @laxhammar2015inductive]. Benjamini–Hochberg (BH) controls the false-discovery rate for independent [@benjamini1995] and positively dependent tests [@benjamini2001control], and @bates2023 showed that conformal p-values sharing one calibration set satisfy the required condition. Adaptive conformal inference (ACI) adjusts the miscoverage level online to keep long-run coverage under drift [@gibbs2021aci], and weighted variants relax exchangeability [@barber2023]. On ESA-ADB, TC-CAD found ACI the best calibrated of its point-level layers, yet it exceeded its point-wise target by 1.15–3.4× and collapsed the event score [@sahin2026tccad]. The present work moves calibration to the event level, selects the layer on the event metric over all training labels and compares scorer families under that one layer. On tabular problems, which our engineered per-step features resemble, gradient-boosted trees often remain strong against deep models [@grinsztajn2022tree].

# Materials and methods {#sec:methods}

## Data, split and metric {#sec:data}

**Data.** We use the ESA Anomaly Dataset, version 1.0 [@esaadb_dataset], Missions 1 and 2, on the official sampling grids of 30 s and 18 s. The official chronological split divides each mission into a training half and a test half. We use the full target channel sets of 58 and 47 channels; the lightweight subsets appear only in secondary analyses. On the target channels the Mission 1 training half holds 105 target events (63 anomalies and 42 rare nominal events). The Mission 1 test half, 7.0 years long, holds 55 anomalies and 36 rare events; the Mission 2 test half holds 6 anomalies and 356 rare events. Mission 3 is not used, because its authors advise against it for algorithm comparison [@kotowski2026esaadb].

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
where $\text{TNR}$ is one minus the share of nominal time covered by alarms. We also report anomaly-only $F_{0.5}$, false-alarm runs per year, all alarm runs per year and alarming precision, that is, true-positive events divided by true-positive events plus redundant detections (further alarm runs on an event already detected). The other pre-registered secondary metrics are in Supplementary Table S3 and Supplementary Table S4.

## Scoring models {#sec:models}

**AR(8).** Each channel is robust-normalised with the median and interquartile range of the nominal samples of a fitting period at the start of the training half. A ridge autoregressive model of order 8, fitted on at most 500,000 randomly sampled nominal rows, forecasts up to 16 steps ahead. The non-conformity score is the absolute forecast error divided by a robust residual scale $\hat\sigma_c$ of channel $c$, floored at $\delta$ = 0.05.

**Out-of-fold scores.** The training half is cut into five contiguous blocks. Each block is scored by a model fitted on the nominal samples of the other blocks, with a one-day purge on either side (2,880 steps on Mission 1). Test scores come from a model fitted on all nominal training rows.

**Supervised scorers.** Every channel and step receives seven causal features built from the normalised value and the out-of-fold AR(8) score: $\text{asinh}\,z$, $\log(1+s)$, its exponential moving averages over 1 h, 12 h and 3 d, the deviation from the trailing one-day mean and a forecast indicator. Arm A2 is a three-class XGBoost model [@chen2016xgboost] (nominal, anomaly, rare event) on these features plus trailing maxima of $\log(1+s)$ and the channel index. Arm A3 is a causal temporal convolutional network [@bai2018tcn] with dilated convolutions, a receptive field of 1021 steps, a learned channel embedding and cross-channel mixing (220,739 parameters on Mission 1), trained with AdamW [@loshchilov2019adamw], a one-cycle schedule [@smith2019superconvergence] and class-weighted cross-entropy. Both were trained out of fold on the same blocks as AR(8), without a hyperparameter search (Supplementary Table S2); their event score is $\log P(\text{anomaly}\cup\text{rare}) - \log P(\text{nominal})$.

## The event-level alarm layer {#sec:layer}

The pool $\mathcal C_c$ of channel $c$ holds nominal training scores of that channel. A test score $s$ receives the conformal p-value
$$
p_c(s) = \frac{1+\#\{s'\in\mathcal C_c: s'\ge s\}}{|\mathcal C_c|+1}.
$$ {#eq:p}
At every step, BH at level $q$ is applied across channels, and a channel whose p-value is rejected is *hot*. A global alarm starts once at least $k$ consecutive steps contain a hot channel. Alarm runs closer than 1 h are merged. The BH guarantee and the result of @bates2023 do not apply here, because the pools are per channel, the scores are correlated across channels, and telemetry is serially dependent and drifts. We therefore treat $q$ as a tuning parameter and report realised false-alarm rates.

**Selection.** $(q, k)$ is chosen by cross-fitting: calibrate on the nominal scores of one half of the training half, raise alarms on the other half and score them with the official metric on that half's labels, then swap the halves and average. Ties go to the smaller alarm fraction. The chosen layer is refitted on the whole training half and applied once to the test scores. The pre-registered grid is $q \in$ [0.01, 0.001, 0.0001, 1e-05] and $k \in$ [1, 3, 10, 30, 100].

**Comparators.** Three comparators were scored on identical scores. The *per-channel threshold* selects its tail fraction from [0.001, 0.0001, 1e-05, 0.0] by the same rule (0 means the maximum nominal score). The *rolling layer* refreshes its pools daily from the trailing 30 days of unlabelled test scores and reuses the $(q,k)$ selected for BH. *Point-level ACI* [@gibbs2021aci] updates its level as
$$
\alpha_{t+1} = \alpha_t + \gamma\left(\alpha - \mathbb 1[s_t > \hat Q_{1-\alpha_t}]\right),
$$ {#eq:aci}
with $\alpha$ = 0.01; the threshold $\hat Q_{1-\alpha_t}$ is infinite while $\alpha_t \le 0$. In the pre-registration $\gamma \in$ [0.001, 0.005, 0.01, 0.05] was chosen so that the point-wise false-alarm rate on the second training half came closest to $\alpha$, a rule adapted from TC-CAD [@sahin2026tccad]. ACI was scored without event logic and with the persistence and merging of the BH layer.

## Arms and the five differences between them {#sec:arms}

A0 is the AR(8) arm of the exploratory phase, with its alarm layer selected on a short calibration window just before the test half. A1 applies the protocol of [@sec:layer] to out-of-fold AR(8) scores; A2 and A3 apply it to the supervised scorers, with five seeds each. An independent review of the code found five differences between A0 and A1 ([@tbl:arms]). The pre-registration named four of them and missed the residual scale.

: The arms, and the five respects in which A0 and A1 differ. {#tbl:arms}

| Factor | A0 (calibration window) | A1 (training half) |
|---|---|---|
| F, AR(8) fit data | fitting period only | all nominal training rows (out of fold for training scores) |
| σ, residual scale | out-of-sample residuals of an early-stopping period (3 months on M1) | in-sample residuals over the training half |
| P, calibration pool | the 792,000 steps (9 months) before the test half (M1) | out-of-fold nominal scores of the whole training half (7,364,161 steps, M1) |
| D, selection data | the calibration window: 9 target events (M1); 43 rare events and no anomaly (M2) | both halves of the training half: 105 target events (M1) |
| G, grid | $q \in$ [0.01, 0.001, 0.0003, 0.0001], $k \in$ [1, 3, 10, 30] | $q \in$ [0.01, 0.001, 0.0001, 1e-05], $k \in$ [1, 3, 10, 30, 100] |

::: {custom-style="Table Note"}
Note: A2 and A3 share A1's protocol and use five seeds each; A0 and A1 are deterministic. F and σ change the test scores themselves; P, D and G change only the alarm layer. M1, M2: Missions 1 and 2.
:::

## Pre-registration, timeline and test exposure {#sec:prereg}

[@fig:timeline] shows the order of the study; all runs took place on 2026-10-24 and 2026-10-25. An exploratory phase produced 128 test-half result rows across missions, channel sets, encoders, alarm layers and seeds, with alarm layers selected on the calibration window. On Mission 1 that approach failed: AR(8) with the BH layer reached 0.074, no deep or pretrained forecaster beat AR(8) under the same calibration, and telecommand inputs did not reduce rare-event alarms.

The pre-registration, written next, recorded what had been seen and fixed the arms, grids, seeds, stopping rule, minimum effect of interest, verdict rules and statistics. The document and 27 code files were hashed into a freeze record (09a98f5e160e); the test function refuses to run if a frozen file has changed. At the freeze, A0's test value was known, A1's training-half selection had been computed and seed-0 training of A2 and A3 had started, but no test-half output of A1, A2 or A3 existed. After the confirmatory numbers existed, the deviation log was completed and the freeze re-issued (681cc7ddec86).

![**Order of the study.** The exploratory phase scored the official test halves before the pre-registered protocol existed. The confirmatory pass scored each new arm once on those halves, and the validation analyses were fixed by a second frozen plan before they were run.](artwork/Figure_1.png){#fig:timeline width=72%}

## Hypotheses and statistics {#sec:stats}

The hypotheses keep the identifiers of the registered document, in which H1–H3 belonged to the exploratory phase: H4 (A3 > A1), H5 (A3 > A2) and H6 (A1 > A0), all on Mission 1, the primary metric and the BH layer. With a minimum effect of interest of +0.05, a hypothesis is *supported* if its Holm-adjusted $p < 0.05$ [@holm1979] and its point estimate is at least +0.05; it is *refuted* if the upper end of its unadjusted 95 % interval lies below +0.05; otherwise it is *inconclusive*. Mission 2 repeats the comparisons without multiplicity correction and outside the confirmatory family. It was given that role before any arm was scored, because a parameter-free out-of-limit rule already scores 0.894 there.

Intervals are percentile intervals from a paired bootstrap over fixed, non-overlapping 30-day blocks of the test half [@carlstein1986], 86 on Mission 1 and 22 on Mission 2, shared by both arms of a comparison; seeds are resampled hierarchically. We drew $B$ = 2,000 replicates with seed 20261024, and two-sided p-values are floored at $1/B$. The pre-registration calls the scheme a moving-block bootstrap [@kunsch1989]; the frozen code uses fixed blocks, and we report it as implemented. Because event-wise $F_{0.5}$ does not decompose over blocks, the frozen bootstrap used an approximate block-wise re-implementation, which agreed with the official point estimates to within 0.0081. The validation analyses replace it with an exact decomposition (Appendix A).

## Validation analyses {#sec:posthoc}

After the confirmatory results existed, a second plan fixed eight further analyses, their code checks and their interpretation rules, and was frozen by hash (217648e6805f, 2026-10-25T12:14:37) before any of them was run ([@tbl:posthoc]). They use the same test halves, so they are exploratory: the plan controls which analyses were run and how they are read, not what had been seen.

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

All runs used one server with 512 logical CPUs and Intel GPUs (PyTorch 2.13.0+xpu XPU back end, Python 3.12.13); one Mission 1 TCN run (five fold models and a final model) took 6170 s, one XGBoost run 193 s. Every number in this paper is inserted at build time from an evidence file that names its source result file. GPU kernels are not bit-deterministic, one reason seeds are reported as distributions.

# Results {#sec:results}

## Confirmatory tests {#sec:conf}

[@tbl:conf] gives the pre-registered comparisons. H6 is supported on Mission 1: every bootstrap replicate favoured A1, so the raw p-value sits at its floor. H4 and H5 are refuted, because neither improvement reaches the minimum effect. The family-wise analysis does not show that the TCN is worse: the H4 interval includes zero, and the H5 interval excludes zero but its raw p-value of 0.035 becomes 0.07 after Holm correction. The pre-registration states that a refuted H5 makes the trees the recommended event scorer; we follow that rule, adding that the trees did not significantly beat AR(8) ([@sec:supervised]). On Mission 2 the direction of H6 reverses (-0.089, unadjusted p-value at the bootstrap floor).

: Pre-registered comparisons on the official test halves (BH layer). {#tbl:conf}

| Mission | Test | A | B | Difference | 95 % CI | $p$ | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | H4: A3 − A1 | 0.205 | 0.428 | -0.222 | [-0.395, +0.042] | 0.101 | refuted |
| 1 | H5: A3 − A2 | 0.205 | 0.544 | -0.338 | [-0.540, -0.033] | 0.070 | refuted |
| 1 | H6: A1 − A0 | 0.428 | 0.074 | +0.353 | [+0.206, +0.488] | ≤ 0.0015 | supported |
| 2 | H4: A3 − A1 | 0.875 | 0.874 | +0.001 | [-0.055, +0.054] | 0.907 | replication |
| 2 | H5: A3 − A2 | 0.875 | 0.904 | -0.029 | [-0.103, +0.037] | 0.382 | replication |
| 2 | H6: A1 − A0 | 0.874 | 0.962 | -0.089 | [-0.143, -0.049] | ≤ 0.0005 | replication |

::: {custom-style="Table Note"}
Note: A and B are official seed-mean $F_{0.5}$; intervals and p-values come from the frozen paired hierarchical block bootstrap. Mission 1 p-values are Holm-adjusted; Mission 2 p-values are unadjusted and outside the confirmatory family.
:::

**Exactness and provenance.** The exact decomposition (P3) matched the official code on 302 stored arm–metric pairs, with a largest difference of 2.1e-16. With it, the H6 interval is [+0.207, +0.491], H5's Holm-adjusted p-value is 0.066, and all three verdicts are unchanged. Re-running the frozen selection for A1 (P8) reproduced the saved choice and every grid value on both missions (largest difference 0).

## What produces the Mission 1 gain {#sec:decomp}

The first and last cells reproduced A0's and A1's official values exactly; [@fig:decomp] shows the steps. On Mission 1 with BH one factor carries the contrast, the data on which the alarm settings are selected. Selecting $(q,k)$ on the 105 target events of the whole training half instead of the 9 of the calibration window raised the score by +0.301 [+0.156, +0.436] on the forward path and +0.366 [+0.234, +0.500] on the reverse path, and it is the only factor the pre-set rule calls material. The calibration pool added +0.043 on both paths, an interval that excludes zero but an effect below the minimum of interest, and the AR(8) fit data added +0.010.

The residual scale had no effect on either path. This is expected: both layers compare a test score with nominal scores of the same channel, so a per-channel rescaling shared by pool and test scores cancels, and the scales of the fold and final models differ little. Neither of the two differences that change the scores ([@tbl:arms]) therefore explains the gain. The grid was order-dependent. It did nothing on the forward path, where the calibration-window selection chose a level present in both grids, but under whole-half selection the extended grid allowed a stricter level, and the score fell by -0.065 [-0.167, +0.014]; with the original grid, whole-half selection would have scored 0.493. Under the per-channel threshold, the pool and the selection data were both material (+0.145 and +0.197 on both paths), the fit data changed it by -0.003, and grid and residual scale not at all (Supplementary Table S11). The Mission 2 reversal has the same source: selecting on the training half rather than on the calibration window, whose 43 events include no anomaly, lowered the score by -0.088 [-0.134, -0.054] on both paths, while the pool changed it by -0.004 and the fit data by +0.003.

![**Decomposition of the training-half protocol's effect.** Change in test-half event-wise $F_{0.5}$ when one of the five differences between A0 and A1 is switched, on the forward path (A0 to A1) and the reverse path (A1 to A0), with exact block-bootstrap 95 % intervals. The dotted line marks the minimum effect of interest (+0.05).](artwork/Figure_2.png){#fig:decomp width=90%}

## Does the gain hold forward in time? {#sec:forward}

Forward chaining asks whether, at a given date and with only earlier data, the history protocol H would have beaten the window protocol W ([@fig:forward]). On Mission 1 the mean fold difference was +0.240, but H won only 2 of 4 folds, so by the pre-set rule the gain does not replicate. The mean is driven by 2005, when W chose a permissive setting that raised 1,173 false alarms per year while H raised none (H − W = +0.825). In 2004 W raised 271 false alarms per year and H none, but H also detected no event, so neither produced a useful score; in 2006 the two tied at 0.357. Under the per-channel threshold the pattern was the same (mean +0.084; H won 2 of 4). On Mission 2, H won all 3 folds but by a mean of only +0.017, below the minimum effect; inside the training half, then, the reversal seen on the Mission 2 test half did not appear.

![**Forward-chaining folds inside the training halves (conformal-BH layer).** Each fold is scored by the window protocol W and the history protocol H, using only data before the fold. Labels give the fold's event-wise $F_{0.5}$ and false alarms per year.](artwork/Figure_3.png){#fig:forward width=90%}

## How general is the protocol's gain? {#sec:general}

Across missions, channel sets and layers, 5 of 8 contrasts favour the training-half protocol (Supplementary Fig. S1; on the lightweight subsets A0's counterpart is the exploratory lightweight AR(8) arm). The Mission 1 gain does not depend on BH: under the per-channel threshold it is +0.340. On the lightweight Mission 1 subset the contrast is -0.383 under BH: the selection chose $q$ = 0.01, at the edge of the grid, and a cross-validated score of 0.528 became 0.025 on test. Its decomposition ([@fig:decomp]) mirrors the full-channel gain: the pool lowered the score by -0.150 and -0.223 on the two paths and the selection data by -0.166 (Supplementary Table S13). On Mission 2 the contrast is negative only under BH (-0.089, against +0.094 under the threshold).

The per-channel threshold, which has no cross-channel multiplicity control, equalled or beat BH in 8 of 10 arm–mission rows; BH was better only for the two supervised scorers on Mission 2. The rolling layer, whose $(q, k)$ were copied from the BH selection, is reported in Supplementary Table S5 and Supplementary Fig. S2.

## Operating points and operational behaviour {#sec:ops}

[@tbl:ops] reports what each arm would mean for an operator. A1's Mission 1 layer is strict ($q$ = 0.0001, $k$ = 10): compared with A0 it cuts false alarms from 91.6 to 1.1 per year (0.15 per 1000 nominal hours) but detects fewer events, 16 of 91 against 42, though quickly (88% within 24 h). Its alarming precision of 0.139 means that most runs touching an event repeat an alarm on an event already detected, so the false-alarm count understates the alarms an operator must handle. The trees detect more than twice as many events as A1 at about seven times the false-alarm rate; the TCN raises the longest runs and the latest detections.

: Test-half operating points and operational behaviour (BH layer). {#tbl:ops}

| Arm | $F_{0.5}$ (sd) | Detected (anomalies) | FA / yr | Runs / yr | Delay (h) | Run (h) | Alarm prec. |
|---|---|---|---|---|---|---|---|
| M1 A0 | 0.074 | 42 (17) | 91.6 | 113 | 0.35 | 0.04 | 0.276 |
| M1 A1 | 0.428 | 16 (8) | 1.1 | 18 | 0.33 | 0.32 | 0.139 |
| M1 A2 | 0.544 (0.236) | 37.8 (11.8) | 8.5 | 33 | 1.07 | 0.80 | 0.213 |
| M1 A3 | 0.205 (0.114) | 24.2 (11.8) | 20.9 | 30 | 2.02 | 5.05 | 0.326 |
| M2 A0 | 0.962 | 323 (1) | 3.4 | 205 | 0.16 | 0.24 | 0.895 |
| M2 A1 | 0.874 | 210 (0) | 0.0 | 136 | 0.16 | 0.20 | 0.857 |
| M2 A2 | 0.904 (0.050) | 257.2 (0.8) | 3.5 | 163 | 0.07 | 0.19 | 0.901 |
| M2 A3 | 0.875 (0.058) | 244.8 (0.8) | 7.7 | 162 | 0.27 | 0.30 | 0.885 |

::: {custom-style="Table Note"}
Note: Seed means for A2 and A3 (sd over five seeds). Detected: target events touched by an alarm run, of 91 (M1) or 362 (M2); anomalies in parentheses. FA, false alarms; runs, all alarm runs; delay and run, medians of first-alarm delay and run length (post hoc, P7; more in Supplementary Table S26); alarm prec., alarming precision.
:::

## Point-level and event-tuned adaptive conformal inference {#sec:aci}

Tuned to a point-wise rate as pre-registered, ACI held its point-wise false-alarm rate on nominal test data between 0.0098 and 0.0100, yet its event-wise $F_{0.5}$ never exceeded 0.0017 over 30 runs ([@tbl:aci]): the global alarm is an OR over channels that each fire on about one per cent of steps, giving up to 1,597,643 alarm runs on a test half. With persistence and merging it reached a best seed mean of 0.046, and 10 of 30 runs raised no alarm at all. The update rule ([@eq:aci]) explains why. The pre-registered selection chose $\gamma$ = 0.05 in 29 of 30 runs; with that value one exceedance moves $\alpha_t$ from 0.01 to -0.0395, and the channel cannot fire again for 80 steps (Appendix B). A channel therefore cannot stay hot through an event, and persistence must be assembled from different channels firing in successive steps.

Tuned on the event objective (P4), ACI behaved differently. For A1 on Mission 1 the selection chose $\alpha$ = 0.0001, $\gamma$ = 0.0001 and $k$ = 10, and the test score was 0.472, above both the BH and the threshold layers on the same scores. With $\gamma$ = 0, which turns ACI into a fixed per-channel conformal threshold, the score was 0.261. Across the arms, event-tuned ACI with $\gamma > 0$ stayed 0.05 or more below the threshold layer on 5 of 7 and reached or exceeded it on 2. Under the pre-set rule, therefore, the claim that ACI is unusable at the event level is not permitted. The collapse in the pre-registered comparison came from tuning $\gamma$ to a point-wise rate, although event-tuned ACI still trailed the simpler per-channel threshold on most arms.

: ACI against the event-level layers on identical scores (test-half event-wise $F_{0.5}$). {#tbl:aci}

| Arm | ACI, point-tuned | ACI + events, point-tuned | ACI, event-tuned, $\gamma>0$ | ACI, event-tuned, $\gamma=0$ | Threshold | Conformal-BH |
|---|---|---|---|---|---|---|
| M1 A1 | 0.0001 | 0.008 | 0.472 | 0.261 | 0.428 | 0.428 |
| M1 A2 | 0.0000 | 0.046 | 0.484 | 0.248 | 0.613 | 0.544 |
| M2 A1 | 0.0007 | 0.000 | 0.925 | 0.977 | 0.977 | 0.874 |

::: {custom-style="Table Note"}
Note: Point-tuned columns are pre-registered; event-tuned columns are post-hoc (P4), with $(\alpha,\gamma,k)$ selected by the frozen cross-fit rule on the event objective. A2 values are means over its 5 seeds. Chosen settings per run are in Supplementary Table S22.
:::

## The supervised scorers {#sec:supervised}

**Seeds.** The supervised arms vary widely between seeds: with the five pre-registered seeds the trees range from 0.157 to 0.769 and the TCN from 0.085 to 0.358. With the seed extension (P6) the trees averaged 0.488 (sd 0.195) over 10 seeds and the TCN 0.176 (sd 0.093) over 9 (the tenth TCN run was lost before scoring and not completed). With all seeds the TCN trailed AR(8) by -0.252 [-0.402, -0.043] and the trees by -0.313 [-0.457, -0.112], while the trees' lead over AR(8) was +0.061 [-0.138, +0.268] (exact bootstrap, unadjusted; Supplementary Table S25).

**Optimistic cross-validation and shift.** Cross-validation over-estimated every Mission 1 arm (AR(8) 0.624 against 0.428 on test, trees 0.727 against 0.544, TCN 0.536 against 0.205). Nominal test samples exceeded the $1-10^{-4}$ quantile of their own out-of-fold pool 0.8 times as often as in cross-validation for AR(8), 5.1–9.3 times for the trees and 2.8–54 times for the TCN (every fourth channel). The shift does not order the runs by test score (Spearman correlation -0.12 over ten runs; Supplementary Fig. S3).

**Where the trees' lead comes from.** The lead lies in rare nominal events, not anomalies. By recall the trees detect about 22 more events per seed than AR(8): about 11 rare events of classes seen in training, 7 of unseen classes and 2.6 anomalies. No arm detected any of the 3 Mission 1 test anomalies whose class is absent from training. The registered anomaly-score arm (P5) points the same way: selected and scored on anomalies only, the anomaly log-odds of the trees reached a mean anomaly-only $F_{0.5}$ of 0.087 (range 0.011–0.232) and those of the TCN 0.097 on Mission 1, against 0.336 for AR(8) selected on the same objective (Supplementary Table S23; on Mission 2, with six test anomalies, the trees reached 0.247 and AR(8) 0.000). Two TCN ablations with three seeds each changed the score by less than its seed spread: dropping the forecast-residual inputs by +0.039 [-0.188, +0.278], and using two classes instead of three by -0.012 [-0.260, +0.164].

# Discussion {#sec:discussion}

## Interpretation {#sec:interp}

**What the evidence shows.** No more complex scoring model produced a gain of the kind the training-half protocol produced on Mission 1, and the decomposition narrows that protocol-level result to one mechanism, the labels on which the alarm settings are chosen. The fitted scores contributed nothing material, so the contrast is not an effect of better forecasting. A plausible reading is that nine events barely discriminate between settings: the calibration-window selection settled on a permissive level that raised 92 false alarms per year, while the cross-fitted objective over 105 events picked the strict setting that the test half rewarded.

**Why the gain does not travel.** The mechanism that helped on Mission 1 can also hurt. On the lightweight Mission 1 subset the training-half labels pushed the selection to the edge of the grid, and pool and selection data together cost more than 0.3. In forward chaining the history protocol chose strict settings in every fold; that removed the window protocol's false alarms, but in 2004 it also removed every detection. More labels make the choice of setting more stable. They do not make it right for a period whose events differ from the older ones. This reading is consistent with the results but was not tested directly.

**The deep scorer.** With all training labels and a receptive field of 1021 steps, the TCN showed no benefit over a linear forecaster or over trees on the same features. Distribution shift, too little labelled data for its capacity (220,739 parameters against 63 training anomalies), seed instability and the absence of a tuning budget remain open explanations; the shift diagnostic alone does not separate good from bad runs. Doubling the seeds did not change the picture: the TCN stayed below both AR(8) and the trees, and the trees' lead over AR(8) stayed within seed noise.

## Relation to prior work {#sec:prior}

Our Mission 1 results exceed the benchmark's full-channel baselines [@kotowski2026esaadb], but those baselines did not tune their thresholds on training labels, whereas A1 selects its layer on them and A2 and A3 are supervised; on the lightweight subset the best published baseline (0.786) is far above anything we obtained. That labels used for threshold selection can matter more than the detector agrees with the benchmark authors' emphasis on thresholding. The pre-registered ACI result agrees with TC-CAD [@sahin2026tccad], and the event-tuned result shows that the collapse belongs to the tuning target rather than to adaptive conformal inference as such. Higher numbers reported elsewhere come from other protocols (Supplementary Table S1) and do not establish a ranking against ours.

## Implications for practice {#sec:practice}

For teams working with telemetry like that of ESA-ADB Mission 1, the evidence supports four working rules. Select alarm settings on the event objective, using out-of-sample labelled history, and confirm the chosen setting on a later held-out period, since in our forward-chaining folds the advantage of more history was erratic. Report false alarms per year, alarm runs per year, alarming precision and detected events next to any $F$ score: A1's higher $F_{0.5}$ came with fewer detected events than A0, 16 against 42. Tune any adaptive calibrator on the quantity the operator cares about, not on a point-wise rate. And do not assume that a deeper scorer will make up for a poor alarm protocol; in our study it did not, although a single untuned TCN cannot settle the question in general.

## Limitations {#sec:limits}

**Test exposure.** Both test halves had been scored in the exploratory phase, and A0's test value was known when the study was designed; the decision to study alarm selection on Mission 1 and the extended grid came after the exploratory results. The pre-registration controls which comparisons are confirmatory. It does not make the test halves untouched, and the validation analyses use the same halves.

**Confounding and statistics.** A0 and A1 differ in five respects. The decomposition attributes the contrast to factors along two orders only, and interactions between factors are not identified. Supervised-arm seeds are few relative to their spread, Mission 2 offers only 22 blocks, and H6's raw p-value sits at the bootstrap's resolution floor. Forward-chaining folds hold 9–10 events on Mission 1, too few to estimate a fold's score precisely.

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
and $F_{0.5}(\mathcal B)$ follows from [@eq:metric]. Times are measured in continuous time, as in the official code. With the identity resample, Eqs. (A.1) and (A.2) reproduce the official value, which we verified to within 2.1e-16 on 302 arm–metric pairs. The approximation used by the frozen bootstrap differed from Eqs. (A.1)–(A.2) in treating each event as one time span and computing the true-negative rate from sample counts.

# Appendix B. Recovery time of point-level ACI {.unnumbered}

Under [@eq:aci], a step with an exceedance changes the level by $\gamma(\alpha - 1)$ and a step without one by $\gamma\alpha$. Starting from $\alpha_t = \alpha$, one exceedance gives $\alpha_{t+1} = \alpha - \gamma(1-\alpha)$, which is negative whenever $\gamma > \alpha/(1-\alpha)$. While $\alpha_t \le 0$ the threshold is infinite and the channel cannot fire, so the number of steps until it can fire again is
$$
n_{\mathrm{rec}} = \left\lfloor \frac{\gamma(1-\alpha) - \alpha}{\gamma\alpha} \right\rfloor + 1.
$$ {#eq:B1}
With $\alpha$ = 0.01 and $\gamma$ = 0.05, $n_{\mathrm{rec}}$ = 80. Persistence of $k \ge 3$ steps then requires different channels to fire in successive steps.

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
