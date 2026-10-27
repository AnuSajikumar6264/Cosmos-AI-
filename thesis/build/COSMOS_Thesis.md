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

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Declaration {.unnumbered}

I declare that this thesis is my own work. It reports the software, experiments and results contained in the COSMOS project repository (`https://github.com/AnuSajikumar6264/Cosmos-AI-`, commit `2525367`, together with the result files listed in the evidence map in Appendix E). Where other work is used, it is cited. Datasets released by third parties are acknowledged in [@sec:datasets].

Signature: ____________________ Date: ____________________

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Acknowledgements {.unnumbered}

[To be written by the author.]

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

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

All 20 of its automated verification checks pass. 10 compare with external references (published tables, closed forms, physical limits, known answers); 9 are self-consistency checks; and one is mis-specified. Several of its coupling outputs are not used downstream, and the debris feature is not connected to the end-to-end chain; both are documented.

**The COSMOS-AI layer.** The layer adds checksum-verified data acquisition and models evaluated on five real public sources: the ESA Anomalies Dataset, OPSSAT-AD, SpotGEO, CelesTrak catalogue snapshots and ESA's list of recovered debris. Simulator-only results are labelled as such.

**The pre-registered study.** The principal study uses the ESA Anomaly Detection Benchmark (ESA-ADB) under its official split, on the full sets of 58 and 47 target channels. Its primary metric is the corrected event-wise $F_{0.5}$.

An exploratory phase established three findings:

- Calibrating per-channel conformal p-values with a Benjamini–Hochberg (BH) step across channels gave 0.962 on Mission 2.
- The best full-channel value published with the benchmark on Mission 2 is 0.079. However, a parameter-free out-of-limit rule already reaches 0.894 there, so Mission 2 is close to saturated.
- On Mission 1 the same layer reached only 0.074. Its settings had been chosen on a nine-month window containing 9 labelled events.

We then pre-registered three hypotheses and froze 28 code and protocol files by SHA-256 hash (freeze `09a98f5e160e`). Each arm was scored once on the test half. The arms use out-of-fold scores over the whole training half, which holds 109 annotated events on Mission 1. Either an unsupervised AR(8) forecaster or a supervised event scorer, trained out of fold, feeds the same event-level conformal alarm layer.

**Confirmatory outcomes on Mission 1:**

- The protocol that fits AR(8) and selects the alarm layer on the whole training half raised AR(8) from 0.074 to 0.428. The difference was +0.353 (95 % CI [+0.206, +0.488], Holm-adjusted *p* ≤ 0.0015). The arm raised 1.1 false alarms and 18 alarm runs in total per year, and it detected about one target event in six. The hypothesis was **supported**. The two arms differ in selection data, calibration pool, forecaster fitting period, residual scale and grid, so this is a comparison of two complete protocols. The gain was of similar size under a per-channel threshold layer (+0.340).
- A causal temporal convolutional event scorer reached 0.205 ± 0.114 over five seeds. Its point estimates were below the forecaster (-0.222) and below gradient-boosted trees on the same features (0.544 ± 0.236; -0.338). Both hypotheses were **refuted**: neither improvement reached the pre-registered minimum effect.
- On the near-saturated Mission 2 replication, whole-half selection was worse than the calibration window under the BH layer (-0.089). Under a per-channel threshold layer it was better (+0.094), so the reversal depends on the layer. On the lightweight Mission 1 channel subset, whole-half selection was worse under both layers.
- A point-level adaptive conformal comparator (ACI at $\alpha$ = 0.01) scored at most 0.002 on either mission. With the selected adaptation rate, its update rule silences a channel for many steps after each exceedance, so persistent alarms are rare and adding event logic did not repair it.
- An exploratory diagnostic found that the supervised scorers' nominal score distributions shift between training and test. Nominal test samples exceeded the TCN's calibration quantile 3–54 times more often than in cross-validation, against 0.8 times for AR(8). The shift is consistent with the deep scorer's weak result, but it does not explain the differences between seeds or models (rank correlation with test score -0.12 over ten supervised runs).

**Post-hoc validation.** Eight further analyses, fixed and frozen before they were run (on the same test halves), qualify these results. A one-factor-at-a-time decomposition attributed the Mission 1 gain to the labels used to select the alarm settings (+0.301 and +0.366 on the two paths), not to the fitted scores. In forward-chaining folds inside the training halves the protocol won only 2 of 4 Mission 1 folds, so the gain did not replicate by the pre-set rule. ACI tuned on the event metric reached 0.472, so its pre-registered collapse reflects the tuning target. More seeds did not change the H4 and H5 picture.

**Supporting studies.** These give the following results:

- *OPSSAT-AD:* a hybrid convolutional network reached F1 0.949 on the official split. The deep models' lead shrank under a time-blocked split.
- *SpotGEO:* a heat-map U-Net reached the official F1 0.746.
- *Re-entry within 68 days, from a leakage-checked catalogue landmark:* gradient-boosted trees outperformed a neural discrete-time hazard model, with AUC-PR 0.821 against 0.688. On uncontrolled objects at 30 days, the physics lifetime model scored highest.

Under operator-grade metrics, the alarm layer and the data used to select it mattered more than model capacity, but the selection effect was conditional on the mission and channel set. The thesis closes with the threats to validity, a register of known defects, and a roadmap towards a journal publication.

**Keywords:** spacecraft telemetry; anomaly detection; conformal prediction; false discovery rate; ESA-ADB; pre-registration; temporal convolutional networks; space debris; re-entry prediction; digital twin.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

::: {custom-style="TOC Heading"}
Contents
:::

```{=openxml}
<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \o "1-2" \h \z \u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Right-click and choose "Update Field" (or press F9) to build the table of contents.</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
```

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# List of Figures {.unnumbered}

::: {custom-style="List Entry"}
Figure 1. Package-level architecture of COSMOS at commit `2525367`.

Figure 2. Data flow of `run_pipeline` (`cosmos/pipeline.py`, l.73–122).

Figure 3. Functional map of COSMOS-AI.

Figure 4. Workflow of the pre-registered ESA-ADB study.

Figure 5. The causal TCN event scorer (arm A3).

Figure 6. The event-level conformal alarm layer shared by all arms of the pre-registered study.

Figure 7. Detection delay of the classical health monitor for every subsystem and fault type (SIMULATED telemetry, seed 7, 24-hour runs).

Figure 8. Exploratory study: corrected event-wise $F_{0.5}$ on the official test halves (full channel sets).

Figure 9. Pre-registered arms, conformal-BH layer.

Figure 10. Confirmatory and exploratory differences with 95 % bootstrap intervals.

Figure 11. Alarm burden of the pre-registered arms (conformal-BH layer, seed means).

Figure 12. Selection objective against test result for every arm with training-half selection (conformal-BH layer, seed means).

Figure 13. Training-half cross-validated $F_{0.5}$ over the selection grid of the conformal-BH layer (training labels only).

Figure 14. Test-half $F_{0.5}$ of the same scores under three alarm layers (seed means; values of [@tbl:layers]).

Figure 15. The selection contrast of [@tbl:seleff].

Figure 16. Exploratory: calibration shift against test-half $F_{0.5}$ for every supervised run on Mission 1 (conformal-BH layer).

Figure 17. Exploratory: nominal test exceedance rate of the $1-10^{-4}$ out-of-fold calibration quantile, relative to the training-half cross-validation rate (Mission 1, every fourth channel).

Figure 18. Primary $F_{0.5}$ (anomalies and rare events) against anomaly-only $F_{0.5}$ for the pre-registered arms (conformal-BH layer, seed means).

Figure 19. Exploratory: share of Mission 1 test events detected, by category and by whether the event class occurs in the training half (seed means, conformal-BH layer).

Figure 20. Post hoc (P1): change in test-half event-wise $F_{0.5}$ when one of the five differences between A0 and A1 is switched, forward (A0 to A1) and reverse (A1 to A0), with exact block-bootstrap 95 % intervals.

Figure 21. Post hoc (P2): forward-chaining folds, window protocol W against history protocol H (conformal-BH layer).

Figure 22. OPSSAT-AD F1 at threshold 0.5 under the official and time-blocked splits (5 seeds).

Figure 23. OPSSAT-AD label efficiency and robustness.

Figure 24. Vision results on SpotGEO and on the synthetic scenes.

Figure 25. Re-entry landmark: AUC-PR for re-entry within 30 and 68 days on the test launch groups.

Figure 26. Re-entry landmark: AUC-PR on all test objects and on the uncontrolled subset, at 30 and 68 days.
:::

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# List of Tables {.unnumbered}

::: {custom-style="List Entry"}
Table 1. Results reported on ESA-ADB after the benchmark paper.

Table 2. Implementation status of the components described in this thesis.

Table 3. Arms of the pre-registered study.

Table 4. Datasets used in this thesis.

Table 5. Labelled events on the target channels of ESA-ADB, by period (`data/DATASET_REPORT.json`).

Table 6. Evaluation protocols.

Table 7. Verification checks of the original pipeline (`tests/test_verify.py`; log `outputs/tests/test_verify_after.log`).

Table 8. Classical health monitor on simulated telemetry, seed 7: number of subsystems in which each fault type was detected within 24 h, and median detection delay.

Table 9. Failure-mode comparison through the whole chain (simulated, 600 Monte Carlo samples per scenario).

Table 10. Corrected event-wise $F_{0.5}$ of label-free baselines on the official test halves.

Table 11. Exploratory paired block-bootstrap comparisons (seed 0, 30-day blocks).

Table 12. Pre-registered arms on the official test halves, conformal-BH layer (mean ± sd over seeds; A0 and A1 are deterministic).

Table 13. Confirmatory comparisons (official seed-mean difference; 95 % interval and p from the paired hierarchical block bootstrap with 2,000 replicates).

Table 14. Test-half $F_{0.5}$ by alarm layer (seed means).

Table 15. Training-half selection minus calibration-window selection for AR(8) (A1 − A0; on the lightweight subsets, A1 minus the exploratory lightweight AR(8) arm).

Table 16. OPSSAT-AD test results (mean ± sd over 5 seeds where applicable; threshold 0.5).

Table 17. SpotGEO results on the full test set with the official scorer (mean ± sd over seeds where several exist).

Table 18. Re-entry landmark on the test launch groups (6,371 objects, 38 events).

Table 19. Defects in the code.

Table 20. Discrepancies between documentation and code or evidence.
:::

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

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

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```


# Introduction {#sec:intro}

## Background and motivation

A launch vehicle leaves material behind at every stage. Spent stages and fairings fall back through the atmosphere within minutes or days. Mission-related objects, rocket bodies and fragments from break-ups can remain in orbit for years or centuries. A newly operating satellite must be watched for faults. Telemetry is often the earliest evidence that something is wrong, and in some cases a fault precedes a fragmentation.

These three activities are usually treated by separate communities with separate tools:

- monitoring the health of an operating spacecraft;
- detecting and cataloguing objects in orbit;
- predicting when and where objects re-enter.

The growth of the catalogued population has raised concern about collision cascades [@kessler1978collision]. It has also increased the operational load of all three activities.

Telemetry monitoring is where machine learning has been studied most directly. Deep forecasting and reconstruction models for spacecraft telemetry have been proposed for several years [@hundman2018]. However, the benchmarks used to evaluate them have well-documented flaws. Several popular datasets contain trivially detectable or mislabelled anomalies [@wu2021flawed], and point-adjusted metrics inflate scores [@kim2022rigorous]. The European Space Agency Anomaly Detection Benchmark (ESA-ADB) was designed to address these problems [@kotowski2026esaadb]:

- it provides multi-year telemetry from three ESA missions, annotated by spacecraft operations engineers;
- it fixes an official training/test split;
- its primary metric is aligned with operator needs: an event-wise $F_{0.5}$ score whose precision is additionally penalised by the false-positive rate on nominal data.

Under this metric, on the full sets of monitored channels, no baseline reported with the benchmark exceeds 0.079. The benchmark's authors attribute much of this failure to thresholding and post-processing [@kotowski2026esaadb].

## The project as it stands

This thesis reports the COSMOS project as it exists in its repository at the time of writing (commit `2525367`). COSMOS has two layers.

**The original physics pipeline.** This is the Python package `cosmos` with the subpackages `health`, `debris` and `reentry`. It contains:

- a telemetry simulator and health monitor;
- a rule that maps the failing subsystem to break-up parameters;
- a debris catalogue and orbital-lifetime toolkit;
- a two-frame optical detection front end;
- a US Standard Atmosphere 1976 model, a drag model and a three-degree-of-freedom re-entry propagator;
- Monte Carlo footprints with conformal impact zones and a casualty-expectation estimate.

Its `run_pipeline` function chains simulated telemetry, health assessment, break-up, footprint and risk. The debris subpackage stands alongside this chain but is not part of it ([@sec:architecture]).

**The machine-learning layer, COSMOS-AI.** This is the package `cosmos.ai`. It imports but does not modify the original code. It adds:

- verified acquisition of public datasets;
- learned models for telemetry, imagery and re-entry;
- uncertainty quantification;
- an experiment record;
- a pre-registered confirmatory study on ESA-ADB.

An earlier project review proposed a different architecture. It described a physics-guided spatio-temporal graph transformer with low-rank adaptation (APSGTT–LoRA), probabilistic diffusion forecasting, federated learning and a dashboard. None of these components exists in the current code. This was verified by searching the source tree; Appendix C lists all discrepancies of this kind. The thesis therefore describes and evaluates only what is implemented. Several headline numbers from the earlier review are not reproduced by the current code, and [@sec:results-orig] reports the values the code now produces.

## Research problem

The central research problem is a practical one:

> On real spacecraft telemetry, evaluated with an operator-grade metric on the full sets of monitored channels, what actually determines whether an anomaly monitor produces few, mostly correct and timely alarms?

This matters because each candidate explanation leads to a different engineering response:

- the detector's capacity, which calls for a deeper model;
- the information the detector is given, for example telecommands;
- the way the detector's scores are turned into alarm events.

The other components of COSMOS raise related but secondary questions:

- Can a health state be coupled to re-entry physics in one chain?
- Do learned models outperform classical ones on real debris imagery and on catalogue-based re-entry prediction?
- Can a neural surrogate replace a physics propagator inside the chain?

## Research questions {#sec:rq}

**RQ1 (system).** Can satellite health, break-up, footprint and casualty risk be chained in one software pipeline whose physical components agree with independent references? What does the chain actually propagate from one stage to the next?

**RQ2 (primary).** On ESA-ADB under its official protocol, using the full channel sets, is event-wise performance improved more by a supervised deep event scorer or by selecting the alarm layer on all training labels? This is answered by pre-registered hypotheses H4–H6 ([@sec:prereg]).

**RQ3.** Does an event-level alarm layer outperform per-channel thresholds and point-level adaptive conformal calibration on the same scores? The layer combines conformal p-values, false-discovery-rate control across channels, persistence and merging.

**RQ4.** On the supporting real datasets (OPSSAT-AD, SpotGEO, and a re-entry landmark built from CelesTrak snapshots), do deep models outperform classical or tree-based models under leakage-controlled protocols?

**RQ5.** How closely does a learned surrogate reproduce the COSMOS re-entry propagator? How much faster is it, and how should it fall back to the propagator outside its training domain?

## Objectives

1. Document the current architecture accurately from the source code, including its limitations and defects.
2. Verify the physical components, against independent references where they exist, and characterise the end-to-end chain with fixed seeds.
3. Build a reproducible data layer that acquires, checks and records every dataset used, with checksums and manifests.
4. Implement and evaluate learned models on real data, under official or leakage-controlled protocols, against strong classical baselines.
5. Pre-register and execute a confirmatory study on ESA-ADB, and report every outcome, including refuted hypotheses.
6. Trace every reported number to a result file, and identify what further evidence would be required for journal publication.

## Scope and limits

The thesis evaluates software and models offline, on recorded or simulated data. It makes **no** claim of:

- real-time operation;
- deployment on a spacecraft or in a control centre;
- validated impact-location prediction. No recorded re-entry trajectories with recovery coordinates were available.

Results obtained only on simulated data are labelled *synthetic* or *simulated* wherever they appear. They say nothing about real spacecraft. Some data could not be obtained:

- the SPARK imagery is restricted;
- Space-Track and ESA DISCOS histories require accounts;
- ESA-ADB Mission 3 is declared unsuitable for benchmarking by its authors.

[@sec:datasets] records each of these cases.

## Contributions {#sec:contributions}

The following contributions are supported by the implementation and the evidence in this thesis:

1. **A pre-registered, hash-frozen confirmatory study on ESA-ADB at full channel scale ([@sec:results-prereg]).**
   - *Design.* Three hypotheses on Mission 1, a fixed seed count, a family-wise Holm correction and a minimum effect size were fixed before any test number of the new arms existed.
   - *Supported.* The protocol that selects the alarm layer on the whole training half (H6). Its gain holds under two alarm layers, but A0 and A1 differ in five respects (two of which change the scores), and the effect reverses on Mission 2 under BH and on the lightweight Mission 1 subset.
   - *Refuted.* A supervised deep event scorer improving on an AR(8) forecaster (H4) or on gradient-boosted trees (H5) by the minimum effect.
   - *Replication.* On Mission 2 the selection effect reversed under BH.
2. **An out-of-fold protocol for the official split ([@sec:oof]).** Every training sample is scored by a model that never saw it. Alarm selection and supervised training can therefore use all training labels without touching the test half.
3. **An encoder-agnostic, event-level conformal alarm layer, compared with simpler and point-level alternatives on identical scores ([@sec:alarm-layer]; [@sec:results-ablations]).** The comparison explains why point-level ACI gains little from event logic, and it shows that aggregation to events matters more than multiplicity control across channels.
4. **A reproduction of the official baselines and a new label-free reference.**
   - The official GlobalSTD detector reproduces the published values to within 0.001 in all eight configurations.
   - A parameter-free out-of-limit rule scores 0.894 on the full Mission 2 channel set. This shows that Mission 2 is close to saturated.
5. **Supporting real-data studies under leakage-controlled protocols ([@sec:results-support]).** These include a negative result for deep survival models on catalogue re-entry prediction.
6. **An integrated physics pipeline with an automated verification suite, an evidence-graded model registry, and a register of defects found during this work (Appendix C).**

The thesis does not claim that any component is the first of its kind. [@sec:literature] positions each contribution against the closest published work, including several studies from 2026 that address related questions.

## Thesis structure

| Chapter | Content |
|---|---|
| [@sec:literature] | Literature review |
| [@sec:theory] | Theoretical foundations: the methods and metrics used |
| [@sec:architecture] | The current architecture, taken from the source code |
| [@sec:methods] | Methodology and implementation of each study |
| [@sec:setup] | Experimental setup |
| [@sec:results] | Results |
| [@sec:discussion] | Discussion of the research questions |
| [@sec:limitations] | Limitations and future work |
| [@sec:conclusion] | Conclusion |

The appendices contain:

- the reproducibility instructions;
- the pre-registration and its deviation log;
- the defect register;
- the journal-readiness assessment;
- an automatically generated evidence map that links every number in the thesis to its source.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Literature Review {#sec:literature}

The review covers the literature for each component of COSMOS. It gives most space to spacecraft telemetry anomaly detection, where the principal study of this thesis sits. Each work is described only to the extent that its full text or abstract was read for this thesis. Where only a title or abstract was available, the description says no more than that source does.

## Anomaly detection in spacecraft telemetry

### Early data-driven monitoring

Data-driven monitoring of spacecraft housekeeping data predates deep learning.

- Yairi et al. combined probabilistic dimensionality reduction with clustering to monitor the telemetry of JAXA's SDS-4 small demonstration satellite [@yairi2017health].
- Ibrahim et al. compared machine-learning techniques for forecasting telemetry parameters of a low-Earth-orbit satellite [@ibrahim2019telemetry].
- General surveys of anomaly detection organise the field into statistical, distance-based, reconstruction-based and one-class methods [@chandola2009anomaly; @pang2022deep; @ruff2021unifying]. Deep one-class classification is one example of the last category [@ruff2018deepsvdd].

### Forecasting approaches and their benchmarks

The most cited deep approach for spacecraft telemetry is the LSTM forecaster of Hundman et al. [@hundman2018]. It predicts each channel ahead in time and flags windows whose smoothed prediction errors exceed a non-parametric dynamic threshold. It was evaluated on telemetry from NASA's SMAP and MSL missions. Related encoder–decoder reconstruction approaches had been proposed shortly before [@malhotra2016lstm].

Subsequent analyses have questioned the benchmarks on which such methods were compared.

- **Flawed datasets.** Wu and Keogh argued that several widely used time-series anomaly benchmarks contain anomalies that are trivially detectable, mislabelled, or placed at positions that reward particular biases. They concluded that such benchmarks "are creating the illusion of progress" [@wu2021flawed].
- **Inflated metrics.** Kim et al. showed that the point-adjustment protocol inflates detection scores: a whole anomalous segment counts as detected if any of its points is flagged. Under this protocol even random scores can appear competitive [@kim2022rigorous].
- **Large comparisons.** These have broadly found that simple and statistical methods are often competitive with deep models when the evaluation is done carefully [@schmidl2022anomaly; @paparrizos2022tsbuad; @liu2024tsbad]. TimeEval provides a common execution framework for such comparisons [@wenig2022timeeval].
- **Metric design.** Range-based precision and recall [@tatbul2018precision] and affiliation-based metrics [@huet2022local] were proposed to make event-level evaluation less sensitive to how events are fragmented. Sehili and Zhang proposed correcting event-wise precision by the false-positive rate on nominal time [@sehili2023flawed]. ESA-ADB adopts that correction.

### The ESA Anomaly Detection Benchmark

ESA-ADB [@kotowski2026esaadb] releases telemetry from three ESA missions. Spacecraft operations engineers annotated anomalies, rare nominal events and communication gaps, each with its affected channels. The benchmark fixes:

- per-mission sampling grids and preprocessing;
- a 50/50 chronological split for Missions 1 and 2. Mission 3 is declared unsuitable for algorithm comparison;
- "target" channel sets (58 and 47 channels) and smaller "lightweight" subsets;
- a hierarchy of metrics with the corrected event-wise $F_{0.5}$ as the primary score.

The benchmark paper reports results for a set of TimeEval algorithms, including Telemanom-ESA, DC-VAE-ESA [@garcia2022dcvae], isolation forests and the GlobalSTD rule. On the full channel sets the best primary scores were:

- 0.061 on Mission 1 (Telemanom-ESA-Pruned);
- 0.079 on Mission 2 (PCC).

On the lightweight subsets the best scores were 0.786 and 0.949. The authors attribute the poor Mission 1 results of the unsupervised algorithms to a massive number of false detections, caused by noise and varying sampling rates. They credit Telemanom's relative success to its dynamic thresholding and pruning, and conclude that proper thresholding and post-processing are important [@kotowski2026esaadb].

### Results reported on ESA-ADB since the benchmark

Several studies published during 2025–2026 bear directly on the questions of this thesis. [@tbl:lit-esa] summarises them.

: Results reported on ESA-ADB after the benchmark paper. The protocols differ, so the numbers are **not** directly comparable with each other or with this thesis. They are listed with their protocol. {#tbl:lit-esa}

| Work | Protocol | Method | Reported primary score |
|---|---|---|---|
| Kotowski et al. [@kotowski2026esaadb] | official split; full / lightweight channels | TimeEval baselines with detector-specific thresholds | M1 0.061, M2 0.079 (full) |
| Brancato et al. [@brancato2027ast] | "full ESA-ADB hierarchical protocol", Mission 1 only (abstract; full text not accessed) | label-free sliding minimum Euclidean distance (SMED); quantile Gaussian-process regression with adaptive retraining | M1 full: SMED 0.902, Q-GPR 0.883, out-of-limit reference 0.754 |
| Alcarria et al. [@alcarria2026] | non-official: monthly incremental retraining over the whole mission | LSTM autoencoder (M1), PCA + k-means + LOF (M2), adaptive extreme-value thresholds | M1 0.700, M2 0.698 |
| PitIA (as cited by Alcarria et al.) [@tejedor2025pitia] | "static train/test" | industrial unsupervised system | M1 0.424, M2 0.882 |
| Allegrini and Pompei [@allegrini2026] | Kaggle challenge on Mission 1, hidden test split; anomalies and rare events not separated | supervised hierarchical ensemble | 0.931 public, 0.853 private leaderboard |
| Nguyen et al. [@nguyen2026] | uncorrected event-wise metrics; lightweight and full channel sets of both missions; own preprocessing | supervised vs unsupervised benchmark | not comparable |
| Şahin, TC-CAD [@sahin2026tccad] | official code; 24 / 19 structurally chosen channels; shortest training windows; 16 seeds; pre-registered | telecommand conditioning; per-channel split, sliding and adaptive conformal thresholds; operator deferral | conditioning: negative; anomaly event-wise $F_{0.5}$ 0.090 with Telemanom's threshold, of order 1e-5 with point-level ACI |
| Heydari et al. [@heydari2026] | two missions, 77 shared channels, fixed scoring pipeline | cross-mission threshold transfer under alert-workload budgets | policy diagnostics (no detector comparison) |

Four observations follow from this body of work.

1. **Protocols vary widely.** Reported Mission 1 numbers range from below 0.1 to about 0.9, but they come from different splits, retraining regimes, channel sets and label treatments. Brancato et al. report 0.754 for an out-of-limit reference on Mission 1. [@sec:results-baselines] shows that an out-of-limit rule under the official protocol, as implemented here, scores far lower. The difference cannot be resolved without their implementation.
2. **Calibration and triage have been studied, at the point level.** The pre-registered TC-CAD study [@sahin2026tccad] has three relevant findings:
   - Conditioning a detector on the telecommand record did not reduce confusion between commanded events and anomalies.
   - Adaptive conformal thresholds per channel and per point were the best calibrated of their layers. They still exceeded the target point-wise false-alarm rate (by 1.15–3.4× according to the deposit), and they collapsed the event-wise score.
   - Two-threshold deferral under an operator budget removed 29-33 percentage points of automatic alarms on rare nominal events. On Mission 1 this test was post hoc.
3. **Workload-constrained alerting has been studied across missions** [@heydari2026].
4. **No work in this list combines** three elements:
   - event-level false-discovery-rate control across channels;
   - selection of the alarm layer on all training labels;
   - a pre-registered comparison of unsupervised, tree-based and deep supervised scorers under the official split at full channel scale.

   This gap is the one the principal study addresses. Its outcome ([@sec:results-prereg]) is reported whether favourable or not.

A further, related strand addresses distribution drift directly. MORA asks whether a local deviation is explained by broader temporal context by reconstructing the same target from paired short- and long-term views [@mou2026mora]. Brancato et al.'s earlier work proposed plug-and-play adaptive thresholds [@brancato2026phmap].

## Thresholds, conformal inference and false-alarm control

The step from scores to decisions has received less attention than the scorers themselves. The standard tools are:

- **Extreme-value thresholds.** SPOT fits a generalised Pareto distribution to the tail of the score distribution and updates it online [@siffer2017].
- **Telemanom's non-parametric dynamic threshold** [@hundman2018].

Conformal prediction provides distribution-free guarantees under exchangeability [@vovk2022alrw; @angelopoulos2023gentle; @lei2018distribution]. Conformalised quantile regression extends this to prediction intervals [@romano2019cqr]. For anomaly detection, conformal anomaly detection [@laxhammar2010conformal] and its inductive variant [@laxhammar2015inductive] convert a non-conformity score into a p-value against a calibration set. Benjamini and Hochberg proved false-discovery-rate control for independent tests [@benjamini1995]. Benjamini and Yekutieli extended it to positively regression-dependent tests [@benjamini2001control]. Bates et al. showed that conformal p-values of several test points, computed against one shared calibration set, satisfy that condition, so BH controls the false-discovery rate over those test points [@bates2023].

Exchangeability rarely holds for telemetry. Two lines of work relax it:

- weighted conformal prediction beyond exchangeability [@barber2023];
- adaptive conformal inference, which adjusts the level online to track a long-run error rate [@gibbs2021aci].

These guarantees concern *points*. The ESA-ADB metric concerns *events*, and the TC-CAD results show that point-level calibration does not translate into event-level performance [@sahin2026tccad].

## Deep and classical models for time series and tabular data

- **Temporal convolutional networks.** These match or exceed recurrent networks on many sequence tasks while remaining parallel and causal [@bai2018tcn].
- **PatchTST.** It tokenises each channel into patches and applies a transformer channel by channel [@nie2023patchtst].
- **Chronos.** It is a family of pretrained forecasting models applied zero-shot [@ansari2024chronos]. The Chronos-Bolt variant used here was released later by the same group, without a separate paper.
- **Trees on tabular data.** For tabular data, Grinsztajn et al. found that tree-based models still outperform deep networks on medium-sized datasets [@grinsztajn2022tree].
- **Gradient boosting.** It remains a standard strong baseline [@chen2016xgboost].
- **Uncertainty and explanations.** Deep ensembles provide a practical uncertainty estimate [@lakshminarayanan2017ensembles]. Calibration of modern networks is often poor [@guo2017calibration]. Gradient-based explanations such as Integrated Gradients [@sundararajan2017ig] should pass randomisation sanity checks before being used as evidence [@adebayo2018sanity].

## OPS-SAT and OPSSAT-AD

OPS-SAT was an ESA nanosatellite designed to demonstrate new satellite-control software under flight conditions [@evans2014opssat]. OPSSAT-AD is a dataset of labelled telemetry fragments from this mission, with an official train/test split, 18 handcrafted features and results for a broad set of baseline models [@ruszczak2025opssat]. The strongest reported model was a fully connected network on the features, with F1 0.946 and AUC-PR 0.979. AdaBoost reached F1 0.836.

## Optical detection of space objects

The SpotGEO challenge released 1280 training and 5120 test sequences of five frames from a low-cost ground-based telescope, labelled with the positions of geostationary and near-geostationary objects [@chen2021spotgeo]. Submissions were ranked on $1-F_1$ and a mean squared error on the full test set, after matching with the Hungarian algorithm [@kuhn1955hungarian]. The winning entry reached $1-F_1$ = 0.0517, i.e. $F_1$ ≈ 0.948. According to the challenge paper, the leading solutions post-processed detections over time, using line detection and trajectory filling [@chen2021spotgeo].

The point-detection formulation used in this thesis follows CenterNet [@zhou2019objects]. Its components are:

- a U-Net encoder–decoder [@ronneberger2015unet];
- or a ResNet-18 encoder [@he2016resnet];
- trained with the penalty-reduced variant [@law2018cornernet] of the focal loss [@lin2017focal].

Box detectors such as YOLO are a standard alternative for resolved objects [@redmon2016yolo].

## Debris environment and re-entry prediction

Collisions between catalogued objects can generate fragments faster than drag removes them, the cascade described by Kessler and Cour-Palais [@kessler1978collision]. ESA's annual space environment report tracks the catalogued population and re-entries [@esa2026environment].

**Orbital lifetime.** This is governed by atmospheric drag and therefore by solar activity. Classical treatments are given by King-Hele [@kinghele1987satellite] and Vallado [@vallado2013fundamentals]. Empirical atmosphere models such as NRLMSISE-00 represent the variation with solar and geomagnetic activity [@picone2002nrlmsise].

**Data-driven re-entry and orbit prediction.** Several approaches have been reported:

- Jung et al. trained a recurrent neural network to predict re-entry trajectories of uncontrolled objects [@jung2021reentry].
- Salmaso et al. developed a deep-learning model with feature engineering for the re-entry prediction of uncontrolled objects in low Earth orbit [@salmaso2023reentry].
- Peng and Bai used machine learning on publicly available data to improve orbit-prediction accuracy [@peng2020orbit].
- Shorten et al. documented pitfalls of wide-scale satellite-lifetime monitoring from public element sets and released a benchmark dataset [@shorten2023lifetimes].

**Survival analysis.** Censoring is treated by survival analysis:

- Cox's proportional-hazards model [@cox1972regression];
- discrete-time hazard models [@tutz2016discrete] and their neural variants [@kvamme2021continuous];
- accelerated-failure-time boosting [@barnwal2022aft].

Evaluation uses the concordance index [@harrell1982yield; @harrell1996multivariable] and the Brier score with censoring weights [@graf1999assessment; @brier1950verification]. Conformal methods for censored outcomes have been developed recently [@candes2023conformalized; @gui2024conformalized].

## Digital twins

The digital-twin concept is usually traced to Grieves' work on product lifecycle management [@grieves2017digital]. Glaessgen and Stargel articulated it for aerospace vehicles: a high-fidelity, probabilistic simulation mirroring the life of a specific vehicle [@glaessgen2012digital]. Tao et al. reviewed industrial practice [@tao2019digital]. COSMOS is described as a "digital twin" in its README. In the sense of these works it is not one: it does not ingest live data from a specific vehicle or update its state from them. The thesis uses the term only when quoting the project's own description.

## Research gap {#sec:gap}

Taken together, the literature establishes the following points.

1. **Operator-grade evaluation on real telemetry.** It is now possible through ESA-ADB, and full-channel results under the official protocol remain low for the published baselines [@kotowski2026esaadb].
2. **Thresholding.** The benchmark authors stress the role of false detections and of thresholding and post-processing [@kotowski2026esaadb]. Point-level conformal calibration does not by itself produce usable event-level alarms [@sahin2026tccad].
3. **Supervised learning.** Supervised approaches have been evaluated on a challenge split of Mission 1 [@allegrini2026]. Whether a supervised deep event scorer helps under the official split, compared with trees and an unsupervised forecaster on the same footing, has not been tested in a pre-registered design.
4. **Protocol heterogeneity.** It makes published ESA-ADB numbers hard to compare. Strong label-free references, such as an out-of-limit rule run through the same evaluation code, are rarely reported.

The principal study of this thesis addresses points 2–4 with a pre-registered design. The supporting studies address narrower gaps:

- deep versus classical models on OPSSAT-AD under a time-blocked split as well as the official one;
- the value of the original COSMOS front end on real GEO imagery;
- whether deep survival models improve on trees for catalogue-based re-entry prediction under a leakage-checked landmark design.

The integration of health, break-up and re-entry in one chain is an engineering contribution. Its scientific value depends on validation against real re-entries, which this thesis could not perform ([@sec:limitations]).


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Theoretical Foundations {#sec:theory}

This chapter defines the concepts, equations and metrics on which the implementation relies. Each definition is given in the form in which the code computes it. Where the code departs from a textbook form, the departure is stated, and the implementing module is named in brackets.

## Telemetry as multichannel time series

Spacecraft telemetry consists of many channels sampled irregularly, at different rates, with gaps. ESA-ADB defines a fixed sampling grid per mission (30 s for Mission 1 and 18 s for Mission 2). Each channel is mapped onto that grid by zero-order hold, i.e. the last observed value is carried forward [@kotowski2026esaadb]. Let $x_{t,c}$ denote channel $c$ at grid step $t$. All learned models in this thesis operate on a robust normalisation:
$$
z_{t,c} = \frac{x_{t,c} - \operatorname{med}_c}{\hat\sigma_c},
$$ {#eq:robust}
Both the median $\operatorname{med}_c$ and the scale $\hat\sigma_c$ are computed on nominal samples of a fitting period. The scale is $\hat\sigma_c = \mathrm{IQR}_c/1.349$, which equals the standard deviation for Gaussian data. Channels with zero interquartile range, such as discrete or zero-order-held channels, fall back to $(Q_{99}-Q_{01})/4.653$ and then to the standard deviation (`telemetry/forecasting.py`, `MissionData`). For monotonic counters, the official preprocessing replaces the value by its forward difference $x_{i+1}-x_i$ [@kotowski2026esaadb]. This introduces a one-sample look-ahead that the implementation inherits unchanged (Appendix C).

### Labels

Annotations in ESA-ADB are intervals attached to channels, grouped into events by an identifier. Each event belongs to one of three categories:

- an **anomaly**;
- a **rare nominal event**, which is unusual but expected behaviour such as a commanded manoeuvre;
- a **communication gap**.

The benchmark's primary task treats anomalies and rare nominal events together as targets. Communication gaps are ignored.

## Forecast residuals as non-conformity scores {#sec:nonconformity}

An unsupervised detector scores how surprising each new sample is relative to a model of nominal behaviour. COSMOS-AI uses forecasting models.

**Autoregressive forecaster.** For each channel, the forecaster fits a ridge-regularised autoregressive model of order 8 by least squares on nominal samples (penalty $10^{-3}$, intercept included):
$$
\hat z_{t,c} = \sum_{k=1}^{8} \phi_{k,c}\, z_{t-k,c} + \phi_{0,c}.
$$ {#eq:ar}
It is iterated over a horizon of $H=16$ steps from forecast origins spaced 16 steps apart. Every sample is therefore predicted between 1 and 16 steps ahead. The non-conformity score is the absolute forecast error divided by a robust residual scale:
$$
s_{t,c} = \frac{|z_{t,c} - \hat z_{t,c}|}{\max(\sigma^{r}_c, \delta)},\qquad
\sigma^{r}_c = \max\!\left(1.4826\,\operatorname{med}|r|,\ \frac{Q_{90}(|r|)}{1.645},\ \delta\right),\quad \delta = 0.05,
$$ {#eq:score-ar}
Here $r$ are the residuals on usable samples (`telemetry/oof.py`).

**Quantile forecasters.** These are the deep encoders of the exploratory study (TCN, PatchTST and Chronos-Bolt). They predict the quantiles $q_{0.25}, q_{0.5}, q_{0.75}$. The score is
$$
s_{t,c} = \frac{|z_{t,c}-q_{0.5}|}{\max(q_{0.75}-q_{0.25},\ \delta)} .
$$ {#eq:score-q}
Quantile models are trained with the pinball loss $\ell_\tau(y,\hat y) = \max(\tau d, (\tau-1)d)$, where $d = y-\hat y$, averaged over the quantile levels $\tau \in \{0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99\}$ (`telemetry/models.py`).

## Conformal p-values and false-discovery-rate control {#sec:theory-conformal}

A score becomes an alarm decision only after calibration. Let $\mathcal C_c$ be a calibration pool of $n_c$ scores of channel $c$ recorded on nominal data. The conformal p-value of a new score $s$ is
$$
p_c(s) = \frac{1 + \#\{s' \in \mathcal C_c : s' \ge s\}}{n_c + 1}.
$$ {#eq:conformal-p}
If the new nominal score and the pool are exchangeable, $p_c$ is super-uniform: $\Pr(p_c \le \alpha) \le \alpha$ for every $\alpha$ [@vovk2022alrw; @bates2023]. With $m$ channels tested at the same time step, the Benjamini–Hochberg (BH) procedure at level $q$ works as follows [@benjamini1995]:

1. Order the p-values, $p_{(1)} \le \dots \le p_{(m)}$.
2. Find the largest $j$ with $p_{(j)} \le jq/m$.
3. Reject the $j$ smallest.

For independent p-values, BH controls the false-discovery rate, the expected fraction of false rejections among the rejections, at $q\,m_0/m$, where $m_0$ is the number of true nulls [@benjamini1995]. Benjamini and Yekutieli extended the result to positively regression-dependent p-values [@benjamini2001control]. Bates et al. showed that conformal p-values of several test points computed against *one shared* calibration set satisfy that condition [@bates2023].

Three caveats apply to telemetry.

- **The pools are per channel.** COSMOS-AI applies BH across channels, each with its own calibration pool, and the channels' scores are correlated. This is not the shared-pool setting of Bates et al., so their result does not apply even under exchangeability.
- **Exchangeability fails.** Consecutive samples are serially dependent, and missions drift, so exchangeability between the pool and the test data does not hold. Weighted and adaptive variants of conformal inference address drift with weaker guarantees [@barber2023; @gibbs2021aci].
- **Finite pools bound the p-values.** The smallest attainable p-value is $1/(n_c+1)$. With BH at level $q$ over $m$ channels, a single channel can be rejected only if $1/(n_c+1) \le q/m$.

COSMOS-AI therefore treats $q$ as a tuning parameter selected on training data and reports realised false-alarm rates. It does not claim a guarantee.

The Simes combination $p_S = \min_j m\,p_{(j)}/j$ gives one global p-value per time step. It rejects exactly when BH rejects at least one hypothesis (`uncertainty.py`); the learned health monitor uses it ([@sec:methods-monitor]).

**Adaptive conformal inference (ACI)** adjusts the miscoverage level online [@gibbs2021aci]:
$$
\alpha_{t+1} = \alpha_t + \gamma\left(\alpha - \mathbb 1[s_t > \hat Q_{1-\alpha_t}(\mathcal C)]\right).
$$ {#eq:aci}
It keeps the long-run rate of exceedances near the target $\alpha$ even under drift. An exceedance lowers $\alpha_t$ by $\gamma(1-\alpha)$; once $\alpha_t < 0$ the quantile is infinite and the channel cannot fire until enough non-exceedances have raised $\alpha_t$ again. It is used here as the point-level comparator of [@sec:alarm-layer], following its use for spacecraft telemetry by Şahin [@sahin2026tccad].

## From hot samples to alarm events {#sec:theory-events}

Let $H_{t,c}\in\{0,1\}$ mark the samples that the calibrator flags ("hot cells"). Operators act on alarm *events*, not samples, so the implementation applies three rules (`telemetry/alarms.py`).

1. **Persistence.** A global alarm starts only after at least $k$ consecutive steps contain a hot channel. The alarm is placed from the step at which persistence is reached.
2. **Merging.** Alarm runs separated by fewer than $\Delta$ steps are joined. $\Delta$ corresponds to one hour.
3. **Attribution.** The channels reported for an alarm run are those hot at any step inside the run, including the $k-1$ steps that established persistence.

The global alarm is the logical OR over channels, as in the benchmark's own pipeline.

## The ESA-ADB metrics {#sec:theory-metrics}

The metrics are computed by the official ESA-ADB code, which is vendored unchanged in `telemetry/esa_metrics/`. Let the ground-truth events be intervals grouped by event identifier, and let the predicted events be the alarm runs.

- **True positive.** A ground-truth event is a true positive ($TP_e$) if at least one predicted event overlaps it. Otherwise it is a false negative ($FN_e$).
- **False positive.** A predicted event is a false positive ($FP_e$) if it overlaps no ground-truth interval.
- **Corrected event-wise precision.** The event-wise precision is multiplied by a time-based true-negative rate [@sehili2023flawed]:

$$
P_e = \frac{TP_e}{TP_e+FP_e}\cdot \mathrm{TNR},\qquad
\mathrm{TNR} = 1 - \frac{\text{duration of predicted alarms inside nominal time}}{\text{total nominal duration}},
$$ {#eq:pe}
$$
R_e = \frac{TP_e}{TP_e + FN_e},\qquad
F_{\beta} = \frac{(1+\beta^2) P_e R_e}{\beta^2 P_e + R_e},\quad \beta = 0.5 .
$$ {#eq:fbeta}

Setting $\beta=0.5$ weights precision twice as heavily as recall. The TNR factor penalises long alarms on nominal data even when they are few. Several secondary metrics are also reported:

- **Alarming precision:** $TP_e/(TP_e + \text{redundant detections})$.
- **ADTQC:** a score for detection timing.
- **Affiliation-based $F_{0.5}$:** a time-weighted precision and recall [@huet2022local].
- **Channel- and subsystem-aware $F_{0.5}$:** these score whether the attributed channels are the affected ones.

The primary metric of every ESA-ADB result in this thesis is the corrected event-wise $F_{0.5}$ over anomalies and rare nominal events. The *anomaly-only* variant counts only anomalies as targets.

## Temporal convolutional networks {#sec:theory-tcn}

A temporal convolutional network (TCN) stacks one-dimensional causal convolutions with exponentially increasing dilation [@bai2018tcn]. A causal convolution with kernel size $\kappa$ and dilation $d$ computes
$$
y_t = \sum_{i=0}^{\kappa-1} w_i\, x_{t - d i},
$$ {#eq:causal-conv}
so the output at $t$ depends only on inputs at or before $t$. A stack of $L$ residual blocks, each holding two convolutions with dilation $2^{\ell}$, has the receptive field
$$
\mathrm{RF} = 1 + 2(\kappa-1)\left(2^{L}-1\right).
$$ {#eq:rf}
For $\kappa = 3$ and $L = 8$ this is 1021 steps.

Group normalisation over time would mix future samples into past outputs. The implementation therefore normalises across feature channels at each time step only (`ChannelNorm`). The unit tests confirm that an input change at step $t$ leaves every earlier output bit-identical (`tests/test_ai.py`).

## Gradient-boosted trees

Gradient boosting fits an additive ensemble of regression trees. Each new tree approximately follows the negative gradient of the loss with respect to the current prediction. XGBoost adds second-order information and regularisation of tree complexity [@chen2016xgboost]. For multi-class problems it optimises the softmax cross-entropy. Trees are invariant to monotone feature transformations and robust to heterogeneous feature scales, and they remain strong baselines on tabular data [@grinsztajn2022tree].

## Statistical comparison {#sec:theory-stats}

Test-set differences between two detectors are uncertain in two ways. The test half is one realisation of a serially dependent process, and stochastic models vary between seeds.

- **Dependence in time.** Block bootstraps resample contiguous blocks of time, which preserves dependence within each block [@kunsch1989]. The implementation uses fixed, non-overlapping 30-day blocks rather than Künsch's moving blocks: 86 on the Mission 1 test half and 22 on Mission 2. For each replicate it draws blocks with replacement, using the same blocks for both arms of a comparison.
- **Seed variation.** Each seeded arm's statistic is the mean over seeds drawn with replacement, a hierarchical bootstrap.

The event-wise $F_{0.5}$ cannot be written as a sum over blocks, so it cannot be resampled directly. The bootstrap therefore uses a block-decomposable re-implementation that tracks:

- detected events by block;
- false-positive alarm runs by block;
- nominal and alarmed nominal time by block.

The official metric gives the point estimate. The re-implementation agrees with it to within 0.0081 on every confirmatory comparison ([@sec:results-prereg]).

The two-sided bootstrap p-value is $2\min\{\Pr(\Delta \le 0), \Pr(\Delta \ge 0)\}$, floored at $1/B$. With $B$ = 2,000 the smallest reportable value is 0.0005. The Holm step-down procedure controls the family-wise error rate over the pre-registered family [@holm1979].

## Optical detection of moving objects {#sec:theory-vision}

### Two-frame detection

The original COSMOS front end detects moving objects in two steps.

1. **Registration.** The two frames are registered by phase correlation. The normalised cross-power spectrum $F_a F_b^{*}/|F_a F_b^{*}|$ has an inverse transform that peaks at the shift between the frames. The peak is refined to sub-pixel precision with a three-point parabola.
2. **Detection.** The registered difference is searched with a difference-of-Gaussians (DoG) filter at three scales.
   - Detections are thresholded against a photon-noise map, since pixels under bright stars are intrinsically noisier.
   - Negative lobes ("dipoles") left by the object's old position are rejected (`debris/detect.py`).

### Learned point detection

COSMOS-AI follows the "objects as points" formulation [@zhou2019objects]. A network predicts a heat map whose peaks mark object centres, together with sub-pixel offsets. Training uses the penalty-reduced focal loss of CornerNet [@law2018cornernet], with $\alpha = 2$ and $\beta = 4$ as in [@zhou2019objects], applied around Gaussian targets with $\sigma = 1.5$ px. It modifies the focal loss of Lin et al. [@lin2017focal]. The backbone is a U-Net [@ronneberger2015unet], or a ResNet-18 encoder [@he2016resnet].

### SpotGEO scoring

The SpotGEO challenge scores each frame separately [@chen2021spotgeo].

1. Predicted and true points are assigned by the Hungarian algorithm [@kuhn1955hungarian].
2. A match within $\tau = 10$ px is a true positive. Unmatched or distant points are false negatives or false positives.
3. The per-frame error sums the matched distances (zero below $\varepsilon = 3$ px) and adds $\tau^2$ for every unmatched point.

The values $\tau$ and $\varepsilon$ are those of the official toolkit (`vision/spotgeo_validation.py`, vendored unchanged). The challenge ranks by $1-F_1$ and uses the mean squared error only to break ties. The toolkit's MSE is the *sum* of the per-sequence MSE values, not their mean.

## Survival analysis for re-entry {#sec:theory-survival}

Predicting *whether an object re-enters within a horizon* is a time-to-event problem with censoring. Objects that have not yet re-entered by the end of follow-up are censored. A landmark design fixes a prediction date $\tau$ and uses only information available at $\tau$. It then follows each object until a common end date, which makes censoring administrative and identical for all objects.

**Discrete-time hazard model.** The follow-up is divided into bins. A network outputs a hazard $h_k(\mathbf x)$ for each bin, and the survival function is
$$
S_k(\mathbf x) = \prod_{j \le k}\bigl(1 - h_j(\mathbf x)\bigr) .
$$ {#eq:discrete-hazard}
Training maximises the Bernoulli likelihood of the event indicator over the bins in which the object is at risk [@tutz2016discrete; @kvamme2021continuous].

**Accelerated failure time (AFT) model.** The AFT model regresses $\log T$ on the features with a parametric error distribution. XGBoost implements it with interval-censored labels [@barnwal2022aft].

**Evaluation.** Five measures are used:

- the area under the precision–recall curve (AUC-PR) for "re-entry within $h$ days";
- Harrell's concordance index [@harrell1982yield];
- the Brier score [@brier1950verification];
- the expected calibration error;
- conformal prediction sets for the binary outcome.

The conformal sets come in two forms. *Marginal* sets use a single quantile of the non-conformity $1-\hat p(y)$. *Mondrian* (label-conditional) sets calibrate a separate quantile for each class, which gives per-class coverage [@vovk2022alrw; @angelopoulos2023gentle].

## Physics of decay and re-entry {#sec:theory-physics}

**Atmosphere.** The US Standard Atmosphere 1976 (US76) defines temperature profiles in seven layers, with boundaries in geopotential altitude up to 84.852 km (86 km geometric), and hydrostatic pressure and density. Above 86 km the implementation interpolates a density table log-linearly up to 1000 km (`reentry/atmosphere.py`). No solar or geomagnetic activity is modelled. Empirical models such as NRLMSISE-00 represent these effects [@picone2002nrlmsise], and their absence is a stated limitation.

**Drag.** Drag acceleration is $\mathbf a_D = -\tfrac12 \rho\, |\mathbf v_{r}|\, C_D \frac{A}{m} \mathbf v_{r}$, where $\mathbf v_r$ is the velocity relative to the atmosphere. The ballistic coefficient is $B = m/(C_D A)$. The drag coefficient blends a continuum curve in the Mach number with the free-molecular limit 2.10, using a logistic weight in $\log_{10}$ of the Knudsen number (`reentry/drag.py`).

**Equations of motion.** In an Earth-fixed rotating frame with angular velocity $\boldsymbol\omega$:
$$
\ddot{\mathbf r} = \mathbf g_{J_2}(\mathbf r) - 2\boldsymbol\omega\times\dot{\mathbf r} - \boldsymbol\omega\times(\boldsymbol\omega\times\mathbf r) + \mathbf a_D ,
$$ {#eq:eom}
where $\mathbf g_{J_2}$ is point-mass gravity plus the $J_2$ oblateness term [@vallado2013fundamentals].

**Orbital lifetime.** For a near-circular orbit, the semi-major axis decays as $\dot a = -C_D (A/m)\,\rho(h)\sqrt{\mu a}$. The implementation integrates this in 5 km steps from the initial altitude down to 120 km (`debris/lifetime.py`).

**Casualty expectation.** For one fragment with debris area $A_d$, the casualty area is $A_c = (\sqrt{A_d} + 0.6)^2$. The 0.6 m is the square root of the 0.36 m² cross-section assumed for a person (`PERSON_AREA_M2`, `risk.py`, l.27). The expected number of casualties is $E_c = \rho_{\text{pop}}\,A_c$, summed over fragments that carry more than 15 J of kinetic energy. The commonly applied acceptability threshold is $10^{-4}$ per re-entry event. Appendix C notes that the pipeline compares a *per-fragment* value with this per-event threshold.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Current System Architecture {#sec:architecture}

This chapter documents the architecture as implemented at commit `2525367`. It was established in three ways:

- by reading every module;
- by running the end-to-end pipeline with fixed seeds;
- by comparing the original package with the snapshot taken before the machine-learning layer was added (`outputs/baseline/cosmos_baseline_src_2026-10-24.tar.gz`).

Every original module except `cosmos/pipeline.py` is byte-identical to that snapshot. `pipeline.py` gained two opt-in switches. Constants are quoted with the file and line where they are defined. Numbers produced by running the code come from the evidence file (Appendix E).

## Overview and design principles

COSMOS consists of two layers ([@fig:overall]).

- **The original package `cosmos`.** It holds 26 Python files and 5,317 lines, in three feature subpackages (`health`, `debris`, `reentry`) and shared modules (`common.py`, `pipeline.py`, `plots.py`, `datasets.py`).
- **The machine-learning layer `cosmos.ai`.** It holds 38 files and 8,749 lines, excluding the vendored official ESA-ADB metric code.

Three design rules govern how the layers interact.

1. **`cosmos.ai` imports the original package but never edits it.** The original behaviour is regression-tested: `test_pipeline_backward_compatible` runs the untouched baseline code in a subprocess and requires an identical pipeline summary.
2. **Learned components enter the end-to-end chain only through opt-in switches.** `run_pipeline(backend="learned")` replaces the health monitor. `run_pipeline(footprint_backend="surrogate")` replaces the reference footprint propagation. The defaults are unchanged.
3. **Every learned model is described by a model card built from the result files** (`cosmos/ai/registry.py`). Each card carries an evidence grade: `validated-real`, `validated-synthetic`, `baseline`, `not-recommended` or `not-run`.

![Package-level architecture of COSMOS at commit `2525367`. Solid arrows form the end-to-end chain of `run_pipeline`. Dashed arrows are optional COSMOS-AI replacements or supporting physics. The debris subpackage is not part of the chain.](figures/diag_overall.png){#fig:overall width=100%}

## Feature 1: satellite health {#sec:arch-health}

### Telemetry simulator (`health/telemetry.py`)

`simulate_telemetry(hours=24, dt_s=30, faults=None, fault_onset_frac=0.55, seed=7)` generates 16 channels: two for each of eight subsystems (`CHANNELS`, l.25–42). Each channel has a base value $b$, an orbital amplitude $A$ and a noise level $\sigma$. With orbital period $T_{orb} = 5700$ s (l.53), channel $k\in\{0,1\}$ of a subsystem is
$$
x_k(t) = b + A\sin\!\left(\frac{2\pi t}{T_{orb}} + 0.7k\right) + b\cdot 2\times10^{-6}\,\frac{t}{3600} + \varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma^2).
$$ {#eq:telemetry}
A fault adds a deterministic deviation and extra noise after the onset:
$$
\Delta x_k(t) = g_k\,\sigma\,6\,s(t) + \mathcal N\!\left(0, [\sigma(\nu-1)]^2\right) s(t).
$$
The severity $s(t)$ either ramps as $\mathrm{linspace}(0,1)^{1.3}$ or steps to 1. The gains $g_k$ and noise multiplier $\nu$ are set per fault type (`FAULTS`, l.45–52): degradation, leak, overheat, stuck, dropout and saturation. The `stuck` fault has $\nu = 0.6$, so $\nu - 1 < 0$. The rerun recorded the resulting error: ValueError('scale < 0'). This is a known defect (Appendix C).

### Classical health monitor (`health/monitor.py`)

`HealthMonitor(method="ewma", z_saturate=6, alpha=0.08, alarm_threshold=0.5, alarm_persistence=5)` proceeds as follows.

1. **Fit (l.127).** It estimates the median and $\hat\sigma = \max(1.4826\,\mathrm{MAD}, 10^{-9})$ for each channel on the first 40 % of the record. That window would be capped at the true fault-onset index (l.129–130), which is ground-truth information. With the simulator's default onset at 55 % of the record, the cap never binds (Appendix C).
2. **Robust z-scores.** It forms $z_{t,c}$ in the form of [@eq:robust], with the MAD-based scale of step 1 instead of the interquartile range.
3. **Smoothing.** It applies an exponentially weighted moving average $y_t = \alpha x_t + (1-\alpha) y_{t-1}$.
4. **Degradation scores.** Subsystem degradation is $D_s(t) = \mathrm{clip}(\mathrm{EWMA}(\max_{c\in s}|z_{t,c}|)/6, 0, 1)$. The aggregate anomaly score $A(t)$ is the same quantity taken over all 16 channels.
5. **Alarm.** The alarm is the first run of five consecutive samples with $A > 0.5$.
6. **Health band.** The band of the worst subsystem is NOMINAL (below 0.20), WATCH (below 0.45), DEGRADED (below 0.70) or CRITICAL (l.24).

An Isolation-Forest option exists, but the pipeline does not use it.

### Coupling of health to break-up (`health/hazard.py`)

`assess_hazard(anomaly_score, health)` implements the project's central coupling claim. With degradations $h_k$ and fragmentation weights $w_k$ (`FRAG_WEIGHT`, l.22: propulsion 3.20, structure 1.60, thermal 1.35, power 1.05, attitude 0.95, payload 0.35, communication 0.12, data storage 0.10):
$$
\mathrm{sev} = \mathrm{clip}\!\left(0.65\,\frac{\max_k w_k h_k}{3.2} + 0.35\,\bar h, 0, 1\right),\qquad
P_{\text{frag}} = \sigma\!\left(-5.2 + 2.10\,a + 3.05\,\frac{\sum_k w_k h_k}{3.2}\right),
$$ {#eq:hazard}
where $a$ is the clipped anomaly score. The rule produces four quantities:

- **Driver.** The driver is the subsystem with the largest *unweighted* degradation. Ties go to the first subsystem in the dictionary (`monitor.py`, l.46–48).
- **Break-up type.** It is "explosion" for propulsion and power and "aerodynamic" otherwise (l.33).
- **Break-up altitude.** It is a driver-specific base altitude minus $14\,\mathrm{sev}$ km (l.28).
- **Footprint spread.** It is $\sigma_{\log_{10}B} = 0.16 + 0.13\,\mathrm{sev}$.

The function also returns $P_{\text{frag}}$ and a state uncertainty $\sigma_{\text{state}}$. Neither value is used by any downstream computation. A break-up is propagated for every scenario, including a nominal one, and the risk is not weighted by $P_{\text{frag}}$ (Appendix C).

### Real-telemetry monitor (`health/realdata.py`)

`score_real` is a separate, stand-alone module that applies the monitor to real telemetry (SMAP/MSL and ESA-AD). It is not called by `run_pipeline`. Its main elements are:

- ridge AR(8) residuals fitted on the first part of the record;
- a 30-day trailing robust baseline shifted by one day;
- per-channel calibration: each channel's $|z|$ is divided by its own 99.9th percentile in the fitting window, and the EWMA of the channel maximum by twice its 99.9th percentile, so that 0.5 marks the fitting window's 99.9th percentile;
- EWMA smoothing, and an alarm at 0.5 held for five samples.

Under the official ESA-ADB split, COSMOS-AI uses it as the "COSMOS monitor" baseline (`telemetry/baselines.py`).

## Feature 2: debris {#sec:arch-debris}

The debris subpackage is a set of tools that stand alongside the end-to-end chain. They are not part of it.

- **`catalogue.py`.** It loads the CelesTrak SATCAT and computes the spatial density of debris in 100 km shells. It extracts four historical break-up clouds by international designator (Cosmos 1408, Iridium 33, Cosmos 2251 and Fengyun-1C) and estimates their survival with Kaplan–Meier. Their lifetimes are measured from the event date to the decay date, censored at the hard-coded date 2026-08-29 (l.110), 13 days after the snapshot was taken.
- **`lifetime.py`.** It integrates the drag-decay law of [@sec:theory-physics] with US76 density, constant above 1000 km and with no solar activity. A single area-to-mass ratio is fitted by grid search on the squared error in $\log_{10}$ lifetime. The loss treats censored objects one-sidedly, and a bias factor is reported relative to fitting on decayed objects only. Because every fragment of a cloud starts from its event altitude, the fit sees only three distinct altitudes. The module also contains a collision-rate and "criticality" estimate per shell; its fragments-per-collision formula is not the Standard Breakup Model law.
- **`scene.py`.** It renders synthetic star-field frame pairs. Each scene has 90–219 stars with Gaussian point-spread functions, limb glow, 5–11 moving objects in four classes, a random inter-frame jitter, and Poisson plus read noise. It exports registered difference images in YOLO format.
- **`detect.py`.** It implements phase-correlation registration, signed frame differencing, a photon-noise map and a multi-scale DoG detector with dipole rejection ([@sec:theory-vision]). It also provides single-frame and frame-pair baselines and a check of the registration sign. No tracking, association across more than two frames, or orbit determination is implemented.
- **`yolo.py`.** It contains a small single-scale YOLO-style detector (TinyYOLO, 320,313 parameters on a stride-8 grid) and an adapter for the Ultralytics library. Its `map50` is computed by centre-distance matching, class-agnostically and without interpolation, so it is not a COCO mAP@0.5 (Appendix C).
- **`realimagery.py`.** It converts external labelled imagery (SPARK, COCO or YOLO layouts) for training. No real imagery of this kind was available.

## Feature 3: re-entry {#sec:arch-reentry}

- **Atmosphere (`atmosphere.py`).** Below 86 km geometric altitude it implements US76 with its seven layers, whose boundaries are geopotential (up to 84.852 km). Above that it interpolates log-linearly over 23 density nodes up to 1000 km and holds the value constant beyond. Sound speed, Sutherland viscosity, mean free path and Knudsen number are derived from it.
- **Drag (`drag.py`).** It blends a continuum $C_D(M)$ curve with a transonic bump and the free-molecular limit 2.10, through a logistic weight in $\log_{10} Kn$. Shape parameters are given for spheres, cylinders, boxes, plates, cones and a "tumbling" average.
- **Break-up (`breakup.py`).** This is a simplified variant of the NASA Standard Breakup Model.
  - *Sizes:* a truncated power law on 0.05–4 m.
  - *Area-to-mass:* a bimodal log-normal in $\log_{10} L_c$.
  - *Area:* $A = 0.540424\,L_c^2$.
  - *Velocity increments:* log-normal $\Delta v$ with mean $0.9\log_{10}(A/m) + \log_{10} f + 1.85$ for every break-up type, where the factor $f$ is 0.20 (explosion), 0.90 (collision) or 0.06 (aerodynamic) (l.28, l.84). The SBM uses type-specific slopes and offsets instead [@johnson2001sbm].
  - *Mass closure:* fragments are kept while the cumulative mass stays within 92 % of the parent, then remnants are added.

  The explosion fragment count is scaled by the parent mass and capped at 1200. The default 1800 kg parent therefore reaches the cap for both break-up types. The audit found that several coefficients differ from the published model (Appendix C).
- **Propagator (`propagator.py`).** It integrates [@eq:eom] in two ways:
  - *Single objects:* an adaptive Dormand–Prince 5(4) integrator with Sutton–Graves heating and ablation.
  - *Ensembles:* a vectorised fixed-step RK4 integrator ($\Delta t = 0.5$ s) without mass loss.
- **Monte Carlo (`montecarlo.py`).** `monte_carlo_footprint` resamples fragments with replacement and perturbs three quantities: the ballistic coefficient by the hazard's $\sigma_{\log_{10}B}$, the density by $10^{\mathcal N(0,0.08)}$, and the velocity increment by an isotropic direction. Two defects are known. One flight-path-angle perturbation is shared by all samples (l.115), and the configured ballistic-coefficient noise and wind dispersion are unused. The module compares Gaussian ellipses with split-conformal ellipse and density zones.
- **Risk (`risk.py`, `population.py`).**
  - *Population density:* GHS-POP 2020 aggregated to 0.25° cells, with Natural Earth land fractions.
  - *Casualty area:* $A_c = (\sqrt{A_d}+0.6)^2$.
  - *Expected casualties:* summed over fragments above 15 J.

  `run_pipeline` reports a *per-fragment* expectation with $A_d = 1$ m². An event-level value is computed only by `corridor_risk`. It assumes that every fragment survives, and it computes each fragment's area as $m/B = C_D A$ with $C_D$ = 1.35 (`breakup.py`, l.140), not as $A$.

## The end-to-end chain {#sec:arch-pipeline}

[@fig:pipeline-flow] shows the call sequence of `run_pipeline`. Its main features are:

- **Inputs.** The only data file it reads is the population grid; everything else is simulated.
- **Stage hand-offs.** The monitor passes the *last* sample's degradation vector and anomaly score to the hazard rule. The hazard rule passes the break-up type, altitude and $\sigma_{\log_{10}B}$ to the break-up and footprint stages.
- **Default run.** With the default arguments (no fault, seed 7), the run reports a final anomaly score of 0.241, health band NOMINAL, and a break-up of type "aerodynamic" at 77.9 km. It generates 868 fragments, propagates 1199 impacts with CEP50 = 459.7 km, and reports risk band NEGLIGIBLE. The values come from `thesis/tools/run_original_pipeline.py`, package version 1.1.0.

![Data flow of `run_pipeline` (`cosmos/pipeline.py`, l.73–122). Grey boxes pass data without computation. The hazard rule's $P_{\text{frag}}$ and $\sigma_{\text{state}}$ are displayed only.](figures/diag_pipeline_flow.png){#fig:pipeline-flow width=80%}

## The COSMOS-AI layer {#sec:arch-ai}

[@fig:ai-layer] groups the machine-learning layer by function.

- **Data (`cosmos/ai/data/`).**
  - `download.py` provides resumable, checksum-verified downloads with manifests. They record MD5 and SHA-256 hashes, source records and GitHub commits. `safe_extract` refuses path traversal.
  - `preprocess_telemetry.py` builds the official ESA-ADB grids. It is verified against a literal port of the official pandas code by a unit test. It also builds the OPSSAT-AD tensors.
  - `preprocess_catalogue.py` builds the re-entry landmark together with a leakage report.
  - `validate.py` recomputes every dataset statistic from the files and writes `data/DATASET_REPORT.{json,md}`.
- **Telemetry (`cosmos/ai/telemetry/`).**
  - *Models (`models.py`):* forecasters (causal TCN, PatchTST) and classifiers (feature MLP, 1-D CNN, patch transformer, hybrid CNN + MLP).
  - *ESA-ADB studies:* `forecasting.py`, `alarms.py`, `esa_benchmark.py`, `oof.py`, `event_model.py`, `event_gbt.py` and `oof_benchmark.py`, with the official metric code vendored unchanged in `esa_metrics/`.
  - *OPSSAT-AD:* `opssat.py` and `robustness.py`.
  - *Learned monitor:* `learned_monitor.py`, for the synthetic pipeline.
- **Vision (`cosmos/ai/vision/`).** It contains the SpotGEO front end and the vendored official scorer, heat-map networks (U-Net and ResNet-18 U-Net), the training loop, the COSMOS synthetic-scene study and a COCO-style mAP implementation.
- **Re-entry.** `survival/landmark.py` holds the catalogue landmark study. `surrogate.py` holds the propagator surrogate.
- **Cross-cutting modules.**
  - *`uncertainty.py`:* conformal p-values, BH, Simes and split-conformal quantiles.
  - *`evaluation.py`:* classification and survival metrics, plus bootstraps.
  - *`explain.py`:* Integrated Gradients with completeness and randomisation checks.
  - *`fusion.py`:* the evidence-combination rule and the surrogate footprint.
  - *`registry.py`:* model cards.
  - *`experiment.py`:* run records with configuration, environment, epoch logs, metrics and checkpoints, appended to `outputs/experiments/index.jsonl`.
  - *`report.py`:* tables and figures.

![Functional map of COSMOS-AI. Dashed arrows show which modules feed the cross-cutting uncertainty, fusion and registry components.](figures/diag_ai_layer.png){#fig:ai-layer width=95%}

Two integration facts matter for interpreting the results.

1. **Pipeline back-ends.** The learned monitor and the surrogate footprint are wired into `run_pipeline` through the opt-in switches. The evidence-combination rule `fusion.decide` is used only in the experiments notebook and the tests. It is not part of the pipeline.
2. **Real-data studies.** The real-data studies (ESA-ADB, OPSSAT-AD, SpotGEO, landmark) are offline evaluations. They do not feed the physics chain, which runs on simulated telemetry.

## Implementation status {#sec:status}

[@tbl:status] classifies every major component by the evidence behind it.

: Implementation status of the components described in this thesis. "Tested" means covered by an automated check that compares with an independent or hand-computed answer. "Evaluated" means scored on held-out data with a result file. {#tbl:status}

| Component | Status | Evidence |
|---|---|---|
| US76 atmosphere, drag limits, geodesy, propagator, break-up mass closure, registration sign, conformal zone coverage | implemented, tested | 20/20 checks in `tests/test_verify.py`, 10 of them against external references ([@sec:results-verify]) |
| Telemetry simulator, classical monitor, hazard coupling, Monte Carlo footprint, casualty estimate | implemented; behaviour characterised on simulated data; **not validated against real events** | [@sec:results-orig] |
| Debris catalogue, lifetime model, synthetic detection front end, TinyYOLO | implemented; evaluated on real catalogue data (lifetimes) and synthetic imagery only | [@sec:results-orig] |
| ESA-ADB official preprocessing, metrics and baselines | implemented, tested against the official code; reproduces published baselines | [@sec:results-baselines] |
| ESA-ADB forecasters, alarm layers, out-of-fold protocol, event scorers | implemented, evaluated on the real test halves; pre-registered | [@sec:results-prereg] |
| OPSSAT-AD classifiers, robustness study | implemented, evaluated on real data | [@sec:results-support] |
| Learned health monitor | implemented, evaluated on **simulated** telemetry only | [@sec:results-support] |
| SpotGEO heat-map detectors; synthetic-scene detectors | implemented, evaluated (real GEO imagery; synthetic scenes) | [@sec:results-support] |
| Re-entry landmark models; propagator surrogate | implemented, evaluated (real catalogue; simulator only) | [@sec:results-support] |
| Evidence fusion (`fusion.decide`) | implemented, unit-tested; not wired into the pipeline; not evaluated | `tests/test_ai.py` |
| Chronos-2 (zero-shot) | loading supported in `forecasting.load_chronos`; **not run** | — |
| LoRA or other fine-tuning of foundation models | **not implemented** (an earlier project summary stated otherwise; Appendix C) | source search |
| Impact-location validation against recovered debris | **not implemented** (no trajectory data) | — |
| APSGTT–LoRA, diffusion forecasting, federated learning, dashboard (Review-1 design) | **not implemented in the current code** | source search |


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Methodology and Implementation {#sec:methods}

This chapter describes how each study was implemented and why the main design decisions were taken. The ESA-ADB studies are described in most detail, because they carry the confirmatory claims.

The configuration values of the pre-registered study are inserted from the frozen code (keys `cfg.*` in Appendix E). For the other studies, constants are quoted with the module that defines them.

## Data acquisition and validation {#sec:methods-data}

Every external dataset is fetched by `cosmos/ai/data/download.py`. The downloader resumes interrupted transfers and verifies each file against the MD5 checksum published by Zenodo (or against the file size where no checksum is published). It records MD5 and SHA-256 hashes, the source record and the download date in a per-dataset `MANIFEST.json`. GitHub code snapshots record the commit that was fetched. Archives are extracted by a function that refuses absolute paths, parent-directory references and links.

The ESA Anomalies Dataset exists in two Zenodo revisions. Rather than re-downloading the 7.9 GB second revision, the second revision was compared with the first member by member using the CRC-32 values in the zip central directories, read over HTTP range requests. The result: "780/780 (M1) and 230/230 (M2) members present in both; the ONLY changed member is ESA-MissionN/channels.csv, which gains a 'Categorical' column (M2 also blanks the physical unit of channel_4 and channel_5). Labels, anomaly types, telecommands and all channel pickles are byte-identical.".

`cosmos/ai/data/validate.py` then recomputes every dataset statistic used in this thesis directly from the files: counts, class balance, missing values, duplicates between splits, coordinate ranges and leakage rules. It writes `data/DATASET_REPORT.json`. Dataset numbers in this thesis come from that report.

## ESA-ADB: official preprocessing {#sec:methods-esa-pre}

`prepare_esa_official` (`cosmos/ai/data/preprocess_telemetry.py`) reproduces the official preprocessing of the benchmark.

1. Each channel is mapped onto the mission grid by zero-order hold: 30 s for Mission 1 and 18 s for Mission 2.
2. Monotonic counters are differenced.
3. String-valued channels on Mission 2 are factorised.
4. The benchmark's "restore" step is applied. When a grid bin contains several samples and an annotated sample would otherwise be lost, that annotated sample is moved onto the next grid point.
5. Telecommands of priority 3 or higher become a 0/1 indicator per grid step.

A unit test compares the vectorised implementation with a literal port of the official pandas code on irregular synthetic channels and requires identical values and labels. The processed grids contain 7,364,161 training and 7,364,161 test steps on Mission 1, and 3,067,201 and 3,062,401 steps on Mission 2. The target channel sets have 58 and 47 channels.

The official test labels are restricted to the target channels and to events lying entirely inside the test half (`official_test_labels`). The scoring adapter (`telemetry/esa_eval.py`) converts per-channel alarms into the event intervals that the vendored official code expects.

## ESA-ADB: the exploratory study (H1–H3) {#sec:methods-explo}

The first study was designed before the work reported here was extended to Q1-journal standards. It is reported as **exploratory**, because the design was revised after its test numbers had been seen ([@sec:methods-prereg]).

**Periods.** The training half of each mission was divided into three periods:

| Purpose | Mission 1 | Mission 2 |
|---|---|---|
| Fitting | 2000-01-01 to 2006-01-01 | 2000-01-01 to 2001-04-01 |
| Early stopping | 2006-01-01 to 2006-04-01 | 2001-04-01 to 2001-05-15 |
| Calibration and selection | 2006-04-01 to 2007-01-01 | 2001-05-15 to 2001-10-01 |

**Encoders.** Five encoders produced forecasts on a shared origin schedule (context 512 steps, horizon 16 steps):

- **Persistence and AR(8).** These are the COSMOS forecasters ([@eq:ar]), fitted per channel on the fitting period.
- **A causal TCN.** It has 7 levels, a receptive field of 509 steps, width 64, a learned channel embedding and seven quantile outputs per horizon step.
- **PatchTST.** It uses patches of 16 steps, 3 layers, width 64 and per-window instance normalisation.
- **Chronos-Bolt-small.** This pretrained forecaster was used zero-shot.

The deep encoders were trained channel-independently with weight sharing:

- **Data:** windows drawn from nominal fitting data.
- **Optimiser:** AdamW [@loshchilov2019adamw], learning rate 0.001, weight decay 0.0001.
- **Budget:** batch 256, at most 20,000 steps.
- **Early stopping:** on the early-stopping period, with a patience of 6 evaluations.

The TCN's normalisation was changed from GroupNorm to a per-time-step LayerNorm (`ChannelNorm`), because GroupNorm over time leaks future samples into earlier outputs. The fix was verified by a causality test.

**Calibrators.** Three calibrators turn scores into hot cells, and all three share the event logic of [@sec:theory-events]:

- a per-channel percentile threshold, with tail fractions [0.001, 0.0001, 1e-05, 0.0];
- conformal p-values with BH across channels at level $q \in$ [0.01, 0.001, 0.0003, 0.0001];
- a rolling variant of the second, whose pool is refreshed daily from the trailing 30 days of *unlabelled* test scores.

The persistence was chosen from $k\in$ [1, 3, 10, 30], and the merge gap was fixed at 1 h. Level and persistence were selected by cross-fitting on the calibration period: calibrate on one half, score the other half with the official metric, swap and average. On Mission 1 this period contains only 9 labelled events.

**Telecommands.** For H3, the deep encoders optionally received telecommand features $\exp(-\Delta t/\tau)$ with $\tau \in \{1, 12\}$ h. These entered through a gated fusion $h' = h + \sigma(W_g[h; e]) \odot W_e e$, with modality dropout during training.

**Hypotheses (exploratory).**

- **H1:** a deep or pretrained encoder beats AR(8) under the same calibration.
- **H2:** conformal-BH beats a per-channel threshold.
- **H3:** telecommand gating reduces alarm runs that touch only rare nominal events.

Deep arms used three seeds.

## ESA-ADB: the pre-registered study {#sec:methods-prereg}

### Why a second study

The exploratory study left one striking asymmetry. The same AR(8) scores with the same conformal-BH layer scored 0.962 on Mission 2 but 0.074 on Mission 1. On Mission 1 the alarm layer had been selected on 9 events. The training half as a whole holds 109 annotated events on the target channels:

- 63 anomalies;
- 42 rare nominal events;
- 4 communication gaps.

These were unavailable to the selection because no out-of-sample scores existed for them. The pre-registered study therefore asks two questions:

- Does selecting the alarm layer on out-of-fold scores of the whole training half fix the Mission 1 failure?
- Does a supervised event scorer trained on those labels add to an unsupervised forecaster?

The decision to study Mission 1 selection was taken *after* Mission 1 had failed on its test half. This is acknowledged as a threat to validity ([@sec:threats]).

[@fig:esa-protocol] summarises the resulting protocol.

![Workflow of the pre-registered ESA-ADB study. Everything above the freeze uses the training half only. The test half is scored once per arm, through a function that verifies the frozen hashes.](figures/diag_esa_protocol.png){#fig:esa-protocol width=75%}

### Out-of-fold scores {#sec:oof}

Out-of-fold scores are produced as follows (`telemetry/oof.py`).

1. The training half is cut into $K$ = 5 contiguous blocks.
2. For block $b$, the AR(8) forecaster of [@eq:ar] is fitted on the nominal samples of the other blocks. A purge margin of one day on either side of $b$ is excluded (2,880 steps on Mission 1, 4,800 on Mission 2).
3. The fitted forecaster scores block $b$ with [@eq:score-ar].
4. Inputs (lagged values) are read from the record itself, as any deployed forecaster would read them.
5. Test-half scores come from a forecaster fitted on every nominal training sample.

This is cross-fitting, not forward chaining: a block can be scored by a model that also saw later blocks. Forward chaining would have discarded the first block. On Mission 1 that block holds the densest year of events. The resulting optimism is a stated threat.

### Event features and scorers {#sec:model}

Each target channel and grid step receives seven causal features (`telemetry/event_model.py`): ['asinh_z', 'log_s', 'ewm_1h', 'ewm_12h', 'ewm_3d', 'asinh_dev_1d', 'finite']. In words, these are:

- $\operatorname{asinh} z$;
- $\log(1+s)$, where $s$ is the out-of-fold AR(8) score;
- the trailing exponential means of $\log(1+s)$ with time constants 1 h, 12 h and 3 days. These give context beyond the network's receptive field, since Mission 1 events often last days.
- $\operatorname{asinh}(z - \bar z^{1\text{d}})$, the value minus its trailing one-day mean;
- an indicator that a forecast exists.

The exponential means are computed with `scipy.signal.lfilter` over the joined training-and-test record, so the test half starts from the state at the end of training. A unit test confirms that a change at step $t$ leaves every earlier feature unchanged.

**A3: causal TCN event scorer.** [@fig:event-tcn] shows the architecture.

- **Input:** a 1×1 convolution from 7 to 64 channels plus a learned channel embedding.
- **Body:** 8 residual blocks with dilations $2^0 \dots 2^7$. Each block holds two causal convolutions (kernel 3) with per-step LayerNorm, SiLU activation and dropout 0.1.
- **Cross-channel mixing:** every 2 blocks the channel mean of the hidden state is projected and added back. This mixing is permutation-equivariant.
- **Head:** a 1×1 convolution producing three logits per channel and step: nominal, anomaly and rare nominal event.

The receptive field is 1021 steps, and the model has 220,739 parameters on Mission 1. Training used the following settings:

- class-weighted cross-entropy with weights [1.0, 5.0, 5.0], with communication gaps ignored;
- 1,500 AdamW steps (learning rate 0.001, weight decay 0.0001, one-cycle schedule [@smith2019superconvergence]);
- batches of 8 windows of 2,048 steps containing all channels. A fraction 0.5 of the windows contained a labelled sample, placed at a uniformly random position inside the part of the window where the loss is computed.

The loss is computed only on the last $T - \mathrm{RF}$ steps of each window (RF = 1021), where the receptive field is complete. The event score is $\log P(\text{anomaly} \cup \text{rare}) - \log P(\text{nominal})$.

**A2: tree baseline** (`telemetry/event_gbt.py`). It uses the same seven features, plus trailing maxima of $\log(1+s)$ over 16 and 256 steps and the channel index as a categorical variable. The model is a three-class XGBoost with these settings:

- objective `multi:softprob`;
- 400 trees of depth 8, learning rate 0.1, subsampling 0.8;
- the same class weights as A3;
- training on 2,000,000 subsampled (step, channel) rows, half of them labelled.

**Out-of-fold protocol for both scorers.** One model is trained per held-out block (on the other blocks, minus the purge margin) and predicts that block. A final model trained on all blocks predicts the test half.

![The causal TCN event scorer (arm A3). The receptive field is 1021 steps. Channel mixing preserves causality because it acts at each time step separately.](figures/diag_event_tcn.png){#fig:event-tcn width=70%}

### The event-level conformal alarm layer {#sec:alarm-layer}

All arms share one layer ([@fig:alarm-layer]; `telemetry/alarms.py`).

1. Each channel's nominal out-of-fold training scores form its calibration pool.
2. Test scores receive conformal p-values ([@eq:conformal-p]).
3. BH at level $q$ is applied across channels at each step.
4. Persistence $k$ and a 1 h merge are applied.
5. Channels are attributed to each alarm run ([@sec:theory-events]).

The pre-registered grids are:

- BH level $q \in$ [0.01, 0.001, 0.0001, 1e-05];
- per-channel threshold tail fractions [0.001, 0.0001, 1e-05, 0.0] (0 means the maximum nominal score);
- persistence $k \in$ [1, 3, 10, 30, 100].

The rolling variant was retained as a secondary layer. It is not selected on its own; it reuses the $(q, k)$ selected for conformal-BH.

![The event-level conformal alarm layer shared by all arms of the pre-registered study.](figures/diag_alarm_layer.png){#fig:alarm-layer width=100%}

**Selection** (`oof_benchmark.select`) cross-fits over the two halves of the training half:

1. Calibrate on the nominal out-of-fold scores of one half.
2. Raise alarms on the other half and score them with the official metric on that half's labels.
3. Swap the halves and average the two results.

Ties go to the smaller alarm fraction. The chosen layer is refitted on the whole training half and applied once to the test scores.

**Point-level comparator.** This is per-channel ACI ([@eq:aci]) with $\alpha$ = 0.01. $\gamma$ is chosen from [0.001, 0.005, 0.01, 0.05] as the value whose nominal false-alarm rate on the second training half is closest to $\alpha$. The rule is adapted from Şahin, who selects $\gamma$ on a window of the test period [@sahin2026tccad]. Its global alarm is the OR over channels. It is evaluated without persistence or merging, and also with the event logic of the layer above (`aci_event`).

### Arms, hypotheses and statistics {#sec:prereg}

[@tbl:arms] defines the arms. A2 and A3 were run with five seeds each (0–4). The secondary arm with anomaly-only scores, listed in the pre-registration, was not run in the confirmatory pass (Appendix B); it was run post hoc as P5 ([@sec:results-posthoc]). Two ablations of A3 were run with three seeds:

- without the forecast-residual inputs (`no_residual`);
- with two output classes instead of three.

: Arms of the pre-registered study. All share the alarm layer of [@sec:alarm-layer]. {#tbl:arms}

| Arm | Scores | Selection of $(q, k)$ | Seeds |
|---|---|---|---|
| A0 | AR(8) from the exploratory study (fitted on the fitting period) | cross-fit on the calibration period | deterministic |
| A1 | AR(8), out of fold ($K = 5$) | cross-fit on the whole training half | deterministic |
| A2 | XGBoost event scorer, out of fold | cross-fit on the whole training half | 5 |
| A3 | causal TCN event scorer, out of fold | cross-fit on the whole training half | 5 |

**Hypotheses.** They concern Mission 1, the primary metric and the conformal-BH layer:

- **H4:** A3 > A1. A supervised deep scorer beats the unsupervised forecaster.
- **H5:** A3 > A2. The deep scorer beats trees on the same features.
- **H6:** A1 > A0. Selection on the whole training half beats selection on the calibration period.

**Statistics.**

- *Test:* a paired block bootstrap (fixed 30-day blocks shared by both arms, seeds resampled hierarchically, 2,000 replicates; [@sec:theory-stats]).
- *Correction:* Holm across the family {H4, H5, H6} at $\alpha = 0.05$.
- *Minimum effect size of interest:* +0.05.
- *Verdicts:*
  - **supported** if the Holm-adjusted $p < 0.05$ *and* the point estimate is at least +0.05;
  - **refuted** if the upper end of the 95 % interval lies below +0.05;
  - **inconclusive** otherwise.

Mission 2 repeats the three comparisons with the same test but without multiplicity correction, outside the confirmatory family. Mission 2 was assigned that role before any arm was scored, because the out-of-limit rule already scores 0.894 there.

### Freeze, gating and deviations

The pre-registration document (`research_design/preregistration/PREREG_TELEMETRY_V1.md`) and 27 code files, 28 files in all, were hashed into `freeze.json` at 2026-10-24T15:56:29 (freeze `09a98f5e160e`). The frozen code covers:

- the out-of-fold scorers;
- the event models;
- the alarm layer;
- the selection and test code;
- the bootstrap;
- the official metric code;
- the preprocessing.

`oof_benchmark.test` refuses to run if any frozen file has changed, and every arm was scored once. 8 deviations are logged with their timing (Appendix B). Five are procedural and changed no computation. The sixth records that arm A0 carries the exploratory selection grid, so H6 compares two complete protocols. After all confirmatory numbers existed, the pre-registration's own deviation log was filled in and the freeze reissued as `681cc7ddec86`.

## Label-free baselines for ESA-ADB {#sec:methods-baselines}

Three label-free baselines are scored through the same metric code (`telemetry/baselines.py`).

- **GlobalSTD.** This is an exact port of the benchmark's own rule. It alarms when $|x - \mu| > \kappa\sigma$, with the mean and standard deviation of nominal training samples and $\kappa \in \{3, 5\}$. Its purpose is to check the whole preprocessing and metric chain against the published values.
- **Out-of-limit (OOL).** This rule alarms whenever a sample leaves the [minimum, maximum] range of its channel's nominal training samples. It has no free parameter. A variant uses all training samples.
- **The COSMOS monitor** of `health/realdata.py`. It is fitted on the official training half and run online over the test half.

## OPSSAT-AD {#sec:methods-opssat}

The OPSSAT-AD study (`telemetry/opssat.py`) uses two inputs:

- the 18 official features, signed-log transformed;
- the raw fragments resampled to 256 points as three channels: per-fragment z-score, $\log(1+\Delta t)$ and value divided by its maximum absolute value.

**Splits.** Two splits are evaluated:

- *official:* 1594 training and 529 test fragments;
- *time-blocked:* all fragments starting before the 75th percentile of start times go to training.

In both splits, a stratified 20 % of the training part is held out for validation.

**Classical models.** The classical baselines are:

- a COSMOS robust-z rule, with its threshold at the 95th percentile of nominal training scores;
- Isolation Forest;
- AdaBoost;
- a random forest;
- logistic regression;
- XGBoost, early-stopped on validation AUC-PR.

**Deep models.** The deep models (`telemetry/models.py`) are:

- the feature MLP (FCNN, 13,249 parameters), which re-implements the strongest published baseline;
- a 1-D CNN on the raw fragment;
- a patch transformer;
- a hybrid network that concatenates CNN and MLP embeddings.

They were trained with binary cross-entropy and AdamW (learning rate $10^{-3}$, batch 64). Early stopping used validation AUC-PR with patience 30.

**Foundation-model probe.** A logistic-regression probe was trained on frozen Chronos-Bolt embeddings.

**Further analyses.** Every model was run with five seeds and decides at $P \ge 0.5$. The study also includes:

- label efficiency (10–100 % of the training data);
- robustness to added noise and to sample dropout (`telemetry/robustness.py`);
- paired bootstrap comparisons on seed-averaged scores.

## Learned health monitor (simulated telemetry) {#sec:methods-monitor}

`telemetry/learned_monitor.py` replaces the classical monitor inside `run_pipeline`. It works in five steps.

1. **Training data.** A PatchTST forecaster (98,424 parameters) is trained on 180 simulated fault-free runs with the pinball loss.
2. **Scoring.** Scores follow [@eq:score-q] and are smoothed by an EWMA.
3. **Calibration.** Scores are calibrated against 20 held-out fault-free runs, which yields conformal p-values per channel.
4. **Combination.** The p-values are combined across the 16 channels with Simes. The anomaly score $a = \mathrm{clip}\big(-\log_{10} p_S / (2\cdot(-\log_{10} q)), 0, 1\big)$ with $q = 10^{-3}$ equals 0.5 exactly at $p_S = q$.
5. **Alarm.** The classical persistence rule raises the alarm.

**Benchmark.** Every subsystem–fault pair except `stuck` (which crashes the simulator) is run, plus eight fault-free runs per seed set. Seed sets 0–4 served for development and 5–9 are held out. Two limitations apply:

- the 20 calibration runs also selected the checkpoint;
- both monitors would cap their healthy normalisation window at the true fault onset, but with the default onset the cap never binds (Appendix C).

## Vision {#sec:methods-vision}

### SpotGEO

The training set of 1280 sequences is split by a hash of the sequence identifier into 1153 training and 127 validation sequences. The 5120 test sequences are scored once with the vendored official scorer. For each frame the front end computes two channels:

- the frame minus the median of the other four registered frames, which reuses the COSMOS phase-correlation registration;
- the frame minus a Gaussian blur with $\sigma = 6$ px.

Both are robust-normalised. The input variants are both channels for all five frames (10 channels), differences only, or high-pass only.

**Networks.** The heat-map U-Net has encoder widths 32–256 and a 512-channel bottleneck, 7,784,015 parameters. The ResNet-18 variant uses an ImageNet-pretrained encoder (14,405,775 parameters). Both output five heat maps (one per frame) with sub-pixel offsets.

**Training.**

- Settings: 40 epochs, batch 12, random 320 px crops, AdamW (learning rate $10^{-3}$ for the U-Net, $3\times10^{-4}$ for ResNet), one-cycle schedule, gradient clipping 5.
- Augmentation: time reversal, flips and gain.
- Selection: every two epochs the validation set is decoded at thresholds 0.1–0.7. The epoch and threshold with the best validation F1 are kept.

**Classical baseline.** The COSMOS DoG detector is applied to the difference channel, with its threshold chosen on validation.

### COSMOS synthetic scenes (SYNTHETIC)

Scenes are drawn from disjoint seed ranges: 800 for training, 200 for validation and 200 for test. The images are COSMOS registered differences. Three detectors are compared:

- the existing TinyYOLO;
- a COCO-pretrained YOLOv8n via Ultralytics;
- the heat-map U-Net with a width–height head.

All three are scored with one COCO-style mAP implementation (101-point interpolation, IoU 0.50:0.05:0.95), because Ultralytics' internal evaluator interpolates differently.

## Re-entry {#sec:methods-reentry}

### Landmark study (real catalogue)

The design is shown below.

- **Snapshots.** The landmark date is $\tau$ = 2026-08-16, with follow-up to 2026-10-23. The CelesTrak SATCAT snapshot of $\tau$ defines the population; a later snapshot defines the outcomes.
- **Population.** An object is included if it is Earth-orbiting and in orbit at $\tau$, has orbit fields, and was launched by $\tau$. This leaves 34,158 objects with 235 re-entries.
- **Features.** Only information available at $\tau$ is used: SATCAT orbit fields, radar cross-section, object type, owner, launch site and age. Where available, the latest general-perturbations element set with epoch at most one day after $\tau$ is added; this covers 55.5 % of objects.
- **Split.** Objects are split by launch group (international designator year and number) into 60/20/20 % of groups, so that fragments of one launch never span splits.
- **Leakage checks.** A leakage report checks five rules, and all pass:
  - no element set after $\tau$;
  - no event at or before $\tau$;
  - launch groups confined to one split;
  - no decay field among the features;
  - a constant censoring time.

**Models.** Five models are compared:

- the COSMOS physics lifetime at the perigee, Platt-scaled on validation;
- logistic regression;
- XGBoost (early-stopped on validation AUC-PR, five seeds);
- XGBoost-AFT;
- a discrete-time hazard MLP with 7-day bins.

The MLP configuration was chosen on validation AUC-PR from a grid of 8 configurations, then run as a five-seed ensemble. Its selected configuration was {"hidden": [256, 256, 128], "lr": 0.0003, "quantile": true}.

**Metrics.** The metrics are:

- AUC-PR and AUC-ROC for re-entry within 30 and 68 days;
- the concordance index;
- the Brier score and the expected calibration error;
- launch-group bootstrap intervals for differences between models;
- marginal and Mondrian conformal sets at $\alpha = 0.1$.

### Propagator surrogate (SIMULATED)

`cosmos/ai/surrogate.py` trains an ensemble on reference simulations from the COSMOS ensemble propagator. Its inputs and outputs are:

- **Inputs (13):** break-up altitude, speed, flight-path angle, heading, latitude, ballistic coefficient, mass, density factor and a velocity increment.
- **Outputs:** downrange and crossrange impact position, fall time, impact speed and an impact flag.

**Data.** The simulations comprise 400,000 training and 40,000 validation cases (`surrogate.SPLITS`), 40,000 test cases, and an out-of-range test set of 20,000 cases with break-up altitudes of 100–120 km.

**Ensemble and training.** The ensemble has five MLPs of four 256-unit layers each (202,245 parameters per member). They were trained with masked MSE plus 0.1 times the impact-flag BCE, using AdamW and a cosine schedule.

**Use inside the pipeline.** `fusion.surrogate_footprint` reproduces the reference sampler's random draws call for call. Samples outside the training ranges fall back to the reference propagator (mode `range`). Two other modes are available: `spread`, which also falls back on large ensemble disagreement, and `none`.

## Experiment records and reproducibility

Every training or evaluation run is wrapped in `experiment.Run`. It writes the configuration, the environment (library versions and device), the epoch logs, the metrics and the checkpoints to a run folder, and appends a line to `outputs/experiments/index.jsonl`. The index holds 188 runs, of which 185 completed and 3 failed; every failed run was superseded by a successful rerun. Seeds are fixed for NumPy, Python and PyTorch. XPU kernels are not bit-deterministic, so repeated deep runs differ slightly, which is one reason seeds are reported as distributions. Appendix A gives the commands that regenerate each result.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Experimental Setup {#sec:setup}

## Datasets {#sec:datasets}

[@tbl:datasets] lists every dataset used, with its status as recorded by the validation report. "REAL" means recorded by an instrument or catalogue. "SYNTHETIC" and "SIM" mean generated by this project, and those results say nothing about real spacecraft.

: Datasets used in this thesis. Sizes are on disk after preparation (`data/DATASET_REPORT.json`). {#tbl:datasets}

| Dataset | Class | Status | Size (MB) | Licence (as recorded) | Used for |
|---|---|---|---|---|---|
| ESA Anomalies Dataset v1.0 (Zenodo 12528696), Missions 1–2 | REAL | downloaded | 8498 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | ESA-ADB studies |
| OPSSAT-AD (Zenodo 15108715) | REAL | downloaded | 22.8 | cc-by-4.0 | fragment classification |
| SpotGEO v2 (Zenodo 4432143) | REAL | downloaded | 45577 | cc-by-4.0 | GEO object detection |
| CelesTrak SATCAT and GP snapshots (2026-08-16, 2026-10-24) | REAL | downloaded | 21.2 | CelesTrak usage policy; public, no credentials; no explicit licence | lifetimes, landmark |
| ESA recovered-debris list (public web page) | REAL | downloaded (public parts only) | 0.2 | — | linkage check only |
| COSMOS synthetic telemetry | SYNTHETIC | generated | 38.9 | — | learned monitor |
| COSMOS synthetic frame pairs | SYNTHETIC | generated | 467.5 | generated by this project | detector comparison |
| COSMOS simulated re-entries | SIM | generated | 67.5 | — | propagator surrogate |
| SpaceSense-Bench | SYNTHETIC (Unreal Engine 5 renders) | inspection sample only (1 of 136 archives) | 1032 | cc-by-nc-4.0 (Hugging Face card); GitHub repo has no licence file | inspected; not used (resolved spacecraft, off-task) |
| SPARK | — | not downloaded | — | — | restricted (by request); not used |

The licence recorded for ESA-ADB cites the DMLR paper. That paper states no licence; the CC BY 3.0 IGO statement is in the arXiv version (2406.17826v2, p. 35).

The following data were **not** available:

- **Space-Track and ESA DISCOS.** Their element-set histories and re-entry records require an account or token, and no access control was bypassed.
- **ESA-ADB Mission 3.** Its authors declare it unsuitable for algorithm comparison [@kotowski2026esaadb].
- **DUST.** The dataset could not be located.
- **Recorded re-entry trajectories with recovery coordinates.** No such data were available, so no impact-location result can be validated.

### ESA-ADB event counts

[@tbl:esa-events] gives the labelled events per period on the target channels.

: Labelled events on the target channels of ESA-ADB, by period (`data/DATASET_REPORT.json`). The *fit*, *es* and *cal* periods partition the training half ([@sec:methods-explo]). {#tbl:esa-events}

| Period | M1 anomalies | M1 rare events | M1 gaps | M2 anomalies | M2 rare events |
|---|---|---|---|---|---|
| fit | 55 | 40 | 4 | 25 | 202 |
| es | 1 | 0 | 0 | 0 | 12 |
| cal | 7 | 2 | 0 | 0 | 43 |
| test half | 55 | 36 | — | 6 | 356 |

Two properties of these counts shape the interpretation of the results.

1. **The calibration windows differ sharply.** The Mission 1 window holds 9 events. The Mission 2 window holds 43, all of them rare nominal events.
2. **The Mission 2 test half is dominated by rare nominal events.** It holds 356 rare events against 6 anomalies. Anomaly-only results on Mission 2 therefore rest on 6 events.

Of the 17 event classes in the Mission 1 test half, 4 never occur in the training half. On Mission 2, 8 of 24 test classes are unseen.

### Other datasets

- **OPSSAT-AD.** It contains 2,123 fragments from 9 channels, with an anomaly fraction of 0.204. Fragment lengths have a median of 70 samples (range 8–1040).
- **SpotGEO.**
  - The training set holds 1,280 sequences, 6,400 frames and 11,250 labelled object positions.
  - The test set holds 5,120 sequences, 25,600 frames and 44,550 positions.
  - There are 0 corrupt images and 0 exact duplicates between training and test.
  - The release gives no per-object signal-to-noise ratio, so no breakdown by brightness is possible.
- **COSMOS synthetic frame pairs.**
  - The splits hold 800, 200 and 200 images, with 6,448, 1,619 and 1,603 objects.
  - The median box size is about 8.4 px.
- **Re-entry landmark.**
  - The population at risk holds 34,158 objects with 235 re-entries during 68 days.
  - Training has 18,914 objects (140 events), validation 8,873 (57) and test 6,371 (38).
  - Re-entries by object type: 173 payloads, 46 debris, 14 rocket bodies and 2 unknown.

## Hardware and software

All experiments ran on one server. The recorded environment is:

- **CPU:** 512 logical CPUs.
- **Accelerators:** Intel Arc Pro B60 GPUs. The run records give the device as "Intel(R) Arc(TM) Pro B60 Graphics", reached through PyTorch's XPU back end.
- **Software:** Python 3.12.13, PyTorch 2.13.0+xpu, NumPy 2.3.5 and pandas 3.0.3 on Linux-6.17.0-1010-intel-x86_64-with-glibc2.41.

The main computational costs were as follows:

- **TCN event scorer.** One Mission 1 run took 6170 s on average (five fold models and a final model; 5 runs).
- **XGBoost event scorer.** One run took 193 s on 64 CPU threads, excluding prediction.
- **Out-of-fold AR(8) scores.** These took 382 s for Mission 1.

## Evaluation protocols

[@tbl:protocols] summarises the protocols. Three rules are common to all studies:

- every test set was scored once for each frozen configuration;
- selection used only training or validation data;
- seeds are reported as distributions.

: Evaluation protocols. "Selection" lists the data used for every choice of hyper-parameters, thresholds or checkpoints. {#tbl:protocols}

| Study | Split | Selection | Test | Seeds |
|---|---|---|---|---|
| ESA-ADB exploratory (H1–H3) | official 50/50 | early stopping on *es*; calibrator by cross-fitting on *cal* | official test half | 3 (deep), 1 (deterministic) |
| ESA-ADB pre-registered (H4–H6) | official 50/50; training half in 5 out-of-fold blocks | cross-fitting on the two training-half halves | official test half, gated by the freeze | 5 (A2, A3), 3 (ablations) |
| OPSSAT-AD | official, and time-blocked (75 % / 25 % by start time) | stratified 20 % of training | held-out test | 5 |
| Learned monitor (SIM) | seed sets 0–4 (development) / 5–9 (held out) | development seeds | held-out seeds | 5 seed sets |
| SpotGEO | 90 % / 10 % hash split of training; official test | validation F1 (epoch, threshold) | 5,120 test sequences | 3 (U-Net), 2 (ResNet), 1 (ablations) |
| Synthetic scenes (SYNTHETIC) | disjoint seed ranges | validation mAP / F1 | 200 test scenes | 3 |
| Re-entry landmark | launch groups 60/20/20 | validation AUC-PR, Platt, conformal | test launches | 5 (ensembles) |
| Surrogate (SIM) | simulated train / val / test / out-of-range | validation loss | test and out-of-range sets | 5 members |

## Metrics

| Study | Metrics |
|---|---|
| ESA-ADB | primary: corrected event-wise $F_{0.5}$ over anomalies and rare events ([@eq:pe; @eq:fbeta]); also anomaly-only $F_{0.5}$, precision, recall, alarming precision, ADTQC, affiliation $F_{0.5}$, channel- and subsystem-aware $F_{0.5}$, false alarms per year, and alarm runs touching only rare events |
| OPSSAT-AD | F1, AUC-PR, AUC-ROC, MCC, Brier score, ECE |
| SpotGEO | the official $F_1$ and MSE |
| Synthetic scenes | COCO mAP@50 and mAP@50–95 |
| Landmark | AUC-PR, AUC-ROC, C-index, Brier score, ECE, conformal coverage |
| Surrogate | position error and fall-time error, ensemble spread, out-of-range AUROC, footprint CEP50 error |

## Statistical procedures

The procedures differ by study:

- **ESA-ADB:** paired block bootstraps with fixed 30-day blocks (86 on Mission 1, 22 on Mission 2) and hierarchical seed resampling ([@sec:theory-stats]), with Holm correction for the pre-registered family. The exploratory comparisons use 1,000 replicates of the same block bootstrap on seed-0 arms. The pre-registered comparisons use 2,000.
- **OPSSAT-AD:** paired i.i.d. bootstraps on seed-averaged test scores (2,000 replicates).
- **Landmark:** bootstraps that resample test launch groups (2,000 replicates).

Intervals are two-sided 95 % percentile intervals.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Results and Evaluation {#sec:results}

The results are presented in the order in which their evidence is strongest:

1. verification of the physics components;
2. the behaviour of the original pipeline;
3. the ESA-ADB studies;
4. the supporting studies.

Every value in this chapter is inserted from the evidence file. Values computed from simulated or synthetic data are labelled.

## Verification of the physics components {#sec:results-verify}

All 20 checks of the verification suite pass ([@tbl:verify]). They differ in strength, and the table states what each one compares against:

- 10 compare with an answer that does not come from the COSMOS code: a published table, a closed form, a physical limit, an imposed shift or a statistical guarantee;
- 9 are self-consistency checks or hold by construction, for example a constant compared with itself or a mass budget that the code enforces;
- 1 check, the hydrostatic one, is mis-specified. It compares $dp/dz$ with $-\rho g_0$ instead of $-\rho g(z)$, so its 0.933 % error is the ratio $g(30\,\text{km})/g_0$, not a numerical error, and the check would fail its own 1 % limit above about 33 km.

The checks print PASS or FAIL but do not raise an exception, so under pytest a failing check would still pass (Appendix C).

: Verification checks of the original pipeline (`tests/test_verify.py`; log `outputs/tests/test_verify_after.log`). "Reference" states what the expected value is based on. {#tbl:verify}

| Check | Reference | Expected | Measured |
|---|---|---|---|
| US76 density vs published table | published table | < 0.01 % | 0.0019 % |
| US76 temperature vs published table | layer constants (by construction) | 0 K | 0.000 K |
| hydrostatic balance dp/dz = -rho g | uses g0, not local g (mis-specified) | < 1 % | 0.933 % |
| WGS-84 geodetic round trip | round trip (self-consistency) | < 1e-6 m | 1.97e-10 m |
| great-circle quarter of the equator | same radius on both sides (self-consistency) | 10007.5 km | 10007.5 km |
| C_D at Kn -> infinity reaches the free-molecular limit | code constant (by construction) | 2.1 | 2.100 |
| C_D at Kn -> 0, Mach 20 reaches the continuum hypersonic value | code constant (by construction) | 0.92 | 0.920 |
| C_D peaks in the transonic regime above the hypersonic value | tuned bump (by construction) | > 0.92 | 1.012 |
| terminal velocity: drag / gravity at impact | physical limit | 1.00 | 0.9984 |
| Coriolis deflection is eastward and below the free-fall value | closed-form bound | 0 < d < 82 m | 81 m |
| impact point converges as rtol tightens 1e-6 -> 1e-12 | convergence (self-consistency) | < 1 m | 0.060 m |
| J2 makes a circular orbit oscillate in altitude | physical expectation | 15-25 km | 18.4 km |
| vacuum orbit does not decay | run shorter than any decay (weak) | no impact | timeout |
| break-up conserves the parent mass | enforced mass closure (by construction) | 1.000 | 1.0000 |
| sampled size distribution follows the truncated power law | closed-form sampling law | < 5 % | 3.80 % |
| registration recovers a known shift with the right SIGN | imposed shift (known answer) | < 0.5 px | 0.225 px |
| residual misalignment after registration | imposed shift (known answer) | < 0.3 px | 0.118 px |
| conformal density zone coverage at nominal 90% | conformal coverage guarantee | >= 0.87 | 0.9065 |
| conformal density zone coverage at nominal 95% | conformal coverage guarantee | >= 0.92 | 0.9495 |
| Gaussian ellipse under-covers on a skewed sample | known property of the ellipse | < 0.95 | 0.9430 |

These checks cover how individual components behave. They do not test the hazard rule, the Monte Carlo footprint, or the risk estimate as a whole. Nor do they test anything against a real re-entry.

The machine-learning layer has 45 checks in `tests/test_ai.py`, all passing with 0 failures. Among other things they compare:

- the preprocessing with a literal port of the official code;
- the official metric with a hand-solved case;
- the conformal p-values and BH procedure with their theoretical properties;
- the TCN and the event model for causality;
- the original pipeline's output with the untouched baseline.

The original dataset tests have one pre-existing error under pandas 3: a safe-unpickler test fails on a pandas `ArrowStringArray`. The last line of that log is "SUMMARY {'PASS': 20, 'ERROR': 1}".

## Behaviour of the original pipeline (simulated) {#sec:results-orig}

All results in this section were regenerated with the current code (package version 1.1.0) by `thesis/tools/run_original_pipeline.py`. They use simulated telemetry and simulated re-entries.

**Health monitor.** Over all 8 subsystems and 5 fault types (excluding `stuck`, which crashes the simulator), the classical monitor raised a sustained alarm in 29 of 40 faulted 24-hour runs. It raised no alarm on the fault-free run (false alarm: False). Detection depended strongly on the fault type ([@tbl:monitor-grid]):

- slow *degradation* was detected in 1 of 8 subsystems;
- step-like *dropouts* were detected in 7 of 8, with a median delay of 6 minutes.

Detection also depends on the subsystem. Thermal Control was detected in 0 of its 5 faulted runs, the data storage unit in 5 and every other subsystem in 4. Apart from slow degradation, which is missed almost everywhere, every miss is on Thermal Control. The likely reason lies in the simulator's constants (`CHANNELS`, l.25–42). A fault's deviation scales with the channel's noise level $\sigma$, while the robust scale of the fitting window is dominated by the orbital sinusoid of amplitude $A$. Thermal Control is the only subsystem in which both channels have $\sigma/A \le 0.11$ (0.04 and 0.11), so its faults stay small in $z$. Every other subsystem has a channel with $\sigma/A \ge 0.13$. [@fig:monitor-grid] shows every run.

![Detection delay of the classical health monitor for every subsystem and fault type (SIMULATED telemetry, seed 7, 24-hour runs). Hatched cells were not detected within the run. Source: `thesis/evidence/original_pipeline_runs.json`.](figures/orig_monitor_grid.png){#fig:monitor-grid width=85%}

: Classical health monitor on simulated telemetry, seed 7: number of subsystems in which each fault type was detected within 24 h, and median detection delay. {#tbl:monitor-grid}

| Fault | Detected | Median delay (min) |
|---|---|---|
| degradation | 1 / 8 | 600 |
| leak | 7 / 8 | 505 |
| overheat | 7 / 8 | 348 |
| dropout | 7 / 8 | 6 |
| saturation | 7 / 8 | 357 |

**Coupling.** The hazard rule maps the same anomaly score to different break-up physics depending on the driving subsystem. In the scenario table, at anomaly score 0.85:

- a thermal-driven scenario gives an aerodynamic break-up at 69.8 km, with $P_{\text{frag}}$ = 0.129;
- a power-driven scenario gives an explosion at 76.6 km.

The all-nominal scenario is run at anomaly score 0.05, with every degradation at the same value 0.05. Its $P_{\text{frag}}$ is 0.009. Because all degradations tie, `max()` returns the first subsystem in the dictionary, Power System. The nominal case is therefore labelled an explosion at 79.3 km. This tie-break is a defect of the rule (Appendix C).

In the end-to-end comparison of failure modes ([@tbl:failure-modes]), the propulsion scenario produces an explosion at 81.8 km with CEP50 = 458.4 km. The thermal scenario produces an aerodynamic break-up at 71.3 km with CEP50 = 386.1 km.

The table also exposes a limitation of the coupling. The *structure* scenario is driven by "Data Storage Unit", because the structural fault is not detected and the subsystem with the largest residual degradation becomes the driver. Its footprint (CEP50 433.3 km) is nearly identical to that of the *nominal* scenario (432.9 km).

: Failure-mode comparison through the whole chain (simulated, 600 Monte Carlo samples per scenario). {#tbl:failure-modes}

| Scenario | Driver | Break-up | Altitude (km) | $P_{\text{frag}}$ | $\sigma_{\log B}$ | Mean downrange (km) | CEP50 (km) |
|---|---|---|---|---|---|---|---|
| nominal | Data Storage Unit | aerodynamic | 77.9 | 0.021 | 0.18 | 796.7 | 432.9 |
| thermal | Thermal Control | aerodynamic | 71.3 | 0.154 | 0.186 | 632.1 | 386.1 |
| structure | Data Storage Unit | aerodynamic | 77.9 | 0.091 | 0.179 | 798.2 | 433.3 |
| propulsion | Propulsion System | explosion | 81.8 | 0.592 | 0.254 | 902.8 | 458.4 |
| comms lost | Communication | aerodynamic | 77.6 | 0.109 | 0.182 | 788.6 | 431.5 |

**Impact zones.** For one simulated footprint (1200 impacts, CEP50 452.3 km), the split-conformal density zone achieved near-nominal held-out coverage:

| Nominal coverage | Conformal density zone | Gaussian ellipse | Area saving of the density zone |
|---|---|---|---|
| 0.90 | 0.9004 | 0.9122 | 60.4 % |
| 0.95 | 0.9437 | 0.945 | 59.8 % |
| 0.99 | 0.9846 | 0.9856 | 57.4 % |

At nominal 0.95 the density zone needs 59.8 % less area than the Gaussian ellipse. On this sample the Gaussian ellipse is itself close to nominal: its coverage shortfall is -1.22, 0.5 and 0.44 percentage points at the three levels, where a negative value means over-coverage. The gain from the conformal zone is therefore a smaller area, not better coverage. The earlier review's statement of "0.9927 versus 0.9750 coverage in 17 % less area" is therefore not reproduced by the current code.

**Two-frame detection (synthetic).** On 16 synthetic scenes, the registered difference with the photon-noise map gave precision 0.987, recall 0.563 and F1 0.717. The single-frame baseline gave precision 0.11 and F1 0.1974.

**Catalogue and lifetimes (real catalogue).** The SATCAT snapshot of 2026-08-16 contains 70,288 objects, 35,834 of them debris. The censoring-aware area-to-mass fit used 7,711 fragments from four break-up clouds: 4,652 decayed and 3,059 censored. It gave an area-to-mass ratio of 0.1491 m² kg⁻¹, which is 1.049× smaller than a fit to decayed fragments only. The ratio of modelled to measured median lifetime was:

- 1.001 for Iridium 33 and 0.863 for Cosmos 2251, both at 789 km;
- 0.292 for Cosmos 1408 (480 km), so the model under-predicts the lifetime of that low cloud by a factor of about three.

With a single area-to-mass ratio and a static atmosphere, the model cannot fit clouds at different altitudes and solar epochs. The earlier review's figure of "1.03× on 1,806 fragments" is not reproduced.

**Risk.** For the four predefined corridors, the per-fragment expected casualties ranged from 0 over the open ocean to 3.37e-04 over peninsular India. The corresponding event-level values computed by `corridor_risk` were 0.061 for India and 0.032 for central Europe. These values assume that every fragment survives to the ground, and they rest on 300 Monte Carlo samples per corridor. They are artefacts of the simulated break-up and of the risk defects listed in Appendix C, not risk assessments of any real object.

## ESA-ADB: reproduction and label-free baselines {#sec:results-baselines}

[@tbl:baselines] compares the label-free baselines with the published values. The ported official GlobalSTD rule reproduces the published primary metric to within 0.001 in all eight configurations. Two Mission 2 cells differ in the third decimal because of rounding. This checks the whole chain: download, preprocessing, scoring adapter and official metric code. The parameter-free out-of-limit rule fails on Mission 1. On Mission 2 it scores 0.894 (full channels) and 0.965 (lightweight subset), above every value published with the benchmark.

: Corrected event-wise $F_{0.5}$ of label-free baselines on the official test halves. Published values (DMLR Table 4 [@kotowski2026esaadb]) are in parentheses. {#tbl:baselines}

| Detector | M1 full | M1 lightweight | M2 full | M2 lightweight |
|---|---|---|---|---|
| best published | 0.061 | 0.786 | 0.079 | 0.949 |
| GlobalSTD3 (official rule) | 0.0004 (<0.001) | 0.0011 (0.001) | 0.0075 (0.008) | 0.0071 (0.007) |
| GlobalSTD5 (official rule) | 0.0029 (0.003) | 0.253 (0.253) | 0.063 (0.064) | 0.075 (0.075) |
| out-of-limit, nominal range | 0.0007 | 0.202 | 0.894 | 0.965 |
| out-of-limit, full range | 0.0051 | 0.000 | 0.261 | 0.080 |
| COSMOS monitor (as shipped) | 0.0038 | 0.202 | 0.047 | 0.788 |

On Mission 1 the out-of-limit rule raises 11,627 false alarms per year. Brancato et al. report 0.754 for an out-of-limit reference on Mission 1 [@brancato2027ast]. That value is not reproduced under the official protocol as implemented here. The likely reason is a difference in protocol or in the definition of the limits, which cannot be checked without their implementation.

## ESA-ADB: exploratory study (H1–H3) {#sec:results-explo}

[@fig:esa-explo] shows the exploratory results on the full channel sets.

![Exploratory study: corrected event-wise $F_{0.5}$ on the official test halves (full channel sets). Bars show seed means; whiskers show the standard deviation where more than one seed exists. The dashed line marks the best published value.](figures/esa_exploratory.png){#fig:esa-explo width=100%}

On **Mission 2**, every forecaster combined with conformal-BH scored far above the published baselines:

- AR(8): 0.962;
- PatchTST: 0.952 ± 0.009;
- zero-shot Chronos-Bolt: 0.945.

These arms detect only 16.7 % of the anomalies, however. Their score comes from the rare nominal events, which dominate the Mission 2 test half.

On **Mission 1**, no arm was usable:

- the best deterministic arm (AR(8) with a threshold) reached 0.088;
- AR(8) with conformal-BH reached 0.074, at 92 false alarms per year;
- the only higher mean, 0.318 ± 0.215, came from the TCN with telecommands and the rolling layer. Its three seeds ranged from 0.072 to 0.464, so it was not a stable arm.

On the lightweight Mission 1 subset, by contrast, AR(8) with conformal-BH reached 0.408. The failure on Mission 1 was a failure at full channel scale.

[@tbl:explo-boot] gives the paired bootstrap comparisons.

: Exploratory paired block-bootstrap comparisons (seed 0, 30-day blocks). "Difference" is the re-implemented $F_{0.5}$ difference A − B with its 95 % interval. {#tbl:explo-boot}

| Comparison | A (official) | B (official) | Difference | 95 % CI |
|---|---|---|---|---|
| H1 M1: patchtst vs COSMOS AR(8), conformal-BH | 0.062 | 0.074 | -0.011 | [-0.031, +0.009] |
| H1 M1: tcn vs COSMOS AR(8), conformal-BH | 0.049 | 0.074 | -0.025 | [-0.048, +0.006] |
| H1 M1: chronos-bolt-small vs COSMOS AR(8), conformal-BH | 0.024 | 0.074 | -0.052 | [-0.075, -0.032] |
| H1 M2: patchtst vs COSMOS AR(8), conformal-BH | 0.941 | 0.962 | -0.021 | [-0.051, +0.007] |
| H1 M2: tcn vs COSMOS AR(8), conformal-BH | 0.936 | 0.962 | -0.027 | [-0.054, +0.002] |
| H1 M2: chronos-bolt-small vs COSMOS AR(8), conformal-BH | 0.945 | 0.962 | -0.018 | [-0.047, +0.014] |
| H2 M1: ar8 conformal-BH vs threshold | 0.074 | 0.088 | -0.012 | [-0.027, +0.000] |
| H2 M2: ar8 conformal-BH vs threshold | 0.962 | 0.882 | +0.080 | [+0.041, +0.133] |
| H2 M2: patchtst conformal-BH vs threshold | 0.941 | 0.817 | +0.124 | [+0.080, +0.177] |
| H3 M1: tcn + telecommands vs without, conformal-BH | 0.019 | 0.049 | -0.033 | [-0.069, -0.012] |
| H3 M2: tcn + telecommands vs without, conformal-BH | 0.948 | 0.936 | +0.013 | [+0.000, +0.025] |
| H3 M1: patchtst + telecommands vs without, conformal-BH | 0.020 | 0.062 | -0.045 | [-0.070, -0.026] |
| H3 M2: patchtst + telecommands vs without, conformal-BH | 0.951 | 0.941 | +0.010 | [-0.011, +0.033] |

**Summary of the exploratory hypotheses.**

- **H1 is not supported.** No deep or pretrained encoder beat AR(8) under the same calibration. On Mission 1, Chronos-Bolt was significantly worse.
- **H2 holds on Mission 2 only.** Conformal-BH beat the per-channel threshold on Mission 2 by +0.080 for AR(8). On Mission 1 the difference was -0.012.
- **H3 is not supported.** H3 predicted fewer alarm runs touching only rare events when the encoder receives telecommands. Seed means of those runs (conformal-BH):
  - TCN: 101 without and 91 with telecommands on Mission 1, and 399 and 393 on Mission 2;
  - PatchTST: 97 and 207 on Mission 1, and 353 and 356 on Mission 2.

  The reductions are small, and PatchTST doubles these runs on Mission 1. The primary metric agrees: telecommands lowered both encoders on Mission 1 (-0.033 and -0.045, both intervals below zero), and changed them by about +0.01 on Mission 2. A trained gated model is also insensitive to its telecommand input: removing the stream at test time left PatchTST on Mission 2 at 0.951 (with the stream: 0.951). This mirrors the pre-registered negative result of TC-CAD [@sahin2026tccad].

**A fixed telecommand rule.** A simple rule was also tested: suppress alarms for 30 minutes after any priority-3 telecommand. On Mission 2 it reduced alarm runs that touch only rare events from 351 to 42, and anomaly recall stayed at 0.167. Because the primary metric counts rare events as targets, the primary score fell at the same time from 0.962 to 0.304.

## ESA-ADB: pre-registered confirmatory study {#sec:results-prereg}

### Test-half results of the arms

[@tbl:prereg-arms] and [@fig:prereg-arms] give the test-half results under the primary conformal-BH layer.

: Pre-registered arms on the official test halves, conformal-BH layer (mean ± sd over seeds; A0 and A1 are deterministic). "CV" is the selection objective, i.e. the cross-fitted training-half $F_{0.5}$ of the chosen setting. {#tbl:prereg-arms}

| Mission | Arm | Test $F_{0.5}$ | Precision | Recall | Anomaly-only $F_{0.5}$ | False alarms / yr | CV (training half) |
|---|---|---|---|---|---|---|---|
| 1 | A0 | 0.074 | 0.061 | 0.462 | 0.032 | 91.6 | — |
| 1 | A1 | 0.428 | 0.667 | 0.176 | 0.336 | 1.1 | 0.624 |
| 1 | A2 | 0.544 ± 0.236 | 0.627 | 0.415 | 0.327 | 8.5 | 0.727 |
| 1 | A3 | 0.205 ± 0.114 | 0.211 | 0.266 | 0.125 | 20.9 | 0.536 |
| 2 | A0 | 0.962 | 0.982 | 0.892 | 0.147 | 3.4 | — |
| 2 | A1 | 0.874 | 1.000 | 0.580 | 0.000 | 0.0 | 0.919 |
| 2 | A2 | 0.904 ± 0.050 | 0.975 | 0.710 | 0.129 | 3.5 | 0.890 |
| 2 | A3 | 0.875 ± 0.058 | 0.958 | 0.676 | 0.180 | 7.7 | 0.906 |

![Pre-registered arms, conformal-BH layer. Bars show seed means; dots show individual seeds. The dotted line marks the out-of-limit rule.](figures/esa_prereg_arms.png){#fig:prereg-arms width=100%}

The seed spread is large for the supervised arms on Mission 1:

- trees range from 0.157 to 0.769;
- the TCN ranges from 0.085 to 0.358.

The best tree seed (seed 4, 0.769) raises 0.1 false alarms per year. Seed 1 (0.157) raises 33.9. No single seed is representative; the confirmatory tests use all of them.

### Confirmatory tests

[@tbl:confirmatory] and [@fig:forest] report the confirmatory tests.

: Confirmatory comparisons (official seed-mean difference; 95 % interval and p from the paired hierarchical block bootstrap with 2,000 replicates). H4: A3 > A1; H5: A3 > A2; H6: A1 > A0. Mission 1 is the pre-registered family (Holm); Mission 2 is a replication outside the family, with unadjusted p-values. Freeze `09a98f5e160e`. {#tbl:confirmatory}

| Mission | Test | A | B | Δ (official) | Δ (re-impl.) | 95 % CI | $p_{\text{Holm}}$ | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | H4 | 0.205 | 0.428 | -0.222 | -0.222 | [-0.395, +0.042] | 0.1010 | **refuted** |
| 1 | H5 | 0.205 | 0.544 | -0.338 | -0.330 | [-0.540, -0.033] | 0.0700 | **refuted** |
| 1 | H6 | 0.428 | 0.074 | +0.353 | +0.352 | [+0.206, +0.488] | 0.0015 | **supported** |
| 2 | H4 | 0.875 | 0.874 | +0.001 | +0.001 | [-0.055, +0.054] | — | replication |
| 2 | H5 | 0.875 | 0.904 | -0.029 | -0.029 | [-0.103, +0.037] | — | replication |
| 2 | H6 | 0.874 | 0.962 | -0.089 | -0.089 | [-0.143, -0.049] | — | replication |

![Confirmatory and exploratory differences with 95 % bootstrap intervals. The dotted line marks the minimum effect size of interest (+0.05).](figures/esa_forest.png){#fig:forest width=90%}

The verdicts follow the pre-registered rules.

- **H6 is supported on Mission 1.** The A1 protocol scored +0.353 above A0. Every bootstrap replicate favoured A1, so the raw $p$ sits at the floor of 0.0005 and the Holm-adjusted value is $p$ ≤ 0.0015. The selected layer is strict: $q$ = 0.0001 and $k$ = 10.
- **H4 and H5 are refuted.** Under the pre-registered rule, *refuted* means that the upper end of the 95 % interval lies below the minimum effect of +0.05. It does not mean that the TCN is shown to be worse. Its point estimates are below both comparators, but the H4 interval includes zero ([-0.395, +0.042]) and the Holm-adjusted $p$ for H5 is 0.07.
- **The Mission 2 replication reverses H6.** Selection on the whole training half was worse than selection on the 4.5-month calibration window, by -0.089.

**What H6 compares.** A0 and A1 differ in more than the selection data, so H6 is a comparison of two complete protocols:

| | A0 | A1 |
|---|---|---|
| AR(8) fitted on | the fitting period (2000-01-01 to 2006-01-01) | the whole training half, out of fold |
| Calibration pool | scores of the calibration period | out-of-fold scores of the whole training half |
| Residual scale | out-of-sample residuals of the early-stopping period | in-sample residuals over the training half |
| Selection data | the calibration period (9 target events) | the whole training half (105 target events) |
| Selection grid | the exploratory grid: $q \in$ [0.01, 0.001, 0.0003, 0.0001], $k \in$ [1, 3, 10, 30] | the pre-registered grid: $q \in$ [0.01, 0.001, 0.0001, 1e-05], $k \in$ [1, 3, 10, 30, 100] |

The pre-registration states that all arms use the same grids. A0 was instead taken from the exploratory study, unchanged. This inconsistency is logged as a deviation (Appendix B). A later code review found a fifth difference, the residual scale; it and the fitting data change the test scores themselves. The study cannot attribute the gain to any one of the five differences.

**Alarm burden.** False alarms fell from 91.6 to 1.1 per year, and alarm runs of any kind from 113 to 18 per year. The false-alarm count understates the operator's load. Most A1 alarm runs fall inside an event that has already been alarmed. Alarming precision is 0.139: of the alarm runs that touch an event, only about one in seven is the first on that event. Mission 1 events often last days, longer than the 1 h merge gap, so one event can produce many runs. Recall is 0.176: A1 detects about one target event in six. [@fig:alarm-burden] sets these quantities side by side for every arm.

![Alarm burden of the pre-registered arms (conformal-BH layer, seed means). Each bar splits the alarm runs per year of test half into runs that touch a target event and false-alarm runs; the label gives the total and the number of target events detected (recall times the number of test events). Source: `oof_protocol/test/*.json`.](figures/esa_alarm_burden.png){#fig:alarm-burden width=100%}

### Cross-validation versus test

The selection objective over-estimated test performance for every Mission 1 arm, but by very different amounts:

| Arm | CV (training half) | Test |
|---|---|---|
| AR(8), A1 | 0.624 | 0.428 |
| trees, A2 | 0.727 | 0.544 |
| TCN, A3 | 0.536 | 0.205 |

[@fig:cv-test] shows the same comparison for both missions and for the lightweight Mission 1 arm, whose cross-validated value fell from 0.528 to 0.025 on test. On Mission 2 every arm lies close to the diagonal.

![Selection objective against test result for every arm with training-half selection (conformal-BH layer, seed means). Points below the dashed line did worse on test than in cross-validation. Sources: `oof_protocol/selection/*.json`, `oof_protocol/test/*.json`.](figures/esa_cv_vs_test.png){#fig:cv-test width=65%}

[@fig:landscape] shows the cross-validated landscape over the selection grid. For AR(8) on Mission 1 the optimum is a narrow ridge at $k = 10$. The trees' landscape is broad, with many settings within a few hundredths of the best. The TCN's (seed 0) is flat and low. On Mission 2 the AR(8) landscape is a broad plateau.

![Training-half cross-validated $F_{0.5}$ over the selection grid of the conformal-BH layer (training labels only).](figures/esa_selection_landscape.png){#fig:landscape width=100%}

## Alarm layers, ablations and the point-level comparator {#sec:results-ablations}

[@tbl:layers] compares the alarm layers on identical scores.

: Test-half $F_{0.5}$ by alarm layer (seed means). ACI is the point-level adaptive conformal comparator; "ACI + events" adds the persistence and merging of the conformal-BH layer. {#tbl:layers}

| Mission | Arm | threshold | conformal-BH | rolling BH | ACI | ACI + events |
|---|---|---|---|---|---|---|
| 1 | A1 | 0.428 | 0.428 | 0.101 | 0.0001 | 0.008 |
| 1 | A2 | 0.613 | 0.544 | 0.087 | 0.0000 | 0.046 |
| 1 | A3 | 0.282 | 0.205 | 0.323 | 0.0001 | 0.003 |
| 1 | A3, no residual inputs | 0.276 | 0.249 | 0.000 | 0.0001 | 0.005 |
| 1 | A3, two classes | 0.363 | 0.196 | 0.084 | 0.0001 | 0.000 |
| 1 | A1, lightweight channels | 0.350 | 0.025 | 0.130 | 0.0001 | 0.000 |
| 2 | A1 | 0.977 | 0.874 | 0.356 | 0.0007 | 0.000 |
| 2 | A2 | 0.819 | 0.904 | 0.244 | 0.0007 | 0.013 |
| 2 | A3 | 0.853 | 0.875 | 0.487 | 0.0014 | 0.005 |
| 2 | A1, lightweight channels | 0.978 | 0.978 | 0.529 | 0.0017 | 0.000 |

![Test-half $F_{0.5}$ of the same scores under three alarm layers (seed means; values of [@tbl:layers]). Where two markers coincide, the layers gave the same value (A1 on Mission 1; A1 lightweight on Mission 2). ACI is omitted because all its values are below 0.002.](figures/esa_layers.png){#fig:layers width=85%}

Four findings stand out.

1. **Point-level ACI fails at the event level.** Point-level ACI kept its point-wise false-alarm rate on nominal test data between 0.0098 and 0.0100 for the target $\alpha$ = 0.01. Its primary metric never exceeded 0.0017 over 30 runs, because it produced up to 1,597,643 alarm runs. With event logic it reached 0.117 in a single tree seed; the best seed mean of any arm was 0.046, and 10 of 30 runs raised no alarm at all. The reason is mechanical. The selection chose $\gamma$ = 0.05 in 29 of 30 runs. With that value, one exceedance moves $\alpha_t$ from 0.01 to -0.0395 ([@eq:aci]). The quantile then becomes infinite, and the channel cannot fire again for 80 steps. In the remaining run (A1, Mission 1, $\gamma$ = 0.01) one exceedance lowers $\alpha_t$ to 0.0001, so the next one needs a far more extreme score. A channel therefore cannot stay hot through an event. Because the global alarm is an OR over channels, persistence can still be met by different channels firing in successive steps, but that happens rarely. The comparison is also not like-for-like: ACI's $\gamma$ was tuned to a point-wise false-alarm rate, while the BH layer's $(q, k)$ were tuned on the event metric. The result concerns ACI at $\alpha$ = 0.01, not point-level calibration in general; tuned on the event metric post hoc, ACI is competitive on some arms ([@sec:results-posthoc]).
2. **No layer dominates.** The per-channel threshold, which has no cross-channel multiplicity control, equalled or beat conformal-BH in 8 of the 10 rows. Conformal-BH was better only for the trees and the TCN on Mission 2. For A1 on Mission 1 the two layers give the same value. The rolling layer copies the $(q, k)$ selected for conformal-BH rather than being selected itself. It cut AR(8) on Mission 1 from 0.428 to 0.101 and the trees from 0.544 to 0.087. It raised only the TCN, whose seeds spread widely under it, and the lightweight Mission 1 arm, whose conformal-BH result had collapsed (0.025 to 0.130).
3. **The TCN ablations change little.** Neither removing the residual inputs nor merging the two event classes changed the TCN by more than its seed spread (exploratory comparisons below).
4. **Mission 1 lightweight channels.** On the lightweight Mission 1 subset, the conformal-BH layer selected on the training half scored only 0.025, with recall 0.031. Its cross-validated value had been 0.528. The selection chose $q$ = 0.01, at the edge of the grid, with $k$ = 30. The threshold layer scored 0.350. Both are far below the best published lightweight value (0.786) and below the exploratory calibration-window arm (0.408).

**The selection effect across channel sets and layers.** [@tbl:seleff] sets the confirmatory H6 result beside the same contrast on the other channel sets and layers. These are point estimates on single deterministic runs, not tests.

: Training-half selection minus calibration-window selection for AR(8) (A1 − A0; on the lightweight subsets, A1 minus the exploratory lightweight AR(8) arm). Only the Mission 1 full-channel conformal-BH cell is a pre-registered test. {#tbl:seleff}

| Channel set | Layer | Mission 1 | Mission 2 |
|---|---|---|---|
| full | conformal-BH | +0.353 | -0.089 |
| full | threshold | +0.340 | +0.094 |
| lightweight | conformal-BH | -0.383 | +0.070 |
| lightweight | threshold | -0.056 | +0.157 |

![The selection contrast of [@tbl:seleff]. The ringed point is the pre-registered H6 comparison; the dotted line marks the minimum effect of interest (+0.05). All other points are exploratory point estimates without intervals.](figures/esa_selection_effect.png){#fig:seleff width=95%}

The gain on Mission 1 at full scale is large under both layers, so it does not depend on BH. Elsewhere the sign changes. On Mission 2 it is negative only under BH. On the lightweight Mission 1 subset it is negative under both layers, because the selection settled on a setting that did not transfer to the test half.

## Exploratory diagnostics (not pre-registered)

**Trees against AR(8).** The trees outperformed AR(8) on Mission 1 in the point estimate, by +0.116 with the official metric. With seeds resampled, the re-implemented difference is +0.108 (95 % CI [-0.133, +0.344]), which is not distinguishable from zero. On Mission 2 the official difference is +0.030 and the re-implemented one +0.030 [-0.016, +0.081].

**Score drift between halves.** [@fig:shift] compares, on nominal samples, how often each scorer exceeds the $1-10^{-4}$ quantile of its own out-of-fold pool on the test half against how often it does so in cross-validation on the training half:

- AR(8): ratio 0.8;
- trees: ratios 5.1–9.3;
- TCN: ratios 2.8–54.2.

The supervised scorers, and above all two TCN seeds, shift their nominal score distribution between training and test: a calibrated layer then raises more alarms on the test half than the selection expected. The shift does not explain the differences in test score, however:

- TCN seed 1 has the second-largest shift (34.8×) and the best TCN score (0.358);
- TCN seed 2 has the smallest shift (2.8×) and scores 0.135;
- the trees shift as much as TCN seeds 2–4 (5.1–9.3× against 2.8–6.3×), yet score far higher.

The rank correlation between shift and test $F_{0.5}$ is 0.00 over the TCN seeds, -0.60 over the tree seeds and -0.12 over all ten supervised runs. With five seeds per arm these correlations are descriptive only, but they give no support to shift as the cause of the TCN's low scores. The shift is a property of the supervised scorers that an operator would have to monitor. It is not shown to be what separates good from bad runs ([@fig:drift-score]).

![Exploratory: calibration shift against test-half $F_{0.5}$ for every supervised run on Mission 1 (conformal-BH layer). Each point is one seed. Sources: `oof_protocol/exploratory/calibration_shift_m1.json`, `oof_protocol/test/*.json`.](figures/esa_drift_vs_score.png){#fig:drift-score width=85%}

![Exploratory: nominal test exceedance rate of the $1-10^{-4}$ out-of-fold calibration quantile, relative to the training-half cross-validation rate (Mission 1, every fourth channel).](figures/esa_calibration_shift.png){#fig:shift width=85%}

**Event classes.** [@fig:classes] breaks detection down by event class.

- No arm detected any of the 3 Mission 1 test anomalies whose class never occurs in training.
- For seen-class anomalies, recall was 0.15 (AR(8)), 0.20 (trees) and 0.23 (TCN).
- The trees' advantage over AR(8) came mainly from rare nominal events. They detected 0.68 of seen-class and 0.88 of unseen-class rare events, against 0.29 and 0.00 for A1. Of their roughly 21 additional detections per seed, about 11 are seen-class rare events, 7 unseen-class rare events and 2.6 anomalies.
- The unseen-class rare events are not detected through learned classes. The unsupervised A0 detects the same share of them (0.875) as the trees, while A1 detects none. They are large deviations that a lenient layer catches and A1's strict layer misses. The trees combine recall of this kind with a better false-alarm record.
- On anomalies alone, the trees' lead over A1 disappears ([@fig:primary-anom]).

![Primary $F_{0.5}$ (anomalies and rare events) against anomaly-only $F_{0.5}$ for the pre-registered arms (conformal-BH layer, seed means). The labels give primary / anomaly-only. Mission 2's anomaly-only values rest on six anomalies.](figures/esa_primary_vs_anomaly.png){#fig:primary-anom width=100%}

![Exploratory: share of Mission 1 test events detected, by category and by whether the event class occurs in the training half (seed means, conformal-BH layer).](figures/esa_event_classes.png){#fig:classes width=85%}

**Ablations of A3.** Compared with the full A3 (three seeds each):

- dropping the residual inputs changed the score by +0.039 [-0.188, +0.278];
- using two classes changed it by -0.012 [-0.260, +0.164].

## ESA-ADB: post-hoc validation analyses (P1–P8) {#sec:results-posthoc}

After the confirmatory analysis, a second plan (`research_design/preregistration/POSTHOC_PLAN_V2.md`) fixed eight further analyses, their code checks and their interpretation rules. It was frozen by hash (217648e6805f, 2026-10-25T12:14:37) before any of them was run. All eight use test halves that had been scored before, so every result in this section is exploratory. What the plan controls is which analyses were run and how they are read.

**P3, exact bootstrap.** The frozen bootstrap used an approximate block-wise metric. P3 replaces it with an exact decomposition that accumulates, per 30-day block, detected and total events, false-positive runs, nominal time and alarmed nominal time (`scripts/posthoc_v2/exact_metric.py`):

- it matched the official code on 302 stored arm–metric pairs, with a largest difference of 2.1e-16;
- with it, the H6 interval is [+0.207, +0.491] and H5's Holm-adjusted p-value 0.066;
- all three confirmatory verdicts are unchanged.

**P8, provenance.** Re-running the frozen selection for A1 reproduced the saved choice and every grid value on both missions (largest difference 0).

**P1, decomposition of H6.** Ten cells, each a full protocol scored once, switch the five differences between A0 and A1 one at a time, in the order G, F, σ, P, D (forward) and in the reverse order. In all six settings the first and last cells reproduced A0's and A1's official values exactly. [@fig:ph-decomp] shows four of the settings.

- Mission 1, BH: the selection data carry the contrast, +0.301 [+0.156, +0.436] forward and +0.366 [+0.234, +0.500] reverse, the only factor the rule calls material. The pool added +0.043 on both paths and the fit data +0.010.
- The residual scale had no effect in any setting. Both layers compare a test score with nominal scores of the same channel, so a per-channel rescaling shared by pool and test cancels.
- The grid was order-dependent: under whole-half selection the extended grid allowed a stricter level and cost -0.065; with the original grid, whole-half selection would have scored 0.493.
- Mission 1, threshold: pool and selection data were both material (+0.145 and +0.197).
- Mission 1, lightweight channels: the reversal comes from the same two factors, the pool (-0.150 and -0.223) and the selection data (-0.166).
- Mission 2, BH: the reversal also comes from the selection data, -0.088 [-0.134, -0.054] on both paths; every other factor changed the score by less than 0.005.

Neither of the two differences that change the scores (fit data and residual scale) explains the gain. The H6 contrast is, in effect, a contrast of selection data.

![Post hoc (P1): change in test-half event-wise $F_{0.5}$ when one of the five differences between A0 and A1 is switched, forward (A0 to A1) and reverse (A1 to A0), with exact block-bootstrap 95 % intervals. Dotted line: minimum effect of interest. Source: `posthoc_v2/p1_decompose/*.json`.](figures/esa_posthoc_decomposition.png){#fig:ph-decomp width=100%}

**P2, forward chaining.** Folds inside the training halves (Mission 1: the years 2003–2006; Mission 2: three quarters of 2001) compare a window protocol W, which mirrors A0, with a history protocol H, which mirrors A1, using only data before each fold ([@fig:ph-forward]).

- Mission 1: mean H − W +0.240, but H won 2 of 4 folds, so by the pre-set rule (at least 3 of 4) the gain does not replicate. The mean comes from 2005, when W raised 1,173 false alarms per year. In 2004 H raised no false alarm but also detected no event.
- Mission 2: H won all 3 folds, by a mean of +0.017, below the minimum effect.
- Each Mission 1 fold holds 9–10 events, so single detections move a fold's score a long way.

![Post hoc (P2): forward-chaining folds, window protocol W against history protocol H (conformal-BH layer). Labels: fold $F_{0.5}$ and false alarms per year. Source: `posthoc_v2/p2_forward/m*_summary.json`.](figures/esa_posthoc_forward.png){#fig:ph-forward width=100%}

**P4, event-tuned ACI.** With $(\alpha, \gamma, k)$ selected on the event objective, ACI on A1 (Mission 1) chose $\gamma$ = 0.0001 and scored 0.472, above BH (0.428). Over the 7 arms, ACI with $\gamma > 0$ reached or exceeded the threshold layer on 2 and stayed at least 0.05 below it on 5. The plan's rule therefore does not allow the claim that ACI is unusable at the event level. The pre-registered collapse ([@sec:results-ablations]) came from tuning $\gamma$ to a point-wise rate.

**P5, the registered anomaly-score arm.** Selected and scored on anomalies only (source: `posthoc_v2/p5_anomaly_arm/m*_summary.json`):

- Mission 1: AR(8) 0.336; the trees' anomaly log-odds 0.087 (range 0.011–0.232, 44.7 false alarms per year); the TCN's 0.097 (0.034–0.211);
- Mission 2, with six test anomalies: AR(8) 0.000, trees 0.247, TCN 0.005.

Supervised anomaly scores therefore did not improve anomaly detection on Mission 1, where the training half holds 63 labelled anomalies.

**P6, seed extension.** Seeds 5–9 of A2 and A3 were trained with the frozen code (source: `posthoc_v2/p6_seeds/summary.json`):

- trees: 0.488 ± 0.195 over 10 seeds (range 0.157–0.769);
- TCN: 0.176 ± 0.093 over 9 seeds (range 0.082–0.358);
- with all seeds (exact bootstrap, unadjusted): TCN − AR(8) -0.252 [-0.402, -0.043]; TCN − trees -0.313 [-0.457, -0.112]; trees − AR(8) +0.061 [-0.138, +0.268].

The confirmatory five-seed verdicts are unchanged; the extension narrows the intervals around the same conclusion.

**P7, operational metrics.** On Mission 1, A1 detected 16 of 91 target events with a median delay of 0.33 h, and raised 0.15 false alarms per 1000 nominal hours. A0 detected 42 at 11.7 per 1000 h. The TCN's runs were the longest (median 5.1 h) and its detections the latest (median 2.0 h).


## Supporting studies {#sec:results-support}

### OPSSAT-AD (real)

[@tbl:opssat] and [@fig:opssat] give the OPSSAT-AD results.

: OPSSAT-AD test results (mean ± sd over 5 seeds where applicable; threshold 0.5). {#tbl:opssat}

| Model | F1 (official) | AUC-PR (official) | F1 (time-blocked) | AUC-PR (time-blocked) |
|---|---|---|---|---|
| hybrid CNN + features | 0.949 ± 0.009 | 0.986 | 0.910 | 0.972 |
| feature MLP (FCNN) | 0.937 ± 0.008 | 0.980 | 0.907 | 0.967 |
| 1-D CNN | 0.939 ± 0.015 | 0.987 | 0.867 | 0.971 |
| patch transformer | 0.898 | 0.948 | 0.880 | 0.942 |
| XGBoost | 0.911 | 0.961 | 0.894 | 0.964 |
| random forest | 0.909 | 0.964 | 0.888 | 0.956 |
| AdaBoost | 0.838 | 0.925 | 0.880 | 0.944 |
| logistic regression | 0.863 | 0.957 | 0.876 | 0.962 |
| Chronos-Bolt probe | 0.889 | 0.955 | 0.808 | 0.919 |
| COSMOS robust-z | 0.436 | 0.471 | 0.315 | 0.673 |
| Isolation Forest | 0.226 | 0.346 | 0.172 | 0.331 |
| published FCNN [@ruszczak2025opssat] | 0.946 | 0.979 | — | — |

![OPSSAT-AD F1 at threshold 0.5 under the official and time-blocked splits (5 seeds).](figures/opssat_f1.png){#fig:opssat width=100%}

**Main comparisons.**

- The re-implemented FCNN reproduces the published model: F1 0.937 against 0.946 published. AdaBoost reaches 0.838 against 0.836 published.
- The hybrid network has the highest F1 on the official split; the 1-D CNN has a marginally higher AUC-PR (0.987 against 0.986). In a paired bootstrap on seed-averaged scores the hybrid exceeds the FCNN in AUC-PR by +0.0059 [+0.0004, +0.0127]. In F1 the difference is not significant (+0.015 [-0.008, +0.041]).
- The hybrid exceeds XGBoost by +0.025 in AUC-PR [+0.008, +0.046].
- On the 18 tabular features alone, the feature MLP beats XGBoost: F1 +0.032 [+0.004, +0.063] and AUC-PR +0.020 [+0.002, +0.039]. On this dataset, trees do not dominate tabular data.

**Time-blocked split.** Under this split the deep models drop and the ranking compresses: the hybrid scores 0.910, the FCNN 0.907 and XGBoost 0.894. Not every model drops: AdaBoost rises from 0.838 to 0.880 and logistic regression from 0.863 to 0.876. The official split's advantage for the deep models is partly an advantage of interleaved training and test fragments.

**Label efficiency.** At 10 % of the training labels, the AUC-PR values are:

- FCNN 0.936;
- CNN 0.912;
- hybrid 0.910;
- XGBoost 0.891;
- Chronos-Bolt probe 0.823.

[@fig:opssat-curves] shows the label-efficiency curves together with the robustness results described next.

![OPSSAT-AD label efficiency and robustness. Left: test AUC-PR against the share of training labels (official split; note the y-axis starts at 0.8). Centre and right: test F1 under added noise and under sample dropout; the 0 level is the unperturbed run of the robustness study. Means over 5 seeds. Sources: `summary_official.csv`, `robustness_summary.csv`.](figures/opssat_curves.png){#fig:opssat-curves width=100%}

**Robustness.** The models were perturbed at test time with noise (σ as a fraction of each fragment's standard deviation) and with sample dropout:

- **Noise at 0.1σ.** F1 fell to 0.856 for the hybrid, 0.807 for the FCNN, 0.766 for XGBoost and 0.568 for the CNN.
- **Larger noise.** At 0.25σ and above, all models fell to about 0.3–0.4.
- **30 % sample dropout.** XGBoost was the most robust (0.851); the deep models dropped to 0.672–0.766.

### Learned health monitor (SIMULATED)

The learned monitor was compared with the classical monitor on the held-out seed sets 5,6,7,8,9, using simulated telemetry:

| Monitor | F1 | Precision | Recall | Localisation accuracy | False alarms (fault-free runs) |
|---|---|---|---|---|---|
| classical EWMA | 0.834 ± 0.009 | 1.000 | 0.715 | 0.830 | 0 |
| learned (PatchTST + conformal) | 0.775 ± 0.053 | 0.929 | 0.665 | 0.960 | 3 |

The learned monitor localises the faulted subsystem more often. It detects fewer faults and raises more false alarms. Both monitors would cap their healthy window at the true fault onset if the onset came before the end of the window. With the simulator's default onset at 55 % of the record and the window at 40 %, the cap never binds, so the results do not use ground truth (Appendix C).

### SpotGEO (real) and synthetic scenes (SYNTHETIC)

[@fig:vision] and [@tbl:spotgeo] give the vision results.

![Vision results on SpotGEO and on the synthetic scenes. Left: official SpotGEO F1 on the 5,120 test sequences. Right: COCO mAP@50 on the synthetic test scenes (SYNTHETIC).](figures/vision_results.png){#fig:vision width=100%}

: SpotGEO results on the full test set with the official scorer (mean ± sd over seeds where several exist). The challenge winner is from [@chen2021spotgeo]. {#tbl:spotgeo}

| Model | Inputs | Training data | F1 | Precision | Recall | Official MSE (sum) |
|---|---|---|---|---|---|---|
| COSMOS front end + DoG (classical) | differences | — | 0.084 | 0.049 | 0.308 | — |
| heat-map U-Net | both | 100 % | 0.746 ± 0.005 | 0.846 | 0.667 | 174,793 |
| heat-map U-Net | high-pass only | 100 % | 0.751 | 0.881 | 0.654 | 174,208 |
| heat-map U-Net | differences only | 100 % | 0.644 | 0.749 | 0.565 | 251,127 |
| heat-map U-Net | both | 25 % | 0.663 | 0.854 | 0.542 | 208,472 |
| heat-map U-Net | both | 10 % | 0.535 | 0.763 | 0.411 | 279,284 |
| ResNet-18 encoder (pretrained) | both | 100 % | 0.765 ± 0.033 | 0.862 | 0.687 | 155,488 |
| challenge winner | — | — | 0.948 | — | — | 33839 |

**SpotGEO findings.**

- The learned detectors improve on the classical COSMOS front end by an order of magnitude in F1. They remain well short of the challenge winner, whose solution added temporal post-processing [@chen2021spotgeo].
- The "differences only" input is the weakest learned variant, and the high-pass-only input comes close to the combined one. This suggests that, once all five frames are given to the network, the COSMOS registration-and-differencing front end adds little. The input ablations were run with one seed each, so the support is weak.
- With 10 % or 25 % of the training sequences, F1 falls to 0.535 and 0.663.

**Synthetic scenes.** On the synthetic scenes, which carry no evidence about real sensors, the COCO-style scores (mAP@50 / mAP@50–95) are:

- heat-map U-Net: 0.780 / 0.370;
- YOLOv8n: 0.431 / 0.244;
- existing TinyYOLO: 0.379 / 0.162.

### Re-entry landmark (real catalogue)

[@tbl:landmark] and [@fig:landmark] give the landmark results.

: Re-entry landmark on the test launch groups (6,371 objects, 38 events). {#tbl:landmark}

| Model | AUC-PR 30 d | AUC-PR 68 d | AUC-ROC 68 d | C-index | Brier 68 d | ECE 68 d |
|---|---|---|---|---|---|---|
| XGBoost | 0.761 | 0.821 | 0.998 | 0.997 | 0.0025 | 0.0022 |
| XGBoost-AFT$^\dagger$ | 0.658 | 0.813 | 0.997 | 0.997 | 0.0026 | 0.0021 |
| hazard MLP ensemble | 0.610 | 0.688 | 0.996 | 0.996 | 0.0033 | 0.0024 |
| hazard MLP without physics feature | 0.587 | 0.662 | 0.995 | 0.995 | 0.0033 | 0.0024 |
| logistic regression | 0.528 | 0.527 | 0.991 | 0.990 | 0.0039 | 0.0017 |
| COSMOS physics lifetime | 0.254 | 0.188 | 0.977 | 0.976 | 0.0056 | 0.0015 |

$^\dagger$ The AFT predictions use all boosting rounds rather than the early-stopped iteration, because of a defect (Appendix C). The AFT row therefore does not represent the intended configuration.

![Re-entry landmark: AUC-PR for re-entry within 30 and 68 days on the test launch groups.](figures/reentry_landmark.png){#fig:landmark width=100%}

**Model comparisons.**

- Trees outperform the neural discrete-time hazard model. At 68 days the difference is -0.134 [-0.221, -0.024], from a launch-group bootstrap.
- On the full test set, the learned models greatly outperform the physics lifetime: XGBoost exceeds physics by +0.633 [+0.427, +0.779]. Most re-entries in the population are payloads (173 of 235), many of them controlled, which a drag-only lifetime cannot represent.
- The physics feature adds a little to the MLP: +0.026 [+0.004, +0.065].
- On the *uncontrolled* subset (3,001 objects, 12 events), the order changes at the short horizon. At 30 days physics has the highest AUC-PR (0.512), against 0.294 for logistic regression, 0.202 for XGBoost and 0.136 for the MLP. At 68 days XGBoost (0.464) is ahead of physics (0.204), and the defective AFT model is highest (0.549). The 30-day comparison rests on 4 positive objects, so it is not conclusive. It suggests that the learned models' advantage comes mainly from objects whose re-entry is not governed by drag ([@fig:landmark-subsets]).

![Re-entry landmark: AUC-PR on all test objects and on the uncontrolled subset, at 30 and 68 days. The positives in each set are given in the panel titles; with four positives, the 30-day uncontrolled values are highly uncertain. Source: `reentry_landmark/results.json`.](figures/reentry_subsets.png){#fig:landmark-subsets width=100%}

- The AUC-ROC and C-index values are close to 1 for every model. Most objects are far from re-entry, so these measures say little. AUC-PR is the informative measure here.

**Conformal sets at 68 days ($\alpha$ = 0.1).**

- *Marginal sets* reach the nominal overall coverage (XGBoost 0.911). They cover the events themselves only 0.000 of the time (MLP: 0.132), because events are rare.
- *Label-conditional (Mondrian) sets* cover events with probability 0.974 and non-events with probability 0.916 for XGBoost.

### Propagator surrogate (SIMULATED)

All errors in this section are measured against the COSMOS propagator, not against real re-entries.

**In-range test set.** On 40,000 test cases the surrogate ensemble reproduced the reference propagator with:

- median impact-position error 16.2 km (95th percentile 69.4 km);
- median relative downrange error 3.2 %;
- fall-time mean absolute error 41.3 s;
- impact classification accuracy 0.995.

Its throughput was 80,107 samples per second.

**Out-of-range test set.** With break-up altitudes of 100–120 km, the median error rose to 38.1 km and the 95th percentile to 362 km. The ensemble spread separates out-of-range from in-range cases with AUROC 0.726.

**Footprints.** Over 12 simulated footprint scenarios:

- the median relative error of CEP50 was 1.7 % (maximum 5.2 %);
- the surrogate's 95 % conformal zone covered a median 0.953 (minimum 0.927) of the reference impacts;
- the median speed-up was 994×.

These scenarios clip their inputs into the training ranges, so they do not test the fallback.

**Inside the pipeline.** Mean over 3 seeds:

| Fallback mode | Seconds per footprint | CEP50 (km) |
|---|---|---|
| reference only | 11.69 | 485 |
| `none` (surrogate only) | 0.10 | 499 |
| `range` | 8.72 | 499 |
| `spread` | 8.87 | 491 |

In `range` mode the surrogate handles 99.0 % of samples. The remainder still dominate the run time, so any fallback gives up most of the speed-up.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Discussion {#sec:discussion}

## RQ1: what the integrated chain establishes

The verification suite shows that the physics components implement their equations ([@tbl:verify]). The checks differ in strength:

- **10 compare with an answer from outside the code.** For example, the US76 density matches the published tables to 0.0019 %, impact speed reaches the terminal-velocity limit, the registration recovers imposed shifts with the correct sign, and the conformal zones reach their nominal coverage.
- **9 are self-consistency checks or hold by construction.** These include the convergence of the impact point to 0.060 m as the tolerance tightens, and the mass balance of the break-up model.
- **The hydrostatic check is mis-specified** (Appendix C).

These results establish the components as implementations of their equations. They do not establish that the equations are adequate for real re-entries. The atmosphere is static, the break-up model is a simplified and partly non-standard variant, and no recorded re-entry with recovery coordinates was available to test predicted landing locations.

The chain does connect health to re-entry physics. The subsystem that fails selects the break-up type, altitude and dispersion ([@tbl:failure-modes]). Running the chain, rather than reading it, reveals five limitations of the coupling.

1. **It depends on detection.** When the monitor misses a fault, the subsystem with the largest residual degradation becomes the driver instead of the faulted one. Slow degradation is missed in all but 1 of 8 subsystems, and every fault on Thermal Control is missed. The resulting footprint is indistinguishable from a nominal one.
2. **Ties are broken by dictionary order.** In the all-nominal scenario every degradation is equal. The rule then names Power System as the driver and labels the case an explosion.
3. **The probabilities are not propagated.** $P_{\text{frag}}$ and $\sigma_{\text{state}}$ are computed but not used. A break-up is propagated for every scenario, and the risk is not weighted by its probability.
4. **The debris feature is not connected.** It stands outside the chain, so the claim that "the objects catalogued in orbit are the same objects that later come down" is not implemented.
5. **The rules are hand-set.** The coupling weights are expert choices. They have not been fitted to, or checked against, recorded fragmentation events. The `discos.py` module was written for that check, but the DISCOS data it needs require an account.

The answer to RQ1 is therefore partial. The chain exists and its parts implement their equations. The coupling, however, is a hand-set rule whose outputs are only partly used. Its scientific value would rest on validation against recorded fragmentations and re-entries.

## RQ2: what determines event-wise performance on ESA-ADB

The pre-registered study gives a clear answer for Mission 1 at full channel scale. Elsewhere the answer is conditional.

### The selection result

On Mission 1 the A1 protocol raised AR(8) from 0.074 to 0.428, and H6 is supported. Three observations qualify what this shows.

1. **It is a protocol effect.** A0 and A1 differ in five respects ([@sec:results-prereg]): the forecaster's fitting data, the residual scale, the calibration pool, the selection data (9 against 105 target events) and the grid. Two of them change the test scores themselves. The post-hoc decomposition ([@sec:results-posthoc]) apportions the gain: the selection data carry it (+0.301 and +0.366 on the two paths), the pool adds +0.043, and neither score-changing difference is material.
2. **It does not depend on BH.** Under the per-channel threshold the same contrast is +0.340 ([@tbl:seleff]).
3. **It does not transfer to every setting.** On the lightweight Mission 1 subset the contrast is negative under both layers. On Mission 2 it is negative under BH and positive under the threshold. Of the 8 contrasts in [@tbl:seleff], 5 favour training-half selection. The cross-validated value over-estimated the lightweight BH result by more than an order of magnitude (0.528 against 0.025 on test). Its selected level lies at the edge of the grid.

The selection effect is therefore real and large for Mission 1 at full scale, where the calibration window was too small to select from. It is not a general rule. A post-hoc reading is that more labelled history helps when the recent window is uninformative, and can hurt when a setting tuned to older data does not transfer to the test period. The post-hoc analyses fit that reading without testing it: the lightweight and Mission 2 reversals come from the same factor (the selection data), and in forward-chaining folds inside the training halves the history protocol won only 2 of 4 Mission 1 folds ([@sec:results-posthoc]).

**What the selected layer does.** The selected layer is strict ($q$ = 0.0001, $k$ = 10). It trades recall (0.176) for precision (0.667), which is what $F_{0.5}$ with a true-negative-rate correction rewards. The benchmark authors credit the relative success of their best Mission 1 baseline to its dynamic thresholding and pruning, a comparable emphasis on suppressing false detections [@kotowski2026esaadb].

For an operator, the strict layer means three things:

- about one false alarm per year;
- about one target event in six detected;
- about 18 alarm runs per year, most of them repeat alarms inside long events.

How many runs reach the operator therefore depends as much on the merging rule as on the detector.

### The deep supervised scorer

The TCN event scorer had full access to the training labels, a receptive field of 1021 steps and cross-channel mixing. It scored 0.205, below the forecaster (-0.222) and below the trees (-0.338). Neither difference reached the minimum effect, so H4 and H5 are refuted. Neither is significant in the other direction either: the H4 interval includes zero, and the Holm-adjusted $p$ for H5 is 0.07. The supported conclusion is that the deep scorer showed no benefit. It is not that the deep scorer is shown to be worse.

Four explanations are compatible with the data. None was pre-registered, and the study cannot separate them.

- **Distribution shift.** The supervised scorers' nominal test exceedances were 2.8–54× (TCN) and 5.1–9.3× (trees) their cross-validated rate, against 0.8× for AR(8). This can explain why cross-validation over-estimates the supervised arms. It does not explain the differences between runs: the rank correlation of shift with test score is -0.12 over the ten supervised runs. The trees shift as much as three of the five TCN seeds (5.1–9.3× against 2.8–6.3×), yet score far higher. Two plausible sources of shift remain: drift of Mission 1 over its record (2000-01-01 to 2014-01-01), and the difference between the final model and the fold models that produced the calibration pool.
- **Capacity relative to labels.** The TCN has 220,739 parameters and learns from 109 annotated events, 63 of them anomalies. It detects fewer unseen-class rare events (0.125) than the trees or the unsupervised A0 (0.875 each). That pattern is consistent with a model that has fitted the recurring classes. The two ablations, without residual inputs and with two classes, changed little, so the result is not tied to those design choices.
- **Seed instability.** The TCN seeds range from 0.085 to 0.358 (sd 0.114). With five seeds, the arm mean is uncertain by roughly ±0.1 or more at 95 %.
- **The layer.** Under the per-channel threshold the TCN scored 0.282. The layer choice changes the size of the gap to the trees but not its direction.

### Trees, AR(8) and what the primary metric rewards

The trees' mean of 0.544 is the best Mission 1 result of the study. Their seed range (0.157–0.769) makes the difference from AR(8) statistically indistinguishable from zero. The event-class analysis locates the advantage. Of the trees' roughly 21 additional detections per seed over A1:

- about 11 are seen-class rare events;
- 7 are unseen-class rare events, which the unsupervised A0 also detects;
- 2.6 are anomalies.

On anomalies alone the two arms are equal: anomaly-only $F_{0.5}$ is 0.327 for the trees and 0.336 for AR(8). The trees are therefore better at the benchmark's combined target, not better at finding anomalies. No arm detected any of the 3 anomalies of unseen classes.

### Mission 2

On Mission 2, whole-half selection was worse than the calibration window under BH (-0.089) but better under the threshold (+0.094). The reversal is therefore a property of the BH selection on this mission, not of the selection data as such.

The mission leaves little room for any effect. Its calibration window holds 43 rare events and immediately precedes the test half. The parameter-free out-of-limit rule already scores 0.894. The four A0 and A1 values lie between 0.874 and 0.977. Its cross-validated landscape is a broad plateau ([@fig:landscape]), on which near-equivalent settings are separated by small differences that need not transfer. Mission 2 provides no evidence either way on the selection-data question.

## RQ3: event-level versus point-level calibration

On identical scores, point-level ACI held its point-wise false-alarm rate at the target. Its primary metric never exceeded 0.0017 ([@tbl:layers]), because it produced up to 1,597,643 alarm runs on a test half. This agrees with the TC-CAD study, which found the same collapse with Telemanom scores on a smaller channel set [@sahin2026tccad]. In that study ACI also exceeded its target false-alarm rate, by 1.15–3.4× according to its deposit; here it did not. The present study adds three points.

- **The collapse occurs with other scores.** It occurs with forecaster and supervised scores, at full channel scale.
- **Event logic does not rescue ACI.** With persistence and merging, the best seed mean was 0.046, and 10 of 30 runs raised no alarm.
- **The reason is mechanical.** With the $\gamma$ selected in 29 of 30 runs, one exceedance silences the channel for 80 steps. A channel therefore cannot stay hot through an event, and persistence must be assembled from different channels firing in turn, which is rare ([@sec:results-ablations]).

A fair event-level ACI would select $\gamma$ on the event metric, or apply persistence before the ACI update. Neither was tested.

**What makes the alarms usable.** In this study, usable alarms come from aggregation to events (persistence and merging), together with a strict operating point selected on the event metric. Multiplicity control across channels is not shown to be necessary. The per-channel threshold, which has none, equalled or beat conformal-BH in 8 of 10 rows, and it gave the same Mission 1 selection effect. The robust conclusion concerns the level of aggregation and the selection, not BH as such.

**No formal guarantee applies here.** BH is run across channels whose p-values come from separate per-channel pools and whose scores are correlated. The guarantee of Bates et al. covers test points that share one calibration pool [@bates2023], and exchangeability over time fails. In this setting, the conformal-BH layer is a scale-free family of thresholds indexed by $(q, k)$, not a procedure with false-discovery-rate control.

**The rolling layer is weak evidence.** The rolling layer recalibrates on trailing unlabelled test scores. It lowered AR(8) and the trees sharply. It raised only the lightweight Mission 1 arm, whose conformal-BH result had collapsed, and the TCN, whose seeds then ranged from 0.000 to 0.569. Its $(q, k)$ were copied from the conformal-BH selection rather than selected for it. It is therefore weak evidence about drift in either direction.

## RQ4: deep versus classical models on the supporting real datasets

The three supporting real-data studies do not show a single pattern by model family.

- **OPSSAT-AD.**
  - On the official split, the deep models lead. The hybrid has the highest F1 and the 1-D CNN the highest AUC-PR. Even on the 18 tabular features, the feature MLP beats XGBoost (F1 +0.032, interval above zero).
  - Under the time-blocked split, which is the more realistic one because the official split interleaves training and test fragments in time, the deep models drop and two classical models rise. The ranking compresses.
  - Gradient-boosted trees are the most robust model to sample dropout.
- **SpotGEO.** The heat-map networks are an order of magnitude better than the classical COSMOS front end. Without temporal post-processing, they remain far from the challenge winner (0.746 against 0.948).
- **Re-entry landmark.**
  - Trees beat the neural hazard model (-0.134 at 68 days), consistent with the general finding for medium-sized tabular data [@grinsztajn2022tree].
  - The physics lifetime, far behind on the full test set, has the highest 30-day AUC-PR on uncontrolled objects, whose re-entry drag governs. That comparison rests on 4 positive objects.

The common thread with RQ2 is that model capacity was rarely the binding constraint. The evaluation protocol changed the rankings more than the choice between model families did: the split on OPSSAT-AD, and the alarm layer and selection data on ESA-ADB.

## RQ5: the propagator surrogate

The surrogate reproduces the COSMOS propagator closely inside its training ranges:

- median position error 16.2 km;
- median CEP50 error 1.7 % over 12 footprint scenarios;
- about 994× faster.

Its ensemble spread is only a moderate out-of-range signal (AUROC 0.726). Inside the pipeline, any fallback to the reference propagator removes most of the speed-up, because a few reference samples dominate the run time. All of these statements concern agreement with the simulator. None concerns real re-entries.

## Relation to the literature

**Comparison with the published baselines.** The official baselines [@kotowski2026esaadb] set the reference on the full channel sets. Our Mission 1 numbers under the same protocol are higher, at 0.428 (AR(8)) and 0.544 (trees), against a published best of 0.061. Three facts limit what this comparison shows.

- **The use of labels differs.** The published baselines did not tune their thresholds on training labels. A1 selects its layer on those labels, and A2 and A3 are supervised. The comparison shows what label-based selection can achieve under the official protocol. It does not show that these scorers are better detectors.
- **Mission 2 is saturated.** The published best full-channel value is 0.079, but the parameter-free out-of-limit rule scores 0.894. The large margin over the published values therefore says more about those values than about our scorers.
- **The advantage does not extend to the lightweight subset.** On the lightweight Mission 1 subset the best published value is 0.786 (Telemanom-ESA-Pruned). The training-half-selected AR(8) reached only 0.350 (threshold) and 0.025 (BH), and the calibration-window arm 0.408.

**Other protocols.** Our numbers are lower than those reported under other protocols:

- 0.700 with monthly retraining over the whole mission [@alcarria2026];
- 0.853 on the Kaggle split [@allegrini2026];
- 0.902 with SMED, a lightweight similarity-search detector [@brancato2027ast].

The protocols differ in retraining, splits and channel selection, so these numbers do not establish a ranking. At least one of these methods, monthly retraining, adapts to new data over time. Adaptation is a plausible direction for future work, although the drift diagnostic here does not by itself show that drift limits the present scorers.

**Calibration and triage.** The TC-CAD study's negative result on telecommand conditioning is replicated here with different detectors [@sahin2026tccad]. Its finding on point-level conformal calibration is extended to the full channel sets, with a mechanical explanation for why event logic does not repair it.

## Practical implications

For an operator choosing an anomaly monitor on telemetry like ESA-ADB's, the evidence supports five practical steps.

1. **Select at the event level.** Evaluate and select alarm settings at the event level, on as much labelled history as can be scored out of sample. Then check the selected setting on a later held-out period, because under BH the selection did not transfer on 2 of the 4 mission and channel-set combinations.
2. **Report the alarm burden.** Report false alarms per year, alarm runs per year and alarming precision next to any $F$ score. With events that last days, the merging rule determines much of the alarm count.
3. **Monitor supervised scorers.** Treat the calibration of supervised scorers as perishable, and monitor their nominal exceedance rate after deployment.
4. **Do not rely on depth.** Do not expect a deeper model to compensate for poor alarm selection.
5. **Report anomaly-only metrics.** The primary metric counts rare nominal events as targets, so a score near 0.9 on Mission 2 can coexist with detecting only a few of its 6 anomalies. On Mission 1, the trees' lead disappears on anomalies alone. Anomaly-only metrics should always be reported alongside the primary score.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Limitations and Future Work {#sec:limitations}

## Threats to validity {#sec:threats}

**Internal validity of the pre-registered study.**

- **Prior exposure of the test halves.** Both missions' test halves had been scored in the exploratory study for other configurations. The new arms had never been scored. However, the decision to study alarm selection on Mission 1 was taken after Mission 1 had failed. This is a researcher degree of freedom that the pre-registration documents but cannot remove.
- **Cross-fitting optimism.** Out-of-fold models are trained on later as well as earlier blocks, so training-half scores are more favourable than deployed scores. The gap between cross-validation and test in [@tbl:prereg-arms] is consistent with this.
- **Selections on grid edges.** Several selections lie on an edge of the pre-registered grid, at $q = 10^{-5}$ or $10^{-2}$, or at $k = 100$. The grid was not widened after the freeze, so better settings may lie outside it.
- **Too few seeds for the supervised arms.** Five seeds are few for arms whose Mission 1 results span 0.157–0.769 (trees). The hierarchical bootstrap accounts for this, but the intervals are wide.
- **Procedural deviations.** 8 deviations are logged (Appendix B). Five are procedural and changed no computation. The sixth concerns the design: arm A0 used the exploratory selection grid, not the pre-registered one, so H6 compares two complete protocols.
- **Confounded H6 contrast.** A0 and A1 differ in fitting period, residual scale, calibration pool, selection data and grid; the first two change the test scores. The confirmatory result cannot be attributed to any one of them.
- **Small bootstrap block counts.** The test halves contain 86 (Mission 1) and 22 (Mission 2) 30-day blocks. With so few blocks, the Mission 2 intervals are themselves uncertain.

**Construct validity.**

- **Rare events count as targets.** The primary metric treats rare nominal events as targets. A detector can therefore score well on Mission 2 while missing most of its 6 anomalies. Anomaly-only metrics are reported for this reason.
- **Exchangeability is violated, and the pools are per channel.** Conformal levels are used as tuning parameters, not as guarantees.
- **The pre-registered point-level comparator is not like-for-like.** ACI's $\gamma$ was tuned to a point-wise rate, the BH layer's $(q, k)$ to the event metric. Tuned post hoc on the event metric (P4), ACI reached 0.472 on A1 (Mission 1), so the pre-registered collapse reflects the tuning target; it still trailed the threshold layer on 5 of 7 arms.

**External validity.**

- **One benchmark.** All telemetry conclusions rest on one benchmark with two usable missions, and Mission 2 is close to saturated. The selection effect also changes sign between the full and lightweight Mission 1 channel sets.
- **Unseen event classes.** 4 Mission 1 test event classes are absent from training.
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
3. **Forward chaining on new data.** The post-hoc forward-chaining folds did not replicate the gain on Mission 1 (2 of 4 folds); a prospective test on a later data release is the decisive check.
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


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Conclusion {#sec:conclusion}

This thesis documented COSMOS as it is currently implemented. COSMOS is a framework that chains simulated satellite health, break-up and re-entry physics, and adds a machine-learning layer evaluated on real public data. The thesis reported every result through an evidence file that traces each number to its source.

**Methodology.**

- The physics components were checked by an automated suite: 10 of its checks compare with external references, and the rest are self-consistency checks.
- The behaviour of the chain was characterised with fixed seeds.
- The learned components were evaluated on ESA-ADB, OPSSAT-AD, SpotGEO and a leakage-checked CelesTrak landmark, under official or leakage-controlled protocols and against strong classical baselines.
- The principal study was pre-registered and hash-frozen before scoring.

**Principal findings.**

1. **Whole-training-half selection was supported on Mission 1.** On ESA-ADB under the official protocol at full channel scale, the protocol that selects an event-level alarm layer on out-of-fold scores of the whole training half raised Mission 1 from 0.074 to 0.428 (H6 supported, Holm-adjusted $p$ ≤ 0.0015). The gain held under a per-channel threshold as well. The two arms differ in five respects, two of which change the scores, so the gain belongs to the protocol as a whole; a post-hoc decomposition traced it to the selection data, and forward-chaining folds inside the training half did not replicate it. The resulting layer raises about one false alarm per year, but about 18 alarm runs in total, and it detects about one target event in six.
2. **The supervised deep event scorer showed no benefit.** Its point estimates were below both the AR(8) forecaster and gradient-boosted trees on the same features (H4 and H5 refuted: neither reached the minimum effect). An exploratory diagnostic found that the supervised scorers' nominal score distributions shift between training and test, but the size of the shift did not track the test score across runs.
3. **The selection effect is conditional.** It reversed on the near-saturated Mission 2 under the BH layer, but not under the threshold. It also reversed on the lightweight Mission 1 subset under both layers. Mission 2 is already handled well by a parameter-free out-of-limit rule (0.894).
4. **Point-level adaptive conformal calibration failed at the event level** on both missions, and adding event logic did not repair it, for a mechanical reason. What made alarms usable was aggregation to events with a strict operating point. A per-channel threshold did as well as cross-channel BH in most comparisons.
5. **The supporting studies show no single pattern by model family.** Deep models led on OPSSAT-AD's official split, even on its tabular features, but the advantage shrank under a time-blocked split. Trees beat the neural hazard model on the re-entry landmark, and the physics lifetime was competitive on uncontrolled objects at 30 days. Deep heat-map detectors improved on the classical optical front end but remained far from the SpotGEO winner.
6. **The physics pipeline implements its equations component by component.** Its coupling of health to re-entry is a hand-set rule whose outputs are only partly used, and it has not been validated against real events.

**Implications.** For spacecraft telemetry evaluated with operator-grade metrics, the alarm layer, the data used to select it and the stability of the score distribution mattered more than model capacity. A selection that works on one channel set may fail on another, so the selected setting should be checked on a held-out period. Claims that deep models solve such benchmarks should be tested under the official protocol, against equally well-calibrated simple scorers, and with anomaly-only metrics reported.

**Remaining challenges.**

- A factorial test of what drives the selection effect, and drift-aware scoring, both on unseen data.
- Validation of predicted re-entry locations against recorded events.
- Correction of the documented defects in the physics chain.

The roadmap in Appendix D describes how the telemetry study can be turned into a focused journal manuscript.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# References {.unnumbered}

::: {#refs}
:::


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Appendix A: Reproducibility {#sec:appendix-repro .unnumbered}

The repository is `https://github.com/AnuSajikumar6264/Cosmos-AI-`, at commit `2525367`. Large data, score arrays and checkpoints are excluded from git (see `.gitignore`) and must be regenerated or obtained separately.

**Environment.** Python 3.12, plus PyTorch with a back end for the available hardware (here Intel XPU), then `pip install -r requirements.txt -r requirements-ai.txt`.

**Data.**

```bash
python -m cosmos.ai.data.download opssat esa_adb spotgeo celestrak   # checksummed, with manifests
python -m cosmos.ai.data.preprocess_telemetry                        # official ESA-ADB grids
python -m cosmos.ai.data.validate                                    # data/DATASET_REPORT.{json,md}
```

**Verification and tests.**

```bash
python tests/test_verify.py      # physics verification checks (prints PASS/FAIL)
python tests/test_ai.py          # COSMOS-AI checks (also under pytest)
```

**ESA-ADB, exploratory study.**

```bash
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --baseline global_std5
python -m cosmos.ai.telemetry.esa_benchmark --mission 1 --encoder ar8
python -m cosmos.ai.telemetry.esa_benchmark --mission 1 --encoder tcn --seed 0
```

**ESA-ADB, pre-registered study.**

```bash
python -m cosmos.ai.telemetry.oof --mission 1
python -m cosmos.ai.telemetry.event_model --mission 1 --seed 0
python -m cosmos.ai.telemetry.event_gbt --mission 1 --seed 0
python scripts/prereg_analysis.py select
python scripts/prereg_analysis.py test        # refuses unless the freeze hashes match
python scripts/prereg_analysis.py stats       # -> oof_protocol/CONFIRMATORY.{json,md}
```

**Supporting studies.**

```bash
python -m cosmos.ai.telemetry.opssat --config configs/telemetry/opssat.yaml
python -m cosmos.ai.vision.train_spotgeo --model unet --inputs both --seed 0
python -m cosmos.ai.survival.landmark
python -m cosmos.ai.surrogate train && python -m cosmos.ai.surrogate evaluate
python -m cosmos.ai.telemetry.learned_monitor train && python -m cosmos.ai.telemetry.learned_monitor heldout
```

**This thesis.**

```bash
python thesis/tools/run_original_pipeline.py   # regenerates original-pipeline evidence
python thesis/tools/build_evidence.py          # thesis/evidence/evidence.json + evidence_map.md
python thesis/tools/make_figures.py            # result figures from the evidence only
python thesis/tools/make_diagrams.py           # architecture diagrams (graphviz)
python thesis/tools/render.py                  # thesis/build/COSMOS_Thesis.docx
```

`render.py` aborts if any number in the text has no evidence entry.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Appendix B: Pre-registration and deviations {#sec:appendix-prereg .unnumbered}

The pre-registration is in `research_design/preregistration/PREREG_TELEMETRY_V1.md`. It states:

- what was known before it was written;
- the question;
- the data and metrics;
- the arms;
- the selection rule;
- hypotheses H4–H6, with test, family, minimum effect size and verdict rules;
- the seeds and stopping rule;
- what would change the claims;
- the known threats;
- a deviation log.

It was frozen with 28 files at 2026-10-24T15:56:29 as `09a98f5e160e`, and reissued as `681cc7ddec86` once the deviation log was filled in. The confirmatory results file records the freeze under which they were computed (`09a98f5e160e`). 8 deviations are logged in `research_design/preregistration/DEVIATIONS.md`. The first five are procedural:

1. The test pass was run by a parallel driver that claims each arm atomically, instead of a sequential loop. The same frozen function scored each arm once.
2. The sequential test process was stopped after it had scored arm A1 on Mission 1.
3. Training-half selection for the late-finishing arms (Mission 2 TCN seeds 3–4 and the six ablation arms) ran after other arms had been scored. Four workers had crashed on the missing selection files before scoring anything, and their locks were removed.
4. The arm-A0 alarms for Mission 2 were regenerated with the frozen function, after the first statistics run failed on the missing file. The function is deterministic and the bootstrap seed fixed.
5. The deviation log was filled in and the freeze reissued.

None of these changed a computation or a selection.

The sixth was found during the review for this thesis. It is an inconsistency in the design, not a change to it. Section 3 of the pre-registration says that all arms use the pre-registered grids. Arm A0, however, is by definition the exploratory arm, and it carries its exploratory selection on the grid $q \in$ [0.01, 0.001, 0.0003, 0.0001] × $k \in$ [1, 3, 10, 30]. A0 and A1 therefore differ in at least four respects (fitting period, calibration pool, selection data and grid), and H6 compares two complete protocols. A later code review found a fifth: the residual scale of the AR(8) score (logged as a further row of the deviation log). The same review found that the secondary arm with anomaly-only scores, listed in section 3, was neither run nor reported as not run. It is reported here as not run.

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Appendix C: Defect and discrepancy register {#sec:defects .unnumbered}

The register was compiled from two independent read-only code audits, from re-running the pipeline, and from comparing the documentation with the code. Defects are listed, not fixed, because several affect results that have already been reported. Fixing them would require new, separately reported runs.

: Defects in the code. "Effect" states which reported results are affected. {#tbl:defects}

| # | Location | Defect | Effect |
|---|---|---|---|
| 1 | `health/telemetry.py:125` | `stuck` fault has noise multiplier 0.6, so a negative noise scale; the call raises `ValueError` | `stuck` excluded from all monitor benchmarks |
| 2 | `health/monitor.py:129–130`, `learned_monitor.py:188–189` | healthy window capped at the true fault-onset index, i.e. ground truth. With the default onset (55 % of the record) and window (40 %) the cap never binds | none for the reported runs; a latent leak if the onset is moved earlier |
| 3 | `health/hazard.py:71`, `monitor.py:46–48` | driver is the unweighted argmax of degradation; undetected faults are attributed to the subsystem with the largest residual degradation; ties go to the first key, so the all-nominal case becomes a Power System explosion at 79.3 km | coupling results for undetected faults ([@tbl:failure-modes]) and the nominal scenario |
| 4 | `health/hazard.py`, `pipeline.py` | $P_{\text{frag}}$ and $\sigma_{\text{state}}$ computed but unused; a break-up is always propagated | risk is not probability-weighted |
| 5 | `reentry/montecarlo.py:115` | one flight-path-angle draw shared by all samples; configured ballistic-coefficient and wind dispersions unused | footprint spread under-dispersed |
| 6 | `reentry/risk.py:209–235` | per-fragment $E_c$ compared with the per-event threshold; the event-level debris area uses $m/B = C_D A$ instead of $A$, so $A_d$ is $C_D$ = 1.35 times too large; $A_c = (\sqrt{A_d}+0.6)^2$ grows by less | risk bands and corridor values (event level modestly over-estimated) |
| 7 | `reentry/breakup.py:28, 84` | simplified SBM: mass-scaled fragment count, non-standard area-to-mass coefficients, single area law. The $\Delta v$ law uses the collision slope 0.9 in $\log_{10}(A/m)$ for every type, with a type factor (0.2, 0.9, 0.06) in place of the SBM's type-specific slope and offset (0.2χ + 1.85 for explosions, 0.9χ + 2.9 for collisions [@johnson2001sbm]) | explosion fragments receive far smaller $\Delta v$ than SBM fragments; break-up clouds are not standard SBM clouds |
| 8 | `reentry/atmosphere.py` | static US76, constant above 1000 km, no solar or geomagnetic activity; the comment "2 km grid" is wrong | lifetimes and footprints |
| 9 | `debris/lifetime.py:158, 172` | fragments-per-collision formula is not the SBM law; possible double counting of the collision rate | collision criticality |
| 10 | `debris/yolo.py` | `map50` uses centre matching, class-agnostically, without interpolation | not a COCO mAP; COSMOS-AI re-scores with `vision/metrics.py` |
| 11 | `debris/scene.py:184–186` | augmentation summary claims a contrast jitter that is not implemented | documentation only |
| 12 | `survival/landmark.py:301` | XGBoost-AFT predicts with all trees, not the early-stopped iteration | AFT row of [@tbl:landmark] |
| 13 | `data/preprocess_telemetry.py` | forward difference of monotonic channels gives a one-sample look-ahead, inherited from the official scripts | negligible; noted for causality claims |
| 14 | `realdata` tests | safe-unpickler test fails under pandas 3 | one pre-existing test error |
| 15 | landmark, learned monitor, OPSSAT | validation data reused for several decisions | mild optimism of the selection |
| 16 | `fusion.decide` | noisy-OR treats evidence items as independent causes; not wired into the pipeline | not used for any result |
| 17 | `tests/test_verify.py:31–37, 60` | `check()` records PASS or FAIL but never raises, so under pytest every check passes; the hydrostatic check compares $dp/dz$ with $-\rho g_0$ rather than $-\rho g(z)$, so its 0.933 % is the gravity offset at 30 km, and it would fail its own 1 % limit above about 33 km | verification evidence: 9 of the checks are self-consistency checks or hold by construction ([@tbl:verify]) |
| 18 | pre-registered study, arm A0 | A0 carries the exploratory selection grid, not the pre-registered grid (Appendix B) | H6 compares two complete protocols |

: Discrepancies between documentation and code or evidence. {#tbl:discrepancies}

| Source | Statement | Current evidence |
|---|---|---|
| Review-1 presentation | APSGTT–LoRA, diffusion forecasting, federated learning, `coupling.py`, dashboard | not present in the code |
| Review-1 presentation | conformal zone 0.9927 vs Gaussian 0.9750 in 17 % less area | not reproduced ([@sec:results-orig]) |
| Review-1 presentation | lifetime 1.03× on 1,806 fragments | not reproduced (ratio 0.292 for Cosmos 1408) |
| Review-1 presentation / `debris/yolo.py` | YOLOv8n/v11n/v12n mAP@50–95 0.976 / 0.976 / 0.969 | numbers of an external reference pipeline (hard-coded in `reference_benchmark()`), not produced by COSMOS |
| `README.md` | step-size convergence 0.13 m | current test log: 0.060 m |
| `data/DATASET_REPORT.json` | ESA-ADB licence "CC BY 3.0 IGO per the DMLR paper" | the DMLR paper states no licence; the statement is in the arXiv version (2406.17826v2, p. 35) |
| `outputs/benchmarks/SUMMARY.md` | Chronos-2 and LoRA fine-tuning implemented | only Chronos-2 zero-shot loading exists |
| `cosmos/__init__.py` docstring | catalogued objects are the objects that later re-enter | debris is not connected to `run_pipeline` |
| `notebooks/COSMOS_Demo.ipynb` | saved outputs | produced by package version 1.0.0; superseded by `thesis/tools/run_original_pipeline.py` |
| several docstrings (CNN kernel, landmark τ wording, learned-monitor "BH", `report.py` output name) | as listed in the audit | minor; code is authoritative |

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Appendix D: Contribution statement, evidence gaps and journal readiness {#sec:appendix-q1 .unnumbered}

## D.1 Research contribution and gap statement {.unnumbered}

**Gap.** ESA-ADB makes operator-grade evaluation on real telemetry possible. Full-channel results under its official protocol remain low for the published baselines, and the benchmark authors stress false detections and the importance of thresholding and post-processing. Point-level conformal calibration has been shown not to fix it [@sahin2026tccad]. No pre-registered study has compared unsupervised, tree-based and deep supervised event scorers under the official split at full channel scale, with the alarm layer selected on all training labels.

**Contribution.** This thesis contributes such a study. Its outcomes are:

- the whole-training-half selection protocol is supported on Mission 1, under both alarm layers;
- the deep supervised scorer shows no benefit over either simpler scorer;
- the selection effect is conditional: it reverses on Mission 2 under BH and on the lightweight Mission 1 subset;
- aggregation to events with a strict, metric-selected operating point succeeds where point-level adaptive calibration fails, for a mechanical reason;
- a diagnostic, exploratory only, shows that supervised scores shift between training and test, but the shift does not track test performance.

The contribution is empirical and methodological. It is not a new architecture, and the main deep-learning result is negative.

## D.2 Unresolved evidence gaps and missing experiments {.unnumbered}

1. **Brancato et al.'s protocol.** Their Mission 1 numbers (0.902; out-of-limit reference 0.754) cannot be reconciled with this study without their full text and code. *Not verified — the full text of [@brancato2027ast] is required.*
2. **Fresh data for a confirmatory test.** No confirmatory test has been run on data never seen by the project. All test halves were seen in the exploratory phase.
3. **Mechanism of the TCN's failure.** The post-hoc shift diagnostic (Mission 1, every fourth channel) does not order the runs by score, so the cause is open. *A pre-registered test is required.*
4. **Source of the selection effect.** A0 and A1 differ in five respects. *A factorial experiment is required to attribute the H6 gain.*
5. **Re-entry locations.** No validation against recorded re-entries exists. *Recorded re-entry events with recovery coordinates are required* (ESA DISCOS or Space-Track access).
6. **The coupling weights.** These are hand-set and have not been checked against fragmentation records. *DISCOS fragmentation data are required.*
7. **Statistical power.** Supervised-arm seed counts (5; 3 for the ablations) are low relative to the seed variance.
8. **Two literature items not read.** Fejjari et al. (Acta Astronautica) and Dai et al. (Advances in Space Research) were identified in the research design, but their full texts were not accessible here. *The author must read them before submission.*
9. **Unverified references.** RevIN and a telemetry survey by Zeng et al. could not be verified and are not cited.

## D.3 Critical assessment from a reviewer's perspective {.unnumbered}

| Criterion | Assessment |
|---|---|
| Originality and significance | Moderate. The pre-registered, full-channel comparison and the event-versus-point calibration result are new. The calibration-and-triage space overlaps closely with TC-CAD [@sahin2026tccad], and the deep-learning result is negative. |
| Clarity of the gap | Good, provided the paper is framed around alarm selection and score stability rather than around a new model. |
| Technical rigour | High for the telemetry study: official metric code, a reproduced baseline, frozen hashes, gated test scoring, hierarchical bootstrap and Holm correction. The five-way confound of H6 and the point-tuned ACI comparator were addressed post hoc, by a decomposition and an event-tuned comparison ([@sec:results-posthoc]), on test halves already seen. Lower for the physics chain, which has documented defects. |
| Experimental design and data | Real, expert-annotated data; but one benchmark, with one informative mission and two usable missions in total. |
| Baselines | Strong: label-free rules, the official GlobalSTD, trees on the same features, and a published comparator re-implemented. The published baselines did not use training labels, so comparisons with them need that context. |
| Statistical reliability | Adequate for H6. Inconclusive for the comparison of trees with AR(8), because of the seed variance. |
| Reproducibility | High: code public, evidence file, one-command rebuild. Large artefacts are not yet archived with a DOI. |
| Generalisation | Limited. The selection effect reverses on Mission 2 under BH and on the lightweight Mission 1 subset. No test on unseen data. |
| Limitations stated | Yes, extensively. |

**Overall.** In its present form, the telemetry study is publishable as a careful empirical paper. Acceptance by a Q1 venue is plausible but not assured, and it depends on the framing and on items 1–4 above. The physics chain and the supporting studies are thesis material rather than journal contributions in their present state.

## D.4 Prioritised action plan {.unnumbered}

1. **Read and position against Brancato et al.** If possible, reproduce their protocol through this code (gap 1).
2. **Run a factorial H6** that crosses scores, calibration pool and selection data on one grid, to attribute the selection effect (gap 4). Done post hoc as a one-factor-at-a-time decomposition (P1): the selection data carry the effect.
3. **Pre-register and run a drift-aware supervised arm.** Use recalibration selected for that purpose or causal retraining, more seeds, and data not yet used, such as a later ESA release or another mission (gaps 2, 3 and 7).
4. **Add a forward-chaining control** to quantify the optimism of cross-fitting. Done post hoc (P2): the gain did not replicate by the pre-set rule.
5. **Write the manuscript** around one claim: selection and stability of the alarm layer matter more than model capacity. State its conditions (Mission 1 at full scale; reversals elsewhere), report the alarm burden, and report the refuted deep-learning hypotheses as a pre-registered negative result.
6. **Archive the code and score arrays on Zenodo** with a DOI, and add a licence and `CITATION.cff`.
7. **Leave the physics chain and supporting studies out of the paper,** or move them to a separate engineering paper after the defects in [@tbl:defects] are fixed.

## D.5 Converting the thesis into a journal manuscript {.unnumbered}

| Manuscript section | Thesis sources | Action |
|---|---|---|
| Introduction | [@sec:intro], [@sec:gap] | condense to the telemetry problem |
| Related work | [@sec:literature] (ESA-ADB, calibration) | keep [@tbl:lit-esa]; add the two unread papers |
| Method | [@sec:oof], [@sec:model], [@sec:alarm-layer], [@sec:prereg] | as written; add the freeze procedure |
| Results | [@sec:results-baselines], [@sec:results-prereg], [@sec:results-ablations] | [@tbl:baselines], [@tbl:prereg-arms], [@tbl:confirmatory], [@tbl:layers] and figures |
| Analysis | exploratory diagnostics | label as exploratory |
| Discussion and limitations | [@sec:discussion], [@sec:threats] | keep the threats section complete |

The draft in `paper/manuscript.tex` already follows this outline. Its tables are generated by `scripts/build_paper_tables.py`.

**Venues.** The candidate venues, with indicative citation statistics, are listed in `research_design/VENUES.md`. The SJR quartiles there could not be verified from this machine and must be checked on the day of submission. On scope, Engineering Applications of Artificial Intelligence, Acta Astronautica and Reliability Engineering & System Safety fit an empirical study of alarm selection on spacecraft telemetry. The choice should follow the journal's current aims and scope, not its impact factor.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# Appendix E: Evidence map for the numbers in this thesis {.unnumbered}

Every number printed in this thesis was inserted from the evidence file `thesis/evidence/evidence.json` by the build script `thesis/tools/render.py`. That file is generated from the project's result files by `thesis/tools/build_evidence.py`. The table lists each value used, how it was obtained, and its source. *measured* means read from a result or log file; *derived* means computed from measured values by the stated rule; *config* means taken from code or configuration; *literature* means reported by a cited publication. Constants that *define* a method (equation coefficients, grid values, architecture sizes quoted with their module) are taken from the code at commit `2525367` and are not results; they are therefore not listed here.

| Evidence key | Value | Kind | Source |
|---|---|---|---|
| `aci.alpha_after_one_hit.g0.01` | 0.0001 | derived (alpha_t + gamma (alpha - 1) starting from alpha) | `update rule of eq. (aci)` |
| `aci.alpha_after_one_hit.g0.05` | -0.0395 | derived (alpha_t + gamma (alpha - 1) starting from alpha) | `update rule of eq. (aci)` |
| `aci.recovery_steps.g0.05` | 80 | derived (non-exceedance steps until alpha_t > 0 again (alpha_t <= 0 means no alarm)) | `update rule of eq. (aci)` |
| `cfg.evm.batch` | 8 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.class_weights` | [1.0, 5.0, 5.0] | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.d_model` | 64 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.dropout` | 0.1 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.event_fraction` | 0.5 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.features` | ['asinh_z', 'log_s', 'ewm_1h', 'ewm_12h', 'ewm_3d', 'asinh_d | config | `cosmos/ai/telemetry/event_model.py` |
| `cfg.evm.kernel` | 3 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.levels` | 8 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.lr` | 0.001 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.mix_every` | 2 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.steps` | 1500 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.weight_decay` | 0.0001 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.evm.window` | 2048 | config | `cosmos/ai/telemetry/event_model.py: EventConfig` |
| `cfg.explo.ks` | [1, 3, 10, 30] | config | `cosmos/ai/telemetry/alarms.py` |
| `cfg.explo.levels.conformal_bh` | [0.01, 0.001, 0.0003, 0.0001] | config | `cosmos/ai/telemetry/alarms.py` |
| `cfg.explo.levels.threshold` | [0.001, 0.0001, 1e-05, 0.0] | config | `cosmos/ai/telemetry/alarms.py` |
| `cfg.fc.batch` | 256 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.context` | 512 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.d_model` | 64 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.horizon` | 16 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.levels` | 7 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.lr` | 0.001 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.n_layers` | 3 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.patch` | 16 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.patience` | 6 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.steps` | 20000 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.fc.weight_decay` | 0.0001 | config | `cosmos/ai/telemetry/forecasting.py: ForecastConfig` |
| `cfg.gbt.eta` | 0.1 | config | `cosmos/ai/telemetry/event_gbt.py: PARAMS` |
| `cfg.gbt.max_depth` | 8 | config | `cosmos/ai/telemetry/event_gbt.py: PARAMS` |
| `cfg.gbt.n_rows` | 2000000 | config | `cosmos/ai/telemetry/event_gbt.py` |
| `cfg.gbt.nthread` | 64 | config | `cosmos/ai/telemetry/event_gbt.py: PARAMS` |
| `cfg.gbt.objective` | multi:softprob | config | `cosmos/ai/telemetry/event_gbt.py: PARAMS` |
| `cfg.gbt.rounds` | 400 | config | `cosmos/ai/telemetry/event_gbt.py` |
| `cfg.gbt.subsample` | 0.8 | config | `cosmos/ai/telemetry/event_gbt.py: PARAMS` |
| `cfg.periods.m1.cal` | 2006-04-01 to 2007-01-01 | config | `cosmos/ai/telemetry/forecasting.py` |
| `cfg.periods.m1.es` | 2006-01-01 to 2006-04-01 | config | `cosmos/ai/telemetry/forecasting.py` |
| `cfg.periods.m1.fit` | 2000-01-01 to 2006-01-01 | config | `cosmos/ai/telemetry/forecasting.py` |
| `cfg.periods.m2.cal` | 2001-05-15 to 2001-10-01 | config | `cosmos/ai/telemetry/forecasting.py` |
| `cfg.periods.m2.es` | 2001-04-01 to 2001-05-15 | config | `cosmos/ai/telemetry/forecasting.py` |
| `cfg.periods.m2.fit` | 2000-01-01 to 2001-04-01 | config | `cosmos/ai/telemetry/forecasting.py` |
| `cfg.prereg.aci_alpha` | 0.01 | config | `cosmos/ai/telemetry/oof_benchmark.py` |
| `cfg.prereg.aci_gammas` | [0.001, 0.005, 0.01, 0.05] | config | `cosmos/ai/telemetry/oof_benchmark.py` |
| `cfg.prereg.ks` | [1, 3, 10, 30, 100] | config | `cosmos/ai/telemetry/oof_benchmark.py` |
| `cfg.prereg.levels.conformal_bh` | [0.01, 0.001, 0.0001, 1e-05] | config | `cosmos/ai/telemetry/oof_benchmark.py` |
| `cfg.prereg.levels.threshold` | [0.001, 0.0001, 1e-05, 0.0] | config | `cosmos/ai/telemetry/oof_benchmark.py` |
| `cfg.prereg.merge_h` | 1 | config | `cosmos/ai/telemetry/oof_benchmark.py` |
| `code.ai.files` | 38 | derived | `cosmos/ai/**/*.py (excl. vendored esa_metrics)` |
| `code.ai.lines` | 8749 | derived | `cosmos/ai/**/*.py (excl. vendored esa_metrics)` |
| `code.original.files` | 26 | derived | `cosmos/{*,health,debris,reentry}/*.py` |
| `code.original.lines` | 5317 | derived | `cosmos/{*,health,debris,reentry}/*.py` |
| `conf.freeze_id` | 09a98f5e160e | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.A_official` | 0.205361 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.B_official` | 0.427806 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.ci95_high` | 0.0423207 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.ci95_low` | -0.395153 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.diff` | -0.222445 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.diff_official` | -0.222445 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.p_holm` | 0.101 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h4.verdict` | refuted | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.A_official` | 0.205361 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.B_official` | 0.543614 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.ci95_high` | -0.0329927 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.ci95_low` | -0.540052 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.diff` | -0.330145 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.diff_official` | -0.338253 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.p_holm` | 0.07 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h5.verdict` | refuted | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.A_official` | 0.427806 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.B_official` | 0.074351 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.ci95_high` | 0.488107 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.ci95_low` | 0.206355 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.diff` | 0.351792 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.diff_official` | 0.353455 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.n_boot` | 2000 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.p_holm` | 0.0015 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m1.h6.verdict` | supported | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h4.A_official` | 0.874943 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h4.B_official` | 0.873529 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h4.ci95_high` | 0.053566 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h4.ci95_low` | -0.0548069 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h4.diff` | 0.00141484 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h4.diff_official` | 0.00141387 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h5.A_official` | 0.874943 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h5.B_official` | 0.903949 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h5.ci95_high` | 0.0370127 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h5.ci95_low` | -0.10283 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h5.diff` | -0.0290051 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h5.diff_official` | -0.0290058 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h6.A_official` | 0.873529 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h6.B_official` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h6.ci95_high` | -0.0493163 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h6.ci95_low` | -0.143205 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h6.diff` | -0.0888818 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.m2.h6.diff_official` | -0.0888814 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.json` |
| `conf.p_floor` | 0.0005 | derived (1 / replicates) | `CONFIRMATORY.json` |
| `conf.p_floor_holm` | 0.0015 | derived (Holm factor 3 x floor) | `CONFIRMATORY.json` |
| `conf.reimpl_gap_max` | 0.00810847 | derived (max |re-implemented - official difference| over the six comparisons) | `CONFIRMATORY.json` |
| `data.celestrak.class` | REAL | measured | `data/DATASET_REPORT.json` |
| `data.celestrak.licence` | CelesTrak usage policy; public, no credentials; no explicit  | measured | `data/DATASET_REPORT.json` |
| `data.celestrak.size_mb` | 21.2 | measured | `data/DATASET_REPORT.json` |
| `data.celestrak.status` | downloaded | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_telemetry.class` | SYNTHETIC | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_telemetry.size_mb` | 38.9 | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_telemetry.status` | generated | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_vision.class` | SYNTHETIC | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_vision.licence` | generated by this project | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_vision.size_mb` | 467.5 | measured | `data/DATASET_REPORT.json` |
| `data.cosmos_synthetic_vision.status` | generated | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.classes.test` | 17 | derived | `data/feature1_health/esa_adb/processed/mission1/labels_with_types.csv` |
| `data.esa.m1.classes.test_unseen` | 4 | derived | `data/feature1_health/esa_adb/processed/mission1/labels_with_types.csv` |
| `data.esa.m1.events_all.train.Anomaly` | 63 | derived (events (unique ID) by first start time vs the official split date) | `data/feature1_health/esa_adb/processed/mission1/labels_with_types.csv` |
| `data.esa.m1.events_all.train.Communication_Gap` | 4 | derived (events (unique ID) by first start time vs the official split date) | `data/feature1_health/esa_adb/processed/mission1/labels_with_types.csv` |
| `data.esa.m1.events_all.train.Rare_Event` | 42 | derived (events (unique ID) by first start time vs the official split date) | `data/feature1_health/esa_adb/processed/mission1/labels_with_types.csv` |
| `data.esa.m1.events_all.train.targets` | 105 | derived (anomalies + rare events in the M1 training half (communication gaps excluded)) | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.cal.Anomaly` | 7 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.cal.Rare_Event` | 2 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.cal.events` | 9 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.es.Anomaly` | 1 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.fit.Anomaly` | 55 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.fit.Communication_Gap` | 4 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.fit.Rare_Event` | 40 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.test.Anomaly` | 55 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.events_target.test.Rare_Event` | 36 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.grid_seconds` | 30 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.target_channels` | 58 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.test.blocks30` | 86 | derived (number of 30-day bootstrap blocks in the test half) | `data/DATASET_REPORT.json` |
| `data.esa.m1.test.end` | 2014-01-01 00:00:00 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.test.n` | 7364161 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.train.n` | 7364161 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m1.train.start` | 2000-01-01 00:00:00 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.classes.test` | 24 | derived | `data/feature1_health/esa_adb/processed/mission2/labels_with_types.csv` |
| `data.esa.m2.classes.test_unseen` | 8 | derived | `data/feature1_health/esa_adb/processed/mission2/labels_with_types.csv` |
| `data.esa.m2.events_target.cal.Rare_Event` | 43 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.events_target.cal.events` | 43 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.events_target.es.Rare_Event` | 12 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.events_target.fit.Anomaly` | 25 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.events_target.fit.Rare_Event` | 202 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.events_target.test.Anomaly` | 6 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.events_target.test.Rare_Event` | 356 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.grid_seconds` | 18 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.target_channels` | 47 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.test.blocks30` | 22 | derived (number of 30-day bootstrap blocks in the test half) | `data/DATASET_REPORT.json` |
| `data.esa.m2.test.n` | 3062401 | measured | `data/DATASET_REPORT.json` |
| `data.esa.m2.train.n` | 3067201 | measured | `data/DATASET_REPORT.json` |
| `data.esa.revision_check` | 780/780 (M1) and 230/230 (M2) members present in both; the O | measured | `data/DATASET_REPORT.json` |
| `data.esa_adb.class` | REAL | measured | `data/DATASET_REPORT.json` |
| `data.esa_adb.licence` | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR  | measured | `data/DATASET_REPORT.json` |
| `data.esa_adb.size_mb` | 8498.1 | measured | `data/DATASET_REPORT.json` |
| `data.esa_adb.status` | downloaded | measured | `data/DATASET_REPORT.json` |
| `data.landmark.events` | 235 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.events_type.DEB` | 46 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.events_type.PAY` | 173 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.events_type.RB` | 14 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.events_type.UNK` | 2 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.follow_up_days` | 68 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.gp_coverage` | 0.555126 | measured | `data/DATASET_REPORT.json` |
| `data.landmark.population_at_risk` | 34158 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.anomaly_fraction` | 0.204428 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.channels` | 9 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.fragments` | 2123 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.n_features` | 18 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.seg_len_max` | 1040 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.seg_len_median` | 70 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.seg_len_min` | 8 | measured | `data/DATASET_REPORT.json` |
| `data.opssat.test` | 529 | derived | `data/DATASET_REPORT.json` |
| `data.opssat.train` | 1594 | derived | `data/DATASET_REPORT.json` |
| `data.opssat_ad.class` | REAL | measured | `data/DATASET_REPORT.json` |
| `data.opssat_ad.licence` | cc-by-4.0 | measured | `data/DATASET_REPORT.json` |
| `data.opssat_ad.size_mb` | 22.8 | measured | `data/DATASET_REPORT.json` |
| `data.opssat_ad.status` | downloaded | measured | `data/DATASET_REPORT.json` |
| `data.reentry_historical.class` | REAL | measured | `data/DATASET_REPORT.json` |
| `data.reentry_historical.size_mb` | 0.2 | measured | `data/DATASET_REPORT.json` |
| `data.reentry_historical.status` | downloaded (public parts only) | measured | `data/DATASET_REPORT.json` |
| `data.reentry_simulated.class` | SIM | measured | `data/DATASET_REPORT.json` |
| `data.reentry_simulated.size_mb` | 67.5 | measured | `data/DATASET_REPORT.json` |
| `data.reentry_simulated.status` | generated | measured | `data/DATASET_REPORT.json` |
| `data.satcat.20260816.rows` | 70288 | measured | `data/DATASET_REPORT.json` |
| `data.satcat.20260816.type.DEB` | 35834 | measured | `data/DATASET_REPORT.json` |
| `data.spacesense_bench.class` | SYNTHETIC (Unreal Engine 5 renders) | measured | `data/DATASET_REPORT.json` |
| `data.spacesense_bench.licence` | cc-by-nc-4.0 (Hugging Face card); GitHub repo has no licence | measured | `data/DATASET_REPORT.json` |
| `data.spacesense_bench.size_mb` | 1031.7 | measured | `data/DATASET_REPORT.json` |
| `data.spacesense_bench.status` | inspection sample only (1 of 136 archives) | measured | `data/DATASET_REPORT.json` |
| `data.spark.status` | not downloaded | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.class` | REAL | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.dup_train_test` | 0 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.licence` | cc-by-4.0 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.size_mb` | 45576.8 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.status` | downloaded | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.test.frames_on_disk` | 25600 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.test.objects` | 44550 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.test.sequences` | 5120 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.train.corrupt_png` | 0 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.train.frames_on_disk` | 6400 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.train.objects` | 11250 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.train.sequences` | 1280 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.valsplit.train_sequences` | 1153 | measured | `data/DATASET_REPORT.json` |
| `data.spotgeo.valsplit.val_sequences` | 127 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.test.images` | 200 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.test.objects` | 1603 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.train.box_median_px` | 8.43264 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.train.images` | 800 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.train.objects` | 6448 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.val.images` | 200 | measured | `data/DATASET_REPORT.json` |
| `data.synvis.val.objects` | 1619 | measured | `data/DATASET_REPORT.json` |
| `env.cpu_count` | 512 | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `env.device_name` | Intel(R) Arc(TM) Pro B60 Graphics | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `env.numpy` | 2.3.5 | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `env.pandas` | 3.0.3 | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `env.platform` | Linux-6.17.0-1010-intel-x86_64-with-glibc2.41 | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `env.python` | 3.12.13 | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `env.torch` | 2.13.0+xpu | measured | `outputs/experiments/20261024-154724_esa_event_m1_target_event_ar8_all_c3_s0_s0/env.json` |
| `evcls.m1.gain_a2_a1.seen.Anomaly` | 2.6 | derived ((detection rate A2 - A1) x number of events) | `event_classes_summary_m1.csv` |
| `evcls.m1.gain_a2_a1.seen.Rare_Event` | 11.004 | derived ((detection rate A2 - A1) x number of events) | `event_classes_summary_m1.csv` |
| `evcls.m1.gain_a2_a1.total` | 20.604 | derived | `event_classes_summary_m1.csv` |
| `evcls.m1.gain_a2_a1.unseen.Rare_Event` | 7 | derived ((detection rate A2 - A1) x number of events) | `event_classes_summary_m1.csv` |
| `evcls.m1.seen.Anomaly.A1` | 0.154 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.seen.Anomaly.A2` | 0.204 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.seen.Anomaly.A3` | 0.227 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.seen.Rare_Event.A1` | 0.286 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.seen.Rare_Event.A2` | 0.679 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.unseen.Anomaly.n` | 3 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.unseen.Rare_Event.A0` | 0.875 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.unseen.Rare_Event.A1` | 0 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.unseen.Rare_Event.A2` | 0.875 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evcls.m1.unseen.Rare_Event.A3` | 0.125 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/event_classes_summary_m1.csv` |
| `evm.m1.gbt.train_seconds_total.mean` | 193.249 | derived (mean over runs (5 fold models + final model per run)) | `outputs/benchmarks/telemetry_esa_adb/event_model/*/meta.json` |
| `evm.m1.params` | 220739 | measured | `outputs/benchmarks/telemetry_esa_adb/event_model/m1_target_event_ar8_all_c3_s4/meta.json` |
| `evm.m1.tcn.train_seconds_total.mean` | 6170.1 | derived (mean over runs (5 fold models + final model per run)) | `outputs/benchmarks/telemetry_esa_adb/event_model/*/meta.json` |
| `evm.m1.tcn.train_seconds_total.n` | 5 | derived (count over runs (5 fold models + final model per run)) | `outputs/benchmarks/telemetry_esa_adb/event_model/*/meta.json` |
| `evm.receptive_field` | 1021 | measured | `outputs/benchmarks/telemetry_esa_adb/event_model/m2_target_event_ar8_all_c3_s4/meta.json` |
| `explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.A_EW_F05_official` | 0.0238061 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.B_EW_F05_official` | 0.074351 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_high` | -0.0315662 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_low` | -0.0749269 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.diff_reimpl` | -0.0516518 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.chronos_bolt_small_vs_cosmos_ar_8_confor.name` | H1 M1: chronos-bolt-small vs COSMOS AR(8), conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official` | 0.0618832 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official` | 0.074351 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_high` | 0.00896075 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_low` | -0.0313526 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.diff_reimpl` | -0.0111968 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.patchtst_vs_cosmos_ar_8_conformal_bh.name` | H1 M1: patchtst vs COSMOS AR(8), conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official` | 0.0490784 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official` | 0.074351 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.ci95_high` | 0.00620728 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.ci95_low` | -0.047729 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.diff_reimpl` | -0.024975 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m1.tcn_vs_cosmos_ar_8_conformal_bh.name` | H1 M1: tcn vs COSMOS AR(8), conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.A_EW_F05_official` | 0.944569 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.B_EW_F05_official` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_high` | 0.0135241 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.ci95_low` | -0.0465582 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.diff_reimpl` | -0.0178425 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.chronos_bolt_small_vs_cosmos_ar_8_confor.name` | H1 M2: chronos-bolt-small vs COSMOS AR(8), conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official` | 0.941063 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_high` | 0.00696948 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.ci95_low` | -0.0511868 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.diff_reimpl` | -0.0213483 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.patchtst_vs_cosmos_ar_8_conformal_bh.name` | H1 M2: patchtst vs COSMOS AR(8), conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.A_EW_F05_official` | 0.935649 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.B_EW_F05_official` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.ci95_high` | 0.00241063 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.ci95_low` | -0.05399 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.diff_reimpl` | -0.0267613 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h1.m2.tcn_vs_cosmos_ar_8_conformal_bh.name` | H1 M2: tcn vs COSMOS AR(8), conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.A_EW_F05_official` | 0.074351 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.B_EW_F05_official` | 0.0882472 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.ci95_high` | 1.41521e-05 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.ci95_low` | -0.0270308 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.diff_reimpl` | -0.0122336 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m1.ar8_conformal_bh_vs_threshold.name` | H2 M1: ar8 conformal-BH vs threshold | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.A_EW_F05_official` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.B_EW_F05_official` | 0.882099 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.ci95_high` | 0.132815 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.ci95_low` | 0.0406 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.diff_reimpl` | 0.0803124 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.n_boot` | 1000 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.ar8_conformal_bh_vs_threshold.name` | H2 M2: ar8 conformal-BH vs threshold | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.A_EW_F05_official` | 0.941063 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.B_EW_F05_official` | 0.8174 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.ci95_high` | 0.177345 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.ci95_low` | 0.080152 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.diff_reimpl` | 0.123663 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h2.m2.patchtst_conformal_bh_vs_threshold.name` | H2 M2: patchtst conformal-BH vs threshold | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.A_EW_F05_official` | 0.019632 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.B_EW_F05_official` | 0.0618832 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.ci95_high` | -0.0257175 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.ci95_low` | -0.0704653 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.diff_reimpl` | -0.0451848 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.patchtst_telecommands_vs_without_conform.name` | H3 M1: patchtst + telecommands vs without, conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.A_EW_F05_official` | 0.0185321 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.B_EW_F05_official` | 0.0490784 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.ci95_high` | -0.0117759 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.ci95_low` | -0.0689801 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.diff_reimpl` | -0.0325064 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m1.tcn_telecommands_vs_without_conformal_bh.name` | H3 M1: tcn + telecommands vs without, conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.A_EW_F05_official` | 0.950842 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.B_EW_F05_official` | 0.941063 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.ci95_high` | 0.0327386 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.ci95_low` | -0.0112314 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.diff_reimpl` | 0.00977922 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.patchtst_telecommands_vs_without_conform.name` | H3 M2: patchtst + telecommands vs without, conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.A_EW_F05_official` | 0.948464 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.B_EW_F05_official` | 0.935649 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.ci95_high` | 0.0251972 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.ci95_low` | 0.000271831 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.diff_reimpl` | 0.0128152 | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.boot.h3.m2.tcn_telecommands_vs_without_conformal_bh.name` | H3 M2: tcn + telecommands vs without, conformal-BH | measured | `outputs/benchmarks/telemetry_esa_adb/bootstrap_comparisons.csv` |
| `explo.m1.full.ar8.conformal_bh.f05.mean` | 0.074351 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.ar8.conformal_bh.fa_yr.mean` | 91.5625 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.ar8.threshold.f05.mean` | 0.0882472 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.cosmon.native.f05.mean` | 0.00378583 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.gstd3.native.f05.mean` | 0.000443491 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.gstd5.native.f05.mean` | 0.00293218 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.ool.native.f05.mean` | 0.000679463 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.ool.native.fa_yr.mean` | 11627 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.oolall.native.f05.mean` | 0.00511105 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.patchtst.conformal_bh.rare_only.mean` | 97 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.patchtst_tc.conformal_bh.rare_only.mean` | 207 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.tcn.conformal_bh.rare_only.mean` | 100.667 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.tcn_tc.conformal_bh.rare_only.mean` | 91.3333 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.tcn_tc.conformal_bh_rolling.f05.max` | 0.463576 | derived (max over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.tcn_tc.conformal_bh_rolling.f05.mean` | 0.318366 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.tcn_tc.conformal_bh_rolling.f05.min` | 0.0719424 | derived (min over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.full.tcn_tc.conformal_bh_rolling.f05.std` | 0.21454 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.lw.ar8.conformal_bh.f05.mean` | 0.408035 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.lw.cosmon.native.f05.mean` | 0.202064 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.lw.gstd3.native.f05.mean` | 0.00108118 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.lw.gstd5.native.f05.mean` | 0.252687 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.lw.ool.native.f05.mean` | 0.201856 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m1.lw.oolall.native.f05.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.ar8.conformal_bh.f05.mean` | 0.962411 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.ar8.conformal_bh.r_anom.mean` | 0.166667 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.chronos.conformal_bh.f05.mean` | 0.944569 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.cosmon.native.f05.mean` | 0.0471464 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.gstd3.native.f05.mean` | 0.00749368 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.gstd5.native.f05.mean` | 0.0634998 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.ool.native.f05.mean` | 0.893611 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.oolall.native.f05.mean` | 0.26092 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.patchtst.conformal_bh.f05.mean` | 0.951814 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.patchtst.conformal_bh.f05.std` | 0.00942801 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.patchtst.conformal_bh.rare_only.mean` | 352.667 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.patchtst_tc.conformal_bh.rare_only.mean` | 356 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.tcn.conformal_bh.rare_only.mean` | 399.333 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.full.tcn_tc.conformal_bh.rare_only.mean` | 393.333 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.lw.cosmon.native.f05.mean` | 0.788052 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.lw.gstd3.native.f05.mean` | 0.00708125 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.lw.gstd5.native.f05.mean` | 0.0749327 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.lw.ool.native.f05.mean` | 0.964808 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.m2.lw.oolall.native.f05.mean` | 0.0802139 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/results.csv` |
| `explo.robust.m2_target_ar8_s0_telecommand_mask.mask0.f05` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_ar8_s0_telecommand_mask.json` |
| `explo.robust.m2_target_ar8_s0_telecommand_mask.mask0.rare_only` | 351 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_ar8_s0_telecommand_mask.json` |
| `explo.robust.m2_target_ar8_s0_telecommand_mask.mask30.f05` | 0.303921 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_ar8_s0_telecommand_mask.json` |
| `explo.robust.m2_target_ar8_s0_telecommand_mask.mask30.r_anom` | 0.166667 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_ar8_s0_telecommand_mask.json` |
| `explo.robust.m2_target_ar8_s0_telecommand_mask.mask30.rare_only` | 42 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_ar8_s0_telecommand_mask.json` |
| `explo.robust.m2_target_patchtst_tc_s0_telecommands_removed.conformal_bh.f05` | 0.950842 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_patchtst_tc_s0_telecommands_removed.json` |
| `explo.robust.m2_target_patchtst_tc_s0_telecommands_removed.conformal_bh.f05_with_tc` | 0.950842 | measured | `outputs/benchmarks/telemetry_esa_adb/robustness/m2_target_patchtst_tc_s0_telecommands_removed.json` |
| `layers.n_rows` | 10 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `layers.n_threshold_ge_bh` | 8 | derived (rows of the alarm-layer table where threshold F0.5 >= conformal-BH F0.5) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `lit.alcarria.m1_f05` | 0.7 | literature | `alcarria2026, Table 1 (arXiv 2609.10017)` |
| `lit.alcarria.m2_f05` | 0.698 | literature | `alcarria2026, Table 1 (arXiv 2609.10017)` |
| `lit.alcarria.pitia_m1` | 0.424 | literature | `alcarria2026, Table 1 (arXiv 2609.10017)` |
| `lit.alcarria.pitia_m2` | 0.882 | literature | `alcarria2026, Table 1 (arXiv 2609.10017)` |
| `lit.allegrini.private` | 0.853 | literature | `allegrini2026 (arXiv 2605.06681)` |
| `lit.allegrini.public` | 0.931 | literature | `allegrini2026 (arXiv 2605.06681)` |
| `lit.brancato.ool_full` | 0.754 | literature | `brancato2027ast, abstract (full text not accessed)` |
| `lit.brancato.qgpr_full` | 0.883 | literature | `brancato2027ast, abstract (full text not accessed)` |
| `lit.brancato.smed_full` | 0.902 | literature | `brancato2027ast, abstract (full text not accessed)` |
| `lit.dmlr.m1.full.best` | 0.061 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m1.full.best_name` | Telemanom-ESA-Pruned | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m1.full.gstd5` | 0.003 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m1.lw.best` | 0.786 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m1.lw.best_name` | Telemanom-ESA-Pruned | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m1.lw.gstd3` | 0.001 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m1.lw.gstd5` | 0.253 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.full.best` | 0.079 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.full.best_name` | PCC | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.full.gstd3` | 0.008 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.full.gstd5` | 0.064 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.lw.best` | 0.949 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.lw.gstd3` | 0.007 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.dmlr.m2.lw.gstd5` | 0.075 | literature | `kotowski2026esaadb, DMLR Table 4 (research_design/references/kotowski2026_esa_adb_dmlr.pdf)` |
| `lit.heydari.shared_channels` | 77 | literature | `heydari2026, abstract` |
| `lit.opssat.adaboost_f1` | 0.836 | literature | `ruszczak2025opssat, Table 3 (research_design/references/opssat_table3.html)` |
| `lit.opssat.fcnn_aucpr` | 0.979 | literature | `ruszczak2025opssat, Table 3 (research_design/references/opssat_table3.html)` |
| `lit.opssat.fcnn_f1` | 0.946 | literature | `ruszczak2025opssat, Table 3 (research_design/references/opssat_table3.html)` |
| `lit.spotgeo.winner_1mf1` | 0.0517 | literature | `chen2021spotgeo, Table 1 (final ranking, full test set)` |
| `lit.spotgeo.winner_f1` | 0.9483 | literature (1 - (1 - F1) as published) | `chen2021spotgeo, Table 1 (final ranking, full test set)` |
| `lit.spotgeo.winner_mse` | 33839 | literature | `chen2021spotgeo, Table 1 (final ranking, full test set)` |
| `lit.tccad.aci_far_overshoot` | 1.15–3.4 | literature | `sahin2026tccad, preregistration/en/PREREGISTRATION_H4.md section 0` |
| `lit.tccad.c0_anom_ewf05` | 0.0895124 | literature | `sahin2026tccad, figures/table7_official_V0_data.csv and fig5_budget_data.csv (Zenodo 23344000)` |
| `lit.tccad.c3_anom_ewf05_order` | 1e-5 | literature | `sahin2026tccad, figures/table7_official_V0_data.csv and fig5_budget_data.csv (Zenodo 23344000)` |
| `lit.tccad.deferral_pp` | 29-33 | literature | `sahin2026tccad, Zenodo description` |
| `lit.tccad.m1_channels` | 24 | literature | `sahin2026tccad, figures/table7_official_V0_data.csv and fig5_budget_data.csv (Zenodo 23344000)` |
| `lit.tccad.m2_channels` | 19 | literature | `sahin2026tccad, figures/table7_official_V0_data.csv and fig5_budget_data.csv (Zenodo 23344000)` |
| `lit.tccad.seeds` | 16 | literature | `sahin2026tccad, figures/table7_official_V0_data.csv and fig5_budget_data.csv (Zenodo 23344000)` |
| `lm.cmp.hazard_mlp_vs_hazard_mlp_nophys.h68.ci_high` | 0.065034 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.hazard_mlp_vs_hazard_mlp_nophys.h68.ci_low` | 0.00413187 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.hazard_mlp_vs_hazard_mlp_nophys.h68.diff` | 0.0257765 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.hazard_mlp_vs_xgboost.h68.ci_high` | -0.0239493 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.hazard_mlp_vs_xgboost.h68.ci_low` | -0.220573 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.hazard_mlp_vs_xgboost.h68.diff` | -0.133609 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.hazard_mlp_vs_xgboost.h68.n_boot` | 2000 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.xgboost_vs_physics.h68.ci_high` | 0.779465 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.xgboost_vs_physics.h68.ci_low` | 0.426619 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.cmp.xgboost_vs_physics.h68.diff` | 0.63309 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.conf.hazard_mlp.marginal.coverage_on_events` | 0.131579 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.conf.xgboost.marginal.coverage` | 0.911474 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.conf.xgboost.marginal.coverage_on_events` | 0 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.conf.xgboost.mondrian.coverage_events` | 0.973684 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.conf.xgboost.mondrian.coverage_nonevents` | 0.915996 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.follow_up` | 2026-10-23 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.all.c_index` | 0.995662 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.all.h30.auc_pr` | 0.609709 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.all.h68.auc_pr` | 0.687716 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.all.h68.auc_roc` | 0.99636 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.all.h68.brier` | 0.00328892 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.all.h68.ece` | 0.0024067 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp.uncontrolled.h30.auc_pr` | 0.136312 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp_nophys.all.c_index` | 0.994618 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp_nophys.all.h30.auc_pr` | 0.58738 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp_nophys.all.h68.auc_pr` | 0.66194 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp_nophys.all.h68.auc_roc` | 0.995375 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp_nophys.all.h68.brier` | 0.00330339 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.hazard_mlp_nophys.all.h68.ece` | 0.00244289 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.all.c_index` | 0.990217 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.all.h30.auc_pr` | 0.528056 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.all.h68.auc_pr` | 0.527164 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.all.h68.auc_roc` | 0.990846 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.all.h68.brier` | 0.00386752 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.all.h68.ece` | 0.00168146 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.logreg.uncontrolled.h30.auc_pr` | 0.294218 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.mlp.grid_size` | 8 | derived | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.mlp.selected` | {"hidden": [256, 256, 128], "lr": 0.0003, "quantile": true} | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.all.c_index` | 0.975976 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.all.h30.auc_pr` | 0.253835 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.all.h68.auc_pr` | 0.188235 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.all.h68.auc_roc` | 0.976649 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.all.h68.brier` | 0.00559902 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.all.h68.ece` | 0.00147456 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.uncontrolled.events` | 12 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.uncontrolled.h30.auc_pr` | 0.511707 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.uncontrolled.h30.positives` | 4 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.uncontrolled.h68.auc_pr` | 0.2041 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.physics.uncontrolled.n` | 3001 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.split.test.events` | 38 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.split.test.n` | 6371 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.split.train.events` | 140 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.split.train.n` | 18914 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.split.val.events` | 57 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.split.val.n` | 8873 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.tau` | 2026-08-16 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.all.c_index` | 0.997265 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.all.h30.auc_pr` | 0.760797 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.all.h68.auc_pr` | 0.821325 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.all.h68.auc_roc` | 0.998217 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.all.h68.brier` | 0.00254114 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.all.h68.ece` | 0.00215699 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.uncontrolled.h30.auc_pr` | 0.202381 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost.uncontrolled.h68.auc_pr` | 0.464158 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.all.c_index` | 0.996747 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.all.h30.auc_pr` | 0.658244 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.all.h68.auc_pr` | 0.81347 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.all.h68.auc_roc` | 0.997469 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.all.h68.brier` | 0.00263809 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.all.h68.ece` | 0.00207475 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `lm.xgboost_aft.uncontrolled.h68.auc_pr` | 0.548909 | measured | `outputs/benchmarks/reentry_landmark/results.json` |
| `mon.heldout.classical_ewma.f1.mean` | 0.83376 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.classical_ewma.f1.std` | 0.00933883 | derived (sample sd over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.classical_ewma.false_alarms_total` | 0 | derived (sum over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.classical_ewma.localisation_accuracy.mean` | 0.83 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.classical_ewma.precision.mean` | 1 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.classical_ewma.recall.mean` | 0.715 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.f1.mean` | 0.774617 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.f1.std` | 0.0532979 | derived (sample sd over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.false_alarms_total` | 3 | derived (sum over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.localisation_accuracy.mean` | 0.96 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.precision.mean` | 0.928715 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.recall.mean` | 0.665 | derived (mean over seed sets) | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `mon.heldout.learned_conformal.seed_sets` | 5,6,7,8,9 | measured | `outputs/benchmarks/telemetry_cosmos_synthetic/heldout/summary_by_seed.csv` |
| `oof.m1.K` | 5 | measured | `outputs/benchmarks/telemetry_esa_adb/oof/m1_target_ar8/folds.json` |
| `oof.m1.purge_steps` | 2880 | measured | `outputs/benchmarks/telemetry_esa_adb/oof/m1_target_ar8/folds.json` |
| `oof.m1.seconds` | 381.669 | measured | `outputs/benchmarks/telemetry_esa_adb/oof/m1_target_ar8/folds.json` |
| `oof.m2.purge_steps` | 4800 | measured | `outputs/benchmarks/telemetry_esa_adb/oof/m2_target_ar8/folds.json` |
| `ops.official.adaboost.f100.auc_pr.mean` | 0.925116 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.adaboost.f100.f1.mean` | 0.838095 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.chronos.f10.auc_pr.mean` | 0.82313 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.chronos.f100.auc_pr.mean` | 0.955346 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.chronos.f100.f1.mean` | 0.888889 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.cnn.f10.auc_pr.mean` | 0.911614 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.cnn.f100.auc_pr.mean` | 0.987041 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.cnn.f100.f1.mean` | 0.938753 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.cnn.f100.f1.std` | 0.015446 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.cosmos_robust_z.f100.auc_pr.mean` | 0.471207 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.cosmos_robust_z.f100.f1.mean` | 0.436364 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.fcnn.f10.auc_pr.mean` | 0.935697 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.fcnn.f100.auc_pr.mean` | 0.979983 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.fcnn.f100.f1.mean` | 0.936508 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.fcnn.f100.f1.std` | 0.00813517 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.fcnn.f100.params` | 13249 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.hybrid.f10.auc_pr.mean` | 0.910214 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.hybrid.f100.auc_pr.mean` | 0.985549 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.hybrid.f100.f1.mean` | 0.948994 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.hybrid.f100.f1.std` | 0.00856504 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.iforest.f100.auc_pr.mean` | 0.34621 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.iforest.f100.f1.mean` | 0.226232 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.logreg.f100.auc_pr.mean` | 0.956913 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.logreg.f100.f1.mean` | 0.862559 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.random_forest.f100.auc_pr.mean` | 0.963675 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.random_forest.f100.f1.mean` | 0.90875 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.transformer.f100.auc_pr.mean` | 0.948313 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.transformer.f100.f1.mean` | 0.898064 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.xgboost.f10.auc_pr.mean` | 0.891342 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.xgboost.f100.auc_pr.mean` | 0.960739 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.official.xgboost.f100.f1.mean` | 0.910936 | measured | `outputs/benchmarks/telemetry_opssat/summary_official.csv` |
| `ops.paired.fcnn_xgboost.aucpr.ci_high` | 0.038761 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.fcnn_xgboost.aucpr.ci_low` | 0.00223272 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.fcnn_xgboost.aucpr.diff` | 0.0195391 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.fcnn_xgboost.f1.ci_high` | 0.0628046 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.fcnn_xgboost.f1.ci_low` | 0.00410465 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.fcnn_xgboost.f1.diff` | 0.0323824 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_fcnn.aucpr.ci_high` | 0.0127088 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_fcnn.aucpr.ci_low` | 0.000410158 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_fcnn.aucpr.diff` | 0.00589887 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_fcnn.f1.ci_high` | 0.0406961 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_fcnn.f1.ci_low` | -0.00801538 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_fcnn.f1.diff` | 0.0147179 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_xgboost.aucpr.ci_high` | 0.0460475 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_xgboost.aucpr.ci_low` | 0.00778197 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.paired.hybrid_xgboost.aucpr.diff` | 0.025438 | measured | `outputs/benchmarks/telemetry_opssat/paired_comparisons.csv` |
| `ops.robust.dropout.0p3.cnn.f1` | 0.6719 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.robust.dropout.0p3.hybrid.f1` | 0.7665 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.robust.dropout.0p3.xgboost.f1` | 0.8505 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.robust.noise.0p1.cnn.f1` | 0.5678 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.robust.noise.0p1.fcnn.f1` | 0.8074 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.robust.noise.0p1.hybrid.f1` | 0.8563 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.robust.noise.0p1.xgboost.f1` | 0.7661 | measured | `outputs/benchmarks/telemetry_opssat/robustness_summary.csv` |
| `ops.time_blocked.adaboost.f100.auc_pr.mean` | 0.944012 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.adaboost.f100.f1.mean` | 0.88 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.chronos.f100.auc_pr.mean` | 0.919065 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.chronos.f100.f1.mean` | 0.807818 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.cnn.f100.auc_pr.mean` | 0.971173 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.cnn.f100.f1.mean` | 0.867164 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.cosmos_robust_z.f100.auc_pr.mean` | 0.673231 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.cosmos_robust_z.f100.f1.mean` | 0.315315 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.fcnn.f100.auc_pr.mean` | 0.966555 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.fcnn.f100.f1.mean` | 0.906921 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.hybrid.f100.auc_pr.mean` | 0.971882 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.hybrid.f100.f1.mean` | 0.909598 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.iforest.f100.auc_pr.mean` | 0.331281 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.iforest.f100.f1.mean` | 0.172101 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.logreg.f100.auc_pr.mean` | 0.961717 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.logreg.f100.f1.mean` | 0.875776 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.random_forest.f100.auc_pr.mean` | 0.955508 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.random_forest.f100.f1.mean` | 0.888197 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.transformer.f100.auc_pr.mean` | 0.941795 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.transformer.f100.f1.mean` | 0.880023 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.xgboost.f100.auc_pr.mean` | 0.963939 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `ops.time_blocked.xgboost.f100.f1.mean` | 0.894146 | measured | `outputs/benchmarks/telemetry_opssat/summary_time_blocked.csv` |
| `orig.corr.central_europe_e.expected_casualties_event` | 0.0319442 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.corr.mid_pacific_e.expected_casualties` | 0 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.corr.peninsular_india_nw.expected_casualties` | 0.000336936 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.corr.peninsular_india_nw.expected_casualties_event` | 0.0610304 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.anomaly_score_final` | 0.241 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.break_up_altitude_km` | 77.9 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.break_up_type` | aerodynamic | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.cep50_km` | 459.7 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.fragments_generated` | 868 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.health_band` | NOMINAL | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.impacts_propagated` | 1199 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.default.risk_band` | NEGLIGIBLE | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.det.registered_difference.f1` | 0.717 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.det.registered_difference.precision` | 0.987 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.det.registered_difference.recall` | 0.563 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.det.single_frame.f1` | 0.1974 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.det.single_frame.precision` | 0.11 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.P_fragmentation` | 0.109 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.breakup_alt_km` | 77.6 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.breakup_type` | aerodynamic | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.cep50_km` | 431.5 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.downrange_mean_km` | 788.6 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.driver` | Communication | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.comms_lost.sigma_logB` | 0.182 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.P_fragmentation` | 0.021 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.breakup_alt_km` | 77.9 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.breakup_type` | aerodynamic | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.cep50_km` | 432.9 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.downrange_mean_km` | 796.7 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.driver` | Data Storage Unit | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.nominal.sigma_logB` | 0.18 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.P_fragmentation` | 0.592 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.breakup_alt_km` | 81.8 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.breakup_type` | explosion | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.cep50_km` | 458.4 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.downrange_mean_km` | 902.8 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.driver` | Propulsion System | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.propulsion.sigma_logB` | 0.254 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.P_fragmentation` | 0.091 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.breakup_alt_km` | 77.9 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.breakup_type` | aerodynamic | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.cep50_km` | 433.3 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.downrange_mean_km` | 798.2 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.driver` | Data Storage Unit | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.structure.sigma_logB` | 0.179 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.P_fragmentation` | 0.154 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.breakup_alt_km` | 71.3 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.breakup_type` | aerodynamic | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.cep50_km` | 386.1 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.downrange_mean_km` | 632.1 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.driver` | Thermal Control | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fm.thermal.sigma_logB` | 0.186 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fp.cep50_km` | 452.3 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.fp.n_impacts` | 1200 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_cal.area_to_mass_censored` | 0.14907 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_cal.bias_factor` | 1.049 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_cal.n_censored` | 3059 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_cal.n_fragments` | 7711 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_cal.n_observed` | 4652 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_val.cosmos_1408.ratio` | 0.292 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_val.cosmos_2251.ratio` | 0.863 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.lifetime_val.iridium_33.ratio` | 1.001 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.degradation.detected` | 1 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.degradation.median_delay_min` | 600.5 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.degradation.n` | 8 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.dropout.detected` | 7 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.dropout.median_delay_min` | 6 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.dropout.n` | 8 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.leak.detected` | 7 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.leak.median_delay_min` | 505 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.leak.n` | 8 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.n_detected` | 29 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.n_faulted` | 40 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.n_faults` | 5 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.n_subsystems` | 8 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.nominal_false_alarm` | False | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.overheat.detected` | 7 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.overheat.median_delay_min` | 347.5 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.overheat.n` | 8 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.saturation.detected` | 7 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.saturation.median_delay_min` | 357 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.saturation.n` | 8 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.sub.data_storage_unit.detected` | 5 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.sub.power_system.detected` | 4 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.monitor.sub.thermal_control.detected` | 0 | derived | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.all_nominal.anomaly_score` | 0.05 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.all_nominal.breakup_altitude_km` | 79.3 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.all_nominal.breakup_type` | explosion | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.all_nominal.driver` | Power System | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.all_nominal.p_fragmentation` | 0.009 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.power_critical.breakup_altitude_km` | 76.6 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.thermal_critical.breakup_altitude_km` | 69.8 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.scen.thermal_critical.p_fragmentation` | 0.129 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.stuck.error` | ValueError('scale < 0') | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.version` | 1.1.0 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.90.conformal_density_coverage` | 0.9004 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.90.density_area_saving_pct` | 60.4 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.90.gaussian_ellipse_coverage` | 0.9122 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.90.gaussian_shortfall_pts` | -1.22 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.95.conformal_density_coverage` | 0.9437 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.95.density_area_saving_pct` | 59.8 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.95.gaussian_ellipse_coverage` | 0.945 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.95.gaussian_shortfall_pts` | 0.5 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.99.conformal_density_coverage` | 0.9846 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.99.density_area_saving_pct` | 57.4 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.99.gaussian_ellipse_coverage` | 0.9856 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `orig.zone.99.gaussian_shortfall_pts` | 0.44 | measured | `thesis/evidence/original_pipeline_runs.json` |
| `ph.p1.m1.lightweight.conformal_bh.forward.D.delta_official` | -0.165865 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_lightweight_conformal_bh.json` |
| `ph.p1.m1.lightweight.conformal_bh.forward.P.delta_official` | -0.149641 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_lightweight_conformal_bh.json` |
| `ph.p1.m1.lightweight.conformal_bh.reverse.P.delta_official` | -0.222666 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_lightweight_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.forward.D.ci95_high` | 0.43552 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.forward.D.ci95_low` | 0.156401 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.forward.D.delta_official` | 0.300603 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.forward.F.delta_official` | 0.0100133 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.forward.P.delta_official` | 0.0428387 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.r1.f05` | 0.493271 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.reverse.D.ci95_high` | 0.500156 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.reverse.D.ci95_low` | 0.233552 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.reverse.D.delta_official` | 0.366068 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.conformal_bh.reverse.G.delta_official` | -0.0654647 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_conformal_bh.json` |
| `ph.p1.m1.target.threshold.forward.D.delta_official` | 0.196915 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_threshold.json` |
| `ph.p1.m1.target.threshold.forward.P.delta_official` | 0.145417 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m1_target_threshold.json` |
| `ph.p1.m2.target.conformal_bh.forward.D.ci95_high` | -0.0542792 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m2_target_conformal_bh.json` |
| `ph.p1.m2.target.conformal_bh.forward.D.ci95_low` | -0.134201 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m2_target_conformal_bh.json` |
| `ph.p1.m2.target.conformal_bh.forward.D.delta_official` | -0.0884476 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p1_decompose/m2_target_conformal_bh.json` |
| `ph.p2.m1.conformal_bh.fold2.W_fa_yr` | 1172.8 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p2_forward/m1_summary.json` |
| `ph.p2.m1.conformal_bh.mean_diff` | 0.239539 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p2_forward/m1_summary.json` |
| `ph.p2.m1.conformal_bh.n_folds` | 4 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p2_forward/m1_summary.json` |
| `ph.p2.m1.conformal_bh.n_folds_H_better` | 2 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p2_forward/m1_summary.json` |
| `ph.p2.m2.conformal_bh.mean_diff` | 0.0170919 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p2_forward/m2_summary.json` |
| `ph.p2.m2.conformal_bh.n_folds` | 3 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p2_forward/m2_summary.json` |
| `ph.p3.m1.h5.p_holm` | 0.066 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p3_exact/confirmatory_exact.json` |
| `ph.p3.m1.h6.ci95_high` | 0.491076 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p3_exact/confirmatory_exact.json` |
| `ph.p3.m1.h6.ci95_low` | 0.207282 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p3_exact/confirmatory_exact.json` |
| `ph.p3.max_abs_diff` | 2.12178e-16 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p3_exact/exactness.json` |
| `ph.p3.n_checks` | 302 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p3_exact/exactness.json` |
| `ph.p4.m1.ar8.gamma_positive.f05` | 0.471698 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p4_aci/m1_target_ar8.json` |
| `ph.p4.m1.ar8.gamma_positive.gamma` | 0.0001 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p4_aci/m1_target_ar8.json` |
| `ph.p4.rule.n_arms` | 7 | derived | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p4_aci/*.json` |
| `ph.p4.rule.n_at_or_above` | 2 | derived | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p4_aci/*.json` |
| `ph.p4.rule.n_below` | 5 | derived | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p4_aci/*.json` |
| `ph.p5.m1.A1.mean` | 0.336133 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A2.fa_yr` | 44.7099 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A2.max` | 0.231777 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A2.mean` | 0.0868645 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A2.min` | 0.010882 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A3.max` | 0.21052 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A3.mean` | 0.0970079 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m1.A3.min` | 0.0343881 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m1_summary.json` |
| `ph.p5.m2.A1.mean` | 0 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m2_summary.json` |
| `ph.p5.m2.A2.mean` | 0.247202 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m2_summary.json` |
| `ph.p5.m2.A3.mean` | 0.00458192 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p5_anomaly_arm/m2_summary.json` |
| `ph.p6.m1.A2.f05.max` | 0.769227 | derived (max over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A2.f05.mean` | 0.488366 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A2.f05.min` | 0.157195 | derived (min over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A2.f05.n` | 10 | derived (count over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A2.f05.std` | 0.194718 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A3.f05.max` | 0.358179 | derived (max over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A3.f05.mean` | 0.175637 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A3.f05.min` | 0.0819908 | derived (min over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A3.f05.n` | 9 | derived (count over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.A3.f05.std` | 0.0931341 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.h4.ci95_high` | -0.0426391 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.h4.ci95_low` | -0.401727 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.h4.diff` | -0.252169 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.h5.ci95_high` | -0.11186 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.h5.ci95_low` | -0.456708 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.h5.diff` | -0.312729 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.x1.ci95_high` | 0.268036 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.x1.ci95_low` | -0.13784 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p6.m1.x1.diff` | 0.0605598 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p6_seeds/summary.json` |
| `ph.p7.m1.A0.detected` | 42 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A0.false_alarms_per_1000_nominal_h` | 11.7487 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A1.delay_h_median` | 0.327818 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A1.detected` | 16 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A1.false_alarms_per_1000_nominal_h` | 0.14663 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A1.n_events` | 91 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A3.delay_h_median` | 2.02365 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p7.m1.A3.run_duration_h_median` | 5.0525 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p7_operational/summary.json` |
| `ph.p8.m1.max_abs_cv_diff` | 0 | measured | `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/p8_provenance/summary.json` |
| `ph.plan_sha256` | 217648e6805f | config | `research_design/preregistration/posthoc_v2_freeze.json` |
| `ph.plan_written` | 2026-10-25T12:14:37 | config | `research_design/preregistration/posthoc_v2_freeze.json` |
| `pre.all.aci.alarm_runs.max` | 1.59764e+06 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci.f05.max` | 0.00171801 | derived (max over all test arms and seeds (30 runs)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci.gamma005_runs` | 29 | derived (runs whose selected gamma is 0.05) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci.gamma_runs` | 30 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci.n` | 30 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci.point_far.max` | 0.00999834 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci.point_far.min` | 0.0098147 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci_event.f05.max` | 0.116959 | derived (max over all test arms and seeds (30 runs)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci_event.f05.max_seedmean` | 0.0460158 | derived (max over arms of the seed-mean) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci_event.n_runs` | 30 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.all.aci_event.zero_alarm_runs` | 10 | derived (runs where aci_event raised no alarm) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A0.conformal_bh.alarm_runs_per_year` | 112.846 | derived (alarm runs / test-half years) | `outputs/benchmarks/telemetry_esa_adb/arms/m1_target_ar8_s0.json` |
| `pre.m1.A0.conformal_bh.f05.mean` | 0.074351 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m1_target_ar8_s0.json` |
| `pre.m1.A0.conformal_bh.f05_anom.mean` | 0.0316175 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m1_target_ar8_s0.json` |
| `pre.m1.A0.conformal_bh.fa_yr.mean` | 91.5625 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m1_target_ar8_s0.json` |
| `pre.m1.A0.conformal_bh.p.mean` | 0.061461 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m1_target_ar8_s0.json` |
| `pre.m1.A0.conformal_bh.r.mean` | 0.461538 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m1_target_ar8_s0.json` |
| `pre.m1.A1.aci.f05.mean` | 5.03012e-05 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.aci_event.f05.mean` | 0.00845125 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.alarm_prec.mean` | 0.13913 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.alarm_runs_per_year` | 17.5697 | derived (alarm runs / test-half years) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.f05.mean` | 0.427806 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.f05_anom.mean` | 0.336133 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.fa_yr.mean` | 1.14275 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.p.mean` | 0.666662 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh.r.mean` | 0.175824 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.conformal_bh_rolling.f05.mean` | 0.10101 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1.threshold.f05.mean` | 0.427806 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1lw.aci.f05.mean` | 0.000141324 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1lw.aci_event.f05.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1lw.conformal_bh.f05.mean` | 0.0246872 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1lw.conformal_bh.r.mean` | 0.0307692 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1lw.conformal_bh_rolling.f05.mean` | 0.12987 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A1lw.threshold.f05.mean` | 0.350124 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.aci.f05.mean` | 4.42638e-05 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.aci_event.f05.mean` | 0.0460158 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.f05.max` | 0.769227 | derived (max over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.f05.mean` | 0.543614 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.f05.min` | 0.157195 | derived (min over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.f05.std` | 0.235659 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.f05_anom.mean` | 0.326712 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.fa_yr.mean` | 8.45632 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.p.mean` | 0.627224 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh.r.mean` | 0.415385 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.conformal_bh_rolling.f05.mean` | 0.0873786 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A2.threshold.f05.mean` | 0.613444 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.aci.f05.mean` | 6.21851e-05 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.aci_event.f05.mean` | 0.00301295 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.f05.max` | 0.358179 | derived (max over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.f05.mean` | 0.205361 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.f05.min` | 0.0854216 | derived (min over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.f05.std` | 0.114168 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.f05_anom.mean` | 0.124969 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.fa_yr.mean` | 20.8837 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.p.mean` | 0.210758 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh.r.mean` | 0.265934 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh_rolling.f05.max` | 0.568828 | derived (max over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh_rolling.f05.mean` | 0.323379 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.conformal_bh_rolling.f05.min` | 0 | derived (min over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3.threshold.f05.mean` | 0.281742 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3c2.aci.f05.mean` | 8.73034e-05 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3c2.aci_event.f05.mean` | 0.000318222 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3c2.conformal_bh.f05.mean` | 0.195847 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3c2.conformal_bh_rolling.f05.mean` | 0.084487 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3c2.threshold.f05.mean` | 0.362719 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3nr.aci.f05.mean` | 5.86195e-05 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3nr.aci_event.f05.mean` | 0.00543196 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3nr.conformal_bh.f05.mean` | 0.249308 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3nr.conformal_bh_rolling.f05.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m1.A3nr.threshold.f05.mean` | 0.275748 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A0.conformal_bh.f05.mean` | 0.962411 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m2_target_ar8_s0.json` |
| `pre.m2.A0.conformal_bh.f05_anom.mean` | 0.147052 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m2_target_ar8_s0.json` |
| `pre.m2.A0.conformal_bh.fa_yr.mean` | 3.43495 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m2_target_ar8_s0.json` |
| `pre.m2.A0.conformal_bh.p.mean` | 0.981705 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m2_target_ar8_s0.json` |
| `pre.m2.A0.conformal_bh.r.mean` | 0.892265 | measured | `outputs/benchmarks/telemetry_esa_adb/arms/m2_target_ar8_s0.json` |
| `pre.m2.A1.aci.f05.mean` | 0.000690695 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.aci_event.f05.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.conformal_bh.f05.mean` | 0.873529 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.conformal_bh.f05_anom.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.conformal_bh.fa_yr.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.conformal_bh.p.mean` | 0.999976 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.conformal_bh.r.mean` | 0.58011 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.conformal_bh_rolling.f05.mean` | 0.355731 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1.threshold.f05.mean` | 0.976529 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1lw.aci.f05.mean` | 0.00171801 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1lw.aci_event.f05.mean` | 0 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1lw.conformal_bh.f05.mean` | 0.977799 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1lw.conformal_bh_rolling.f05.mean` | 0.52931 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A1lw.threshold.f05.mean` | 0.977799 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.aci.f05.mean` | 0.000681249 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.aci_event.f05.mean` | 0.0130901 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh.f05.mean` | 0.903949 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh.f05.std` | 0.049534 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh.f05_anom.mean` | 0.128763 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh.fa_yr.mean` | 3.54945 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh.p.mean` | 0.975287 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh.r.mean` | 0.710497 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.conformal_bh_rolling.f05.mean` | 0.244057 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A2.threshold.f05.mean` | 0.819155 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.aci.f05.mean` | 0.00144371 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.aci_event.f05.mean` | 0.00540604 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh.f05.mean` | 0.874943 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh.f05.std` | 0.058205 | derived (sample sd over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh.f05_anom.mean` | 0.179673 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh.fa_yr.mean` | 7.67139 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh.p.mean` | 0.957819 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh.r.mean` | 0.676243 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.conformal_bh_rolling.f05.mean` | 0.48655 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.m2.A3.threshold.f05.mean` | 0.853376 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `pre.seed.m1_target_event_ar8_all_c3_s1_event.conformal_bh.f05` | 0.358179 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/m1_target_event_ar8_all_c3_s1_event.json` |
| `pre.seed.m1_target_event_ar8_all_c3_s2_event.conformal_bh.f05` | 0.134942 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/m1_target_event_ar8_all_c3_s2_event.json` |
| `pre.seed.m1_target_gbt_ar8_s1_event.conformal_bh.f05` | 0.157195 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/m1_target_gbt_ar8_s1_event.json` |
| `pre.seed.m1_target_gbt_ar8_s1_event.conformal_bh.fa_yr` | 33.8538 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/m1_target_gbt_ar8_s1_event.json` |
| `pre.seed.m1_target_gbt_ar8_s4_event.conformal_bh.f05` | 0.769227 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/m1_target_gbt_ar8_s4_event.json` |
| `pre.seed.m1_target_gbt_ar8_s4_event.conformal_bh.fa_yr` | 0.142843 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/m1_target_gbt_ar8_s4_event.json` |
| `prereg.freeze_id` | 09a98f5e160e | measured | `research_design/preregistration/freeze_09a98f5e160e.json` |
| `prereg.freeze_written` | 2026-10-24T15:56:29 | measured | `research_design/preregistration/freeze_09a98f5e160e.json` |
| `prereg.frozen_code_files` | 27 | derived (frozen files minus the document) | `research_design/preregistration/freeze_09a98f5e160e.json` |
| `prereg.frozen_files` | 28 | derived | `research_design/preregistration/freeze_09a98f5e160e.json` |
| `prereg.n_deviations` | 8 | derived | `research_design/preregistration/DEVIATIONS.md` |
| `prereg.refreeze_id` | 681cc7ddec86 | measured | `research_design/preregistration/freeze.json` |
| `runs.failed` | 3 | derived | `outputs/experiments/index.jsonl` |
| `runs.ok` | 185 | derived | `outputs/experiments/index.jsonl` |
| `runs.total` | 188 | derived (line count) | `outputs/experiments/index.jsonl` |
| `sel.m1.A1.conformal_bh.cv.mean` | 0.623717 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m1.A1lw.conformal_bh.cv.mean` | 0.527775 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m1.A1lw.conformal_bh.k.mean` | 30 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m1.A1lw.conformal_bh.q.mean` | 0.01 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m1.A2.conformal_bh.cv.mean` | 0.727324 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m1.A3.conformal_bh.cv.mean` | 0.536193 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m1.n_train_events` | 109 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/m1_target_ar8.json` |
| `sel.m1_target_ar8.conformal_bh.k` | 10 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/m1_target_ar8.json` |
| `sel.m1_target_ar8.conformal_bh.q` | 0.0001 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/m1_target_ar8.json` |
| `sel.m2.A1.conformal_bh.cv.mean` | 0.918858 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m2.A2.conformal_bh.cv.mean` | 0.889757 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `sel.m2.A3.conformal_bh.cv.mean` | 0.906206 | derived (mean over seeds) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/selection/*.json` |
| `seleff.bh.n` | 4 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.bh.n_negative` | 2 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.m1.full.conformal_bh` | 0.353455 | derived (A1 minus A0) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.m1.full.threshold` | 0.339559 | derived (A1 minus A0) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.m1.lw.conformal_bh` | -0.383348 | derived (A1 lightweight minus exploratory lightweight AR(8) (calibration-window selection)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json + results.csv` |
| `seleff.m1.lw.threshold` | -0.0563854 | derived (A1 lightweight minus exploratory lightweight AR(8) (calibration-window selection)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json + results.csv` |
| `seleff.m2.full.conformal_bh` | -0.0888814 | derived (A1 minus A0) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.m2.full.threshold` | 0.0944298 | derived (A1 minus A0) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.m2.lw.conformal_bh` | 0.0697168 | derived (A1 lightweight minus exploratory lightweight AR(8) (calibration-window selection)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json + results.csv` |
| `seleff.m2.lw.threshold` | 0.156722 | derived (A1 lightweight minus exploratory lightweight AR(8) (calibration-window selection)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json + results.csv` |
| `seleff.n` | 8 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `seleff.n_positive` | 5 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `shift.m1.A2.ratio_max` | 9.34987 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.A2.ratio_min` | 5.08005 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.A2.spearman_f05` | -0.6 | derived (Spearman rho over 5 seeds; descriptive only) | `calibration_shift_m1.json + test/*.json` |
| `shift.m1.A3.ratio_max` | 54.1576 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.A3.ratio_min` | 2.78689 | derived | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.A3.spearman_f05` | 0 | derived (Spearman rho over 5 seeds; descriptive only) | `calibration_shift_m1.json + test/*.json` |
| `shift.m1.a1_ar8.ratio` | 0.815346 | derived (test rate / CV rate) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.a3_s1.ratio` | 34.845 | derived (test rate / CV rate) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.a3_s2.ratio` | 2.78689 | derived (test rate / CV rate) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.a3_s4.ratio` | 6.3156 | derived (test rate / CV rate) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/calibration_shift_m1.json` |
| `shift.m1.supervised.spearman_f05` | -0.115152 | derived (Spearman rho over the 10 supervised runs) | `calibration_shift_m1.json + test/*.json` |
| `spot.classical.diff.f100.test_f1.mean` | 0.0838401 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.classical.diff.f100.test_precision.mean` | 0.0485185 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.classical.diff.f100.test_recall.mean` | 0.308238 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.resnet18_pretrained.both.f100.params.mean` | 1.44058e+07 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.resnet18_pretrained.both.f100.test_f1.mean` | 0.764629 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.resnet18_pretrained.both.f100.test_f1.std` | 0.0334634 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.resnet18_pretrained.both.f100.test_precision.mean` | 0.861827 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.resnet18_pretrained.both.f100.test_recall.mean` | 0.687239 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.run.resnet18_pretrained.both.f100.mse_sum.mean` | 155488 | derived (mean over seeds) | `outputs/benchmarks/vision_spotgeo/runs/*.json` |
| `spot.run.unet.both.f10.mse_sum.mean` | 279284 | derived (mean over seeds) | `outputs/benchmarks/vision_spotgeo/runs/*.json` |
| `spot.run.unet.both.f100.mse_sum.mean` | 174793 | derived (mean over seeds) | `outputs/benchmarks/vision_spotgeo/runs/*.json` |
| `spot.run.unet.both.f25.mse_sum.mean` | 208472 | derived (mean over seeds) | `outputs/benchmarks/vision_spotgeo/runs/*.json` |
| `spot.run.unet.diff.f100.mse_sum.mean` | 251127 | derived (mean over seeds) | `outputs/benchmarks/vision_spotgeo/runs/*.json` |
| `spot.run.unet.raw.f100.mse_sum.mean` | 174208 | derived (mean over seeds) | `outputs/benchmarks/vision_spotgeo/runs/*.json` |
| `spot.unet.both.f10.test_f1.mean` | 0.534516 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f10.test_precision.mean` | 0.763087 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f10.test_recall.mean` | 0.411313 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f100.params.mean` | 7.78402e+06 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f100.test_f1.mean` | 0.746008 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f100.test_f1.std` | 0.00534405 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f100.test_precision.mean` | 0.846021 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f100.test_recall.mean` | 0.66719 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f25.test_f1.mean` | 0.663031 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f25.test_precision.mean` | 0.854442 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.both.f25.test_recall.mean` | 0.541684 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.diff.f100.test_f1.mean` | 0.64403 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.diff.f100.test_precision.mean` | 0.748795 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.diff.f100.test_recall.mean` | 0.564983 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.raw.f100.test_f1.mean` | 0.750695 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.raw.f100.test_precision.mean` | 0.880567 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `spot.unet.raw.f100.test_recall.mean` | 0.654209 | measured | `outputs/benchmarks/vision_spotgeo/summary.csv` |
| `sur.fp.cep50_relerr_max` | 0.0518922 | derived | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.fp.cep50_relerr_median` | 0.0173736 | derived (|CEP50 sur - ref| / ref) | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.fp.coverage_median` | 0.953257 | derived | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.fp.coverage_min` | 0.926667 | derived | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.fp.n` | 12 | derived | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.fp.speedup_median` | 994.437 | derived | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.mode.n_seeds` | 3 | derived | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.none.cep50_mean` | 498.7 | derived | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.none.seconds_mean` | 0.0966667 | derived (mean over 3 seeds) | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.range.cep50_mean` | 498.733 | derived | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.range.seconds_mean` | 8.71667 | derived (mean over 3 seeds) | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.range.used_mean` | 0.99 | derived | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.ref.cep50_mean` | 485 | derived | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.ref.seconds_mean` | 11.69 | derived (mean over 3 seeds) | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.spread.cep50_mean` | 490.933 | derived | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.mode.spread.seconds_mean` | 8.86667 | derived (mean over 3 seeds) | `outputs/benchmarks/reentry_surrogate/pipeline_fallback_modes.csv` |
| `sur.ood_auroc` | 0.726127 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.ood_test.n` | 20000 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.ood_test.position_error_km_median` | 38.0568 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.ood_test.position_error_km_p95` | 362.05 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.downrange_relative_error_median` | 0.0321906 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.fall_time_mae_s` | 41.2777 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.impact_classification_accuracy` | 0.995075 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.n` | 40000 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.position_error_km_median` | 16.173 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.position_error_km_p95` | 69.3508 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `sur.test.surrogate_samples_per_s` | 80107.3 | measured | `outputs/benchmarks/reentry_surrogate/results.json` |
| `syn.heatmap.map50.mean` | 0.780142 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `syn.heatmap.map50_95.mean` | 0.3705 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `syn.tinyyolo.map50.mean` | 0.378738 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `syn.tinyyolo.map50_95.mean` | 0.161963 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `syn.tinyyolo.params.mean` | 320313 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `syn.yolov8n.map50.mean` | 0.431364 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `syn.yolov8n.map50_95.mean` | 0.244344 | measured | `outputs/benchmarks/vision_cosmos_synthetic/summary.csv` |
| `tests.ai.failed` | 0 | measured | `outputs/tests/test_ai_after_prereg.log` |
| `tests.ai.passed` | 45 | measured | `outputs/tests/test_ai_after_prereg.log` |
| `tests.datasets.log_tail` | SUMMARY {'PASS': 20, 'ERROR': 1} | measured | `outputs/tests/test_datasets_after_per_test.log` |
| `tests.verify.passed` | 20 | measured | `outputs/tests/test_verify_after.log` |
| `tests.verify.total` | 20 | measured | `outputs/tests/test_verify_after.log` |
| `verify.1.expected` | < 0.01 % | measured | `outputs/tests/test_verify_after.log` |
| `verify.1.measured` | 0.0019 % | measured | `outputs/tests/test_verify_after.log` |
| `verify.1.name` | US76 density vs published table | measured | `outputs/tests/test_verify_after.log` |
| `verify.1.reference` | published table | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.10.expected` | 0 < d < 82 m | measured | `outputs/tests/test_verify_after.log` |
| `verify.10.measured` | 81 m | measured | `outputs/tests/test_verify_after.log` |
| `verify.10.name` | Coriolis deflection is eastward and below the free-fall valu | measured | `outputs/tests/test_verify_after.log` |
| `verify.10.reference` | closed-form bound | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.11.expected` | < 1 m | measured | `outputs/tests/test_verify_after.log` |
| `verify.11.measured` | 0.060 m | measured | `outputs/tests/test_verify_after.log` |
| `verify.11.name` | impact point converges as rtol tightens 1e-6 -> 1e-12 | measured | `outputs/tests/test_verify_after.log` |
| `verify.11.reference` | convergence (self-consistency) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.12.expected` | 15-25 km | measured | `outputs/tests/test_verify_after.log` |
| `verify.12.measured` | 18.4 km | measured | `outputs/tests/test_verify_after.log` |
| `verify.12.name` | J2 makes a circular orbit oscillate in altitude | measured | `outputs/tests/test_verify_after.log` |
| `verify.12.reference` | physical expectation | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.13.expected` | no impact | measured | `outputs/tests/test_verify_after.log` |
| `verify.13.measured` | timeout | measured | `outputs/tests/test_verify_after.log` |
| `verify.13.name` | vacuum orbit does not decay | measured | `outputs/tests/test_verify_after.log` |
| `verify.13.reference` | run shorter than any decay (weak) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.14.expected` | 1.000 | measured | `outputs/tests/test_verify_after.log` |
| `verify.14.measured` | 1.0000 | measured | `outputs/tests/test_verify_after.log` |
| `verify.14.name` | break-up conserves the parent mass | measured | `outputs/tests/test_verify_after.log` |
| `verify.14.reference` | enforced mass closure (by construction) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.15.expected` | < 5 % | measured | `outputs/tests/test_verify_after.log` |
| `verify.15.measured` | 3.80 % | measured | `outputs/tests/test_verify_after.log` |
| `verify.15.name` | sampled size distribution follows the truncated power law | measured | `outputs/tests/test_verify_after.log` |
| `verify.15.reference` | closed-form sampling law | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.16.expected` | < 0.5 px | measured | `outputs/tests/test_verify_after.log` |
| `verify.16.measured` | 0.225 px | measured | `outputs/tests/test_verify_after.log` |
| `verify.16.name` | registration recovers a known shift with the right SIGN | measured | `outputs/tests/test_verify_after.log` |
| `verify.16.reference` | imposed shift (known answer) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.17.expected` | < 0.3 px | measured | `outputs/tests/test_verify_after.log` |
| `verify.17.measured` | 0.118 px | measured | `outputs/tests/test_verify_after.log` |
| `verify.17.name` | residual misalignment after registration | measured | `outputs/tests/test_verify_after.log` |
| `verify.17.reference` | imposed shift (known answer) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.18.expected` | >= 0.87 | measured | `outputs/tests/test_verify_after.log` |
| `verify.18.measured` | 0.9065 | measured | `outputs/tests/test_verify_after.log` |
| `verify.18.name` | conformal density zone coverage at nominal 90% | measured | `outputs/tests/test_verify_after.log` |
| `verify.18.reference` | conformal coverage guarantee | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.19.expected` | >= 0.92 | measured | `outputs/tests/test_verify_after.log` |
| `verify.19.measured` | 0.9495 | measured | `outputs/tests/test_verify_after.log` |
| `verify.19.name` | conformal density zone coverage at nominal 95% | measured | `outputs/tests/test_verify_after.log` |
| `verify.19.reference` | conformal coverage guarantee | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.2.expected` | 0 K | measured | `outputs/tests/test_verify_after.log` |
| `verify.2.measured` | 0.000 K | measured | `outputs/tests/test_verify_after.log` |
| `verify.2.name` | US76 temperature vs published table | measured | `outputs/tests/test_verify_after.log` |
| `verify.2.reference` | layer constants (by construction) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.20.expected` | < 0.95 | measured | `outputs/tests/test_verify_after.log` |
| `verify.20.measured` | 0.9430 | measured | `outputs/tests/test_verify_after.log` |
| `verify.20.name` | Gaussian ellipse under-covers on a skewed sample | measured | `outputs/tests/test_verify_after.log` |
| `verify.20.reference` | known property of the ellipse | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.3.expected` | < 1 % | measured | `outputs/tests/test_verify_after.log` |
| `verify.3.measured` | 0.933 % | measured | `outputs/tests/test_verify_after.log` |
| `verify.3.name` | hydrostatic balance dp/dz = -rho g | measured | `outputs/tests/test_verify_after.log` |
| `verify.3.reference` | uses g0, not local g (mis-specified) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.4.expected` | < 1e-6 m | measured | `outputs/tests/test_verify_after.log` |
| `verify.4.measured` | 1.97e-10 m | measured | `outputs/tests/test_verify_after.log` |
| `verify.4.name` | WGS-84 geodetic round trip | measured | `outputs/tests/test_verify_after.log` |
| `verify.4.reference` | round trip (self-consistency) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.5.expected` | 10007.5 km | measured | `outputs/tests/test_verify_after.log` |
| `verify.5.measured` | 10007.5 km | measured | `outputs/tests/test_verify_after.log` |
| `verify.5.name` | great-circle quarter of the equator | measured | `outputs/tests/test_verify_after.log` |
| `verify.5.reference` | same radius on both sides (self-consistency) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.6.expected` | 2.1 | measured | `outputs/tests/test_verify_after.log` |
| `verify.6.measured` | 2.100 | measured | `outputs/tests/test_verify_after.log` |
| `verify.6.name` | C_D at Kn -> infinity reaches the free-molecular limit | measured | `outputs/tests/test_verify_after.log` |
| `verify.6.reference` | code constant (by construction) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.7.expected` | 0.92 | measured | `outputs/tests/test_verify_after.log` |
| `verify.7.measured` | 0.920 | measured | `outputs/tests/test_verify_after.log` |
| `verify.7.name` | C_D at Kn -> 0, Mach 20 reaches the continuum hypersonic val | measured | `outputs/tests/test_verify_after.log` |
| `verify.7.reference` | code constant (by construction) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.8.expected` | > 0.92 | measured | `outputs/tests/test_verify_after.log` |
| `verify.8.measured` | 1.012 | measured | `outputs/tests/test_verify_after.log` |
| `verify.8.name` | C_D peaks in the transonic regime above the hypersonic value | measured | `outputs/tests/test_verify_after.log` |
| `verify.8.reference` | tuned bump (by construction) | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.9.expected` | 1.00 | measured | `outputs/tests/test_verify_after.log` |
| `verify.9.measured` | 0.9984 | measured | `outputs/tests/test_verify_after.log` |
| `verify.9.name` | terminal velocity: drag / gravity at impact | measured | `outputs/tests/test_verify_after.log` |
| `verify.9.reference` | physical limit | derived (classification of the reference) | `tests/test_verify.py` |
| `verify.n_external` | 10 | derived | `tests/test_verify.py` |
| `verify.n_internal` | 9 | derived | `tests/test_verify.py` |
| `verify.n_misspecified` | 1 | derived | `tests/test_verify.py` |
| `xcomp.m1.x1.ci95_high` | 0.344141 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x1.ci95_low` | -0.132519 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x1.diff` | 0.1077 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x1.diff_official` | 0.115808 | derived (official seed mean A2 minus A1 (conformal_bh)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |
| `xcomp.m1.x4.ci95_high` | 0.277958 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x4.ci95_low` | -0.188238 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x4.diff` | 0.0386365 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x5.ci95_high` | 0.164194 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x5.ci95_low` | -0.25979 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m1.x5.diff` | -0.0119558 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m2.a1_minus_a0.threshold` | 0.0944298 | derived (A1 minus A0 under the threshold layer (not a pre-registered test)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json + arms` |
| `xcomp.m2.x1.ci95_high` | 0.0813228 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m2.x1.ci95_low` | -0.016109 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m2.x1.diff` | 0.0304199 | measured | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/exploratory/exploratory_comparisons.json` |
| `xcomp.m2.x1.diff_official` | 0.0304197 | derived (official seed mean A2 minus A1 (conformal_bh)) | `outputs/benchmarks/telemetry_esa_adb/oof_protocol/test/*.json` |