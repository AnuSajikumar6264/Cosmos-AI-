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

Under this metric, on the full sets of monitored channels, no baseline reported with the benchmark exceeds {{lit.dmlr.m2.full.best:.3f}}. The benchmark's authors attribute much of this failure to thresholding and post-processing [@kotowski2026esaadb].

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
   - A parameter-free out-of-limit rule scores {{explo.m2.full.ool.native.f05.mean:.3f}} on the full Mission 2 channel set. This shows that Mission 2 is close to saturated.
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
