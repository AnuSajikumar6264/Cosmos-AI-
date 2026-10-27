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
- "target" channel sets ({{data.esa.m1.target_channels}} and {{data.esa.m2.target_channels}} channels) and smaller "lightweight" subsets;
- a hierarchy of metrics with the corrected event-wise $F_{0.5}$ as the primary score.

The benchmark paper reports results for a set of TimeEval algorithms, including Telemanom-ESA, DC-VAE-ESA [@garcia2022dcvae], isolation forests and the GlobalSTD rule. On the full channel sets the best primary scores were:

- {{lit.dmlr.m1.full.best:.3f}} on Mission 1 ({{lit.dmlr.m1.full.best_name}});
- {{lit.dmlr.m2.full.best:.3f}} on Mission 2 ({{lit.dmlr.m2.full.best_name}}).

On the lightweight subsets the best scores were {{lit.dmlr.m1.lw.best:.3f}} and {{lit.dmlr.m2.lw.best:.3f}}. The authors attribute the poor Mission 1 results of the unsupervised algorithms to a massive number of false detections, caused by noise and varying sampling rates. They credit Telemanom's relative success to its dynamic thresholding and pruning, and conclude that proper thresholding and post-processing are important [@kotowski2026esaadb].

### Results reported on ESA-ADB since the benchmark

Several studies published during 2025–2026 bear directly on the questions of this thesis. [@tbl:lit-esa] summarises them.

: Results reported on ESA-ADB after the benchmark paper. The protocols differ, so the numbers are **not** directly comparable with each other or with this thesis. They are listed with their protocol. {#tbl:lit-esa}

| Work | Protocol | Method | Reported primary score |
|---|---|---|---|
| Kotowski et al. [@kotowski2026esaadb] | official split; full / lightweight channels | TimeEval baselines with detector-specific thresholds | M1 {{lit.dmlr.m1.full.best:.3f}}, M2 {{lit.dmlr.m2.full.best:.3f}} (full) |
| Brancato et al. [@brancato2027ast] | "full ESA-ADB hierarchical protocol", Mission 1 only (abstract; full text not accessed) | label-free sliding minimum Euclidean distance (SMED); quantile Gaussian-process regression with adaptive retraining | M1 full: SMED {{lit.brancato.smed_full:.3f}}, Q-GPR {{lit.brancato.qgpr_full:.3f}}, out-of-limit reference {{lit.brancato.ool_full:.3f}} |
| Alcarria et al. [@alcarria2026] | non-official: monthly incremental retraining over the whole mission | LSTM autoencoder (M1), PCA + k-means + LOF (M2), adaptive extreme-value thresholds | M1 {{lit.alcarria.m1_f05:.3f}}, M2 {{lit.alcarria.m2_f05:.3f}} |
| PitIA (as cited by Alcarria et al.) [@tejedor2025pitia] | "static train/test" | industrial unsupervised system | M1 {{lit.alcarria.pitia_m1:.3f}}, M2 {{lit.alcarria.pitia_m2:.3f}} |
| Allegrini and Pompei [@allegrini2026] | Kaggle challenge on Mission 1, hidden test split; anomalies and rare events not separated | supervised hierarchical ensemble | {{lit.allegrini.public:.3f}} public, {{lit.allegrini.private:.3f}} private leaderboard |
| Nguyen et al. [@nguyen2026] | uncorrected event-wise metrics; lightweight and full channel sets of both missions; own preprocessing | supervised vs unsupervised benchmark | not comparable |
| Şahin, TC-CAD [@sahin2026tccad] | official code; {{lit.tccad.m1_channels}} / {{lit.tccad.m2_channels}} structurally chosen channels; shortest training windows; {{lit.tccad.seeds}} seeds; pre-registered | telecommand conditioning; per-channel split, sliding and adaptive conformal thresholds; operator deferral | conditioning: negative; anomaly event-wise $F_{0.5}$ {{lit.tccad.c0_anom_ewf05:.3f}} with Telemanom's threshold, of order {{lit.tccad.c3_anom_ewf05_order}} with point-level ACI |
| Heydari et al. [@heydari2026] | two missions, {{lit.heydari.shared_channels}} shared channels, fixed scoring pipeline | cross-mission threshold transfer under alert-workload budgets | policy diagnostics (no detector comparison) |

Four observations follow from this body of work.

1. **Protocols vary widely.** Reported Mission 1 numbers range from below 0.1 to about 0.9, but they come from different splits, retraining regimes, channel sets and label treatments. Brancato et al. report {{lit.brancato.ool_full:.3f}} for an out-of-limit reference on Mission 1. [@sec:results-baselines] shows that an out-of-limit rule under the official protocol, as implemented here, scores far lower. The difference cannot be resolved without their implementation.
2. **Calibration and triage have been studied, at the point level.** The pre-registered TC-CAD study [@sahin2026tccad] has three relevant findings:
   - Conditioning a detector on the telecommand record did not reduce confusion between commanded events and anomalies.
   - Adaptive conformal thresholds per channel and per point were the best calibrated of their layers. They still exceeded the target point-wise false-alarm rate (by {{lit.tccad.aci_far_overshoot}}× according to the deposit), and they collapsed the event-wise score.
   - Two-threshold deferral under an operator budget removed {{lit.tccad.deferral_pp}} percentage points of automatic alarms on rare nominal events. On Mission 1 this test was post hoc.
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

OPS-SAT was an ESA nanosatellite designed to demonstrate new satellite-control software under flight conditions [@evans2014opssat]. OPSSAT-AD is a dataset of labelled telemetry fragments from this mission, with an official train/test split, {{data.opssat.n_features}} handcrafted features and results for a broad set of baseline models [@ruszczak2025opssat]. The strongest reported model was a fully connected network on the features, with F1 {{lit.opssat.fcnn_f1:.3f}} and AUC-PR {{lit.opssat.fcnn_aucpr:.3f}}. AdaBoost reached F1 {{lit.opssat.adaboost_f1:.3f}}.

## Optical detection of space objects

The SpotGEO challenge released {{data.spotgeo.train.sequences}} training and {{data.spotgeo.test.sequences}} test sequences of five frames from a low-cost ground-based telescope, labelled with the positions of geostationary and near-geostationary objects [@chen2021spotgeo]. Submissions were ranked on $1-F_1$ and a mean squared error on the full test set, after matching with the Hungarian algorithm [@kuhn1955hungarian]. The winning entry reached $1-F_1$ = {{lit.spotgeo.winner_1mf1:.4f}}, i.e. $F_1$ ≈ {{lit.spotgeo.winner_f1:.3f}}. According to the challenge paper, the leading solutions post-processed detections over time, using line detection and trajectory filling [@chen2021spotgeo].

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
