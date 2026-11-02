---
title: "COSMOS-AI: a physics-aware, uncertainty-calibrated digital twin for orbital debris analysis, with atmospheric re-entry prediction and satellite health monitoring"
author:
  - "[Author name(s)]^a^"
  - "^a^[Affiliation, postal address, country]"
  - "Corresponding author: [name, e-mail]"
abstract: |
  Orbital debris analysis must forecast how long objects stay in orbit and how likely they are to collide, yet practice rests on uncalibrated physics propagators. Prior lifetime studies used a few hundred to 1,800 objects, single-epoch predictions and only decayed objects; tabular foundation models have been assessed on curated benchmarks, not censored catalogues under drift. This study proposes an artificial-intelligence architecture for the debris environment, each component measured against the practice it replaces. COSMOS-AI is a digital twin holding one state per object. Debris forecasts collision risk from conjunction messages; re-entry predicts residual lifetime by a reliability-gated mixture of experts, a learned gate weighting a Monte-Carlo decay prior against 2025–2026 tabular models (TabM; in-context foundation models TabICLv2 and Kumo-Tabular); health scores telemetry with Chronos-2 zero-shot; quantiles receive Mondrian conformal and isotonic calibration. Four couplings make this one twin, not three independent models: breakup-family statistics feed the lifetime learner, per-object survival sums into cloud forecasts, operational status selects the decay regime, and health flags escalate high-risk conjunctions. Public data throughout: 64,516 catalogued objects in two forward-in-time periods, the European Space Agency collision-avoidance benchmark and NASA SMAP/MSL telemetry. In 2016–2021 the foundation models cut re-entry pinball 13 % below the best gradient-boosted baseline and 44 % below the physics expert, though tied with it in 2011–2015. Conformal intervals covered 72.6 %–90.6 % of decayed objects against 12.2 %–17.1 % for the operational ±20 % rule, designed for the final days before re-entry, and history-conditioned state estimation cut 72-day over-prediction from 9.5-fold to 1.4-fold. Collision risk improved 37 % over persistence, yet none beat the latest-risk baseline on the official metric; zero-shot telemetry detection did not beat simple detectors. Three couplings were testable — family evidence helped in one period, cloud survival reached a tenth of a cloud, the status gain was not significant — and the fourth lacks public test data. Calibrated learned fusion advances debris forecasting where the full distribution matters, and covariate availability is a leakage channel. A dated, hashed forecast enables blind verification.

  **Keywords:** Tabular foundation models; In-context learning; Mixture of experts; Conformal prediction; Digital twin; Space debris; Atmospheric re-entry prediction; Space situational awareness
---


# Introduction {#sec:intro}

More than 30,000 tracked objects share low Earth orbit with the satellites that operators must keep healthy, steer clear of debris and eventually bring down. These three tasks are usually handled by separate tools and teams. Satellite health monitoring looks for anomalies in telemetry; conjunction assessment estimates the probability that two objects collide; and re-entry prediction estimates when an object will decay and what that implies for people on the ground. Yet the tasks share state. A satellite that loses its ability to manoeuvre stops station-keeping and begins to decay naturally; fragments of one breakup share an origin, and so a range of area-to-mass ratios, that shapes how long they stay in orbit; and how long fragments stay up determines the debris environment that conjunction assessment faces.

A digital twin of the tracked population, with one state per object that all three functions read and update, is the natural way to make these links explicit. Digital twins have been proposed for space systems mostly as architectures and reviews [@liu2024digital], or for single functions such as orbit prediction [@xu2025digital]; we found no system that couples health, debris and re-entry functions and measures what the coupling contributes. Building one on public data meets a hard constraint: there is no public dataset that follows the same object through telemetry, conjunctions and re-entry. Telemetry benchmarks are anonymised [@kotowski2026esaadb; @ruszczak2025opssat; @hundman2018], conjunction messages are anonymised [@uriot2022spacecraft], and catalogue data carry no telemetry. A credible twin therefore cannot be a joint model trained on unpaired data. It has to be a set of modules, each validated on its own real data, joined by couplings that are defined explicitly and tested wherever a real test exists.

Re-entry prediction is the module where the gap to practice is largest. Operational re-entry predictions are physics-based and concentrate on the final weeks, with windows of about ±20 % of the residual lifetime that cover the true epoch roughly 90 % of the time [@pardini2018assessing]. Machine-learning studies of re-entry and lifetime have used Space-Track orbital histories of a few hundred to about 1,800 objects, point predictions and, as far as we could verify, only decayed objects [@salmaso2023machine; @jung2021recurrent; @lotti2025exploring]. A catalogue-wide model that handles objects still in orbit, is evaluated forward in time across solar-cycle phases and produces intervals with a coverage guarantee was not found in the literature we screened (Section [-@sec:related]).

This paper presents COSMOS-AI, a digital-twin framework for satellite health monitoring, orbital debris analysis and atmospheric re-entry risk prediction, and evaluates it on real data only. Its objectives are:

1. to build and validate a catalogue-wide re-entry module that fuses a physics expert with learned experts and calibrates its uncertainty without distributional assumptions;
2. to validate a debris module (conjunction-risk forecasting and breakup-family evidence) and a health module (telemetry anomaly detection) on public benchmarks;
3. to define the couplings between the modules and measure, on real data, what each coupling contributes.

The contributions are:

* **A twin framework with tested couplings.** One state per object, three modules sharing one uncertainty principle (conformal calibration of alarms, intervals and probabilities), and four couplings, three of which are tested on real data: breakup-family evidence for re-entry prediction, breakup-cloud survival forecasting, and operational status as health evidence that selects the re-entry regime. The fourth (health × debris → collision response) has no public test data and is labelled a demonstration. The twin's forecasts for objects already in orbit condition each object's area-to-mass ratio on its catalogue history. They are scored against later catalogue decays in an as-of-date test (T1) on a window that had been analysed earlier in the project; a dated, hashed forecast is released for a blind test.
* **A licence-clean, leakage-controlled re-entry benchmark** built from GCAT [@mcdowell2020gcat] and CelesTrak [@celestrak2026satcat]: 64,516 objects with low-Earth-orbit canonical orbits, administrative censoring at the forecast origin, and two independent five-year test periods.
* **A reliability-gated re-entry model** that averages the quantile functions of a Monte-Carlo physics expert and a learned expert, with Mondrian conformal intervals. The physics expert uses orbit-averaged King-Hele drag, exact for eccentric orbits, with NRLMSIS 2.1 densities, and area-to-mass ratios from design values, class priors or, for fragments, the NASA Standard Breakup Model. The gate is driven by the physics expert's own reliability descriptors.
* **A documented outcome leak in catalogue covariates.** Whether a catalogued object has a published radar cross-section depends on how long it stayed in orbit. Using it broke interval coverage in the test period whose missingness pattern differed from training; we exclude it and provide a check for such leaks.
* **An honest account of where learning helps and where it does not**, including results in which the proposed components do not beat simpler alternatives.

# Related work {#sec:related}

**Re-entry and lifetime prediction.** Semi-analytic decay theory [@kinghele1987satellite; @frey2019extension] and numerical propagators with empirical atmospheres underpin operational lifetime tools such as ESA's DRAMA/OSCAR, whose predictions were compared with 981 observed lifetimes by @operti2017assessing. Assessments of the IADC re-entry campaigns show mean errors of about 10 % of the remaining lifetime and a ±20 % window covering about 90 % of epochs near re-entry [@pardini2018assessing]. Learning-based studies predict the re-entry epoch from orbital histories: @salmaso2023machine trained a sequence model on 417 Space-Track objects with tracking-and-impact messages and tested it on 15 IADC objects; @jung2021recurrent used a recurrent network on TLE-derived altitude histories; physics-informed networks for lifetime are emerging [@lotti2025exploring]. A recent physics-based backtest reports calibrated lifetime distributions for 1,283 objects but calibrates on the same backtest set [@shambaugh2026validation]. Uncertainty in thermospheric density, the main driver, has been modelled with calibrated neural networks [@licata2022uncertainty].

**Debris analysis and conjunction risk.** The ESA Kelvins Collision Avoidance Challenge released Conjunction Data Messages with the final collision risk of each event [@uriot2022spacecraft; @uriot2021cacdataset]. The latest-risk baseline scored 0.694 on the official metric and only 12 of 97 teams beat it; the best entry scored 0.556 [@uriot2022spacecraft]. Later entries report scores on the same split (0.628 for @tulczyjew2021predicting), and a pre-registered re-evaluation found that no model beats the latest-risk forecast with confidence [@vazquez2026latest]. Debris families have been studied for origin attribution [@wu2023investigation; @celletti2021reconnecting]; the Kelvins "Space debris: the origin" data are simulated [@efthymiopoulos2021origin]. We found no study that uses breakup-family information as a predictor of fragment lifetime.

**Telemetry anomaly detection.** ESA-ADB [@kotowski2026esaadb], OPSSAT-AD [@ruszczak2025opssat] and NASA SMAP/MSL [@hundman2018] are the public spacecraft benchmarks; flawed metrics such as point adjustment can make random detectors look strong [@kim2022rigorous; @wu2021flawed], so event-wise and point-wise metrics without adjustment are used here.

**Uncertainty.** Conformal prediction gives finite-sample, distribution-free coverage under exchangeability [@vovk2005algorithmic; @angelopoulos2023gentle]; conformalised quantile regression adapts interval width to the input [@romano2019cqr]; Mondrian conformal prediction conditions coverage on groups [@vovk2005algorithmic]. Averaging quantile functions is a well-studied way to combine forecasts [@lichtendahl2013average]. In our reading, conformal intervals have not been applied to re-entry or lifetime prediction.

# Data {#sec:data}

All data are public and real; their provenance, licences, checksums and limitations are listed in the supplementary dataset audit.

**Catalogue (re-entry and debris modules).** GCAT [@mcdowell2020gcat] (release 1.8.8, CC-BY-4.0, retrieved 2026-10-27) lists for each catalogued object one *canonical orbit*, "representative of the spacecraft's early operations, after initial positioning", with its epoch; design mass and dimensions; the parent object (for fragments, the object that broke up); and the end of its current phase with a status code that separates natural decay from active de-orbit, landing or docking. The CelesTrak SATCAT [@celestrak2026satcat] adds operational status. CelesTrak's space-weather file provides F10.7 and Ap since 1957; only observed rows of days that ended before the prediction time are used. The SATCAT's radar cross-section (RCS) is *not* used as a predictor, because its availability encodes the outcome. Values were mostly never published for objects that decayed quickly, and none are published for objects catalogued after about 2018. In the origin-2016 training cohort, 75.9 % of objects without an RCS decayed within 30 days, against 17.2 % of those with one. Adjusted for orbit, class and decade, the odds ratio is 3.59 (95 % CI 2.09–6.17). The effect is quantified in Section [-@sec:results]. The cohort holds 64,516 objects whose canonical orbit has perigee below 2000 km, of which 32,970 decayed naturally; the exclusions are counted in the build log. Fig. [-@fig:data-reentry] shows the cohort's orbits, its distribution over time with the cohorts of the two forecast origins, the observed lifetimes and the space-weather record.

![**The re-entry dataset.** (a) GCAT canonical orbits of the cohort by object class; payloads are outside the primary population. The classes overlap heavily at these densities, so the panel shows where each class lies rather than its density; the largest class is drawn first. (b) Objects per year of their canonical-orbit epoch, with the training, tuning and test cohorts of the two forecast origins. (c) Time from the canonical-orbit epoch to natural decay, or to censoring for objects still in orbit at the snapshot or whose phase ended without natural decay (a controlled re-entry, a landing, a docking or another non-decay status code); the learning target is capped at five years. (d) F10.7 solar flux from the CelesTrak space-weather file; its 81-day mean is the model input, and the bars mark the two test periods.](artwork/Figure_1.png){#fig:data-reentry width=100%}

**Conjunctions (debris module).** The Kelvins Collision Avoidance Challenge data [@uriot2021cacdataset] contain anonymised CDMs received by ESA in 2015–2019. Following the challenge, inputs are the CDMs released at least two days before closest approach and the target is the risk (log10 collision probability) of the final CDM. Training events are filtered to mirror the test inputs, leaving 8,293 training events (66 high-risk, i.e. final risk ≥ −6) and the official 2,167 test events (150 high-risk). Fig. [-@fig:data-debris] shows the breakup clouds of the catalogue and the conjunction messages.

![**Debris data.** (a) Catalogued fragments in orbit over time for the four largest breakup clouds of the cohort (fragments with a GCAT canonical orbit with perigee below 2000 km). (b) Gabbard diagram of the largest cloud, from the canonical orbits of its fragments. (c) Risk sequences of the conjunction data messages of 75 training events of the ESA Kelvins challenge: 60 drawn at random from the events whose final risk is below −6 and 15 from the 66 high-risk events (fixed seed), so high-risk events are deliberately oversampled. The model inputs are the messages received at least two days before closest approach. (d) Final risk of the training events: most lie at the floor of −30, and few are high-risk.](artwork/Figure_2.png){#fig:data-debris width=100%}

**Telemetry (health module).** NASA SMAP/MSL [@hundman2018]: 81 channels with 68 SMAP and 36 MSL labelled anomaly sequences (the label file lists channel P-2 twice; the two rows are merged). The alarm layer of the health module was developed and tested on ESA-ADB Missions 1 and 2 [@kotowski2026esaadb] in a companion pre-registered study by the same authors, whose results are summarised in Section [-@sec:res-health]. Fig. [-@fig:data-telemetry] shows examples of both.

![**Telemetry data.** (a–d) Two channels of each NASA spacecraft drawn at random (fixed seed): training series (grey), test series (black) and labelled anomaly sequences (shaded). (e) ESA-ADB Mission 1 around the first labelled anomaly of the test half, for the target channels it affects (values min–max scaled per channel within the window; shaded: labels). ESA-ADB is used in the companion study.](artwork/Figure_3.png){#fig:data-telemetry width=100%}

**What is not available.** Orbital-element histories at catalogue scale require a Space-Track account whose user agreement forbids redistribution, and no legitimate public archive exists; DISCOS and ESA re-entry predictions need accounts. The re-entry module therefore predicts from a single dated orbit per object, not from orbital histories.

# The COSMOS-AI framework {#sec:framework}

## Twin state

For object *i* at time *t* the twin holds $s_i(t) = (o_i, \pi_i, h_i, F_i, e, C_i)$: the latest orbit $o_i$ (perigee, apogee, inclination), design properties $\pi_i$, health evidence $h_i$ (catalogue operational status for payloads; the health module's alarm state where telemetry exists), the debris family $F_i$ with sibling statistics restricted to what is known at *t*, space weather $e$, and a conjunction summary $C_i$ when CDMs exist. Every module output attached to the twin records its model, validation status and uncertainty (Fig. [-@fig:arch]).

![**COSMOS-AI digital twin.** Three modules read and update one state per object. Solid arrows are couplings tested on real data in this paper; the dashed coupling has no public test data.](artwork/Figure_4.png){#fig:arch width=100%}

## Re-entry module

*Physics expert.* The semi-major axis $a$ and eccentricity $e$ are integrated with King-Hele's orbit-averaged drag equations for an atmosphere locally exponential about perigee [@kinghele1987satellite]:

$$\Delta a_{\text{rev}} = -2\pi\delta a^2\rho_p e^{-c}\left[I_0 + 2eI_1 + \tfrac{3}{4}e^2(I_0+I_2)\right], \qquad c = ae/H,$$ {#eq:kh}

with $I_n(c)$ modified Bessel functions. $\rho_p$ and $H$ are the density and scale height at perigee from NRLMSIS 2.1 [@emmert2021nrlmsis], with the trailing 81-day F10.7 and Ap at $t_0$ held constant. $\delta = F\,C_D A/m$ is the ballistic coefficient times the atmospheric-rotation factor, and re-entry is declared when the perigee falls below 120 km.

The series (shown truncated) holds only for small eccentricity. Its perigee rate is 0.68 of the exact value at $e = 0.2$ and changes sign near $e = 0.3$. For $e > 0.05$ we therefore integrate the same orbit averages exactly over the eccentric anomaly $E$: $\Delta a_{\text{rev}} = -\delta\rho_p a^2 \int_0^{2\pi} e^{c(\cos E - 1)} (1+e\cos E)^{3/2}(1-e\cos E)^{-1/2}\,dE$. The trapezoid rule is spectrally accurate for this periodic integrand; its relative error is below $10^{-12}$ up to $c = 3000$.

The area-to-mass ratio is sampled per object from one of three sources:
- the design value (log-normal, 0.25 dex);
- for intact payloads and rocket stages without design values, a design value drawn from the training-set distribution of their class;
- for fragments and components without design values, the NASA Standard Breakup Model's bimodal distribution [@johnson2001nasa], with a characteristic length drawn from the training-set distribution of radar-derived sizes of their class.

Each object has its own random stream, so its predictive distribution does not depend on the other objects in a batch. Thirty-two Monte-Carlo samples give a predictive distribution with quantiles $q^{\text{phys}}_\tau$.

*Learned experts.* Three recent models predict quantiles $q^{\text{ml}}_\tau$ of $z = \log_{10}\min(T, 5\,\text{yr})$:
- TabM, a parameter-efficient ensemble of MLPs [@gorishniy2025tabm];
- TabICLv2 [@qu2026tabiclv2], an in-context tabular foundation model;
- Kumo-Tabular [@nvidia2026kumo], also an in-context tabular foundation model.

Their inputs are the orbit, design properties, space weather, family statistics and summaries of the physics distribution. TabM gets a non-crossing quantile head and the same search budget as every tuned learner. The two foundation models are used with their default inference settings. Kumo-Tabular's context is capped at 16,000 training rows by device memory. The candidate with the lowest tuning loss becomes the expert.

Gradient-boosted trees [@chen2016xgboost; @ke2017lightgbm; @prokhorenkova2018catboost] and a plain non-crossing quantile MLP are fitted and reported as classic reference baselines; they are not eligible as the expert. A censoring-aware accelerated-failure-time model (XGBoost AFT), trained on all objects including those still in orbit, is a further reference baseline.

*Reliability-gated fusion.* With reliability descriptors $r_i$ of the physics expert (whether its area-to-mass ratio comes from design values or a class prior, the spread of its own distribution, eccentricity, perigee band, object class),

$$q_\tau(i) = g(r_i)\,q^{\text{phys}}_\tau(i) + \bigl(1-g(r_i)\bigr)\,q^{\text{ml}}_\tau(i), \qquad g(r) = \sigma(w^\top r + b).$$ {#eq:gate}

A convex combination of two quantile functions is a quantile function [@lichtendahl2013average]. $(w, b)$ minimise the pinball loss on out-of-fold predictions of the training period.

*Mondrian conformal calibration.* On a calibration split disjoint from fitting and gating, the scores $S = \max(q_{\text{lo}} - z, z - q_{\text{hi}})$ of each group $k$ (object class × perigee band, with fallback to class and pooled) give the correction $\hat Q_k$, the $\lceil(1-\alpha)(n_k+1)\rceil$-th smallest score, and the interval $[q_{\text{lo}} - \hat Q_k,\; q_{\text{hi}} + \hat Q_k]$ [@romano2019cqr; @vovk2005algorithmic]. Probabilities $P(T \le H)$ are read from the fused quantile function and calibrated on the same split by isotonic regression [@zadrozny2002transforming]. Rank metrics are computed on the raw probabilities, because the isotonic map ties scores; Brier score and ECE are computed on the calibrated ones. The target has a point mass at the 5-year cap, so predicted quantiles within $10^{-6}$ of the cap (in log10 days) are set to it exactly.

*Twin state estimation for objects already in orbit.* The re-entry module is fitted and validated at canonical-orbit epochs. A twin, however, forecasts objects that have been in orbit for years. Their survival since the canonical orbit selects low area-to-mass ratios, which neither the class priors nor the learned experts know about.

For each fragment, rocket stage and component, the twin therefore conditions the area-to-mass ratio on the object's catalogue history, a coarse two-orbit form of ballistic-coefficient estimation from orbital histories [@gondelach2017ballistic; @sang2013estimation; @pardini2008impact]:
1. 128 prior samples are propagated from the GCAT canonical orbit to the as-of date with the realised trailing space weather.
2. Samples that would already have re-entered get weight zero.
3. The others are weighted by a Gaussian likelihood of the current perigee and apogee, with $\sigma = 10\,\text{km} + 0.3\,|\Delta|$, where $\Delta$ is the observed change since the canonical orbit.
4. 32 posterior samples are drawn by systematic resampling. If the effective sample size is below 4, survival alone is used; if no sample survives, the 8 longest survivors are used.

The physics expert then propagates the current orbit with the posterior samples. Payloads keep the class priors, because their past manoeuvres make the decay history uninformative.

## Debris module

*Family evidence.* For the launch family and, for fragments, the breakup family of object *i*: the number of siblings catalogued by $t_0$, the number and fraction that decayed naturally before $t_0$, and their mean log-lifetime. Only events before $t_0$ enter.

*Conjunction risk.* A persistence expert (the latest risk plus the training distribution of final-minus-latest changes in its risk bin) and the same learned experts (TabM, TabICLv2, Kumo-Tabular, with the classic baselines alongside) predict quantiles of the final risk; they are fused by the gate of Eq. [-@eq:gate] and calibrated as above. Because the official metric scores a point forecast, a calibrated P(high risk) also gates the persistence point (predict the latest risk if P(high) exceeds a threshold chosen on out-of-fold training data, otherwise −6.001).

## Health module

Per-channel non-conformity scores from a forecaster fitted on nominal data are turned into conformal p-values against the channel's nominal pool, aggregated across channels with Benjamini–Hochberg control and persistence into alarm runs, as in the companion pre-registered ESA-ADB study. For SMAP/MSL, the level and persistence are fixed in advance and the detectors are:
- the Chronos-2 time-series foundation model [@ansari2025chronos2], used zero-shot: rolling 8-step forecasts of the telemetry value from up to 512 past steps, scored by the absolute residual of the forecast median;
- three classic baselines: ridge autoregression, PCA reconstruction, and nearest-neighbour window distance;
- the scores published with telemanom [@hundman2018].

Each channel's alarms come from conformal p-values against its own training-series scores:
- Chronos-2 is never fitted to these series, so its training scores are out of sample.
- For the fitted baselines, the calibration scores are computed out of fold, in contiguous blocks with a purge gap.
- An alarm is raised causally, at the third consecutive step with $p \le \max(10^{-3}, 1/(n_\text{cal}+1))$.

## Couplings

* **C1 debris → re-entry:** family statistics are inputs of the re-entry experts.
* **C2 re-entry → debris environment:** the expected number of a breakup cloud's fragments in orbit at time *t* is $\sum_{i:\,t_{0,i}\le t} S_i(t - t_{0,i})$, with $S_i$ the fused survival function. $S_i$ is read in two ways:
  - raw, from the quantile function;
  - calibrated, by isotonic maps of $P(T \le a)$ fitted on tune-B fragments at 25 ages and interpolated in log-age.
* **C3 health → re-entry:** a payload reported operational is in an operator-controlled regime, where drag physics does not determine re-entry. The twin withholds the drag forecast and uses the payload base rate of the preceding window. Non-operational payloads, rocket stages, components and fragments are in the natural-decay regime, and payloads with no reported status form a separate unknown regime.
* **C4 health × debris → response:** a high-risk conjunction for a payload with an active health alarm or non-operational status is escalated because its manoeuvre capability is uncertain (demonstration only).

# Experimental protocol {#sec:protocol}

**Re-entry.** Prediction time $t_0$ is the canonical-orbit epoch. The event is natural decay; controlled endings are censored at their date and objects in orbit at the snapshot. Under the protocol *as of origin O*, the training cohort has $t_0 < O - 7$ yr, the tuning cohort $O - 7 \le t_0 < O - 5$ yr (split by launch family into tune-A for early stopping, selection and checks, and tune-B for calibration), and all their outcomes are censored at *O*; and the test cohort starts at $O$ and ends where the five-year follow-up is complete at the snapshot, or at the next origin: $t_0 < $ 2016-01-01 at origin 2011 (2011–2015) and $t_0 < $ 2021-10-27 at origin 2016 (2016–2021). The two origins, 2011-01-01 and 2016-01-01, therefore give two disjoint test periods. The primary population is objects that cannot manoeuvre (fragments, rocket stages, components); payloads are excluded from it because many are operator-controlled. Each tuned learner receives the same budget of 12 random hyperparameter configurations scored on tune-A, and every learner five seeds; TabICLv2 and Kumo-Tabular are used with their defaults. Metrics:
- mean pinball loss over seven quantiles;
- MAE of the median in log10 days;
- absolute error in days for objects that decayed within five years;
- coverage, width and Winkler score of 80 % and 90 % intervals;
- Harrell's C on all test objects [@harrell1982evaluating];
- AUC-PR, AUC-ROC, Brier score and ECE for re-entry within 30 days, six months, one year and five years.

Confidence intervals come from 2,000 cluster-bootstrap replicates over launch families, paired for model differences. Learner predictions are cached under a fingerprint of the data, features and settings, so a stale result cannot enter a comparison. As a labelled leak demonstration (never a candidate), the selected learner is also refitted with the radar cross-section added.

**Debris.** The official Kelvins split is used.
- Training events are split into five folds. For each fold, the tuned learners select their configuration by random search on the other four folds; the selected model then predicts the fold.
- These nested out-of-fold predictions drive learner selection, the gate, isotonic calibration, the decision threshold and the conformal calibration.
- The test models are tuned on a held-out fold of the training events and averaged over five seeds.
- TabICLv2 and Kumo-Tabular have no search: their out-of-fold models use the other four folds as context, and their test models use all training events. Their out-of-fold losses therefore come from more training data than those of the tuned learners, which hold out one of the four folds for their search; this favours them in selection.
- Predicted quantiles within $10^{-4}$ of the risk floor (−30) are set to it exactly, because most final risks lie at the floor and the float32 spacing there is about $2\times10^{-6}$.

The official metric is $L = \text{MSE}_{\text{HR}}/F_2$ with low-risk predictions clipped at −6.001 [@uriot2022spacecraft]. We also report:
- pinball loss;
- AUC-PR and AUC-ROC of the raw P(high risk);
- the Brier score and ECE of its isotonic-calibrated value;
- interval coverage.

**Health.** ESA-ADB as in the companion pre-registration (official split and corrected event-wise $F_{0.5}$); SMAP/MSL with thresholds fixed in advance and no test labels used.

**Couplings.** C1: ablation of the family features. C2: predicted against catalogue counts of fragments in orbit, for every breakup cloud with at least 30 fragments in a test cohort, at 1–5 years after its first catalogued fragment. T1 (twin re-entry forecasts on current orbits): the twin is run as of 2026-08-16, the data date of a CelesTrak snapshot, and its P(re-entry within 30 and 72 days) is scored against the decays recorded in the 2026-10-27 snapshot. Three variants are compared for fragments, rocket stages and components:
- the lifetime pipeline, refitted as of 2026-08-16;
- the physics expert with class priors;
- the history-conditioned physics expert, which is the twin's choice.

All three were specified before the test, but the window is not fresh (see *Deviations and prior use of test data* below). T1 is therefore an as-of-date test on a previously analysed window, not a blind prospective test. The hashed forecast of 2026-10-27 is the blind test.

C3: as-of-date test on the same previously analysed window, with the CelesTrak operational status of 2026-08-16 and decays up to 2026-10-27, for Earth-orbiting payloads with perigee below 600 km.
- *As-of-date* policies use only information available on 2026-08-16:
  - physics for every payload;
  - the previous 72-day payload base rate for every payload;
  - the status-aware twin (base rate for operational payloads, physics otherwise).
- Platt recalibrations, status-blind and status-specific, are fitted on the follow-up window itself and cross-fitted by launch. They are reported only as an in-window diagnostic of attainable calibration.
- Confidence intervals resample launches.

**Deviations and prior use of test data.** The test sets of this study were not untouched when the reported results were produced, and we list what had been seen.
- *Learner pool.* The first complete re-entry run (run 1; Supplementary Table S1) chose the expert among XGBoost, LightGBM, CatBoost, the plain MLP and TabICLv2. After its test results were known at both origins, and after the collision-risk test set had been scored with the same pool, we restricted the eligible pool of both the re-entry and the collision-risk experts to learners published 2022–2026 (TabM, TabICLv2, Kumo-Tabular), so that the expert would be chosen among current methods; the classic learners became reference baselines. The selection rule, budget, seeds, features, gate and calibration were not changed. The reported run is run 2. Run 1 selected the MLP at both origins, and its 2011 gated model was much worse than run 2's (Section [-@sec:results]), so the 2011 gated result depends on this change.
- *Smoke tests.* Before run 2, engineering tests of the new learners printed two test-set scores: Kumo-Tabular's origin-2016 pinball loss (0.051) and TabM's origin-2011 pinball loss (0.0729). No setting was chosen with them.
- *Repeated scoring.* An earlier version of every benchmark was scored on its test set and then corrected after an internal review; the superseded results are archived with the code. The re-entry and collision-risk test sets were therefore scored more than once during development.
- *The T1 and C3 window.* An earlier version of the project fitted and tested re-entry classifiers (physics, logistic regression, XGBoost, XGBoost-AFT and a hazard network) on the window from 2026-08-16 to 2026-10-23, and the aggregate decay count of the window was seen while the lifetime pipeline's bias on current orbits was diagnosed, before the history-conditioned variant was designed. A defect in the twin's payload prior (payloads without design values fell back to the fragment breakup law) was found and fixed after T1 had first been scored; the fix changes payload probabilities only.
- *Post-hoc analyses.* The class breakdown, the gate applied to every learner, the head-to-head comparisons, the calibrated C2 reading and the C2 bootstrap were added after test results had been seen. They are labelled as post hoc, and all intervals are nominal 95 % intervals without correction for multiplicity.


# Results {#sec:results}

## Re-entry prediction at canonical-orbit epochs {#sec:res-reentry}

Table [-@tbl:reentry] lists every re-entry model on both test periods and Fig. [-@fig:reentry] shows their pinball losses.

| Model | Pinball | MAE | Cov. 90 % | Width 90 % | C | Pinball | MAE | Cov. 90 % | Width 90 % | C |
|---|---|---|---|---|---|---|---|---|---|---|
| Physics expert alone | 0.0892 | 0.230 | 0.876 | 0.67 | 0.739 | 0.0527 | 0.148 | 0.974 | 1.03 | 0.766 |
| XGBoost (baseline) | 0.0576 | 0.190 | 0.921 | 0.80 | 0.888 | 0.0648 | 0.231 | 0.894 | 1.01 | 0.807 |
| LightGBM (baseline) | 0.0662 | 0.191 | 0.971 | 1.00 | 0.887 | 0.0683 | 0.221 | 0.981 | 1.18 | 0.811 |
| CatBoost (baseline) | 0.0611 | 0.207 | 0.911 | 0.93 | 0.878 | 0.0627 | 0.201 | 0.788 | 0.98 | 0.821 |
| Quantile MLP (baseline) | 0.0578 | 0.193 | 0.892 | 0.84 | 0.851 | 0.1147 | 0.390 | 0.852 | 1.40 | 0.747 |
| XGBoost AFT, censoring-aware (baseline) | 0.0657 | 0.185 | 0.901 | 0.62 | 0.866 | 0.0531 | 0.156 | 0.961 | 0.87 | 0.813 |
| TabM (2025) | 0.0593 | 0.192 | 0.877 | 0.68 | 0.861 | 0.0994 | 0.354 | 0.918 | 1.38 | 0.783 |
| TabICLv2 (2026) | 0.0499 | 0.166 | 0.962 | 1.00 | 0.894 | 0.0596 | 0.160 | 0.919 | 1.29 | 0.844 |
| Kumo-Tabular (2026) | 0.0508 | 0.162 | 0.974 | 1.07 | 0.899 | 0.0616 | 0.163 | 0.966 | 1.34 | 0.845 |
| Physics + expert, equal weights | 0.0691 | 0.205 | 0.838 | 0.64 | 0.859 | 0.0527 | 0.148 | 0.970 | 1.04 | 0.851 |
| Physics + expert, constant weight | 0.0601 | 0.193 | 0.877 | 0.67 | 0.861 | 0.0608 | 0.162 | 0.960 | 1.31 | 0.847 |
| Reliability-gated, gate fitted on tune-A | 0.0695 | 0.202 | 0.832 | 0.62 | 0.863 | 0.0616 | 0.163 | 0.959 | 1.34 | 0.845 |
| **Reliability-gated (COSMOS)** | 0.0604 | 0.193 | 0.874 | 0.67 | 0.860 | 0.0583 | 0.157 | 0.959 | 1.21 | 0.848 |

: **Re-entry prediction on the two test periods** (left: origin 2016, test 2016–2021; right: origin 2011, test 2011–2015). Pinball loss over seven quantiles and MAE of the median, both in log10 days; coverage and mean width (log10 days) of the Mondrian conformal 90 % interval; Harrell's C. Models marked "baseline" are not eligible as the gated expert. {#tbl:reentry}

![**Test pinball loss of every re-entry model** on the two test periods. Grey: physics expert; blue: reference baselines, fixed-weight fusions and the variant whose gate is fitted on tune-A; green: the 2025–2026 learned experts; orange: the reliability-gated model.](artwork/Figure_5.png){#fig:reentry width=100%}

### Test period 2016–2021 (origin 2016)

**Selection.** On tune-A, TabM had the lowest loss among the recent learners (Supplementary Fig. S1 plots tune-A against
test loss for every learner):

| Learner | Tune-A pinball |
|---|---|
| TabM | 0.0283 |
| TabICLv2 | 0.0369 |
| Kumo-Tabular | 0.0433 |

On the test cohort the ranking is different, and TabM falls behind both foundation models
(Table [-@tbl:reentry]):

| Model | Test pinball |
|---|---|
| TabICLv2 | 0.0499 |
| Kumo-Tabular | 0.0508 |
| TabM | 0.0593 |
| Classic baselines (best: XGBoost) | 0.0576 |
| Physics expert alone | 0.0892 |

Both foundation models are significantly better than the selected TabM (post-hoc paired
comparisons, Supplementary Table S16; a negative difference favours the first model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| TabICLv2 minus TabM | -0.0093 | -0.0162 to -0.0036 |
| Kumo-Tabular minus TabM | -0.0085 | -0.0167 to -0.0017 |
| TabICLv2 minus Kumo-Tabular | -0.0008 | -0.0036 to +0.0024 |

The two foundation models cannot be separated from each other.

**Coverage and concordance.** The raw 90 % quantile intervals of the two in-context foundation
models already cover 95.8 % (TabICLv2) and
97.3 % (Kumo-Tabular) of test objects. TabM's cover
80.8 %.

After Mondrian calibration on tune-B, TabM and the gated model remain below nominal coverage
(87.7 % and 87.4 %). This is consistent with a
loss of exchangeability between the 2009–2010 calibration cohort and the test cohort, although
the shortfall is small relative to the launch-clustered sampling error.
Their intervals are narrower, and their Winkler scores slightly lower:

| Model | Winkler score (90 %) |
|---|---|
| TabM | 1.099 |
| TabICLv2 | 1.129 |

Supplementary Fig. S2 plots coverage against interval width for every model, and
Fig. [-@fig:reentry-outputs] shows individual forecasts and, for every test object with a known
five-year outcome, predicted against observed outcomes.

The foundation models also have the best concordance (C = 0.894 and
0.899). A tuning cohort five to seven years older than the test cohort
did not identify the model that generalises best.

**Gated model.** The reliability-gated model fuses the physics expert with TabM; its mean gate
weight on physics on the test cohort is 0.11. Its
pinball loss of 0.0604 compares as follows (paired cluster
bootstrap; a positive difference favours the gated model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| Physics expert alone | +0.0289 | +0.0148 to +0.0469 |
| TabM alone | -0.0011 | -0.0028 to +0.0002 |
| TabICLv2 | -0.0104 | -0.0179 to -0.0043 |
| Kumo-Tabular | -0.0096 | -0.0181 to -0.0023 |

The gated model is clearly better than physics alone and no different from TabM alone. It is
significantly worse than both foundation models.

**Horizon probabilities.** The foundation models also lead for re-entry within 30 days:

| Model | AUC-PR | Brier score |
|---|---|---|
| Kumo-Tabular | 0.883 | 0.036 |
| TabICLv2 | 0.874 | 0.035 |
| Gated model | 0.829 | 0.051 |
| Physics expert alone | 0.719 | 0.051 |

**Solar activity.** Solar cycle 25 rose faster than the persistence assumption allows. Replaying
the realised F10.7 lowers the physics expert's pinball loss from
0.0892 to 0.0783.
This is a diagnostic only: solar activity after the prediction time accounts for a substantial
part of the physics expert's error in this period.

![**What the re-entry module outputs** (origin 2016, test 2016–2021). (a) Forty test objects drawn at random (fixed seed) among those with a known five-year outcome, sorted by outcome: median and 90 % Mondrian conformal interval of the physics expert and of the reliability-gated model, against the observed time to decay (diamonds) or survival beyond the five-year cap (triangles). (b–d) Predicted median against observed outcome for every such test object, for the physics expert, TabICLv2 and the gated model.](artwork/Figure_6.png){#fig:reentry-outputs width=100%}

### Test period 2011–2015 (origin 2011)

**Selection.** At this origin the tuning cohort is small: 248 objects in
tune-A and 332 in tune-B. Kumo-Tabular had the lowest tune-A loss:

| Learner | Tune-A pinball |
|---|---|
| Kumo-Tabular | 0.0547 |
| TabM | 0.0561 |
| TabICLv2 | 0.0581 |

**Physics expert against the foundation models.** The physics expert alone has the lowest point
estimate of the test pinball loss (0.0527), but it cannot be
separated from the foundation models or from the gated model (post-hoc paired comparisons,
Supplementary Table S16; a negative difference favours the physics expert):

| Physics expert minus | Pinball difference | 95 % CI |
|---|---|---|
| TabICLv2 | -0.0069 | -0.0217 to +0.0122 |
| Kumo-Tabular | -0.0089 | -0.0253 to +0.0133 |
| Gated model | -0.0056 | -0.0199 to +0.0139 |

The physics expert's 90 % conformal coverage is 97.4 %, but its
concordance is low (C = 0.766).

**TabM.** TabM, the expert selected at 2016, has a much higher loss here
(0.0994), as does the plain MLP (0.1147).

**By object class** (post hoc, Supplementary Table S2 and Fig. S3):

| Model | Fragments | Rocket stages | Components |
|---|---|---|---|
| TabM | 0.1122 | 0.0473 | 0.0597 |
| Physics expert | 0.0432 | 0.0733 | 0.0959 |

TabM's error is concentrated in fragments. The physics expert has the lowest loss for fragments,
while the learned experts have lower losses for rocket stages and components (e.g. TabICLv2:
0.0469 and 0.0703).

**Gated model.** The gated model fuses the physics expert with Kumo-Tabular (mean physics weight
0.12) and scores 0.0583.
Paired differences (a positive difference favours the gated model):

| Comparison | Pinball difference | 95 % CI |
|---|---|---|
| Kumo-Tabular alone | +0.0033 | +0.0007 to +0.0055 |
| XGBoost | +0.0066 | +0.0017 to +0.0136 |
| LightGBM | +0.0101 | +0.0023 to +0.0215 |
| CatBoost | +0.0044 | -0.0011 to +0.0122 |
| TabM | +0.0412 | +0.0011 to +0.0905 |
| TabICLv2 | +0.0013 | -0.0016 to +0.0044 |
| Physics expert alone | -0.0056 | -0.0199 to +0.0139 |

The 95 % intervals exclude zero for Kumo-Tabular alone, XGBoost, LightGBM and TabM, but not for
CatBoost, TabICLv2 or the physics expert. The censoring-aware AFT baseline scores
0.0531.

**Solar activity.** Replaying the realised F10.7 changes the physics expert's pinball loss only
from 0.0527 to
0.0516.

### Across both origins

All intervals in this section are nominal 95 % intervals without correction for multiplicity.
Apart from the comparisons of the gated model, the analyses below were added after the test
results were seen, so we read them as exploratory. Fig. [-@fig:reentry-benchmark] collects the
head-to-head comparisons.

1. **TabICLv2 was best or tied in both periods.** At 2016 the two foundation models are
   significantly better than the physics expert, the gated model and the best tree baseline;
   for example, TabICLv2 minus XGBoost is
   -0.0076 (-0.0121 to -0.0036).
   At 2011 TabICLv2, the physics expert and the gated model are statistically tied; Kumo-Tabular
   is tied with TabICLv2 and the physics expert but significantly worse than the gated model that
   fuses it (+0.0033,
   +0.0007 to +0.0055).
2. **Selection on the tuning cohort chose a significantly worse expert at 2016.** There, TabM was
   selected over the two foundation models. At 2011 the selected Kumo-Tabular is tied with
   TabICLv2.
3. **Gating improved the expert it fused at 2011 and was neutral at 2016.** In a post-hoc
   sensitivity analysis that gates every learner (Supplementary Table S3):
   - at 2011 the intervals exclude zero for every learner to which the gate gave non-negligible
     weight (TabICLv2: +0.0027);
   - at 2016 no interval excludes zero, and the point estimates are mostly slightly negative
     (TabICLv2 -0.0046, -0.0112 to
     +0.0007). This is when the physics expert's own error
     was largest.

**First learner set.** The deviation record (Section [-@sec:protocol], Supplementary Table S1)
used XGBoost, LightGBM, CatBoost, the plain MLP and TabICLv2 as candidates. It selected the MLP
at both origins. Its gated model scored 0.1074 at 2011 and
0.0592 at 2016. The reported 2011 gated model therefore depends on
the change of the candidate pool.

**Operational windows** (Supplementary Table S5). The ±20 % residual-lifetime window around the
physics median, the rule used for the final days before re-entry [@pardini2018assessing], covers
12.2 % (2016) and
17.1 % (2011) of the test objects that decayed within
five years, against 72.6 % and
90.6 % for the physics expert's 90 % conformal
interval. The shortfall is a property of forecasts made from a canonical orbit years earlier, not of
the horizon: the window covers a similar share of the objects that decayed within a week of their
canonical-orbit epoch. The rule's reported ~90 % coverage refers to the final days before re-entry,
when the orbit is tracked continuously. Fig. [-@fig:coverage] shows the same rules over all test
objects, including those still in orbit after five years, where the ±20 % window scores higher
because it is wide enough to contain the five-year cap.

![**Head-to-head re-entry comparisons** (post hoc): differences in test pinball loss with paired launch-family cluster-bootstrap 95 % intervals; a negative difference favours the first model. Filled markers: the nominal interval, uncorrected for multiplicity, excludes zero. "Its expert" is TabM at origin 2016 and Kumo-Tabular at origin 2011.](artwork/Figure_7.png){#fig:reentry-benchmark width=100%}

![**Interval coverage on the test periods**, over all test objects, including those still in orbit after five years: the operational ±20 % and ±30 % residual-lifetime windows around the physics median, the raw quantile intervals of the gated model, and its marginal and Mondrian conformal intervals. Supplementary Table S5 gives the same rules restricted to the objects that decayed within five years.](artwork/Figure_8.png){#fig:coverage width=100%}

### An outcome leak through catalogue covariates {#sec:res-leak}

Whether an object has a radar cross-section in the 2026 catalogue depends on its fate.

**Evidence of the leak.** In the origin-2016 training cohort:
- 75.9 % of objects without a value decayed within
  30 days of their canonical-orbit epoch, against 17.2 %
  of objects with one.
- Adjusted for orbit, class and decade, the odds ratio is
  3.59. Its 95 % CI,
  2.09–6.17,
  uses cluster-robust standard errors over launch families.

The test periods differ in how this availability relates to the outcome:
- In the 2011–2015 test cohort, only 1.9 % of objects lack a
  value, and 84.0 % of them decayed within 30 days.
- In the 2016–2021 cohort, the value is missing for 99.9 %
  of objects catalogued from 2018 on, whatever their fate.

**Effect on the selected learner** (refitted with the 2026 RCS added, a labelled demonstration
and never a candidate):

| Origin | Pinball with RCS | Pinball without | Difference (95 % CI) | 90 % coverage with RCS | 90 % coverage without |
|---|---|---|---|---|---|
| 2011 | 0.0578 | 0.0616 | -0.0038 (-0.0072 to +0.0011) | 96.1 % | 96.6 % |
| 2016 | 0.0664 | 0.0593 | +0.0071 (-0.0010 to +0.0151) | 76.8 % | 87.7 % |

**Pinball loss.** The change from adding the RCS is within noise at both origins. It is also not
consistent across learner sets: with the first set, adding it worsened the selected MLP at 2011
(0.1221 against 0.1147).

**Interval coverage.** The consistent effect is on coverage in the period whose missingness
pattern differs from training. At 2016, 90 % coverage falls from 87.7 % to
76.8 %. With the first learner set it falls from
89.2 % to 72.1 %.

**Conclusion.** The association between availability and outcome is clear, and using the
covariate breaks calibration when that association changes. Its effect on point accuracy was not
detectable here.

### Feature groups and coupling C1 (family evidence) {#sec:res-c1}

Each feature group was removed from the selected learner and the learner refitted with the same
settings and seeds. Fig. [-@fig:ablation] gives the paired launch-family bootstrap of the change
in test pinball loss; a positive change means the group helps. The intervals are nominal 95 %
intervals without correction for the eight comparisons.

| Group removed | Origin 2016 (TabM) | 95 % CI | Origin 2011 (Kumo-Tabular) | 95 % CI |
|---|---|---|---|---|
| Family evidence (C1) | +0.0056 | +0.0015 to +0.0101 | +0.0024 | -0.0058 to +0.0103 |
| Physics summaries | -0.0050 | -0.0086 to -0.0022 | +0.0022 | +0.0002 to +0.0039 |
| Design properties | +0.0071 | +0.0035 to +0.0121 | +0.0008 | -0.0001 to +0.0017 |
| Space weather | +0.0038 | -0.0057 to +0.0129 | +0.0009 | -0.0016 to +0.0038 |

**Family evidence (C1).** Siblings from the same launch and breakup that were catalogued, and
had decayed, before the prediction time improve TabM at 2016. At 2011 the point estimate is
positive but the interval includes zero.

**Physics summaries.** They help Kumo-Tabular at 2011, by a margin that would not survive a
multiplicity correction, and hurt TabM at 2016. A plausible reason, not tested here: in 2016–2021 the
physics expert's own error grew with solar cycle 25, and a learner that leans on its summaries
inherits that error.

**Design properties.** They help at 2016.

![**Feature-group ablations of the selected learner** (paired cluster bootstrap over launch families). In the first four rows a group is removed, so a positive change means the group helps. The last row instead *adds* the leaking 2026 radar cross-section, so there a positive change means the covariate makes the model worse.](artwork/Figure_9.png){#fig:ablation width=100%}

## Debris module: collision risk {#sec:res-cac}

**Setup.** The official Kelvins test set has 2,167 events, of which
150 are high-risk (Table [-@tbl:cac], Fig. [-@fig:cac]). Kumo-Tabular
had the lowest out-of-fold pinball loss among the recent learners and became the expert. The
gate then put almost no weight on the persistence expert (mean
0.0002), so the gated model is in effect Kumo-Tabular.

**Probabilistic forecast.** Kumo-Tabular's final-risk distribution has the lowest pinball loss.
Paired bootstrap over test events (post hoc; a negative difference favours Kumo-Tabular):
against TabICLv2 -0.039 (-0.053 to -0.025),
against XGBoost -0.202 (-0.233 to -0.171)
and against persistence -0.392 (-0.451 to -0.337).

| Model | Pinball loss | AUC-PR of P(high risk) | AUC-ROC of P(high risk) |
|---|---|---|---|
| Kumo-Tabular | 0.653 | 0.733 | 0.970 |
| TabICLv2 | 0.692 | 0.665 | 0.961 |
| Best tree baseline (XGBoost) | 0.855 | 0.606 | 0.945 |
| Persistence | 1.045 | 0.664 | 0.944 |

Its conformal 90 % intervals cover 94.2 % of all test events but only
48.0 % of the high-risk ones. With
66 high-risk training events, the group-conditional calibration is
coarse exactly where it matters.

**Official point metric.** It does not reward these gains.
- Every model's median forecast falls below the high-risk threshold for most high-risk events, so
  its $F_2$ is low: 129 of the 150 high-risk
  events receive a median below −6 from Kumo-Tabular, giving $F_2$ = 0.17. Most of
  those medians are not at the −30 floor but just below the threshold
  (41 lie between −10 and −6), so the floor is not the main cause
  (Fig. [-@fig:cac-outputs]).
- The latest-risk baseline scores $L$ = 0.694, the published
  value.
- Gating the latest-risk point with each model's calibrated P(high risk) brings the best models
  close to the baseline:
  - TabM: 0.698 (difference from the baseline, 95 % CI
    -0.005 to +0.019);
  - the gated model: 0.704
    (-0.055 to +0.071).
- No model beats the baseline.

This agrees with the challenge, where few teams beat the latest risk, and with a pre-registered
re-evaluation [@uriot2022spacecraft; @vazquez2026latest].

| Model | L (clipped) | F2 | MSE_HR | Pinball | AUC-PR P(high) | Brier P(high) | Cov. 90 % | Cov. 90 %, high risk |
|---|---|---|---|---|---|---|---|---|
| Latest risk (challenge baseline) | 0.694 | 0.739 | 0.513 | — | — | — | — | — |
| Persistence + residual climatology | ∞ | 0.000 | 1.752 | 1.045 | 0.664 | 0.0523 | 0.910 | 0.593 |
| XGBoost (baseline) | ∞ | 0.000 | 1.752 | 0.855 | 0.606 | 0.0552 | 0.917 | 0.433 |
| LightGBM (baseline) | ∞ | 0.000 | 1.752 | 0.861 | 0.541 | 0.0601 | 0.905 | 0.253 |
| CatBoost (baseline) | ∞ | 0.000 | 1.752 | 0.869 | 0.542 | 0.0564 | 0.912 | 0.467 |
| Quantile MLP (baseline) | ∞ | 0.000 | 1.752 | 1.035 | 0.434 | 0.0653 | 0.936 | 0.853 |
| TabM (2025) | ∞ | 0.000 | 1.752 | 0.885 | 0.512 | 0.0591 | 0.919 | 0.727 |
| TabICLv2 (2026) | 30.059 | 0.057 | 1.728 | 0.692 | 0.665 | 0.0489 | 0.918 | 0.327 |
| Kumo-Tabular (2026) | 9.892 | 0.168 | 1.662 | 0.653 | 0.733 | 0.0453 | 0.942 | 0.480 |
| Persistence + expert, constant weight | 9.898 | 0.168 | 1.663 | 0.653 | 0.733 | 0.0453 | 0.935 | 0.480 |
| **Reliability-gated (COSMOS)** | 9.896 | 0.168 | 1.662 | 0.653 | 0.733 | 0.0453 | 0.940 | 0.480 |

: **ESA Kelvins Collision Avoidance Challenge, official test set.** $L = \text{MSE}_\text{HR}/F_2$ of the median (predictions clipped at -6.001 as in the challenge); pinball loss of the final-risk quantiles; AUC-PR and Brier score of P(final risk >= -6); coverage of the conformal 90 % interval over all events and over the high-risk events. {#tbl:cac}

![**ESA Kelvins collision-risk test set.** (a) Pinball loss of the final-risk distribution; (b) the official score of the latest-risk point forecast and of the same forecast gated by each model's calibrated P(high risk).](artwork/Figure_10.png){#fig:cac width=100%}

![**What the collision-risk module outputs** (official test set). (a) Final risk against the latest risk available at least two days before closest approach, the input of the persistence expert. (b) Final risk against Kumo-Tabular's median forecast. 129 of the 150 high-risk events (orange) receive a median below the −6 threshold, 41 of them between −10 and −6 and 37 within one decade of the −30 floor, so the median forecast reaches $F_2$ = 0.17 on the official metric. Dashed lines mark the −6 threshold and the dotted line is equality. (c–e) Three high-risk test events drawn at random (fixed seed): the input messages, Kumo-Tabular's predictive quantiles (open circle: median) and the final risk, both drawn to the right of the input window.](artwork/Figure_11.png){#fig:cac-outputs width=100%}

## Couplings {#sec:res-couplings}

### C2: from per-fragment re-entry to breakup-cloud survival {#sec:res-c2}

**Setup.** C2 covers the test folds of both origins: 160
cloud-years in 27 breakup clouds, each with at least 30
catalogued fragments. For each cloud we predict the number of its fragments still in orbit one
to five years after its first catalogued fragment, by summing per-fragment survival
probabilities, and compare it with the catalogue count (Fig. [-@fig:c2]).

**Results** (mean absolute error as a fraction of the cloud, with 95 % intervals from a
bootstrap over clouds; Supplementary Table S17):

| Model | Raw reading | 95 % CI | Calibrated reading | 95 % CI |
|---|---|---|---|---|
| Physics expert alone | 0.129 | 0.078–0.191 | 0.145 | 0.111–0.187 |
| TabICLv2 | 0.153 | 0.119–0.188 | 0.095 | 0.068–0.123 |
| Gated model | 0.155 | 0.121–0.192 | 0.122 | 0.085–0.161 |

- **Raw reading.** The physics expert has the lowest point error. Its mean bias is
  +0.007, and 73.1 % of
  cloud-years are within 10 %. The learned experts under-predict survival on this reading, but
  the physics expert cannot be separated from the gated model (difference
  -0.025, 95 % CI
  -0.084 to
  +0.033).
- **Calibrated reading** (isotonic maps of $P(T \le a)$ fitted on tune-B fragments). Calibration
  significantly lowers the error of TabICLv2
  (+0.058,
  +0.034 to +0.082)
  and of the gated model (+0.033,
  +0.007 to
  +0.055). Calibrated TabICLv2 has the lowest
  point error, with 63.1 % of cloud-years within 10 %, but it
  cannot be separated from the raw physics expert
  (+0.035,
  -0.007 to
  +0.078). Calibration does not help the physics
  expert.

The calibrated reading and the cloud bootstrap were added after the raw reading had been seen
and criticised in an internal review; they are post hoc.

**What C2 shows.** With 27 clouds, the errors of the best
readings are about a tenth of a cloud's size, and the uncertainty is too large to rank the
experts. How survival is read off the predictive distribution matters for the learned experts.
In the twin, cloud forecasts are sums of the history-conditioned physics probabilities. T1 tests
the calibration of those probabilities on objects in orbit (Section [-@sec:res-t1]).

![**C2: predicted against observed fraction of each breakup cloud's fragments still in orbit**, one point per cloud-year, (a) raw and (b) calibrated survival read-off.](artwork/Figure_12.png){#fig:c2 width=100%}

### C3: operational status as health evidence for the re-entry regime {#sec:res-c3}

**Cohort.** On 2026-08-16, 13,794 Earth-orbiting payloads had perigee below 600 km. In the
following 72 days, 182 of them decayed; the decays came from
86 of their 1,070 launches.

**Physics by status.** The drag physics over-predicts re-entry for payloads reported
operational:

| Group | n | Decays | Mean predicted P | Observed rate |
|---|---|---|---|---|
| Operational | 13,478 | 164 | 0.039 | 0.012 |
| Non-operational | 161 | 17 | 0.076 | 0.106 |

- **Operational payloads.** The physics probabilities nevertheless rank them well (AUC-ROC
  0.959), consistent with operators letting some low
  payloads decay while keeping others in orbit.
- **Non-operational payloads.** The physics expert under-predicts somewhat, with AUC-ROC
  0.935.

**As-of-date policies** (information at 2026-08-16 only; Fig. [-@fig:c3]):

| Policy | Brier score | ECE |
|---|---|---|
| Drag physics for every payload | 0.0179 | 0.0267 |
| Previous-window base rate (0.0127) for every payload | 0.0130 | 0.0005 |
| Status-aware twin (base rate for operational payloads, physics otherwise) | 0.0125 | 0.0008 |

The status-aware twin has the lowest Brier score, but neither difference is significant once
launches are resampled:

| Status-aware twin minus | Brier difference | 95 % CI |
|---|---|---|
| Physics for every payload | -0.0054 | -0.0120 to 0.0006 |
| Base rate for every payload | -0.0005 | -0.0012 to 0.0001 |

With 182 decays in one window, the test can show the direction of the effect but
cannot establish it. The window is also not fresh: an earlier version of the project had fitted
and tested re-entry classifiers on it (Section [-@sec:protocol]).

**In-window diagnostic (not a forecast).** We also fitted Platt recalibrations on the follow-up
window itself, cross-fitted by launch. Recalibrating per status rather than for all payloads
changes the Brier score by
-0.0004
(95 % CI
-0.0010
to
0.0001).
Even with look-ahead, then, status adds little once the physics probabilities are rescaled. The
regime split is what keeps the twin from issuing drag forecasts for payloads whose operators
control their orbits.

![**C3: mean predicted probability of re-entry within 72 days** by operational status, against the observed rate (dashed). The in-window Platt map is fitted on the follow-up outcomes and is shown only as a reference.](artwork/Figure_13.png){#fig:c3 width=100%}

### T1: the twin's re-entry forecasts for objects in orbit {#sec:res-t1}

**Cohort.** As of 2026-08-16, the twin held 13,186 fragments, rocket stages and
components with perigee below 2000 km. Within 72 days, 64
of them decayed, 29 within 30 days
(Fig. [-@fig:t1]).

**Results at 72 days** (expected re-entries, and their ratio to the observed number):

| Variant | Expected | Exp./obs. | 95 % CI | Brier score |
|---|---|---|---|---|
| History-conditioned physics | 87 | 1.37 | 1.20–1.63 | 0.00217 |
| Physics with class priors | 233 | 3.64 | 2.90–4.68 | 0.00625 |
| Lifetime pipeline (gated TabM) | 610 | 9.5 | 6.9–13.0 | 0.01344 |

- **Lifetime pipeline.** Applied to current orbits, it over-predicts re-entry about ninefold,
  although it was the validated model at canonical-orbit epochs. Objects that have stayed in
  orbit for years are the survivors, with low area-to-mass ratios, and a model fitted at
  canonical-orbit epochs does not know this. Its TabM expert is refitted on the GPU, whose
  training is not bit-reproducible: an earlier run of the same configuration gave a ratio of 8.9.
- **Physics with class priors.** It over-predicts about fourfold.
- **History-conditioned physics.** Conditioning the area-to-mass ratio on the object's catalogue
  history cuts the excess to a factor of about 1.4. This is still significantly above one.

**Brier differences** (launch-cluster bootstrap):

| History-conditioned physics minus | Brier difference | 95 % CI |
|---|---|---|
| Physics with class priors | -0.00408 | -0.00598 to -0.00266 |
| Lifetime pipeline | -0.01127 | -0.01812 to -0.00632 |

**Ranking.** Rank metrics use the raw probabilities. By AUC-PR the history-conditioned variant
ranks best at 72 days: 0.737, against
0.601 (class priors) and
0.616 (lifetime pipeline). By AUC-ROC the three
are within 0.01 of each other (0.991,
0.988 and
0.995), so all three separate decaying objects
from the rest about equally well. What differs is mainly calibration. At 30 days the AUC-PR
order is the same, and by AUC-ROC the history-conditioned variant is slightly lower
(0.982, against
0.999 and
0.998).

**Payloads.**
- *Operational payloads* (operator-controlled regime). The C3 policy expected
  170 decays against
  164 observed. Drag physics would
  have expected 498. The
  base-rate policy cannot rank payloads, though: its AUC-ROC is
  0.58, against
  0.96 for drag physics.
- *Non-operational payloads.* The physics expert expected
  11 against
  17 observed, the same under-prediction as
  in C3.

**The whole twin.** Over all 30,826 objects, the twin's per-regime choice expected
272 re-entries against
246 observed (Brier 0.0065).

**Caveats.**
- T1 was specified before it was run, but it is not a blind test. An earlier version of the
  project had fitted and tested re-entry classifiers on the same window, and the window's
  aggregate decay count was seen while the lifetime pipeline's bias on current orbits was being
  diagnosed, before the history-conditioned variant was designed.
- A defect in the payload prior was corrected after T1 had first been scored (Section
  [-@sec:protocol]). Payload numbers changed; those of fragments, rocket stages and components
  did not.

![**T1: as-of-date test of the twin's re-entry forecasts** for fragments, rocket stages and components in orbit on 2026-08-16. (a) Reliability at 72 days, in six fixed probability bins; marker area grows with the number of objects in the bin, bins with fewer than ten objects are dropped, and both axes are symmetric-log so that bins with no observed decay sit on the zero line. (b) Expected against observed re-entries.](artwork/Figure_14.png){#fig:t1 width=100%}

## Health module {#sec:res-health-wrap}

### NASA SMAP/MSL telemetry and the companion ESA-ADB study {#sec:res-health}

**Setup.** On the 81 channels (68 SMAP and
36 MSL anomaly sequences), detectors are compared at a level and
persistence fixed in advance.

**Ranking** (Fig. [-@fig:telemetry-benchmark]a). The published telemanom scores [@hundman2018] rank anomalous steps best. Their
per-channel mean AUC-PR is 0.576 (SMAP) and
0.473 (MSL). Used zero-shot, the Chronos-2 foundation
model reaches 0.343 and
0.367. That is level with the ridge, PCA and
nearest-neighbour baselines, and above the random-score floor of
0.129 and 0.141.

**Alarms** (Fig. [-@fig:telemetry-benchmark]b). On the ESA-ADB corrected event-wise $F_{0.5}$, which charges alarm time on nominal
data, the highest detector scores are PCA on SMAP (0.249) and the
nearest-neighbour detector on MSL (0.334). Chronos-2's residual
alarms are too frequent on MSL (508 false-positive segments),
giving a corrected $F_{0.5}$ of 0.035 there, against
0.170 on SMAP. That contrast is not a property of the spacecraft:
one MSL channel contributes 414 of those segments, and three contribute 95 % of them. They are
channels whose training series is constant, so every calibration score sits at the score floor and
any non-zero test residual raises an alarm.

**Why these alarm scores are weak evidence.**
- An always-on alarm reaches an uncorrected event $F_1$ of
  0.548 (SMAP) and 0.537 (MSL),
  higher than every detector. Its corrected $F_{0.5}$ is 0.
- 93.7 % of labelled steps lie in the second half of
  their channel's test series. A label-free reference that keeps each detector's amount of alarm
  time but places it as one block at the end of the series scores higher on the corrected
  $F_{0.5}$ than the detector itself in seven of the eight detector–spacecraft pairs; for
  example, 0.306 against 0.249 for
  PCA on SMAP. The exception is the nearest-neighbour detector on MSL
  (0.334 against 0.281).

We therefore read skill against a null that keeps each detector's alarm positions but
reassigns them across channels. Every fitted detector exceeds it on both spacecraft. For
example, ridge's event $F_1$ is 0.295 against
0.136 under the null (permutation $p$ =
0.005). This shows channel-specific detection, but a
modest one: on this benchmark, alarms that are merely late are rewarded about as much.
Fig. [-@fig:smap-outputs] shows the scores and alarms on one typical channel of each spacecraft.

**Companion ESA-ADB study.** The health module's alarm layer (conformal p-values,
Benjamini–Hochberg aggregation across channels and persistence) was developed and tested in a separate,
pre-registered study on ESA-ADB [@kotowski2026esaadb] by the same authors. On Mission 1, fitting
an AR(8) forecaster and selecting the alarm settings on the whole training half, rather than on a
short calibration window, raised the corrected event-wise $F_{0.5}$ from
0.074 to 0.428
(difference +0.353, 95 % CI
+0.206 to +0.488), at about
1 false alarm per year. On Mission 2 the same
contrast reversed (-0.089,
-0.143 to -0.049). Validation analyses of
that study attributed the Mission 1 gain to the labels used to select the alarm settings, not to
the fitted scores.

**Summary.** The recent forecasting foundation model does not improve on simple detectors on
SMAP/MSL, and the alarm protocol that the twin's health module uses helped on one ESA-ADB
mission and hurt on the other. The health module is therefore the least validated of the three.

![**Telemetry benchmark on SMAP/MSL.** (a) Per-channel mean AUC-PR of anomalous steps. (b) Corrected event-wise F0.5 of each detector's alarms (filled) and of a label-free reference that keeps the detector's amount of alarm time but places it as one block at the end of each series (open).](artwork/Figure_15.png){#fig:telemetry-benchmark width=100%}

![**What the health module outputs** on one channel per spacecraft, chosen by rule as the median channel by Chronos-2 per-channel AUC-PR, taking the lower of the two middle channels where that number is even (SMAP E-6, MSL C-2). Top: test series and labelled anomalies. Middle: conformal p-values of PCA reconstruction and Chronos-2 against the channel's own training scores, with the alarm level. Bottom: labelled sequences and each detector's causal alarms. The training series of C-2 is constant, so every Chronos-2 calibration score sits at the score floor and any non-zero test residual alarms; Chronos-2 is in alarm from step 210 on, the most extreme Chronos-2 alarm behaviour of the 27 MSL channels. The recomputed alarms reproduce the benchmark's event counts for both channels.](artwork/Figure_16.png){#fig:smap-outputs width=100%}

## The dated twin forecast of 2026-10-27 {#sec:res-twin}

**Scope.** As of 2026-10-27 the twin holds 31,396 objects in orbit with perigee below
2000 km:
- 13,787 in the natural-decay regime;
- 16,386 operator-controlled payloads;
- 1,223 payloads without a reported status.

**History conditioning** (Fig. [-@fig:twin-state] shows three examples). Among fragments, rocket stages and components:
- for 12,606, the catalogue history determined the area-to-mass
  posterior (effective sample size of at least 4);
- for 410, survival alone was used;
- for 56, no prior sample survived, and the longest
  survivors were used;
- 166 objects without a usable canonical orbit kept the class prior.

**Expected re-entries in the next 30 days** (natural-decay regime): 24.5.
For fragments, rocket stages and components, the three variants differ by an order of
magnitude:

| Variant | Expected re-entries, 30 days |
|---|---|
| History-conditioned physics | 21.6 |
| Physics with class priors | 76.3 |
| Gated lifetime pipeline, fitted at canonical-orbit epochs | 229.1 |

T1 indicates which of these to believe (Section [-@sec:res-t1]). Fig. [-@fig:twin-forecast] maps the forecast and
lists the natural-decay objects with the highest 30-day probability.

**Clouds and verification.** The forecast includes breakup-cloud forecasts for the
61 clouds with at least 30 fragments in orbit. Its record file is fixed by its
SHA-256 hash, `19a1fede1fd4871f380ff802f285d2ad03732f6cd3179790e67336c7d202a6b5`, so that later catalogue snapshots can score it without the
possibility of revision. An earlier forecast of the same date (hash beginning `fad1c619`)
contained the payload-prior defect described in Section [-@sec:protocol]. It was replaced
before any later snapshot existed, and it is archived with the code.

![**Twin state estimation**, for three objects drawn at random (one per class; fixed seed) among natural-decay objects whose catalogue history determined the posterior (effective sample size at least 4) and whose history-conditioned median forecast is within five years. (a) Prior area-to-mass samples (grey: the object's design value where GCAT has one, otherwise a draw from its class prior) and the posterior after conditioning on the change from the canonical to the current orbit (orange). (b) The re-entry forecasts the same samples produce. The recomputed posteriors reproduce the stored forecasts.](artwork/Figure_17.png){#fig:twin-state width=100%}

![**The dated twin forecast of 2026-10-27.** (a) Current orbits of the objects in the twin, by regime; the regimes overlap, and the largest is drawn first. (b) The twin's probability of re-entry within 30 days against current perigee, natural-decay regime. (c) The 15 natural-decay objects with the highest probability, ranked by that probability and then by the median forecast: median and 90 % Monte-Carlo interval of the forecast re-entry time (history-conditioned physics for fragments, rocket stages and components; class-prior physics for non-operational payloads). Two rows have a degenerate interval at exactly 10 days: their perigee is already within 1 km of the 120 km re-entry threshold, where the integrator's maximum step of 10 days sets the resolution.](artwork/Figure_18.png){#fig:twin-forecast width=100%}

### Computational cost {#sec:res-cost}

Times are measured on the origin-2016 run (25,334 training objects,
2,993 test objects; Fig. [-@fig:cost]). Hardware:
- CPU learners, TabICLv2 and the physics expert: a two-socket Intel Xeon 6980P node (512 hardware
  threads).
- TabM and Kumo-Tabular: an Intel Arc Pro B60 GPU.

| Model | Fit and predict per seed (s) | One call over the test cohort (s) |
|---|---|---|
| Physics expert (32 Monte-Carlo samples) | 128 | 34 |
| XGBoost | 26 | 0.029 |
| TabM | 60 | 0.063 |
| Kumo-Tabular | 94 | 95 |
| TabICLv2 | 915 | 617 |

**What is timed.** The first column is the wall-clock of the fit-and-predict loop divided by the
five seeds: it contains four refits (the seed-0 fit happens during tuning and is not timed) and
the predictions of all five seeds on the tuning and test cohorts. It is therefore an upper bound
on fitting alone, and for the in-context models it is dominated by prediction. The second column
is one call predicting the test cohort.

**In-context foundation models.** Their prediction cost includes encoding the training context,
up to 16,000 rows for Kumo-Tabular and 25,334 for TabICLv2 on CPU. Predicting
the same cohort takes them three to five orders of magnitude longer than a tree ensemble
(Kumo-Tabular 95 s and TabICLv2
617 s, against 0.029 s for
XGBoost and 0.005 s for CatBoost). Because the context encoding
is a fixed cost per call rather than per object, these times cannot be rescaled to a different
number of objects, and we do not extrapolate them to the full catalogue; both models predicted the
whole test cohort in one call.

**The physics features are part of every learner's cost.** All learners take five summaries of the
physics expert's Monte-Carlo distribution as inputs, so a deployed pipeline also pays that run:
34 s for the 2,993 test objects, which
dominates the tree ensembles' own prediction time by three orders of magnitude. The second column
above times the learner alone.

![**Computational cost**, origin 2016, on log scales. (a) Wall-clock of the fit-and-predict loop per seed: four refits (the seed-0 fit happens during tuning and is not timed) plus the predictions of all five seeds on the tuning and test cohorts, so it bounds fitting alone from above and for the in-context models is dominated by prediction; for the physics expert, its Monte-Carlo run over the training set. (b) One call predicting the test cohort, which for the in-context models includes encoding the training context, a fixed cost per call rather than per object.](artwork/Figure_19.png){#fig:cost width=100%}



# Discussion {#sec:discussion}

**Which re-entry expert is best depends on the test period.**
- For 2016–2021, two in-context tabular foundation models released in 2026, TabICLv2 and
  Kumo-Tabular, were significantly more accurate than the physics expert, the gated model, the
  selected TabM and the best tree ensemble. They also had the best concordance and near-nominal
  raw coverage.
- For 2011–2015, the physics expert alone had the lowest point estimate, but it could not be
  separated from the foundation models or from the gated model. It had the lowest loss for
  fragments in that period. The one significant difference among them is that Kumo-Tabular, the
  learner selected there, is worse than the gated model that fuses it.

TabICLv2 was thus best or tied in both periods. One change between the periods is measured:
replaying the realised F10.7 lowers the physics expert's error much more in the rising phase of
cycle 25 than in cycle 24. Other changes in the test cohorts were not isolated here. Two test
periods, of which several comparisons were added after the first results were seen, are not
enough to say whether the foundation models will keep their lead.

**Selecting a learner on a tuning cohort five to seven years older than the test cohort did not
reliably pick the model that generalised best.**
- At 2016, TabM led on tune-A by a wide margin and was significantly worse than both foundation
  models on test.
- At 2011, the small tune-A set (248 objects) chose Kumo-Tabular, which
  is tied with TabICLv2 on test.
- The first learner set, recorded as a deviation, selected its MLP at both origins, and the MLP
  and TabM failed on 2011 fragments.

The temporal protocol is what exposes this. A random split would have hidden it.

**The reliability gate helped when the physics expert was reliable, and it did not beat the best
single expert.**
- At 2011 it significantly improved every learner to which it gave non-negligible weight
  (post-hoc sensitivity analysis). The reported 2011 gated model depends on the change of the
  candidate pool: with the first learner set, its gated model was much worse.
- At 2016, when solar activity degraded the physics expert, it improved none and cost a little.
  The cost was not statistically significant.

The gate is fitted on out-of-fold predictions of the training period, so it cannot anticipate
that a learner will degrade on a future cohort. Its descriptors (area-to-mass source, the physics
expert's spread, orbit, class) describe the physics expert's reliability, not the learner's
reliability under shift. A gate that also sees descriptors of covariate shift is a natural next
step, but it would have to be validated on a third test period.

**The radar-cross-section leak is a warning for catalogue-based learning.**
- A covariate that "is a size measurement" encodes the outcome through its availability
  (adjusted odds ratio 3.6, cluster-robust
  95 % CI 2.1–6.2).
- Adding it changed the test pinball loss by amounts within noise at both origins, so point
  accuracy would not have revealed the leak.
- At 2016, where its missingness pattern differs from training, it broke interval coverage.

Any study that joins catalogue snapshots taken long after the prediction time should test the
missingness of each covariate against the outcome. Our code includes such a check.

**What the couplings show.**
- **Family evidence (C1)** improved the learner in one of two periods.
- **Cloud survival (C2)** is accurate to about a tenth of a cloud's size for the best readings,
  but with 27 clouds the experts cannot be ranked.
- **Operational status (C3)** marks a regime in which drag physics over-predicts re-entry by a
  factor of about three. A status-aware policy lowered the Brier score in one as-of-date window,
  but not significantly once outcomes are clustered by launch.
- **The twin's current-state forecasts (T1)** are where the coupling of history and physics
  matters most. Conditioning on the catalogue history reduced the over-prediction of 72-day
  re-entries from 9.5-fold (lifetime
  pipeline) and 3.6-fold (class
  priors) to 1.37-fold, in one
  window that had been analysed earlier in the project.

C1, C2 and C3 are small or uncertain effects. T1 is a large effect, but it comes from one
window that had been analysed before. Together they are what a digital twin that only joins
separate tools cannot provide: a measured account of which links between the modules carry
information.

## Limitations and threats to validity {#sec:limits}

**One orbit per object.**
- Orbital-element histories at catalogue scale are not legitimately public: Space-Track
  requires an account and forbids redistribution.
- The re-entry module therefore predicts from the canonical orbit, and the twin uses two orbits:
  canonical and current.
- Operational tools that fit the ballistic coefficient to weeks of TLE data
  [@pardini2008impact; @gondelach2017ballistic] use far more information near re-entry. Our
  months-to-years forecasts are not comparable with their final-days accuracy.

**Validation of current-state forecasts.**
- The lifetime study validates forecasts at canonical-orbit epochs.
- Forecasts for objects already in orbit are validated only by T1. T1 covers one 72-day window,
  and it is not blind: an earlier version of the project fitted and tested re-entry classifiers
  on the same window, and its aggregate decay count was seen before the history-conditioned
  variant was designed.
- The hashed forecast of 2026-10-27 is a blind test that a later catalogue snapshot will
  score.

**Priors built from catalogue values.**
- The class prior of characteristic length comes from training objects that received a
  radar-cross-section value, so it is biased towards long-lived objects.
- Among components, the availability of design data is associated with early decay (adjusted
  odds ratio 3.4). We
  treat it as known at launch, because GCAT design values come from launch documentation. The
  `no_design` ablation bounds its influence.

**Resolution of the decay integrator.** A step may not drop the perigee by more than 2 % of its
height above 120 km, with a floor of 1 km, and may not exceed 10 days. For an object whose perigee
is already within 1 km of the 120 km threshold, the first step can therefore be a full 10 days, and
the forecast is reported at the end of that step. Two of the 13,787
natural-decay objects of the dated forecast have a degenerate 90 % interval at exactly 10 days for
this reason. The affected objects are within days of re-entry in any case, and no reported
comparison depends on them.

**Physics simplifications.**
- The density is a global NRLMSIS 2.1 average without latitude, local-time or seasonal
  variation.
- The atmosphere is exponential about perigee within an orbit.
- Lunisolar and J2 effects on perigee are not modelled. They matter for transfer-orbit stages.
- Solar activity is held at its value at the prediction time. The oracle diagnostic shows how
  much of the error this causes (Section [-@sec:results]).

**Small calibration sets at the 2011 origin.** The 2004–2005 tuning cohort is small, so model
selection and calibration at that origin are noisy.

**Test-set reuse and post-hoc analyses.**
- The re-entry and collision-risk test sets were scored more than once during development:
  before an internal review, with a first learner set, and in smoke tests of the new learners.
- The pool of eligible learners was changed after the first set's test results were known.
- The head-to-head comparisons, the gate applied to every learner and the calibrated C2 reading
  were added after results were seen. Their intervals are nominal and uncorrected for
  multiplicity.

All of this is recorded in Section [-@sec:protocol] and in the released logs. It weakens the
confirmatory value of the comparisons, and a fresh test period is needed to confirm them.

**Bit-level reproducibility.** Seeded training of TabM on the GPU is not bit-reproducible; on CPU
it is. The re-entry study reuses cached predictions, so its numbers are fixed, but a refit gives
slightly different results: two runs of the T1 lifetime-pipeline variant gave expected/observed
ratios of 8.9 and 9.5. Chronos-2
inference on the GPU varies at the same level; repeating the SMAP/MSL run moved its per-channel
AUC-PR by at most $2 \times 10^{-5}$ and left every reported value unchanged at three decimals.

**Health evidence is a proxy.**
- Catalogue operational status is the only health evidence available for every payload. It is
  not a measurement of manoeuvre capability.
- No public dataset links telemetry, conjunction messages and catalogue objects, so the
  health-to-debris coupling (C4) is a demonstration only.

**Event metrics for telemetry.** Event-wise metrics can be satisfied by trivial alarm rules, so
they are reported next to trivial references and with the corrected precision of ESA-ADB. Even
so, a reference that only places alarms late in each series beats most detectors on SMAP/MSL.
The alarm protocol of the health module helped on one ESA-ADB mission and hurt on the other, so
the health module is the least validated part of the twin.

# Conclusions {#sec:conclusions}

COSMOS-AI joins satellite health monitoring, orbital debris analysis and atmospheric re-entry
prediction in one digital twin. Each module is validated on public data, and the couplings are
tested wherever a real test exists. Five findings stand out.

1. **Re-entry prediction across solar-cycle phases.**
   - Two 2026 tabular foundation models were significantly the most accurate in 2016–2021.
   - In 2011–2015 TabICLv2 was statistically tied with the physics expert and the gated model;
     Kumo-Tabular was significantly worse than the gated model that fuses it.
   - Selection on an older tuning cohort chose a significantly worse learner in 2016.

   The reliability gate improved the learners it weighted at the 2011 origin and was neutral to
   slightly harmful at 2016. It did not beat the best single expert.
2. **Catalogue covariates can carry the outcome in their availability.** Whether an object has a
   radar cross-section is strongly associated with early decay. Using it broke interval coverage
   in the period whose missingness pattern differed from training, while its effect on point
   accuracy was within noise.
3. **For objects already in orbit, the twin's state estimate matters more than the choice of
   learner.** Conditioning each object's area-to-mass ratio on its catalogue history reduced the
   over-prediction of 72-day re-entries from
   9.5-fold (the lifetime
   pipeline) and 3.6-fold
   (physics with class priors) to
   1.4-fold.
4. **Operational status separates payloads whose re-entry drag physics cannot predict.** Using
   it lowered the Brier score in one 72-day window, but not significantly.
5. **Conjunction and telemetry: the newest models did not beat the established baselines on the
   operational metrics.**
   - On the ESA collision-risk benchmark, a 2026 tabular foundation model gave significantly the
     best probabilistic forecasts. Neither it nor any other model beat the latest-risk forecast
     on the official point metric.
   - A zero-shot time-series foundation model did not beat simple detectors on SMAP/MSL.

**Limits.**
- The twin rests on single catalogue orbits rather than orbital histories.
- Findings 3 and 4 come from one as-of-date window that had been analysed earlier in the
  project, so they are not blind tests.
- The test sets were scored more than once during development, and several analyses were added
  after results were seen (Section [-@sec:protocol]).

**Verification.** The hashed forecast of 2026-10-27 is the first blind test of the twin and
will be scored against later catalogue snapshots. The code and derived data needed to repeat
every number are released with the paper.



# Declarations {.unnumbered}

**CRediT authorship contribution statement.** [Author 1]: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization. [Author 2]: Supervision, Writing – review & editing. [To be completed by the authors.]

**Data availability.** All data used are public. GCAT [@mcdowell2020gcat] is available under CC-BY-4.0; CelesTrak catalogue and space-weather files [@celestrak2026satcat] under CelesTrak's usage policy; the ESA Kelvins Collision Avoidance Challenge data [@uriot2021cacdataset] under CC-BY-4.0; ESA-ADB [@esaadb_dataset] (companion study) and NASA SMAP/MSL [@hundman2018] under the terms stated by their authors. The derived cohort table, the NRLMSIS density table, all predictions and result files, and the dated twin forecast will be deposited in Zenodo with a DOI before publication [DOI to be added]. No Space-Track data are used or redistributed. Pretrained model weights are obtained from their publishers under their own licences:
- TabICLv2: BSD-3-Clause;
- Kumo-Tabular: OpenMDW-1.1;
- Chronos-2: Apache-2.0;
- NRLMSIS 2.1, via pymsis.

**Code availability.** The code (data fetchers with checksummed manifests, the re-entry, conjunction and telemetry modules, the twin engine, and the scripts that produce every table and figure from the result files) is available on GitHub [@cosmos_software]; an archived release with a DOI will be created before publication.

**Funding.** [To be completed by the authors.]

**Declaration of competing interest.** [To be completed by the authors.]

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work the author(s) used Claude (Anthropic) to assist with writing analysis and plotting code and with drafting and editing the text. After using this tool, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article. All figures were rendered with Matplotlib from the study's result files; no figure was generated or altered by a generative AI model. [The authors must review this statement and adjust it to their actual use.]

**What was not validated.** The C4 coupling (health × debris → collision response) is a demonstration: no public dataset pairs conjunction messages with the health of the same spacecraft, and operational status is a proxy for manoeuvre capability, not a measurement of it. The twin forecast as of 2026-10-27 is a prospective forecast; it will be verified against later catalogue snapshots. The as-of-date test T1 covers one 72-day window that had been analysed earlier in the project, so it is not a blind test (Section [-@sec:protocol]).

# References {.unnumbered}

::: {#refs}
:::
