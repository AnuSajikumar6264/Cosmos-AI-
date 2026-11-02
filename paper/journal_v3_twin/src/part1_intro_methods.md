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

**Catalogue (re-entry and debris modules).** GCAT [@mcdowell2020gcat] (release 1.8.8, CC-BY-4.0, retrieved 2026-10-27) lists for each catalogued object one *canonical orbit*, "representative of the spacecraft's early operations, after initial positioning", with its epoch; design mass and dimensions; the parent object (for fragments, the object that broke up); and the end of its current phase with a status code that separates natural decay from active de-orbit, landing or docking. The CelesTrak SATCAT [@celestrak2026satcat] adds operational status. CelesTrak's space-weather file provides F10.7 and Ap since 1957; only observed rows of days that ended before the prediction time are used. The SATCAT's radar cross-section (RCS) is *not* used as a predictor, because its availability encodes the outcome. Values were mostly never published for objects that decayed quickly, and none are published for objects catalogued after about 2018. In the origin-2016 training cohort, {{leak.o2016.train.p_decay30_given_missing:pct}} of objects without an RCS decayed within 30 days, against {{leak.o2016.train.p_decay30_given_present:pct}} of those with one. Adjusted for orbit, class and decade, the odds ratio is {{leak.o2016.train.adjusted_or_rcs_missing.odds_ratio:.2f}} (95 % CI {{leak.o2016.train.adjusted_or_rcs_missing.ci95.0:.2f}}–{{leak.o2016.train.adjusted_or_rcs_missing.ci95.1:.2f}}). The effect is quantified in Section [-@sec:results]. The cohort holds {{data.cohort.rows_out:,}} objects whose canonical orbit has perigee below 2000 km, of which {{data.cohort.events:,}} decayed naturally; the exclusions are counted in the build log. Fig. [-@fig:data-reentry] shows the cohort's orbits, its distribution over time with the cohorts of the two forecast origins, the observed lifetimes and the space-weather record.

![**The re-entry dataset.** (a) GCAT canonical orbits of the cohort by object class; payloads are outside the primary population. The classes overlap heavily at these densities, so the panel shows where each class lies rather than its density; the largest class is drawn first. (b) Objects per year of their canonical-orbit epoch, with the training, tuning and test cohorts of the two forecast origins. (c) Time from the canonical-orbit epoch to natural decay, or to censoring for objects still in orbit at the snapshot or whose phase ended without natural decay (a controlled re-entry, a landing, a docking or another non-decay status code); the learning target is capped at five years. (d) F10.7 solar flux from the CelesTrak space-weather file; its 81-day mean is the model input, and the bars mark the two test periods.](artwork/Figure_1.png){#fig:data-reentry width=100%}

**Conjunctions (debris module).** The Kelvins Collision Avoidance Challenge data [@uriot2021cacdataset] contain anonymised CDMs received by ESA in 2015–2019. Following the challenge, inputs are the CDMs released at least two days before closest approach and the target is the risk (log10 collision probability) of the final CDM. Training events are filtered to mirror the test inputs, leaving {{cac.data.train_events_kept:,}} training events ({{cac.data.train_high_risk}} high-risk, i.e. final risk ≥ −6) and the official {{cac.data.test_events:,}} test events ({{cac.data.test_high_risk}} high-risk). Fig. [-@fig:data-debris] shows the breakup clouds of the catalogue and the conjunction messages.

![**Debris data.** (a) Catalogued fragments in orbit over time for the four largest breakup clouds of the cohort (fragments with a GCAT canonical orbit with perigee below 2000 km). (b) Gabbard diagram of the largest cloud, from the canonical orbits of its fragments. (c) Risk sequences of the conjunction data messages of 75 training events of the ESA Kelvins challenge: 60 drawn at random from the events whose final risk is below −6 and 15 from the {{cac.data.train_high_risk}} high-risk events (fixed seed), so high-risk events are deliberately oversampled. The model inputs are the messages received at least two days before closest approach. (d) Final risk of the training events: most lie at the floor of −30, and few are high-risk.](artwork/Figure_2.png){#fig:data-debris width=100%}

**Telemetry (health module).** NASA SMAP/MSL [@hundman2018]: {{smap.config.channels}} channels with {{smap.SMAP.ar_ridge.events:.0f}} SMAP and {{smap.MSL.ar_ridge.events:.0f}} MSL labelled anomaly sequences (the label file lists channel P-2 twice; the two rows are merged). The alarm layer of the health module was developed and tested on ESA-ADB Missions 1 and 2 [@kotowski2026esaadb] in a companion pre-registered study by the same authors, whose results are summarised in Section [-@sec:res-health]. Fig. [-@fig:data-telemetry] shows examples of both.

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
