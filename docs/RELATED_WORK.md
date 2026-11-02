# Related work: orbital debris analysis and atmospheric re-entry risk prediction

Compiled 2026-10-27 for the Q1 re-entry rebuild (branch `q1-reentry-rebuild`).

- Bibliography: `docs/references_reentry.bib`. Citation keys are written `[@key]`.
- Candidate research gaps and the evidence for each: `docs/LIT_NOTES_GAP.md`.

## 0. How this review was built, and how far to trust each entry

**Sources searched (2026-10-27):**
- OpenAlex: boolean title/abstract queries, until the daily quota ran out in the afternoon.
- Crossref: bibliographic queries.
- Semantic Scholar: paper records and citation contexts.
- The arXiv API, which was heavily rate-limited.
- The ESA Space Debris Office proceedings database: title and full-text search over SDC1–9, the NEO and Debris Detection conferences and the Re-entry Workshops.
- Zenodo and DataCite.
- Publisher pages and official websites: Kelvins, CelesTrak, Space-Track documentation, GCAT/planet4589, the ESA re-entry portal, and the Aerospace Corporation via the Wayback Machine.

**How entries were verified:**
- Every one of the 157 entries in the `.bib` was checked against an authoritative record, and every entry is cited in this file or in `LIT_NOTES_GAP.md`.
- All 126 DOIs were re-resolved automatically against Crossref or DataCite. All titles matched. The only year differences are online-first versus issue year, and each is stated in the entry's `note`.
- The 31 entries without a DOI were checked against the proceedings page or PDF, the arXiv abstract page, or the official website.
- The `.bib` parses cleanly with pandoc 3.9 (biblatex reader), and both documents render with `--citeproc` with no missing keys.

**Evidence labels used below:**

| Label | Meaning |
|---|---|
| **FT** | Full text read (publisher PDF, open conference PDF or arXiv). Numbers are quoted from it. |
| **ABS** | Abstract only. Numbers come from the abstract. |
| **META** | Metadata only. Closed access; ScienceDirect, MDPI, Wiley, ADS and Aerospace.org return bot walls or 403, and these were not circumvented. Nothing beyond the title is claimed. |
| **PRE** | Not peer reviewed: an arXiv, Zenodo or company report. |

**Scope note:**
- "Re-entry" here means natural decay and uncontrolled re-entry of catalogued objects.
- Hypersonic re-entry vehicles, capsule aerothermodynamics and "re-entry" in other fields (mining, clinical, social science) came up in the searches. They were screened out.

---

## 1. Re-entry and orbital-lifetime prediction

### 1.1 Physics-based and semi-analytic methods

**Analytic and semi-analytic theory.**
- King-Hele's orbit-contraction theory remains the analytic reference for drag-driven decay [@kinghele1987satellite].
- Frey, Colombo and Lemmens extend it to non-circular orbits and to density profiles with variable scale height (the superimposed King-Hele method) [@frey2019extension] (ABS).
  - They report accuracy "comparable to numerical Gauss-Legendre quadrature" at very low computational load.
  - The method works on averaged elements. It is the natural "physics prior" for a learned residual.

**Long-term propagators and tools.**
- CNES's **STELA** is the reference semi-analytic tool for French Space Operations Act compliance [@lefevre2014compliance] (META).
  - Deleflie et al. fit STELA's mean-element model to TLE time series [@deleflie2019reentry] (FT).
  - The fit estimates the ballistic coefficient and area-to-mass ratio jointly with the initial mean state.
  - They note that the a-priori values deduced from individual TLEs are "frequently far" from the best-fitting ones (Tiangong-1 example).
- ESA's **DRAMA** suite contains **OSCAR**, which computes orbital lifetime after disposal, and **SARA**, which computes on-ground re-entry risk [@braun2020drama] (ABS). The paper reports more than 1,000 users.

**Catalogue-scale validation of a lifetime tool.**
- Operti et al. compare OSCAR lifetimes with observed lifetimes [@operti2017assessing] (FT). Data: TLEs plus DISCOS physical properties for 575 rocket bodies, 333 payloads and 73 small satellites.
- The metric is the whole-lifetime relative error (computed − observed) / observed:
  - LEO rocket bodies: mean −4.06%, median −8.87%, standard deviation 43.10%.
  - LEO small satellites: mean +13.36%, median +10.11%, standard deviation 36.57%.
- They conclude that "the bias in the orbital lifetime estimate cannot be corrected by a simple global constant correction coefficient".
- This is the closest physics-based, catalogue-scale lifetime error distribution to the project's task, and a natural baseline.

**SGP4 and TLE semantics.**
- TLEs are mean elements tied to SGP4/SDP4 [@hoots1980models; @vallado2006revisiting].
- The drag term B* is a fitted parameter that absorbs model error. It is not a physical ballistic coefficient [@salmaso2023machine] (FT).
- Drag and density modelling limits are reviewed by Vallado and Finkleman [@vallado2014critical] (META).
- ESA's operational re-entry chain (orbit theories, atmosphere models, break-up) is described by Klinkrad [@klinkrad2013methods] (FT).

**Recent physics-based uncertainty work:**
- Multifidelity Monte Carlo re-entry windows, tested on GOCE and Kosmos-482 [@sztamfatergarcia2026multifidelity] (ABS). The windows "safely contain" the true decay at a fraction of crude Monte Carlo cost. These are case studies, with no coverage statistics over many objects.
- A semi-analytic decay model with solar radiation pressure, process noise and Monte Carlo re-entry time domains [@chen2025uncertainty] (ABS).
- Drag-coefficient bounds from the relation between SGP4 mean and osculating elements, applied to the Shenzhou XVIII orbital module [@chen2025debris; @chen2025abandoned] (ABS). The median is within one day; the second paper reports "sub-day" accuracy.
- Segmented mass-to-area calibration on public TLEs [@remis2026predicting] (PRE, ABS). It reproduces TIP times to about ±1 h in the terminal phase.
- Two further items are metadata only: an analysis of assumptions in re-entry software [@flannery2025assumptions] (META) and a high-fidelity RISAT-2 prediction [@srivastava2026risat] (META).

### 1.2 TLE-based re-entry epoch prediction and the IADC campaigns

**TLE pre-processing is part of the method:**
- Lidtke et al. show, on 96 historical re-entries, that TLE filtering is needed on *all* orbital elements used, "to avoid considerably inaccurate results" [@lidtke2019optimising] (ABS).
- Gondelach et al. estimate ballistic coefficients from semi-major-axis data alone, for 101 rocket bodies in GTO [@gondelach2017ballistic] (ABS).
- Related work: ballistic coefficients of low-altitude debris from historical TLEs [@sang2013estimation] (META), and the effect of the time span used to fit the ballistic parameter [@pardini2008impact] (META).
- Shorten et al. document undocumented TLE quirks that are "a particularly hazardous data type" for ML researchers [@shorten2023wide] (FT).

**IADC re-entry test campaigns (Pardini and Anselmo, ISTI-CNR).**
- Over the first 20 campaigns (316 predictions), they define the percentage error on the residual lifetime as PE_RL = (T_PRED − T_REF) / (T_REF − T_IN) × 100 [@pardini2018assessing] (FT, conference version). T_IN is the epoch of the propagated state.
- Mean error:
  - about 10% overall, rising to 15% in the last 6 h;
  - rocket bodies about 5%, rising to about 8% in the last day.
- Window sizing:
  - a window of about ±20% around the nominal time gives 90% confidence;
  - this should rise to about ±25% in the last couple of days;
  - ±30% is needed for at least 95%.
- Empirical misses: 29 of the 316 issued windows (about 9%) did not contain the reference epoch, and 17 of 127 (about 13%) missed in the last two days.
- This is the only published *empirical coverage* figure for operational re-entry windows found in this review.

**Fifteen campaigns, 1998–2012** [@pardini2013reentry] (FT):
- The mean error was below 10% of residual lifetime in 11 of 15 campaigns and below 20% in all 15.
- High and variable solar or geomagnetic activity gives mean errors of 15–20%, "with possible excursions up to 30%".
- They recommend "a relative prediction error of at least ±20%" for uncertainty windows.

**Density models at high solar activity** [@pardini2003performance] (ABS):
- 16 objects, predicted about one month, one week and one day before decay.
- JR-71 and MSIS-86 gave errors of about 10%; TD-88 gave about 15–25%.

**ESA's automatic re-entry service** [@bastidavirgili2021impact] (FT):
- Compared NRLMSIS 2.0 with NRLMSIS-00, DTM-2013 and GOST-2004 (18 configurations) on 22 objects re-entering between December 2020 and March 2021.
- No configuration was best for every object.
- The relative error is correlated with time-to-re-entry. The authors attribute this to error being absorbed into the drag coefficient over moving fit windows.
- They warn that results "are different than the ones obtained in previous assessments … one should be careful with extrapolating the knowledge obtained for future events".
- That warning is direct support for temporal (solar-cycle-aware) evaluation.

**Other operational re-entry work:**
- A KASI re-entry study [@choi2017study] (ABS, partial).
- ISRO's in-house algorithms [@dutt2023assessment] (META).
- GOCE six-degree-of-freedom uncertainty analysis compared against TIP and ESA predictions [@geul2018analysis] (ABS).
- The Long March 5B radar campaign [@montaruli2025third] (META).
- Re-entry predictions for air-traffic management [@bernellizazzera2022reentry] (ABS).
- Re-entry statistics:
  - more than 24,400 catalogued objects have re-entered;
  - about 450 large intact objects re-entered uncontrolled between 2008 and 2017 [@pardini2019uncontrolled] (ABS).
- Earlier ISTI-CNR methodology [@pardini2004accuracy; @anselmo2005computational; @anselmo2013satellite] (META).

### 1.3 Uncertainty sources: solar flux, storms and the lower atmosphere

**Solar flux dominates long-horizon error:**
- Vallado and Oltrogge use a percentile approach over the last five solar cycles to generate space-weather scenarios for lifetime studies [@vallado2021satellite] (FT).
- RESONANCE predicts the 13-month smoothed F10.7 index 1–24 months ahead with an RMS error of 5–27 sfu (3–16 sfu for F30), and was evaluated on ESA re-entry campaigns from 2006 to June 2019 [@petrova2021medium] (ABS).

**Leonid Space's backtest.** Shambaugh backtests a physics pipeline (propagator, TLE-derived ballistic coefficients, density corrections, in-house space-weather forecasts and Monte Carlo) [@shambaugh2026validation] (PRE, FT).
- Population: 1,283 deorbited non-payload objects. The selection uses the Space-Track bulk TLE archive, the SATCAT and a DISCOS payload filter.
- Task: one year ahead, with deorbit defined as semi-major-axis altitude below 300 km.
- Median CRPS:
  - 2.4 days (0.7%) with perfect knowledge;
  - 35.3 days (9.7%) when fully predictive.
- Flux forecasts contribute "75% of the variance".
- Calibration is scored with a rank-percentile RMS calibration error (RMSCE).
- **Caveat:** a global dispersion factor is *tuned to minimise RMSCE* on the same backtest set in each scenario, so the reported calibration is in-sample.

**Solar Cycle 25 underestimation:**
- The density in low Earth orbit rose to 2–3× the 2019 SWPC forecast (Capella SAR constellation) [@shambaugh2025doing].
- In a population analysis of 1,597 payloads, the cumulative drag impulse ran 5–6σ beyond the forecast's stated uncertainty [@shambaugh2026billion] (PRE).

**Storms:**
- February 2022: 38 Starlink satellites were lost after moderate storms [@kataoka2022unexpected] (ABS).
  - Drag at about 200 km rose "up to 50%".
  - NRLMSIS 2.0 predicted less than 25%.
- 523 Starlink re-entries (2020–2024) re-entered faster, and with larger prediction errors, as geomagnetic activity increased [@deoliveira2025tracking] (ABS).
- During the May 2024 storm, TRISAT's decay rose from about 0.4 to more than 1 km/day, and DRAMA/OSCAR estimates that it lost about one week of lifetime [@kosanic2026impact] (ABS).
- Storm-time decay assessment with a decay factor and a density proxy, demonstrated on FY-3G [@wu2025space] (ABS).

**Lower atmosphere:**
- A major sudden stratospheric warming at solar minimum shifts re-entry time by 0.2–0.3 days, comparable to a 2-sfu change in solar flux [@yue2026satellite] (ABS).

**Density estimation:**
- TLE data assimilation with Kalman-filter uncertainty, using the TLEs of 17 objects and validated against CHAMP and GRACE [@gondelach2020realtime] (ABS).
- Density-model biases at the 23rd sunspot maximum [@pardini2012thermospheric] (META).
- ML density models are covered in §3.

**Further catalogue-level lifetime studies (META):**
- The correlation between ballistic coefficient and natural decay time for VLEO debris [@lee2025correlation].
- Lifetime modelling of 1U–3U CubeSats [@russell2026lifetime].

### 1.4 Machine-learning and deep-learning approaches to re-entry, lifetime and decay

Table 1. ML re-entry, lifetime and decay studies. "Split" means how test objects or time were separated from training.

| Study | Task / horizon | Data | Method | Split | Reported result | Level |
|---|---|---|---|---|---|---|
| Jung et al. 2021 [@jung2021recurrent] | Re-entry trajectory / epoch | TLEs; per Salmaso et al., the average-altitude evolution is the only feature | RNN | Unknown | Unknown (closed access) | META |
| Jung et al. 2024 [@jung2024reentry] | Re-entry prediction with "AI techniques" | Unknown | Unknown | Unknown | Unknown | META |
| Salmaso et al. 2023 [@salmaso2023machine] | Terminal re-entry epoch: altitude profile from 200 km to 80 km (TIP reference). Most training trajectories have 0.5–3 days of residual lifetime. | Space-Track OMM/TLE for decayed objects with TIP, 2000-01-01 to 2021-10-07, filtered to 417 bodies. Test: 15 IADC campaign objects (2005–2018). | Seq2Seq GRU. Features: average altitude, B*, F10.7, area-to-mass ratio. | 80/20 object split, with train and validation chosen to have similar B*/eccentricity distributions. Test epochs interleave in time with training. | Category 1 (9 objects with in-distribution B*): worst ε_abs 0.2225 h (13.35 min, NORAD 40138, start at 180 km); ε_rel up to 20.3% from 120 km. Category 2: Molniya 3-39 ε_rel 964–2532%, Phobos-Grunt 1.34 h. Point predictions only. | FT (preprint) |
| Li et al. 2023 [@li2023backpropagation] | Orbital lifetime of LEO satellites | Unknown | BP neural network on orbital parameters (as described by Lotti et al. 2025 and Chen et al. 2025) | Unknown | Unknown | META |
| Chen et al. 2025 [@chen2025lowearth] | Orbital lifetime of LEO RSOs, large-scale | **Simulated** with a semi-analytic propagator | Sensitivity analysis plus a fully connected NN; black-winged-kite hyperparameter search | Simulation | "Accurately determines" lifetime, faster than propagation; no numbers in abstract | ABS |
| Guan et al. 2025 [@guan2025intelligent]; Guan and Li 2026 [@guan2026orbital] | Orbit-life prediction; orbit prediction fused with an aerodynamic NN | Unknown | Unknown | Unknown | Unknown | META |
| He et al. 2024 [@he2024expandable] | Precise orbit-decay prediction in VLEO | Spacecraft in VLEO (details not in abstract) | Continual learning ("expandable orbit decay propagator") | Not stated | "At least a 70% improvement in accuracy" over a high-precision propagator | ABS |
| Zhou et al. 2026 [@zhou2026orbital] | Decay at given altitudes under storms | GRACE Level-1B along-track density, aerodynamic coefficients, mass | Random forest | 113 ICME events tested independently | Test R² 0.749 (2005); 85% of variance explained; May 2024 storm: 48-h semi-major-axis decay of about 168 m | ABS |
| Lotti et al. 2025 [@lotti2025exploring] | Orbital lifetime (semi-major-axis decay to perigee < 150 km) | Space-Track TLEs of decayed payloads with lifetime ≥ 1 yr (3,590, of which 1,783 survived filtering). Train: 1,435 (351 decayed + 1,084 not decayed). Val: 348 decayed. | Physics-informed 1-D convolutional NN; residual loss on the orbit-averaged drag equations for semi-major axis and eccentricity; initial conditions enforced through a theory-of-functional-connections form | Decayed objects split 50/50 into validation, stratified by four lifetime classes; not temporal | Preliminary. The data-only network memorises the decay index and fails when time sampling changes; generalisation "not yet fully satisfactory" | FT |
| Andaristiyan et al. 2025 [@andaristiyan2025ballistic] | Ballistic coefficient from historical TLEs | 16 HASDM objects; NRLMSIS 2.1 | Fitting | n/a | Absolute relative error < 13% | ABS |
| Andaristiyan et al. 2026 [@andaristiyan2026future] | Forecasting future ballistic coefficient for re-entry prediction | Unknown | Deep learning (title) | Unknown | Unknown | META |
| Chen et al. 2026 [@chen2026highprecision] | Re-entry of low-ballistic-coefficient debris | Unknown | Physics- and geometry-constrained GAN (title) | Unknown | Unknown | META |
| Dyreby et al. 2025 [@dyreby2025analyzing] | Ephemeris quality and decay of a mega-constellation | About 1,500 Starlink satellites, 2 months, multiple public sources | Physics-informed ML to extract non-conservative accelerations | n/a | Public-ephemeris position RMSE about 300 m (not deorbiting) vs about 600 m (deorbiting) | ABS (arXiv) |
| Shang et al. 2026 [@shang2026rapid] | Rapid orbital-decay prediction | Unknown | Sparse multivariate polynomial regression | Unknown | Unknown | META |
| Tani et al. 2025 [@tani2025hybrid] | Debris classification and "decay pattern" clustering | DISCOS records with decay dates (access route not stated) | Autoencoder + MLP; XGBoost; UMAP + DBSCAN | Not stated | 93% *classification* accuracy (not a decay-time metric) | ABS |
| Cimmino et al. 2023 / 2025 [@cimmino2023machine; @cimmino2025recurrent] | Ballistic-coefficient estimation | 2025: **synthetic** high-accuracy propagation | 2025: GRU on unevenly sampled orbital elements | Synthetic | 2025: mean percentage error 10% | META / ABS |
| Gao et al. 2023 [@gao2023reentry] | Ground risk: debris landing velocity, longitude and latitude | Simulated break-up components as simple shapes | SVR, decision-tree regression, MLP | Not stated | "High-accuracy" warnings within 10 s; no numbers in abstract | ABS |

**What the readable ML studies have in common:**
- Every study whose design could be read predicts a **point** value (Salmaso, Lotti, Chen 2025, He, Zhou, Tani).
- They use **object-level random or stratified splits**, or simulation, rather than a forward-in-time split.
- Salmaso et al. train only on objects that have decayed and have a TIP.
- In Salmaso et al., the altitude profiles of training, validation *and* test objects are generated by the same fitting procedure (Fig. 5), and that procedure sets the reference time to the TIP re-entry epoch (Sec. 4.1, Eq. 16). The authors note that operationally the last TLE epoch would have to be used, "we thus expect to observe some differences in performance". This is a form of label information entering the inputs, and it limits comparability with operational predictions.
- None of these studies reports interval coverage or a calibration metric, and none compares against the ±20% operational rule [@pardini2018assessing].

A survey of ML for orbit estimation covers orbit determination, orbit prediction and density modelling [@caldas2024machine].

### 1.5 Survival and hazard formulations

**Re-entry.** No statistical time-to-event (censored) formulation of catalogue re-entry was found; the searches are listed in `LIT_NOTES_GAP.md`. In this literature, "re-entry survival analysis" means *demise survivability* of fragments (for example ORSAT- or SARA-type tools), not survival analysis of decay time.

**Satellite health.** Survival methods are established for on-orbit **reliability**:
- Kaplan–Meier and Weibull analysis of satellite and subsystem failures [@castet2009satellite].
- Statistical reliability of Earth-orbiting satellites operating from 1991 to 2020 [@grile2025statistical] (META).

These are relevant to the project's satellite-health component.

**Methods that could be transferred:**
- Accelerated-failure-time gradient boosting [@barnwal2022survival].
- Conformalized survival analysis, giving distribution-free lower bounds on survival time under censoring [@candes2023conformalized; @gui2024conformalized].
- Neither has been applied to space-object decay in the records checked.

### 1.6 Data used by the re-entry and lifetime studies

| Study | Orbit data | Labels / truth | Physical properties | Other |
|---|---|---|---|---|
| [@salmaso2023machine] | Space-Track OMM/TLE | TIP epochs (two manually corrected per IADC) | B* as proxy; area-to-mass ratio | F10.7 |
| [@lotti2025exploring] | Space-Track TLEs | Perigee below 150 km (own definition) | Ballistic coefficient from median non-negative B* | Solar-cycle phase |
| [@operti2017assessing] | USSTRATCOM TLEs | Observed lifetime | DISCOS mass, shape and cross-section | OSCAR propagation |
| [@shambaugh2026validation] | Space-Track bulk historical TLE archive | Semi-major-axis altitude below 300 km | Ballistic coefficient from TLEs | SATCAT filters; DISCOS payload flag; in-house space-weather forecasts |
| [@shambaugh2026billion] | Space-Track TLEs | Not stated here (physics analysis) | DISCOSweb masses | GCAT end-of-operations records; CelesTrak |
| [@pardini2018assessing], [@pardini2013reentry] | TLEs and IADC exchange data | IADC reference epochs | Fitted ballistic parameter | Campaign data |
| [@bastidavirgili2021impact] | TLE (20-TLE and variable spans) | Space-Track assessed re-entry time | Fitted drag coefficient | Daily or 3-hourly Ap; a-posteriori space weather |
| [@deoliveira2025tracking] | TLEs of 523 Starlink satellites | Re-entry epochs | — | Geomagnetic indices |
| [@tani2025hybrid] | — | DISCOS decay dates | DISCOS | — |
| [@zhou2026orbital] | GRACE Level-1B | Observed decay | Aerodynamic coefficients, mass | ICME catalogue |

- No ML re-entry study in this set uses **GCAT** or **CORDS** as a data source (see §4).
- The DISCOS-based studies need an institutional ESA account.

---

## 2. Debris analysis

### 2.1 ESA Kelvins Collision Avoidance Challenge (CAC): exact facts

**Source.** Uriot et al., *Astrodynamics* 6(2):121–140, 2022 [@uriot2022spacecraft] (FT; open access; online 2021-04-07). The official pages are [@esa2019kelvinscac] and the dataset is on Zenodo, DOI 10.5281/zenodo.4463683 (CC BY 4.0), which includes the formerly private test labels [@uriot2021cacdataset].

**Data:**
- ESA Space Debris Office CDMs, 2015–2019: 15,321 events, 199,082 CDMs, 103 attributes (Table 1).
- The risk is stored as log10 of the collision probability, floored at −30.
- The training file has 13,154 events (162,634 rows). The test file has 2,167 events (24,484 rows): 150 high-risk and 2,017 low-risk.
- The split was hand-picked to over-represent high-risk events. There is no chaser identity and there are no absolute epochs.

**Prediction setting (Sec. 4.2):**
- The target is the risk in an event's last CDM.
- Test events must have at least 2 CDMs, a last CDM less than 1 day before TCA, and a first CDM at least 2 days before TCA.
- All CDMs with time-to-TCA under 2 days are removed from the inputs.

**Official metric (Sec. 4.3, Eqs. 1–4):** lower is better.

> L(r̂) = MSE_HR(r, r̂) / F₂
>
> MSE_HR = (1/N*) Σᵢ 1ᵢ (rᵢ − r̂ᵢ)², where 1ᵢ = 1 if the *true* final risk rᵢ ≥ −6 (log10), else 0, and N* = Σᵢ 1ᵢ
>
> F_β = (1+β²)·p·q / (β²·p + q) with β = 2, computed over the whole test set with classes r ≥ −6 and r < −6

- The paper reports every score after **clipping** low-risk predictions to −6 − ε with ε = 0.001, that is −6.001.
- The official Kelvins results page lists **unclipped** scores, which differ. For example, team madks scored 14.596 officially but 0.634 clipped in the paper's Table 3. The paper's "top ten" is therefore ranked by clipped score.

**Baselines (Sec. 4.4, Table 3):**
- **Latest-risk prediction (LRP):** r̂ equals the risk of the latest CDM at least 2 days before TCA, or −6.001 if that risk is below −6. Its score is **L = 0.694** (MSE_HR 0.513, F₂ 0.739).
  - "Of the 97 teams, 12 managed to submit better solutions."
  - The 12th team's score equals LRP's to five decimals [@vazquez2026latest] (PRE).
  - The paper itself gives the team count as 96 in two places and 97 in another.
- **Constant prediction** of −5: L = 2.5, beaten by 38 teams.

**Winners (Table 3, clipped; L / MSE_HR / F₂):**

| Rank | Team | L | MSE_HR | F₂ | Official unclipped L |
|---|---|---|---|---|---|
| 1 | sesc | **0.556** | 0.407 | 0.733 | 0.5553 |
| 2 | dietmarw | **0.571** | 0.437 | 0.765 | 0.5745 |
| 3 | Magpies | **0.585** | 0.441 | 0.753 | 0.5849 |
| 4 | Vidente | 0.610 | 0.436 | 0.714 | 0.6098 |
| 5 | DeCRA | 0.615 | 0.457 | 0.743 | 0.6149 |

- The winning entry is a hand-tuned threshold cascade on the latest risk. It scores worse than LRP on the training set (1.050 against 0.804).
- After the competition, a LightGBM model beat LRP on both splits, with test L = 0.684, which is 1.4% better (Sec. 6.1).
- 11 of the 150 high-risk events were missed by all top-10 teams. The authors therefore suggest predicting a risk *distribution* rather than a point (Sec. 5.4).

### 2.2 Later work on the CAC data: what is comparable

**Comparable** (same 2,167-event test set and the same clipped L):
- Tulczyjew et al. [@tulczyjew2021predicting] (META plus abstract). This is team DunderMifflin, 7th, with 0.628 clipped and 0.6276 official.
- Vazquez 2026 [@vazquez2026latest] (PRE, FT; Zenodo, not peer reviewed, generative AI disclosed). A pre-registered re-evaluation.
  - It reproduces LRP exactly.
  - No model beats LRP with confidence. The best primary result is L = 0.661 (−4.7%, 97.5% CI [−14.0, +3.6]%).
  - It reports Platt-calibrated *probability* forecasts on pooled cross-validation: log loss 0.0503, against 0.0556 for a calibrated latest-risk baseline.
  - It argues that the organisers' paired t-test, which ignores overlapping splits, drops to p = 0.36 once corrected.

**Not comparable** (different data, split or metric):
- [@abay2021benchmarking]: internal ESA set of 2.2 M CDMs; random 80:20 split; F1/F2/RMSE.
- [@pinto2020towards] (PRE): Bayesian LSTM predicting the next CDM; random 85/15 split; next-CDM MSE.
- [@acciarini2021kessler]: the Kessler library and a synthetic CDM generator.
- [@catulo2023predicting] (PRE): Bayesian HMM; its 95% HDIs "have poor coverage".
- [@ouari2026probabilistic] (abstract): F₂ 0.808 and AUC 0.981 on the *pooled* 15,321 events.
- [@zerrouki2026deep] (META).
- [@sanchez2024robust]: belief-function classes; internal ESA mission data with cross-mission transfer.
- [@mashiku2025nasa]: NASA CARA internal data, about 450k CDMs; emphasises data scarcity and stochasticity.
- [@olson2026contextual]: Slingshot data; high-covariance conjunctions.
- [@tok2026early] (PRE): TCN-Transformer on in-house TÜBİTAK CDMs.
- [@arshad2026leakage]: 149,044 SOCRATES events with a pair-disjoint split.
- [@rahman2026forecast]: MS thesis claiming conformal bounds on CDM data; embargoed until 2028, so unverifiable.

### 2.3 Debris origin and fragment-to-parent attribution

**The Kelvins "Space debris: the origin" dataset is simulated.**
- Zenodo DOI 10.5281/zenodo.4986292, CC BY 4.0, with creators Efthymiopoulos, Märtens, Gaitanas and Izzo [@efthymiopoulos2021origin].
- The official challenge page's background section is headed "Background (fictional)" [@esa2021kelvinsorigin].
- The page states the exact equations of motion the debris obey: Kepler, J2, C22/S22, analytic Sun and Moon, and solar radiation pressure scaled by Cr·A/m.
- The training labels give "the epoch at which the debris was generated".
- All 100 parent trajectories sit on an exact 10-day grid, and every debris and generation epoch is a multiple of 5 days.
- The Zenodo text never uses the word "simulated", so this is an inference, but a strong one.

**Challenge format:**
- The task is GEO only, with 100 debris and 100 candidate parents.
- Score: G = (G1 + 1)(G2 + 1) − 1, where G1 is the fraction of wrong parents and G2 is the MSE of Cr·A/m.
- The winner scored G = 0.2252.
- Test labels are not public, and no paper using the dataset was found.

**Catalogue-based attribution with real or simulated breakups:**
- Orbital-similarity criteria (D-criteria and MOID) for parent identification and breakup epoch [@dimare2019inorbit].
- PUZZLE back-propagation with pruning and clustering [@romano2021puzzle], and its 2026 follow-up on two real events [@ottoboni2026reconstruction].
- FRED, which estimates the fragmentation epoch from a single fragment orbit determination [@montaruli2023stochastic].
- Breakup type and mass versus NASA standard breakup model (SBM) simulations; the Hitomi epoch is estimated to within minutes of JSpOC's [@liu2023determining].
- Proper elements to reconnect fragments with parents [@celletti2021reconnecting; @celletti2022proper], combined with clustering in [@celletti2026perturbative].
- **Neural networks on numerically computed proper elements of real LEO TLE families**, applied to analyst objects of unknown origin [@wu2023investigation]. This is the closest real-catalogue ML attribution work.
- A pairwise "same parent?" classifier on synthetic SBM families without drag [@ling2026supervised].
- The NASA SBM itself [@johnson2001nasa].

**Labels and motivation:**
- GCAT supplies catalogue-level fragment-to-parent labels: type D, names of the form "deb X", and Parent/SDate fields [@mcdowell2020gcat].
- ESA's 2026 environment report says further work to correlate unidentified objects to fragmentation events may reduce the unidentified class [@esa2026ser].

### 2.4 Graph neural networks and network science on catalogues and conjunctions

**GNN work found** (all-versus-all conjunction screening framed as link prediction):
- [@stevenson2022towards] (abstract only).
- [@chu2024conjunction] (GCN; F1 0.8847).
- [@chu2026conjunction] (META).
- [@li2024dynamic] (dynamic GNN; GC-LSTM AUC 0.944).
- [@peri2025graph] (PRE; self-reported).

**Network-science representations:**
- An early network of the debris population for active-removal targeting [@lewis2010new].
- RSONet, which builds a network risk index from near-collision events [@romano2024resident].
- Centrality combined with DilutionMaxPc [@lin2024risk].

**Not found:** no GNN or graph transformer for CDM-sequence risk forecasting, for fragment-to-parent attribution, or for lifetime or re-entry prediction. A targeted arXiv query returned only an unrelated tether-net capture paper.

---

## 3. Integrated architectures, physics-guided ML and uncertainty calibration

### 3.1 Integrated or fused debris + re-entry + health systems

- A "digital twin of the space environment" review reports no quantitative results; its scope is mission environments [@liu2024digital].
- A digital-twin orbit-prediction system (STL-TCN on ILRS and TLE data) is the only quantitative SSA digital-twin paper found. It covers orbit prediction only [@xu2025digital].
- No peer-reviewed model combining debris risk, re-entry and satellite-health evidence was found; only unreviewed preprints exist.
- A Zenodo record uses the term "physics-gated neural network" for Starlink drag correction [@em2026physics] (PRE). It reports a 12.6% reduction in mean position error during severe storms.
- The project's previous telemetry work is the health component. Its literature is in `paper/journal_v2/refs_journal.bib` and is not repeated here.

### 3.2 Physics-guided ML for orbit and decay

**ML correction of physics-based orbit prediction:**
- SVM error learning on 11 objects using TLE and ILRS data [@peng2020machine].
- A Gaussian-process version with uncertainty [@peng2019gaussian].
- **EKF fusion of the ML correction with classical prediction**, with analytic rather than learned fusion [@peng2021fusion]. This is the nearest peer-reviewed physics–ML fusion, and it is for state prediction.
- A medium-scale TLE study [@peng2023medium] (META) and error-pattern transfer [@li2021improved] (META).

**Residual learning on TLE/SGP4:**
- A neural network trained on more than 4,500 inactive objects over 10 months corrects state-vector errors "by at least 40% for 70% of the TLE", with Student-t residuals [@bizalion2023systematic].
- BiLSTM compensation on 340 LEO objects [@lin2026bilstm].
- A survey calls for foundation models and multi-regime datasets [@lee2026toward].

**Differentiable SGP4.** ∂SGP4 and ML-∂SGP4 [@acciarini2025closing] (FT of arXiv v5) are published in *Acta Astronautica* 226 (2025).
- Data: Starlink, 1,519 satellites, 2023-02-18 to 2023-02-21.
- Test set: 225 satellites below 539 km.
- Test position RMSE: SGP4 107.42 km; NN output correction 88.66 km; ML-∂SGP4 71.25 km.
- Limitation: the data cover a 3-day window, so there is no temporal validation.

**Density and ballistic coefficient as physics inputs:**
- ML density models cut MAPE from 40–60% to about 20% on POD-derived densities [@acciarini2024improving]. The split design is unverified.
- ML reduced-order density models with data assimilation feed a high-precision propagator to predict deorbit time, in case studies [@briden2026atmospheric] (ABS).
- Machine-learned ballistic coefficients [@cimmino2023machine; @cimmino2025recurrent; @andaristiyan2026future].

**Physics-informed lifetime.**
- The only physics-informed NN for orbital lifetime found is a work-in-progress EUCASS paper with unsatisfactory generalisation [@lotti2025exploring].

### 3.3 Uncertainty calibration in SSA tasks

**Thermospheric density** (Licata, Mehta and co-workers):
- They compare MC dropout with direct (μ, σ) prediction under an NLPD loss [@licata2022uncertainty] (FT).
- Calibration is scored with a **calibration error score**: the mean absolute gap between expected and observed coverage over 20 prediction intervals.
- The CHAMP model is "within 2% of perfect calibration".
- HASDM-ML [@licata2022machine] and MSIS-UQ [@licata2022msis] use the same score.
- **All three use interleaved 8-week/1-week/1-week train/validation/test blocks.** Chronological or long-segment splits were dropped because they "did not well-generalize", so the calibration is not tested forward in time.
- Related: propagation of density uncertainty to orbits [@paul2023advanced] and probabilistic F10.7 ensembles [@daniell2023probabilistic].

**Conformal prediction in SSA:**
- The only conformal-prediction SSA publication found in a journal or proceedings is Chen et al. 2013 (SPIE): anomaly detection and collision alerts on NORAD catalogue data, with exchangeability assumed [@chen2013conformal].
- The relevant method literature: split and conformalized quantile regression [@romano2019cqr], adaptive conformal inference under shift [@gibbs2021aci], an overview [@angelopoulos2023gentle] and deep ensembles [@lakshminarayanan2017ensembles].

**Collision risk:** Bayesian and calibrated approaches are listed in §2.2. Calibrated *probabilities* exist [@vazquez2026latest]; distribution-free *intervals* on the official split were not found.

**Re-entry and lifetime:**
- Operational windows are heuristic (±20–30% of residual lifetime) with about 91% empirical coverage [@pardini2018assessing].
- Physics Monte Carlo windows are case-study based [@sztamfatergarcia2026multifidelity].
- The one catalogue-scale calibrated backtest is in-sample tuned and not peer reviewed [@shambaugh2026validation].

**Tabular foundation models.** TabPFN [@hollmann2025accurate] has no SSA application in the indexes checked.

---

## 4. Datasets and data access

### 4.1 Is there a public historical TLE archive without Space-Track?

**Only partially; there is no legitimate bulk, catalogue-wide archive.**

**CelesTrak Special Data Request** [@celestrak2026archives]:
- "Data is available for all unclassified objects from 1957 through the latest GP data update", with no account.
- Limits: "10 requests per 24-hour period", up to 100 satellites per request, and at most 20,000 records per request.
- A CAPTCHA was added specifically against scripted requests.
- For bulk needs the page says "you should consider using Space Track". That makes it impractical for about 35k decayed objects.
- The CelesTrak static archives for 1980–2004 cover only a curated set of about 196 satellites.

**J. McDowell's historical TLE pages** [@mcdowell2022elements]: public, but by his own account they do "not contain recent element sets". Spot checks found sparse post-2004 coverage. No licence is stated.

**Shorten et al. benchmark** [@shorten2023wide]: 15 manoeuvring satellites with TLEs downloaded from Space-Track. The repository has no licence file. It is a manoeuvre benchmark, not a decay one.

**Kaggle "Historical satellite TLE Dataset"** [@kaggle2026historicaltle]: self-described as Space-Track TLEs for 2004–2025, about 32 GB, labelled CC0. That label conflicts with the Space-Track agreement, so it is not suitable as a primary source.

**Space-Track** [@spacetrack2019agreement]:
- `GP_HISTORY` holds "over 138 Million elsets" and needs a free account.
- The user agreement says: "The User agrees not to transfer any data or technical information received from this website … including the analysis of data, to any other entity without prior express approval."
- The safe route is to release code, NORAD IDs and query specifications rather than derived TLE tables.

**Operational hazards in 2026:**
- CelesTrak ran out of 5-digit catalogue numbers on 2026-07-11, so new objects have no data in TLE format.
- On 28 May 2026 the 18th Space Defense Squadron changed `RCSVALUE` from average RCS in m² to "median radar cross section in meters" (Space-Track FAQ). For most users `RCSVALUE` is published as 0, so the change matters only to accounts that can see numeric values. Any RCS-based feature should still be checked for a break at that date.
- Both points were checked against the CelesTrak request page and the Space-Track documentation page fetched on 2026-10-27.

### 4.2 GCAT and McDowell's re-entry catalogue

**GCAT** [@mcdowell2020gcat]:
- Release 1.8.8, 2026-09-03, with a data update on 2026-10-25. Licence CC-BY-4.0.
- Per object: mass and dimensions, `DDate` (decay date), Parent/SDate, and **one** orbit (`ODate`, perigee, apogee, inclination). There is no history.

**McDowell's re-entry catalogue** [@mcdowell2026reentry] (updated 2026-09-09):
- A TSV with OrbLife, PMLife (post-mission life), end-of-mission perigee, apogee and inclination, masses, manoeuvre flags and disposal codes, for payloads, rocket bodies and parts.
- It is effectively a public lifetime-label table. The page states no licence, so attribution should be confirmed with the author.

**Use in ML:** no ML paper using GCAT as training data was found. Examples of non-ML use:
- Wright et al. use GCAT rocket-body re-entry data for airspace-risk statistics [@wright2025airspace].
- Longuet et al. cite GCAT only for satellite counts [@longuet2025formal].

### 4.3 ESA DISCOS

DISCOS has "orbital data histories for all trackable, unclassified objects", more than 40,000, plus size, mass and shape [@esa2026sdup; @klinkrad1991discos].
- Accounts are restricted to research institutes, government bodies and companies of ESA Member States, "not as an individual".
- The API documentation is behind a login, so field names are unverified.

ML or statistical uses:
- A taxonomy autoencoder (PRE) [@guimaraes2023taxonomy].
- Classification and clustering [@tani2025hybrid].
- Drag-coefficient statistics from B* plus DISCOS mass and shape [@hakima2022analysis].
- A natural-language question-answering system over DISCOS [@darm2023discosqa].
- OSCAR validation [@operti2017assessing].
- Shambaugh's population analyses [@shambaugh2026validation; @shambaugh2026billion].

### 4.4 CORDS, ESA re-entry portal and Space-Track TIP/DECAY

**Aerospace Corporation CORDS** [@aerospace2026cords]:
- A public table of re-entries since 2000, showing the *predicted* re-entry time with a ± window. A CSV is linked for older data.
- The live site blocks scripts; it was read through a Wayback capture. CSV contents are unverified.
- No ML paper using CORDS was found.

**ESA re-entry portal** [@esa2026reentry]: the history and recovered-debris pages are public. The predictions endpoint returns HTTP 401.

**Space-Track TIP and DECAY:**
- An account is needed.
- TIP latitude and longitude refer to the point above which the object reaches 10 km altitude, not the impact point.
- Salmaso et al. use TIP epochs as labels [@salmaso2023machine].

### 4.5 Labelled benchmark datasets relevant to this project

| Dataset | Real or simulated | Public | Task | Notes |
|---|---|---|---|---|
| CAC (Zenodo 4463683) [@uriot2021cacdataset] | Real ESA CDMs | Yes, CC BY 4.0, including test labels | Final-CDM risk | Anonymised; 150 high-risk test events |
| Space debris: the origin (Zenodo 4986292) [@efthymiopoulos2021origin] | **Simulated** (fictional GEO scenario) | Training only; test labels not public | Fragment → parent, Cr·A/m | SRP-dominated dynamics |
| Kessler generator [@acciarini2021kessler] | Synthetic CDMs | Yes (software) | CDM sequences | Calibrated to Kelvins distributions |
| Shorten et al. [@shorten2023wide] | Real TLEs (Space-Track) | Yes, no licence file | Manoeuvre detection | 15 satellites |
| GCAT and McDowell re-entry catalogue [@mcdowell2020gcat; @mcdowell2026reentry] | Real catalogue | Yes (GCAT CC-BY-4.0; re-entry catalogue licence not stated) | Decay dates, post-mission life | No orbit history |
| CelesTrak SATCAT [@celestrak2026satcat] | Real catalogue | Yes, no account | Decay dates, current orbit | Snapshots must be archived by the user |

---

## 5. Reported results: read the comparability column first

Table 2. Headline numbers. Rows in the same block are **not** comparable unless the last column says so.

| Task | Study | Data / split | Metric (definition) | Reported value | Comparable with |
|---|---|---|---|---|---|
| Re-entry epoch, last ~10–14 days | [@pardini2018assessing] | 20 IADC campaigns, 316 predictions | Mean \|PE_RL\|, % of residual lifetime | ≈10% (15% last 6 h); rocket bodies ≈5% (≈8% last day); 9% of windows missed truth | [@pardini2013reentry] (same group, similar residual-lifetime metric; its 15 campaigns are the first 15 of the 20) |
| Re-entry epoch, last ~10 days | [@pardini2013reentry] | 15 IADC campaigns, 1998–2012 | Mean relative error, % of residual lifetime | < 10% in 11/15; < 20% in 15/15 | Row above |
| Re-entry epoch, 1 month / 1 week / 1 day | [@pardini2003performance] | 16 objects, high solar activity | Residual-lifetime error | ≈10% (JR-71, MSIS-86); ≈15–25% (TD-88) | Not comparable (different period and objects) |
| Re-entry epoch, 0–28 days | [@bastidavirgili2021impact] | 22 objects, Dec 2020–Mar 2021 | Relative error and relative model error | No model best overall | Not comparable |
| Terminal re-entry, 200 → 80 km | [@salmaso2023machine] | 417 bodies, 80/20 by object; 15 IADC test objects | ε_abs (h); ε_rel = \|Δt\| / (t_actual − t_initial) | In-distribution worst 13.35 min (from 180 km); Molniya 3-39 ε_rel up to 2532% | **Not** comparable with IADC rows: horizon of hours to days, and inputs fitted with the known TIP epoch |
| Whole orbital lifetime | [@operti2017assessing] | 575 R/B, 333 P/L, 73 small sats (DISCOS + TLE) | (C − O)/O | LEO R/B mean −4.06%, σ 43.10%; small sats mean +13.36%, σ 36.57% | Not comparable with re-entry-window rows (lifetime of years versus days) |
| 1-year-ahead deorbit (to 300 km) | [@shambaugh2026validation] (PRE) | 1,283 non-payload objects, 1968–2026 | Median CRPS (days, % of 365 d); RMSCE | 2.4 d (0.7%) perfect knowledge; 35.3 d (9.7%) fully predictive | Not comparable with Operti (different horizon and objects); calibration tuned in-sample |
| Decay under storms | [@zhou2026orbital] | GRACE density; 113 ICME events | R² | 0.749 (2005); 85% of variance | Not comparable |
| Orbit decay in VLEO | [@he2024expandable] | VLEO spacecraft | Accuracy improvement vs high-precision propagator | ≥ 70% | Not comparable |
| Lifetime via PINN | [@lotti2025exploring] | 1,435 train / 348 validation (Space-Track) | Qualitative | Fails to capture the timescale | n/a |
| F10.7 forecast, 1–24 months | [@petrova2021medium] | Solar-flux record; ESA campaigns 2006–2019 | RMS error (sfu) | 5–27 sfu | n/a |
| SSW effect on re-entry | [@yue2026satellite] | WACCM-X + POD, January 2009 | Re-entry time shift | 0.2–0.3 day (≈ 2 sfu) | n/a |
| CAC final risk | [@uriot2022spacecraft] | Official 2,167-event test, clipped | L = MSE_HR / F₂ | LRP 0.694; winner 0.556; LightGBM 0.684 | Comparable with [@tulczyjew2021predicting] (0.628) and [@vazquez2026latest] (0.661, PRE) |
| CAC-derived risk | [@ouari2026probabilistic] | Pooled 15,321 events | F₂, AUC | 0.808, 0.981 | **Not** comparable with CAC L |
| ESA CDM risk (internal) | [@abay2021benchmarking] | 2.2 M CDMs, random 80:20 | F2 | Naive 0.638; LDA 0.640 | Not comparable |
| Next-CDM forecast | [@pinto2020towards] (PRE) | Kelvins training data, random 85/15 | Normalised MSE | Persistence 0.2433; BNN (50 samples) 0.1419 | Not comparable with CAC L |
| Debris origin (simulated GEO) | [@esa2021kelvinsorigin] | 100 test debris | G = (G1+1)(G2+1) − 1 | Winner 0.2252 | Only with challenge entries |
| Orbit prediction (position) | [@acciarini2025closing] | Starlink, 3 days, test below 539 km | RMSE (km) | 107.42 / 88.66 / 71.25 | Not comparable with decay tasks |
| TLE state correction | [@bizalion2023systematic] | More than 4,500 objects, 10 months | Error reduction | ≥ 40% for 70% of TLEs | Not comparable |
| Density | [@acciarini2024improving] | CHAMP, GOCE, GRACE, SWARM | MAPE | 40–60% → ≈20% | Not comparable |
| Density UQ | [@licata2022uncertainty] | CHAMP; interleaved blocks | Calibration error score | Within 2% of perfect calibration | Not out-of-time |
| Conjunction screening (GNN) | [@chu2024conjunction] | Catalogue | F1 | 0.8847 | Not comparable with CAC |

---

## 6. Checked but excluded, or not verifiable

- **Excluded as off-topic despite matching titles:**
  - Zhang et al. 2023, *Knowledge* ("post-blast re-entry time"), which is about mine ventilation.
  - Hypersonic glide or re-entry vehicle trajectory papers.
  - "Re-entry" papers from medicine and social science.
- **Not verifiable** (not cited, or cited only through their metadata):
  - The details of every META entry above, in particular Jung 2021, Li 2023, Guan 2025, Andaristiyan 2026 and Chen 2026. These are the most relevant closed ML papers, and the authors should read them before submission.
  - The ISO standard on orbit-lifetime estimation: the BSI DOI record carries no standard number or edition, so it is left out.
  - Newland's 2012 PhD thesis, known only second-hand.
  - The Metz MSc thesis, cited by Abay 2021, which could not be located.
- **Unreviewed items cited only with the PRE label:** Vazquez 2026, Shambaugh 2026a/b, Remis 2026, Peri 2025, EM 2026, Pinto 2020, Catulo 2023, Tok 2026, Guimarães 2023.
- **Context:** the collision cascade motivating debris work is Kessler and Cour-Palais [@kessler1978collision].
