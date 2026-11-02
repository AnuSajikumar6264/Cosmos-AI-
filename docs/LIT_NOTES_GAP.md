# Literature notes: what is not yet done (candidate research gaps)

Compiled 2026-10-27, as a companion to `docs/RELATED_WORK.md`. Citation keys refer to `docs/references_reentry.bib`.

**How to read the status labels:**

| Status | Meaning |
|---|---|
| **Verified absent in N papers checked** | The method sections of N papers were read, at least the abstract and in most cases the full text, and none does the thing. The index searches listed with each gap also returned nothing relevant. This is evidence of absence within those papers and indexes. It is not proof that nobody has done it. |
| **Uncertain** | At least one plausibly relevant paper could not be read (closed access, embargo), or a key index could not be fully searched. |
| **Partly done** | Something close exists; the remaining gap is stated precisely. |

**Search coverage and its limits:**
- OpenAlex: title/abstract boolean queries, until the daily credit limit was reached.
- Crossref: bibliographic queries.
- Semantic Scholar: records and citation contexts, used to find what citing papers say about closed papers.
- arXiv API: abstract-field boolean queries, paced to cope with rate limits.
- The ESA SDO proceedings database (title and full-text fields).
- Google Scholar and Scopus were not available.
- Closed Elsevier papers whose abstracts are not exposed by Crossref, OpenAlex or Semantic Scholar could only be checked by title. They are listed explicitly under each gap because they are the main source of uncertainty:
  - [@jung2021recurrent], [@li2023backpropagation], [@guan2025intelligent], [@guan2026orbital], [@andaristiyan2026future], [@chen2026highprecision], [@zerrouki2026deep] and [@chu2026conjunction];
  - also the IAA proceedings papers [@jung2024reentry] and [@wacker2025largescale].

---

## Ranking (strongest first)

1. **G1 + G2 + G3 together:** catalogue-wide, censoring-aware re-entry and lifetime prediction, evaluated forward in time across solar-cycle phases, with distribution-free (adaptive) conformal intervals benchmarked against the operational ±20% rule. Each part is individually absent or only weakly present, and the combination was not found anywhere.
2. **G5:** breakup-family information as predictors of fragment lifetime and re-entry, and as context for conjunction risk.
3. **G4:** learned (gated) fusion of a physics lifetime estimate with an ML estimate. Likely open, but three closed 2026 hybrid papers must be read first.
4. **G7:** conformal intervals and leave-mission-out evaluation on the Kelvins CAC split. Narrow, and partly pre-empted by a 2026 preprint.
5. **G6:** tabular foundation models for SSA. Absent, but weak as a stand-alone contribution.
6. **G8–G11:** graph models, integrated debris + re-entry + health systems, an open decay benchmark, and a real-catalogue attribution benchmark. All absent or partial, but each needs a demonstrated benefit to count as more than engineering.

---

## G1. Catalogue-wide ML re-entry or lifetime prediction evaluated forward in time

**Gap.** An ML model for lifetime or decay date, trained on objects that decayed before a cut-off date and tested on objects that decay after it, ideally across a solar-cycle change. It should be compared with a physics baseline on the same objects.

**What exists:**
- **ML studies use object-level random or stratified splits, or simulation:**
  - Salmaso et al. use an 80/20 object split chosen so that training and validation have matched B* and eccentricity distributions. Their 15 test objects (IADC, 2005–2018) interleave in time with the training re-entries (2005–2021) [@salmaso2023machine].
  - Lotti et al. put 50% of decayed objects into validation, stratified by lifetime class [@lotti2025exploring].
  - Chen et al. train on simulation output [@chen2025lowearth].
  - For He et al. [@he2024expandable] and Zhou et al. [@zhou2026orbital], no forward-in-time split is stated in the abstract.
- **Salmaso et al.'s horizon is terminal:** 200 km down to 80 km, with most residual lifetimes between 0.5 and 3 days. Inputs are built from a fit that uses the known TIP epoch (their Sec. 4.1).
- **Catalogue-scale backtests exist only for physics pipelines:**
  - OSCAR lifetimes for 981 objects: LEO rocket-body (C − O)/O standard deviation 43.10% [@operti2017assessing].
  - Leonid Space: 1,283 objects, one year ahead [@shambaugh2026validation] (not peer reviewed).
- **Temporal shift is documented as the dominant risk:**
  - Solar Cycle 25 drag ran 5–6σ beyond forecast uncertainty [@shambaugh2026billion].
  - Flux forecasts account for 75% of the variance in one-year deorbit predictions [@shambaugh2026validation].
  - ESA warns against extrapolating density-model rankings across solar-cycle phases [@bastidavirgili2021impact].
  - The density-UQ group dropped chronological splits because they generalised poorly [@licata2022uncertainty; @licata2022msis].

**Papers checked (ML, design readable):** 9.
- [@salmaso2023machine], [@lotti2025exploring], [@chen2025lowearth], [@he2024expandable], [@zhou2026orbital], [@tani2025hybrid], [@dyreby2025analyzing], [@cimmino2025recurrent], [@gao2023reentry].
- Not readable: [@jung2021recurrent], [@jung2024reentry], [@li2023backpropagation], [@guan2025intelligent], [@andaristiyan2026future], [@chen2026highprecision], [@shang2026rapid].

**Searches run:**
- OpenAlex title/abstract: re-entry/reentry/orbital lifetime/orbital decay AND (ML terms), from 2018 and 2022; about 230 records screened in total.
- Crossref: "re-entry prediction machine learning space debris", "orbital lifetime prediction neural network" and "orbital decay prediction deep learning two-line elements", 100 rows each, from 2022.
- arXiv: `ti:"re-entry" AND ti:prediction`; `abs:"orbital lifetime"`; `abs:"orbital decay" AND abs:learning`; `abs:"two-line element" AND abs:learning`; `abs:"re-entry prediction"`; `abs:"reentry prediction"`.
- Semantic Scholar: citations of Jung 2021, Salmaso 2023 and Li 2023.

**Status: Uncertain.** Verified absent in the 9 ML papers whose design could be read. 7 closed or unread ML papers could not be checked, including three from 2025–2026.

**Contribution needed to claim it:**
- A pre-declared temporal cut, for example train on re-entries up to 2019 and test on 2020–2026 (Solar Cycle 25 rise and maximum).
- Report per-object-class errors in the same units as the operational literature: PE_RL in % of residual lifetime [@pardini2018assessing], or (C − O)/O for whole-lifetime prediction [@operti2017assessing].
- Compare against at least one physics baseline run on the same objects and horizons, such as King-Hele/SI-KH [@frey2019extension] or an OSCAR- or STELA-type propagator.

## G2. Distribution-free (conformal, CQR or adaptive conformal) intervals for re-entry epoch or lifetime

**Gap.** Prediction intervals with stated coverage guarantees, and coverage measured out of time, for re-entry epoch or lifetime.

**What exists:**
- **Operational windows are heuristic:**
  - ±20% of residual lifetime gives about 90% confidence; ±25% applies in the last two days; ±30% gives at least 95%.
  - Measured coverage is about 91% overall (29 of 316 windows missed) and about 87% in the last two days (17 of 127 missed) [@pardini2018assessing].
  - The ±20% rule is also recommended in [@pardini2013reentry].
- **Physics Monte Carlo windows** are tested on two objects [@sztamfatergarcia2026multifidelity] or on a single case [@chen2025uncertainty; @chen2025debris].
- **Catalogue-scale probabilistic backtest:** calibration is scored with RMSCE, but the dispersion factor is tuned on the same backtest set, so calibration is in-sample [@shambaugh2026validation] (not peer reviewed).
- **ML density UQ** reports a calibration error score, but on interleaved weekly blocks [@licata2022uncertainty; @licata2022machine; @licata2022msis].
- **Conformal methods exist** but have not been applied here: [@romano2019cqr], [@gibbs2021aci], [@angelopoulos2023gentle], and for censored outcomes [@candes2023conformalized; @gui2024conformalized].

**Searches run:**
- OpenAlex (data strand): 9 queries combining conformal / CQR / quantile regression / prediction interval / coverage with orbit / satellite / debris / conjunction / thermosphere / re-entry / lifetime; about 320 records screened.
- arXiv:
  - `abs:conformal AND (re-entry OR reentry OR "orbital lifetime" OR "orbital decay")`: 6 hits, none relevant.
  - `abs:"conformal prediction" AND (orbit OR orbital)`: 1 hit, not relevant.
  - `abs:"quantile regression" AND (re-entry OR reentry OR "orbital decay" OR "orbital lifetime" OR "collision probability")`: 0 hits.
  - `abs:conformal AND abs:"space debris"`: 0 hits.
- Semantic Scholar: 2 queries, top 20 each.
- Closest SSA conformal work: anomaly detection and collision alerts, with exchangeability assumed [@chen2013conformal].

**Papers checked:** 9 readable ML re-entry/lifetime papers (listed under G1), plus 4 physics UQ papers ([@pardini2018assessing], [@sztamfatergarcia2026multifidelity], [@shambaugh2026validation], [@operti2017assessing]) and 3 density-UQ papers. None uses conformal methods.

**Status: Verified absent in the 16 papers checked and in the indexes listed.**

**Contribution needed:**
- Split-conformal or CQR intervals on the temporal test set of G1.
- Adaptive conformal inference [@gibbs2021aci] for the drift across solar-cycle phases.
- Report empirical coverage and interval width against the ±20% / ±25% / ±30% rules on the same objects.
- Caveat: exchangeability fails under solar-cycle shift. The paper must therefore report coverage per period and per object class, not just overall.

## G3. Time-to-event (censored) formulation of catalogue re-entry

**Gap.** Treat each catalogued object as a survival unit at a landmark date. Objects still in orbit are right-censored, and the model predicts the distribution of the decay date (or the probability of re-entry within a horizon).

**Why this is a gap and not only a reframing:**
- Every readable ML re-entry study trains on decayed objects only. Salmaso et al. use decayed objects with TIP [@salmaso2023machine]; the physics backtests also use deorbited objects [@operti2017assessing; @shambaugh2026validation].
- Lotti et al. include 1,084 non-decayed satellites in training, but as trajectories to fit, not as censored lifetimes [@lotti2025exploring].
- Training on completed lifetimes only is a selection effect. It over-represents short-lived objects in any finite observation window.

**What exists:**
- Survival analysis of satellite *reliability* (on-orbit failure): Kaplan–Meier and Weibull [@castet2009satellite; @grile2025statistical].
- Generic tools: accelerated-failure-time boosting [@barnwal2022survival] and conformalized survival [@candes2023conformalized; @gui2024conformalized].
- In the re-entry literature, "survival analysis" means demise survivability (ORSAT/SARA type), not time-to-event modelling.

**Searches run:**
- OpenAlex: (survival analysis OR survival model OR hazard model OR Cox proportional OR accelerated failure time OR random survival forest OR Kaplan-Meier OR censored) AND (orbital lifetime OR orbital decay OR re-entry OR reentry OR decay date OR space debris OR two-line element). 40 records screened; all relevant hits were demise or reliability.
- OpenAlex: (satellite OR spacecraft OR space object) AND (Kaplan-Meier OR Cox OR Weibull OR survival analysis OR time-to-event) AND (decay OR orbital lifetime OR reentry OR re-entry OR on-orbit). 30 records; reliability only.
- arXiv: `abs:"survival analysis" AND (satellite OR debris) AND (decay OR reentry OR lifetime)`: 0 hits.
- Crossref: "censored survival analysis satellite decay re-entry time-to-event", 60 rows: none.

**Status: Verified absent in the 9 readable ML papers and the 3 physics backtests; no hits in the index searches above.**

**Contribution needed:**
- A landmark design built from public catalogue snapshots, with censoring, evaluated by time-dependent concordance and integrated Brier score.
- Intervals or lower bounds from conformalized survival analysis.
- The project's existing CelesTrak SATCAT snapshot pipeline (see `data/DATASET_REPORT.md`) is already a landmark design of this kind. That is a project fact, not a literature claim.

## G4. Learned (gated) fusion of physics and ML lifetime predictors

**Gap.** An input-dependent gate, or a mixture of experts, that weights a physics lifetime or epoch estimate against an ML estimate. Examples of the physics estimate are King-Hele/SI-KH [@frey2019extension] or a propagator with a TLE-fitted ballistic coefficient. Ideally the gate is calibrated and interpretable: when does physics win, and when does ML?

**What exists:**
- **Analytic EKF fusion** of an ML correction with classical *state* prediction [@peng2021fusion].
- **Identity-initialised NN wrappers** around a differentiable SGP4 (position) [@acciarini2025closing].
- **Residual nets on TLE/SGP4** [@bizalion2023systematic; @lin2026bilstm].
- **Serial physics + ML pipelines:** ML density feeding a propagator [@briden2026atmospheric], and an ML-forecast ballistic coefficient for re-entry [@andaristiyan2026future] (closed, inferred from the title).
- **A PINN for lifetime**, which has not yet generalised [@lotti2025exploring].
- **A physics-informed ML decay analysis** of Starlink [@dyreby2025analyzing].
- **A "physics-gated" NN** for drag correction, not lifetime, in a non-peer-reviewed Zenodo record [@em2026physics]. Watch for naming overlap.

**Searches run:**
- OpenAlex (data strand): gated / gating / mixture of experts / stacking / model fusion AND physics AND orbit / drag / thermosphere / re-entry, from 2019 (218 hits, top 30 screened). Also SGP4 AND ML AND (correction OR residual OR hybrid) (29), differentiable AND SGP4 (6), "orbit prediction" AND ML (65, top 30), and physics-informed/physics-guided AND (thermosphere / drag / decay / re-entry / propagation) (81, top 40).
- arXiv: `(gated OR gating OR "mixture of experts") AND ("orbit prediction" OR "orbital decay" OR re-entry OR reentry OR "orbital lifetime")`: 5 hits, none relevant.
- arXiv: `("physics-informed" OR "physics-guided") AND ("orbital decay" OR "re-entry" OR "orbital lifetime" OR "atmospheric drag")`: 0 hits.
- OpenAlex: (re-entry … OR decay prediction) AND (hybrid OR residual OR correction OR physics-informed OR physics-guided) AND ML, from 2018: 76 hits, top 40 screened.

**Status: Uncertain.** Verified absent among the readable peer-reviewed work (about 15 hybrid papers read at abstract level or better). However, three closed 2026 papers combine physics and ML and must be read before any novelty claim:
- [@andaristiyan2026future];
- [@chen2026highprecision] (a "multi-physics … constrained" GAN);
- [@guan2026orbital] (a simplified perturbation model "fused by" an aerodynamic NN).

**Contribution needed:**
- Show that the gate beats both components and a simple average or stacking under the temporal split of G1.
- Show that it carries calibrated uncertainty into G2.
- Report where the gate prefers physics, for example high eccentricity, sparse TLEs or solar maximum. Salmaso et al.'s Molniya failure (ε_rel up to 2532%) is a concrete example where a data-driven model needs a physics fallback [@salmaso2023machine].

## G5. Breakup-family information as predictors

**Gap.** Fragment lifetime and re-entry models, and conjunction-risk models, have not used information from the fragment's breakup family:
- the parent object;
- the breakup epoch;
- the family's area-to-mass distribution, as learned from siblings that already decayed;
- the family's decay history.

**What exists:**
- **Attribution and reconstruction methods** (they identify families but do not feed predictors) [@dimare2019inorbit; @romano2021puzzle; @ottoboni2026reconstruction; @montaruli2023stochastic; @liu2023determining; @celletti2021reconnecting; @celletti2022proper; @celletti2026perturbative; @wu2023investigation; @ling2026supervised].
- **Large-scale area-to-mass ratios of fragments** [@wacker2025largescale] (metadata only, not read).
- **Labels:** GCAT gives catalogue-level parent labels (type D, "deb X", Parent/SDate) [@mcdowell2020gcat].
- **CDM risk models** use object type or RCS context at most [@olson2026contextual]; in the CAC winning rules, chaser object type [@uriot2022spacecraft].
- **CAC is anonymised.** There is no chaser identity, so family features could only be proxies built from chaser orbit and AMR fields [@uriot2022spacecraft].

**Papers checked:**
- 9 readable ML re-entry/lifetime papers (G1).
- 11 CDM papers: [@uriot2022spacecraft], [@abay2021benchmarking], [@pinto2020towards], [@catulo2023predicting], [@vazquez2026latest], [@olson2026contextual], [@sanchez2024robust], [@tok2026early], [@tulczyjew2021predicting] (abstract), [@ouari2026probabilistic] (abstract), [@mashiku2025nasa] (abstract).
- 10 attribution papers (listed above).

**Searches run:**
- OpenAlex (debris strand): (graph OR GNN OR network) AND (debris family OR fragmentation family OR breakup family OR proper elements). Only proper-element papers came back.
- Crossref: "breakup fragments orbital lifetime decay prediction debris cloud" (60 rows). Debris-cloud propagation and break-up modelling only; no learned predictor using family features.

**Status: Verified absent in the 20 predictive papers checked** (9 re-entry/lifetime ML and 11 CDM). Uncertain for unindexed venues.

**Contribution needed:**
- Hierarchical (partially pooled) family priors for fragment ballistic coefficient and lifetime, with an ablation (with versus without family features) under the temporal split.
- For a fragment from a recent breakup, the earlier-decaying siblings are informative about its area-to-mass ratio. Use them only through information available at the landmark date, to avoid leakage.

## G6. Tabular foundation models for SSA prediction tasks

**Gap.** No use of TabPFN-style models [@hollmann2025accurate] for re-entry, lifetime or CDM risk.

**Searches run:**
- OpenAlex (data strand): (TabPFN OR "tabular foundation model" OR "prior-data fitted network") AND (satellite / orbit / debris / re-entry / thermosphere / spacecraft). 24 hits, all Earth-observation, agriculture or solar-energy.
- arXiv: `abs:TabPFN AND (orbit OR satellite OR debris OR spacecraft)`: 3 hits, none SSA.
- Crossref: "TabPFN satellite orbit space debris" and "tabular foundation model orbit re-entry space": no SSA hit.

**Status: Verified absent in the indexes checked.**

**Caveat:**
- "First to apply model X" is weak novelty on its own.
- It is worth including as a strong small-data baseline for the CAC high-risk subset (150 test positives) and for per-family lifetime models.
- A survey explicitly calls for foundation models for orbit prediction [@lee2026toward].
- Tree ensembles remain the strong default on typical tabular data [@grinsztajn2022tree], including in survival form [@barnwal2022survival].

## G7. CDM risk forecasting with calibrated uncertainty on the official CAC split

**Status: Partly done. The remaining gap is uncertain.**

**Already done:**
- Calibrated *probability* forecasts against a calibrated latest-risk baseline, with log loss and Brier score on pooled cross-validation [@vazquez2026latest] (Zenodo preprint, 2026-10-18, not peer reviewed).
- A claimed "well-calibrated" ensemble on pooled data [@ouari2026probabilistic].
- Bayesian LSTM predictive distributions [@pinto2020towards].
- Bayesian HMM intervals with poor coverage [@catulo2023predicting].
- Uncertainty-aware full-horizon CDM-sequence forecasting [@zerrouki2026deep] (closed, not read).

**Not found:**
- Conformal or distribution-free intervals for the final log-risk on the official test split, with coverage reported (13 papers checked).
- Leave-mission-out evaluation, which Uriot et al. Sec. 4.5 recommend.
- arXiv `abs:conformal AND abs:conjunction AND abs:collision`: 2 hits, none relevant.
- An embargoed MS thesis claims conformal bounds on unspecified CDM data [@rahman2026forecast].

**Constraints any claim must respect:**
- The official metric is L = MSE_HR / F₂, computed after clipping at −6.001 [@uriot2022spacecraft].
- The latest-risk baseline is L = 0.694, and the winner scored 0.556.
- Only about 12 of 97 teams beat the baseline.
- The test set has 150 high-risk events, so effects of a few percent are within noise. Vazquez estimates the minimum detectable effect at about 5–11% of L [@vazquez2026latest].
- Report L, with confidence intervals, alongside any calibration metric.

## G8. Graph models for risk forecasting, attribution or lifetime

**What exists:**
- GNNs only for all-versus-all conjunction screening framed as link prediction [@stevenson2022towards; @chu2024conjunction; @chu2026conjunction; @li2024dynamic; @peri2025graph].
- Network science of the catalogue [@lewis2010new; @romano2024resident; @lin2024risk].

**Status: Verified absent in the 18 works checked** (debris strand). Additional check: arXiv `"graph neural" AND ("space debris" OR "resident space objects" OR conjunction)` returned 62 hits; in the 40 most recent screened, none applies GNNs to risk forecasting, attribution or lifetime.

**Caveat:** a graph method must beat strong tabular baselines on the same split, or it adds complexity without evidence.

## G9. An integrated debris + re-entry + satellite-health model with quantitative evaluation

**What exists:**
- A narrative digital-twin review [@liu2024digital].
- An orbit-prediction-only digital twin [@xu2025digital].
- Unreviewed preprints only for any combination of two components.

**Status: Verified absent among about 130 records screened** (data strand: digital twin, multi-task, and telemetry AND debris/re-entry queries).

**Caveat:** integration by itself is engineering. A publishable claim needs a measured benefit of sharing evidence across components; for example, that health-derived end-of-mission dates improve lifetime prediction for payloads.

## G10. A public, licence-clean, catalogue-wide decay benchmark

**What exists:**
- **Labels are public:**
  - GCAT `DDate`, CC-BY-4.0 [@mcdowell2020gcat];
  - McDowell's re-entry catalogue, with OrbLife, PMLife and end-of-mission orbit [@mcdowell2026reentry] (licence not stated);
  - CelesTrak SATCAT `DECAY_DATE` [@celestrak2026satcat].
- **Orbit histories at scale are not:**
  - they need Space-Track, whose agreement forbids transferring data "including the analysis of data" without approval [@spacetrack2019agreement];
  - or rate-limited manual CelesTrak requests [@celestrak2026archives];
  - the Kaggle mirror's CC0 label conflicts with those terms [@kaggle2026historicaltle].

**Status: Verified absent.** No such benchmark was found, and the data constraints explain why.

**Feasible contribution:**
- A benchmark built only from public catalogue snapshots (current orbit plus physical properties at each landmark date) and public decay labels, released with code.
- TLE-history features can be added as an optional, user-downloaded layer, with code and NORAD IDs released but no TLE data.

## G11. ML intervals benchmarked against operational re-entry windows

**Gap.** No ML paper compares its intervals with:
- the ±20% residual-lifetime rule and its measured coverage of about 91% [@pardini2018assessing];
- TIP or CORDS windows [@aerospace2026cords];
- ESA predictions [@esa2026reentry].

**Status: Verified absent in the 9 readable ML papers.** Uncertain for the 7 closed ones.

**Note:** the CORDS predicted-window table is public in the web list, but its CSV contents were not verified.

## G12. A real-catalogue fragment-to-parent benchmark with public labels

**What exists:**
- **Kelvins "Space debris: the origin" is simulated:**
  - the challenge page says "Background (fictional)";
  - the dynamics are stated exactly;
  - the generation epochs lie on a regular grid;
  - it covers GEO only;
  - its test labels are not public [@efthymiopoulos2021origin; @esa2021kelvinsorigin].
- **Wu and Rosengren** train neural networks on known real TLE families [@wu2023investigation].
- **Other evaluations** use 2–4 real events or synthetic SBM data [@ottoboni2026reconstruction; @celletti2021reconnecting; @ling2026supervised].

**Status: Partly done.** No shared real-catalogue benchmark with a standard metric was found in the 10 attribution papers checked. GCAT parent labels make one feasible [@mcdowell2020gcat].

---

## Pre-emption watch-list (read before claiming novelty)

- **Vazquez 2026** [@vazquez2026latest]: CAC re-evaluation, calibrated probabilities, and "latest message is still the best forecast". It pre-empts generic claims about CAC calibration or beating the latest-risk baseline.
- **Shambaugh 2026a** [@shambaugh2026validation]: a catalogue-scale probabilistic lifetime backtest with CRPS and RMSCE. It is physics-based, calibrated in-sample and not peer reviewed. A project paper must position its out-of-time, distribution-free calibration explicitly against it.
- **Closed 2025–2026 ML or hybrid re-entry papers**, metadata only: [@andaristiyan2026future], [@chen2026highprecision], [@guan2025intelligent], [@guan2026orbital], [@jung2024reentry], [@li2023backpropagation]. They need full-text reading through a library.
- **"Physics-gated" naming:** [@em2026physics] (Zenodo, drag correction).
- **Lotti et al. (EUCASS 2025)** [@lotti2025exploring]: a physics-informed NN for orbital lifetime, still work in progress. A journal version may appear.
