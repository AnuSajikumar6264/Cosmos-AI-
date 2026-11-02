# Adversarial code review of 2026-10-27 and the fixes applied

## How the review worked

- **Specialist reviewers.** Each examined the new modules through one lens: leakage,
  statistics, physics, the twin, or conjunction and telemetry.
- **Skeptics.** Every finding then went to two independent skeptics, who tried to refute it by
  reading the code and re-running checks on the data.

**Counts.** 38 findings: 34 confirmed by both skeptics, 3 disputed (split verdict), 1 refuted.
Some findings duplicate one another across lenses, so the confirmed set comes to about 27
distinct defects. The raw review output is in
`/data/anupa/COSMOS_work/.scratch/review_findings.json`.

**Superseded results.**
- The results that the review invalidated are kept, unused, under
  `outputs/benchmarks/*/_superseded_pre_review_2026-10-27/`, each with a README.
- A copy of the pre-fix sources is in `/data/anupa/COSMOS_work/.scratch/pre_fix_2026-10-27/`
  (a git repository). It was taken after the first edit to `fusion.py` (the snapping helper).

Finding numbers below are indices into the confirmed list.

## Re-entry module (`cosmos/ai/lifetime/`)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 0 | critical | Radar cross-section (RCS) from the 2026 SATCAT. Whether a value exists depends on the outcome, and it entered the learners (`log_rcs`), the physics area-to-mass source (`rcs_sbm`) and the gate (`src == rcs_sbm`). | RCS removed from the main feature set (`run.FULL`), from the physics source choice and from the gate descriptors. It remains only in the class-level size prior (training objects only) and in a labelled leak demonstration `<best>_plus_rcs_2026`. |
| 3 | minor | The learner cache was keyed by tag only, and stale caches were silently re-aligned. | The cache is keyed by an md5 fingerprint of the object keys, labels, feature matrices and settings, and the re-alignment path is removed. All caches from before the review were moved aside. |
| 5 | major | Float32 values at the 5-year cap were not snapped to the cap, which broke raw coverage and P(T ≤ 5 yr). | `models.snap_bounds` sets values within 1e-6 of a bound to the bound. It is applied in every learner's output, the physics quantiles, every fusion and inside `MondrianCQR`. |
| 6 | major | Horizon AUC-PR and AUC-ROC were computed on tied isotonic outputs. | Rank metrics use the raw probability; the isotonic output is used only for Brier and ECE. |
| 11 | major | King-Hele's small-eccentricity series was used up to e = 0.95, where the perigee rose under drag. | For e > 0.05 the orbit averages are integrated exactly over the eccentric anomaly (257-node trapezoid, relative error below 1e-12 up to c = 3000). A raised perigee now raises an error. |
| 12 | major | The breakup-fragment A/m law was applied to intact payloads without design data. | Payloads and rocket stages without design values draw A/m from the training distribution of design A/m for their class; the breakup law is used only for fragments and components. |
| 13 | minor | Physics Monte-Carlo draws depended on which other objects were in the batch. | Each object has its own generator, seeded with (seed, md5 of its key), and pool values are drawn through the empirical quantile function. Batch independence was checked: the maximum difference is exactly 0. |
| disputed | minor | Space-weather windows included the whole day of t0. | The windows now end on the last complete day before t0, in the cohort, C3 and the engine. |

Further changes made for consistency:
- A gate descriptor that is constant on the fitting set gets scale 1, not 1e-9 (otherwise it
  would blow up at prediction time).
- Tune-B predictions are saved (`tuneB_predictions.npz`) for the C2 survival calibration.

## Digital twin (`cosmos/ai/twin/`)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1, 20 | major | The C3 Platt recalibration was fitted on outcomes from the follow-up window (look-ahead) but reported as prospective. | The prospective policies use only information available at tau: physics, the previous-window base rate, and the status-aware twin. The Platt maps are reported separately as an *in-window, cross-fitted* diagnostic. |
| 2, 8, 14, 16 | major | C3 folds used Python's salted `hash()`, so results changed from run to run. | md5 folds, as everywhere else in the project. |
| 22 | minor | The C3 bootstrap resampled payloads i.i.d., although outcomes cluster by launch. | Launch-cluster bootstrap (`run.cluster_bootstrap`). |
| 24 | minor | The C3 cohort was not restricted to Earth-centred orbits. | `ORBIT_CENTER == "EA"` filter added. |
| 15 | critical | The engine crashed in `current_state` (`fillna` was given a NumPy array). | Fixed; the fallback is a Series aligned to the frame. |
| 7, 17 | major | Twin probabilities were raw quantile read-offs with a 2.5–4 % floor, which inflated expected re-entries. | Per-horizon isotonic maps are fitted on the as-of tune-B split and applied to uncontrolled objects. Payloads use the physics Monte-Carlo share, which is the reading tested in C3. |
| 18 | major | The C2 survival read-off biased predicted counts low, by a different amount for each model. | C2 now also reports a *calibrated* reading: isotonic maps of P(T ≤ a) on tune-B fragments at 25 ages, interpolated in log-age. Both readings are reported. |
| 19 | major | The deployed re-entry module differed from the validated one but was labelled "validated-real". | The engine reproduces the validated configuration for uncontrolled objects: uncontrolled population, 5-seed ensemble, train-OOF gate, tune-B CQR and isotonic maps. Payloads get the physics expert only. Every record carries the validation that actually supports it, including the domain-shift caveat. |
| 21 | minor | C3 was loaded but never applied, and the C2 step was missing. | Operational payloads get the C3 status-aware policy (previous-window base rate by perigee band; drag forecast withheld). A C2 cloud table (`c2_clouds.csv`) is written. |
| 23 | minor | `mass_uncertain` was forced to False when serving. | Taken from GCAT `MassFlag`, as in training. |
| disputed | minor | Unknown-status payloads were counted in the natural-decay regime. | Reported as a separate `unknown` regime with their own expected-re-entry sum. |

## Collision risk (`cosmos/ai/conjunction/`) and SMAP/MSL telemetry (`cosmos/ai/telemetry/smap_msl.py`)

These modules were fixed in a separate workflow: one fixer per module, two adversarial reviewers
per fix, and repair rounds. The findings concerned were:
- conjunction: 10, 26, 30 and the disputed AUC finding;
- telemetry: 4, 9, 25, 27, 28, 29, 31, 32, 33.

Both modules passed with every finding verified fixed, after one repair round for conjunction
(none needed) and two for telemetry. The full reports are in
`/data/anupa/COSMOS_work/.scratch/fix_workflow_result.json`.

**Collision risk (`cosmos/ai/conjunction/run.py`)**

| # | Fix |
|---|---|
| 10 + disputed | AUC-PR and AUC-ROC of P(high) are computed on the raw P(high). Brier, ECE, the decision threshold and gated persistence use the isotonic-calibrated value. |
| 26 | Every candidate (persistence, learners, fusions; OOF and test) is sorted and snapped to [-30, 0] before anything is fitted or scored. The interval endpoints are snapped too. The apparent coverage loss from gating was mostly this rounding artefact: in the smoke run, 80 % coverage was 0.784 before the fix and 0.845 after. |
| 30 | Nested selection: in every outer fold the same configuration draws are searched on the other folds. TabICLv2 and Kumo-Tabular (no search) use all of the other folds as context. Per-fold selections are recorded in `summary.json`. A permuted-label "poison" run confirmed that no test label is used before scoring. |
| verifier note | The snap tolerance was raised to 1e-4 in log10 risk, because the float32 spacing at -30 (1.9e-6) exceeds 1e-6. |

**SMAP/MSL telemetry (`cosmos/ai/telemetry/smap_msl.py`)**

| # | Fix |
|---|---|
| 33 | Label rows are merged by channel: 81 channels, 104 sequences (P-2 had been counted twice). |
| 4, 31 | Out-of-fold calibration: 5 contiguous blocks with a 30-window purge, and test scores from the full refit (cross-conformal approximation). |
| 9, 27 | alpha_eff = max(1e-3, 1/(n_cal+1)) per channel. The 9 channels with alpha raised are reported, and results are also given on the channels where alpha is attainable. |
| 25 | A false positive is a maximal alarm segment outside every sequence. The ESA-ADB corrected event-wise precision and the nominal alarm fraction are reported. An always-on reference is reported, and surrogate references show what event scores trivial alarms reach (`_swap` is the null for channel-specific detection; `_late` is a late-block reference). |
| 28 | Exact SVD for PCA (deterministic). |
| 29 | All point-wise metrics are on the steps telemanom scores. The primary AUC-PR is the mean over channels. The pooled AUC-PR uses conformal p-values; telemanom's raw pooled value is in its own column. |
| 32 | Causal alarms (third consecutive hot step, no backdating). |

The reviewers also found a withdrawn claim. That a detector beats the circular-shift surrogate is
not evidence of skill: 93.7 % of labelled steps lie in the second half of their channel's test
series, so late alarms score well without detecting anything. Skill is read against the `_swap`
null and the `_late` reference.

## Found after the review: domain shift of the re-entry module on current orbits

Smoke-testing the corrected twin engine as of 2026-10-27 produced too many re-entries. The
engine used the validated lifetime pipeline, with tune-B isotonic calibration, for fragments,
rocket stages and components. It expected 139 of these objects to re-enter in 30 days. The
catalogue shows 30–34 decays of these classes in each of the three preceding 30-day windows.
The physics expert with its class priors alone expected 76.

**Cause.**
- The lifetime study predicts from the canonical-orbit epoch.
- An object still in orbit years later has survived since that epoch.
- Survival selects low area-to-mass ratios, which neither the class priors nor the learned
  experts (fitted at canonical-orbit epochs) know about.

Review finding #19 had flagged this as a labelling issue; it is also a quantitative failure.

**Change (design fixed before any test of it).** The twin conditions each uncontrolled object's
area-to-mass ratio on its catalogue history (`physics.history_posterior`):
1. 128 prior samples are propagated from the GCAT canonical orbit to the as-of date with the
   realised trailing space weather.
2. Samples that would have re-entered get weight 0.
3. The others are weighted by a Gaussian likelihood of the current perigee and apogee, with
   σ = 10 km + 0.3 × the observed change.
4. Fallbacks: survival alone if the effective sample size is below 4; the 8 longest survivors
   if no sample survives.
5. 32 samples are drawn by systematic resampling.

This is a coarse, two-orbit version of the ballistic-coefficient estimation from orbital
histories used in re-entry practice (Gondelach et al. 2017; Sang et al. 2013; Pardini & Anselmo
2008). It is not claimed as new.

**Test and disclosure.** The three variants are tested prospectively in T1
(`cosmos/ai/twin/prospective.py`): the lifetime pipeline, physics with class priors, and
history-conditioned physics. The test runs as of 2026-08-16 with outcomes to 2026-10-27.

The aggregate number of decays in that window was seen during the diagnosis above, before the
variant was designed. The variant's settings were fixed before it was evaluated on the window,
and no variant was changed afterwards. T1 is therefore pre-specified but not fully blind. The
dated forecast of 2026-10-27, hashed in `outputs/twin/as_of_2026-10-27/FORECAST_SHA256.txt`, is
the fully blind test, to be scored against a later snapshot.

## Deviation: learner set restricted to models published 2022–2026 (user request, 2026-10-27)

After the first full re-entry run (run 1), the user asked that the models be the latest ones
(2022–2026).

**Change.** The learned-expert candidates are now TabM (Gorishniy et al., ICLR 2025), TabICLv2
(2026) and Kumo-Tabular (NVIDIA, 2026). The expert is still the candidate with the lowest
tune-A pinball loss. XGBoost, LightGBM, CatBoost and the plain quantile MLP are still fitted
and reported, as classic reference baselines that are not eligible as the expert.

**Models considered and not used:**
- RealMLP (NeurIPS 2024): its package has no quantile loss.
- TabPFN-3.5 (2026): its weights are licensed for non-commercial use only.

**What had been seen.**
- Run 1's test results were known for both origins when run 2 started, and the Kelvins
  collision-risk test set had been scored with run 1's learner pool.
- Smoke tests of the new learners printed two test-set scores before run 2: Kumo-Tabular's
  origin-2016 pinball (0.051) and TabM's origin-2011 pinball (0.0729).
- Before the first review, every benchmark had been scored once on its test set (archived under
  `_superseded_pre_review_2026-10-27/`).
- The T1/C3 window was not fresh: the earlier landmark benchmark (commit 2525367,
  `cosmos/ai/survival/landmark.py`, TAU 2026-08-16, FOLLOW 2026-10-23) fitted and tested physics,
  logistic regression, XGBoost, XGBoost-AFT and hazard-MLP classifiers on it, and its aggregate
  decay count was seen during the diagnosis of the domain shift.

No setting was chosen with this information: the selection rule, budget, seeds, features, gate
and calibration are unchanged, and cached learners are reused through their fingerprints. The
manuscript states all of the above in its "Deviations and prior use of test data" paragraph and
no longer says that test sets were scored once.

**Records.** Run 1 is kept in `outputs/benchmarks/reentry_lifetime/origin<O>_uncontrolled/run1_learner_set_v1/`
and is reported alongside run 2.

**Engineering constraints of the new learners (no effect on the protocol):**
- Kumo-Tabular runs on an Intel XPU.
- Its attention is split along the query dimension, which is exact: the maximum difference was
  4.8e-7 in a forced-chunking test.
- Its context is capped at 16,000 training rows, drawn at random per seed, because a 25k-row
  context exceeds a 24 GB card. TabICLv2 uses the full context, on CPU.
- TabM runs on an XPU in full precision.

## Final adversarial review of the manuscript and code (2026-10-27, second pass)

Four finders (claims, statistics, leakage and disclosure, code) and one or two skeptics per
finding. The verified findings and what was done:

| Finding | Verdict | Fix |
|---|---|---|
| Twin engine and T1 gave payloads without design values the fragment breakup A/m law (the class pools were built from D/R/C training rows only) | confirmed, major | `engine.fit_as_of` keeps the validated D/R/C pools (`class_prior_from(train)`) and adds only the payload design-A/m pool from payloads of the same training period. An interim fix that rebuilt every pool from all rows (changing the shared L_c pool from 25,319 to 29,672 values, including 16 rows with undefined outcomes) was caught by the review and replaced before its results were used. Twin forecast and T1 rerun; the superseded outputs are in `_superseded_payload_prior/` |
| Headline "no expert best in both periods; physics led 2011–2015" rests on non-significant differences | confirmed, major | Post-hoc pairwise comparisons added (`diagnostics.pairwise_effects`, Supplementary Table S16). Text now says: foundation models significantly best at 2016; physics, foundation models and gated model tied at 2011 |
| RCS leak "flatters 2011 / worsens 2016" pinball claims are within noise and not reproduced with run 1 | confirmed, major | Leak text now reports the pinball changes with their CIs as within noise; the coverage collapse at 2016 and the association itself are the robust effects |
| Learner-pool change after run-1 test results not disclosed as such; test sets described as "scored once" | confirmed, major | Deviations paragraph in the protocol; all "scored once" statements in the manuscript, docs and docstrings corrected |
| T1/C3 window was the earlier landmark benchmark window | confirmed, major (disclosure) | Disclosed; "prospective" replaced by "as-of-date test on a previously analysed window" |
| C2 expert ranking has no uncertainty | confirmed, major | Cloud bootstrap (`couplings.c2_bootstrap`, Table S17); ranking claims withdrawn; calibrated reading labelled post hoc |
| Gate-on-every-learner analysis presented without a post-hoc label or multiplicity note | partly refuted (the joint "every learner" claim needs no correction) | Labelled post hoc; nominal-interval statement added |
| RCS odds-ratio CI ignored launch-family clustering | confirmed, minor | Cluster-robust SEs (OR 3.59, 95 % CI 2.09–6.17) |
| T1 rank metrics computed on tied isotonic outputs | confirmed, minor | Rank metrics on raw probabilities (`p_lifetime_raw_*`); AUC-ROC reported next to AUC-PR |
| SMAP/MSL late reference not reported; Data section had pre-merge counts | confirmed, minor | Late reference reported (it beats the detector in 7 of 8 pairs); counts from the result files |
| Health module's ESA-ADB/OPSSAT-AD validation described but not reported | confirmed, minor | Companion ESA-ADB results reported (Mission 1 gain, Mission 2 reversal); OPSSAT-AD no longer presented as part of this paper |
| "Significantly better than the GBDT baselines" false for CatBoost at 2011 | confirmed, minor | Narrowed; CatBoost row added |
| Debris protocol text out of date (snap tolerance, untuned Kumo-Tabular) | confirmed, minor | Text corrected; the selection advantage of the untuned models is stated |
| "Still minutes" for a 30,000-object catalogue false for TabICLv2 | confirmed, minor | About two hours per seed on CPU stated |
| Lotti et al. "only decayed objects" | refuted (the primary source confirms it) | Only the object count was corrected ("a few hundred to about 1,800") |

Also added after this review (post hoc): head-to-head pinball comparisons on the Kelvins test set
(`conjunction.run.pinball_pairs`, Table S18).

## Figure review (2026-10-28): dataset, output and benchmark figures

The manuscript was extended from 8 to 19 figures plus 3 supplementary figures: views of each
dataset (Figures 1–3), per-object model outputs (6, 11, 16, 17), head-to-head benchmarks (7, 15)
and the dated forecast (18). Seven reviewers (one per figure group, plus one on manuscript
integration) checked every figure against the data, its caption and every sentence citing it;
each finding was then checked by a skeptic. 42 findings were confirmed and 2 refuted. The ones
that changed a number, an image or a claim:

| Finding | Verdict | Fix |
|---|---|---|
| `near_zero_spread` used `s_tr.std()`, which is NaN for Chronos-2 (its first 64 context steps have no score), so the flag read False on all 81 channels | confirmed, major (code) | `np.nanstd`; 16 channels now flagged. SMAP/MSL rerun; only the flag and the subset columns derived from it changed. Repeating the run also moved Chronos-2's per-channel AUC-PR by at most 2e-5 (GPU non-determinism), leaving every reported value unchanged at three decimals. Previous outputs in `_superseded_nanstd_flag/` |
| Kelvins final-risk histogram: bins centred on integers, so the high-risk colour and the threshold line sat half a bin low (115 bars coloured against 66 high-risk events) | confirmed, major | Integer bin edges with one edge at −6, colour by left edge, line at −6, and an assertion that the coloured bars sum to the high-risk count |
| Figure 1 drew both test periods as origin + 5.83 yr, so the origin-2011 test bar overlapped the origin-2016 one | confirmed, major | Boundaries read from `run.ORIGINS`; the 2011 test cohort ends 2016-01-01 |
| Cost figure: "fitting time per seed" is really the fit-and-predict loop (the seed-0 fit is not timed), and the per-1,000-object rate is one call dominated by a fixed context encoding, so the "two hours" / "quarter of an hour" catalogue extrapolations were unsupported | confirmed, major | Panels relabelled to the measured quantities; extrapolation removed; the ratio stated as three to five orders of magnitude; the physics-feature cost (which every learner needs) added |
| Caption blamed the −30 risk floor for the collision-risk model's near-zero F2 | confirmed, major | 129 of 150 high-risk events get a median below −6, 41 of them between −10 and −6; F2 = 0.17 stated instead of "near zero" |
| Caption and in-figure note blamed C-2's continuous Chronos-2 alarm on a short training series; the cause is a constant training series | confirmed, major | Reworded to the real cause; noted as the most extreme of the 27 MSL channels |
| "At 2011 the foundation models, the physics expert and the gated model are statistically tied" — Kumo-Tabular is significantly worse than the gated model that fuses it | confirmed, major | Narrowed in the results, the discussion and the conclusions |
| The operational-window paragraph cited Figure 8 for decayed-only coverage the figure does not show, and attributed the shortfall to long horizons | confirmed, major | Cites Table S5; the shortfall is attributed to forecasting from a canonical orbit, not to the horizon; the figure's axis now says it covers all test objects |
| Two rows of the dated forecast have a degenerate interval at exactly 10 days, set by the integrator's maximum step rather than by physics | confirmed, major | Disclosed in the caption and in the limitations |
| Figure 18's grey series was labelled "class prior" but is a design value for most objects, and panel (b) did not come from the samples in panel (a) | confirmed, minor | Forecasts recomputed from the plotted draws; legend and caption corrected |
| Example-channel rule ("AUC-PR closest to the spacecraft median") ties exactly whenever the channel count is even | confirmed, minor | Replaced by the median channel, taking the lower of the two middle channels; derived in code, not hard-coded |
| Architecture figure listed OPSSAT-AD, which the paper no longer reports | confirmed, minor | Removed from both boxes |
| Figures were not numbered in order of first citation | confirmed, minor | `FIGURE_ORDER` and `SUPP_ORDER` reordered; the build asserts the file number equals the displayed number, and the citation sequence is now monotone |
| Occlusion and label collisions: legend over the −30 floor, overplotted classes and regimes, colliding labels in S1 and S3, overprinting axis labels in Figure 8, stacked markers hiding two variants at the origin of Figure 14 | confirmed, minor | Legend moved, largest class/regime drawn first with the overlap disclosed, leader lines added, one shared axis label and one shared legend, open markers in Figure 14 |
| Figure 1(c) showed 848 zero lifetimes at the axis edge, and called all censoring "ended by control" | confirmed, minor | Annotated, and the label now says "phase ended without natural decay" |
| Caption said the 75 plotted conjunction events were "drawn at random" | confirmed, minor | Disclosed as 60 low-risk and 15 high-risk drawn at random, so high-risk events are oversampled |

Two findings were refuted: that panel (a) of the collision-risk figure mislabels the latest-risk
input as "persistence", and that the dated-forecast caption misstates which model supplies three
of its rows.

**Reproducibility of the figures.** `make_data_figures.py` recomputes the detector scores of two
SMAP/MSL channels and the area-to-mass posteriors of three objects, and asserts that they
reproduce the stored benchmark event counts and forecasts before drawing them. Every example is
chosen by a seed or by a rule stated in the code and the caption.
