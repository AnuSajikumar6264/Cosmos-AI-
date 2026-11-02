# Research gap and novelty

Sources: `docs/RELATED_WORK.md` (157 verified references in `docs/references_reentry.bib`) and the
gap analysis `docs/LIT_NOTES_GAP.md`. "Verified absent" means absent in the papers that could be
read in full; closed papers that were checked by metadata only are listed at the end and must
be read before submission.

## The gap this work addresses

1. **Integrated health + debris + re-entry with quantitative evaluation (G9).** Verified absent
   among about 130 screened records: there are narrative digital-twin reviews and an
   orbit-prediction-only twin, but no system that couples the three functions and measures what
   the coupling buys. Integration alone is engineering; the claim here is restricted to couplings
   whose effect is measured on real data (C1–C3) and labels the rest (C4) as demonstration.
2. **Catalogue-wide, censoring-aware, forward-in-time re-entry prediction with distribution-free
   intervals (G1 + G2 + G3).** No paper found combines these. Conformal intervals for re-entry or
   lifetime: verified absent in 16 papers. Censoring-aware formulation: verified absent (the
   readable ML studies train on decayed objects only). Forward-in-time, catalogue-wide ML:
   absent in the 9 readable ML papers; 7 closed papers unchecked.
3. **Breakup-family evidence as predictors (G5).** Verified absent in 20 predictive papers.
4. **Comparison of ML intervals with the operational ±20 % residual-lifetime rule (G11)**,
   whose measured coverage in IADC campaigns is about 91 % (Pardini & Anselmo 2018).
5. **A licence-clean, catalogue-wide decay benchmark built from public data only (G10)**: GCAT
   (CC-BY-4.0) and CelesTrak, released with code; no TLE data are redistributed.

## What is not claimed

* Beating the latest-risk baseline on the ESA Kelvins challenge: a pre-registered 2026
  re-evaluation (Vazquez 2026, not peer reviewed) already finds that no model does so with
  confidence and reports calibrated probabilities. The conjunction module is reported as one
  module of the twin, with its result stated whatever it is.
* Tabular foundation models as a contribution in themselves (G6: absent but weak).
* Novelty of gradient-boosted trees, MLPs, TabICL, King-Hele decay, NRLMSIS, the breakup model,
  conformal prediction or gating as individual techniques.
* The name "physics-gated" is already used by a 2026 Zenodo record (drag correction); this work
  uses "reliability-gated".

## Closest work and how this work differs

| Work | What it does | Difference |
|---|---|---|
| Salmaso, Trisolini & Colombo 2023 | GRU re-entry epoch over the last 200→80 km, 417 objects with Space-Track TIP data, 15 IADC test objects | point predictions; final-days regime; Space-Track data; here: catalogue-wide (64,516 objects), months-to-years horizon, intervals, public data |
| Pardini & Anselmo 2018 | assessment of 316 IADC predictions; ±20 % window ≈ 90 % coverage | operational physics predictions near re-entry; used here as the interval benchmark |
| Operti et al. 2017 | OSCAR lifetimes against 981 observed lifetimes | physics only, no learning, no intervals |
| Shambaugh 2026 (arXiv) | physics lifetime backtest, 1,283 objects, CRPS, calibration tuned on the backtest | physics-based, in-sample calibration; here: out-of-time conformal calibration and learned experts |
| Vazquez 2026 (Zenodo) | pre-registered CAC re-evaluation | pre-empts CAC calibration claims; here CAC serves the twin's debris module |
| Licata & Mehta 2022 | calibrated density uncertainty | density, interleaved splits; here lifetime, forward splits |

## Must be read in full before claiming novelty (metadata only so far)

Jung 2021 and 2024; Li 2023; Guan 2025 and 2026; Andaristiyan 2026; Chen 2026 (physics-constrained
GAN for re-entry, Aerospace Science and Technology); Zerrouki 2026; Lotti et al. 2025 (journal
version may appear).

## Status of each claim after the experiments

Status as of 2026-10-27, from `docs/RESULTS_AND_ANALYSIS.md`. Where the measurements do not
support a claim, it is narrowed or withdrawn here.

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Integrated twin with couplings measured on real data | **Supported, narrowed** | C1: family evidence helps the selected learner in one of two periods. C2: the best readings are accurate to about a tenth of a cloud, but with the cloud bootstrap the experts cannot be ranked (calibrated reading added post hoc). C3: status-aware policy, gain not significant in one as-of-date window. T1: history-conditioned twin forecasts cut the over-prediction of 72-day re-entries several-fold (see `RESULTS_AND_ANALYSIS.md` for the numbers); the window had been analysed earlier in the project, so T1 is not blind. C4: demonstration only. |
| 2 | Catalogue-wide, censoring-aware, forward-in-time re-entry prediction with distribution-free intervals | **Supported as a benchmark and protocol; coverage not guaranteed under shift** | Two temporal test periods, 64,516 objects. Mondrian conformal 90 % coverage ranges from 0.79 to 0.98 across models and periods: below nominal for TabM and the gated model at 2016, because the calibration cohort is 5–7 years older than the test cohort. |
| 3 | Reliability-gated fusion of physics and learned experts | **Withdrawn as a superiority claim** | The gate significantly improved every weighted learner at 2011 and was neutral to slightly harmful at 2016. It never beat the best single expert. At 2011 it is tied with physics and the foundation models; at 2016 it is significantly worse than TabICLv2 and Kumo-Tabular. The gate-on-every-learner analysis is post hoc. |
| 4 | Breakup-family evidence as a predictor | **Partly supported** | Significant gain for TabM at 2016; not significant for Kumo-Tabular at 2011. |
| 5 | Comparison with the operational ±20 % residual-lifetime rule | **Supported** | At month-to-year horizons the ±20 % window covers only a small share of objects that decayed (about 17 % at 2011); conformal intervals reach nominal coverage. The rule's 91 % coverage refers to the final days before re-entry. |
| 6 | Licence-clean benchmark (GCAT + CelesTrak) | **Supported** | Released with checksummed manifests. No Space-Track data. |
| new | Outcome leak through catalogue covariate availability (RCS) | **Documented** | Adjusted odds ratio 3.6 for 30-day decay (cluster-robust 95 % CI 2.1–6.2). Its effect on test pinball loss is within noise at both origins; it breaks interval coverage at 2016. Not claimed as a methodological novelty; it is a caution. |
| new | History-conditioned area-to-mass for current-state forecasts | **Supported in one as-of-date window (not blind)** | A coarse form of established ballistic-coefficient estimation; claimed as a twin component, not as a new method. |
