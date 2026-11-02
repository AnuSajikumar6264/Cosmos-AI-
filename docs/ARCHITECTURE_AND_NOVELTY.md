# COSMOS-AI: a digital-twin framework for satellite health, orbital debris and re-entry risk

## 1. Design goal and the constraint that shapes it

COSMOS-AI keeps one **twin** per tracked object: a state that is updated from catalogue data,
telemetry, conjunction messages and space weather, and three modules that read and write it:
satellite health monitoring (H), orbital debris analysis (D) and atmospheric re-entry risk
prediction (R). The modules are coupled where a physical or operational link exists.

The public data do not contain one object followed through all three: telemetry benchmarks
(ESA-ADB, OPSSAT-AD, SMAP/MSL) are anonymised or belong to missions outside the debris
population, conjunction messages (ESA Kelvins) are anonymised, and catalogue data carry no
telemetry. COSMOS-AI therefore does **not** train a joint model over unpaired datasets. Each
module is trained and validated on its own real data; each coupling is a defined function
between module outputs and the twin state, tested on real data where a test exists and
otherwise reported as a demonstration (`cosmos/ai/twin/`, `cosmos/ai/fusion.py`).

## 2. Twin state

For object *i* at time *t*:

    s_i(t) = ( o_i(t), pi_i, h_i(t), F_i(t), e(t), C_i(t) )

* `o_i(t)`: orbit (perigee, apogee, inclination) at its latest epoch ≤ t;
* `pi_i`: design properties (mass, length, diameter, span, shape, mass-uncertainty flag). The
  radar cross-section is shown in the twin record but is not a model input: its availability in
  the catalogue encodes the outcome (`docs/TARGET_DEFINITIONS.md`);
* `h_i(t)`: health evidence — catalogue operational status (payloads) and, where telemetry
  exists, the health module's alarm state;
* `F_i(t)`: debris family (same launch; for fragments, same breakup parent) with sibling
  statistics restricted to what is known at *t*;
* `e(t)`: space weather (trailing F10.7 and Ap, observed values only);
* `C_i(t)`: conjunction summary (CDM sequence) when one is available.

Every module output attached to the twin records its model, its validation status and its own
uncertainty (`cosmos/ai/twin/state.py`).

## 3. Modules

### H — satellite health
Per-channel non-conformity scores `s_c(t)` from a forecaster fitted on nominal data; conformal
p-values against the channel's nominal calibration pool,

    p_c(t) = (1 + #{s' in C_c : s' >= s_c(t)}) / (|C_c| + 1),

Benjamini–Hochberg across channels at each step, persistence and merging into alarm runs
(the pre-registered ESA-ADB study; OPSSAT-AD; SMAP/MSL). Output to the twin: alarm state and,
for catalogue objects, operational status.

### D — orbital debris analysis
(a) *Family evidence* `phi_F(i, t)`: number of siblings catalogued by *t*, number and fraction that
decayed naturally before *t*, and their mean log-lifetime (launch family; breakup family for
fragments).
(b) *Conjunction risk*: quantiles of the final log10 collision probability of an encounter from
its CDMs released ≥ 2 days before closest approach. A persistence expert (latest risk plus the
training distribution of final-minus-latest changes in the latest-risk bin) and learned experts
are fused by the gate of section 4; conformal intervals and a calibrated P(high risk).

### R — re-entry risk
*Physics expert.* Orbit-averaged drag equations for the semi-major axis and eccentricity, with
NRLMSIS 2.1 densities at perigee (trailing F10.7 and Ap held at their values at *t0*) and
ballistic coefficient `B = C_D A/m`:
- for e <= 0.05, King-Hele's Bessel-function series;
- for e > 0.05, the same orbit averages integrated exactly over the eccentric anomaly. The
  truncated series raises the perigee for e above about 0.3.

Area-to-mass is sampled from one of three sources:
- design values (log-normal, 0.25 dex);
- for intact payloads and rocket stages without design values, the training distribution of
  design values of their class;
- for fragments and components without design values, the NASA Standard Breakup Model
  distribution (Johnson et al. 2001), with a characteristic length from the training
  distribution of radar-derived sizes of their class.

Every object has its own random stream. Monte-Carlo samples give the physics predictive
distribution of the capped time to re-entry, with quantiles `q^phys_tau(i)`.

*Learned expert.* Quantile regression `q^ml_tau(x_i)` of `z = log10 min(T, 5 yr)` on
`x_i = (o_i, pi_i, e, phi_F, physics summaries)`.
- The candidates are models published 2022–2026: TabM (ICLR 2025), TabICLv2 (2026) and
  Kumo-Tabular (2026). The one with the lowest tuning-split pinball loss is chosen.
- Gradient-boosted trees (XGBoost, LightGBM, CatBoost), a plain quantile MLP and XGBoost-AFT
  are reference baselines.

*Twin state estimation (objects already in orbit).* The lifetime model is fitted at
canonical-orbit epochs; objects in orbit today have survived since then. The twin therefore
conditions the area-to-mass distribution on each object's catalogue history:
1. prior samples are propagated from the canonical orbit to the as-of date with the realised
   space weather;
2. samples that would have re-entered get weight zero;
3. the others are weighted by a Gaussian likelihood of the current perigee and apogee;
4. 32 samples are drawn by resampling.

The physics expert then forecasts from the current orbit with these samples
(`physics.history_posterior`). Tested in the as-of-date test T1.

## 4. Fusion and uncertainty

*Reliability-gated quantile averaging.* For object *i* with reliability descriptors `r_i` of the
physics expert (area-to-mass source: design value, class design prior or class breakup prior;
spread of its own Monte-Carlo distribution, eccentricity, perigee band, object class, physics
survival probability):

    q_tau(i) = g(r_i) q^phys_tau(i) + (1 - g(r_i)) q^ml_tau(i),   g(r) = sigmoid(w^T r + b).

A convex combination of two monotone quantile functions is a monotone quantile function, so the
output is a valid predictive distribution. `(w, b)` minimise the mean pinball loss on
out-of-fold predictions of the training period (L2-penalised). Baselines that the comparison
includes: `g = 1` (physics), `g = 0` (learned), `g = 1/2`, constant `g`.

*Mondrian conformal calibration.* On a calibration split disjoint from fitting and gating, the
score `S = max(q_lo - z, z - q_hi)` is computed and its `ceil((1-alpha)(n_k+1))`-th smallest value
`Q_k` in group *k* (object class × perigee band, with fallback to class, then pooled) widens the
interval to `[q_lo - Q_k, q_hi + Q_k]`. Horizon probabilities `P(T <= H)` are read from the fused
quantile function and calibrated by isotonic regression on the calibration split. Rank metrics
use the raw probability, because isotonic calibration ties scores. Quantiles within 1e-6 of the
5-year cap are set exactly to the cap, which is a point mass of the target.

## 5. Couplings (implemented in `cosmos/ai/twin/couplings.py` and `cosmos/ai/twin/engine.py`)

| Coupling | Definition | Test on real data |
|---|---|---|
| C1 debris → re-entry | `phi_F` enters `x_i` | ablation of the family features in the re-entry study (two temporal test folds) |
| C2 re-entry → debris environment | expected number of a breakup cloud's fragments in orbit at time *t*: `E[N(t)] = sum_{i : t0_i <= t} S_i(t - t0_i)` with `S_i` the fused survival function, read raw and calibrated (isotonic maps of `P(T <= a)` on tune-B fragments at 25 ages) | predicted against catalogue counts for every breakup cloud with ≥ 30 fragments in the test folds |
| C3 health → re-entry | regime `rho_i`: operator-controlled for payloads reported operational, natural decay for non-operational payloads, rocket stages, components and fragments, unknown for payloads with no status. In the operator-controlled regime the drag forecast is withheld and the previous-window payload base rate is used | as-of-date test on a previously analysed window: CelesTrak operational status at 2026-08-16, decays to 2026-10-27, Earth-orbiting payloads below 600 km. The policies compared use only information at 2026-08-16. Platt recalibrations fitted on the follow-up window are reported only as an in-window diagnostic. Launch-cluster CIs |
| T1 twin state → re-entry (all classes) | history-conditioned physics for fragments, rocket stages and components; C3 policies for payloads | as-of-date test: twin as of 2026-08-16 (SATCAT snapshot), decays to 2026-10-27, 13,186 uncontrolled objects plus all payloads. Compared with the lifetime pipeline and class-prior physics. The window had been used by an earlier landmark benchmark and its decay count was seen during the diagnosis, so the test is not blind (disclosed) |
| C4 health × debris → response | a high-risk conjunction for a payload with an active health alarm or non-operational status is escalated (manoeuvre capability uncertain) | none possible with public data: demonstration only |

## 6. What is and is not new

Not claimed as new by itself: gradient-boosted trees, MLPs, foundation models, King-Hele decay,
NRLMSIS, the breakup model, conformal prediction, quantile averaging or gating. The claims to be
tested (checked against the literature review in `docs/RELATED_WORK.md` and
`docs/RESEARCH_GAP_AND_NOVELTY.md`) are:

1. a digital-twin framework in which the three modules share one uncertainty principle
   (conformal calibration of alarms, intervals and probabilities) and are coupled by explicit
   functions, each tested on real data or labelled untested;
2. catalogue-wide re-entry prediction with a leakage-controlled temporal protocol (as-of
   censoring, two independent test periods, 64,516 objects) and calibrated intervals;
3. reliability-gated fusion of a Monte-Carlo physics distribution with learned quantiles;
4. causal debris-family evidence for re-entry prediction and its use for breakup-cloud survival
   forecasting;
5. operational status as health evidence that selects the re-entry regime.

Whether each component improves on its simpler alternative is decided by the measurements in
`docs/RESULTS_AND_ANALYSIS.md`, not by this document.
