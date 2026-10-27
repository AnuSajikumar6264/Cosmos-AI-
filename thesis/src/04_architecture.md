# Current System Architecture {#sec:architecture}

This chapter documents the architecture as implemented at commit `2525367`. It was established in three ways:

- by reading every module;
- by running the end-to-end pipeline with fixed seeds;
- by comparing the original package with the snapshot taken before the machine-learning layer was added (`outputs/baseline/cosmos_baseline_src_2026-10-24.tar.gz`).

Every original module except `cosmos/pipeline.py` is byte-identical to that snapshot. `pipeline.py` gained two opt-in switches. Constants are quoted with the file and line where they are defined. Numbers produced by running the code come from the evidence file (Appendix E).

## Overview and design principles

COSMOS consists of two layers ([@fig:overall]).

- **The original package `cosmos`.** It holds {{code.original.files}} Python files and {{code.original.lines:int}} lines, in three feature subpackages (`health`, `debris`, `reentry`) and shared modules (`common.py`, `pipeline.py`, `plots.py`, `datasets.py`).
- **The machine-learning layer `cosmos.ai`.** It holds {{code.ai.files}} files and {{code.ai.lines:int}} lines, excluding the vendored official ESA-ADB metric code.

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
The severity $s(t)$ either ramps as $\mathrm{linspace}(0,1)^{1.3}$ or steps to 1. The gains $g_k$ and noise multiplier $\nu$ are set per fault type (`FAULTS`, l.45–52): degradation, leak, overheat, stuck, dropout and saturation. The `stuck` fault has $\nu = 0.6$, so $\nu - 1 < 0$. The rerun recorded the resulting error: {{orig.stuck.error}}. This is a known defect (Appendix C).

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
- **`yolo.py`.** It contains a small single-scale YOLO-style detector (TinyYOLO, {{syn.tinyyolo.params.mean:int}} parameters on a stride-8 grid) and an adapter for the Ultralytics library. Its `map50` is computed by centre-distance matching, class-agnostically and without interpolation, so it is not a COCO mAP@0.5 (Appendix C).
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
- **Default run.** With the default arguments (no fault, seed 7), the run reports a final anomaly score of {{orig.default.anomaly_score_final}}, health band {{orig.default.health_band}}, and a break-up of type "{{orig.default.break_up_type}}" at {{orig.default.break_up_altitude_km}} km. It generates {{orig.default.fragments_generated}} fragments, propagates {{orig.default.impacts_propagated}} impacts with CEP50 = {{orig.default.cep50_km}} km, and reports risk band {{orig.default.risk_band}}. The values come from `thesis/tools/run_original_pipeline.py`, package version {{orig.version}}.

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
| US76 atmosphere, drag limits, geodesy, propagator, break-up mass closure, registration sign, conformal zone coverage | implemented, tested | {{tests.verify.passed}}/{{tests.verify.total}} checks in `tests/test_verify.py`, {{verify.n_external}} of them against external references ([@sec:results-verify]) |
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
