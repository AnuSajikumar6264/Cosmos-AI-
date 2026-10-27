# Methodology and Implementation {#sec:methods}

This chapter describes how each study was implemented and why the main design decisions were taken. The ESA-ADB studies are described in most detail, because they carry the confirmatory claims.

The configuration values of the pre-registered study are inserted from the frozen code (keys `cfg.*` in Appendix E). For the other studies, constants are quoted with the module that defines them.

## Data acquisition and validation {#sec:methods-data}

Every external dataset is fetched by `cosmos/ai/data/download.py`. The downloader resumes interrupted transfers and verifies each file against the MD5 checksum published by Zenodo (or against the file size where no checksum is published). It records MD5 and SHA-256 hashes, the source record and the download date in a per-dataset `MANIFEST.json`. GitHub code snapshots record the commit that was fetched. Archives are extracted by a function that refuses absolute paths, parent-directory references and links.

The ESA Anomalies Dataset exists in two Zenodo revisions. Rather than re-downloading the 7.9 GB second revision, the second revision was compared with the first member by member using the CRC-32 values in the zip central directories, read over HTTP range requests. The result: "{{data.esa.revision_check}}".

`cosmos/ai/data/validate.py` then recomputes every dataset statistic used in this thesis directly from the files: counts, class balance, missing values, duplicates between splits, coordinate ranges and leakage rules. It writes `data/DATASET_REPORT.json`. Dataset numbers in this thesis come from that report.

## ESA-ADB: official preprocessing {#sec:methods-esa-pre}

`prepare_esa_official` (`cosmos/ai/data/preprocess_telemetry.py`) reproduces the official preprocessing of the benchmark.

1. Each channel is mapped onto the mission grid by zero-order hold: {{data.esa.m1.grid_seconds}} s for Mission 1 and {{data.esa.m2.grid_seconds}} s for Mission 2.
2. Monotonic counters are differenced.
3. String-valued channels on Mission 2 are factorised.
4. The benchmark's "restore" step is applied. When a grid bin contains several samples and an annotated sample would otherwise be lost, that annotated sample is moved onto the next grid point.
5. Telecommands of priority 3 or higher become a 0/1 indicator per grid step.

A unit test compares the vectorised implementation with a literal port of the official pandas code on irregular synthetic channels and requires identical values and labels. The processed grids contain {{data.esa.m1.train.n:int}} training and {{data.esa.m1.test.n:int}} test steps on Mission 1, and {{data.esa.m2.train.n:int}} and {{data.esa.m2.test.n:int}} steps on Mission 2. The target channel sets have {{data.esa.m1.target_channels}} and {{data.esa.m2.target_channels}} channels.

The official test labels are restricted to the target channels and to events lying entirely inside the test half (`official_test_labels`). The scoring adapter (`telemetry/esa_eval.py`) converts per-channel alarms into the event intervals that the vendored official code expects.

## ESA-ADB: the exploratory study (H1–H3) {#sec:methods-explo}

The first study was designed before the work reported here was extended to Q1-journal standards. It is reported as **exploratory**, because the design was revised after its test numbers had been seen ([@sec:methods-prereg]).

**Periods.** The training half of each mission was divided into three periods:

| Purpose | Mission 1 | Mission 2 |
|---|---|---|
| Fitting | {{cfg.periods.m1.fit}} | {{cfg.periods.m2.fit}} |
| Early stopping | {{cfg.periods.m1.es}} | {{cfg.periods.m2.es}} |
| Calibration and selection | {{cfg.periods.m1.cal}} | {{cfg.periods.m2.cal}} |

**Encoders.** Five encoders produced forecasts on a shared origin schedule (context {{cfg.fc.context}} steps, horizon {{cfg.fc.horizon}} steps):

- **Persistence and AR(8).** These are the COSMOS forecasters ([@eq:ar]), fitted per channel on the fitting period.
- **A causal TCN.** It has {{cfg.fc.levels}} levels, a receptive field of 509 steps, width {{cfg.fc.d_model}}, a learned channel embedding and seven quantile outputs per horizon step.
- **PatchTST.** It uses patches of {{cfg.fc.patch}} steps, {{cfg.fc.n_layers}} layers, width 64 and per-window instance normalisation.
- **Chronos-Bolt-small.** This pretrained forecaster was used zero-shot.

The deep encoders were trained channel-independently with weight sharing:

- **Data:** windows drawn from nominal fitting data.
- **Optimiser:** AdamW [@loshchilov2019adamw], learning rate {{cfg.fc.lr}}, weight decay {{cfg.fc.weight_decay}}.
- **Budget:** batch {{cfg.fc.batch}}, at most {{cfg.fc.steps:int}} steps.
- **Early stopping:** on the early-stopping period, with a patience of {{cfg.fc.patience}} evaluations.

The TCN's normalisation was changed from GroupNorm to a per-time-step LayerNorm (`ChannelNorm`), because GroupNorm over time leaks future samples into earlier outputs. The fix was verified by a causality test.

**Calibrators.** Three calibrators turn scores into hot cells, and all three share the event logic of [@sec:theory-events]:

- a per-channel percentile threshold, with tail fractions {{cfg.explo.levels.threshold}};
- conformal p-values with BH across channels at level $q \in$ {{cfg.explo.levels.conformal_bh}};
- a rolling variant of the second, whose pool is refreshed daily from the trailing 30 days of *unlabelled* test scores.

The persistence was chosen from $k\in$ {{cfg.explo.ks}}, and the merge gap was fixed at 1 h. Level and persistence were selected by cross-fitting on the calibration period: calibrate on one half, score the other half with the official metric, swap and average. On Mission 1 this period contains only {{data.esa.m1.events_target.cal.events}} labelled events.

**Telecommands.** For H3, the deep encoders optionally received telecommand features $\exp(-\Delta t/\tau)$ with $\tau \in \{1, 12\}$ h. These entered through a gated fusion $h' = h + \sigma(W_g[h; e]) \odot W_e e$, with modality dropout during training.

**Hypotheses (exploratory).**

- **H1:** a deep or pretrained encoder beats AR(8) under the same calibration.
- **H2:** conformal-BH beats a per-channel threshold.
- **H3:** telecommand gating reduces alarm runs that touch only rare nominal events.

Deep arms used three seeds.

## ESA-ADB: the pre-registered study {#sec:methods-prereg}

### Why a second study

The exploratory study left one striking asymmetry. The same AR(8) scores with the same conformal-BH layer scored {{explo.m2.full.ar8.conformal_bh.f05.mean:.3f}} on Mission 2 but {{explo.m1.full.ar8.conformal_bh.f05.mean:.3f}} on Mission 1. On Mission 1 the alarm layer had been selected on {{data.esa.m1.events_target.cal.events}} events. The training half as a whole holds {{sel.m1.n_train_events}} annotated events on the target channels:

- {{data.esa.m1.events_all.train.Anomaly}} anomalies;
- {{data.esa.m1.events_all.train.Rare_Event}} rare nominal events;
- {{data.esa.m1.events_all.train.Communication_Gap}} communication gaps.

These were unavailable to the selection because no out-of-sample scores existed for them. The pre-registered study therefore asks two questions:

- Does selecting the alarm layer on out-of-fold scores of the whole training half fix the Mission 1 failure?
- Does a supervised event scorer trained on those labels add to an unsupervised forecaster?

The decision to study Mission 1 selection was taken *after* Mission 1 had failed on its test half. This is acknowledged as a threat to validity ([@sec:threats]).

[@fig:esa-protocol] summarises the resulting protocol.

![Workflow of the pre-registered ESA-ADB study. Everything above the freeze uses the training half only. The test half is scored once per arm, through a function that verifies the frozen hashes.](figures/diag_esa_protocol.png){#fig:esa-protocol width=75%}

### Out-of-fold scores {#sec:oof}

Out-of-fold scores are produced as follows (`telemetry/oof.py`).

1. The training half is cut into $K$ = {{oof.m1.K}} contiguous blocks.
2. For block $b$, the AR(8) forecaster of [@eq:ar] is fitted on the nominal samples of the other blocks. A purge margin of one day on either side of $b$ is excluded ({{oof.m1.purge_steps:int}} steps on Mission 1, {{oof.m2.purge_steps:int}} on Mission 2).
3. The fitted forecaster scores block $b$ with [@eq:score-ar].
4. Inputs (lagged values) are read from the record itself, as any deployed forecaster would read them.
5. Test-half scores come from a forecaster fitted on every nominal training sample.

This is cross-fitting, not forward chaining: a block can be scored by a model that also saw later blocks. Forward chaining would have discarded the first block. On Mission 1 that block holds the densest year of events. The resulting optimism is a stated threat.

### Event features and scorers {#sec:model}

Each target channel and grid step receives seven causal features (`telemetry/event_model.py`): {{cfg.evm.features}}. In words, these are:

- $\operatorname{asinh} z$;
- $\log(1+s)$, where $s$ is the out-of-fold AR(8) score;
- the trailing exponential means of $\log(1+s)$ with time constants 1 h, 12 h and 3 days. These give context beyond the network's receptive field, since Mission 1 events often last days.
- $\operatorname{asinh}(z - \bar z^{1\text{d}})$, the value minus its trailing one-day mean;
- an indicator that a forecast exists.

The exponential means are computed with `scipy.signal.lfilter` over the joined training-and-test record, so the test half starts from the state at the end of training. A unit test confirms that a change at step $t$ leaves every earlier feature unchanged.

**A3: causal TCN event scorer.** [@fig:event-tcn] shows the architecture.

- **Input:** a 1×1 convolution from 7 to {{cfg.evm.d_model}} channels plus a learned channel embedding.
- **Body:** {{cfg.evm.levels}} residual blocks with dilations $2^0 \dots 2^7$. Each block holds two causal convolutions (kernel {{cfg.evm.kernel}}) with per-step LayerNorm, SiLU activation and dropout {{cfg.evm.dropout}}.
- **Cross-channel mixing:** every {{cfg.evm.mix_every}} blocks the channel mean of the hidden state is projected and added back. This mixing is permutation-equivariant.
- **Head:** a 1×1 convolution producing three logits per channel and step: nominal, anomaly and rare nominal event.

The receptive field is {{evm.receptive_field}} steps, and the model has {{evm.m1.params:int}} parameters on Mission 1. Training used the following settings:

- class-weighted cross-entropy with weights {{cfg.evm.class_weights}}, with communication gaps ignored;
- {{cfg.evm.steps:int}} AdamW steps (learning rate {{cfg.evm.lr}}, weight decay {{cfg.evm.weight_decay}}, one-cycle schedule [@smith2019superconvergence]);
- batches of {{cfg.evm.batch}} windows of {{cfg.evm.window:int}} steps containing all channels. A fraction {{cfg.evm.event_fraction}} of the windows contained a labelled sample, placed at a uniformly random position inside the part of the window where the loss is computed.

The loss is computed only on the last $T - \mathrm{RF}$ steps of each window (RF = {{evm.receptive_field}}), where the receptive field is complete. The event score is $\log P(\text{anomaly} \cup \text{rare}) - \log P(\text{nominal})$.

**A2: tree baseline** (`telemetry/event_gbt.py`). It uses the same seven features, plus trailing maxima of $\log(1+s)$ over 16 and 256 steps and the channel index as a categorical variable. The model is a three-class XGBoost with these settings:

- objective `{{cfg.gbt.objective}}`;
- {{cfg.gbt.rounds}} trees of depth {{cfg.gbt.max_depth}}, learning rate {{cfg.gbt.eta}}, subsampling {{cfg.gbt.subsample}};
- the same class weights as A3;
- training on {{cfg.gbt.n_rows:int}} subsampled (step, channel) rows, half of them labelled.

**Out-of-fold protocol for both scorers.** One model is trained per held-out block (on the other blocks, minus the purge margin) and predicts that block. A final model trained on all blocks predicts the test half.

![The causal TCN event scorer (arm A3). The receptive field is {{evm.receptive_field}} steps. Channel mixing preserves causality because it acts at each time step separately.](figures/diag_event_tcn.png){#fig:event-tcn width=70%}

### The event-level conformal alarm layer {#sec:alarm-layer}

All arms share one layer ([@fig:alarm-layer]; `telemetry/alarms.py`).

1. Each channel's nominal out-of-fold training scores form its calibration pool.
2. Test scores receive conformal p-values ([@eq:conformal-p]).
3. BH at level $q$ is applied across channels at each step.
4. Persistence $k$ and a {{cfg.prereg.merge_h}} h merge are applied.
5. Channels are attributed to each alarm run ([@sec:theory-events]).

The pre-registered grids are:

- BH level $q \in$ {{cfg.prereg.levels.conformal_bh}};
- per-channel threshold tail fractions {{cfg.prereg.levels.threshold}} (0 means the maximum nominal score);
- persistence $k \in$ {{cfg.prereg.ks}}.

The rolling variant was retained as a secondary layer. It is not selected on its own; it reuses the $(q, k)$ selected for conformal-BH.

![The event-level conformal alarm layer shared by all arms of the pre-registered study.](figures/diag_alarm_layer.png){#fig:alarm-layer width=100%}

**Selection** (`oof_benchmark.select`) cross-fits over the two halves of the training half:

1. Calibrate on the nominal out-of-fold scores of one half.
2. Raise alarms on the other half and score them with the official metric on that half's labels.
3. Swap the halves and average the two results.

Ties go to the smaller alarm fraction. The chosen layer is refitted on the whole training half and applied once to the test scores.

**Point-level comparator.** This is per-channel ACI ([@eq:aci]) with $\alpha$ = {{cfg.prereg.aci_alpha}}. $\gamma$ is chosen from {{cfg.prereg.aci_gammas}} as the value whose nominal false-alarm rate on the second training half is closest to $\alpha$. The rule is adapted from Şahin, who selects $\gamma$ on a window of the test period [@sahin2026tccad]. Its global alarm is the OR over channels. It is evaluated without persistence or merging, and also with the event logic of the layer above (`aci_event`).

### Arms, hypotheses and statistics {#sec:prereg}

[@tbl:arms] defines the arms. A2 and A3 were run with five seeds each (0–4). The secondary arm with anomaly-only scores, listed in the pre-registration, was not run in the confirmatory pass (Appendix B); it was run post hoc as P5 ([@sec:results-posthoc]). Two ablations of A3 were run with three seeds:

- without the forecast-residual inputs (`no_residual`);
- with two output classes instead of three.

: Arms of the pre-registered study. All share the alarm layer of [@sec:alarm-layer]. {#tbl:arms}

| Arm | Scores | Selection of $(q, k)$ | Seeds |
|---|---|---|---|
| A0 | AR(8) from the exploratory study (fitted on the fitting period) | cross-fit on the calibration period | deterministic |
| A1 | AR(8), out of fold ($K = 5$) | cross-fit on the whole training half | deterministic |
| A2 | XGBoost event scorer, out of fold | cross-fit on the whole training half | 5 |
| A3 | causal TCN event scorer, out of fold | cross-fit on the whole training half | 5 |

**Hypotheses.** They concern Mission 1, the primary metric and the conformal-BH layer:

- **H4:** A3 > A1. A supervised deep scorer beats the unsupervised forecaster.
- **H5:** A3 > A2. The deep scorer beats trees on the same features.
- **H6:** A1 > A0. Selection on the whole training half beats selection on the calibration period.

**Statistics.**

- *Test:* a paired block bootstrap (fixed 30-day blocks shared by both arms, seeds resampled hierarchically, 2,000 replicates; [@sec:theory-stats]).
- *Correction:* Holm across the family {H4, H5, H6} at $\alpha = 0.05$.
- *Minimum effect size of interest:* +0.05.
- *Verdicts:*
  - **supported** if the Holm-adjusted $p < 0.05$ *and* the point estimate is at least +0.05;
  - **refuted** if the upper end of the 95 % interval lies below +0.05;
  - **inconclusive** otherwise.

Mission 2 repeats the three comparisons with the same test but without multiplicity correction, outside the confirmatory family. Mission 2 was assigned that role before any arm was scored, because the out-of-limit rule already scores {{explo.m2.full.ool.native.f05.mean:.3f}} there.

### Freeze, gating and deviations

The pre-registration document (`research_design/preregistration/PREREG_TELEMETRY_V1.md`) and {{prereg.frozen_code_files}} code files, {{prereg.frozen_files}} files in all, were hashed into `freeze.json` at {{prereg.freeze_written}} (freeze `{{prereg.freeze_id}}`). The frozen code covers:

- the out-of-fold scorers;
- the event models;
- the alarm layer;
- the selection and test code;
- the bootstrap;
- the official metric code;
- the preprocessing.

`oof_benchmark.test` refuses to run if any frozen file has changed, and every arm was scored once. {{prereg.n_deviations}} deviations are logged with their timing (Appendix B). Five are procedural and changed no computation. The sixth records that arm A0 carries the exploratory selection grid, so H6 compares two complete protocols. After all confirmatory numbers existed, the pre-registration's own deviation log was filled in and the freeze reissued as `{{prereg.refreeze_id}}`.

## Label-free baselines for ESA-ADB {#sec:methods-baselines}

Three label-free baselines are scored through the same metric code (`telemetry/baselines.py`).

- **GlobalSTD.** This is an exact port of the benchmark's own rule. It alarms when $|x - \mu| > \kappa\sigma$, with the mean and standard deviation of nominal training samples and $\kappa \in \{3, 5\}$. Its purpose is to check the whole preprocessing and metric chain against the published values.
- **Out-of-limit (OOL).** This rule alarms whenever a sample leaves the [minimum, maximum] range of its channel's nominal training samples. It has no free parameter. A variant uses all training samples.
- **The COSMOS monitor** of `health/realdata.py`. It is fitted on the official training half and run online over the test half.

## OPSSAT-AD {#sec:methods-opssat}

The OPSSAT-AD study (`telemetry/opssat.py`) uses two inputs:

- the {{data.opssat.n_features}} official features, signed-log transformed;
- the raw fragments resampled to 256 points as three channels: per-fragment z-score, $\log(1+\Delta t)$ and value divided by its maximum absolute value.

**Splits.** Two splits are evaluated:

- *official:* {{data.opssat.train}} training and {{data.opssat.test}} test fragments;
- *time-blocked:* all fragments starting before the 75th percentile of start times go to training.

In both splits, a stratified 20 % of the training part is held out for validation.

**Classical models.** The classical baselines are:

- a COSMOS robust-z rule, with its threshold at the 95th percentile of nominal training scores;
- Isolation Forest;
- AdaBoost;
- a random forest;
- logistic regression;
- XGBoost, early-stopped on validation AUC-PR.

**Deep models.** The deep models (`telemetry/models.py`) are:

- the feature MLP (FCNN, {{ops.official.fcnn.f100.params:int}} parameters), which re-implements the strongest published baseline;
- a 1-D CNN on the raw fragment;
- a patch transformer;
- a hybrid network that concatenates CNN and MLP embeddings.

They were trained with binary cross-entropy and AdamW (learning rate $10^{-3}$, batch 64). Early stopping used validation AUC-PR with patience 30.

**Foundation-model probe.** A logistic-regression probe was trained on frozen Chronos-Bolt embeddings.

**Further analyses.** Every model was run with five seeds and decides at $P \ge 0.5$. The study also includes:

- label efficiency (10–100 % of the training data);
- robustness to added noise and to sample dropout (`telemetry/robustness.py`);
- paired bootstrap comparisons on seed-averaged scores.

## Learned health monitor (simulated telemetry) {#sec:methods-monitor}

`telemetry/learned_monitor.py` replaces the classical monitor inside `run_pipeline`. It works in five steps.

1. **Training data.** A PatchTST forecaster (98,424 parameters) is trained on 180 simulated fault-free runs with the pinball loss.
2. **Scoring.** Scores follow [@eq:score-q] and are smoothed by an EWMA.
3. **Calibration.** Scores are calibrated against 20 held-out fault-free runs, which yields conformal p-values per channel.
4. **Combination.** The p-values are combined across the 16 channels with Simes. The anomaly score $a = \mathrm{clip}\big(-\log_{10} p_S / (2\cdot(-\log_{10} q)), 0, 1\big)$ with $q = 10^{-3}$ equals 0.5 exactly at $p_S = q$.
5. **Alarm.** The classical persistence rule raises the alarm.

**Benchmark.** Every subsystem–fault pair except `stuck` (which crashes the simulator) is run, plus eight fault-free runs per seed set. Seed sets 0–4 served for development and 5–9 are held out. Two limitations apply:

- the 20 calibration runs also selected the checkpoint;
- both monitors would cap their healthy normalisation window at the true fault onset, but with the default onset the cap never binds (Appendix C).

## Vision {#sec:methods-vision}

### SpotGEO

The training set of {{data.spotgeo.train.sequences}} sequences is split by a hash of the sequence identifier into {{data.spotgeo.valsplit.train_sequences}} training and {{data.spotgeo.valsplit.val_sequences}} validation sequences. The {{data.spotgeo.test.sequences}} test sequences are scored once with the vendored official scorer. For each frame the front end computes two channels:

- the frame minus the median of the other four registered frames, which reuses the COSMOS phase-correlation registration;
- the frame minus a Gaussian blur with $\sigma = 6$ px.

Both are robust-normalised. The input variants are both channels for all five frames (10 channels), differences only, or high-pass only.

**Networks.** The heat-map U-Net has encoder widths 32–256 and a 512-channel bottleneck, {{spot.unet.both.f100.params.mean:int}} parameters. The ResNet-18 variant uses an ImageNet-pretrained encoder ({{spot.resnet18_pretrained.both.f100.params.mean:int}} parameters). Both output five heat maps (one per frame) with sub-pixel offsets.

**Training.**

- Settings: 40 epochs, batch 12, random 320 px crops, AdamW (learning rate $10^{-3}$ for the U-Net, $3\times10^{-4}$ for ResNet), one-cycle schedule, gradient clipping 5.
- Augmentation: time reversal, flips and gain.
- Selection: every two epochs the validation set is decoded at thresholds 0.1–0.7. The epoch and threshold with the best validation F1 are kept.

**Classical baseline.** The COSMOS DoG detector is applied to the difference channel, with its threshold chosen on validation.

### COSMOS synthetic scenes (SYNTHETIC)

Scenes are drawn from disjoint seed ranges: {{data.synvis.train.images}} for training, {{data.synvis.val.images}} for validation and {{data.synvis.test.images}} for test. The images are COSMOS registered differences. Three detectors are compared:

- the existing TinyYOLO;
- a COCO-pretrained YOLOv8n via Ultralytics;
- the heat-map U-Net with a width–height head.

All three are scored with one COCO-style mAP implementation (101-point interpolation, IoU 0.50:0.05:0.95), because Ultralytics' internal evaluator interpolates differently.

## Re-entry {#sec:methods-reentry}

### Landmark study (real catalogue)

The design is shown below.

- **Snapshots.** The landmark date is $\tau$ = {{lm.tau}}, with follow-up to {{lm.follow_up}}. The CelesTrak SATCAT snapshot of $\tau$ defines the population; a later snapshot defines the outcomes.
- **Population.** An object is included if it is Earth-orbiting and in orbit at $\tau$, has orbit fields, and was launched by $\tau$. This leaves {{data.landmark.population_at_risk:int}} objects with {{data.landmark.events}} re-entries.
- **Features.** Only information available at $\tau$ is used: SATCAT orbit fields, radar cross-section, object type, owner, launch site and age. Where available, the latest general-perturbations element set with epoch at most one day after $\tau$ is added; this covers {{data.landmark.gp_coverage:pct1}} of objects.
- **Split.** Objects are split by launch group (international designator year and number) into 60/20/20 % of groups, so that fragments of one launch never span splits.
- **Leakage checks.** A leakage report checks five rules, and all pass:
  - no element set after $\tau$;
  - no event at or before $\tau$;
  - launch groups confined to one split;
  - no decay field among the features;
  - a constant censoring time.

**Models.** Five models are compared:

- the COSMOS physics lifetime at the perigee, Platt-scaled on validation;
- logistic regression;
- XGBoost (early-stopped on validation AUC-PR, five seeds);
- XGBoost-AFT;
- a discrete-time hazard MLP with 7-day bins.

The MLP configuration was chosen on validation AUC-PR from a grid of {{lm.mlp.grid_size}} configurations, then run as a five-seed ensemble. Its selected configuration was {{lm.mlp.selected}}.

**Metrics.** The metrics are:

- AUC-PR and AUC-ROC for re-entry within 30 and 68 days;
- the concordance index;
- the Brier score and the expected calibration error;
- launch-group bootstrap intervals for differences between models;
- marginal and Mondrian conformal sets at $\alpha = 0.1$.

### Propagator surrogate (SIMULATED)

`cosmos/ai/surrogate.py` trains an ensemble on reference simulations from the COSMOS ensemble propagator. Its inputs and outputs are:

- **Inputs (13):** break-up altitude, speed, flight-path angle, heading, latitude, ballistic coefficient, mass, density factor and a velocity increment.
- **Outputs:** downrange and crossrange impact position, fall time, impact speed and an impact flag.

**Data.** The simulations comprise 400,000 training and 40,000 validation cases (`surrogate.SPLITS`), {{sur.test.n:int}} test cases, and an out-of-range test set of {{sur.ood_test.n:int}} cases with break-up altitudes of 100–120 km.

**Ensemble and training.** The ensemble has five MLPs of four 256-unit layers each (202,245 parameters per member). They were trained with masked MSE plus 0.1 times the impact-flag BCE, using AdamW and a cosine schedule.

**Use inside the pipeline.** `fusion.surrogate_footprint` reproduces the reference sampler's random draws call for call. Samples outside the training ranges fall back to the reference propagator (mode `range`). Two other modes are available: `spread`, which also falls back on large ensemble disagreement, and `none`.

## Experiment records and reproducibility

Every training or evaluation run is wrapped in `experiment.Run`. It writes the configuration, the environment (library versions and device), the epoch logs, the metrics and the checkpoints to a run folder, and appends a line to `outputs/experiments/index.jsonl`. The index holds {{runs.total}} runs, of which {{runs.ok}} completed and {{runs.failed}} failed; every failed run was superseded by a successful rerun. Seeds are fixed for NumPy, Python and PyTorch. XPU kernels are not bit-deterministic, so repeated deep runs differ slightly, which is one reason seeds are reported as distributions. Appendix A gives the commands that regenerate each result.
