# Theoretical Foundations {#sec:theory}

This chapter defines the concepts, equations and metrics on which the implementation relies. Each definition is given in the form in which the code computes it. Where the code departs from a textbook form, the departure is stated, and the implementing module is named in brackets.

## Telemetry as multichannel time series

Spacecraft telemetry consists of many channels sampled irregularly, at different rates, with gaps. ESA-ADB defines a fixed sampling grid per mission (30 s for Mission 1 and 18 s for Mission 2). Each channel is mapped onto that grid by zero-order hold, i.e. the last observed value is carried forward [@kotowski2026esaadb]. Let $x_{t,c}$ denote channel $c$ at grid step $t$. All learned models in this thesis operate on a robust normalisation:
$$
z_{t,c} = \frac{x_{t,c} - \operatorname{med}_c}{\hat\sigma_c},
$$ {#eq:robust}
Both the median $\operatorname{med}_c$ and the scale $\hat\sigma_c$ are computed on nominal samples of a fitting period. The scale is $\hat\sigma_c = \mathrm{IQR}_c/1.349$, which equals the standard deviation for Gaussian data. Channels with zero interquartile range, such as discrete or zero-order-held channels, fall back to $(Q_{99}-Q_{01})/4.653$ and then to the standard deviation (`telemetry/forecasting.py`, `MissionData`). For monotonic counters, the official preprocessing replaces the value by its forward difference $x_{i+1}-x_i$ [@kotowski2026esaadb]. This introduces a one-sample look-ahead that the implementation inherits unchanged (Appendix C).

### Labels

Annotations in ESA-ADB are intervals attached to channels, grouped into events by an identifier. Each event belongs to one of three categories:

- an **anomaly**;
- a **rare nominal event**, which is unusual but expected behaviour such as a commanded manoeuvre;
- a **communication gap**.

The benchmark's primary task treats anomalies and rare nominal events together as targets. Communication gaps are ignored.

## Forecast residuals as non-conformity scores {#sec:nonconformity}

An unsupervised detector scores how surprising each new sample is relative to a model of nominal behaviour. COSMOS-AI uses forecasting models.

**Autoregressive forecaster.** For each channel, the forecaster fits a ridge-regularised autoregressive model of order 8 by least squares on nominal samples (penalty $10^{-3}$, intercept included):
$$
\hat z_{t,c} = \sum_{k=1}^{8} \phi_{k,c}\, z_{t-k,c} + \phi_{0,c}.
$$ {#eq:ar}
It is iterated over a horizon of $H=16$ steps from forecast origins spaced 16 steps apart. Every sample is therefore predicted between 1 and 16 steps ahead. The non-conformity score is the absolute forecast error divided by a robust residual scale:
$$
s_{t,c} = \frac{|z_{t,c} - \hat z_{t,c}|}{\max(\sigma^{r}_c, \delta)},\qquad
\sigma^{r}_c = \max\!\left(1.4826\,\operatorname{med}|r|,\ \frac{Q_{90}(|r|)}{1.645},\ \delta\right),\quad \delta = 0.05,
$$ {#eq:score-ar}
Here $r$ are the residuals on usable samples (`telemetry/oof.py`).

**Quantile forecasters.** These are the deep encoders of the exploratory study (TCN, PatchTST and Chronos-Bolt). They predict the quantiles $q_{0.25}, q_{0.5}, q_{0.75}$. The score is
$$
s_{t,c} = \frac{|z_{t,c}-q_{0.5}|}{\max(q_{0.75}-q_{0.25},\ \delta)} .
$$ {#eq:score-q}
Quantile models are trained with the pinball loss $\ell_\tau(y,\hat y) = \max(\tau d, (\tau-1)d)$, where $d = y-\hat y$, averaged over the quantile levels $\tau \in \{0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99\}$ (`telemetry/models.py`).

## Conformal p-values and false-discovery-rate control {#sec:theory-conformal}

A score becomes an alarm decision only after calibration. Let $\mathcal C_c$ be a calibration pool of $n_c$ scores of channel $c$ recorded on nominal data. The conformal p-value of a new score $s$ is
$$
p_c(s) = \frac{1 + \#\{s' \in \mathcal C_c : s' \ge s\}}{n_c + 1}.
$$ {#eq:conformal-p}
If the new nominal score and the pool are exchangeable, $p_c$ is super-uniform: $\Pr(p_c \le \alpha) \le \alpha$ for every $\alpha$ [@vovk2022alrw; @bates2023]. With $m$ channels tested at the same time step, the Benjamini–Hochberg (BH) procedure at level $q$ works as follows [@benjamini1995]:

1. Order the p-values, $p_{(1)} \le \dots \le p_{(m)}$.
2. Find the largest $j$ with $p_{(j)} \le jq/m$.
3. Reject the $j$ smallest.

For independent p-values, BH controls the false-discovery rate, the expected fraction of false rejections among the rejections, at $q\,m_0/m$, where $m_0$ is the number of true nulls [@benjamini1995]. Benjamini and Yekutieli extended the result to positively regression-dependent p-values [@benjamini2001control]. Bates et al. showed that conformal p-values of several test points computed against *one shared* calibration set satisfy that condition [@bates2023].

Three caveats apply to telemetry.

- **The pools are per channel.** COSMOS-AI applies BH across channels, each with its own calibration pool, and the channels' scores are correlated. This is not the shared-pool setting of Bates et al., so their result does not apply even under exchangeability.
- **Exchangeability fails.** Consecutive samples are serially dependent, and missions drift, so exchangeability between the pool and the test data does not hold. Weighted and adaptive variants of conformal inference address drift with weaker guarantees [@barber2023; @gibbs2021aci].
- **Finite pools bound the p-values.** The smallest attainable p-value is $1/(n_c+1)$. With BH at level $q$ over $m$ channels, a single channel can be rejected only if $1/(n_c+1) \le q/m$.

COSMOS-AI therefore treats $q$ as a tuning parameter selected on training data and reports realised false-alarm rates. It does not claim a guarantee.

The Simes combination $p_S = \min_j m\,p_{(j)}/j$ gives one global p-value per time step. It rejects exactly when BH rejects at least one hypothesis (`uncertainty.py`); the learned health monitor uses it ([@sec:methods-monitor]).

**Adaptive conformal inference (ACI)** adjusts the miscoverage level online [@gibbs2021aci]:
$$
\alpha_{t+1} = \alpha_t + \gamma\left(\alpha - \mathbb 1[s_t > \hat Q_{1-\alpha_t}(\mathcal C)]\right).
$$ {#eq:aci}
It keeps the long-run rate of exceedances near the target $\alpha$ even under drift. An exceedance lowers $\alpha_t$ by $\gamma(1-\alpha)$; once $\alpha_t < 0$ the quantile is infinite and the channel cannot fire until enough non-exceedances have raised $\alpha_t$ again. It is used here as the point-level comparator of [@sec:alarm-layer], following its use for spacecraft telemetry by Şahin [@sahin2026tccad].

## From hot samples to alarm events {#sec:theory-events}

Let $H_{t,c}\in\{0,1\}$ mark the samples that the calibrator flags ("hot cells"). Operators act on alarm *events*, not samples, so the implementation applies three rules (`telemetry/alarms.py`).

1. **Persistence.** A global alarm starts only after at least $k$ consecutive steps contain a hot channel. The alarm is placed from the step at which persistence is reached.
2. **Merging.** Alarm runs separated by fewer than $\Delta$ steps are joined. $\Delta$ corresponds to one hour.
3. **Attribution.** The channels reported for an alarm run are those hot at any step inside the run, including the $k-1$ steps that established persistence.

The global alarm is the logical OR over channels, as in the benchmark's own pipeline.

## The ESA-ADB metrics {#sec:theory-metrics}

The metrics are computed by the official ESA-ADB code, which is vendored unchanged in `telemetry/esa_metrics/`. Let the ground-truth events be intervals grouped by event identifier, and let the predicted events be the alarm runs.

- **True positive.** A ground-truth event is a true positive ($TP_e$) if at least one predicted event overlaps it. Otherwise it is a false negative ($FN_e$).
- **False positive.** A predicted event is a false positive ($FP_e$) if it overlaps no ground-truth interval.
- **Corrected event-wise precision.** The event-wise precision is multiplied by a time-based true-negative rate [@sehili2023flawed]:

$$
P_e = \frac{TP_e}{TP_e+FP_e}\cdot \mathrm{TNR},\qquad
\mathrm{TNR} = 1 - \frac{\text{duration of predicted alarms inside nominal time}}{\text{total nominal duration}},
$$ {#eq:pe}
$$
R_e = \frac{TP_e}{TP_e + FN_e},\qquad
F_{\beta} = \frac{(1+\beta^2) P_e R_e}{\beta^2 P_e + R_e},\quad \beta = 0.5 .
$$ {#eq:fbeta}

Setting $\beta=0.5$ weights precision twice as heavily as recall. The TNR factor penalises long alarms on nominal data even when they are few. Several secondary metrics are also reported:

- **Alarming precision:** $TP_e/(TP_e + \text{redundant detections})$.
- **ADTQC:** a score for detection timing.
- **Affiliation-based $F_{0.5}$:** a time-weighted precision and recall [@huet2022local].
- **Channel- and subsystem-aware $F_{0.5}$:** these score whether the attributed channels are the affected ones.

The primary metric of every ESA-ADB result in this thesis is the corrected event-wise $F_{0.5}$ over anomalies and rare nominal events. The *anomaly-only* variant counts only anomalies as targets.

## Temporal convolutional networks {#sec:theory-tcn}

A temporal convolutional network (TCN) stacks one-dimensional causal convolutions with exponentially increasing dilation [@bai2018tcn]. A causal convolution with kernel size $\kappa$ and dilation $d$ computes
$$
y_t = \sum_{i=0}^{\kappa-1} w_i\, x_{t - d i},
$$ {#eq:causal-conv}
so the output at $t$ depends only on inputs at or before $t$. A stack of $L$ residual blocks, each holding two convolutions with dilation $2^{\ell}$, has the receptive field
$$
\mathrm{RF} = 1 + 2(\kappa-1)\left(2^{L}-1\right).
$$ {#eq:rf}
For $\kappa = 3$ and $L = 8$ this is {{evm.receptive_field}} steps.

Group normalisation over time would mix future samples into past outputs. The implementation therefore normalises across feature channels at each time step only (`ChannelNorm`). The unit tests confirm that an input change at step $t$ leaves every earlier output bit-identical (`tests/test_ai.py`).

## Gradient-boosted trees

Gradient boosting fits an additive ensemble of regression trees. Each new tree approximately follows the negative gradient of the loss with respect to the current prediction. XGBoost adds second-order information and regularisation of tree complexity [@chen2016xgboost]. For multi-class problems it optimises the softmax cross-entropy. Trees are invariant to monotone feature transformations and robust to heterogeneous feature scales, and they remain strong baselines on tabular data [@grinsztajn2022tree].

## Statistical comparison {#sec:theory-stats}

Test-set differences between two detectors are uncertain in two ways. The test half is one realisation of a serially dependent process, and stochastic models vary between seeds.

- **Dependence in time.** Block bootstraps resample contiguous blocks of time, which preserves dependence within each block [@kunsch1989]. The implementation uses fixed, non-overlapping 30-day blocks rather than Künsch's moving blocks: {{data.esa.m1.test.blocks30}} on the Mission 1 test half and {{data.esa.m2.test.blocks30}} on Mission 2. For each replicate it draws blocks with replacement, using the same blocks for both arms of a comparison.
- **Seed variation.** Each seeded arm's statistic is the mean over seeds drawn with replacement, a hierarchical bootstrap.

The event-wise $F_{0.5}$ cannot be written as a sum over blocks, so it cannot be resampled directly. The bootstrap therefore uses a block-decomposable re-implementation that tracks:

- detected events by block;
- false-positive alarm runs by block;
- nominal and alarmed nominal time by block.

The official metric gives the point estimate. The re-implementation agrees with it to within {{conf.reimpl_gap_max:.4f}} on every confirmatory comparison ([@sec:results-prereg]).

The two-sided bootstrap p-value is $2\min\{\Pr(\Delta \le 0), \Pr(\Delta \ge 0)\}$, floored at $1/B$. With $B$ = {{conf.m1.h6.n_boot:int}} the smallest reportable value is {{conf.p_floor:g}}. The Holm step-down procedure controls the family-wise error rate over the pre-registered family [@holm1979].

## Optical detection of moving objects {#sec:theory-vision}

### Two-frame detection

The original COSMOS front end detects moving objects in two steps.

1. **Registration.** The two frames are registered by phase correlation. The normalised cross-power spectrum $F_a F_b^{*}/|F_a F_b^{*}|$ has an inverse transform that peaks at the shift between the frames. The peak is refined to sub-pixel precision with a three-point parabola.
2. **Detection.** The registered difference is searched with a difference-of-Gaussians (DoG) filter at three scales.
   - Detections are thresholded against a photon-noise map, since pixels under bright stars are intrinsically noisier.
   - Negative lobes ("dipoles") left by the object's old position are rejected (`debris/detect.py`).

### Learned point detection

COSMOS-AI follows the "objects as points" formulation [@zhou2019objects]. A network predicts a heat map whose peaks mark object centres, together with sub-pixel offsets. Training uses the penalty-reduced focal loss of CornerNet [@law2018cornernet], with $\alpha = 2$ and $\beta = 4$ as in [@zhou2019objects], applied around Gaussian targets with $\sigma = 1.5$ px. It modifies the focal loss of Lin et al. [@lin2017focal]. The backbone is a U-Net [@ronneberger2015unet], or a ResNet-18 encoder [@he2016resnet].

### SpotGEO scoring

The SpotGEO challenge scores each frame separately [@chen2021spotgeo].

1. Predicted and true points are assigned by the Hungarian algorithm [@kuhn1955hungarian].
2. A match within $\tau = 10$ px is a true positive. Unmatched or distant points are false negatives or false positives.
3. The per-frame error sums the matched distances (zero below $\varepsilon = 3$ px) and adds $\tau^2$ for every unmatched point.

The values $\tau$ and $\varepsilon$ are those of the official toolkit (`vision/spotgeo_validation.py`, vendored unchanged). The challenge ranks by $1-F_1$ and uses the mean squared error only to break ties. The toolkit's MSE is the *sum* of the per-sequence MSE values, not their mean.

## Survival analysis for re-entry {#sec:theory-survival}

Predicting *whether an object re-enters within a horizon* is a time-to-event problem with censoring. Objects that have not yet re-entered by the end of follow-up are censored. A landmark design fixes a prediction date $\tau$ and uses only information available at $\tau$. It then follows each object until a common end date, which makes censoring administrative and identical for all objects.

**Discrete-time hazard model.** The follow-up is divided into bins. A network outputs a hazard $h_k(\mathbf x)$ for each bin, and the survival function is
$$
S_k(\mathbf x) = \prod_{j \le k}\bigl(1 - h_j(\mathbf x)\bigr) .
$$ {#eq:discrete-hazard}
Training maximises the Bernoulli likelihood of the event indicator over the bins in which the object is at risk [@tutz2016discrete; @kvamme2021continuous].

**Accelerated failure time (AFT) model.** The AFT model regresses $\log T$ on the features with a parametric error distribution. XGBoost implements it with interval-censored labels [@barnwal2022aft].

**Evaluation.** Five measures are used:

- the area under the precision–recall curve (AUC-PR) for "re-entry within $h$ days";
- Harrell's concordance index [@harrell1982yield];
- the Brier score [@brier1950verification];
- the expected calibration error;
- conformal prediction sets for the binary outcome.

The conformal sets come in two forms. *Marginal* sets use a single quantile of the non-conformity $1-\hat p(y)$. *Mondrian* (label-conditional) sets calibrate a separate quantile for each class, which gives per-class coverage [@vovk2022alrw; @angelopoulos2023gentle].

## Physics of decay and re-entry {#sec:theory-physics}

**Atmosphere.** The US Standard Atmosphere 1976 (US76) defines temperature profiles in seven layers, with boundaries in geopotential altitude up to 84.852 km (86 km geometric), and hydrostatic pressure and density. Above 86 km the implementation interpolates a density table log-linearly up to 1000 km (`reentry/atmosphere.py`). No solar or geomagnetic activity is modelled. Empirical models such as NRLMSISE-00 represent these effects [@picone2002nrlmsise], and their absence is a stated limitation.

**Drag.** Drag acceleration is $\mathbf a_D = -\tfrac12 \rho\, |\mathbf v_{r}|\, C_D \frac{A}{m} \mathbf v_{r}$, where $\mathbf v_r$ is the velocity relative to the atmosphere. The ballistic coefficient is $B = m/(C_D A)$. The drag coefficient blends a continuum curve in the Mach number with the free-molecular limit 2.10, using a logistic weight in $\log_{10}$ of the Knudsen number (`reentry/drag.py`).

**Equations of motion.** In an Earth-fixed rotating frame with angular velocity $\boldsymbol\omega$:
$$
\ddot{\mathbf r} = \mathbf g_{J_2}(\mathbf r) - 2\boldsymbol\omega\times\dot{\mathbf r} - \boldsymbol\omega\times(\boldsymbol\omega\times\mathbf r) + \mathbf a_D ,
$$ {#eq:eom}
where $\mathbf g_{J_2}$ is point-mass gravity plus the $J_2$ oblateness term [@vallado2013fundamentals].

**Orbital lifetime.** For a near-circular orbit, the semi-major axis decays as $\dot a = -C_D (A/m)\,\rho(h)\sqrt{\mu a}$. The implementation integrates this in 5 km steps from the initial altitude down to 120 km (`debris/lifetime.py`).

**Casualty expectation.** For one fragment with debris area $A_d$, the casualty area is $A_c = (\sqrt{A_d} + 0.6)^2$. The 0.6 m is the square root of the 0.36 m² cross-section assumed for a person (`PERSON_AREA_M2`, `risk.py`, l.27). The expected number of casualties is $E_c = \rho_{\text{pop}}\,A_c$, summed over fragments that carry more than 15 J of kinetic energy. The commonly applied acceptability threshold is $10^{-4}$ per re-entry event. Appendix C notes that the pipeline compares a *per-fragment* value with this per-event threshold.
