# Debris module: collision risk and cloud population {#sec:debris}

## Family evidence

For object $i$ at time $t$ the module computes $\varphi_F(i,t)$: the number of siblings catalogued by $t$ in the same launch family and, for fragments, the same breakup family; the number and fraction of those siblings that decayed naturally before $t$; and their mean log-lifetime. Only events strictly before $t$ enter, so the feature is causal by construction. This is the quantity that coupling C1 passes to the re-entry module.

## Collision risk

Inputs are the conjunction data messages of an encounter released at least two days before closest approach, reduced to one feature row per event: the latest risk; the minimum, maximum, mean and first risk; the change and the three-point slope; the last step; the logarithmic ratio of miss distance and of the covariance scales between the first and the last message; the message count; and the span. The target is the risk of the final message.

A **persistence expert** predicts the latest risk plus the training distribution of (final $-$ latest) changes within the same latest-risk bin, which is an honest persistence forecast with uncertainty rather than a bare point. The same learned experts as the re-entry module are applied to the event features, and the two are fused by Eq. [-@eq:gate]. Learner selection, the gate, the conformal calibration, the isotonic map and the decision threshold are all driven by nested out-of-fold predictions over five event-hash folds, so no test message informs any choice. Quantiles within $10^{-4}$ of the risk floor are snapped to it, because most final risks lie exactly at the floor and the spacing of single-precision floats there is about $2 \times 10^{-6}$.

Because the official metric scores a *point* forecast, a decision rule is also required: predict the latest risk when the calibrated $P(r \geq -6)$ exceeds a threshold chosen on out-of-fold training events, and $-6.001$ otherwise.

## Cloud population

Coupling C2 turns per-object survival into a debris-environment quantity. For a breakup cloud the expected number of catalogued fragments still in orbit at time $t$ is

$$\mathbb{E}[N(t)] = \sum_{i:\, t_{0,i} \leq t} S_i\bigl(t - t_{0,i}\bigr),$$ {#eq:cloud}

where $S_i$ is the survival function of fragment $i$ read from the fused predictive distribution. It is read two ways: directly from the quantile function, and after isotonic maps of $P(T \leq a)$ fitted on calibration fragments at 25 ages and interpolated in log-age. Reading it two ways is necessary because the two readings disagree, as Section [-@sec:results] shows.
