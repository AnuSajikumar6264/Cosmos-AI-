# Results {#sec:results}

## The mixture of experts

Table [-@tbl:moe] and Figure [-@fig:moe] give the eight-arm comparison that tests the paper's central claim.

{{TABLE:moe}}

: **Mixture-of-experts arms on both test periods.** Columns are the mean pinball loss, the mean absolute error of the median in $\log_{10}$ days, and the empirical coverage of the 90 % conformal interval; the first three numeric columns are origin 2016 and the last three origin 2011. {{TABLE:moe_note}} {#tbl:moe}

![**All eight fusion arms on both test periods.** The dotted line marks the best arm at each origin.](artwork_eaai/eaai_moe.png){#fig:moe width=100%}

The headline comparison is favourable to the learned experts and unfavourable to the fusion. For 2016–2021 the in-context foundation models are the most accurate arms, at {{re.o2016.tabicl.pinball:.4f}} and {{re.o2016.kumo.pinball:.4f}} against {{re.o2016.physics_only.pinball:.4f}} for the physics prior and {{re.o2016.xgboost.pinball:.4f}} for the strongest gradient-boosted baseline, reductions of {{imp.re.o2016.tabicl_vs_physics_pct:.0f}} % and {{imp.re.o2016.tabicl_vs_xgboost_pct:.0f}} % respectively. For 2011–2015 the ordering changes: the physics prior has the lowest point estimate, {{re.o2011.physics_only.pinball:.4f}}, and TabM — the arm selected by the tuning split at the other origin — is the worst of all at {{re.o2011.tabm.pinball:.4f}}.

The proposed reliability gate reaches {{re.o2016.cosmos_gated.pinball:.4f}} and {{re.o2011.cosmos_gated.pinball:.4f}}. **It does not attain the best arm at either origin.** It beats the physics prior decisively at 2016 and beats its own learned expert at 2011, but an oracle choosing the single best expert per period would do better in both. Three further points follow from the table. The equal-weight average is the second-best arm at 2011 and among the worst at 2016, so a fixed mixing weight is not a safe default. The learned constant gate, which is the same model without reliability features, is close to the full gate at 2016 and worse at 2011, so the reliability descriptors contribute little at one origin and something at the other. And fitting the gate on the tuning split rather than on out-of-fold training predictions is clearly worse at both origins, which shows the fitting protocol matters more than the parameterisation.

Table [-@tbl:reentry-pairs] gives the paired comparisons with intervals.

{{TABLE:reentry_pairs}}

: **Head-to-head differences in test pinball loss** with paired launch-family cluster bootstraps. A negative difference favours the first model. These comparisons were added after test results had been seen and are nominal. {#tbl:reentry-pairs}

At 2016 both foundation models are significantly better than the physics prior, the gated mixture, TabM and XGBoost. At 2011 the physics prior cannot be separated from either foundation model or from the gated mixture; the single significant difference among them is that Kumo-Tabular is *worse* than the gated mixture that fuses it, {{re.o2011.pair.kumo.cosmos_gated.diff:+.4f}} ({{re.o2011.pair.kumo.cosmos_gated.lo:+.4f}} to {{re.o2011.pair.kumo.cosmos_gated.hi:+.4f}}). Figure [-@fig:gate] shows what the gate actually does: it places a mean weight of {{re.o2016.gate.cosmos_gated.weight_mean:.2f}} and {{re.o2011.gate.cosmos_gated.weight_mean:.2f}} on the physics prior, more when the area-to-mass ratio comes from the breakup model than when a design value exists, which is the behaviour the descriptors were designed to produce.

![**Gate behaviour.** (a) Mean weight placed on the physics prior, by the provenance of the area-to-mass ratio. (b) The fitted reliability coefficients; a positive coefficient shifts weight towards the physics prior.](artwork_eaai/eaai_gate.png){#fig:gate width=100%}

## Model selection under temporal shift

Selection on a tuning cohort five to seven years older than the test cohort did not identify the arm that generalised best. At 2016 TabM led the tuning split by a wide margin and was significantly worse than both foundation models on test. At 2011 the small tuning cohort ({{re.o2011.known5.tuneA}} objects) selected Kumo-Tabular, which is tied with TabICLv2 on test. A random split would have concealed this; only the forward-in-time protocol exposes it.

## What the module outputs look like

Figure [-@fig:reentry-outputs] shows the predictive distributions for individual objects and for the cohort.

![**Re-entry module outputs, origin 2016.** (a) Forty test objects drawn at random among those with a known five-year outcome: median and 90 % conformal interval of the physics prior and of the gated mixture, against the observed outcome. (b–d) Predicted median against observed outcome for every such test object.](artwork/Figure_6.png){#fig:reentry-outputs width=100%}

## Collision risk

{{TABLE:cac}}

: **Collision risk on the official test set.** $L$ is the challenge metric, infinite when $F_2 = 0$ because every median falls below the high-risk threshold and the metric divides by zero. {#tbl:cac}

Kumo-Tabular gives the best predictive distribution, {{cac.kumo.pinball:.3f}} against {{cac.persistence.pinball:.3f}} for persistence, a reduction of {{imp.cac.kumo_vs_persistence_pct:.0f}} %, and the best discrimination of high-risk events at AUC-PR {{cac.kumo.auc_pr_high:.3f}}. Table [-@tbl:cac-pairs] shows the margin is significant against every comparator.

{{TABLE:cac_pairs}}

: **Paired bootstrap over test events of the collision-risk pinball loss.** A negative difference favours Kumo-Tabular. {#tbl:cac-pairs}

On the official point metric, however, no model beats the latest-risk baseline at {{cac.latest_risk_point.L_clip:.3f}}. The reason is visible in Figure [-@fig:cac-outputs]: {{cacq.kumo.n_high_below:.0f}} of the {{cac.data.test_high_risk}} high-risk events receive a median below the threshold, {{cacq.kumo.n_high_in_m10_m6:.0f}} of them between $-10$ and $-6$, so $F_2$ reaches only {{cac.kumo.F2:.2f}}. The conformal interval covers {{cac.kumo.cov90:pct}} of all events but only {{cac.kumo.cov90_high:pct}} of the high-risk ones, because the calibration groups are coarse exactly where the decisions are made. This reproduces the challenge's own finding and an independent re-evaluation [@uriot2022spacecraft; @vazquez2026latest].

![**Collision-risk outputs.** (a) Final risk against the latest available risk. (b) Final risk against the median forecast. (c–e) Three high-risk test events drawn at random: input messages, predictive quantiles and the final risk.](artwork/Figure_11.png){#fig:cac-outputs width=100%}

## Health

{{TABLE:telemetry}}

: **Telemetry detectors.** AUC-PR is the per-channel mean over anomalous steps; $F_{0.5}$ is the corrected event-wise score, which charges alarm time spent on nominal data. "Late ref." keeps each detector's amount of alarm time but places it as one block at the end of the series. {#tbl:telemetry}

The zero-shot foundation model does not beat the simple detectors. Its per-channel AUC-PR is {{smap.SMAP.chronos2.pointwise_auc_pr:.3f}} and {{smap.MSL.chronos2.pointwise_auc_pr:.3f}}, level with ridge, principal-component and nearest-neighbour scoring and far below the published telemanom scores at {{smap.SMAP.telemanom.pointwise_auc_pr:.3f}} and {{smap.MSL.telemanom.pointwise_auc_pr:.3f}}. Two benchmark artefacts must be stated alongside these numbers. The always-on alarm reaches an uncorrected event $F_1$ of {{smap.SMAP.always_on.event_f1:.3f}} and {{smap.MSL.always_on.event_f1:.3f}}, higher than every detector, while its corrected $F_{0.5}$ is zero; and because {{smap.config.labelled_steps_in_second_half:pct}} of labelled steps lie in the second half of their series, the late-alarm reference outscores the detector itself in seven of the eight detector–spacecraft pairs. Figure [-@fig:smap-outputs] shows the scores and alarms on one channel of each spacecraft.

![**Health module outputs** on the median channel of each spacecraft by zero-shot AUC-PR. Top: test series and labels. Middle: conformal $p$-values with the alarm level. Bottom: labels and each detector's causal alarms.](artwork/Figure_16.png){#fig:smap-outputs width=100%}
