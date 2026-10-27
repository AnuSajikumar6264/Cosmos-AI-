---
title: "Supplementary Material"
subtitle: "Beyond model complexity: a pre-registered evaluation of event-level alarm protocols for spacecraft telemetry anomaly detection"
---

This file accompanies the manuscript. Every number is inserted from the study's evidence file (`thesis/evidence/evidence.json`), which records the result file each value comes from. Section S4 reports the validation analyses of the second frozen plan (`POSTHOC_PLAN_V2.md`, SHA-256 prefix `217648e6805f`, frozen 2026-10-25T12:14:37). Abbreviations: A0–A3, study arms; BH, Benjamini–Hochberg; ACI, adaptive conformal inference; M1, M2, Missions 1 and 2; FA, false alarms.

# S1. Pre-registration, deviation log and validation plan

The following documents are provided as separate files in the repository and in the archived release:

- `research_design/preregistration/PREREG_TELEMETRY_V1.md`: the pre-registration, frozen as `09a98f5e160e`, with 28 files hashed;
- `research_design/preregistration/DEVIATIONS.md`: every departure from the pre-registration, with its timing;
- `research_design/preregistration/freeze_09a98f5e160e.json` and `freeze.json`: the original freeze and the re-issued one (`681cc7ddec86`);
- `research_design/preregistration/POSTHOC_PLAN_V2.md` and `posthoc_v2_freeze.json`: the validation plan (P1–P8), frozen before any of its analyses was run.

# S2. Results reported on ESA-ADB

: **Table S1.** **Results reported on ESA-ADB.** Primary scores as published; the protocols differ, so the numbers are not comparable.

| Study | Protocol | Method | Reported score |
|---|---|---|---|
| @kotowski2026esaadb | official split; full and lightweight channels | TimeEval baselines, detector-specific thresholds | M1 full 0.061; M2 full 0.079; M1 lightweight 0.786 |
| @sahin2026tccad | official code; 24 / 19 channels; 16 seeds; pre-registered | telecommand conditioning; split, sliding and adaptive conformal thresholds | conditioning negative; point-level ACI collapses the event score |
| @nguyen2026 | own preprocessing; uncorrected event-wise metrics | supervised CNN and graph networks against unsupervised detectors | not comparable |
| @brancato2027ast | Mission 1, hierarchical protocol (as stated in the abstract) | similarity search (SMED); quantile GP regression with retraining | M1 full 0.902 (SMED) |
| @alcarria2026 | monthly incremental retraining | LSTM autoencoder; PCA, k-means and LOF; adaptive thresholds | M1 0.700; M2 0.698 |
| @allegrini2026 | Kaggle split of Mission 1, hidden test | supervised hierarchical ensemble | 0.853 (private leaderboard) |
| @heydari2026 | two missions, 77 shared channels | threshold transfer under alert budgets | policy diagnostics |
| This study | official split; full target channels; pre-registered | AR(8), XGBoost and TCN under one event-level layer | M1 0.428 (AR(8)), 0.544 (trees) |

::: {custom-style="Table Note"}
Note: GP, Gaussian process; LOF, local outlier factor; SMED, the similarity-search detector of Brancato et al.
:::

# S3. Models and pre-registered arms

: **Table S2.** **Model settings.** No setting was tuned on test data; the supervised scorers received no hyperparameter search.

| Model | Setting | Value |
|---|---|---|
| AR(8) | order; forecast horizon; maximum fitted rows | 8; 16 steps; 500,000 |
| AR(8) | out-of-fold blocks; purge | 5 contiguous blocks; 1 d on either side |
| AR(8) | score | absolute forecast error / max(residual scale, 0.05) |
| XGBoost (A2) | trees; depth; learning rate | 400; 8; 0.1 |
| XGBoost (A2) | row and column subsampling; minimum child weight | 0.8; 0.8; 5 |
| XGBoost (A2) | training rows; classes; class weights | 2,000,000 (half labelled); 3; [1.0, 5.0, 5.0] |
| TCN (A3) | levels; kernel; width; receptive field | 8; 3; 64; 1021 steps |
| TCN (A3) | cross-channel mixing; parameters (M1) | every 2 levels; 220,739 |
| TCN (A3) | optimiser; steps; learning rate; weight decay; dropout | AdamW, one-cycle; 1,500; 0.001; 0.0001; 0.1 |
| TCN (A3) | batch; window; share of windows with a labelled sample | 8; 2,048 steps; 0.5 |
| A2, A3 | per-step features | asinh z; log(1 + s); its EWMA over 1 h, 12 h, 3 d; deviation from trailing 1 d mean; forecast indicator |

: **Table S3.** **Secondary metrics, Mission 1.** Conformal-BH layer, test half, seed means.

| Arm | Precision | Recall | Alarming precision | ADTQC | Affiliation | Rare-only runs | Channel | Subsystem |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.061 | 0.462 | 0.276 | 0.821 | 0.564 | 81 | 0.219 | 0.308 |
| A1 | 0.667 | 0.176 | 0.139 | 0.862 | 0.445 | 11 | 0.077 | 0.134 |
| A2 | 0.627 | 0.415 | 0.213 | 0.732 | 0.627 | 136 | 0.248 | 0.348 |
| A3 | 0.211 | 0.266 | 0.326 | 0.646 | 0.463 | 42 | 0.106 | 0.218 |

::: {custom-style="Table Note"}
Note: ADTQC, alarm detection timing quality; affiliation, affiliation-based $F_{0.5}$; rare-only runs, alarm runs touching only rare events; channel and subsystem, channel- and subsystem-aware $F_{0.5}$.
:::

: **Table S4.** **Secondary metrics, Mission 2.** Conformal-BH layer, test half, seed means.

| Arm | Precision | Recall | Alarming precision | ADTQC | Affiliation | Rare-only runs |
|---|---|---|---|---|---|---|
| A0 | 0.982 | 0.892 | 0.895 | 0.897 | 0.931 | 351 |
| A1 | 1.000 | 0.580 | 0.857 | 0.930 | 0.733 | 238 |
| A2 | 0.975 | 0.710 | 0.901 | 0.969 | 0.814 | 277 |
| A3 | 0.958 | 0.676 | 0.885 | 0.779 | 0.791 | 268 |

::: {custom-style="Table Note"}
Note: ADTQC, alarm detection timing quality; affiliation, affiliation-based $F_{0.5}$; rare-only runs, alarm runs touching only rare events.
:::

: **Table S5.** **Alarm layers on identical scores.** Test-half event-wise $F_{0.5}$, seed means.

| Mission | Arm | Threshold | Conformal-BH | Rolling BH | ACI | ACI + events |
|---|---|---|---|---|---|---|
| 1 | A1 | 0.428 | 0.428 | 0.101 | 0.0001 | 0.008 |
| 1 | A2 | 0.613 | 0.544 | 0.087 | 0.0000 | 0.046 |
| 1 | A3 | 0.282 | 0.205 | 0.323 | 0.0001 | 0.003 |
| 1 | A3, no residual inputs | 0.276 | 0.249 | 0.000 | 0.0001 | 0.005 |
| 1 | A3, two classes | 0.363 | 0.196 | 0.084 | 0.0001 | 0.000 |
| 1 | A1, lightweight | 0.350 | 0.025 | 0.130 | 0.0001 | 0.000 |
| 2 | A1 | 0.977 | 0.874 | 0.356 | 0.0007 | 0.000 |
| 2 | A2 | 0.819 | 0.904 | 0.244 | 0.0007 | 0.013 |
| 2 | A3 | 0.853 | 0.875 | 0.487 | 0.0014 | 0.005 |
| 2 | A1, lightweight | 0.978 | 0.978 | 0.529 | 0.0017 | 0.000 |

::: {custom-style="Table Note"}
Note: ACI and ACI + events are the point-level comparator as pre-registered ($\gamma$ tuned to a point-wise rate).
:::

![**Fig. S1.** **Generality of the training-half protocol's effect.** Training-half protocol minus calibration-window protocol for AR(8), test-half event-wise $F_{0.5}$, by mission, channel set and alarm layer. The ringed point is the pre-registered H6 comparison; all other points are exploratory single runs.](artwork/Figure_S1.png){width=70%}

![**Fig. S2.** **Alarm layers on identical scores.** Test-half event-wise $F_{0.5}$ of the same scores under the threshold, conformal-BH and rolling layers (seed means). Coinciding markers mean equal values; ACI is omitted because all its values are below 0.002.](artwork/Figure_S2.png){width=60%}

: **Table S6.** **Per-seed results of the supervised arms.** Mission 1, conformal-BH layer, test half (pre-registered seeds 0–4).

| Seed | A2 $F_{0.5}$ | A2 FA / yr | A3 $F_{0.5}$ | A3 FA / yr |
|---|---|---|---|---|
| 0 | 0.653 | 1.7 | 0.085 | 67.7 |
| 1 | 0.157 | 33.9 | 0.358 | 4.1 |
| 2 | 0.504 | 4.6 | 0.135 | 9.7 |
| 3 | 0.635 | 2.0 | 0.158 | 13.7 |
| 4 | 0.769 | 0.1 | 0.290 | 9.1 |

: **Table S7.** **Exploratory phase.** AR(8) with the layer selected on the calibration window, and the label-free baselines; test-half $F_{0.5}$ (before the pre-registration).

| Detector | M1 full | M1 lightweight | M2 full | M2 lightweight |
|---|---|---|---|---|
| AR(8), conformal-BH | 0.0744 | 0.408 | 0.962 | 0.908 |
| AR(8), threshold | 0.0882 | 0.407 | 0.882 | 0.821 |
| out-of-limit rule | 0.0007 | 0.202 | 0.894 | 0.965 |
| GlobalSTD5 | 0.0029 | 0.253 | 0.063 | 0.075 |

![**Fig. S3.** **Calibration shift against test score (exploratory).** For each supervised run on Mission 1 (BH layer), the ratio of the nominal test exceedance rate to the cross-validation rate at the $1-10^{-4}$ quantile, against test-half event-wise $F_{0.5}$.](artwork/Figure_S3.png){width=60%}

# S4. Validation analyses (P1–P8)

All analyses in this section were specified and frozen before they were run, after the confirmatory analysis. They use test halves that had been used before.

**P3, exact bootstrap.** The block-decomposable metric of Appendix A was compared with the official value on 302 stored arm–metric pairs; the largest absolute difference was 2.1e-16 (criterion passed).

: **Table S8.** **P1 decomposition cells, Mission 1, target channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: passed.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | 0.074 | 91.6 | 0.01 | 10 |
| c1 | S0 | cal | cal | G1 | 0.074 | 91.6 | 0.01 | 10 |
| c2 | S½ | cal | cal | G1 | 0.084 | 52.0 | 0.001 | 10 |
| c3 | S1 | cal | cal | G1 | 0.084 | 52.0 | 0.001 | 10 |
| c4 | S1 | half | cal | G1 | 0.127 | 34.6 | 0.01 | 10 |
| c5 | S1 | half | half | G1 | 0.428 | 1.1 | 0.0001 | 10 |
| r1 | S1 | half | half | G0 | 0.493 | 1.6 | 0.0003 | 10 |
| r2 | S1 | half | cal | G0 | 0.127 | 34.6 | 0.01 | 10 |
| r3 | S1 | cal | cal | G0 | 0.084 | 52.0 | 0.001 | 10 |
| r4 | S½ | cal | cal | G0 | 0.084 | 52.0 | 0.001 | 10 |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S9.** **P1 steps, Mission 1, target channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | +0.000 [+0.000, +0.000] | -0.065 [-0.167, +0.014] | order-dependent |
| AR(8) fit data | +0.010 [-0.009, +0.035] | +0.010 [-0.009, +0.033] | not material |
| residual scale | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| calibration pool | +0.043 [+0.015, +0.083] | +0.043 [+0.014, +0.081] | not material |
| selection data | +0.301 [+0.156, +0.436] | +0.366 [+0.234, +0.500] | material |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S10.** **P1 decomposition cells, Mission 1, target channels, threshold layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: passed.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | 0.088 | 64.4 | 0.0001 | 10 |
| c1 | S0 | cal | cal | G1 | 0.088 | 64.4 | 0.0001 | 10 |
| c2 | S½ | cal | cal | G1 | 0.085 | 49.3 | 1e-05 | 10 |
| c3 | S1 | cal | cal | G1 | 0.085 | 49.3 | 1e-05 | 10 |
| c4 | S1 | half | cal | G1 | 0.231 | 13.1 | 0.0001 | 10 |
| c5 | S1 | half | half | G1 | 0.428 | 1.1 | 0 | 10 |
| r1 | S1 | half | half | G0 | 0.428 | 1.1 | 0 | 10 |
| r2 | S1 | half | cal | G0 | 0.231 | 13.1 | 0.0001 | 10 |
| r3 | S1 | cal | cal | G0 | 0.085 | 49.3 | 1e-05 | 10 |
| r4 | S½ | cal | cal | G0 | 0.085 | 49.3 | 1e-05 | 10 |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S11.** **P1 steps, Mission 1, target channels, threshold layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| AR(8) fit data | -0.003 [-0.021, +0.017] | -0.003 [-0.020, +0.016] | not material |
| residual scale | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| calibration pool | +0.145 [+0.071, +0.263] | +0.145 [+0.070, +0.265] | material |
| selection data | +0.197 [+0.060, +0.327] | +0.197 [+0.063, +0.338] | material |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S12.** **P1 decomposition cells, Mission 1, lightweight channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: passed.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | 0.408 | 5.0 | 0.001 | 3 |
| c1 | S0 | cal | cal | G1 | 0.408 | 5.0 | 0.001 | 3 |
| c2 | S½ | cal | cal | G1 | 0.340 | 3.0 | 0.01 | 10 |
| c3 | S1 | cal | cal | G1 | 0.340 | 3.0 | 0.01 | 10 |
| c4 | S1 | half | cal | G1 | 0.191 | 20.6 | 0.01 | 3 |
| c5 | S1 | half | half | G1 | 0.025 | 11.9 | 0.01 | 30 |
| r1 | S1 | half | half | G0 | 0.025 | 11.9 | 0.01 | 30 |
| r2 | S1 | half | cal | G0 | 0.191 | 20.6 | 0.01 | 3 |
| r3 | S1 | cal | cal | G0 | 0.413 | 3.1 | 0.0003 | 3 |
| r4 | S½ | cal | cal | G0 | 0.413 | 3.1 | 0.0003 | 3 |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S13.** **P1 steps, Mission 1, lightweight channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| AR(8) fit data | -0.068 [-0.159, +0.024] | +0.005 [-0.078, +0.083] | order-dependent |
| residual scale | +0.000 [+0.000, +0.000] | +0.000 [-0.017, +0.022] | not material |
| calibration pool | -0.150 [-0.258, -0.049] | -0.223 [-0.335, -0.110] | order-dependent |
| selection data | -0.166 [-0.243, -0.043] | -0.166 [-0.245, -0.042] | not material |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S14.** **P1 decomposition cells, Mission 2, target channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: passed.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | 0.962 | 3.4 | 0.01 | 30 |
| c1 | S0 | cal | cal | G1 | 0.962 | 3.4 | 0.01 | 30 |
| c2 | S½ | cal | cal | G1 | 0.965 | 2.9 | 0.01 | 30 |
| c3 | S1 | cal | cal | G1 | 0.966 | 2.9 | 0.01 | 30 |
| c4 | S1 | half | cal | G1 | 0.962 | 1.1 | 0.01 | 30 |
| c5 | S1 | half | half | G1 | 0.874 | 0.0 | 0.001 | 30 |
| r1 | S1 | half | half | G0 | 0.874 | 0.0 | 0.001 | 30 |
| r2 | S1 | half | cal | G0 | 0.962 | 1.1 | 0.01 | 30 |
| r3 | S1 | cal | cal | G0 | 0.966 | 2.9 | 0.01 | 30 |
| r4 | S½ | cal | cal | G0 | 0.965 | 2.9 | 0.01 | 30 |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S15.** **P1 steps, Mission 2, target channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| AR(8) fit data | +0.003 [-0.004, +0.011] | +0.003 [-0.004, +0.011] | not material |
| residual scale | +0.001 [-0.000, +0.002] | +0.001 [-0.000, +0.002] | not material |
| calibration pool | -0.004 [-0.017, +0.015] | -0.004 [-0.018, +0.014] | not material |
| selection data | -0.088 [-0.134, -0.054] | -0.088 [-0.134, -0.055] | not material |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S16.** **P1 decomposition cells, Mission 2, target channels, threshold layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: passed.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | 0.882 | 0.0 | 0 | 30 |
| c1 | S0 | cal | cal | G1 | 0.882 | 0.0 | 0 | 30 |
| c2 | S½ | cal | cal | G1 | 0.880 | 0.0 | 0 | 30 |
| c3 | S1 | cal | cal | G1 | 0.884 | 0.0 | 0 | 30 |
| c4 | S1 | half | cal | G1 | 0.977 | 1.7 | 0.001 | 30 |
| c5 | S1 | half | half | G1 | 0.977 | 1.7 | 0.001 | 30 |
| r1 | S1 | half | half | G0 | 0.977 | 1.7 | 0.001 | 30 |
| r2 | S1 | half | cal | G0 | 0.977 | 1.7 | 0.001 | 30 |
| r3 | S1 | cal | cal | G0 | 0.884 | 0.0 | 0 | 30 |
| r4 | S½ | cal | cal | G0 | 0.880 | 0.0 | 0 | 30 |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S17.** **P1 steps, Mission 2, target channels, threshold layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| AR(8) fit data | -0.002 [-0.011, +0.005] | -0.002 [-0.011, +0.005] | not material |
| residual scale | +0.005 [-0.001, +0.013] | +0.005 [-0.001, +0.012] | not material |
| calibration pool | +0.092 [+0.054, +0.142] | +0.092 [+0.055, +0.143] | material |
| selection data | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S18.** **P1 decomposition cells, Mission 2, lightweight channels, conformal-BH layer.** Each cell is a full protocol scored once on the test half. Reproduction of A0 and A1: passed.

| Cell | Scores | Pool | Selection data | Grid | $F_{0.5}$ | FA / yr | Level | $k$ |
|---|---|---|---|---|---|---|---|---|
| c0 | S0 | cal | cal | G0 | 0.908 | 9.2 | 0.01 | 30 |
| c1 | S0 | cal | cal | G1 | 0.908 | 9.2 | 0.01 | 30 |
| c2 | S½ | cal | cal | G1 | 0.913 | 18.3 | 0.001 | 10 |
| c3 | S1 | cal | cal | G1 | 0.903 | 21.2 | 0.001 | 10 |
| c4 | S1 | half | cal | G1 | 0.945 | 0.0 | 0.01 | 30 |
| c5 | S1 | half | half | G1 | 0.978 | 0.6 | 0.001 | 10 |
| r1 | S1 | half | half | G0 | 0.978 | 0.6 | 0.001 | 10 |
| r2 | S1 | half | cal | G0 | 0.945 | 0.0 | 0.01 | 30 |
| r3 | S1 | cal | cal | G0 | 0.903 | 21.2 | 0.001 | 10 |
| r4 | S½ | cal | cal | G0 | 0.913 | 18.3 | 0.001 | 10 |

::: {custom-style="Table Note"}
Note: Forward path c0 → c5 changes G, F, σ, P, D in turn; reverse path c5 → r1 → r2 → r3 → r4 → c0 changes G, D, P, σ, F. S½ is A1's forecast errors on A0's residual scale.
:::

: **Table S19.** **P1 steps, Mission 2, lightweight channels, conformal-BH layer.** Change in test-half $F_{0.5}$ when one factor is switched, with exact block-bootstrap 95 % intervals, and the verdict of the pre-set rule.

| Factor | Forward path | Reverse path | Verdict |
|---|---|---|---|
| grid | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | not material |
| AR(8) fit data | +0.005 [-0.027, +0.035] | +0.005 [-0.027, +0.035] | not material |
| residual scale | -0.009 [-0.019, -0.002] | -0.009 [-0.019, -0.002] | not material |
| calibration pool | +0.042 [-0.022, +0.132] | +0.042 [-0.025, +0.135] | not material |
| selection data | +0.033 [+0.018, +0.052] | +0.033 [+0.018, +0.051] | not material |

::: {custom-style="Table Note"}
Note: The rule defines *material* for gains only (≥ +0.05 on both paths); a factor that lowers the score on both paths is *not material* under it.
:::

: **Table S20.** **P2 forward chaining, Mission 1.** Per fold: events of the fold, $F_{0.5}$ and false alarms per year for the calibration-window protocol W and the training-history protocol H.

| Layer | Fold start | Events | W $F_{0.5}$ | H $F_{0.5}$ | H − W | W FA / yr | H FA / yr |
|---|---|---|---|---|---|---|---|
| conformal-BH | 2003-01-01 | 9 | 0.092 | 0.238 | +0.146 | 23.0 | 2.0 |
| conformal-BH | 2004-01-01 | 10 | 0.013 | 0.000 | -0.013 | 271.4 | 0.0 |
| conformal-BH | 2005-01-01 | 10 | 0.008 | 0.833 | +0.825 | 1172.8 | 0.0 |
| conformal-BH | 2006-01-01 | 10 | 0.357 | 0.357 | +0.000 | 0.0 | 0.0 |
| threshold | 2003-01-01 | 9 | 0.000 | 0.345 | +0.345 | 0.0 | 3.0 |
| threshold | 2004-01-01 | 10 | 0.016 | 0.000 | -0.016 | 76.8 | 0.0 |
| threshold | 2005-01-01 | 10 | 0.682 | 0.769 | +0.087 | 0.0 | 0.0 |
| threshold | 2006-01-01 | 10 | 0.357 | 0.278 | -0.079 | 0.0 | 1.0 |

::: {custom-style="Table Note"}
Note: Pooled over folds (BH): H − W = +0.455 [+0.189, +0.659]. Replication criterion met: no.
:::

: **Table S21.** **P2 forward chaining, Mission 2.** Per fold: events of the fold, $F_{0.5}$ and false alarms per year for the calibration-window protocol W and the training-history protocol H.

| Layer | Fold start | Events | W $F_{0.5}$ | H $F_{0.5}$ | H − W | W FA / yr | H FA / yr |
|---|---|---|---|---|---|---|---|
| conformal-BH | 2001-01-01 | 29 | 0.917 | 0.929 | +0.012 | 0.0 | 0.0 |
| conformal-BH | 2001-04-01 | 29 | 0.957 | 0.977 | +0.020 | 4.0 | 0.0 |
| conformal-BH | 2001-07-01 | 26 | 0.941 | 0.960 | +0.020 | 7.9 | 4.0 |
| threshold | 2001-01-01 | 29 | 0.917 | 0.940 | +0.023 | 0.0 | 0.0 |
| threshold | 2001-04-01 | 29 | 1.000 | 0.985 | -0.015 | 0.0 | 0.0 |
| threshold | 2001-07-01 | 26 | 0.960 | 0.951 | -0.009 | 4.0 | 4.0 |

::: {custom-style="Table Note"}
Note: Pooled over folds (BH): H − W = +0.017 [-0.007, +0.044]. Replication criterion met: no.
:::

: **Table S22.** **P4, ACI tuned on the event objective.** Chosen $(\alpha, \gamma, k)$ and test-half $F_{0.5}$ for the best setting overall, the best with $\gamma > 0$, and $\gamma = 0$ (a fixed per-channel conformal threshold).

| Arm | Best: $\alpha$, $\gamma$, $k$ | $F_{0.5}$ | Best with $\gamma>0$ | $F_{0.5}$ | $\gamma=0$ $F_{0.5}$ |
|---|---|---|---|---|---|
| M1 A1 | 0.0001, 0.0001, 10 | 0.472 | 0.0001, 0.0001, 10 | 0.472 | 0.261 |
| M1 A2 seed 0 | 0.0001, 0, 100 | 0.252 | 0.001, 0.0001, 100 | 0.407 | 0.252 |
| M1 A2 seed 1 | 0.001, 0.0001, 100 | 0.535 | 0.001, 0.0001, 100 | 0.535 | 0.212 |
| M1 A2 seed 2 | 0.0001, 0, 100 | 0.218 | 0.001, 0.0001, 100 | 0.513 | 0.218 |
| M1 A2 seed 3 | 0.001, 0.0001, 100 | 0.453 | 0.001, 0.0001, 100 | 0.453 | 0.265 |
| M1 A2 seed 4 | 0.0001, 0, 100 | 0.290 | 0.001, 0.0001, 100 | 0.510 | 0.290 |
| M2 A1 | 0.001, 0, 30 | 0.977 | 0.01, 0.0001, 30 | 0.925 | 0.977 |

: **Table S23.** **P5, registered anomaly-score arm, Mission 1.** Anomaly-only test-half $F_{0.5}$ with the layer selected on the anomaly-only objective.

| Arm | Runs | Mean | sd | Min | Max | FA / yr |
|---|---|---|---|---|---|---|
| A1 | 1 | 0.336 | 0.000 | 0.336 | 0.336 | 1.1 |
| A2 | 5 | 0.087 | 0.091 | 0.011 | 0.232 | 44.7 |
| A3 | 5 | 0.097 | 0.072 | 0.034 | 0.211 | 18.5 |

::: {custom-style="Table Note"}
Note: A1 is deterministic; A2 and A3 use their anomaly log-odds.
:::

: **Table S24.** **P5, registered anomaly-score arm, Mission 2.** Anomaly-only test-half $F_{0.5}$ with the layer selected on the anomaly-only objective.

| Arm | Runs | Mean | sd | Min | Max | FA / yr |
|---|---|---|---|---|---|---|
| A1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| A2 | 5 | 0.247 | 0.176 | 0.081 | 0.500 | 3.0 |
| A3 | 5 | 0.005 | 0.010 | 0.000 | 0.023 | 6.6 |

::: {custom-style="Table Note"}
Note: A1 is deterministic; A2 and A3 use their anomaly log-odds.
:::

: **Table S25.** **P6, seed extension.** Mission 1, conformal-BH layer: seed statistics with all available seeds, and the comparisons with the exact bootstrap (unadjusted, exploratory).

| Arm or comparison | Seeds | Mean $F_{0.5}$ / difference | sd | Min / 95 % CI | Max / $p$ |
|---|---|---|---|---|---|
| A2 | 10 | 0.488 | 0.195 | 0.157 | 0.769 |
| A3 | 9 | 0.176 | 0.093 | 0.082 | 0.358 |
| H4: A3 − A1 |  | -0.252 |  | [-0.402, -0.043] | p = 0.020 |
| H5: A3 − A2 |  | -0.313 |  | [-0.457, -0.112] | p = 0.002 |
| X1: A2 − A1 |  | +0.061 |  | [-0.138, +0.268] | p = 0.528 |

: **Table S26.** **P7, operational metrics.** Test halves, conformal-BH layer; seed means for A2 and A3.

| Mission | Arm | Detected / events | Anomalies | Rare events | Delay, median [IQR] (h) | Within 1 h | Within 24 h | Runs per detected event | FA per 1000 nominal h |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A0 | 42.0 / 91 | 17.0 | 25.0 | 0.35 [0.13, 6.54] | 0.67 | 0.86 | 2.0 | 11.749 |
| 1 | A1 | 16.0 / 91 | 8.0 | 8.0 | 0.33 [0.17, 3.73] | 0.62 | 0.88 | 1.0 | 0.147 |
| 1 | A2 | 37.8 / 91 | 11.8 | 26.0 | 1.07 [0.40, 2.52] | 0.47 | 0.95 | 1.4 | 1.085 |
| 1 | A3 | 24.2 / 91 | 11.8 | 12.4 | 2.02 [0.16, 5.75] | 0.43 | 0.92 | 1.6 | 2.680 |
| 2 | A0 | 323.0 / 362 | 1.0 | 322.0 | 0.16 [0.15, 0.19] | 0.97 | 1.00 | 1.0 | 0.411 |
| 2 | A1 | 210.0 / 362 | 0.0 | 210.0 | 0.16 [0.15, 0.19] | 0.95 | 1.00 | 1.0 | 0.000 |
| 2 | A2 | 257.2 / 362 | 0.8 | 256.4 | 0.07 [0.04, 0.10] | 0.98 | 1.00 | 1.0 | 0.425 |
| 2 | A3 | 244.8 / 362 | 0.8 | 244.0 | 0.27 [0.24, 0.30] | 0.97 | 1.00 | 1.0 | 0.918 |

::: {custom-style="Table Note"}
Note: Delay is first-alarm time minus event start for detected events; IQR, interquartile range.
:::

**P8, provenance.** Re-running the frozen selection for the A1 arms reproduced the saved choice and every grid value: Mission 1 passed (largest difference 0), Mission 2 passed (largest difference 0).

# S5. Code, data and reproduction

The repository contains the frozen code of the pre-registered study, the validation scripts (`scripts/posthoc_v2/`), the evidence builder (`thesis/tools/build_evidence.py`) and the manuscript build (`paper/journal_v2/tools/`). The ESA Anomaly Dataset is obtained from Zenodo (DOI 10.5281/zenodo.12528696). Derived score and alarm arrays will be deposited with the archived release.

# References

::: {#refs}
:::

