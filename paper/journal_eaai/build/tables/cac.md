| Model | Pinball | AUC-PR | AUC-ROC | $F_2$ | $L$ | Cov. 90 % | Cov. high risk |
|---|--:|--:|--:|--:|--:|--:|--:|
| Persistence + residual climatology | 1.045 | 0.664 | 0.944 | 0.000 | $\infty$ | 0.910 | 0.593 |
| XGBoost | 0.855 | 0.606 | 0.945 | 0.000 | $\infty$ | 0.917 | 0.433 |
| LightGBM | 0.861 | 0.541 | 0.944 | 0.000 | $\infty$ | 0.905 | 0.253 |
| CatBoost | 0.869 | 0.542 | 0.935 | 0.000 | $\infty$ | 0.912 | 0.467 |
| Quantile multilayer perceptron | 1.035 | 0.434 | 0.901 | 0.000 | $\infty$ | 0.936 | 0.853 |
| TabM (2025) | 0.885 | 0.512 | 0.937 | 0.000 | $\infty$ | 0.919 | 0.727 |
| TabICLv2 (2026) | 0.692 | 0.665 | 0.961 | 0.057 | 30.06 | 0.918 | 0.327 |
| Kumo-Tabular (2026) | 0.653 | 0.733 | 0.970 | 0.168 | 9.89 | 0.942 | 0.480 |
| Learned constant gate | 0.653 | 0.733 | 0.970 | 0.168 | 9.90 | 0.935 | 0.480 |
| Reliability-gated mixture (proposed) | 0.653 | 0.733 | 0.970 | 0.168 | 9.90 | 0.940 | 0.480 |
| Latest-risk point forecast (challenge baseline) |  |  |  |  | 0.69 |  |  |
