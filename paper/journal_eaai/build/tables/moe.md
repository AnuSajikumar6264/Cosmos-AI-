| Fusion arm | Type | Pinball | MAE | Cov. | Pinball | MAE | Cov. |
|---|---|--:|--:|--:|--:|--:|--:|
| Physics prior only ($g=1$) | single expert | 0.0892 | 0.230 | 0.876 | 0.0527 | 0.148 | 0.974 |
| TabM (2025) | single expert | 0.0593 | 0.192 | 0.877 | 0.0994 | 0.354 | 0.918 |
| TabICLv2 (2026) | single expert | 0.0499 | 0.166 | 0.962 | 0.0596 | 0.160 | 0.919 |
| Kumo-Tabular (2026) | single expert | 0.0508 | 0.162 | 0.974 | 0.0616 | 0.163 | 0.966 |
| XGBoost | baseline | 0.0576 | 0.190 | 0.921 | 0.0648 | 0.231 | 0.894 |
| Equal-weight average ($g=1/2$) | fixed fusion | 0.0691 | 0.205 | 0.838 | 0.0527 | 0.148 | 0.970 |
| Learned constant gate | learned fusion | 0.0601 | 0.193 | 0.877 | 0.0608 | 0.162 | 0.960 |
| Reliability gate fitted on tune-A | learned fusion | 0.0695 | 0.202 | 0.832 | 0.0616 | 0.163 | 0.959 |
| Reliability-gated mixture (proposed) | learned fusion | 0.0604 | 0.193 | 0.874 | 0.0583 | 0.157 | 0.959 |
