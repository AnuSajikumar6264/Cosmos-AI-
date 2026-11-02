# Supplementary material: COSMOS-AI digital twin

All tables are generated from the result files named under each table (`paper/journal_v3_twin/tools/make_supplementary.py`).

## Supplementary figures

![**Figure S1.** Tune-A pinball loss, used to select the expert, against test pinball loss, every learner. The selected learner is outlined. A tuning cohort five to seven years older than the test cohort did not rank the learners as the test cohort did.](/data/anupa/COSMOS_work/Final_Project/paper/journal_v3_twin/artwork/Figure_S1.png){width=100%}

![**Figure S2.** Empirical 90 % coverage against mean width of the 90 % Mondrian conformal interval, for the physics expert, the six learners and the gated model (data of Table S0; the fixed-weight fusions and the ablations of Table S0 are not plotted). The dashed line is the nominal level.](/data/anupa/COSMOS_work/Final_Project/paper/journal_v3_twin/artwork/Figure_S2.png){width=100%}

![**Figure S3.** Re-entry test pinball loss by object class for the main models (post hoc; data of Table S2). The counts in each panel title are the fragments, rocket stages and components of that test cohort.](/data/anupa/COSMOS_work/Final_Project/paper/journal_v3_twin/artwork/Figure_S3.png){width=100%}

## Table S0. Re-entry, all candidates (run 2, the reported run)

| origin | model | pinball | mae_log10 | rmse_log10 | medae_days_decayed | cov80 | cov90 | width90_log10 | winkler90 | raw_cov90 | c_index |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | tabicl | 0.0499 | 0.1657 | 0.3419 | 170.7512 | 0.9019 | 0.9621 | 1.0031 | 1.1288 | 0.9584 | 0.8938 |
| 2016 | kumo | 0.0508 | 0.1624 | 0.3402 | 176.3003 | 0.9459 | 0.9738 | 1.0691 | 1.1607 | 0.9731 | 0.8988 |
| 2016 | tabm_no_physics | 0.0543 | 0.1752 | 0.3398 | 179.1792 | 0.7864 | 0.8966 | 0.6793 | 1.0860 | 0.8559 | 0.8769 |
| 2016 | xgboost | 0.0576 | 0.1898 | 0.3519 | 201.8214 | 0.8026 | 0.9211 | 0.8001 | 1.0924 | 0.8842 | 0.8877 |
| 2016 | mlp | 0.0578 | 0.1929 | 0.3503 | 225.4566 | 0.7774 | 0.8915 | 0.8351 | 1.1957 | 0.8768 | 0.8510 |
| 2016 | tabm | 0.0593 | 0.1919 | 0.3566 | 216.7584 | 0.7277 | 0.8768 | 0.6772 | 1.0992 | 0.8083 | 0.8607 |
| 2016 | tabm_marginal_conformal | 0.0593 | 0.1919 | 0.3566 | 216.7584 | 0.7367 | 0.9218 | 0.6506 | 1.1033 | 0.8083 | 0.8607 |
| 2016 | stack_constant | 0.0601 | 0.1929 | 0.3590 | 223.8272 | 0.7169 | 0.8771 | 0.6736 | 1.0996 | 0.7965 | 0.8607 |
| 2016 | cosmos_gated_marginal_conformal | 0.0604 | 0.1933 | 0.3597 | 233.5241 | 0.7354 | 0.9204 | 0.6432 | 1.1193 | 0.7921 | 0.8602 |
| 2016 | cosmos_gated | 0.0604 | 0.1933 | 0.3597 | 233.5241 | 0.7169 | 0.8741 | 0.6707 | 1.1002 | 0.7921 | 0.8602 |
| 2016 | catboost | 0.0611 | 0.2065 | 0.3737 | 214.2932 | 0.8073 | 0.9107 | 0.9276 | 1.2152 | 0.7794 | 0.8778 |
| 2016 | tabm_no_space_weather | 0.0631 | 0.2033 | 0.3643 | 209.2504 | 0.7327 | 0.8539 | 0.7058 | 1.1654 | 0.7999 | 0.8584 |
| 2016 | tabm_no_family | 0.0649 | 0.2081 | 0.3714 | 250.6741 | 0.7129 | 0.8791 | 0.8591 | 1.2895 | 0.8425 | 0.8462 |
| 2016 | lightgbm | 0.0662 | 0.1909 | 0.3533 | 148.2671 | 0.9120 | 0.9708 | 0.9958 | 1.1588 | 0.9523 | 0.8874 |
| 2016 | tabm_no_design | 0.0664 | 0.2063 | 0.3829 | 226.4505 | 0.7492 | 0.8529 | 0.6734 | 1.3613 | 0.8073 | 0.8601 |
| 2016 | tabm_plus_rcs_2026 | 0.0664 | 0.2114 | 0.3474 | 148.2818 | 0.6833 | 0.7683 | 0.7707 | 1.4072 | 0.6891 | 0.8728 |
| 2016 | equal_average | 0.0691 | 0.2046 | 0.3874 | 301.7900 | 0.7045 | 0.8385 | 0.6393 | 1.2854 | 0.7196 | 0.8594 |
| 2016 | cosmos_gated_tuneA | 0.0695 | 0.2020 | 0.3872 | 317.5479 | 0.6954 | 0.8324 | 0.6152 | 1.3481 | 0.6985 | 0.8627 |
| 2016 | physics_only | 0.0892 | 0.2303 | 0.4502 | 396.1003 | 0.7381 | 0.8764 | 0.6706 | 1.6001 | 0.7089 | 0.7388 |
| 2011 | physics_only | 0.0527 | 0.1478 | 0.3784 | 179.2500 | 0.9438 | 0.9738 | 1.0284 | 1.3025 | 0.8886 | 0.7660 |
| 2011 | equal_average | 0.0527 | 0.1479 | 0.3516 | 117.2448 | 0.9288 | 0.9696 | 1.0369 | 1.2638 | 0.9303 | 0.8512 |
| 2011 | kumo_plus_rcs_2026 | 0.0578 | 0.1649 | 0.3828 | 238.0633 | 0.9318 | 0.9614 | 1.1593 | 1.3789 | 0.9580 | 0.8224 |
| 2011 | cosmos_gated | 0.0583 | 0.1570 | 0.3592 | 122.0176 | 0.9231 | 0.9591 | 1.2139 | 1.3914 | 0.9486 | 0.8481 |
| 2011 | cosmos_gated_marginal_conformal | 0.0583 | 0.1570 | 0.3592 | 122.0176 | 0.9085 | 0.9486 | 1.1933 | 1.4036 | 0.9486 | 0.8481 |
| 2011 | tabicl | 0.0596 | 0.1604 | 0.3598 | 114.9210 | 0.8211 | 0.9194 | 1.2858 | 1.4760 | 0.9093 | 0.8443 |
| 2011 | stack_constant | 0.0608 | 0.1618 | 0.3642 | 121.2032 | 0.9243 | 0.9599 | 1.3104 | 1.4666 | 0.9501 | 0.8465 |
| 2011 | cosmos_gated_tuneA | 0.0616 | 0.1633 | 0.3665 | 119.3121 | 0.9243 | 0.9588 | 1.3422 | 1.4929 | 0.9486 | 0.8454 |
| 2011 | kumo | 0.0616 | 0.1633 | 0.3665 | 119.2795 | 0.9329 | 0.9655 | 1.3424 | 1.4931 | 0.9580 | 0.8453 |
| 2011 | kumo_marginal_conformal | 0.0616 | 0.1633 | 0.3665 | 119.2795 | 0.9209 | 0.9580 | 1.3244 | 1.5055 | 0.9580 | 0.8453 |
| 2011 | kumo_no_design | 0.0624 | 0.1626 | 0.3650 | 118.3518 | 0.9329 | 0.9685 | 1.3712 | 1.5159 | 0.9606 | 0.8441 |
| 2011 | kumo_no_space_weather | 0.0625 | 0.1417 | 0.3652 | 115.1775 | 0.9340 | 0.9678 | 1.4059 | 1.5890 | 0.9648 | 0.8704 |
| 2011 | catboost | 0.0627 | 0.2005 | 0.3714 | 163.4023 | 0.7285 | 0.7882 | 0.9776 | 1.2586 | 0.4244 | 0.8211 |
| 2011 | kumo_no_physics | 0.0639 | 0.1632 | 0.3635 | 126.4689 | 0.9393 | 0.9704 | 1.4474 | 1.5763 | 0.9640 | 0.8493 |
| 2011 | kumo_no_family | 0.0640 | 0.2023 | 0.4044 | 128.4458 | 0.8924 | 0.9614 | 1.1721 | 1.3398 | 0.9509 | 0.8307 |
| 2011 | xgboost | 0.0648 | 0.2313 | 0.3945 | 157.5153 | 0.7900 | 0.8935 | 1.0054 | 1.2279 | 0.6955 | 0.8075 |
| 2011 | lightgbm | 0.0683 | 0.2210 | 0.3853 | 138.7507 | 0.9606 | 0.9809 | 1.1824 | 1.3429 | 0.9531 | 0.8113 |
| 2011 | tabm | 0.0994 | 0.3540 | 0.5996 | 171.2114 | 0.8774 | 0.9175 | 1.3766 | 1.6903 | 0.7559 | 0.7829 |
| 2011 | mlp | 0.1147 | 0.3903 | 0.6829 | 165.8177 | 0.8196 | 0.8519 | 1.3977 | 1.9293 | 0.7027 | 0.7475 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/regression_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/regression_test.csv`. 

## Table S1. Re-entry, first learner set (run 1; XGBoost, LightGBM, CatBoost, MLP, TabICLv2; deviation record)

| origin | model | pinball | mae_log10 | rmse_log10 | medae_days_decayed | cov80 | cov90 | width90_log10 | winkler90 | raw_cov90 | c_index |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | tabicl | 0.0499 | 0.1657 | 0.3419 | 170.7512 | 0.9019 | 0.9621 | 1.0031 | 1.1288 | 0.9584 | 0.8938 |
| 2016 | xgboost | 0.0576 | 0.1898 | 0.3519 | 201.8214 | 0.8026 | 0.9211 | 0.8001 | 1.0924 | 0.8842 | 0.8877 |
| 2016 | mlp | 0.0578 | 0.1929 | 0.3503 | 225.4566 | 0.7774 | 0.8915 | 0.8351 | 1.1957 | 0.8768 | 0.8510 |
| 2016 | mlp_marginal_conformal | 0.0578 | 0.1929 | 0.3503 | 225.4566 | 0.7488 | 0.8768 | 0.7646 | 1.2000 | 0.8768 | 0.8510 |
| 2016 | stack_constant | 0.0586 | 0.1955 | 0.3574 | 249.8953 | 0.7599 | 0.8892 | 0.7846 | 1.1732 | 0.8711 | 0.8532 |
| 2016 | cosmos_gated | 0.0592 | 0.1948 | 0.3615 | 271.4207 | 0.7411 | 0.8751 | 0.7269 | 1.1690 | 0.8519 | 0.8535 |
| 2016 | cosmos_gated_marginal_conformal | 0.0592 | 0.1948 | 0.3615 | 271.4207 | 0.7770 | 0.8801 | 0.6325 | 1.1809 | 0.8519 | 0.8535 |
| 2016 | mlp_no_family | 0.0604 | 0.1993 | 0.3575 | 230.9672 | 0.7707 | 0.8862 | 0.9023 | 1.2935 | 0.8825 | 0.8421 |
| 2016 | mlp_plus_rcs_2026 | 0.0610 | 0.2034 | 0.3464 | 136.3972 | 0.6041 | 0.7210 | 0.9606 | 1.3192 | 0.7032 | 0.8514 |
| 2016 | catboost | 0.0611 | 0.2065 | 0.3737 | 214.2932 | 0.8073 | 0.9107 | 0.9276 | 1.2152 | 0.7794 | 0.8778 |
| 2016 | mlp_no_design | 0.0641 | 0.2102 | 0.3752 | 219.8173 | 0.7599 | 0.8808 | 0.8526 | 1.2790 | 0.8341 | 0.8521 |
| 2016 | mlp_no_physics | 0.0644 | 0.2113 | 0.3934 | 206.4974 | 0.7451 | 0.8690 | 0.8771 | 1.2559 | 0.8455 | 0.8528 |
| 2016 | mlp_no_space_weather | 0.0647 | 0.2117 | 0.3908 | 241.5738 | 0.7102 | 0.8556 | 0.7778 | 1.2269 | 0.8039 | 0.8396 |
| 2016 | cosmos_gated_tuneA | 0.0650 | 0.2004 | 0.3820 | 296.6999 | 0.7105 | 0.8684 | 0.6611 | 1.2775 | 0.7794 | 0.8558 |
| 2016 | equal_average | 0.0653 | 0.2060 | 0.3860 | 296.1305 | 0.6988 | 0.8566 | 0.6961 | 1.2421 | 0.8015 | 0.8536 |
| 2016 | lightgbm | 0.0662 | 0.1909 | 0.3533 | 148.2671 | 0.9120 | 0.9708 | 0.9958 | 1.1588 | 0.9523 | 0.8874 |
| 2016 | physics_only | 0.0892 | 0.2303 | 0.4502 | 396.1003 | 0.7381 | 0.8764 | 0.6706 | 1.6001 | 0.7089 | 0.7388 |
| 2011 | physics_only | 0.0527 | 0.1478 | 0.3784 | 179.2500 | 0.9438 | 0.9738 | 1.0284 | 1.3025 | 0.8886 | 0.7660 |
| 2011 | tabicl | 0.0596 | 0.1604 | 0.3598 | 114.9210 | 0.8211 | 0.9194 | 1.2858 | 1.4760 | 0.9093 | 0.8443 |
| 2011 | catboost | 0.0627 | 0.2005 | 0.3714 | 163.4023 | 0.7285 | 0.7882 | 0.9776 | 1.2586 | 0.4244 | 0.8211 |
| 2011 | mlp_no_family | 0.0633 | 0.1885 | 0.3919 | 214.3297 | 0.8826 | 0.8740 | 1.0963 | 1.4961 | 0.8436 | 0.7885 |
| 2011 | xgboost | 0.0648 | 0.2313 | 0.3945 | 157.5153 | 0.7900 | 0.8935 | 1.0054 | 1.2279 | 0.6955 | 0.8075 |
| 2011 | lightgbm | 0.0683 | 0.2210 | 0.3853 | 138.7507 | 0.9606 | 0.9809 | 1.1824 | 1.3429 | 0.9531 | 0.8113 |
| 2011 | equal_average | 0.0796 | 0.2581 | 0.4574 | 156.9555 | 0.8733 | 0.9149 | 1.1740 | 1.5337 | 0.6865 | 0.7833 |
| 2011 | mlp_no_space_weather | 0.0825 | 0.2646 | 0.4630 | 184.7546 | 0.8140 | 0.8958 | 1.1831 | 1.5485 | 0.7563 | 0.7966 |
| 2011 | mlp_no_design | 0.0849 | 0.2717 | 0.4753 | 178.1871 | 0.8688 | 0.9145 | 1.3580 | 1.6789 | 0.7795 | 0.7814 |
| 2011 | mlp_no_physics | 0.0902 | 0.2578 | 0.4454 | 157.6230 | 0.8733 | 0.8853 | 1.4272 | 1.7164 | 0.7529 | 0.8086 |
| 2011 | cosmos_gated_tuneA | 0.0986 | 0.3327 | 0.5827 | 170.8924 | 0.8590 | 0.9055 | 1.2780 | 1.7084 | 0.6959 | 0.7595 |
| 2011 | cosmos_gated_marginal_conformal | 0.1074 | 0.3637 | 0.6345 | 165.5545 | 0.7019 | 0.7398 | 1.1889 | 2.1117 | 0.6970 | 0.7534 |
| 2011 | cosmos_gated | 0.1074 | 0.3637 | 0.6345 | 165.5545 | 0.8549 | 0.9029 | 1.3466 | 1.8112 | 0.6970 | 0.7534 |
| 2011 | stack_constant | 0.1085 | 0.3679 | 0.6411 | 163.2117 | 0.8478 | 0.9036 | 1.3598 | 1.8236 | 0.6978 | 0.7527 |
| 2011 | mlp | 0.1147 | 0.3903 | 0.6829 | 165.8177 | 0.8196 | 0.8519 | 1.3977 | 1.9293 | 0.7027 | 0.7475 |
| 2011 | mlp_marginal_conformal | 0.1147 | 0.3903 | 0.6829 | 165.8177 | 0.6828 | 0.7150 | 1.2351 | 2.3738 | 0.7027 | 0.7475 |
| 2011 | mlp_plus_rcs_2026 | 0.1221 | 0.3707 | 0.6568 | 174.8182 | 0.7919 | 0.8470 | 1.3487 | 2.5471 | 0.6974 | 0.7616 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/run1_learner_set_v1/regression_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/run1_learner_set_v1/regression_test.csv`. The expert was selected among all five learners (the MLP at both origins).

## Table S2. Re-entry test pinball loss by object class (post hoc)

| origin | model | C | D | R |
|---|---|---|---|---|
| 2011 | catboost | 0.0725 | 0.0612 | 0.0618 |
| 2011 | cosmos_gated | 0.0623 | 0.0590 | 0.0473 |
| 2011 | cosmos_gated_tuneA | 0.0630 | 0.0631 | 0.0476 |
| 2011 | equal_average | 0.0714 | 0.0498 | 0.0523 |
| 2011 | kumo | 0.0630 | 0.0631 | 0.0476 |
| 2011 | kumo_no_design | 0.0629 | 0.0637 | 0.0515 |
| 2011 | kumo_no_family | 0.0616 | 0.0666 | 0.0463 |
| 2011 | kumo_no_physics | 0.0640 | 0.0660 | 0.0466 |
| 2011 | kumo_no_space_weather | 0.0641 | 0.0642 | 0.0465 |
| 2011 | kumo_plus_rcs_2026 | 0.0622 | 0.0589 | 0.0435 |
| 2011 | lightgbm | 0.0884 | 0.0610 | 0.1014 |
| 2011 | mlp | 0.0685 | 0.1302 | 0.0489 |
| 2011 | physics_only | 0.0959 | 0.0432 | 0.0733 |
| 2011 | stack_constant | 0.0629 | 0.0621 | 0.0473 |
| 2011 | tabicl | 0.0703 | 0.0594 | 0.0469 |
| 2011 | tabm | 0.0597 | 0.1122 | 0.0473 |
| 2011 | xgboost | 0.0689 | 0.0645 | 0.0624 |
| 2016 | catboost | 0.0858 | 0.0555 | 0.0641 |
| 2016 | cosmos_gated | 0.0770 | 0.0579 | 0.0545 |
| 2016 | cosmos_gated_tuneA | 0.0954 | 0.0644 | 0.0673 |
| 2016 | equal_average | 0.0980 | 0.0638 | 0.0645 |
| 2016 | kumo | 0.0521 | 0.0510 | 0.0477 |
| 2016 | lightgbm | 0.0997 | 0.0559 | 0.0846 |
| 2016 | mlp | 0.0740 | 0.0551 | 0.0535 |
| 2016 | physics_only | 0.1331 | 0.0809 | 0.0845 |
| 2016 | stack_constant | 0.0771 | 0.0575 | 0.0548 |
| 2016 | tabicl | 0.0608 | 0.0482 | 0.0471 |
| 2016 | tabm | 0.0744 | 0.0570 | 0.0543 |
| 2016 | tabm_no_design | 0.1253 | 0.0562 | 0.0539 |
| 2016 | tabm_no_family | 0.0943 | 0.0592 | 0.0619 |
| 2016 | tabm_no_physics | 0.0633 | 0.0521 | 0.0564 |
| 2016 | tabm_no_space_weather | 0.0730 | 0.0618 | 0.0586 |
| 2016 | tabm_plus_rcs_2026 | 0.0801 | 0.0637 | 0.0658 |
| 2016 | xgboost | 0.0710 | 0.0546 | 0.0586 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/class_breakdown_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/class_breakdown_test.csv`. D fragments, R rocket stages, C components.

## Table S3. Post-hoc sensitivity analysis (added after the origin-2016 results were seen): the reliability gate applied to every learner

| origin | learner | pinball_alone | pinball_gated | gain | gain_ci95_lo | gain_ci95_hi | mae_log10_alone | mae_log10_gated | cov90_alone_marginal | cov90_gated_marginal | gate_weight_test_mean | oof_pinball |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | xgboost | 0.0576 | 0.0590 | -0.0014 | -0.0043 | 0.0010 | 0.1898 | 0.1916 | 0.9184 | 0.9167 | 0.2629 | 0.0750 |
| 2016 | lightgbm | 0.0662 | 0.0684 | -0.0023 | -0.0107 | 0.0057 | 0.1909 | 0.2169 | 0.9523 | 0.8835 | 0.5146 | 0.1175 |
| 2016 | catboost | 0.0611 | 0.0614 | -0.0003 | -0.0027 | 0.0015 | 0.2065 | 0.2072 | 0.9070 | 0.9177 | 0.2281 | 0.0804 |
| 2016 | mlp | 0.0578 | 0.0592 | -0.0014 | -0.0049 | 0.0017 | 0.1929 | 0.1948 | 0.8768 | 0.8801 | 0.3099 | 0.0769 |
| 2016 | tabm | 0.0593 | 0.0609 | -0.0016 | -0.0038 | 0.0001 | 0.1919 | 0.1941 | 0.9218 | 0.9238 | 0.1527 | 0.0752 |
| 2016 | tabicl | 0.0499 | 0.0546 | -0.0046 | -0.0112 | 0.0007 | 0.1657 | 0.1817 | 0.9584 | 0.9238 | 0.5060 | 0.0753 |
| 2016 | kumo | 0.0508 | 0.0494 | 0.0014 | -0.0023 | 0.0045 | 0.1624 | 0.1723 | 0.9731 | 0.9530 | 0.3627 | 0.0687 |
| 2011 | xgboost | 0.0648 | 0.0648 | 0.0000 | 0.0000 | 0.0000 | 0.2313 | 0.2312 | 0.9591 | 0.9591 | 0.0012 | 0.0746 |
| 2011 | lightgbm | 0.0683 | 0.0578 | 0.0105 | 0.0069 | 0.0161 | 0.2210 | 0.2084 | 0.9663 | 0.9693 | 0.2834 | 0.1225 |
| 2011 | catboost | 0.0627 | 0.0626 | 0.0000 | 0.0000 | 0.0000 | 0.2005 | 0.2005 | 0.9438 | 0.9434 | 0.0008 | 0.0772 |
| 2011 | mlp | 0.1147 | 0.1074 | 0.0073 | 0.0009 | 0.0147 | 0.3903 | 0.3637 | 0.7150 | 0.7398 | 0.1007 | 0.0745 |
| 2011 | tabm | 0.0994 | 0.0951 | 0.0043 | 0.0003 | 0.0088 | 0.3540 | 0.3367 | 0.8549 | 0.8493 | 0.0699 | 0.0736 |
| 2011 | tabicl | 0.0596 | 0.0569 | 0.0027 | 0.0007 | 0.0043 | 0.1604 | 0.1553 | 0.9093 | 0.9299 | 0.1119 | 0.0676 |
| 2011 | kumo | 0.0616 | 0.0583 | 0.0033 | 0.0007 | 0.0055 | 0.1633 | 0.1570 | 0.9580 | 0.9486 | 0.1186 | 0.0666 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/gate_all_learners.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/gate_all_learners.csv`. gain > 0: gating helps; nominal 95 % cluster-bootstrap CI, not corrected for multiplicity.

## Table S4. Feature-group ablations of the selected learner

| origin | learner | ablation | pinball_ablation_minus_full | ci95_lo | ci95_hi |
|---|---|---|---|---|---|
| 2016 | tabm | no_design | 0.0071 | 0.0035 | 0.0121 |
| 2016 | tabm | no_family | 0.0056 | 0.0015 | 0.0101 |
| 2016 | tabm | no_physics | -0.0050 | -0.0086 | -0.0022 |
| 2016 | tabm | no_space_weather | 0.0038 | -0.0057 | 0.0129 |
| 2016 | tabm | plus_rcs_2026 | 0.0071 | -0.0010 | 0.0151 |
| 2011 | kumo | no_design | 0.0008 | -0.0001 | 0.0017 |
| 2011 | kumo | no_family | 0.0024 | -0.0058 | 0.0103 |
| 2011 | kumo | no_physics | 0.0022 | 0.0002 | 0.0039 |
| 2011 | kumo | no_space_weather | 0.0009 | -0.0016 | 0.0038 |
| 2011 | kumo | plus_rcs_2026 | -0.0038 | -0.0072 | 0.0011 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/ablation_effects_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/ablation_effects_test.csv`. > 0: removing the group (or adding the leaky RCS) worsens the test pinball loss.

## Table S5. Operational +/-20 % and +/-30 % windows against conformal intervals

| origin | rule | coverage | width_log10 | coverage_decayed |
|---|---|---|---|---|
| 2016 | physics_only_pm20 | 0.6021 | 0.1754 | 0.1223 |
| 2016 | physics_only_pm30 | 0.6383 | 0.2677 | 0.2036 |
| 2016 | physics_only_conformal80 | 0.7381 | 0.3544 | 0.4195 |
| 2016 | physics_only_conformal90 | 0.8764 | 0.6706 | 0.7257 |
| 2016 | cosmos_gated_pm20 | 0.5285 | 0.1761 | 0.1831 |
| 2016 | cosmos_gated_pm30 | 0.5991 | 0.2688 | 0.2796 |
| 2016 | cosmos_gated_conformal80 | 0.7169 | 0.4329 | 0.5426 |
| 2016 | cosmos_gated_conformal90 | 0.8741 | 0.6707 | 0.8351 |
| 2011 | physics_only_pm20 | 0.7469 | 0.1750 | 0.1713 |
| 2011 | physics_only_pm30 | 0.7657 | 0.2669 | 0.2343 |
| 2011 | physics_only_conformal80 | 0.9438 | 0.6534 | 0.8053 |
| 2011 | physics_only_conformal90 | 0.9738 | 1.0284 | 0.9063 |
| 2011 | cosmos_gated_pm20 | 0.6513 | 0.1761 | 0.2006 |
| 2011 | cosmos_gated_pm30 | 0.6944 | 0.2688 | 0.2826 |
| 2011 | cosmos_gated_conformal80 | 0.9231 | 0.8768 | 0.7760 |
| 2011 | cosmos_gated_conformal90 | 0.9591 | 1.2139 | 0.8843 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/interval_rules.json, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/interval_rules.json`. coverage_decayed: objects that decayed within five years.

## Table S6. Horizon probabilities (AUC-PR, AUC-ROC on raw scores; Brier, ECE after tune-B isotonic calibration)

| origin | model | horizon | n | positives | auc_pr | auc_roc | brier | ece |
|---|---|---|---|---|---|---|---|---|
| 2016 | physics_only | 30d | 2979 | 312 | 0.719 | 0.963 | 0.051 | 0.048 |
| 2016 | physics_only | 6mo | 2979 | 587 | 0.842 | 0.934 | 0.064 | 0.033 |
| 2016 | physics_only | 1yr | 2979 | 691 | 0.853 | 0.920 | 0.069 | 0.028 |
| 2016 | physics_only | 5yr | 2978 | 1316 | 0.827 | 0.844 | 0.133 | 0.047 |
| 2016 | xgboost | 30d | 2979 | 312 | 0.838 | 0.977 | 0.055 | 0.081 |
| 2016 | xgboost | 6mo | 2979 | 587 | 0.891 | 0.959 | 0.079 | 0.091 |
| 2016 | xgboost | 1yr | 2979 | 691 | 0.899 | 0.954 | 0.078 | 0.073 |
| 2016 | xgboost | 5yr | 2978 | 1316 | 0.912 | 0.924 | 0.125 | 0.091 |
| 2016 | tabm | 30d | 2979 | 312 | 0.834 | 0.978 | 0.051 | 0.075 |
| 2016 | tabm | 6mo | 2979 | 587 | 0.901 | 0.972 | 0.076 | 0.091 |
| 2016 | tabm | 1yr | 2979 | 691 | 0.912 | 0.969 | 0.086 | 0.089 |
| 2016 | tabm | 5yr | 2978 | 1316 | 0.884 | 0.901 | 0.142 | 0.073 |
| 2016 | tabicl | 30d | 2979 | 312 | 0.874 | 0.982 | 0.035 | 0.043 |
| 2016 | tabicl | 6mo | 2979 | 587 | 0.910 | 0.973 | 0.060 | 0.049 |
| 2016 | tabicl | 1yr | 2979 | 691 | 0.923 | 0.970 | 0.070 | 0.055 |
| 2016 | tabicl | 5yr | 2978 | 1316 | 0.929 | 0.931 | 0.098 | 0.039 |
| 2016 | kumo | 30d | 2979 | 312 | 0.883 | 0.985 | 0.036 | 0.046 |
| 2016 | kumo | 6mo | 2979 | 587 | 0.915 | 0.971 | 0.052 | 0.035 |
| 2016 | kumo | 1yr | 2979 | 691 | 0.922 | 0.964 | 0.060 | 0.039 |
| 2016 | kumo | 5yr | 2978 | 1316 | 0.953 | 0.960 | 0.081 | 0.038 |
| 2016 | cosmos_gated | 30d | 2979 | 312 | 0.829 | 0.977 | 0.051 | 0.073 |
| 2016 | cosmos_gated | 6mo | 2979 | 587 | 0.900 | 0.971 | 0.074 | 0.089 |
| 2016 | cosmos_gated | 1yr | 2979 | 691 | 0.911 | 0.968 | 0.082 | 0.085 |
| 2016 | cosmos_gated | 5yr | 2978 | 1316 | 0.884 | 0.901 | 0.141 | 0.073 |
| 2016 | stack_constant | 30d | 2979 | 312 | 0.828 | 0.977 | 0.051 | 0.074 |
| 2016 | stack_constant | 6mo | 2979 | 587 | 0.901 | 0.971 | 0.076 | 0.090 |
| 2016 | stack_constant | 1yr | 2979 | 691 | 0.911 | 0.969 | 0.082 | 0.086 |
| 2016 | stack_constant | 5yr | 2978 | 1316 | 0.884 | 0.901 | 0.141 | 0.073 |
| 2016 | equal_average | 30d | 2979 | 312 | 0.788 | 0.971 | 0.053 | 0.063 |
| 2016 | equal_average | 6mo | 2979 | 587 | 0.892 | 0.967 | 0.068 | 0.076 |
| 2016 | equal_average | 1yr | 2979 | 691 | 0.904 | 0.964 | 0.073 | 0.069 |
| 2016 | equal_average | 5yr | 2978 | 1316 | 0.885 | 0.902 | 0.140 | 0.073 |
| 2011 | physics_only | 30d | 2669 | 212 | 0.845 | 0.974 | 0.026 | 0.013 |
| 2011 | physics_only | 6mo | 2668 | 333 | 0.864 | 0.945 | 0.053 | 0.071 |
| 2011 | physics_only | 1yr | 2668 | 414 | 0.837 | 0.926 | 0.062 | 0.071 |
| 2011 | physics_only | 5yr | 2667 | 683 | 0.846 | 0.908 | 0.093 | 0.075 |
| 2011 | xgboost | 30d | 2669 | 212 | 0.891 | 0.983 | 0.026 | 0.013 |
| 2011 | xgboost | 6mo | 2668 | 333 | 0.887 | 0.962 | 0.053 | 0.080 |
| 2011 | xgboost | 1yr | 2668 | 414 | 0.849 | 0.938 | 0.074 | 0.090 |
| 2011 | xgboost | 5yr | 2667 | 683 | 0.831 | 0.895 | 0.192 | 0.244 |
| 2011 | tabm | 30d | 2669 | 212 | 0.815 | 0.937 | 0.031 | 0.032 |
| 2011 | tabm | 6mo | 2668 | 333 | 0.791 | 0.911 | 0.123 | 0.162 |
| 2011 | tabm | 1yr | 2668 | 414 | 0.746 | 0.883 | 0.157 | 0.174 |
| 2011 | tabm | 5yr | 2667 | 683 | 0.761 | 0.869 | 0.203 | 0.188 |
| 2011 | tabicl | 30d | 2669 | 212 | 0.874 | 0.971 | 0.029 | 0.022 |
| 2011 | tabicl | 6mo | 2668 | 333 | 0.885 | 0.953 | 0.068 | 0.100 |
| 2011 | tabicl | 1yr | 2668 | 414 | 0.855 | 0.934 | 0.066 | 0.081 |
| 2011 | tabicl | 5yr | 2667 | 683 | 0.818 | 0.893 | 0.101 | 0.070 |
| 2011 | kumo | 30d | 2669 | 212 | 0.866 | 0.969 | 0.029 | 0.025 |
| 2011 | kumo | 6mo | 2668 | 333 | 0.883 | 0.954 | 0.110 | 0.166 |
| 2011 | kumo | 1yr | 2668 | 414 | 0.850 | 0.937 | 0.082 | 0.105 |
| 2011 | kumo | 5yr | 2667 | 683 | 0.875 | 0.929 | 0.089 | 0.065 |
| 2011 | cosmos_gated | 30d | 2669 | 212 | 0.876 | 0.972 | 0.028 | 0.020 |
| 2011 | cosmos_gated | 6mo | 2668 | 333 | 0.885 | 0.956 | 0.117 | 0.180 |
| 2011 | cosmos_gated | 1yr | 2668 | 414 | 0.854 | 0.939 | 0.095 | 0.132 |
| 2011 | cosmos_gated | 5yr | 2667 | 683 | 0.876 | 0.931 | 0.091 | 0.070 |
| 2011 | stack_constant | 30d | 2669 | 212 | 0.869 | 0.970 | 0.029 | 0.026 |
| 2011 | stack_constant | 6mo | 2668 | 333 | 0.883 | 0.954 | 0.106 | 0.165 |
| 2011 | stack_constant | 1yr | 2668 | 414 | 0.851 | 0.937 | 0.085 | 0.110 |
| 2011 | stack_constant | 5yr | 2667 | 683 | 0.876 | 0.930 | 0.092 | 0.071 |
| 2011 | equal_average | 30d | 2669 | 212 | 0.883 | 0.975 | 0.027 | 0.013 |
| 2011 | equal_average | 6mo | 2668 | 333 | 0.879 | 0.953 | 0.083 | 0.146 |
| 2011 | equal_average | 1yr | 2668 | 414 | 0.852 | 0.935 | 0.097 | 0.154 |
| 2011 | equal_average | 5yr | 2667 | 683 | 0.874 | 0.930 | 0.091 | 0.069 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/horizon_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/horizon_test.csv`. 

## Table S7. Re-entry subgroups (main models)

| origin | model | subgroup | n | pinball | mae_log10 | cov90 |
|---|---|---|---|---|---|---|
| 2016 | physics_only | type=C | 451 | 0.1331 | 0.3677 | 0.6275 |
| 2016 | physics_only | type=D | 2154 | 0.0809 | 0.2028 | 0.9266 |
| 2016 | physics_only | type=R | 373 | 0.0845 | 0.2229 | 0.8874 |
| 2016 | physics_only | am_source=class_prior_sbm | 2322 | 0.0813 | 0.2065 | 0.9220 |
| 2016 | physics_only | am_source=design | 656 | 0.1174 | 0.3145 | 0.7149 |
| 2016 | tabm | type=C | 451 | 0.0744 | 0.2393 | 0.8293 |
| 2016 | tabm | type=D | 2154 | 0.0570 | 0.1847 | 0.8844 |
| 2016 | tabm | type=R | 373 | 0.0543 | 0.1760 | 0.8901 |
| 2016 | tabm | am_source=class_prior_sbm | 2322 | 0.0584 | 0.1878 | 0.8833 |
| 2016 | tabm | am_source=design | 656 | 0.0626 | 0.2061 | 0.8537 |
| 2016 | cosmos_gated | type=C | 451 | 0.0770 | 0.2459 | 0.8404 |
| 2016 | cosmos_gated | type=D | 2154 | 0.0579 | 0.1850 | 0.8784 |
| 2016 | cosmos_gated | type=R | 373 | 0.0545 | 0.1773 | 0.8901 |
| 2016 | cosmos_gated | am_source=class_prior_sbm | 2322 | 0.0591 | 0.1877 | 0.8790 |
| 2016 | cosmos_gated | am_source=design | 656 | 0.0647 | 0.2129 | 0.8567 |
| 2011 | physics_only | type=C | 331 | 0.0959 | 0.2714 | 0.9577 |
| 2011 | physics_only | type=D | 2078 | 0.0432 | 0.1234 | 0.9803 |
| 2011 | physics_only | type=R | 258 | 0.0733 | 0.1857 | 0.9419 |
| 2011 | physics_only | am_source=class_prior_sbm | 2263 | 0.0495 | 0.1412 | 0.9779 |
| 2011 | physics_only | am_source=design | 404 | 0.0704 | 0.1851 | 0.9505 |
| 2011 | kumo | type=C | 331 | 0.0630 | 0.2056 | 0.9396 |
| 2011 | kumo | type=D | 2078 | 0.0631 | 0.1587 | 0.9827 |
| 2011 | kumo | type=R | 258 | 0.0476 | 0.1460 | 0.8605 |
| 2011 | kumo | am_source=class_prior_sbm | 2263 | 0.0643 | 0.1671 | 0.9797 |
| 2011 | kumo | am_source=design | 404 | 0.0468 | 0.1420 | 0.8861 |
| 2011 | cosmos_gated | type=C | 331 | 0.0623 | 0.2025 | 0.9184 |
| 2011 | cosmos_gated | type=D | 2078 | 0.0590 | 0.1511 | 0.9769 |
| 2011 | cosmos_gated | type=R | 258 | 0.0473 | 0.1463 | 0.8682 |
| 2011 | cosmos_gated | am_source=class_prior_sbm | 2263 | 0.0604 | 0.1597 | 0.9717 |
| 2011 | cosmos_gated | am_source=design | 404 | 0.0465 | 0.1417 | 0.8886 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/subgroups_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/subgroups_test.csv`. 

## Table S8. Radar-cross-section availability and 30-day decay

| split | n | share_rcs_missing | p_decay30_given_missing | p_decay30_given_present | n_missing | share_rcs_missing_t0_from_2018 |
|---|---|---|---|---|---|---|
| o2016.train | 25350 | 0.271 | 0.759 | 0.172 | 6880 |  |
| o2016.tuneB | 1560 | 0.017 | 0.889 | 0.055 | 27 |  |
| o2016.test | 2979 | 0.682 | 0.134 | 0.041 | 2032 | 0.999 |
| o2011.train | 20793 | 0.324 | 0.764 | 0.191 | 6745 |  |
| o2011.tuneB | 332 | 0.099 | 0.576 | 0.171 | 33 |  |
| o2011.test | 2669 | 0.019 | 0.840 | 0.065 | 50 |  |

Source: `outputs/benchmarks/reentry_lifetime/leak_check.json`. 

## Table S9. C2 cloud survival, all candidates

| model | reading | origins | n_cloud_years | n_clouds | mae_count | mae_fraction | bias_fraction | within_10pct |
|---|---|---|---|---|---|---|---|---|
| kumo_no_space_weather | calibrated | 2011 | 85 | 17 | 8.037 | 0.071 | -0.039 | 0.694 |
| tabicl | calibrated | 2011,2016 | 160 | 27 | 9.432 | 0.095 | -0.046 | 0.631 |
| tabm_no_physics | calibrated | 2016 | 75 | 15 | 8.585 | 0.101 | -0.026 | 0.693 |
| kumo | calibrated | 2011,2016 | 160 | 27 | 10.732 | 0.101 | -0.065 | 0.613 |
| cosmos_gated_tuneA | calibrated | 2011,2016 | 160 | 27 | 11.178 | 0.115 | -0.058 | 0.581 |
| kumo_no_design | calibrated | 2011 | 85 | 17 | 14.009 | 0.118 | -0.111 | 0.529 |
| tabm_no_design | calibrated | 2016 | 75 | 15 | 10.795 | 0.118 | -0.025 | 0.653 |
| tabm_no_space_weather | calibrated | 2016 | 75 | 15 | 11.966 | 0.118 | -0.065 | 0.653 |
| tabm | calibrated | 2011,2016 | 160 | 27 | 13.229 | 0.119 | -0.055 | 0.606 |
| stack_constant | calibrated | 2011,2016 | 160 | 27 | 11.975 | 0.121 | -0.073 | 0.575 |
| tabm_no_family | calibrated | 2016 | 75 | 15 | 11.893 | 0.122 | -0.002 | 0.733 |
| cosmos_gated | calibrated | 2011,2016 | 160 | 27 | 12.142 | 0.122 | -0.072 | 0.550 |
| equal_average | calibrated | 2011,2016 | 160 | 27 | 11.995 | 0.126 | -0.075 | 0.575 |
| kumo_no_physics | calibrated | 2011 | 85 | 17 | 14.703 | 0.127 | -0.112 | 0.506 |
| mlp | calibrated | 2011,2016 | 160 | 27 | 15.459 | 0.131 | -0.076 | 0.606 |
| kumo_no_family | calibrated | 2011 | 85 | 17 | 16.545 | 0.132 | -0.131 | 0.506 |
| catboost | calibrated | 2011,2016 | 160 | 27 | 11.518 | 0.139 | -0.055 | 0.537 |
| kumo_plus_rcs_2026 | calibrated | 2011 | 85 | 17 | 15.939 | 0.141 | -0.099 | 0.541 |
| physics_only | calibrated | 2011,2016 | 160 | 27 | 11.010 | 0.145 | -0.054 | 0.438 |
| lightgbm | calibrated | 2011,2016 | 160 | 27 | 14.953 | 0.166 | -0.098 | 0.519 |
| tabm_plus_rcs_2026 | calibrated | 2016 | 75 | 15 | 15.599 | 0.175 | -0.141 | 0.547 |
| xgboost | calibrated | 2011,2016 | 160 | 27 | 15.015 | 0.177 | -0.103 | 0.475 |
| tabm_no_physics | raw | 2016 | 75 | 15 | 8.753 | 0.107 | -0.016 | 0.707 |
| tabm_no_design | raw | 2016 | 75 | 15 | 10.366 | 0.118 | -0.010 | 0.680 |
| tabm_no_space_weather | raw | 2016 | 75 | 15 | 12.370 | 0.128 | -0.046 | 0.640 |
| physics_only | raw | 2011,2016 | 160 | 27 | 9.981 | 0.129 | 0.007 | 0.731 |
| tabm_no_family | raw | 2016 | 75 | 15 | 12.456 | 0.140 | -0.026 | 0.627 |
| tabm_plus_rcs_2026 | raw | 2016 | 75 | 15 | 13.013 | 0.143 | -0.126 | 0.667 |
| equal_average | raw | 2011,2016 | 160 | 27 | 14.222 | 0.147 | -0.072 | 0.606 |
| tabicl | raw | 2011,2016 | 160 | 27 | 15.758 | 0.153 | -0.121 | 0.512 |
| cosmos_gated | raw | 2011,2016 | 160 | 27 | 15.631 | 0.155 | -0.099 | 0.519 |
| stack_constant | raw | 2011,2016 | 160 | 27 | 16.056 | 0.158 | -0.104 | 0.512 |
| cosmos_gated_tuneA | raw | 2011,2016 | 160 | 27 | 16.503 | 0.162 | -0.077 | 0.556 |
| kumo | raw | 2011,2016 | 160 | 27 | 17.167 | 0.163 | -0.124 | 0.469 |
| kumo_no_space_weather | raw | 2011 | 85 | 17 | 19.969 | 0.173 | -0.167 | 0.435 |
| tabm | raw | 2011,2016 | 160 | 27 | 21.448 | 0.180 | -0.132 | 0.550 |
| kumo_plus_rcs_2026 | raw | 2011 | 85 | 17 | 19.809 | 0.182 | -0.156 | 0.424 |
| catboost | raw | 2011,2016 | 160 | 27 | 16.982 | 0.191 | -0.141 | 0.475 |
| lightgbm | raw | 2011,2016 | 160 | 27 | 17.834 | 0.192 | -0.119 | 0.475 |
| kumo_no_design | raw | 2011 | 85 | 17 | 21.097 | 0.194 | -0.189 | 0.376 |
| mlp | raw | 2011,2016 | 160 | 27 | 22.199 | 0.197 | -0.156 | 0.519 |
| xgboost | raw | 2011,2016 | 160 | 27 | 17.983 | 0.202 | -0.151 | 0.469 |
| kumo_no_physics | raw | 2011 | 85 | 17 | 21.894 | 0.203 | -0.196 | 0.365 |
| kumo_no_family | raw | 2011 | 85 | 17 | 23.362 | 0.208 | -0.208 | 0.365 |

Source: `outputs/benchmarks/twin_couplings/c2_summary.csv`. 

## Table S10. C3 by operational status and policy

| status | policy | mean_p | event_rate | brier | ece |
|---|---|---|---|---|---|
| non-operational | physics_status_blind | 0.0755 | 0.1056 | 0.0610 | 0.0728 |
| non-operational | base_rate_status_blind | 0.0127 | 0.1056 | 0.1031 | 0.0929 |
| non-operational | twin_status_aware_baserate | 0.0755 | 0.1056 | 0.0610 | 0.0728 |
| non-operational | physics_recalibrated_status_blind | 0.0244 | 0.1056 | 0.0901 | 0.0929 |
| non-operational | twin_status_aware_recalibrated | 0.0940 | 0.1056 | 0.0627 | 0.0614 |
| operational | physics_status_blind | 0.0390 | 0.0122 | 0.0176 | 0.0271 |
| operational | base_rate_status_blind | 0.0127 | 0.0122 | 0.0120 | 0.0005 |
| operational | twin_status_aware_baserate | 0.0127 | 0.0122 | 0.0120 | 0.0005 |
| operational | physics_recalibrated_status_blind | 0.0152 | 0.0122 | 0.0076 | 0.0039 |
| operational | twin_status_aware_recalibrated | 0.0138 | 0.0122 | 0.0075 | 0.0027 |
| unknown | physics_status_blind | 0.0167 | 0.0065 | 0.0058 | 0.0123 |
| unknown | base_rate_status_blind | 0.0127 | 0.0065 | 0.0064 | 0.0062 |
| unknown | twin_status_aware_baserate | 0.0167 | 0.0065 | 0.0058 | 0.0123 |
| unknown | physics_recalibrated_status_blind | 0.0055 | 0.0065 | 0.0040 | 0.0061 |
| unknown | twin_status_aware_recalibrated | 0.0139 | 0.0065 | 0.0046 | 0.0095 |

Source: `outputs/benchmarks/twin_couplings/c3_summary.json`. 

## Table S11. NASA SMAP/MSL, all detectors

| spacecraft | model | channels | events | event_tp | event_fp | event_precision | event_recall | event_f1 | esa_ew_precision | esa_ew_f0.5 | nominal_alarm_fraction | median_delay_steps | event_f1_swap | event_f1_swap_p | event_f1_late | esa_ew_f0.5_late | pointwise_auc_pr | pointwise_auc_pr_pooled | pointwise_auc_pr_pooled_raw | point_adjusted_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSL | ar_ridge | 27.000 | 36.000 | 11.000 | 36.000 | 0.234 | 0.306 | 0.265 | 0.241 | 0.252 | 0.167 | 9.000 | 0.154 | 0.005 | 0.185 | 0.297 | 0.329 | 0.126 |  | 0.097 |
| MSL | chronos2 | 27.000 | 36.000 | 16.000 | 508.000 | 0.031 | 0.444 | 0.057 | 0.028 | 0.035 | 0.100 | 11.000 | 0.038 | 0.050 | 0.161 | 0.224 | 0.367 | 0.137 |  | 0.308 |
| MSL | knn | 27.000 | 36.000 | 15.000 | 40.000 | 0.273 | 0.417 | 0.330 | 0.318 | 0.334 | 0.087 | 19.000 | 0.166 | 0.005 | 0.203 | 0.281 | 0.364 | 0.151 |  | 0.239 |
| MSL | pca | 27.000 | 36.000 | 15.000 | 69.000 | 0.179 | 0.417 | 0.250 | 0.189 | 0.212 | 0.094 | 16.000 | 0.126 | 0.005 | 0.226 | 0.314 | 0.328 | 0.141 |  | 0.301 |
| MSL | random | 27.000 | 36.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  | 0.000 | 1.000 | 0.000 | 0.000 | 0.141 | 0.119 |  | 0.000 |
| MSL | telemanom (published scores) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.473 |  | 0.309 |  |
| MSL | always_on (reference) | 27.000 | 36.000 | 36.000 | 62.000 | 0.367 | 1.000 | 0.537 | 0.000 | 0.000 | 1.000 | 0.000 |  |  |  |  |  |  |  | 0.208 |
| SMAP | ar_ridge | 54.000 | 68.000 | 31.000 | 111.000 | 0.218 | 0.456 | 0.295 | 0.215 | 0.240 | 0.126 | 47.000 | 0.136 | 0.005 | 0.293 | 0.413 | 0.355 | 0.244 |  | 0.478 |
| SMAP | chronos2 | 54.000 | 68.000 | 30.000 | 185.000 | 0.140 | 0.441 | 0.212 | 0.147 | 0.170 | 0.018 | 51.500 | 0.059 | 0.005 | 0.155 | 0.197 | 0.343 | 0.329 |  | 0.634 |
| SMAP | knn | 54.000 | 68.000 | 35.000 | 326.000 | 0.097 | 0.515 | 0.163 | 0.095 | 0.114 | 0.057 | 48.000 | 0.100 | 0.005 | 0.218 | 0.272 | 0.341 | 0.380 |  | 0.612 |
| SMAP | pca | 54.000 | 68.000 | 31.000 | 108.000 | 0.223 | 0.456 | 0.300 | 0.224 | 0.249 | 0.046 | 50.000 | 0.151 | 0.005 | 0.250 | 0.306 | 0.322 | 0.425 |  | 0.649 |
| SMAP | random | 54.000 | 68.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  | 0.000 | 1.000 | 0.000 | 0.000 | 0.129 | 0.132 |  | 0.000 |
| SMAP | telemanom (published scores) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.576 |  | 0.382 |  |
| SMAP | always_on (reference) | 54.000 | 68.000 | 68.000 | 112.000 | 0.378 | 1.000 | 0.548 | 0.000 | 0.000 | 1.000 | 0.000 |  |  |  |  |  |  |  | 0.234 |

Source: `outputs/benchmarks/telemetry_smap_msl/summary.csv`. 

## Table S12. ESA Kelvins collision-risk test set, all models

| model | L | F2 | MSE_HR | L_clip | L_decision_clip | pinball | auc_pr_high | auc_roc_high | brier_high | ece_high | cov80 | cov90 | cov90_high | width90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latest_risk_point | 36.0491 | 0.7391 | 26.6430 | 0.6940 |  |  |  |  |  |  |  |  |  |  |
| persistence | inf | 0.0000 | 563.9289 | inf | 2.3222 | 1.0454 | 0.6643 | 0.9443 | 0.0523 | 0.0530 | 0.8463 | 0.9105 | 0.5933 | 8.5190 |
| xgboost | inf | 0.0000 | 308.7600 | inf | 2.9717 | 0.8555 | 0.6064 | 0.9446 | 0.0552 | 0.0537 | 0.8708 | 0.9174 | 0.4333 | 8.7473 |
| lightgbm | inf | 0.0000 | 259.1361 | inf | 4.6482 | 0.8607 | 0.5414 | 0.9439 | 0.0601 | 0.0586 | 0.8579 | 0.9049 | 0.2533 | 9.3536 |
| catboost | inf | 0.0000 | 295.9138 | inf | 3.7759 | 0.8692 | 0.5419 | 0.9351 | 0.0564 | 0.0536 | 0.8329 | 0.9119 | 0.4667 | 9.7233 |
| mlp | inf | 0.0000 | 337.9009 | inf | 9.0363 | 1.0354 | 0.4337 | 0.9010 | 0.0653 | 0.0608 | 0.8616 | 0.9363 | 0.8533 | 10.9389 |
| tabm | inf | 0.0000 | 462.6131 | inf | 2.9604 | 0.8847 | 0.5115 | 0.9372 | 0.0591 | 0.0557 | 0.8353 | 0.9188 | 0.7267 | 8.8725 |
| tabicl | 3766.4255 | 0.0575 | 216.4612 | 30.0595 | 2.5690 | 0.6919 | 0.6652 | 0.9607 | 0.0489 | 0.0487 | 0.8293 | 0.9183 | 0.3267 | 8.3384 |
| kumo | 1334.4935 | 0.1680 | 224.1949 | 9.8916 | 2.5097 | 0.6534 | 0.7326 | 0.9696 | 0.0453 | 0.0456 | 0.9114 | 0.9423 | 0.4800 | 7.8911 |
| cosmos_gated | 1334.6257 | 0.1680 | 224.2171 | 9.8955 | 2.5107 | 0.6534 | 0.7327 | 0.9696 | 0.0453 | 0.0456 | 0.8916 | 0.9400 | 0.4800 | 7.8913 |
| stack_constant | 1334.6935 | 0.1680 | 224.2285 | 9.8975 | 2.5112 | 0.6534 | 0.7328 | 0.9696 | 0.0453 | 0.0456 | 0.8823 | 0.9349 | 0.4800 | 7.8914 |

Source: `outputs/benchmarks/conjunction_kelvins_cac/test_results.csv`. 

## Table S13. Latest-risk point forecast gated by calibrated P(high risk)

| model | threshold_oof | F2 | MSE_HR | L | precision | recall | tp | fp | fn | L_ci95 | L_minus_latest_ci95 | p_better_than_latest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latest_risk_point |  | 0.7391 | 0.5129 | 0.6940 | 0.6461 | 0.7667 | 115 | 63 | 35 | [0.4792599478558545, 0.9681414952419644] | nan |  |
| catboost_gated_persistence | 0.0152 | 0.7134 | 0.5029 | 0.7050 | 0.6433 | 0.7333 | 110 | 61 | 40 | nan | [-0.04439880078297968, 0.061553741416146036] | 0.3235 |
| cosmos_gated_gated_persistence | 0.0400 | 0.7228 | 0.5087 | 0.7038 | 0.7078 | 0.7267 | 109 | 45 | 41 | nan | [-0.055233898266173885, 0.0710182369383869] | 0.3765 |
| kumo_gated_persistence | 0.0400 | 0.7228 | 0.5087 | 0.7038 | 0.7078 | 0.7267 | 109 | 45 | 41 | nan | [-0.055233898266173885, 0.0710182369383869] | 0.3765 |
| lightgbm_gated_persistence | 0.0300 | 0.5997 | 0.8473 | 1.4128 | 0.7350 | 0.5733 | 86 | 31 | 64 | nan | [0.3264247761193668, 1.280610863602614] | 0.0000 |
| mlp_gated_persistence | 0.0193 | 0.7077 | 0.5407 | 0.7640 | 0.6626 | 0.7200 | 108 | 55 | 42 | nan | [-0.025871124337191725, 0.20563250206152214] | 0.0860 |
| persistence_gated_persistence | 0.0000 | 0.7391 | 0.5129 | 0.6940 | 0.6461 | 0.7667 | 115 | 63 | 35 | nan | [0.0, 0.0] | 0.0000 |
| stack_constant_gated_persistence | 0.0400 | 0.7228 | 0.5087 | 0.7038 | 0.7078 | 0.7267 | 109 | 45 | 41 | nan | [-0.055233898266173885, 0.0710182369383869] | 0.3765 |
| tabicl_gated_persistence | 0.0300 | 0.6711 | 0.5982 | 0.8913 | 0.6897 | 0.6667 | 100 | 45 | 50 | nan | [0.03635924414317057, 0.41337709780502124] | 0.0015 |
| tabm_gated_persistence | 0.0049 | 0.7364 | 0.5139 | 0.6979 | 0.6552 | 0.7600 | 114 | 60 | 36 | nan | [-0.005372107579710852, 0.019482478346674128] | 0.3655 |
| xgboost_gated_persistence | 0.0200 | 0.7162 | 0.6184 | 0.8635 | 0.6770 | 0.7267 | 109 | 52 | 41 | nan | [-0.05511769799325124, 0.5254374453252811] | 0.1400 |

Source: `outputs/benchmarks/conjunction_kelvins_cac/gated_persistence_test.csv`. 

## Table S14. As-of-date test T1 on a previously analysed window (as of 2026-08-16, decays to 2026-10-27)

| horizon | group | variant | n | events | expected | mean_p | brier | ece | auc_roc | auc_pr |
|---|---|---|---|---|---|---|---|---|---|---|
| 30d | uncontrolled | physics_history | 13186 | 29 | 43.7500 | 0.0033 | 0.0013 | 0.0013 | 0.9823 | 0.7140 |
| 30d | uncontrolled | physics_prior | 13186 | 29 | 118.0000 | 0.0089 | 0.0032 | 0.0067 | 0.9987 | 0.5749 |
| 30d | uncontrolled | lifetime | 13186 | 29 | 304.7205 | 0.0231 | 0.0087 | 0.0209 | 0.9980 | 0.6192 |
| 30d | class_D | physics_history | 10603 | 21 | 31.0312 | 0.0029 | 0.0012 | 0.0012 | 0.9757 | 0.6707 |
| 30d | class_D | physics_prior | 10603 | 21 | 92.3750 | 0.0087 | 0.0032 | 0.0067 | 0.9986 | 0.5846 |
| 30d | class_D | lifetime | 10603 | 21 | 231.9499 | 0.0219 | 0.0082 | 0.0199 | 0.9976 | 0.5819 |
| 30d | class_R | physics_history | 1465 | 6 | 8.8438 | 0.0060 | 0.0012 | 0.0019 | 0.9997 | 0.8571 |
| 30d | class_R | physics_prior | 1465 | 6 | 9.9375 | 0.0068 | 0.0015 | 0.0033 | 0.9995 | 0.7901 |
| 30d | class_R | lifetime | 1465 | 6 | 30.1305 | 0.0206 | 0.0071 | 0.0171 | 0.9994 | 0.8179 |
| 30d | class_C | physics_history | 1118 | 2 | 3.8750 | 0.0035 | 0.0016 | 0.0017 | 0.9996 | 0.6667 |
| 30d | class_C | physics_prior | 1118 | 2 | 15.6875 | 0.0140 | 0.0057 | 0.0122 | 0.9980 | 0.3667 |
| 30d | class_C | lifetime | 1118 | 2 | 42.6401 | 0.0381 | 0.0154 | 0.0364 | 0.9991 | 0.7500 |
| 30d | payload_natural_decay | twin | 559 | 7 | 5.1875 | 0.0093 | 0.0082 | 0.0093 | 0.9230 | 0.5161 |
| 30d | payload_natural_decay | physics_prior | 559 | 7 | 5.1875 | 0.0093 | 0.0082 | 0.0093 | 0.9230 | 0.5161 |
| 30d | payload_operator_controlled | twin | 15858 | 65 | 57.0638 | 0.0036 | 0.0041 | 0.0005 | 0.5756 | 0.0048 |
| 30d | payload_operator_controlled | physics_prior | 15858 | 65 | 186.2188 | 0.0117 | 0.0043 | 0.0076 | 0.9819 | 0.7830 |
| 30d | payload_unknown | twin | 1223 | 0 | 0.7812 | 0.0006 | 0.0003 | 0.0006 |  |  |
| 30d | payload_unknown | physics_prior | 1223 | 0 | 0.7812 | 0.0006 | 0.0003 | 0.0006 |  |  |
| 30d | all_twin | twin | 30826 | 101 | 106.7825 | 0.0035 | 0.0028 | 0.0009 | 0.8443 | 0.2387 |
| 72d | uncontrolled | physics_history | 13186 | 64 | 87.4375 | 0.0066 | 0.0022 | 0.0019 | 0.9913 | 0.7367 |
| 72d | uncontrolled | physics_prior | 13186 | 64 | 232.9375 | 0.0177 | 0.0062 | 0.0128 | 0.9884 | 0.6011 |
| 72d | uncontrolled | lifetime | 13186 | 64 | 609.5597 | 0.0462 | 0.0134 | 0.0414 | 0.9949 | 0.6158 |
| 72d | class_D | physics_history | 10603 | 45 | 64.5312 | 0.0061 | 0.0018 | 0.0018 | 0.9993 | 0.7706 |
| 72d | class_D | physics_prior | 10603 | 45 | 179.4688 | 0.0169 | 0.0055 | 0.0127 | 0.9984 | 0.6251 |
| 72d | class_D | lifetime | 10603 | 45 | 457.7024 | 0.0432 | 0.0125 | 0.0389 | 0.9947 | 0.6186 |
| 72d | class_R | physics_history | 1465 | 15 | 16.2812 | 0.0111 | 0.0030 | 0.0034 | 0.9989 | 0.8469 |
| 72d | class_R | physics_prior | 1465 | 15 | 20.2188 | 0.0138 | 0.0044 | 0.0062 | 0.9972 | 0.7857 |
| 72d | class_R | lifetime | 1465 | 15 | 68.6753 | 0.0469 | 0.0099 | 0.0366 | 0.9981 | 0.7960 |
| 72d | class_C | physics_history | 1118 | 4 | 6.6250 | 0.0059 | 0.0044 | 0.0046 | 0.8730 | 0.3080 |
| 72d | class_C | physics_prior | 1118 | 4 | 33.2500 | 0.0297 | 0.0161 | 0.0262 | 0.8607 | 0.3842 |
| 72d | class_C | lifetime | 1118 | 4 | 83.1821 | 0.0744 | 0.0266 | 0.0708 | 0.9825 | 0.5378 |
| 72d | payload_natural_decay | twin | 559 | 17 | 11.3125 | 0.0202 | 0.0177 | 0.0162 | 0.9607 | 0.6247 |
| 72d | payload_natural_decay | physics_prior | 559 | 17 | 11.3125 | 0.0202 | 0.0177 | 0.0162 | 0.9607 | 0.6247 |
| 72d | payload_operator_controlled | twin | 15858 | 164 | 170.4293 | 0.0107 | 0.0102 | 0.0004 | 0.5761 | 0.0122 |
| 72d | payload_operator_controlled | physics_prior | 15858 | 164 | 498.2188 | 0.0314 | 0.0133 | 0.0211 | 0.9604 | 0.5930 |
| 72d | payload_unknown | twin | 1223 | 1 | 2.5000 | 0.0020 | 0.0005 | 0.0013 | 1.0000 | 1.0000 |
| 72d | payload_unknown | physics_prior | 1223 | 1 | 2.5000 | 0.0020 | 0.0005 | 0.0013 | 1.0000 | 1.0000 |
| 72d | all_twin | twin | 30826 | 246 | 271.6793 | 0.0088 | 0.0065 | 0.0009 | 0.8473 | 0.2482 |

Source: `outputs/benchmarks/twin_prospective_t1/t1_summary.json`. 

## Table S16. Re-entry head-to-head comparisons (post hoc; paired launch-family bootstrap; < 0: a is better)

| origin | a | b | pinball_a_minus_b | ci95_lo | ci95_hi |
|---|---|---|---|---|---|
| 2016 | physics_only | tabicl | 0.0393 | 0.0217 | 0.0622 |
| 2016 | physics_only | kumo | 0.0385 | 0.0203 | 0.0613 |
| 2016 | physics_only | tabm | 0.0300 | 0.0147 | 0.0493 |
| 2016 | physics_only | cosmos_gated | 0.0289 | 0.0148 | 0.0469 |
| 2016 | tabicl | kumo | -0.0008 | -0.0036 | 0.0024 |
| 2016 | tabicl | xgboost | -0.0076 | -0.0121 | -0.0036 |
| 2016 | kumo | xgboost | -0.0068 | -0.0134 | -0.0012 |
| 2016 | tabicl | cosmos_gated | -0.0104 | -0.0179 | -0.0043 |
| 2016 | kumo | cosmos_gated | -0.0096 | -0.0181 | -0.0023 |
| 2016 | tabicl | tabm | -0.0093 | -0.0162 | -0.0036 |
| 2016 | kumo | tabm | -0.0085 | -0.0167 | -0.0017 |
| 2011 | physics_only | tabicl | -0.0069 | -0.0217 | 0.0122 |
| 2011 | physics_only | kumo | -0.0089 | -0.0253 | 0.0133 |
| 2011 | physics_only | tabm | -0.0468 | -0.1078 | 0.0085 |
| 2011 | physics_only | cosmos_gated | -0.0056 | -0.0199 | 0.0139 |
| 2011 | tabicl | kumo | -0.0020 | -0.0058 | 0.0024 |
| 2011 | tabicl | xgboost | -0.0053 | -0.0124 | 0.0001 |
| 2011 | kumo | xgboost | -0.0032 | -0.0126 | 0.0035 |
| 2011 | tabicl | cosmos_gated | 0.0013 | -0.0016 | 0.0044 |
| 2011 | kumo | cosmos_gated | 0.0033 | 0.0007 | 0.0055 |
| 2011 | tabicl | tabm | -0.0399 | -0.0881 | 0.0006 |
| 2011 | kumo | tabm | -0.0378 | -0.0862 | 0.0009 |

Source: `outputs/benchmarks/reentry_lifetime/origin2016_uncontrolled/pairwise_test.csv, outputs/benchmarks/reentry_lifetime/origin2011_uncontrolled/pairwise_test.csv`. Nominal 95 % intervals, not corrected for multiplicity.

## Table S17. C2 mean absolute fraction errors and paired differences with cloud-cluster bootstrap CIs

| comparison | mae_fraction | ci95_lo | ci95_hi |
|---|---|---|---|
| pred_catboost | 0.191 | 0.157 | 0.227 |
| predcal_catboost | 0.139 | 0.096 | 0.191 |
| pred_cosmos_gated | 0.155 | 0.121 | 0.192 |
| predcal_cosmos_gated | 0.122 | 0.085 | 0.161 |
| pred_cosmos_gated_tuneA | 0.162 | 0.121 | 0.205 |
| predcal_cosmos_gated_tuneA | 0.115 | 0.081 | 0.150 |
| pred_equal_average | 0.147 | 0.114 | 0.182 |
| predcal_equal_average | 0.126 | 0.087 | 0.167 |
| pred_kumo | 0.163 | 0.126 | 0.200 |
| predcal_kumo | 0.101 | 0.071 | 0.132 |
| pred_lightgbm | 0.192 | 0.160 | 0.225 |
| predcal_lightgbm | 0.166 | 0.110 | 0.226 |
| pred_mlp | 0.197 | 0.153 | 0.253 |
| predcal_mlp | 0.131 | 0.085 | 0.186 |
| pred_physics_only | 0.129 | 0.078 | 0.191 |
| predcal_physics_only | 0.145 | 0.111 | 0.187 |
| pred_stack_constant | 0.158 | 0.123 | 0.196 |
| predcal_stack_constant | 0.121 | 0.084 | 0.160 |
| pred_tabicl | 0.153 | 0.119 | 0.188 |
| predcal_tabicl | 0.095 | 0.068 | 0.123 |
| pred_tabm | 0.180 | 0.137 | 0.236 |
| predcal_tabm | 0.119 | 0.079 | 0.163 |
| pred_tabm_no_design | 0.118 | 0.077 | 0.171 |
| predcal_tabm_no_design | 0.118 | 0.053 | 0.191 |
| pred_tabm_no_family | 0.140 | 0.095 | 0.203 |
| predcal_tabm_no_family | 0.122 | 0.052 | 0.213 |
| pred_tabm_no_physics | 0.107 | 0.070 | 0.151 |
| predcal_tabm_no_physics | 0.101 | 0.045 | 0.171 |
| pred_tabm_no_space_weather | 0.128 | 0.086 | 0.176 |
| predcal_tabm_no_space_weather | 0.118 | 0.055 | 0.196 |
| pred_tabm_plus_rcs_2026 | 0.143 | 0.083 | 0.224 |
| predcal_tabm_plus_rcs_2026 | 0.175 | 0.086 | 0.281 |
| pred_xgboost | 0.202 | 0.165 | 0.240 |
| predcal_xgboost | 0.177 | 0.130 | 0.227 |
| pred_kumo_no_design | 0.194 | 0.140 | 0.251 |
| predcal_kumo_no_design | 0.118 | 0.074 | 0.168 |
| pred_kumo_no_family | 0.208 | 0.150 | 0.271 |
| predcal_kumo_no_family | 0.132 | 0.080 | 0.195 |
| pred_kumo_no_physics | 0.203 | 0.147 | 0.264 |
| predcal_kumo_no_physics | 0.127 | 0.079 | 0.182 |
| pred_kumo_no_space_weather | 0.173 | 0.122 | 0.225 |
| predcal_kumo_no_space_weather | 0.071 | 0.038 | 0.108 |
| pred_kumo_plus_rcs_2026 | 0.182 | 0.131 | 0.234 |
| predcal_kumo_plus_rcs_2026 | 0.141 | 0.087 | 0.195 |
| pred_physics_only - predcal_tabicl | 0.035 | -0.007 | 0.078 |
| pred_physics_only - predcal_physics_only | -0.016 | -0.040 | 0.011 |
| pred_cosmos_gated - predcal_cosmos_gated | 0.033 | 0.007 | 0.055 |
| pred_tabicl - predcal_tabicl | 0.058 | 0.034 | 0.082 |
| pred_physics_only - pred_cosmos_gated | -0.025 | -0.084 | 0.033 |

Source: `outputs/benchmarks/twin_couplings/c2_bootstrap.csv`. pred_: raw read-off; predcal_: calibrated read-off (added after the raw reading was criticised in the first review).

## Table S18. Collision-risk pinball-loss comparisons (post hoc; paired bootstrap over test events; < 0: a is better)

| a | b | pinball_a_minus_b | ci95_lo | ci95_hi |
|---|---|---|---|---|
| kumo | tabicl | -0.0385 | -0.0531 | -0.0251 |
| kumo | xgboost | -0.2021 | -0.2333 | -0.1711 |
| kumo | tabm | -0.2314 | -0.2639 | -0.1993 |
| kumo | persistence | -0.3920 | -0.4513 | -0.3367 |

Source: `outputs/benchmarks/conjunction_kelvins_cac/pinball_pairs_test.csv`. Nominal 95 % intervals, not corrected for multiplicity.

## Table S15. Dataset manifest (files with checksums)

| dataset | class | path | bytes | licence | retrieved_utc |
|---|---|---|---|---|---|
| cosmos_synthetic_telemetry | SYNTHETIC | nominal_runs.npy |  | nan | nan |
| esa_adb | REAL | raw/ESA-Mission1.zip | 3776246054.0000 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | 2026-10-24T08:47:44+00:00 |
| esa_adb | REAL | raw/ESA-Mission2.zip | 4098539912.0000 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | 2026-10-24T08:47:44+00:00 |
| esa_adb | REAL | raw/ESA-Mission3.zip | 3734403447.0000 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | 2026-10-24T08:47:44+00:00 |
| esa_adb | REAL | raw/kplabs-pl__ESA-ADB-67194d389c4a.tar.gz | 18741008.0000 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | 2026-10-24T08:47:44+00:00 |
| esa_adb | REAL | raw/zenodo_15237121/members/ESA-Mission1_channels.csv | 3782.0000 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | 2026-10-24T08:47:44+00:00 |
| esa_adb | REAL | raw/zenodo_15237121/members/ESA-Mission2_channels.csv | 4350.0000 | not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper | 2026-10-24T08:47:44+00:00 |
| opssat_ad | REAL | raw/LICENSE | 1091.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| opssat_ad | REAL | raw/dataset.csv | 507550.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| opssat_ad | REAL | raw/dataset_generator.ipynb | 8089.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| opssat_ad | REAL | raw/kplabs-pl__OPS-SAT-AD-dafd0d35bfd6.tar.gz | 5493.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| opssat_ad | REAL | raw/modeling_examples.ipynb | 7174.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| opssat_ad | REAL | raw/requirements.txt | 169.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| opssat_ad | REAL | raw/segments.csv | 18987091.0000 | cc-by-4.0 | 2026-10-24T08:03:31+00:00 |
| smap_msl | REAL | raw/labeled_anomalies_telemanom.csv | 3956.0000 | Caltech copyright with a BSD-style redistribution licence (telemanom LICENSE.txt); Kaggle lists 'Data files (c) Original Authors' | 2026-10-27T04:17:36+00:00 |
| smap_msl | REAL | raw/nasa-smap-msl.zip | 86004011.0000 | Caltech copyright with a BSD-style redistribution licence (telemanom LICENSE.txt); Kaggle lists 'Data files (c) Original Authors' | 2026-10-27T04:17:25+00:00 |
| celestrak | REAL | gp-active.csv | 2468651.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| celestrak | REAL | gp-cosmos-1408-debris.csv | 702.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| celestrak | REAL | gp-cosmos-2251-debris.csv | 91572.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| celestrak | REAL | gp-fengyun-1c-debris.csv | 296976.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| celestrak | REAL | gp-iridium-33-debris.csv | 17257.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| celestrak | REAL | satcat.csv | 6703172.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| celestrak | REAL | snapshots/2026-10-24/SW-All.csv | 2891940.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:05:03+00:00 |
| celestrak | REAL | snapshots/2026-10-24/gp-active.csv | 2543559.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:04:50+00:00 |
| celestrak | REAL | snapshots/2026-10-24/gp-cosmos-1408-debris.csv | 706.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:04:54+00:00 |
| celestrak | REAL | snapshots/2026-10-24/gp-cosmos-2251-debris.csv | 90914.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:04:57+00:00 |
| celestrak | REAL | snapshots/2026-10-24/gp-fengyun-1c-debris.csv | 303374.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:04:59+00:00 |
| celestrak | REAL | snapshots/2026-10-24/gp-iridium-33-debris.csv | 17007.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:05:01+00:00 |
| celestrak | REAL | snapshots/2026-10-24/gp-last-30-days.csv | 38724.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:04:52+00:00 |
| celestrak | REAL | snapshots/2026-10-24/satcat.csv | 6781157.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-24T08:04:48+00:00 |
| celestrak | REAL | snapshots/2026-10-27/SW-All.csv | 2892082.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:56+00:00 |
| celestrak | REAL | snapshots/2026-10-27/gp-active.csv | 2544175.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:42+00:00 |
| celestrak | REAL | snapshots/2026-10-27/gp-cosmos-1408-debris.csv | 704.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:47+00:00 |
| celestrak | REAL | snapshots/2026-10-27/gp-cosmos-2251-debris.csv | 90601.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:49+00:00 |
| celestrak | REAL | snapshots/2026-10-27/gp-fengyun-1c-debris.csv | 303742.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:51+00:00 |
| celestrak | REAL | snapshots/2026-10-27/gp-iridium-33-debris.csv | 17014.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:54+00:00 |
| celestrak | REAL | snapshots/2026-10-27/gp-last-30-days.csv | 39248.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:44+00:00 |
| celestrak | REAL | snapshots/2026-10-27/satcat.csv | 6782722.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-10-27T04:17:38+00:00 |
| celestrak | REAL | spatial_density.csv | 963.0000 | CelesTrak usage policy; public, no credentials; no explicit licence | 2026-08 (see data/README.md) |
| cosmos_synthetic_vision | SYNTHETIC | yolo_v1/data.yaml |  | generated by this project | nan |
| cosmos_synthetic_vision | SYNTHETIC | yolo_v1/scenes.jsonl |  | generated by this project | nan |
| gcat | REAL | raw/2026-10-27/auxcat.tsv | 3430620.0000 | CC-BY-4.0 (cite: McDowell, J., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat) | 2026-10-27T04:13:45+00:00 |
| gcat | REAL | raw/2026-10-27/ftocat.tsv | 511727.0000 | CC-BY-4.0 (cite: McDowell, J., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat) | 2026-10-27T04:13:49+00:00 |
| gcat | REAL | raw/2026-10-27/launch.tsv | 13813643.0000 | CC-BY-4.0 (cite: McDowell, J., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat) | 2026-10-27T04:13:51+00:00 |
| gcat | REAL | raw/2026-10-27/orgs.tsv | 711974.0000 | CC-BY-4.0 (cite: McDowell, J., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat) | 2026-10-27T04:13:53+00:00 |
| gcat | REAL | raw/2026-10-27/psatcat.tsv | 5232656.0000 | CC-BY-4.0 (cite: McDowell, J., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat) | 2026-10-27T04:13:47+00:00 |
| gcat | REAL | raw/2026-10-27/satcat.tsv | 19297558.0000 | CC-BY-4.0 (cite: McDowell, J., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat) | 2026-10-27T04:13:43+00:00 |
| kelvins_cac | REAL | raw/Collision Avoidance Challenge - Dataset.zip | 221128642.0000 | cc-by-4.0 | 2026-10-27T04:16:59+00:00 |
| kelvins_debris_origin | SEE NOTES (simulated or real is determined in docs/DATASET_AUDIT.md) | raw/space-debris-the-origin.zip | 2903013.0000 | cc-by-4.0 | 2026-10-27T04:14:16+00:00 |
| spacesense_bench | SYNTHETIC (Unreal Engine 5 renders) | raw/ACE.tar.gz | 510933895.0000 | cc-by-nc-4.0 (Hugging Face card); GitHub repo has no licence file | 2026-10-24T10:07:33+00:00 |
| spark | SYNTHETIC renders + Zero-G lab images (per the dataset pages) | nan |  | nan | nan |
| spotgeo | REAL | raw/SpotGEOv2.zip | 4230495720.0000 | cc-by-4.0 | 2026-10-24T09:12:26+00:00 |
| spotgeo | REAL | starter_kit/raw/spotGEO_metric.pdf | 307611.0000 | cc-by-4.0 | 2026-10-24T09:12:26+00:00 |
| spotgeo | REAL | starter_kit/raw/spotGEO_starter_kit.ipynb | 179279.0000 | cc-by-4.0 | 2026-10-24T09:12:26+00:00 |
| spotgeo | REAL | starter_kit/raw/validation.py | 8260.0000 | cc-by-4.0 | 2026-10-24T09:12:26+00:00 |
| reentry_historical | REAL | raw/esa_reentry_portal/api_home.json | 125779.0000 | nan | nan |
| reentry_historical | REAL | recovered_debris_esa.csv | 31793.0000 | nan | nan |
| reentry_processed | DERIVED (from REAL catalogue data) | lifetime_cohort_2026-10-27.csv.gz | 3922207.0000 | nan | 2026-10-27T04:55:42+00:00 |
| reentry_simulated | SIM | propagator_surrogate_v1/ood_test.parquet |  | nan | nan |
| reentry_simulated | SIM | propagator_surrogate_v1/test.parquet |  | nan | nan |
| reentry_simulated | SIM | propagator_surrogate_v1/train.parquet |  | nan | nan |
| reentry_simulated | SIM | propagator_surrogate_v1/val.parquet |  | nan | nan |

Source: `data/DATASET_MANIFEST.csv`. 
