| Study | Scale | Censoring | Output | Evaluation split | Cross-module couplings |
|---|---|---|---|---|---|
| Salmaso et al. [@salmaso2023machine] | 417 bodies; test 15 campaign objects | decayed only | point epoch | interleaved in time | none |
| Lotti et al. [@lotti2025exploring] | 1,783 after filtering; 1,435 train | decayed only | point lifetime | random, stratified | none |
| Jung et al. [@jung2021recurrent] | not readable (closed access) | not readable | point epoch | not readable | none |
| Shambaugh [@shambaugh2026validation] | 1,283 objects | decayed only | distribution | backtest, calibrated on itself | none |
| This work | {{data.cohort.rows_out:,}} objects, two periods | censoring-aware | seven quantiles + intervals | forward in time, two origins | three tested |
