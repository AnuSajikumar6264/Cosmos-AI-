# Run 1: first learner set (kept as a record; reported as a deviation)

Learners: XGBoost, LightGBM, CatBoost, quantile MLP and TabICLv2; the expert was selected among
all five on tune-A. On the user's request of 2026-10-27, the learned-expert candidates were then
restricted to models published 2022-2026 (TabM, TabICLv2, Kumo-Tabular), with the others kept
as reference baselines. The run with that set replaces this one in `../`.

When the change was made, the test results of this run had been seen for origin 2011 and, by
the time the new run started, also for origin 2016. Everything else (cohort, physics, features,
seeds, budget, gate, calibration) is unchanged; cached learners are reused through their
fingerprints.
