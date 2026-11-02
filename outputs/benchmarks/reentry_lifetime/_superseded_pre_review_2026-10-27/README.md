# Superseded results (pre-review), kept for provenance only

These re-entry outputs were produced before the adversarial code review of 2026-10-27 and
must not be used or cited. The review found, among others:

- outcome-dependent leakage through the 2026 radar cross-section (feature, physics area-to-mass
  source and gate descriptor);
- King-Hele's small-eccentricity series used for eccentric orbits;
- a breakup-fragment area-to-mass law applied to intact objects;
- cap values that were not snapped to the bound;
- rank metrics computed on tied isotonic outputs;
- a learner cache that was not fingerprinted.

The corrected pipeline writes to `../origin<O>_uncontrolled/`. The pre-review cohort table is
`lifetime_cohort_2026-10-27_pre_review.csv.gz`; its space-weather windows include the day of t0.
