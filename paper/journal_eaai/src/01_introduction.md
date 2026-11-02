# Introduction {#sec:intro}

More than 30,000 tracked objects share low Earth orbit with the satellites that operators must keep healthy, steer clear of debris and eventually bring down. Three questions govern the resulting debris environment. How long does each object remain in orbit? How likely is a given encounter to become a collision? And which objects are still under control, and therefore not decaying freely at all? In operational practice each question is answered by a separate tool, and the uncertainty those tools report is rarely calibrated.

The three questions are not independent. A spacecraft that loses the ability to manoeuvre stops station-keeping and begins to decay; fragments of one breakup share an origin, and therefore a range of area-to-mass ratios that shapes how long they stay in orbit; and how long fragments remain aloft determines the population that conjunction assessment must screen. A digital twin of the tracked population, with one state per object that every function reads and updates, is the natural way to make these links explicit [@liu2024digital].

Building such a twin on public data meets a hard constraint: no public dataset follows the same object through telemetry, conjunction messages and re-entry. Telemetry benchmarks are anonymised [@kotowski2026esaadb; @hundman2018], conjunction messages are anonymised [@uriot2022spacecraft], and catalogue data carry no telemetry. A credible twin therefore cannot be a single model trained over unpaired data. It must be a set of modules, each validated on its own real data, joined by couplings that are defined explicitly and tested wherever a real test exists.

This paper presents such a system. Its contribution in artificial intelligence is a **reliability-gated mixture of experts with distribution-free calibration**: a learned gate weights a physics-based prior against recent tabular neural models, and every interval and probability the system emits is calibrated by conformal prediction and isotonic regression. Its application in engineering is **forecasting the low-Earth-orbit debris environment**, across residual orbital lifetime, collision risk and spacecraft health.

The contributions are:

1. A reliability-gated mixture-of-experts architecture for censored regression on an operational catalogue, evaluated against eight alternative fusion arms including the physics prior alone, each learned expert alone, an equal-weight average, a learned gate without reliability features, and a gradient-boosted baseline.
2. A licence-clean, leakage-controlled benchmark of {{data.cohort.rows_out:,}} catalogued objects with administrative censoring and two independent forward-in-time test periods.
3. Four explicit cross-module couplings, three of which are tested on real data, with their measured contribution and their statistical significance reported.
4. Evidence that conformal prediction's marginal guarantee degrades under temporal drift, and that a catalogue covariate can leak the outcome through its availability rather than its value.
5. An honest account of where learning helps and where it does not, including a central negative result: the learned gate never beat the best single expert.
