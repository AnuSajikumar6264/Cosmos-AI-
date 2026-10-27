---
title: "COSMOS thesis: contribution, evidence gaps, Q1 readiness and roadmap"
---

# Appendix D: Contribution statement, evidence gaps and journal readiness {#sec:appendix-q1 .unnumbered}

## D.1 Research contribution and gap statement 

**Gap.** ESA-ADB makes operator-grade evaluation on real telemetry possible. Full-channel results under its official protocol remain low for the published baselines, and the benchmark authors stress false detections and the importance of thresholding and post-processing. Point-level conformal calibration has been shown not to fix it [@sahin2026tccad]. No pre-registered study has compared unsupervised, tree-based and deep supervised event scorers under the official split at full channel scale, with the alarm layer selected on all training labels.

**Contribution.** This thesis contributes such a study. Its outcomes are:

- the whole-training-half selection protocol is supported on Mission 1, under both alarm layers;
- the deep supervised scorer shows no benefit over either simpler scorer;
- the selection effect is conditional: it reverses on Mission 2 under BH and on the lightweight Mission 1 subset;
- aggregation to events with a strict, metric-selected operating point succeeds where point-level adaptive calibration fails, for a mechanical reason;
- a diagnostic, exploratory only, shows that supervised scores shift between training and test, but the shift does not track test performance.

The contribution is empirical and methodological. It is not a new architecture, and the main deep-learning result is negative.

## D.2 Unresolved evidence gaps and missing experiments 

1. **Brancato et al.'s protocol.** Their Mission 1 numbers (0.902; out-of-limit reference 0.754) cannot be reconciled with this study without their full text and code. *Not verified — the full text of [@brancato2027ast] is required.*
2. **Fresh data for a confirmatory test.** No confirmatory test has been run on data never seen by the project. All test halves were seen in the exploratory phase.
3. **Mechanism of the TCN's failure.** The post-hoc shift diagnostic (Mission 1, every fourth channel) does not order the runs by score, so the cause is open. *A pre-registered test is required.*
4. **Source of the selection effect.** A0 and A1 differ in five respects. *A factorial experiment is required to attribute the H6 gain.*
5. **Re-entry locations.** No validation against recorded re-entries exists. *Recorded re-entry events with recovery coordinates are required* (ESA DISCOS or Space-Track access).
6. **The coupling weights.** These are hand-set and have not been checked against fragmentation records. *DISCOS fragmentation data are required.*
7. **Statistical power.** Supervised-arm seed counts (5; 3 for the ablations) are low relative to the seed variance.
8. **Two literature items not read.** Fejjari et al. (Acta Astronautica) and Dai et al. (Advances in Space Research) were identified in the research design, but their full texts were not accessible here. *The author must read them before submission.*
9. **Unverified references.** RevIN and a telemetry survey by Zeng et al. could not be verified and are not cited.

## D.3 Critical assessment from a reviewer's perspective 

| Criterion | Assessment |
|---|---|
| Originality and significance | Moderate. The pre-registered, full-channel comparison and the event-versus-point calibration result are new. The calibration-and-triage space overlaps closely with TC-CAD [@sahin2026tccad], and the deep-learning result is negative. |
| Clarity of the gap | Good, provided the paper is framed around alarm selection and score stability rather than around a new model. |
| Technical rigour | High for the telemetry study: official metric code, a reproduced baseline, frozen hashes, gated test scoring, hierarchical bootstrap and Holm correction. The five-way confound of H6 and the point-tuned ACI comparator were addressed post hoc, by a decomposition and an event-tuned comparison (thesis Section 7.8), on test halves already seen. Lower for the physics chain, which has documented defects. |
| Experimental design and data | Real, expert-annotated data; but one benchmark, with one informative mission and two usable missions in total. |
| Baselines | Strong: label-free rules, the official GlobalSTD, trees on the same features, and a published comparator re-implemented. The published baselines did not use training labels, so comparisons with them need that context. |
| Statistical reliability | Adequate for H6. Inconclusive for the comparison of trees with AR(8), because of the seed variance. |
| Reproducibility | High: code public, evidence file, one-command rebuild. Large artefacts are not yet archived with a DOI. |
| Generalisation | Limited. The selection effect reverses on Mission 2 under BH and on the lightweight Mission 1 subset. No test on unseen data. |
| Limitations stated | Yes, extensively. |

**Overall.** In its present form, the telemetry study is publishable as a careful empirical paper. Acceptance by a Q1 venue is plausible but not assured, and it depends on the framing and on items 1–4 above. The physics chain and the supporting studies are thesis material rather than journal contributions in their present state.

## D.4 Prioritised action plan 

1. **Read and position against Brancato et al.** If possible, reproduce their protocol through this code (gap 1).
2. **Run a factorial H6** that crosses scores, calibration pool and selection data on one grid, to attribute the selection effect (gap 4). Done post hoc as a one-factor-at-a-time decomposition (P1): the selection data carry the effect.
3. **Pre-register and run a drift-aware supervised arm.** Use recalibration selected for that purpose or causal retraining, more seeds, and data not yet used, such as a later ESA release or another mission (gaps 2, 3 and 7).
4. **Add a forward-chaining control** to quantify the optimism of cross-fitting. Done post hoc (P2): the gain did not replicate by the pre-set rule.
5. **Write the manuscript** around one claim: selection and stability of the alarm layer matter more than model capacity. State its conditions (Mission 1 at full scale; reversals elsewhere), report the alarm burden, and report the refuted deep-learning hypotheses as a pre-registered negative result.
6. **Archive the code and score arrays on Zenodo** with a DOI, and add a licence and `CITATION.cff`.
7. **Leave the physics chain and supporting studies out of the paper,** or move them to a separate engineering paper after the defects in thesis Table 19 are fixed.

## D.5 Converting the thesis into a journal manuscript 

| Manuscript section | Thesis sources | Action |
|---|---|---|
| Introduction | thesis Chapter 1, thesis Section 2.8 | condense to the telemetry problem |
| Related work | thesis Chapter 2 (ESA-ADB, calibration) | keep thesis Table 1; add the two unread papers |
| Method | thesis Section 5.4.2, thesis Section 5.4.3, thesis Section 5.4.4, thesis Section 5.4.5 | as written; add the freeze procedure |
| Results | thesis Section 7.3, thesis Section 7.5, thesis Section 7.6 | thesis Table 10, thesis Table 12, thesis Table 13, thesis Table 14 and figures |
| Analysis | exploratory diagnostics | label as exploratory |
| Discussion and limitations | thesis Chapter 8, thesis Section 9.1 | keep the threats section complete |

The draft in `paper/manuscript.tex` already follows this outline. Its tables are generated by `scripts/build_paper_tables.py`.

**Venues.** The candidate venues, with indicative citation statistics, are listed in `research_design/VENUES.md`. The SJR quartiles there could not be verified from this machine and must be checked on the day of submission. On scope, Engineering Applications of Artificial Intelligence, Acta Astronautica and Reliability Engineering & System Safety fit an empirical study of alarm selection on spacecraft telemetry. The choice should follow the journal's current aims and scope, not its impact factor.



# References
