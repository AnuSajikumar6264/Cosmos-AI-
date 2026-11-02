# Candidate journals for the COSMOS-AI digital-twin manuscript

Checked on 2026-10-27 by a 24-agent verification workflow. The sources were:
- Elsevier's Scopus source-title list of September 2026;
- live Scopus source records;
- SCImago pages and widgets, or archived copies where the live SCImago site blocked scripted
  access (Cloudflare HTTP 403);
- publishers' journal and APC pages.

A quartile is given only when it was read from SCImago itself, or from a copy of SCImago's data
export. Anything else is marked *unverified*. SCImago released its 2025 edition in April 2026,
but for several journals only the 2024 page could be read.

**Nothing here guarantees acceptance.** Re-check the quartile, the APC and the scope on the live
pages before submitting. Raw records with every source URL are in
`/data/anupa/COSMOS_work/.scratch/venues_verified.json`.

| Journal (publisher) | Scopus | SJR (year, source) | Best quartile (verified year) | Scope fit (1–5) | Open access / APC |
|---|---|---|---|---|---|
| Acta Astronautica (Elsevier for IAA) | yes | 1.078 (2025, Scopus); 1.277 (2024, SCImago) | Q1 Aerospace Eng. (2024); 2025 unverified | 4 | hybrid; APC USD 3,750 |
| Advances in Space Research (Elsevier for COSPAR) | yes | 0.675 (2025, SCImago export copy) | Q1 Aerospace Eng. and Earth & Planetary Sci. (misc.) (2025); Q2 in its other categories | 4 | hybrid; APC USD 3,058 |
| Aerospace Science and Technology (Elsevier) | yes | 1.39 (2025, SCImago widget) | Q1 Aerospace Eng. (2025) | 4 | hybrid; APC USD 4,280 |
| Engineering Applications of Artificial Intelligence (Elsevier, IFAC) | yes | 1.782 (2025, Scopus and SCImago export) | Q1 AI, Control & Systems Eng., EEE (2025) | 4.5 | hybrid; APC USD 3,040 |
| Reliability Engineering & System Safety (Elsevier) | yes | 2.838 (2025, Scopus); 2.647 (2024, SCImago) | Q1 all three categories (2024); 2025 unverified | 4 (only if framed as risk) | hybrid; APC USD 4,750 |
| IEEE Trans. Aerospace and Electronic Systems | yes | 1.528 (2024, SCImago) | Q1 Aerospace Eng. and EEE (2024); 2025 unverified | 4 | hybrid; APC USD 2,800 |
| Expert Systems with Applications (Elsevier) | yes | 1.854 (2024, SCImago) | Q1 all three categories (2024); 2025 unverified | 4 | hybrid; APC USD 3,630 |
| Knowledge-Based Systems (Elsevier) | yes | 1.934 (2024, SCImago) | Q1 all four categories (2024); 2025 unverified | 4 | hybrid; APC USD 3,240 |
| Astrodynamics (Tsinghua Univ. Press / Springer) | yes | 2.422 (2025, SCImago widget and Scopus) | Q1 Aerospace Eng. (2025) | 3 | hybrid; APC USD 4,090 |
| Journal of Spacecraft and Rockets (AIAA) | yes | 0.805 (2025, Scopus); 0.821 (2024, SCImago) | Q1 Aerospace Eng., Q2 Space & Planetary Sci. (2024); 2025 unverified | 3.5 | hybrid; APC USD 2,700 |
| Journal of Space Safety Engineering (Elsevier for IAASS) | yes | 0.551 (2025, Scopus) | **unverified**; a secondary site reports Q2 | 4 | hybrid; APC USD 2,930 |
| Aerospace (MDPI) | yes | 0.584 (2025, Scopus); 0.580 (2024, SCImago) | **Q2** Aerospace Eng. (2024); 2025 unverified | 5 | gold OA; APC CHF 2,400 |

The scope-fit scores are the reviewing agents' judgement against each journal's published aims
and scope. They are not a metric.

## Points that affect the choice

- **Acta Astronautica.** The strongest topical fit among the Q1 aerospace journals. It publishes
  collision-probability estimation, telemetry anomaly detection and re-entry papers, and two
  calls are open:
  - the IAA Conference on Space Situational Awareness special issue (deadline 30 Nov 2026);
  - "AI in Space" (deadline 15 Dec 2026).

  Eligibility for either call was not checked. Frame the machine-learning parts around
  space-safety outcomes, not as an ML benchmark.
- **Advances in Space Research.** Submit to section AD (astrodynamics). Reviewers there will
  expect astrodynamics-grade validation:
  - re-entry error against observed decay dates;
  - calibration of probabilities;
  - comparison with the physics propagator.

  The telemetry module is furthest from its core.
- **Engineering Applications of AI.** The journal states three desk-rejection rules:
  1. the abstract must say which part is the AI contribution and which is the engineering
     application;
  2. no undefined acronyms in the title or abstract (spell out or drop COSMOS-AI, NRLMSIS,
     GCAT, CDM, ESA-ADB);
  3. single-column format.
- **Journal of Spacecraft and Rockets.** The title may have at most 12 words and no acronyms.
  The abstract must be 100–200 words, also without acronyms.
- **Expert Systems with Applications.** It no longer considers military or defence applications.
  Present conjunction work strictly as civil space safety.
- **Journal of Space Safety Engineering** and **Aerospace (MDPI)** are good topical fits but not
  verified Q1.

## Suggested order, given the results

The measured results support the following:
- a re-entry benchmark on two temporal origins, where no expert wins both periods;
- a prospective test showing that history-conditioned physics fixes the twin's re-entry forecasts
  for objects in orbit;
- a documented outcome leak in catalogue covariates;
- a collision-risk module whose best probabilistic forecaster does not beat the latest-risk
  baseline on the official metric;
- a supporting telemetry benchmark;
- couplings with honest confidence intervals.

That profile fits an astrodynamics or space-safety venue better than an AI-methods venue.

1. **Acta Astronautica** (Q1 2024; strong fit; open SSA and AI-in-Space calls).
2. **Advances in Space Research**, section AD (Q1 Aerospace Eng. 2025).
3. **Aerospace Science and Technology** (Q1 2025). Lead with the re-entry contribution.

If the manuscript is reframed around the gate and conformal calibration as one AI method,
**Engineering Applications of AI** (Q1 2025) is the AI-side option; observe its rules above.
