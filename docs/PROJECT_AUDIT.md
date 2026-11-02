# COSMOS project audit (2026-10-27, before the re-entry rebuild)

Checkpoint: the state audited here is commit `2427cf8` on branch `q1-reentry-rebuild` (the
`main` branch is unchanged at `2525367`). Every statement below was checked against the code,
the data manifests or the result files named in it.

## 1. What the repository contains

| Area | Code | Data class | Status before the rebuild |
|---|---|---|---|
| Telemetry anomaly detection | `cosmos/ai/telemetry/` | REAL (ESA-ADB, OPSSAT-AD) and SYNTHETIC (COSMOS pipeline) | Mature. Pre-registered ESA-ADB study with post-hoc validation analyses P1–P8 and an Elsevier-format manuscript (`paper/journal_v2/`). OPSSAT-AD benchmark with five seeds. |
| Debris detection in images | `cosmos/ai/vision/`, `cosmos/debris/detect.py` | REAL (SpotGEO) and SYNTHETIC (COSMOS frame pairs) | Working. SpotGEO official F1 0.746 ± 0.005 (3 seeds). |
| Debris catalogue and lifetime | `cosmos/debris/catalogue.py`, `cosmos/debris/lifetime.py` | REAL (CelesTrak) | Physics lifetime model with a static atmosphere (see 3.1). |
| Re-entry within a horizon | `cosmos/ai/survival/landmark.py` | REAL (CelesTrak snapshots) | One landmark (2026-08-16), 68-day follow-up, 38 test events (12 uncontrolled). |
| Re-entry trajectory, breakup and ground risk | `cosmos/reentry/` | SIMULATED | Physics chain with documented defects (thesis Appendix C); surrogate MLP of the propagator (SIM). |
| Fusion of evidence | `cosmos/ai/fusion.py` | rule-based | Combines monitor evidence and a simulated footprint; not evaluated against real outcomes. |

## 2. What is validated on real observations

- **Telemetry:** ESA-ADB Missions 1 and 2 under the official metric (pre-registered); OPSSAT-AD on the official split. These are the only components with confirmatory, real-data validation.
- **Debris imagery:** SpotGEO with the official scorer.
- **Re-entry:** only the single-landmark study above. ESA recovered-debris records (235 fragments, 87 objects) were used for record linkage and latitude-band checks, not as impact-point ground truth.
- Nothing in the re-entry trajectory, breakup or ground-risk chain has been compared with real re-entry observations; those results are simulator outputs and are labelled SIM.

## 3. Defects and weaknesses found (relevant to the new objective)

1. **Physics lifetime model** (`cosmos/debris/lifetime.py`): circular orbits only, US-76 static density (no solar or geomagnetic dependence), one area-to-mass value for all objects. Not a fair physical baseline for re-entry prediction.
2. **Breakup model** (`cosmos/reentry/breakup.py`): the area-to-mass sampler does not match the NASA Standard Breakup Model (Johnson et al. 2001). Its spacecraft-fragment mixture weight, means and spreads differ from the published equations (checked against ESA's reference implementation, commit ddd96572); the delta-v law was already listed as non-standard in the thesis defect register.
3. **Single landmark** for re-entry prediction: 68 days of follow-up and 38 test events give wide intervals and cannot support a temporal-generalisation claim.
4. **No paired data** link telemetry health to catalogue objects: ESA-ADB and OPSSAT-AD missions are anonymised or unrelated to the debris population, so a telemetry-to-re-entry fusion has no real ground truth.
5. **Data access limits**: Space-Track (TLE histories, decay messages), ESA DISCOS and ESA re-entry predictions need accounts that were not available; no public archive of historical TLEs was found (Zenodo searches, 2026-10-27).

## 4. What the rebuild keeps and adds

- Kept unchanged: all telemetry, vision and simulator code and results; the earlier landmark study (as a secondary analysis).
- Added (this rebuild): `cosmos/ai/lifetime/` (catalogue-wide re-entry study on GCAT + CelesTrak, physics expert with NRLMSIS 2.1 and the published SBM, learned experts, reliability-gated fusion, Mondrian conformal calibration); fetchers for GCAT, the ESA Kelvins datasets and NASA SMAP/MSL in `cosmos/ai/data/download.py`; documentation in `docs/`.
- The defective `cosmos/reentry/breakup.py` sampler and `cosmos/debris/lifetime.py` model are left in place for the existing simulator results; the new study does not use them.
