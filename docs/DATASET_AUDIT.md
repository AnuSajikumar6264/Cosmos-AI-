# Dataset audit (retrievals of 2026-10-27 unless stated)

Every file listed here has an entry with size, MD5 and SHA-256 in its folder's `MANIFEST.json`;
`data/DATASET_MANIFEST.csv` collects them (`python scripts/build_dataset_manifest.py`). Raw
downloads live under `raw/`, derived tables under `processed/` or `derived/`, and generated or
simulated data in separate `cosmos_synthetic/` and `simulated/` folders. The `class` column of
the manifest says REAL, DERIVED, SYNTHETIC or SIM.

## Downloaded and used

| Dataset | Folder | Class | Licence | Content and size | Used for | Limitations found |
|---|---|---|---|---|---|---|
| GCAT, J. McDowell, release 1.8.8 (catalogue updated 2026-10-26) | `feature2_debris/gcat/raw/2026-10-27/` | REAL | CC-BY-4.0 (citation required) | `satcat.tsv`: 69,999 objects (catalogue numbers 1–69,999); `auxcat`, `psatcat`, `ftocat`, `launch`, `orgs` tables | re-entry study (canonical orbits, decay dates, status, design mass and dimensions, breakup parents) | One canonical orbit per object (no history). Design mass and dimensions are 0 (unknown) for all fragments and about half of the components. `?` flags mark uncertain values. Continuously revised, hence dated snapshots. |
| CelesTrak SATCAT, GP sets, `SW-All.csv` | `feature2_debris/celestrak/snapshots/2026-10-27/` (and 2026-08, 2026-10-24) | REAL | CelesTrak usage policy; no explicit licence | 71,149 SATCAT rows (35,749 decayed); space weather 1957-10-01 to 2041-10-01 | operational status (health proxy), current orbits for the twin, space-weather covariates; radar cross-section only for the class-level size prior of the physics expert | RCS is missing for 54 % of rows overall and for **100 % of objects launched in the 2020s** (57 % for the 2010s, 11 % for the 1990s and 2000s). **Whether an RCS value exists depends on the outcome**: objects that decayed within weeks were mostly never given one (origin-2016 training split: P(decay ≤ 30 d) = 0.76 without RCS against 0.17 with it). RCS is therefore not used as a predictor (`docs/TARGET_DEFINITIONS.md`). `SW-All.csv` mixes observed (OBS, INT) and predicted (PRD, PRM) rows; only observed rows are used. |
| ESA Kelvins Collision Avoidance Challenge (Uriot et al.) | `feature2_debris/kelvins_cac/` | REAL | CC-BY-4.0 | 162,634 training CDMs (13,154 events), 24,484 test CDMs (2,167 events) with the previously private final risks; raw data 2015–2019 | conjunction-risk module | CDMs anonymised (no object identities or dates), so they cannot be linked to catalogue objects. Only 66 high-risk events (final log10 risk ≥ −6) among the 8,293 training events that match the test protocol; 150 in the test set. Covariance determinants reach 1e46 (log-transformed). |
| NASA SMAP/MSL telemetry (Hundman et al., KDD 2018) | `feature1_health/smap_msl/` | REAL | Caltech copyright, BSD-style redistribution licence; Kaggle copy "data files © original authors" | 82 channels (55 SMAP, 27 MSL), 105 labelled anomaly sequences; the original telemanom run outputs | health module (supporting benchmark) | The original S3 archive returned HTTP 403 on 2026-10-27; the Kaggle copy linked from the telemanom README was used, and its labels are byte-identical to the repository's. Values are scaled with the test-set min/max (stated by the authors). |
| ESA-ADB v1.0, OPSSAT-AD | `feature1_health/esa_adb/`, `feature1_health/opssat_ad/` | REAL | see manifests | as in the telemetry study | health module | ESA-ADB missions are anonymised; neither dataset can be linked to catalogue objects. |
| ESA recovered debris (DISCOS web, public list) | `feature3_reentry/historical/` | REAL | ESA web content, no explicit data licence | 235 fragment records (87 objects), retrieved 2026-10-24 | context and record linkage only | Not an impact-point ground truth (no re-entry state vectors). |
| NRLMSIS 2.1 density table (pymsis 0.13.0) | `feature3_reentry/processed/msis21_density_table.npz` | DERIVED (model) | pymsis: MIT-style | log density on altitude × F10.7 × Ap, averaged over season, local time and latitude | physics expert | A global average: no latitude, local-time or seasonal variation along the orbit. |
| Re-entry cohort table | `feature3_reentry/processed/lifetime_cohort_2026-10-27.csv.gz` | DERIVED | inherits GCAT/CelesTrak terms | 64,516 objects (26,066 fragments, 25,304 payloads, 7,578 components, 5,568 rocket stages); 32,970 natural decays | re-entry study | see `docs/TARGET_DEFINITIONS.md`; exclusions are counted in the build log |

## Downloaded but not used in the main studies

| Dataset | Reason |
|---|---|
| ESA Kelvins "Space debris: the origin" (Zenodo 4986292, CC-BY-4.0) | **SIMULATED**: the challenge page's background is titled "fictional", the equations of motion are stated exactly and the generation epochs lie on a regular 5- and 10-day grid (see `docs/RELATED_WORK.md`). Geostationary orbits only (semi-major axis about 42,260 km), unrelated to atmospheric re-entry; test labels are not public. Kept on disk, not used. |

## Not obtained, with the exact reason

| Source | What it would add | Status |
|---|---|---|
| Space-Track.org (TLE histories, decay and TIP messages) | orbital-element time series up to re-entry, i.e. re-entry epoch prediction in the IADC style | requires an account and acceptance of the user agreement; no credentials were available, and no public archive of historical TLEs was found (Zenodo searches of 2026-10-27). Not attempted. |
| ESA DISCOSweb | physical properties and fragmentation events | requires a personal token; not attempted |
| ESA re-entry predictions (`/api/protected/reentries`) | operational re-entry predictions for comparison | HTTP 401 (account required), checked 2026-10-24 |
| Aerospace Corp. CORDS re-entry database | predicted versus actual re-entry windows | HTTP 403 to scripted access on 2026-10-27; access controls were not bypassed |
| REAT re-entry emission inventory 2019–2025 (Zenodo 22981689, 3.4 GB) | rocket-body re-entry scenarios with mass profiles | not downloaded: GCAT already supplies re-entry dates and design masses for these objects |

## Real versus synthetic results

All results of the re-entry, conjunction and telemetry modules in `docs/RESULTS_AND_ANALYSIS.md`
are computed on the REAL datasets above. The earlier simulator outputs (`feature3_reentry/simulated/`,
`feature1_health/cosmos_synthetic/`, `feature2_debris/cosmos_synthetic/`) are not used in the new
studies and remain labelled SIM or SYNTHETIC.
