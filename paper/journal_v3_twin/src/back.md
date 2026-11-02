# Declarations {.unnumbered}

**CRediT authorship contribution statement.** [Author 1]: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization. [Author 2]: Supervision, Writing – review & editing. [To be completed by the authors.]

**Data availability.** All data used are public. GCAT [@mcdowell2020gcat] is available under CC-BY-4.0; CelesTrak catalogue and space-weather files [@celestrak2026satcat] under CelesTrak's usage policy; the ESA Kelvins Collision Avoidance Challenge data [@uriot2021cacdataset] under CC-BY-4.0; ESA-ADB [@esaadb_dataset] (companion study) and NASA SMAP/MSL [@hundman2018] under the terms stated by their authors. The derived cohort table, the NRLMSIS density table, all predictions and result files, and the dated twin forecast will be deposited in Zenodo with a DOI before publication [DOI to be added]. No Space-Track data are used or redistributed. Pretrained model weights are obtained from their publishers under their own licences:
- TabICLv2: BSD-3-Clause;
- Kumo-Tabular: OpenMDW-1.1;
- Chronos-2: Apache-2.0;
- NRLMSIS 2.1, via pymsis.

**Code availability.** The code (data fetchers with checksummed manifests, the re-entry, conjunction and telemetry modules, the twin engine, and the scripts that produce every table and figure from the result files) is available on GitHub [@cosmos_software]; an archived release with a DOI will be created before publication.

**Funding.** [To be completed by the authors.]

**Declaration of competing interest.** [To be completed by the authors.]

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work the author(s) used Claude (Anthropic) to assist with writing analysis and plotting code and with drafting and editing the text. After using this tool, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article. All figures were rendered with Matplotlib from the study's result files; no figure was generated or altered by a generative AI model. [The authors must review this statement and adjust it to their actual use.]

**What was not validated.** The C4 coupling (health × debris → collision response) is a demonstration: no public dataset pairs conjunction messages with the health of the same spacecraft, and operational status is a proxy for manoeuvre capability, not a measurement of it. The twin forecast as of 2026-10-27 is a prospective forecast; it will be verified against later catalogue snapshots. The as-of-date test T1 covers one 72-day window that had been analysed earlier in the project, so it is not a blind test (Section [-@sec:protocol]).

# References {.unnumbered}

::: {#refs}
:::
