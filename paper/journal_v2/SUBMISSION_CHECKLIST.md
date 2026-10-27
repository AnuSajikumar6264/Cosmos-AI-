# Submission checklist (Elsevier guide for authors)

Status: **done** = in the build; **authors** = needs information or an action only the authors can provide.

| Guide item | Where | Status |
|---|---|---|
| Title: concise, informative, no abbreviations | `src/paper.md` front matter | done |
| Author names, affiliations, corresponding author with e-mail | front matter placeholders | **authors** |
| Abstract ≤ 250 words: purpose, procedures, findings, conclusions; no references; abbreviations defined | front matter; the build checks the word count | done |
| 1–6 keywords, without "and" or "of" | front matter (6); the build checks them | done |
| Classification codes (up to 6 Inspec codes) | front matter (5 codes, confirmed only on archived IET pages from 2013–2019) | **authors**: check against the current Inspec list, or delete if the journal does not ask for them |
| Highlights: 3–5 bullets, ≤ 85 characters, separate editable file | `build/highlights.docx` | done |
| Graphical abstract: separate file, ≥ 1328 × 531 px, readable at 13 × 5 cm, TIFF/EPS/PDF | `artwork/Graphical_abstract.tiff` (1535 × 708 px), `.pdf` | done |
| Units: SI | s, h, d; "per year" for alarm rates | done |
| Mathematical formulae: editable, solidus for small fractions, italic variables, consecutive display numbering | Word equations; Eqs. (1)–(6) | done |
| Appendix equations numbered (A.1), (A.2) … | Eqs. (A.1), (A.2), (B.1) | done |
| Tables: editable, numbered in order of citation, caption above, notes below, no vertical rules, no duplication of figures | Tables 1–5 | done |
| Figures: separate files, vector with embedded fonts, numbered in order of citation | `artwork/Figure_1.pdf` … `Figure_3.pdf` (TrueType fonts embedded) | done |
| Figure captions: title plus description, not on the figure | captions in the manuscript | done |
| Colour accessible to colour-blind readers | blue/orange pairs plus marker shape (circle/square) | done |
| Generative-AI use in images | figures are rendered by Matplotlib from result files; stated in the declaration | done |
| Supplementary material: cited in the text, captioned | `build/Supplementary_Material.docx`; Tables S1–S26, Figs. S1–S3, numbers resolved at build time | done |
| Research data, Option C: deposit, cite and link; data statement | ESA dataset cited as `[dataset]`; data statement written | **authors**: deposit derived scores, alarms and evidence file on Zenodo and insert the DOI |
| Code | GitHub repository cited as `[software]` | **authors**: push the current code (validation scripts and manuscript build are not yet committed), archive a release with a DOI and update the `[software]` reference |
| Article structure: numbered sections; Introduction with objectives; reproducible Methods; Results; Discussion without repeating results; Conclusions | sections 1–6 | done |
| CRediT authorship statement | template in Declarations | **authors** |
| Funding, competing interests | placeholders | **authors** |
| Declaration of generative AI in the writing process | written; names the tool and its use | **authors**: confirm it matches your actual use |
| References: author–year, alphabetical list, DOIs, abbreviated journal titles (LTWA), `[dataset]`/`[software]` entries, published versions instead of preprints | `refs_journal.bib`, `tools/elsevier-harvard.csl` | done (three works exist only as preprints and are marked as such) |
| Journal-specific template and length limit | — | **authors**: choose the journal, then apply its template |

## Before you press submit

1. Choose the journal (`VENUES.md`) and check its current Scopus status and quartile on Scopus or SJR.
2. Fill in every placeholder in square brackets in `src/paper.md` (authors, affiliations, CRediT, funding, competing interests, data DOI) and rebuild.
3. Read Brancato et al. (2027), which is paywalled, and check that the manuscript describes its protocol correctly.
4. Commit and push the code, archive it, and update the `[software]` reference.
5. Write a cover letter that states the pre-registration, the validation plan and the reuse of the test halves openly.
