# Submission checklist: COSMOS-AI digital-twin manuscript

## Build

```
python paper/journal_v3_twin/tools/build_numbers.py      # numbers.json from the result files
python paper/journal_v3_twin/tools/make_architecture.py  # artwork/Figure_4 (architecture)
python paper/journal_v3_twin/tools/make_figures.py       # result and benchmark figures, Figure_S1..S3 (PDF + PNG)
PYTHONPATH=. COSMOS_DEVICE=xpu:0 python paper/journal_v3_twin/tools/make_data_figures.py
                                                         # dataset and model-output figures (examples cached in outputs/figure_data/)
python paper/journal_v3_twin/tools/make_supplementary.py # build/supplementary.{md,docx}
python paper/journal_v3_twin/tools/render.py             # build/manuscript.{md,docx,tex}
python paper/journal_v3_twin/tools/render_doc.py paper/journal_v3_twin/src/highlights.src.md paper/journal_v3_twin/build/highlights.md
cd paper/journal_v3_twin/build && mkdir -p tex && cp manuscript.tex tex/ && cp -r ../artwork tex/ \
  && cd tex && /data/anupa/COSMOS_work/.tools/bin/tectonic -X compile manuscript.tex   # LaTeX PDF
```

The Markdown sources (`src/*.md`) and `build/manuscript.docx` are editable. The build aborts if
any number is missing.

## What is ready

- [x] Manuscript sections:
  - title, abstract (227 words), keywords;
  - introduction, related work, data, framework, protocol, results, discussion, limitations,
    conclusions, declarations.
- [x] Every number is a placeholder filled from a named result file.
- [x] Figures 1–19: Matplotlib, vector PDF + 300 dpi PNG, files numbered in order of appearance
  (`FIGURE_ORDER` in `tools/make_figures.py`): dataset views (1–3), architecture (4), results,
  model outputs and benchmarks (5–19). Figure 4 (architecture) can serve as the graphical
  abstract. Examples in the output figures are chosen by seeds or stated rules.
- [x] Supplementary Figures S1–S3.
- [x] Highlights: 5 items, each at most 85 characters (`build/highlights.md`).
- [x] Supplementary Tables S0–S18 (`build/supplementary.md` / `.docx`).
- [x] References: `refs.bib`, entries verified via Crossref or the arXiv API (the `note` field
  records how).
- [x] Deviations and disclosures:
  - the learner-pool change after run 1;
  - T1 not fully blind;
  - post-hoc analyses labelled as such.

  They are in the manuscript and in `docs/REVIEW_AND_FIXES.md`.
- [x] Candidate journals with verified indexing and quartiles: `VENUES.md`.

## What only the authors can do

- [ ] Author names, affiliations, corresponding author, ORCID; CRediT statement (`src/back.md`).
- [ ] Funding and competing-interest statements.
- [ ] Review and adapt the generative-AI declaration to the actual use.
- [ ] Choose the journal (see `VENUES.md`) and apply its template. Check its limits:
  - abstract length;
  - number of keywords;
  - acronyms in the title or abstract (Engineering Applications of AI and the AIAA journals
    forbid them);
  - single-column format.
- [ ] Deposit the derived data and the dated forecast in Zenodo, and insert the DOI (`src/back.md`).
- [ ] Push the code to a public repository and archive a release.
- [ ] Read in full the closed-access papers listed in `docs/RESEARCH_GAP_AND_NOVELTY.md`
  ("Must be read in full before claiming novelty").
- [ ] Decide whether the non-commercial TabPFN-3.5 weights may be used. They are currently
  excluded; see `docs/SOTA_MODEL_SELECTION.md` section 7.
- [ ] After a later CelesTrak snapshot, score the hashed forecast of 2026-10-27
  (`outputs/twin/as_of_2026-10-27/`). The SHA-256 of `twins.jsonl` is printed in the manuscript.

No submission or acceptance is guaranteed by any of the above.
