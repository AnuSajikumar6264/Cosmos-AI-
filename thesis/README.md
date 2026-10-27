# COSMOS thesis: sources, evidence and build

| Path | What it is |
|---|---|
| `build/COSMOS_Thesis.docx` | **the thesis**: editable Word, with native equations, numbered figures, tables and equations, and IEEE citations |
| `build/COSMOS_Thesis.pdf` | **the thesis as PDF**: table of contents with page numbers and clickable bookmarks. Exported by LibreOffice, which substitutes Liberation Serif, a metric-compatible equivalent, for Times New Roman |
| `build/Q1_Readiness_and_Roadmap.docx` | Appendix D as a separate document: contribution and gap statement, evidence gaps, reviewer-style assessment, roadmap |
| `build/COSMOS_Thesis.md` | the rendered Markdown, with every number substituted |
| `src/*.md` | chapter sources. Numbers appear only as `{{evidence.key}}` placeholders |
| `evidence/evidence.json` | every reported value, with its source file, kind (measured / derived / config / literature) and derivation |
| `evidence/evidence_map.md` | the same, as a table |
| `evidence/original_pipeline_runs.json` | original-pipeline results regenerated with the current code |
| `evidence/reference_verification_log.tsv` | how each reference was verified (DOI, arXiv or publisher record) |
| `refs.bib` | the verified bibliography |
| `figures/` | result figures, drawn from `evidence.json` only, and architecture diagrams (graphviz `.dot` sources included) |
| `tools/` | `build_evidence.py`, `run_original_pipeline.py`, `make_figures.py`, `make_diagrams.py`, `make_reference_docx.py`, `render.py` |

## Rebuild

```bash
.venv/bin/python thesis/tools/run_original_pipeline.py
.venv/bin/python thesis/tools/build_evidence.py
.venv/bin/python thesis/tools/make_figures.py
.venv/bin/python thesis/tools/make_diagrams.py
.venv/bin/python thesis/tools/render.py     # aborts if a placeholder has no evidence
/data/anupa/COSMOS_work/.tools_lo/opt/libreoffice26.8/program/python thesis/tools/export_pdf.py   # PDF
```

`render.py` needs pandoc 3.11 and pandoc-crossref. Both were installed outside the project, under `/data/anupa/COSMOS_work/.tools*`. In Word, refresh the table of contents with right-click → *Update Field*.

## Items the author must complete

- Degree title, institution and department (title page).
- Acknowledgements and a signed declaration.
- Read Brancato et al. (AST 2027), Fejjari et al. and Dai et al., then complete the related work.
- Verify the journal quartiles on scimagojr.com before choosing a venue.
