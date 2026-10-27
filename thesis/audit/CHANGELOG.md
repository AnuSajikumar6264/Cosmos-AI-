# Thesis change log

Backup of the state before the Stage-1 audit (thesis/, paper/, research_design/preregistration/):
`/data/anupa/COSMOS_work/backups/thesis_paper_prereg_2026-10-25_stage0.tar.gz`
(SHA-256 `09a4ac88ae5d962b98f2f9193740fe71a6009d5f70d438dc6bf57842166cbe3f`).

No result file under `outputs/` and no frozen file (`research_design/preregistration/freeze.json`) has been changed. `DEVIATIONS.md` gained one appended row (deviation 6). Nothing has been committed or pushed.

## 2026-10-25, pass 1: corrections after the independent QA review

| File | Original issue | Change | Reason | Verification | Remaining uncertainty |
|---|---|---|---|---|---|
| `thesis/src/00_front.md` | Abstract said the verification checks were "against independent references"; it called H6 a selection effect; it said a diagnostic "located" the TCN failure | Rewritten with evidence keys: 10/9/1 split of the checks; H6 as a four-way protocol comparison; shift "consistent with" the result, not its cause; alarm burden added | QA items (a)1, (b)1–3, 17 | Values re-read from `evidence.json`; render resolves every placeholder | — |
| `thesis/src/02_literature.md` | Range-based metrics credited to Huet et al.; focal-loss variant credited to Lin et al.; BH/PRDS/Bates attributions; Nguyen "mostly lightweight"; Grieves/Glaessgen order; TC-CAD ACI "near target"; DMLR "main weakness" | Citations corrected (Tatbul 2018, CornerNet, Benjamini–Yekutieli 2001); claims reworded to what the papers state | QA items (a)13–14, 16–17; (b)10, 21 | Bibliographic records verified (log in `evidence/reference_verification_log.tsv`) | Fejjari et al., Dai et al. and the Brancato et al. full text are not read |
| `thesis/src/03_theory.md` | BH guarantee implied for per-channel pools; "moving-block" bootstrap; "within 0.008"; SpotGEO MSE "averaged"; US76 boundary; "0.6 m radius" | Caveat added (per-channel pools, no FDR guarantee); fixed 30-day blocks with block counts; derived gap 0.0081; summed MSE, ties only; 84.852 km geopotential; √0.36 m² | QA items (a)3, 15; (b)10–11; (c) | Code read: `risk.py` l.27, `event_bootstrap.py`, `spotgeo_validation.py` l.141 | — |
| `thesis/src/04_architecture.md` | Onset-cap "leak"; IQR scale for the monitor; tie-break missing; realdata scaling; scene ranges 5–12/90–220; "log-loss"; Δv law; event-level area | Corrected from the code | QA items (b)18–20; (c) | `monitor.py` l.127–130 (cap never binds: 1,152 < 1,584), `scene.py` l.171–179, `lifetime.py` l.71–84, `breakup.py` l.28, 84, 140 | — |
| `thesis/src/05_methods.md` | Windows "centred on labelled samples"; γ rule "used by" TC-CAD; freeze "28 code files"; rolling layer not described as copied; anomaly arm not reported | Corrected; anomaly arm reported as not run; AdamW and one-cycle cited | QA items (a)11; (b)14, 16, 21; (c) | `event_model.py` l.221–222; `oof_benchmark.py` l.128–130; freeze file count | — |
| `thesis/src/07_results.md` | H6 reported as a pure selection effect; "worse than both"; ACI "at most 0.117"; H3 metric missing; OPSSAT "every model drops"; "hybrid best"; uncontrolled landmark subset omitted; event-class explanation wrong; alarm burden missing | Rewritten. Added: four-difference table, alarm burden, ACI mechanism (γ chosen in 29/30 runs), H3 rare-only runs, OPSSAT corrections, uncontrolled subset, event-class decomposition, Table 15 (selection effect by channel set and layer) | QA items (a)2, 6–9; (b)3–9, 13, 15–17 | Every number is an evidence key; new derived keys documented in `build_evidence.py` | Table 15 cells other than M1-full-BH are single deterministic runs without intervals |
| `thesis/src/08_discussion.md` | Overstated mechanism, multiplicity credit, RQ4 contradiction, literature comparison without label-use context | Rewritten as evidence-based analysis | QA items (a)8; (b)2, 4–8 | Keys only | Explanations listed as untested |
| `thesis/src/09_limitations.md`, `10_conclusion.md` | Same overstatements; deviations "procedural" only | Corrected; factorial H6 added to future work | as above | — | — |
| `thesis/src/11_appendices.md` | Appendix B listed 5 procedural deviations; Appendix C lacked the tie-break, check()/hydrostatic, A0 grid; #2, #6, #7 wrong | Deviation 6 and the not-run anomaly arm added; defects 17–18 added; #2, #3, #6, #7 corrected | QA items (a)2, 4, 10; (b)14, 19–20 | Code read | — |
| `research_design/preregistration/DEVIATIONS.md` | A0 grid inconsistency and not-run anomaly arm not logged | One row appended (no earlier row edited) | Pre-registration §3 | — | — |
| `thesis/refs.bib` | DMLR entry had arXiv authors/title; missing references | DMLR entry corrected; added benjamini2001control, simes1986, law2018cornernet, tatbul2018precision, johnson2001sbm, loshchilov2019adamw, smith2019superconvergence; Sehili updated to the LNCS version | QA (c) | Crossref / arXiv records, logged | wu2021flawed left at the 2021 early-access record (Crossref shows no 2023 issue data) |
| `thesis/tools/build_evidence.py` | Derived quantities typed by hand in the text | Added derived keys: `seleff.*`, `evcls.m1.gain_a2_a1.*`, `shift.m1.*.spearman_f05`, `conf.reimpl_gap_max`, `conf.p_floor*`, `verify.{i}.reference`, `verify.n_*`, `layers.n_threshold_ge_bh`, `pre.all.aci.gamma*_runs`, `pre.m*.*.alarm_runs_per_year`, `data.esa.m*.test.blocks30` | No hand-typed numbers | Rebuilt: 15,023 entries | The verification-check classification is a judgement (recorded as `derived`) |
| `thesis/tools/render.py`, `make_reference_docx.py` | Equal table column widths; rows split across pages; justified lists of figures/tables | Content-based column widths, `cantSplit` rows, "List Entry" style | Layout | PDF pages inspected | — |
| `thesis/tools/make_figures.py` | A0 labelled "9-month selection" (wrong for Mission 2); forest X1 used the re-implemented difference | Relabelled "calibration-window"; official difference | QA (a)5, (c) | Figures re-rendered and inspected | — |

## 2026-10-25, pass 2: additional result figures (user request "add more result images")

All new figures are drawn by `thesis/tools/make_figures.py` from `thesis/evidence/evidence.json` (the monitor grid from `thesis/evidence/original_pipeline_runs.json`). No new experiment was run.

| Figure file | Content | Source keys | Placed in |
|---|---|---|---|
| `esa_selection_effect.png` | A1 − calibration-window selection, 2 missions × 2 channel sets × 2 layers; pre-registered cell ringed | `seleff.*` | Results, after Table 15 |
| `esa_alarm_burden.png` | Alarm runs per year split into event-touching and false; events detected | `pre.m*.A*.conformal_bh.{alarm_runs_per_year, fa_yr.mean, r.mean}`, event counts | Results, "Alarm burden" |
| `esa_drift_vs_score.png` | Shift ratio against test F0.5 per supervised seed, with Spearman ρ | `shift.m1.*`, `pre.seed.*` | Results, diagnostics |
| `esa_primary_vs_anomaly.png` | Primary against anomaly-only F0.5 per arm | `pre.m*.A*.conformal_bh.{f05, f05_anom}.mean` | Results, event classes |
| `esa_layers.png` | Threshold / conformal-BH / rolling per arm (Table 14) | `pre.m*.*.{threshold, conformal_bh, conformal_bh_rolling}.f05.mean` | Results, alarm layers |
| `esa_cv_vs_test.png` | Training-half CV against test F0.5 | `sel.m*.*.conformal_bh.cv.mean`, `pre.m*.*` | Results, CV versus test |
| `opssat_curves.png` | Label efficiency; robustness to noise and dropout | `ops.official.*.f{10,25,50,100}.auc_pr.mean`, `ops.robust.*` | Results, OPSSAT-AD |
| `reentry_subsets.png` | AUC-PR on all objects and on the uncontrolled subset | `lm.*.{all, uncontrolled}.h{30,68}.auc_pr` | Results, landmark |
| `orig_monitor_grid.png` | Detection delay per subsystem × fault (SIMULATED) | `original_pipeline_runs.json` → `monitor_grid` | Results, original pipeline |

Build after pass 2: `thesis/build/COSMOS_Thesis.pdf`, 187 pages, 24 figures (15 before this pass, 9 new), 20 tables, 996 evidence values, no unresolved placeholder.

## 2026-10-25, pass 3: journal manuscript and the corrections found while writing it

**Journal manuscript.** A new journal manuscript was written in `paper/journal_v2/`. It is 15 pages, has 5 figures and 4 tables, and resolves 221 evidence values. It does not replace `paper/manuscript.tex`, which is unchanged. An independent fact-check of the draft against the result files and code found the issues below. Every one was checked in the code before it was fixed.

| File | Original issue | Change | Verification |
|---|---|---|---|
| paper; thesis 07/08/00/01 | ACI said to make persistence "impossible" and to need 79 recovery steps | The global alarm is an OR over channels, so persistence is rare, not impossible: 20 of 30 event-logic runs raised alarms. Recovery takes 80 steps, because α_t ≤ 0 gives an infinite threshold | `oof_benchmark.py` l.162–180 and l.256–260; key `aci.recovery_steps.g0.05` = 80 |
| paper abstract; thesis 00 | "Best published detectors 0.061/0.079" | Changed to "the benchmark's published baselines"; Brancato et al. and Alcarria et al. report higher values under other protocols | literature keys |
| paper; thesis 05/07 | Mission 2 described as "without inference" | It uses the same test without multiplicity correction (PREREG §5); unadjusted p-values now reported | `conf.m2.*.p_boot` |
| paper; thesis 07/08 | "9 against 109 events": the 109 includes 4 communication gaps | Changed to 105 target events | `data.esa.m1.events_all.train.targets` |
| paper; thesis 00/01/07/08/09/10/11 | "Four differences" between A0 and A1 | Five, including the residual scale. Two of the five change the test scores | `forecasting.py` l.391–399 against `oof.py` l.112–118 |
| paper | Timeline figure said "code frozen before any new arm was trained" | A2/A3 seed-0 training and the A1 selection predate the freeze. The text now says "scored on test" and lists what existed at the freeze | PREREG §0; file times |
| paper | "Every outcome is reported" | Narrowed to the three hypotheses. Added: secondary metrics are in the supplement; the anomaly arm was not run; the §7 rule on trees; the M1 baselines; a one-line H1–H3 summary | PREREG §3, §7 |
| paper | Bootstrap called moving-block | It uses fixed non-overlapping blocks (Carlstein 1986). Discrepancy stated; `carlstein1986` added after a Crossref check | `event_bootstrap.py` l.48–57 |
| paper | "Selecting … raised"; "the selection data is the most plausible contributor" | Reframed as a protocol-level gain; the speculation removed | — |
| paper | Missing settings: TCN, XGBoost, 500k AR rows, seeds, percentile CIs, bootstrap seed, untuned models | Added | `cfg.*` keys; `oof.py` l.37; `event_model.py` l.307 |
| thesis 07/08; paper | Rolling layer "raised only the TCN" | It also raised the lightweight Mission 1 arm (0.025 → 0.130) | `pre.m1.A1lw.*` |
| `research_design/preregistration/DEVIATIONS.md` | Residual-scale difference and bootstrap scheme not logged | One row appended; no earlier row edited | — |
| `thesis/tools/make_diagrams.py` | Test half labelled "(sealed)" | Relabelled "already used in the exploratory phase" | — |
| `thesis/refs.bib` | Two duplicate entries (AdamW, one-cycle) | Duplicates removed | — |
| `thesis/tools/build_evidence.py` | — | New keys: `data.esa.m*.cal.*`, `*.events_detected`, `data.esa.m*.events_target.test.total`, `data.esa.m1.events_all.train.targets`, `evcls.m1.gain_a2_a1.recall_based`; corrected `aci.recovery_steps` formula | rebuilt: 15,039 entries |

**Venue list.** `paper/journal_v2/VENUES.md` was written from OpenAlex data fetched on 2026-10-25, saved in `research_design/references/openalex_venues_2026-10-25.json`. The Scopus status and SJR quartiles of the listed venues are not verified.

## 2026-10-25, pass 4: validation analyses (P1–P8) and the Elsevier-format manuscript

**Validation plan.** `research_design/preregistration/POSTHOC_PLAN_V2.md` fixed eight analyses, their code checks and interpretation rules, and was frozen (`posthoc_v2_freeze.json`, SHA-256 217648e6805f…, 12:14:37) before any of them ran. Scripts are in `scripts/posthoc_v2/`; every script refuses to run if the plan hash has changed and records its own SHA-256 and the plan's in its output. Results are in `outputs/benchmarks/telemetry_esa_adb/posthoc_v2/`. All of them use test halves that had been scored before and are reported as post hoc. A row was appended to `DEVIATIONS.md`.

| Analysis | Code check | Result |
|---|---|---|
| P1 decomposition of H6 (6 settings, 10 cells each) | first and last cells reproduce A0 and A1 to < 1e-9 in all 6 settings; A0 scores recomputed to ≤ 4.8e-6 | M1 BH: selection data +0.301 / +0.366 (material); pool +0.043; fit data +0.010; residual scale 0; grid order-dependent (−0.066 on the reverse path). M1 threshold: pool +0.145 and selection +0.197 (both material). M1 lightweight and M2 reversals: the same factors, with negative sign |
| P2 forward chaining | — | M1: H won 2 of 4 folds (mean +0.240; driven by 2005, where W raised 1,173 FA/yr): not replicated. M2: 3 of 3, mean +0.017 < 0.05: not replicated |
| P3 exact bootstrap | 302 arm–metric pairs, max difference 2.1e-16 | verdicts unchanged; H6 [+0.207, +0.491] |
| P4 event-tuned ACI (7 arms) | — | A1 M1 0.472 (above BH 0.428); γ > 0 below the threshold layer by ≥ 0.05 on 5 of 7 arms: the claim "ACI unusable at the event level" is not permitted |
| P5 registered anomaly-score arm | — | M1 anomaly-only F0.5: AR(8) 0.336, trees 0.087, TCN 0.097 |
| P6 seed extension (seeds 5–9 of A2, A3) | selection reused only when the saved source matches; rows of runs lost with `/tmp` rebuilt from saved alarms reproduce the logged scores | 10 tree seeds 0.488 ± 0.195; TCN 0.176 ± 0.093 over 9 seeds (seed 9 lost with `/tmp` before scoring and rerun under the P6 stopping rule). TCN − AR(8) −0.252 [−0.402, −0.043]; TCN − trees −0.313 [−0.457, −0.112]; trees − AR(8) +0.061 [−0.138, +0.268] (exact bootstrap, unadjusted) |
| P7 operational metrics | detections match recall × events for every arm | A1 M1: 16 of 91 events, median delay 0.33 h, 0.15 FA per 1000 nominal h |
| P8 provenance | frozen selection re-run | identical choice and grid values on both missions |

**Manuscript (`paper/journal_v2/`), restructured to the Elsevier guide.** Abstract ≤ 250 words with abbreviations defined and no citations; 6 keywords; Inspec codes (checked on archived IET pages only); numbered sections with a Materials and methods section and a table of the validation analyses; equations numbered with right-aligned numbers and Appendix equations (A.1), (A.2), (B.1); table notes below tables; captions as bold title plus description; figures as separate vector files (`artwork/Figure_1–3.pdf`) and supplementary figures S1–S3; separate highlights file and graphical abstract (TIFF/PDF, 1535 × 708 px at 13 cm); author–year references from a checked bibliography (`refs_journal.bib`: published versions, DOIs, LTWA abbreviations, `[dataset]` and `[software]` entries) with a patched Elsevier Harvard style; data statement under Option C; CRediT template; generative-AI declaration that names the tool (to be reviewed by the authors). The literature table, the generality figure and the layer and drift figures moved to the supplementary material, whose table and figure numbers are resolved at build time.

**Corrections made while writing.** A draft sentence said the forward-chaining mean came from two high-false-alarm folds; in 2004 the history protocol detected nothing, so the text now attributes the mean to 2005 alone. A draft practice rule claimed a "sixfold" difference in detected events between A0 and A1; the counts are 42 and 16, and the sentence now gives them.

**Thesis.** New section "ESA-ADB: post-hoc validation analyses (P1–P8)" in `07_results.md` with two figures (`esa_posthoc_decomposition.png`, `esa_posthoc_forward.png`); passages in 05, 07, 08, 09 and 11 that called the decomposition, forward chaining, the fair ACI comparison and the anomaly arm future work were updated.

**Formatting fixes found in the final check (2026-10-27).** Six inline formulas in the thesis (and one in the manuscript abstract) ended or began with a bare operator, such as `$K =$ 5`, which LibreOffice renders as a red "¿"; the operators were moved outside the math. The re-entry figure `fig:landmark-subsets` was not registered as a figure because a list item followed it without a blank line; a blank line was added. Thesis: 197 pages, no unresolved reference or placeholder.
