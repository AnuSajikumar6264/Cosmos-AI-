\newpage

# Appendix A: Reproducibility {#sec:appendix-repro .unnumbered}

The repository is `https://github.com/AnuSajikumar6264/Cosmos-AI-`, at commit `2525367`. Large data, score arrays and checkpoints are excluded from git (see `.gitignore`) and must be regenerated or obtained separately.

**Environment.** Python 3.12, plus PyTorch with a back end for the available hardware (here Intel XPU), then `pip install -r requirements.txt -r requirements-ai.txt`.

**Data.**

```bash
python -m cosmos.ai.data.download opssat esa_adb spotgeo celestrak   # checksummed, with manifests
python -m cosmos.ai.data.preprocess_telemetry                        # official ESA-ADB grids
python -m cosmos.ai.data.validate                                    # data/DATASET_REPORT.{json,md}
```

**Verification and tests.**

```bash
python tests/test_verify.py      # physics verification checks (prints PASS/FAIL)
python tests/test_ai.py          # COSMOS-AI checks (also under pytest)
```

**ESA-ADB, exploratory study.**

```bash
python -m cosmos.ai.telemetry.esa_benchmark --mission 2 --baseline global_std5
python -m cosmos.ai.telemetry.esa_benchmark --mission 1 --encoder ar8
python -m cosmos.ai.telemetry.esa_benchmark --mission 1 --encoder tcn --seed 0
```

**ESA-ADB, pre-registered study.**

```bash
python -m cosmos.ai.telemetry.oof --mission 1
python -m cosmos.ai.telemetry.event_model --mission 1 --seed 0
python -m cosmos.ai.telemetry.event_gbt --mission 1 --seed 0
python scripts/prereg_analysis.py select
python scripts/prereg_analysis.py test        # refuses unless the freeze hashes match
python scripts/prereg_analysis.py stats       # -> oof_protocol/CONFIRMATORY.{json,md}
```

**Supporting studies.**

```bash
python -m cosmos.ai.telemetry.opssat --config configs/telemetry/opssat.yaml
python -m cosmos.ai.vision.train_spotgeo --model unet --inputs both --seed 0
python -m cosmos.ai.survival.landmark
python -m cosmos.ai.surrogate train && python -m cosmos.ai.surrogate evaluate
python -m cosmos.ai.telemetry.learned_monitor train && python -m cosmos.ai.telemetry.learned_monitor heldout
```

**This thesis.**

```bash
python thesis/tools/run_original_pipeline.py   # regenerates original-pipeline evidence
python thesis/tools/build_evidence.py          # thesis/evidence/evidence.json + evidence_map.md
python thesis/tools/make_figures.py            # result figures from the evidence only
python thesis/tools/make_diagrams.py           # architecture diagrams (graphviz)
python thesis/tools/render.py                  # thesis/build/COSMOS_Thesis.docx
```

`render.py` aborts if any number in the text has no evidence entry.

# Appendix B: Pre-registration and deviations {#sec:appendix-prereg .unnumbered}

The pre-registration is in `research_design/preregistration/PREREG_TELEMETRY_V1.md`. It states:

- what was known before it was written;
- the question;
- the data and metrics;
- the arms;
- the selection rule;
- hypotheses H4–H6, with test, family, minimum effect size and verdict rules;
- the seeds and stopping rule;
- what would change the claims;
- the known threats;
- a deviation log.

It was frozen with {{prereg.frozen_files}} files at {{prereg.freeze_written}} as `{{prereg.freeze_id}}`, and reissued as `{{prereg.refreeze_id}}` once the deviation log was filled in. The confirmatory results file records the freeze under which they were computed (`{{conf.freeze_id}}`). {{prereg.n_deviations}} deviations are logged in `research_design/preregistration/DEVIATIONS.md`. The first five are procedural:

1. The test pass was run by a parallel driver that claims each arm atomically, instead of a sequential loop. The same frozen function scored each arm once.
2. The sequential test process was stopped after it had scored arm A1 on Mission 1.
3. Training-half selection for the late-finishing arms (Mission 2 TCN seeds 3–4 and the six ablation arms) ran after other arms had been scored. Four workers had crashed on the missing selection files before scoring anything, and their locks were removed.
4. The arm-A0 alarms for Mission 2 were regenerated with the frozen function, after the first statistics run failed on the missing file. The function is deterministic and the bootstrap seed fixed.
5. The deviation log was filled in and the freeze reissued.

None of these changed a computation or a selection.

The sixth was found during the review for this thesis. It is an inconsistency in the design, not a change to it. Section 3 of the pre-registration says that all arms use the pre-registered grids. Arm A0, however, is by definition the exploratory arm, and it carries its exploratory selection on the grid $q \in$ {{cfg.explo.levels.conformal_bh}} × $k \in$ {{cfg.explo.ks}}. A0 and A1 therefore differ in at least four respects (fitting period, calibration pool, selection data and grid), and H6 compares two complete protocols. A later code review found a fifth: the residual scale of the AR(8) score (logged as a further row of the deviation log). The same review found that the secondary arm with anomaly-only scores, listed in section 3, was neither run nor reported as not run. It is reported here as not run.

# Appendix C: Defect and discrepancy register {#sec:defects .unnumbered}

The register was compiled from two independent read-only code audits, from re-running the pipeline, and from comparing the documentation with the code. Defects are listed, not fixed, because several affect results that have already been reported. Fixing them would require new, separately reported runs.

: Defects in the code. "Effect" states which reported results are affected. {#tbl:defects}

| # | Location | Defect | Effect |
|---|---|---|---|
| 1 | `health/telemetry.py:125` | `stuck` fault has noise multiplier 0.6, so a negative noise scale; the call raises `ValueError` | `stuck` excluded from all monitor benchmarks |
| 2 | `health/monitor.py:129–130`, `learned_monitor.py:188–189` | healthy window capped at the true fault-onset index, i.e. ground truth. With the default onset (55 % of the record) and window (40 %) the cap never binds | none for the reported runs; a latent leak if the onset is moved earlier |
| 3 | `health/hazard.py:71`, `monitor.py:46–48` | driver is the unweighted argmax of degradation; undetected faults are attributed to the subsystem with the largest residual degradation; ties go to the first key, so the all-nominal case becomes a Power System explosion at {{orig.scen.all_nominal.breakup_altitude_km}} km | coupling results for undetected faults ([@tbl:failure-modes]) and the nominal scenario |
| 4 | `health/hazard.py`, `pipeline.py` | $P_{\text{frag}}$ and $\sigma_{\text{state}}$ computed but unused; a break-up is always propagated | risk is not probability-weighted |
| 5 | `reentry/montecarlo.py:115` | one flight-path-angle draw shared by all samples; configured ballistic-coefficient and wind dispersions unused | footprint spread under-dispersed |
| 6 | `reentry/risk.py:209–235` | per-fragment $E_c$ compared with the per-event threshold; the event-level debris area uses $m/B = C_D A$ instead of $A$, so $A_d$ is $C_D$ = 1.35 times too large; $A_c = (\sqrt{A_d}+0.6)^2$ grows by less | risk bands and corridor values (event level modestly over-estimated) |
| 7 | `reentry/breakup.py:28, 84` | simplified SBM: mass-scaled fragment count, non-standard area-to-mass coefficients, single area law. The $\Delta v$ law uses the collision slope 0.9 in $\log_{10}(A/m)$ for every type, with a type factor (0.2, 0.9, 0.06) in place of the SBM's type-specific slope and offset (0.2χ + 1.85 for explosions, 0.9χ + 2.9 for collisions [@johnson2001sbm]) | explosion fragments receive far smaller $\Delta v$ than SBM fragments; break-up clouds are not standard SBM clouds |
| 8 | `reentry/atmosphere.py` | static US76, constant above 1000 km, no solar or geomagnetic activity; the comment "2 km grid" is wrong | lifetimes and footprints |
| 9 | `debris/lifetime.py:158, 172` | fragments-per-collision formula is not the SBM law; possible double counting of the collision rate | collision criticality |
| 10 | `debris/yolo.py` | `map50` uses centre matching, class-agnostically, without interpolation | not a COCO mAP; COSMOS-AI re-scores with `vision/metrics.py` |
| 11 | `debris/scene.py:184–186` | augmentation summary claims a contrast jitter that is not implemented | documentation only |
| 12 | `survival/landmark.py:301` | XGBoost-AFT predicts with all trees, not the early-stopped iteration | AFT row of [@tbl:landmark] |
| 13 | `data/preprocess_telemetry.py` | forward difference of monotonic channels gives a one-sample look-ahead, inherited from the official scripts | negligible; noted for causality claims |
| 14 | `realdata` tests | safe-unpickler test fails under pandas 3 | one pre-existing test error |
| 15 | landmark, learned monitor, OPSSAT | validation data reused for several decisions | mild optimism of the selection |
| 16 | `fusion.decide` | noisy-OR treats evidence items as independent causes; not wired into the pipeline | not used for any result |
| 17 | `tests/test_verify.py:31–37, 60` | `check()` records PASS or FAIL but never raises, so under pytest every check passes; the hydrostatic check compares $dp/dz$ with $-\rho g_0$ rather than $-\rho g(z)$, so its {{verify.3.measured}} is the gravity offset at 30 km, and it would fail its own 1 % limit above about 33 km | verification evidence: {{verify.n_internal}} of the checks are self-consistency checks or hold by construction ([@tbl:verify]) |
| 18 | pre-registered study, arm A0 | A0 carries the exploratory selection grid, not the pre-registered grid (Appendix B) | H6 compares two complete protocols |

: Discrepancies between documentation and code or evidence. {#tbl:discrepancies}

| Source | Statement | Current evidence |
|---|---|---|
| Review-1 presentation | APSGTT–LoRA, diffusion forecasting, federated learning, `coupling.py`, dashboard | not present in the code |
| Review-1 presentation | conformal zone 0.9927 vs Gaussian 0.9750 in 17 % less area | not reproduced ([@sec:results-orig]) |
| Review-1 presentation | lifetime 1.03× on 1,806 fragments | not reproduced (ratio {{orig.lifetime_val.cosmos_1408.ratio}} for Cosmos 1408) |
| Review-1 presentation / `debris/yolo.py` | YOLOv8n/v11n/v12n mAP@50–95 0.976 / 0.976 / 0.969 | numbers of an external reference pipeline (hard-coded in `reference_benchmark()`), not produced by COSMOS |
| `README.md` | step-size convergence 0.13 m | current test log: {{verify.11.measured}} |
| `data/DATASET_REPORT.json` | ESA-ADB licence "CC BY 3.0 IGO per the DMLR paper" | the DMLR paper states no licence; the statement is in the arXiv version (2406.17826v2, p. 35) |
| `outputs/benchmarks/SUMMARY.md` | Chronos-2 and LoRA fine-tuning implemented | only Chronos-2 zero-shot loading exists |
| `cosmos/__init__.py` docstring | catalogued objects are the objects that later re-enter | debris is not connected to `run_pipeline` |
| `notebooks/COSMOS_Demo.ipynb` | saved outputs | produced by package version 1.0.0; superseded by `thesis/tools/run_original_pipeline.py` |
| several docstrings (CNN kernel, landmark τ wording, learned-monitor "BH", `report.py` output name) | as listed in the audit | minor; code is authoritative |

# Appendix D: Contribution statement, evidence gaps and journal readiness {#sec:appendix-q1 .unnumbered}

## D.1 Research contribution and gap statement {.unnumbered}

**Gap.** ESA-ADB makes operator-grade evaluation on real telemetry possible. Full-channel results under its official protocol remain low for the published baselines, and the benchmark authors stress false detections and the importance of thresholding and post-processing. Point-level conformal calibration has been shown not to fix it [@sahin2026tccad]. No pre-registered study has compared unsupervised, tree-based and deep supervised event scorers under the official split at full channel scale, with the alarm layer selected on all training labels.

**Contribution.** This thesis contributes such a study. Its outcomes are:

- the whole-training-half selection protocol is supported on Mission 1, under both alarm layers;
- the deep supervised scorer shows no benefit over either simpler scorer;
- the selection effect is conditional: it reverses on Mission 2 under BH and on the lightweight Mission 1 subset;
- aggregation to events with a strict, metric-selected operating point succeeds where point-level adaptive calibration fails, for a mechanical reason;
- a diagnostic, exploratory only, shows that supervised scores shift between training and test, but the shift does not track test performance.

The contribution is empirical and methodological. It is not a new architecture, and the main deep-learning result is negative.

## D.2 Unresolved evidence gaps and missing experiments {.unnumbered}

1. **Brancato et al.'s protocol.** Their Mission 1 numbers ({{lit.brancato.smed_full:.3f}}; out-of-limit reference {{lit.brancato.ool_full:.3f}}) cannot be reconciled with this study without their full text and code. *Not verified — the full text of [@brancato2027ast] is required.*
2. **Fresh data for a confirmatory test.** No confirmatory test has been run on data never seen by the project. All test halves were seen in the exploratory phase.
3. **Mechanism of the TCN's failure.** The post-hoc shift diagnostic (Mission 1, every fourth channel) does not order the runs by score, so the cause is open. *A pre-registered test is required.*
4. **Source of the selection effect.** A0 and A1 differ in five respects. *A factorial experiment is required to attribute the H6 gain.*
5. **Re-entry locations.** No validation against recorded re-entries exists. *Recorded re-entry events with recovery coordinates are required* (ESA DISCOS or Space-Track access).
6. **The coupling weights.** These are hand-set and have not been checked against fragmentation records. *DISCOS fragmentation data are required.*
7. **Statistical power.** Supervised-arm seed counts (5; 3 for the ablations) are low relative to the seed variance.
8. **Two literature items not read.** Fejjari et al. (Acta Astronautica) and Dai et al. (Advances in Space Research) were identified in the research design, but their full texts were not accessible here. *The author must read them before submission.*
9. **Unverified references.** RevIN and a telemetry survey by Zeng et al. could not be verified and are not cited.

## D.3 Critical assessment from a reviewer's perspective {.unnumbered}

| Criterion | Assessment |
|---|---|
| Originality and significance | Moderate. The pre-registered, full-channel comparison and the event-versus-point calibration result are new. The calibration-and-triage space overlaps closely with TC-CAD [@sahin2026tccad], and the deep-learning result is negative. |
| Clarity of the gap | Good, provided the paper is framed around alarm selection and score stability rather than around a new model. |
| Technical rigour | High for the telemetry study: official metric code, a reproduced baseline, frozen hashes, gated test scoring, hierarchical bootstrap and Holm correction. The five-way confound of H6 and the point-tuned ACI comparator were addressed post hoc, by a decomposition and an event-tuned comparison ([@sec:results-posthoc]), on test halves already seen. Lower for the physics chain, which has documented defects. |
| Experimental design and data | Real, expert-annotated data; but one benchmark, with one informative mission and two usable missions in total. |
| Baselines | Strong: label-free rules, the official GlobalSTD, trees on the same features, and a published comparator re-implemented. The published baselines did not use training labels, so comparisons with them need that context. |
| Statistical reliability | Adequate for H6. Inconclusive for the comparison of trees with AR(8), because of the seed variance. |
| Reproducibility | High: code public, evidence file, one-command rebuild. Large artefacts are not yet archived with a DOI. |
| Generalisation | Limited. The selection effect reverses on Mission 2 under BH and on the lightweight Mission 1 subset. No test on unseen data. |
| Limitations stated | Yes, extensively. |

**Overall.** In its present form, the telemetry study is publishable as a careful empirical paper. Acceptance by a Q1 venue is plausible but not assured, and it depends on the framing and on items 1–4 above. The physics chain and the supporting studies are thesis material rather than journal contributions in their present state.

## D.4 Prioritised action plan {.unnumbered}

1. **Read and position against Brancato et al.** If possible, reproduce their protocol through this code (gap 1).
2. **Run a factorial H6** that crosses scores, calibration pool and selection data on one grid, to attribute the selection effect (gap 4). Done post hoc as a one-factor-at-a-time decomposition (P1): the selection data carry the effect.
3. **Pre-register and run a drift-aware supervised arm.** Use recalibration selected for that purpose or causal retraining, more seeds, and data not yet used, such as a later ESA release or another mission (gaps 2, 3 and 7).
4. **Add a forward-chaining control** to quantify the optimism of cross-fitting. Done post hoc (P2): the gain did not replicate by the pre-set rule.
5. **Write the manuscript** around one claim: selection and stability of the alarm layer matter more than model capacity. State its conditions (Mission 1 at full scale; reversals elsewhere), report the alarm burden, and report the refuted deep-learning hypotheses as a pre-registered negative result.
6. **Archive the code and score arrays on Zenodo** with a DOI, and add a licence and `CITATION.cff`.
7. **Leave the physics chain and supporting studies out of the paper,** or move them to a separate engineering paper after the defects in [@tbl:defects] are fixed.

## D.5 Converting the thesis into a journal manuscript {.unnumbered}

| Manuscript section | Thesis sources | Action |
|---|---|---|
| Introduction | [@sec:intro], [@sec:gap] | condense to the telemetry problem |
| Related work | [@sec:literature] (ESA-ADB, calibration) | keep [@tbl:lit-esa]; add the two unread papers |
| Method | [@sec:oof], [@sec:model], [@sec:alarm-layer], [@sec:prereg] | as written; add the freeze procedure |
| Results | [@sec:results-baselines], [@sec:results-prereg], [@sec:results-ablations] | [@tbl:baselines], [@tbl:prereg-arms], [@tbl:confirmatory], [@tbl:layers] and figures |
| Analysis | exploratory diagnostics | label as exploratory |
| Discussion and limitations | [@sec:discussion], [@sec:threats] | keep the threats section complete |

The draft in `paper/manuscript.tex` already follows this outline. Its tables are generated by `scripts/build_paper_tables.py`.

**Venues.** The candidate venues, with indicative citation statistics, are listed in `research_design/VENUES.md`. The SJR quartiles there could not be verified from this machine and must be checked on the day of submission. On scope, Engineering Applications of Artificial Intelligence, Acta Astronautica and Reliability Engineering & System Safety fit an empirical study of alarm selection on spacecraft telemetry. The choice should follow the journal's current aims and scope, not its impact factor.
