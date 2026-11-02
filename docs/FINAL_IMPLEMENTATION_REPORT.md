# COSMOS-AI rebuild: final implementation report

Work branch: `q1-reentry-rebuild`. It was checkpointed in commit 2427cf8 before any change;
`main` is untouched. This report describes:
- what was built;
- how to run it;
- what was checked, and what was found and fixed;
- the known limits.

The measured results are in `docs/RESULTS_AND_ANALYSIS.md` and the manuscript in
`paper/journal_v3_twin/`.

## 1. Objective and scope

The goal is one digital-twin framework for three functions:
- satellite health monitoring;
- orbital debris analysis;
- atmospheric re-entry risk prediction.

Every module is validated on real public data, and the couplings between the modules are
tested on real data wherever a test exists. No public dataset follows one object through
telemetry, conjunctions and re-entry. The twin is therefore a set of separately validated
modules joined by explicit couplings, not a joint model trained on unpaired data
(`docs/ARCHITECTURE_AND_NOVELTY.md`).

## 2. What was built

| Component | Code | Data | Entry point |
|---|---|---|---|
| Re-entry cohort (leakage-controlled) | `cosmos/ai/lifetime/catalogue.py` | GCAT 2026-10-27, CelesTrak SATCAT and SW-All | `python -m cosmos.ai.lifetime.catalogue` |
| Physics expert (King-Hele and exact orbit averages, NRLMSIS 2.1, Monte-Carlo A/m, history posterior) | `cosmos/ai/lifetime/physics.py` | NRLMSIS 2.1 table (pymsis) | used by all of the below |
| Learned experts. Candidates published 2022–2026: TabM, TabICLv2, Kumo-Tabular. Reference baselines: XGBoost, LightGBM, CatBoost, quantile MLP | `cosmos/ai/lifetime/models.py` (`RECENT`, `LEARNERS`) | cohort | |
| Reliability-gated fusion, Mondrian CQR | `cosmos/ai/lifetime/fusion.py` | | |
| Re-entry study (two temporal origins, ablations, bootstrap) | `cosmos/ai/lifetime/run.py` | cohort | `python -m cosmos.ai.lifetime.run --origin {2016,2011}` |
| Censoring-aware AFT baseline | `cosmos/ai/lifetime/survival.py` | cohort | `python -m cosmos.ai.lifetime.survival --origin O` |
| Oracle-solar diagnostic, operational ±20/30 % windows | `cosmos/ai/lifetime/diagnostics.py` | | `python -m cosmos.ai.lifetime.diagnostics --origin O` |
| Gate applied to every learner (sensitivity) | `cosmos/ai/lifetime/gate_all.py` | | `python -m cosmos.ai.lifetime.gate_all --origin O` |
| RCS leak evidence | `cosmos/ai/lifetime/leak_check.py` | | `python -m cosmos.ai.lifetime.leak_check` |
| Collision-risk module (ESA Kelvins CAC) | `cosmos/ai/conjunction/` | Kelvins CAC (Zenodo 4463683) | `python -m cosmos.ai.conjunction.run` |
| Telemetry benchmark (NASA SMAP/MSL): Chronos-2 zero-shot, classic baselines, published telemanom scores | `cosmos/ai/telemetry/smap_msl.py` | SMAP/MSL (Kaggle copy of the telemanom release) | `python -m cosmos.ai.telemetry.smap_msl` |
| Post-hoc diagnostics: per-class breakdown, paired ablation CIs | `cosmos/ai/lifetime/diagnostics.py` (`class_breakdown`, `ablation_effects`) | saved test predictions | see section 3 |
| ESA-ADB and OPSSAT-AD health studies | `cosmos/ai/telemetry/` (pre-registered study, unchanged) | ESA-ADB, OPSSAT-AD | see `README.md` |
| Twin state and records | `cosmos/ai/twin/state.py` | | |
| Couplings C2, C3, C4 | `cosmos/ai/twin/couplings.py` | lifetime outputs, SATCAT snapshots 2026-08-16 and 2026-10-27 | `python -m cosmos.ai.twin.couplings {c2,c3,c4}` |
| Twin engine (dated, hashed forecast) | `cosmos/ai/twin/engine.py` | SATCAT snapshot of the as-of date | `python -m cosmos.ai.twin.engine --as-of 2026-10-27` |
| As-of-date test T1 of the twin's re-entry forecasts | `cosmos/ai/twin/prospective.py` | snapshots 2026-08-16 and 2026-10-27 | `python -m cosmos.ai.twin.prospective` |
| Data fetchers and manifests | `cosmos/ai/data/download.py`, `scripts/build_dataset_manifest.py` | | see `docs/DATASET_AUDIT.md` |
| Manuscript build | `paper/journal_v3_twin/tools/` | result and data files only | `build_numbers.py`, `make_architecture.py`, `make_figures.py`, `make_data_figures.py`, `make_supplementary.py`, `render.py` |

## 3. Order of execution

```
python -m cosmos.ai.lifetime.catalogue                      # cohort (seconds)
python -m cosmos.ai.lifetime.run --origin 2016               # hours (TabICL on CPU)
python -m cosmos.ai.lifetime.run --origin 2011
python -m cosmos.ai.lifetime.survival --origin 2016          # and 2011
python -m cosmos.ai.lifetime.diagnostics --origin 2016       # and 2011; then diagnostics.interval_rules
python -m cosmos.ai.lifetime.gate_all --origin 2016          # and 2011
python -m cosmos.ai.lifetime.leak_check
python -m cosmos.ai.conjunction.run
python -m cosmos.ai.telemetry.smap_msl
python -m cosmos.ai.twin.couplings c2 ; ... c3 ; ... c4
python -m cosmos.ai.twin.prospective                        # T1
python -m cosmos.ai.twin.engine --as-of 2026-10-27           # dated forecast
python paper/journal_v3_twin/tools/build_numbers.py
python paper/journal_v3_twin/tools/make_architecture.py      # Figure 4
python paper/journal_v3_twin/tools/make_figures.py           # result and benchmark figures, S1-S3
PYTHONPATH=. python paper/journal_v3_twin/tools/make_data_figures.py   # dataset and model-output figures
python paper/journal_v3_twin/tools/make_supplementary.py
python paper/journal_v3_twin/tools/render.py                 # Word and LaTeX
```

Environment variables:
- `HF_HOME` for the TabICL, Kumo-Tabular and Chronos-2 weights;
- `COSMOS_DEVICE` to pin TabM, Kumo-Tabular and Chronos-2 to one Intel XPU;
- `COSMOS_KUMO_CONTEXT` for the Kumo-Tabular context cap (default 16,000 rows);
- `COSMOS_TORCH_THREADS` for the MLP threads;
- `COSMOS_PHYSICS_JOBS` for the processes of the history posterior.

## 4. Quality controls

- **Checkpoint and backups.** The repository checkpoint is commit 2427cf8. Pre-fix module sources
  are in `/data/anupa/COSMOS_work/.scratch/pre_fix_2026-10-27/`. Every superseded result is
  kept under `_superseded_pre_review_2026-10-27/` with a README.
- **Leakage controls.**
  - "As of origin" censoring; splits by time and by launch family.
  - Space weather only from days completed before the prediction time.
  - No covariate whose availability depends on the outcome: radar cross-section excluded, with
    evidence in `leak_check.json`.
  - Learner caches keyed by a fingerprint of the data, features and settings.
- **Test sets.** Read only for scoring, but scored more than once during development: before
  the first review, with the first learner set (run 1), in smoke tests of the new learners, and
  for the reported run. The T1/C3 window had been used by an earlier landmark benchmark. All of
  this is listed in `docs/REVIEW_AND_FIXES.md` and in the manuscript's deviations paragraph.
- **Adversarial code review.** Specialist reviewers, then two skeptics per finding: 34 confirmed
  findings and 3 disputed. All were fixed and documented (`docs/REVIEW_AND_FIXES.md`).
  - The conjunction and telemetry fixes were themselves reviewed adversarially, with repair
    rounds.
  - After the fixes, the corrected re-entry module was found to have a domain shift on current
    orbits. This led to the history-conditioned state estimation and the as-of-date test T1.
- **Number traceability.** Every number in the manuscript is a `{{key}}` placeholder filled from
  the result files by `build_numbers.py`; the build aborts if a value is missing.

**Model-set deviation.** On the user's request (2026-10-27) the learned-expert candidates were
restricted to models published 2022–2026. The first full run, with the earlier set, is kept in
`run1_learner_set_v1/` and reported; see `docs/REVIEW_AND_FIXES.md`.

## 5. Known limits

- **No orbital-element histories.** Space-Track requires an account and forbids redistribution.
  The re-entry module predicts from one dated orbit per object; the twin uses two (canonical and
  current).
- **Validation of current-state forecasts.**
  - The lifetime pipeline is validated at canonical-orbit epochs only.
  - Current-state forecasts are validated by T1, which covers one 72-day window and is not fully
    blind (see the disclosure).
  - The hashed 2026-10-27 forecast awaits verification against a later catalogue snapshot.
- **Survivorship bias of the size prior.** The class prior of characteristic length comes from
  training objects that received a radar cross-section, and these are biased towards long-lived
  objects.
- **Design-data availability.** Among components it is associated with early decay (adjusted
  odds ratio about 3.4). It is treated as known at launch, and the `no_design` ablation bounds
  its influence.
- **Operational status is not manoeuvre capability.** C4 is a demonstration only.
- **No pairing across datasets.** No public dataset links telemetry, conjunction messages and
  catalogue objects, so health-to-debris effects cannot be measured.
