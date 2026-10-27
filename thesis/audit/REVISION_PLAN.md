# Deliverable C: Prioritised revision plan, and Deliverable F (design part): experiment plans

**Status of every experiment below: PROPOSED.** None has been run. Items marked ⚠ need your approval (Stage 2), because they use the already-seen test halves, shared GPU time, or change original-package code.

## 1. Critical before submission

| ID | Action | Kind | Depends on | Effort | New experiment? |
|---|---|---|---|---|---|
| C1 | **Decompose the H6 gain** (plan F1): one-factor-at-a-time paths from A0 to A1 and from A1 to A0 across the four differences | experiment ⚠ | approval of the evaluation set (F1/F2) | ≈ 1 day including a short pre-registration; CPU only | yes |
| C2 | **Forward-chaining (pseudo-prospective) replication** of the A0-type versus A1-type protocols inside the training halves, which no study has used for evaluation (plan F2) | experiment ⚠ | C1 design | ≈ 1 day; CPU | yes |
| C3 | **Terminology on test exposure.** Thesis and manuscript must say "pre-registered analysis of a previously used test half". A0's test value and A1's training-half CV were known before the freeze | editing | — | 1–2 h | no |
| C4 | **Fair point-level comparator** (plan F3), or narrow the ACI claim to "ACI as configured" | experiment, or editing | — | ½ day; CPU | optional |
| C5 | **Intervals for the exploratory contrasts** the paper relies on: Table 15 and the threshold-versus-BH pairs (plan F5) | analysis | stored alarm runs | ½ day | no new scoring, only bootstrap |
| C6 | **Literature audit**: read Brancato et al. (AST 2026/27), Fejjari et al., Dai et al.; build the comparison matrix | reading ⚠ (needs full texts) | your access to the papers | 1–2 days | no |
| C7 | **Align `paper/manuscript.tex`** with the corrected claims; it still says "identical features", "locates the failure", "moving-block" | editing | C3 | ½ day | no |
| C8 | **Official-equivalent bootstrap.** Re-implement the bootstrap with per-interval events, off-grid events and second-based TNR in a new, non-frozen module; recompute the H4–H6 intervals and p-values and report them beside the frozen ones (STAGE1_AUDIT §5.3) | analysis | — | ½ day; CPU | no new scoring |
| C9 | **Provenance check of the A1 selections.** Re-run `oof_benchmark.select` (frozen code, training half only) for `m1_target_ar8` and `m2_target_ar8` and compare with the saved files (§5.4) | analysis | — | 1 h; CPU | training half only |
| C10 | **Correct "four differences" to five** (residual scale σ) throughout the thesis, the manuscript and the deviation log (append a row; do not edit earlier rows); mention the official restore step (§5.2) | editing | — | 1 h | no |

## 2. Important for acceptance probability

| ID | Action | Kind | Effort | New experiment? |
|---|---|---|---|---|
| I1 | More seeds: A2 to 20 seeds (≈ 3 min each on CPU); A3 to 10 seeds (≈ 1.7 GPU-h each on Mission 1) ⚠ (plan F4) | experiment | A2: 1 h; A3: ≈ 9 GPU-h | yes |
| I2 | Operational metrics from the stored alarm runs: detection delay, alarm duration, alarm runs per detected event, missed-event list by class (plan F6) | analysis | ½ day | no |
| I3 | Metric-adapter unit tests: overlapping alarms, alarms spanning communication gaps, events crossing the split, TNR with gaps (plan F7) | test code | ½ day | no |
| I4 | A pre-registered tuning budget for the TCN, so that "no benefit" is not a tuning artefact ⚠ | experiment | ≈ 14 GPU-h for 8 configurations | yes |
| I5 | Identify the target journal from its current scope and author guidelines (verified on the day) | reading | 1 h | no |
| I6 | Archive code and score arrays with a DOI (Zenodo); licence; `CITATION.cff` ⚠ (needs your account) | packaging | ½ day | no |

## 3. Editorial improvements

- Narrow the abstract to the telemetry question; move the physics chain into a secondary paragraph.
- Title: replace "evidence-graded framework" with a statement of what was tested. Candidates are listed in the Stage-2 questions.
- Front matter: degree, institution, supervisor, declaration and acknowledgements. These need your information.
- Use one name per arm and per layer across text, tables and figures. A check is planned for Stage 5.
- Journal version: drop the physics, vision and re-entry material, keeping one sentence of context.

## 4. Future research beyond the immediate paper

- Drift-aware supervised scoring (recalibration selected for the purpose; causal retraining) on data not yet used.
- Physics chain: per-sample flight-path-angle draws, a per-event risk threshold, use of P_frag, a defined tie-break, a standard SBM Δv law, and solar-activity density. Each needs re-verification, and the chain could become a separate engineering paper.
- Re-entry validation against DISCOS or Space-Track records (needs an account).
- Mission 3 of ESA-ADB as an external sanity check only. It holds 8 anomalies and 25 rare events, has no official split in the code, and its authors advise against benchmarking on it.

---

## F. Experiment plans (design only)

### F1 Decomposition of the H6 protocol gain (AR(8), deterministic)

The five differences between A0 and A1 (STAGE1_AUDIT §5.1) are not independent factors. Out-of-fold scores do not exist for the fitting period under the A0 forecaster: it is fitted on 2000–2006, so its training-half scores are in-sample. A full 2⁴ factorial therefore contains invalid cells. The design uses **one-factor-at-a-time paths in both directions** over the valid cells. Each step changes exactly one factor, and every other setting is fixed at the frozen code defaults.

| Factor | Level A0 | Level A1 |
|---|---|---|
| S, score model (test scores) | AR(8) fitted on the fitting period | AR(8) fitted on all nominal training samples (OOF for training-half scores) |
| σ, residual scale | out-of-sample residuals on the `es` period | in-sample residuals over the whole training half |
| P, calibration pool | nominal scores of the calibration window | nominal OOF scores of the whole training half (scored block excluded during selection) |
| D, selection data | cross-fit within the calibration window | cross-fit over the two training-half halves |
| G, grid | exploratory {1e-2, 1e-3, 3e-4, 1e-4} × {1, 3, 10, 30} | pre-registered {1e-2, 1e-3, 1e-4, 1e-5} × {1, 3, 10, 30, 100} |

- **Forward path:** A0 → (G) → (σ) → (S) → (P) → A1 (= D). **Reverse path:** A1 → (D) → (P) → (S) → (σ) → (G = A0).
- That gives 8 new cells per mission and channel set. Both missions and both channel sets give 32 runs, all deterministic.
- Fixed: merge 1 h, tie-break by alarm fraction, metric `anomaly_and_rare/EW_F0.5`, official labels.
- **Primary outcome:** test-half F0.5 contribution of each step. **Secondary outcomes:** false alarms/yr, recall, anomaly-only F0.5.
- **Statistics:** paired block bootstrap on the same fixed 30-day blocks, 2,000 replicates. No Holm family: these are decompositions, not tests.
- **Interpretation rule:** a factor is material only if its step exceeds +0.05 on **both** paths. If the two paths disagree by more than 0.05, the factors interact, and attribution is reported as order-dependent.
- **Evaluation set ⚠:** the official test half (previously exposed, so the results are exploratory) and/or the forward-chaining folds of F2 (not previously evaluated). Recommended: F2 as primary, test half as secondary.
- **Compute:** OOF scores already exist. Each cell needs a selection pass and one scoring, about 5–15 min CPU, so the total is ≈ 4–8 CPU-hours.

### F2 Forward-chaining evaluation inside the training halves

- **Mission 1.** For each evaluation year y ∈ {2003, 2004, 2005, 2006}:
  - fit AR(8) on nominal samples before y − 1 day;
  - select the layer either on the 9 months before y (A0-type) or on all history before y (A1-type), by the same cross-fit rule;
  - score year y once.
- The training half holds roughly 8–14 labelled events per year in 2001–2006. This count covers all channels; target-channel counts are to be computed.
- **Mission 2.** Quarterly folds from 2000Q3 to 2001Q3.
- **Outcome:** per-fold and pooled F0.5, false alarms/yr and recall.
- **Statistics:** paired comparison across folds, with a fold-level sign test and a pooled block bootstrap.
- **Caveat:** these years' labels entered the A1 training-half cross-validation objective, so they are not unseen labels. They have never been used as an evaluation period, and no protocol decision was based on them. Report the result as a pseudo-prospective check, not as fresh data.
- **Compute:** ≈ 1 CPU-hour per mission.

### F3 Fair point-level comparator

Run ACI with α ∈ {1e-2, 1e-3, 1e-4} and γ ∈ {0.001, 0.005, 0.01, 0.05}, selected on the **event** objective by the same cross-fit rule as the BH layer. Test two variants:

- (a) event logic after ACI, as now;
- (b) a persistence-aware update, in which α_t is updated on the persisted alarm indicator.

Scores: A1 and the A2 seeds. Compute: CPU, ≈ 2–4 h. Report it as an exploratory comparator. It would let the thesis say something about point-level calibration beyond "ACI at α = 0.01".

### F4 Additional seeds ⚠

Run A2 seeds 5–19 (CPU) and A3 seeds 5–9 (XPU, ≈ 1.7 h each). Pre-register them as a separate seed extension with an analysis plan, and report it beside, not instead of, the frozen 5-seed result.

### F5 Intervals for exploratory contrasts

Paired block bootstrap (same blocks, 2,000 replicates) for the Table 15 cells and the threshold-versus-BH pairs of Table 14, using the stored test alarm runs. No model is refitted.

### F6 Operational metrics

From the stored test alarm runs and labels, compute per arm:

- detection delay (alarm start minus event start) for detected events;
- alarm-run durations;
- runs per detected event;
- false-alarm runs per 1,000 nominal hours;
- missed events by class.

Each figure is generated from these outputs only.

### F7 Metric verification tests

Hand-constructed cases with known official outputs:

- an alarm overlapping two events;
- two alarms on one event;
- an alarm inside a communication gap;
- an event that crosses the split;
- an empty prediction;
- an all-alarm prediction.

Each case is passed through `esa_eval` and compared with direct calls to the vendored official code. The re-implemented bootstrap metric is compared with the official one on every stored arm.
