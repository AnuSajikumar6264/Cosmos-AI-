# Novelty note: COSMOS-AI telemetry study vs. the 2026 ESA-ADB literature

Written 2026-10-24. Sources were read from open full texts where possible. Paths are under
`research_design/references/`. Everything below is what the sources state. Where a source
could not be read, this note says so.

## 1. Closest prior work

| Work | Venue / status | Protocol | What it does | Headline numbers |
|---|---|---|---|---|
| Kotowski et al., ESA-ADB | DMLR 2026 (arXiv 2406.17826) | official: 50/50 split, 58/47 target channels, corrected event-wise F0.5 | benchmark with detector-specific thresholds (TimeEval) | full channels: M1 0.061 (Telemanom-ESA-Pruned), M2 0.079 (PCC); lightweight: M1 0.786, M2 0.949 |
| Brancato, Lucchetti, Kimura, Omata, Giglio | **Aerospace Science and Technology, 2026-10-02**, doi:10.1016/j.ast.2026.113957 (full text paywalled; abstract read) | "full ESA-ADB hierarchical protocol", **Mission 1 only** | label-free SMED (sliding minimum Euclidean distance) and Q-GPR with adaptive retraining; event-morphology taxonomy (8 archetypes) | M1 full: SMED **0.902**, Q-GPR 0.883, out-of-limit 0.754; M1 lightweight: Q-GPR 0.888, SMED 0.811 |
| Şahin, "TC-CAD" | Zenodo 10.5281/zenodo.23322711 (2026-10-13); paper under review (artefacts + preregistrations read: `competitors/zenodo_23344000_tc-cad/`) | ESA-ADB code, Telemanom-ESA, 24 (M1) / 19 (M2) structurally chosen target channels, shortest training windows (3 / 1 months), 16 seeds | (H2) telecommand conditioning (concatenation, FiLM, shuffled control); (H3) per-channel split / sliding / adaptive conformal (ACI) thresholds; (H4) two-threshold deferral under an operator budget | conditioning: preregistered **negative** (FiLM significantly worse; shuffled = real). ACI best point-level calibration, but anomaly event-wise F0.5 drops from 0.090 (C-0) to ~1e-5 (1.6–35 M predicted events). Deferral removes 29–33 pp of automatic rare-nominal alarms at fixed budget |
| Heydari, Mostafavi, Nyarko | IEEE Access 2026, doi:10.1109/ACCESS.2026.3714708 (abstract read) | cross-mission | threshold-policy transfer under alert-workload budgets | policy comparison, not detector design; no telecommands |
| Alcarria et al. | SPAICE 2026 / arXiv 2609.10017 (read) | **non-official**: monthly incremental retraining over the whole mission (not the test half), 6 h grouping | LSTM-AE (M1) / PCA+K-means+LOF (M2), adaptive EVT threshold | F0.5 0.700 (M1, P 0.852, R 0.408), 0.698 (M2); cites PitIA (Tejedor et al., BiDS'25): 0.424 (M1) and 0.882 (M2), "static train/test" |
| Allegrini & Pompei | ECML-PKDD workshops 2026 / arXiv 2605.06681 (read) | Kaggle ESA-ADB challenge, Mission 1, 58 channels, hidden test, anomalies and rare events not separated | supervised hierarchical ensemble (XGBoost/LR, shapelets) | 0.931 public / 0.853 private leaderboard |
| Nguyen et al. | arXiv 2607.07335 (read) | uncorrected event-wise metrics, mostly lightweight subsets | supervised vs unsupervised benchmark (CNN, GCN, GAT, ECOD, ...) | M2 lightweight best F0.5 0.712; not comparable |
| Mou et al., MORA | arXiv 2610.09473 (read) | generic TSAD benchmarks, no ESA-ADB | drift-robust "temporal change disambiguation" | related in spirit (separating explained from unexplained change) |
| Brancato et al. | PHM-AP 2025, doi:10.36001/phmap.2025.v5i1.4465 (abstract) | — | plug-and-play adaptive thresholding TSAD | precursor of the AST paper |
| Fejjari et al. 2026; Dai et al. 2026 | ScienceDirect (blocked here) | — | — | **must be read by the author before submission** |

## 2. What this does to the COSMOS-AI claims

1. **Telecommand gating (our H3) is no longer a contribution.** Şahin's preregistered
   16-seed negative result covers it more rigorously. Report our H3 as a replication on a
   different detector family and channel set (full 58/47 targets, forecasters rather than
   Telemanom) and cite TC-CAD.
2. **"Calibration plus triage for rare nominal events" (the planned H4/H5) is largely
   pre-empted.** TC-CAD has conformal layers, a deferral layer and operator budgets.
   Heydari has workload budgets. Do not build the paper on this.
3. **One distinction that survives, and that the data support:** TC-CAD calibrates
   *points per channel*, and its conformal layers destroy the event-wise score (millions of
   alarm events). COSMOS-AI calibrates *alarm events across channels*: conformal p-values,
   Benjamini–Hochberg across channels, persistence and merging, with the level selected by
   cross-fitting. On Mission 2 (full 47 targets) this layer gives 0.962 at 3.4 false alarms
   per year, against 0.079 (DMLR) and 0.882 (PitIA, as cited by Alcarria). We are not aware
   of a higher full-channel Mission 2 number under the official split. Brancato report
   Mission 1 only.
4. **Mission 1 is the weak point.** Our best stable Mission 1 arm (0.088) is an order of
   magnitude below Brancato's label-free SMED (0.902) and below their out-of-limit
   reference (0.754). Brancato and Alcarria both adapt over time (adaptive retraining /
   monthly retraining). Our detectors are frozen at the end of the training half.
   Mission 1 must be diagnosed before any Mission 1 claim (section 4).
5. **H1 (deep vs classical encoder) remains a negative result.** It is honest, but a
   negative result alone does not carry a Q1 paper.

## 3. Positioning that would survive review

> *The alarm layer, not the encoder, decides event-wise performance on spacecraft
> telemetry. An encoder-agnostic, event-level conformal false-discovery-rate layer, with
> selection by cross-fitting, makes simple and deep forecasters competitive on both
> ESA-ADB missions at full channel scale. Point-level conformal calibration (TC-CAD) does
> not.*

This requires:

- (a) the Mission 1 gap closed, or explained, under a pre-registered design;
- (b) strong label-free baselines on the same evaluation code: out-of-limit, and a
  SMED-like nearest-neighbour detector;
- (c) a direct comparison with point-level conformal / ACI layers on the same scores;
- (d) seeds and statistics at TC-CAD's standard (pre-registration, a fixed family,
  Holm, a minimum effect size).

## 4. Diagnostics run for this note (2026-10-24)

1. **Out-of-limit baseline** (`baselines.out_of_limits`; alarm outside the [min, max] of
   each channel's nominal training samples; no parameters), scored once with the official
   metric code:

   | | M1 full | M1 lightweight | M2 full | M2 lightweight |
   |---|---|---|---|---|
   | nominal training range | 0.0007 (11,627 FA/yr) | 0.202 | **0.894** (20.6 FA/yr) | **0.965** |
   | full training range | 0.0051 | 0.000 | 0.261 | 0.080 |

   - **Mission 2 is close to saturated by a parameter-free rule.** Our 0.962 is 0.07
     above it, not 0.88 above the DMLR best.
   - On Mission 1 the rule fails, as every DMLR baseline does. Brancato et al.'s
     out-of-limit value of 0.754 must therefore come from a different protocol or a
     different definition (perhaps limits supplied by operators, or post-processing).
     Their paper has to be read to resolve this.
2. **Alarm selection on the whole training half** (`oof.py` + `oof_benchmark.select`, using
   training labels only):
   - Mission 1 AR(8) + conformal-BH reaches a training-half CV EW F0.5 of 0.624 (109
     events). The 9-event selection behind the exploratory test value of 0.074 had very
     little to work with.
   - The CV landscape is sharp: k = 10 gives 0.62, k = 30 gives 0.12.
   - This motivated the pre-registered study in `preregistration/PREREG_TELEMETRY_V1.md`
     (frozen as `freeze.json`, ID 09a98f5e160e). Its H4–H6 test (a) whether this
     selection carries over to the test half, and (b) whether a supervised deep event
     scorer adds to it.

## 5. Outcome of the pre-registered study (2026-10-24, freeze 09a98f5e160e)

The full tables are in `outputs/benchmarks/telemetry_esa_adb/oof_protocol/CONFIRMATORY.md` and
`paper/tables/`. All numbers are on the test half, conformal-BH layer, seed means.

| | A0: AR(8), 9-month selection | A1: AR(8), whole-half selection | A2: trees (5 seeds) | A3: TCN (5 seeds) |
|---|---|---|---|---|
| Mission 1 | 0.074 | **0.428** | 0.544 ± 0.236 | 0.205 ± 0.114 |
| Mission 2 | 0.962 | 0.874 | 0.904 ± 0.050 | 0.875 ± 0.058 |

- **H6 supported on Mission 1** (+0.353 [+0.206, +0.488], Holm p 0.0015), **reversed on
  the Mission 2 replication** (−0.089 [−0.143, −0.049]).
- **H4 and H5 refuted.** The deep TCN event scorer loses to AR(8) and to the trees.
- **Exploratory:**
  - trees vs AR(8) on Mission 1: +0.108 [−0.133, +0.344], not significant;
  - supervised scorers' nominal score distributions drift between halves (trees 5–9×, TCN
    3–54× the cross-validated exceedance rate; AR(8) 0.8×);
  - point-level ACI scores ≤ 0.002 everywhere.

**Positioning that the evidence now supports.**

> *On full-channel spacecraft telemetry, how the alarm layer is selected and how stable the
> scorer's nominal distribution is decide operator-grade performance more than model
> capacity. A pre-registered test refutes the hypothesis that a supervised deep event scorer
> helps.*

This is a rigorous empirical paper with a pre-registered negative result for deep learning
and one positive, mission-dependent finding (selection). It is **not** a paper whose main
contribution is a deep-learning method.

- **Strengths for review:** pre-registration with a hash freeze; strong label-free and tree
  baselines; a direct comparison with TC-CAD's point-level approach; honest replication;
  and a mechanism (the calibration-drift diagnostic).
- **Weaknesses:**
  - the selection effect does not replicate;
  - anomaly recall stays at or below 0.33;
  - the supervised arms have large seed variance;
  - Brancato et al.'s 0.902 remains unexplained.
