# COSMOS-AI — Deep-Learning Research Methodology Design

Prepared 2026-10-24. Companion file: `A_existing_project_audit.md` (full audit, section A).

**How to read the evidence labels.** Every literature fact carries a verification label:
**[P]** = the publisher / proceedings / arXiv / official page was opened during this work;
**[M]** = only bibliographic metadata (Crossref/OpenAlex) was seen, content not read;
**[AX]** = arXiv-only, not peer-reviewed as far as could be found. Statements marked
*(inference)* are reasoning, not page facts. Data facts about COSMOS come from files on disk.
Web verification used direct HTTP requests (curl) because the built-in web tools were
unavailable; ScienceDirect, MDPI, IEEE Xplore article bodies, OpenReview notes and DBLP blocked
automated access, so items behind those sites are labelled accordingly. **Re-verify every [M]
and [AX] item yourself before it goes into a manuscript.**

---

## A. Existing project audit — summary

Full detail: `A_existing_project_audit.md`. Findings that drive the methodology:

1. **Only one learned model exists** (TinyYOLO, ≈0.32 M parameters, synthetic data). Telemetry,
   catalogue and re-entry are statistical or numerical.
2. **ESA-ADB results are on a COSMOS-specific protocol** (own resampling, 20 % fit window, own
   event metric). "F1 = 1.0" on Missions 1–2 is degenerate (alarm duty 1.0). Mission 3 is
   excluded from comparison by the benchmark authors. **No COSMOS telemetry number is
   currently comparable with published work.**
3. **Subsystem attribution on real data does not beat "blame the largest subsystem".**
4. **No validation split anywhere**; variant choice was made on test data.
5. **Telecommands (821 files) are on disk but unused** — a genuinely paired second modality.
6. **No real imagery**; SPARK/DUST folders empty; the detector's "mAP@50" is a class-agnostic,
   centre-distance AP on 12 synthetic images.
7. **No TLE/element-set history**; SATCAT orbit fields for decayed objects are last-known values
   (label leakage if used as features).
8. **Compute is ample:** 8 × Intel Arc Pro B60 (23.9 GB each), 512 CPU cores, ≈1 TB RAM;
   PyTorch 2.13 + XPU works in the `sgl-xpu` conda env.
9. Numerical physics is verified (20/20 checks) — that is **numerical verification, not
   empirical validation** of any learned model.

---

## B. Candidate deep-learning methodologies

### B.1 What the literature and data permit (evidence base for the choice)

**Telemetry (strongest data position).**
- ESA-ADB is now peer-reviewed: Kotowski, Haskamp, Andrzejewski, Ruszczak, Nalepa, Lakey,
  Collins, Martínez-Heras, De Canio, *"European Space Agency Dataset and Benchmark for Real-World
  Anomaly Detection in Spacecraft Time Series"*, DMLR 3(23):1–76, 2026 [P]
  (https://data.mlr.press/assets/pdf/v03-23.pdf; arXiv 2406.17826; code
  https://github.com/kplabs-pl/ESA-ADB, MIT; data Zenodo 10.5281/zenodo.12528696, CC BY 3.0 IGO
  per the paper). Official protocol: first half of each mission trains (last 3 months =
  validation), second half is tested online; primary metric **corrected event-wise F0.5**
  (event precision × time-level TNR, so permanent alarming is penalised), then channel-/
  subsystem-aware F0.5, alarming precision, ADTQC timing, affiliation F0.5. Binary outputs only;
  VUS/AUC rejected. Mission 3 must not be used for algorithm comparison.
- Published state (DMLR Table 4 [P]): **full channel sets — best 0.061 (Mission 1,
  Telemanom-ESA-Pruned) and 0.079 (Mission 2, PCC)**; lightweight subsets — 0.786 (M1) and 0.949
  (M2). Authors report that no algorithm meets requirements R7 (suppress rare nominal events) or
  R8 (native irregular sampling), and that DC-VAE was "especially disappointing".
- Follow-ups found among the 36 Semantic-Scholar citers (searched 2026-10-24):
  - **M1-lightweight only (saturated):**
    - PSTG, Entropy 28:426, 2026 [P via Europe PMC]: 0.917.
    - Goetze, Schlippe, Lakey, IEEE SCC 2025 [P arXiv 2603.29375]: 0.927, or 0.888 at 59 KB RAM.
  - **Full channels, non-standard split:** Alcarria et al., LSTM-AE + adaptive EVT [AX
    2609.10017]: 0.700 (M1) and 0.698 (M2), with 6 h alarm merging and no channel attribution.
  - **False-alarm budgets under shift (closest competitor for calibration claims):** Heydari,
    Mostafavi, Nyarko, IEEE Access 2026 [M, abstract only]. They report that naive threshold
    transfer between missions violates workload budgets.
  - **Non-deep Kaggle-challenge ensemble:** Allegrini & Pompei, ECML-PKDD workshop [P arXiv
    HTML]: 0.931 public / 0.853 private leaderboard. **Which data subset the challenge used must
    be checked before any "full-channel gap" claim.**
- **No time-series foundation model has been evaluated on ESA-ADB or OPSSAT-AD** (searched both
  cited-by lists and arXiv; not found).
- Evaluation hygiene matters:
  - Under point adjustment, a *random* score reaches PA-F1 0.931 (MSL) and 0.961 (SMAP)
    (Kim et al., AAAI 2022, Table 2 [P]).
  - The Anomaly Transformer, DCdetector and Time-Series-Library code bases set thresholds on
    train+test scores and then point-adjust (code inspected [P]).
  - SMAP/MSL also contain unlabelled anomaly look-alikes (Wu & Keogh, IEEE TKDE 2021 [P]).

**Vision (weak data position).** The only public, real, labelled benchmark with unresolved
multi-frame targets is **SpotGEO**:
- CVPR Workshops 2021 [P]; data on Zenodo, CC BY 4.0.
- 6,400 five-frame sequences, 640×480, about 4.5″/px, 40 s exposures.
- The second-place pipeline used no learning and finished within 0.004 (1−F1) of the deep winner.

SPARK, SPEED/SPEED+, SpaceSense-Bench and TriM-SOD show resolved spacecraft — a different
problem. DUST was not found.

**Re-entry (good novelty, data must be acquired).**
- No survival / censoring-aware ML for re-entry time was found (OpenAlex, arXiv and Crossref
  searches).
- Closest deep prior: Salmaso, Trisolini, Colombo, Aerospace 10(3):297, 2023 [P arXiv
  2303.10183].
  - Design: Seq2Seq GRU, 417 decayed objects, final days of life only, no censoring.
  - Leakage concern: its altitude profiles are fitted with the TIP re-entry epoch as time
    reference *(inference from the source)*.
- Data: Space-Track GP_History with CREATION_DATE. It needs an account. Space-Track's
  documentation gives blanket approval to redistribute basic SSA data with citation [P].
- Prior expectation: tree models stay state of the art on medium tabular data of about 10K rows
  (Grinsztajn et al., NeurIPS 2022 D&B [P]).

**Multimodal.**
- Across features, no paired data exists: ESA-ADB is anonymised, and SpotGEO has no object
  identifiers *(inference)*.
- The MMT light-curve database states that "most MMT data are still not publicly available" [P].
- **Real paired modalities exist inside two features:**
  - telemetry + telecommands (ESA-ADB);
  - per-object TLE history + SATCAT + space weather + optional DISCOS (re-entry), joined by
    NORAD ID.

### B.2 The candidates

**Candidate 1 — Deep temporal and vision intelligence (breadth).**
- Components: one deep model per feature, a modular calibrated output layer, no shared learning.
  - telemetry: TCN/transformer forecaster;
  - vision: CNN/YOLO on difference images;
  - catalogue: MLP/sequence survival model.
- Strength: covers all of COSMOS.
- Weakness: three medium contributions, none decisive. Vision lacks real data. The re-entry
  data still has to be acquired. A reviewer will read it as "routine integration".

**Candidate 2 — Self-supervised representation learning and transfer.**
- Components:
  - masked or forecasting pretraining on nominal telemetry, and MAE/contrastive pretraining on
    unlabelled sky frames (SpotGEO train, Frigate);
  - few-label fine-tuning;
  - cross-dataset tests (M1↔M2, ESA-ADB→OPSSAT-AD, synthetic→SpotGEO).
- Strength: directly addresses label scarcity, and its transfer claims are testable.
- Weakness: self-supervised gains on the telemetry benchmark are untested in the literature
  (good for novelty, high risk). Visual self-supervised learning on faint point sources needs
  bespoke augmentations (standard crops and colour jitter could erase targets — *inference*).

**Candidate 3 — Reliability-aware deep multimodal integration.**
- Components: deep encoders per compatible modality, gated or cross-attention fusion,
  missing-modality handling, ensembles plus conformal prediction, selective prediction.
- Strength: matches the reliability goals.
- Weakness: needs paired multimodal data. Across features it does not exist. Within features
  it does — telemetry + telecommands, and TLE + catalogue + space weather — but not at the
  "multimodal satellite intelligence" scale the candidate implies.

**Candidate 4 (proposed) — Calibrated, telecommand-aware deep forecasting for spacecraft
telemetry.**
- It takes the encoder idea from C1/C2 and the reliability layer from C3, and keeps fusion only
  where pairing is real.
- Components:
  - a channel-independent deep forecaster (pretrained time-series foundation model fine-tuned
    on nominal telemetry; from-scratch TCN/PatchTST as deep baselines);
  - per-channel conformal p-values aggregated with false-discovery-rate control into binary
    event alarms, which also names the anomalous channels and subsystem;
  - telecommand-conditioned gating to suppress rare nominal events.
- Evaluated on full ESA-ADB Missions 1–2 under the official protocol.
- Vision and re-entry stay as independent supporting studies with their own benchmarks
  (sections D, F, G).

### B.3 Comparison

| Criterion | C1 Breadth | C2 SSL / transfer | C3 Multimodal | **C4 Calibrated telemetry** |
|---|---|---|---|---|
| Scientific novelty | Low–medium (integration) | Medium–high | Medium (fusion is routine unless the pairing is new) | **Medium–high**: no foundation-model, calibrated-alarm or telecommand-fusion result on full ESA-ADB found |
| Fit to codebase | Good | Medium | Low (no paired data across features) | **Good**: real ESA-ADB loader, safe unpickler, event metrics and conformal code already exist |
| Data requirement | Real data missing for vision and re-entry | Unlabelled data plentiful | Paired data scarce | **Met now**: ≈11.6 GB ESA-ADB incl. telecommands on disk |
| Training cost | Medium | High (pretraining) | Medium | Medium: fine-tuning, not pretraining from scratch |
| Difficulty | Medium | High | High | Medium–high |
| Reproducibility | Mixed | Medium | Low | **High**: official split, metrics and MIT code |
| Baselines available | Few comparable | Few | Few | **Official baselines with published scores** |
| Potential improvement | Unclear | Unclear | Unclear | Large headroom: full-channel official best ≤ 0.079 |
| Publication fit | Thesis chapter | ML venue | ML venue | Applied-AI and aerospace journals (section J) |
| Risk of failure from data | High (vision, re-entry) | Medium | High | **Low on data; medium on outcome** (deep may not beat calibrated classical — a falsifiable result, still informative) |

**Choice: Candidate 4 as the primary methodology.**
- It is the smallest design that tests a real hypothesis on real, labelled, officially split
  data with published baselines.
- It contains a self-supervised/foundation element (from C2) and a genuine paired-modality
  fusion element (from C3), without inventing cross-feature pairings.
- C2's visual half becomes the supporting vision study (synthetic→SpotGEO). C3's per-object
  fusion becomes the supporting re-entry study.

---

## C. Selected base paper and justification

### C.1 Candidate base papers

| # | Paper | Problem similarity | Method quality | Dataset suitability | Code / weights | Reproducibility | Protocol strength | Extendable limitation | DL-journal relevance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Kotowski et al., ESA-ADB, DMLR 2026** [P] | Exact | Benchmark + 9 baselines incl. Telemanom-ESA, DC-VAE | Exact (data on disk) | MIT, TimeEval-based | High (official pipeline) | **Strongest** (corrected F0.5, channel-aware, timing) | Full-channel ≤ 0.079; R7/R8 unmet; hand-tuned thresholds | High |
| 2 | Hundman et al., Telemanom, KDD 2018 [P] | High | LSTM + nonparametric dynamic thresholding | SMAP/MSL (flawed per Wu & Keogh) | Caltech BSD-style | Medium | Event-wise, but legacy | Absorbed into #1 as Telemanom-ESA | Medium |
| 3 | GDN, AAAI 2021 [P arXiv; repo MIT] | Medium–high | Learned sensor graph + attention, root-cause oriented | SWaT/WADI (PA-era) | MIT | Medium | Weak (PA-era) | Channel attribution untested on ESA-ADB | Medium |
| 4 | MOMENT / Chronos / TimesFM (ICML 2024 / TMLR 2024 / ICML 2024) [P arXiv/README] | Medium | Pretrained channel-independent encoders | Generic corpora | MIT / Apache-2.0 / Apache-2.0 weights | High | Generic | Never tested on spacecraft telemetry benchmarks | High |
| 5 | Bates, Candès, Lei, Romano, Sesia, *Testing for outliers with conformal p-values*, Ann. Statist. 51(1):149–178, 2023 [P] | Method | Exact marginal FDR control; FPR confidence bounds | Any detector | — | High | Theoretical | Assumes exchangeability; never applied to the ESA-ADB metric | High (reliability) |
| 6 | Alcarria et al., LSTM-AE + adaptive EVT, 2026 [AX] | Exact | Reasonable | Full ESA-ADB | No code | Low | Non-standard split | No attribution, non-official | Medium |
| 7 | Gil et al., adaptive conformal AD with foundation models, 2026 [AX 2604.20122; code in IBM granite-tsfm] | High | Foundation-model forecasts + adaptive conformal | Not ESA-ADB; PA-F1 | Yes | Medium | Weak (PA) | Not on the official protocol | High |

### C.2 Selection

- **Primary base paper: ESA-ADB (Kotowski et al., DMLR 2026).**
  - **Reproduced:** Telemanom-ESA and Telemanom-ESA-Pruned, plus Windowed iForest and PCC, run
    through the official pipeline on lightweight and full channel sets of Missions 1 and 2.
    Acceptance: the lightweight-subset scores land within tolerance of DMLR Table 4.
  - **Modified:**
    - the per-channel LSTM forecaster → a channel-independent pretrained foundation model
      (fine-tuned), with from-scratch TCN/PatchTST as deep baselines;
    - the nonparametric dynamic threshold → conformal p-values with false-discovery-rate
      aggregation across channels;
    - telecommands added as a gated second modality.
- **Supporting methodology paper 1: Bates et al., Ann. Statist. 2023** (conformal p-values;
  marginal FDR control) [P], with Gibbs & Candès, *Adaptive Conformal Inference Under
  Distribution Shift*, NeurIPS 2021 [P] for drift.
- **Supporting methodology paper 2: a time-series foundation model.**
  - Primary arm: Chronos (TMLR 2024 per the official README; Apache-2.0 weights; probabilistic
    quantile forecasts). Chronos-2 adds multivariate inputs and covariates — README claim, venue
    not verified.
  - Alternative arm: MOMENT (ICML 2024; MIT weights; channel-independent; evaluated on anomaly
    detection).
  - The arm is chosen **on the validation split only**, and the choice is recorded before the
    test half is touched.
- **Benchmark papers for the other features:**
  - Vision: SpotGEO challenge paper (CVPR Workshops 2021) [P].
  - Re-entry:
    - Salmaso et al. 2023 [P arXiv] as domain comparator;
    - Dynamic-DeepHit (Lee, Yoon, van der Schaar, IEEE TBME 67(1):122–133, 2020) [P Europe PMC]
      and Cox-Time / pycox (Kvamme, Borgan, Scheel, JMLR 20(129), 2019) [P] as methods;
    - Candès, Lei, Ren, *Conformalized survival analysis*, JRSSB 85(1):24–45, 2023 [P] for
      calibrated lower bounds;
    - XGBoost AFT (Barnwal, Cho, Hocking, JCGS 31(4), 2022) [P] as the GBT baseline.

**The new hypothesis beyond the base paper.** On full Missions 1–2, three components each add
measurable, separable value under the official protocol:
1. a pretrained channel-independent forecaster;
2. conformal FDR-controlled alarming;
3. telecommand-aware gating.

Each is tested by swapping one component while holding the others fixed (section G).

---

## D. Proposed deep-learning methodology and architecture

### D.1 Data flow (diagram specification)

```
 ESA-ADB channels (irregular, mixed rates, numeric + categorical)    ESA-ADB telecommands (id, priority, time)
        │  official split: train half (last 3 mo = validation) | test half (online)        │
        ▼                                                                                   ▼
 [1] Grid + mask: resample each channel to the mission's dominant rate,           [T] Event encoder:
     keep a binary "observed" mask (no imputation is hidden)                          learned embedding(id) + priority
        │                                                                              + time-since-command (log-scaled),
 [2] Per-channel robust normalisation (median / IQR from the TRAIN half only)          within a look-back window
        │                                                                                   │
 [3] Channel-independent encoder  ─── shared weights across channels + channel-ID embedding │
     options: TCN (scratch) | PatchTST (scratch) | Chronos or MOMENT (pretrained, fine-tuned)│
        │                                                                                   │
 [4] Gated fusion  h' = h + σ(W_g[h; e]) ⊙ (W_e e)   ◄──────────────────────────────────────┘
        │       (gate = 0 recovers the telemetry-only model; telecommand stream may be missing)
 [5] Quantile forecast head (τ = 0.05 … 0.95) for the next H steps   — categorical channels: softmax head
        │
 [6] Non-conformity score per channel & time: normalised quantile residual
        │
 [7] Conformal p-value per channel vs a calibration set of NOMINAL validation residuals
     (rolling recalibration on recent nominal-labelled training-half data only)
        │
 [8] Cross-channel aggregation: Benjamini–Hochberg at level q  →  rejected-channel set R(t)
        │
 [9] Event logic: alarm if |R(t)| ≥ 1 for ≥ k consecutive steps; merge within a fixed gap
        │
 [10] Outputs: binary alarm stream (official metrics) · affected channels = R(t) · subsystem = map(R(t))
      · calibrated anomaly score = −log10 min-p · abstain flag when OOD score is high
        │
 [11] COSMOS adapter: MonitorResult-compatible object → existing hazard.py / pipeline.py unchanged
```

### D.2 Telemetry encoder (G1)

| Item | Specification |
|---|---|
| Input tensor | `[B, 1, L]` per channel (channel-independent) + channel-ID embedding; observed-mask channel `[B, 1, L]`; optional telecommand events `[B, E, d_e]` |
| Windowing | Context L and horizon H tuned on validation (start: L = 512, H = 16); stride = H at inference; online, causal — no future samples |
| Sampling | Mission-dominant rate (M1 ≈ 0.033 Hz, M2 ≈ 0.056 Hz per DMLR [P]); mask marks missing grid points. R8 (native irregular sampling) is **not** met — stated as a limitation |
| Missing data | No imputation inside the model: masked inputs are zeroed with the mask fed as input; masked targets excluded from the loss and never scored; benchmark "communication gap" / "invalid segment" spans follow the official handling |
| Normalisation | Robust per channel (median / IQR) from the training half only; reversible instance normalisation inside windows as an ablation |
| Encoders compared | TCN (dilated causal, Bai et al. 2018 [AX; MIT]); PatchTST (ICLR 2023; Apache-2.0); Chronos or MOMENT (pretrained; frozen + head, or LoRA / full fine-tuning) |
| Training objective | **Self-supervised on nominal data**: next-H quantile forecasting with pinball loss over training-half windows that carry no anomaly label. Supervised fine-tuning on labelled training-half anomalies is a separate arm (ESA-ADB R4 permits learning from anomalies) |
| Anomaly score | Conformal p-value of the normalised quantile residual (D.1 steps 6–7) |
| Fault localisation | Channel set rejected by BH at the alarm; subsystem by the official channel→subsystem map; evaluated with the official channel-/subsystem-aware F0.5 |
| Explanation | Primary: per-channel p-values (intrinsic, exactly the decision rule). Secondary: Integrated Gradients (Sundararajan et al., ICML 2017 [P]) over the input window, evaluated with the Turbé et al. protocol (Nature Mach. Intell. 5:250–260, 2023 [P]); attention maps are not used as evidence (Jain & Wallace, NAACL 2019 [P]) |

### D.3 Vision encoder (G2) — supporting study

| Item | Specification |
|---|---|
| Task | Point-target localisation in 5-frame sequences (SpotGEO format); output centroids, not boxes |
| Input | 640×480 grayscale × 5 frames; registered difference stack from the existing COSMOS front end (kept) **vs** raw stack into a learned early-fusion / Siamese model (FC-Siam, Daudt et al., ICIP 2018 [P arXiv]) |
| Backbone / head | Small U-Net or ResNet-18-FPN → centre heat-map + sub-pixel offset head (objects-as-points style) |
| Pretraining | MAE-style masked reconstruction on unlabelled SpotGEO train frames (+ optional Frigate, Sci. Data 12:1967, 2025, CC0 [P]). MAE reference code is CC BY-NC 4.0 [P], so implement it independently or check licence compatibility |
| Augmentation | Physically consistent only: PSF width, read/shot noise, background gradients, star density, inter-frame shift; no colour jitter or random crops that remove faint targets |
| Loss | Focal heat-map loss + L1 offset |
| Evaluation | Official SpotGEO metric (F1 + localisation MSE); since every target is small, report by SNR band |
| Domain-shift tests | COSMOS-synthetic → SpotGEO zero-shot; fine-tuning with 1 / 10 / 100 % of SpotGEO train; SatSim (github.com/ssc-ai/satsim; no licence file [P]) as an independent generator |
| Baselines | Existing register + difference + DoG; TinyYOLO; the published SpotGEO table |

### D.4 Catalogue and re-entry encoder (G3) — supporting study

- **Target, defined first:** residual lifetime from a landmark date τ, i.e. S(t | x, τ).
  - Reported as P(re-entry within 30 / 90 / 365 d) plus a conformal lower bound on residual
    lifetime.
  - Restricted to LEO objects with perigee ≲ 600 km at τ.
  - Controlled de-orbits and manoeuvring payloads are a competing risk, or excluded and stated.
- **Features, all observed at or before τ:**
  - Space-Track GP_History elements with CREATION_DATE ≤ τ (not EPOCH);
  - SATCAT static fields excluding orbit fields;
  - F10.7 / Ap with trailing averages only (CelesTrak LAST81, never CENTER81; GFZ data CC BY 4.0
    except sunspot number, CC BY-NC [P]);
  - optional DISCOS mass and area (account and terms required [P]) — a natural missing-modality
    case.
- **Models compared:**
  - XGBoost AFT (censoring-native GBT baseline);
  - Cox-Time and PC-Hazard (pycox, BSD-2) on landmark features;
  - a sequence hazard model over the pre-τ element history (GRU/TCN encoder + discrete-time
    hazard head, Dynamic-DeepHit-style);
  - a physics-offset AFT variant, log T = log T_COSMOS + f(x) + ε, where the existing lifetime
    model supplies T_COSMOS. This learns the physics error and is not a PINN.
- **Censoring:**
  - right-censoring at the data cutoff;
  - left truncation (fragments enter the catalogue after break-up) handled by landmarking.
- **Conformal:** Candès et al. 2023 lower predictive bounds, or the adaptive cut-off version
  (Gui, Hore, Ren, Barber, Biometrika [P arXiv 2211.01227]).
  - Finite-sample validity needs Type-I-like censoring with exogenous censoring times. Censoring
    at the data cutoff is close to that; "lost" objects are not.
  - Report empirical coverage per stratum. Only **marginal** coverage is claimed (Barber et al.,
    Inf. Inference 2020 [P]: exact conditional coverage is impossible distribution-free).
- **Deep learning must earn its place:** if the sequence model does not beat XGBoost AFT within
  confidence intervals, the GBT model becomes the reported model and the deep model is an
  honest ablation.

### D.5 Integration (G4)

| Option | Where the data supports it | Decision |
|---|---|---|
| 1. Independent deep models | All three features | **Default across features** |
| 2. Decision-level fusion | Re-entry: tabular + sequence model outputs for the same NORAD object | Baseline for option 3 |
| 3. Gated fusion | Telemetry + telecommands (same mission and time); re-entry sequence + static + optional DISCOS | **Used** (H3; re-entry ablation) |
| 4. Cross-attention / shared latent | Telemetry tokens × telecommand events | Ablation only; a shared latent across features is **not** used (no paired data) |

- Telemetry ↔ catalogue ↔ imagery stay **interface-coupled, not jointly trained**.
- The health → break-up coupling in `hazard.py` stays a documented, hand-specified scenario
  layer and is excluded from learned-model claims.
- Handling rules:
  - missing modality: gate value 0, trained with modality dropout (Neverova et al., IEEE TPAMI
    2016 [P]);
  - timestamps: everything aligned to the telemetry grid, using only past events;
  - uncertain object identity (re-entry): objects without an unambiguous NORAD match are
    excluded;
  - OOD input: the abstain flag (D.6).

### D.6 Reliability and explainability (G5)

| Quantity | Method | Guarantee type |
|---|---|---|
| False-alarm control | Conformal p-values + Benjamini–Hochberg (Bates et al. 2023) | Formal marginal FDR **if** nominal residual windows are exchangeable with calibration. Telemetry is serially dependent and drifts, so the guarantee is approximate. Measure realised versus nominal false alarms and report the gap |
| Drift | Rolling recalibration; weighted conformal (Barber et al., Ann. Statist. 51:816–845, 2023 [P]) | Bounded coverage loss, not zero |
| Forecast intervals | Conformalised quantile regression (Romano et al., NeurIPS 2019 [P]) | Marginal coverage |
| Epistemic signal | Deep ensemble of 5 (Lakshminarayanan et al., NeurIPS 2017 [P]) | Heuristic; strongest empirically under shift (Ovadia et al., NeurIPS 2019 [P]) |
| OOD / abstention | Mahalanobis or energy score on encoder features, wrapped as a conformal p-value | Formal false-positive-rate control for the abstain decision under exchangeability |
| Selective prediction | Risk–coverage curves; thresholds chosen on validation | — |
| Explanations | Channel p-values; Integrated Gradients with sanity checks (Adebayo et al., NeurIPS 2018 [P]) and stability across seeds (rank correlation) | Not a guarantee; faithfulness measured |
| Not used as guarantees | MC dropout (Foong et al., NeurIPS 2020 [P] show its "in-between" uncertainty pathology); deep evidential models (critiques: Meinert et al., AAAI 2023 [P]; Bengs et al., NeurIPS 2022 [P]); temperature scaling (fails under shift) | Heuristic only |

---

## E. Research gap and specific contributions

### E.1 Gap matrix

| Existing paper | DL method | Dataset | Strength | Limitation | COSMOS-AI opportunity |
|---|---|---|---|---|---|
| Kotowski et al., DMLR 2026 [P] | Telemanom-ESA (LSTM), DC-VAE | ESA-ADB M1/M2 | Official protocol and metrics | **Author-stated:** full-channel scores ≤ 0.079; R7/R8 unmet; manually chosen thresholds | Deep, calibrated, telecommand-aware detector on full channels |
| Hundman et al., KDD 2018 [P] | LSTM + dynamic threshold | SMAP/MSL | Operational origin | **Author-stated:** precision "overly optimistic"; **verified by others:** labels flawed (Wu & Keogh [P]) | Use only as a legacy comparison, without point adjustment |
| PSTG, Entropy 2026 [P] | Spatio-temporal graph | M1 lightweight | Official pipeline, 15 runs | **Verified by inspection:** 6 channels only; no code statement | Full-channel evaluation |
| Goetze et al., SCC 2025 [P arXiv] | XceptionTime forecaster | M1 lightweight | Edge-device efficiency | **Verified:** lightweight subset only | Efficiency reporting at full scale |
| Alcarria et al., 2026 [AX] | LSTM-AE + EVT | Full M1/M2 | Strongest full-channel numbers | **Author-stated:** not comparable to the official split; **verified:** no channel scores | Official split + channel-aware scores + calibrated thresholds |
| Heydari et al., IEEE Access 2026 [M] | Not verified | ESA-ADB M1↔M2 | False-alarm workload budgets | **Abstract:** naive transfer violates budgets | **Must be read in full**; potential overlap with H2 |
| Gil et al., 2026 [AX] | Foundation model + adaptive conformal | Not ESA-ADB | Conformal on foundation-model residuals | **Verified:** PA-F1 reporting | Bring the idea to the official binary-output protocol, with FDR across channels |
| MOMENT / Chronos (2024) [P] | Pretrained encoders | Generic corpora | Zero/few-shot transfer | **Inferred gap:** never tested on spacecraft benchmarks | H1 |
| Bates et al., Ann. Statist. 2023 [P] | — (statistics) | — | Exact marginal FDR control | **Author-stated:** exchangeability assumption | Empirical test of how far the guarantee survives on real telemetry |
| Kumar, Ramesh et al. (Amrita), IEEE Access 2026 [P Xplore] | LSTM/GRU vs (S)ARIMA forecasting + control limits | One decommissioned satellite's gyro, one anomaly | Real ISRO-related data | **Verified:** one subsystem, one anomaly, no uncertainty quantification | Multi-channel, many events, calibrated alarms |
| Fejjari et al., Acta Astronautica 2026 [M]; Dai et al., Adv. Space Res. 2026 [M] | Transformers for telemetry anomaly detection | Not verified | Shows the topic is active in target venues | **Not readable** (ScienceDirect blocked) | Read before submission to avoid an overlap claim |

**Separating the evidence types.**
- **Limitations acknowledged by prior authors:**
  - ESA-ADB: R7/R8 unmet, poor full-channel scores.
  - Telemanom: optimistic precision.
  - Alcarria: non-comparable split.
- **Limitations verified by our own inspection:**
  - lightweight-only evaluations (PSTG, Goetze);
  - threshold selection on test scores plus point adjustment in Anomaly Transformer, DCdetector
    and Time-Series-Library code.
- **Gaps inferred across papers:**
  - no foundation-model evaluation on ESA-ADB or OPSSAT-AD;
  - no calibrated false-alarm results with channel attribution on full channels under the
    official protocol;
  - no telecommand-conditioned deep model.
- **Hypotheses still to test:** H1–H3 below.
- **No claim of "no prior work" is made.** Statements are bounded to the searches performed
  (36 and 60 cited-by lists, arXiv queries, 2026-10-24).

### E.2 Contributions (three, each falsifiable)

**C1 — Pretrained channel-independent forecasting for real spacecraft telemetry (H1).**
- *Mechanism:* a foundation-model encoder fine-tuned on nominal training-half telemetry, with
  quantile forecasts as the basis for anomaly scores.
- *Expected advantage:* better residuals on heterogeneous, drifting channels with few labels.
- *Baselines:* Telemanom-ESA(-Pruned); TCN and PatchTST from scratch; COSMOS robust-z / AR(8);
  Windowed iForest. **All under the same calibrator.**
- *Experiment:* E3–E5, full M1 and M2, ≥5 seeds, block-bootstrap confidence intervals.
- *Falsified if:* the 95 % CI of the corrected-F0.5 difference against the best calibrated
  non-pretrained model includes 0 on both missions.

**C2 — Conformal FDR-controlled alarming with channel attribution (H2).**
- *Mechanism:* per-channel conformal p-values → Benjamini–Hochberg → event alarms; the rejected
  set names the channels.
- *Expected advantage:*
  - a stated, empirically checked false-alarm rate;
  - better channel-/subsystem-aware F0.5 than nonparametric dynamic thresholds or EVT;
  - attribution above the "largest subsystem" baseline, which COSMOS currently fails to beat.
- *Baselines:* the same encoders with nonparametric dynamic thresholds, EVT/POT and fixed
  percentile thresholds.
- *Experiment:* E5, E11.
- *Falsified if:*
  - the realised test false-alarm rate on nominal spans exceeds nominal by more than its CI; or
  - channel-aware F0.5 is not better than the best alternative calibrator at no loss of
    corrected F0.5.

**C3 — Telecommand-aware gating to suppress rare nominal events (H3).**
- *Mechanism:* the gated fusion in D.1 step 4, with telecommand event embeddings; modality
  dropout so the model still works without the telecommand stream.
- *Expected advantage:* fewer false alarms on the benchmark's "rare nominal event" category
  (requirement R7, which no published algorithm meets).
- *Baselines:* no telecommands; a rule that masks k minutes after any command; cross-attention
  instead of gating.
- *Experiment:* E6, E10.
- *Falsified if:* the false-positive reduction on rare nominal events is not significant, or
  anomaly recall falls by more than a pre-set margin (2 points of event recall).

**If novelty narrows.** If the full Heydari et al. paper already provides calibrated alarming on
ESA-ADB, C2 narrows to "FDR-controlled *channel attribution*". C3 — telecommand fusion, R7 — is
the most original component and should then lead.

---

## F. Dataset plan and data-association strategy

| Dataset | Access URL | Licence / restriction | Class | Fields / labels (verified) | DL task | Split | Leakage risks | Cross-dataset use | Linkable to other modalities? |
|---|---|---|---|---|---|---|---|---|---|
| **ESA-ADB** (primary) | zenodo.org/records/12528696 ; github.com/kplabs-pl/ESA-ADB | CC BY 3.0 IGO (per paper; Zenodo field empty — confirm) | REAL | M1: 76 ch (58 target), 14 yr; M2: 100 ch (47 target), 3.5 yr; per-channel annotations (anomaly / rare nominal event / comm gap / invalid); 821 telecommands with priority 0–3 [P]; on disk ≈ 11.6 GB | Self-supervised forecasting; detection; attribution | **Official**: train half (last 3 mo validation) / test half, online | Using test-half statistics for normalisation or thresholds; tuning on test; event-boundary imprecision (author-stated) | M1↔M2 transfer; M3 for OOD only, never comparison | Telecommands: **yes** (same mission and time). Catalogue: **no** (anonymised) |
| **OPSSAT-AD** (secondary) | zenodo.org/records/12588358 ; github.com/kplabs-pl/OPS-SAT-AD | Data CC BY 4.0; code MIT [P] | REAL | 2,123 labelled univariate fragments, 9 channels, ≈20 % anomalous; 18 handcrafted features [P] | Frozen-encoder transfer, few-shot classification | Official stratified split; add a by-channel split | Random split mixes fragments from the same period *(inference)*; 20 % prevalence ≠ operational rate (author-stated by OpsSatNet [P]) | ESA-ADB → OPS-SAT encoder transfer | One spacecraft — cannot support cross-object learning |
| **NASA SMAP/MSL** (legacy) | github.com/khundman/telemanom | Code Caltech BSD-style; data licence not verified | REAL | 82 channels, 105 anomaly sequences [P]; on disk | Legacy comparability only | Official train/test, **no point adjustment** | Known label flaws (Wu & Keogh) | — | No |
| **COSMOS simulated telemetry** | generated | own | SYNTHETIC | Injected fault type and onset | Unit tests, controlled fault-shape studies | Seeds | Same generator for design and test | — | No |
| **SpotGEO** (vision primary) | zenodo.org/records/3874325 → 4432143 | CC BY 4.0 [P] | REAL | 6,400 five-frame sequences, 32,000 PNG 640×480, ≈4.5″/px, 40 s exposures; per-frame coordinates; 20/80 train/test; 4.23 GB [P] | Detection / localisation | Official; validation carved from train by sequence | Frames of one sequence across splits | Synthetic → real | No object IDs *(inference)* |
| **Frigate** (vision, optional) | Sci. Data s41597-025-06220-0; figshare 29545667 | Data CC0; article CC BY 4.0 [P] | REAL, unlabelled | 2,000 FITS frames 9600×6422, 0.5 s exposures; 88.5 GB [P] | Self-supervised pretraining; qualitative stress test | Hold out by night/sequence | — | Cross-sensor | No |
| **COSMOS synthetic frame pairs** | generated | own | SYNTHETIC | Re-configure to SpotGEO geometry | Training source | Seeds | — | → SpotGEO | No |
| **SatSim** | github.com/ssc-ai/satsim | No LICENSE file (AFRL public release) [P] | SYNTHETIC | Point sources, streaks, FITS, masks | Independent generator check | — | — | → SpotGEO | No |
| SPARK / SPEED+ / SpaceSense-Bench / TriM-SOD | (see H.2) | Various, incl. non-commercial | SYNTH / HYBRID | **Resolved spacecraft** | **Not used** for validation; cited as contrast | — | — | — | — |
| **CelesTrak SATCAT** | celestrak.org/satcat | Usage policy; no explicit licence seen | REAL | 70,288 objects on disk; DECAY_DATE | Labels + static features (orbit fields **excluded**) | Temporal by landmark | Orbit fields are last-known (measured: decayed median perigee 199 km) | — | **NORAD ID** |
| **Space-Track GP_History** | space-track.org (account) | User agreement; blanket redistribution approval for basic SSA data with citation [P] | REAL | > 138 M element sets with CREATION_DATE; decay/TIP messages [P] | Sequence features before τ | Temporal: e.g. train τ ≤ 2015, validate 2016–19, test ≥ 2020 | EPOCH vs CREATION_DATE; smoothed filters using future TLEs; duplicate landmarks | — | **NORAD ID** |
| **Space weather** | celestrak.org/SpaceData ; kp.gfz.de | GFZ CC BY 4.0 (SN CC BY-NC 4.0) [P] | REAL | F10.7 obs/adj, Ap, Kp, LAST81 / CENTER81 | Exogenous covariates | By date | CENTER81 and predicted (PRD/PRM) rows leak the future | — | By date |
| **ESA DISCOSweb** (optional) | discosweb.esoc.esa.int | Account + terms; source reference required; eligibility not verified [P] | REAL | Mass, shape, dimensions, cross-sections, re-entry epoch | Missing-modality physical features | Same as above | — | — | **NORAD (satno)** |
| **Aerospace Corp. re-entry database** | aerospace.org/reentries (archived copies opened) | "All Rights Reserved" | REAL | Predicted re-entry time ± window; NORAD number [P] | Comparison with operational predictions only | — | — | — | NORAD ID |
| Recovered-debris records | Aerospace CORDS archive (2015 copy) | Not stated | REAL | 70 narrative items, **no coordinates** [P] | **None** — impact-location error cannot be evaluated | — | — | — | Text only |

**Association rules.**
1. **Never** present unpaired datasets as a multimodal training set.
2. Pairings that genuinely exist:
   - ESA-ADB telemetry ↔ telecommands, by mission and time;
   - SATCAT ↔ Space-Track history ↔ DISCOS, by NORAD ID;
   - space weather, by date.
3. Telemetry, imagery and catalogue are never jointly trained.

**Label scarcity — options and limits.**
- **Self-supervised forecasting** on nominal data. Limit: anomalies inside the training half
  contaminate "nominal". Exclude labelled spans; unlabelled ones remain a threat.
- **Foundation-model transfer.** Limit: the pretraining corpora are unknown at fine grain, so
  check that ESA-ADB is not in them.
- **Augmentation.** Physically consistent only.
- **Controlled simulation.** COSMOS telemetry for unit and fault-shape studies only; never
  claims about real spacecraft.

---

## G. Training, benchmarking, ablation and statistical evaluation

### G.1 Experiment matrix — primary study (telemetry)

| ID | Experiment | Arms | Data | Primary metric | Tests hypothesis |
|---|---|---|---|---|---|
| E1 | Reproduce base paper | Telemanom-ESA, -Pruned, Windowed iForest, PCC | M1, M2 lightweight + full | Corrected event-wise F0.5 | Reproduction |
| E2 | Existing COSMOS monitor on the official protocol | Robust-z/EWMA, AR(8) residuals | M1, M2 full | Same | Honest baseline |
| E3 | Deep from scratch | TCN, PatchTST (+ Telemanom threshold) | Full | Same | H1 context |
| E4 | Pretrained foundation model | zero-shot / head-only / LoRA / full fine-tune | Full | Same | H1 |
| E5 | Calibrator swap | {E2, E3, E4 best} × {dynamic threshold, EVT/POT, percentile, conformal + BH, conformal + BH + rolling} | Full | Corrected F0.5; channel-/subsystem-aware F0.5 | H1 (deep vs classical at equal calibration), H2 |
| E6 | Telecommand fusion | none / rule mask / gated / cross-attention | Full | Rare-nominal-event false positives; anomaly recall | H3 |
| E7 | Self-supervised vs supervised | self-supervised forecasting / + supervised fine-tune on train-half labels | Full | Corrected F0.5 | Candidate-2 question |
| E8 | Ablations | channel-independent vs mixing; context L; quantile vs point head; calibration-window length; persistence k; BH vs Bonferroni vs Cauchy combination; normalisation | Validation, then test once | Corrected F0.5 | Component value |
| E9 | Generalisation | train M1 → test M2 and reverse (recalibrate on the target train half only); per mission phase; OPSSAT-AD frozen encoder; SMAP/MSL event-wise without point adjustment | Cross-dataset | Same + OPSSAT F1/MCC | Transfer |
| E10 | Robustness | random sample / channel dropout; telecommand stream removed at test (missing modality); noise | Full | Degradation curve | Missing-modality robustness |
| E11 | Reliability | realised vs nominal false alarms/year; p-value uniformity on nominal test spans; forecast-interval coverage; risk–coverage; OOD AUROC (M3 or held-out phases as OOD) | Full | Calibration gap | H2 |
| E12 | Efficiency | all arms | Full | Parameters, size, peak memory, latency per telemetry-day, throughput, training GPU-hours | Cost |

### G.2 Supporting studies

| ID | Experiment | Metric |
|---|---|---|
| V1–V7 | Classical baseline on SpotGEO; CNN synthetic-only zero-shot; + self-supervised pretraining; fine-tune 1/10/100 %; learned pair model vs difference input; COSMOS vs SatSim generator; conformal risk control of the miss rate (Angelopoulos et al., ICLR 2024 [P]) | Official SpotGEO F1 / MSE; gap curve |
| R1–R8 | COSMOS physics lifetime + 10–20 % operational rule (Aerospace Corp. CORDS archived page [P]); XGBoost AFT; Cox-Time / PC-Hazard; sequence hazard; physics offset; conformal lower bounds; missing DISCOS modality; leave-one-break-up-event-out for fragments | Uno C, Antolini Ctd, IPCW integrated Brier score, time-dependent AUC at 30/90/365 d, calibration, relative error \|T̂−T\|/(T−τ), empirical coverage |
| I1–I3 | Re-entry: independent / late fusion / gated fusion of sequence + static + DISCOS | Integrated Brier score, C-index |

### G.3 Metrics

| Group | Metrics |
|---|---|
| Telemetry | **Official ESA-ADB set:** corrected event-wise F0.5, channel-aware and subsystem-aware F0.5, alarming precision, ADTQC, affiliation F0.5. Also event P/R/F1, false alarms/year (realised vs nominal), detection delay, attribution vs the "largest subsystem" and random baselines. PR-AUC only as a secondary, threshold-free score (the benchmark rejects it as primary) |
| Vision | SpotGEO F1 and MSE; precision, recall; IoU-based mAP@50 / 50–95 only when boxes are meaningful; per-SNR breakdown; latency |
| Re-entry | Survival metrics as in G.2, with the risk-score transformation reported (Sonabend et al., Bioinformatics 2022 [P]). Impact-location error **not reported** — no coordinate ground truth exists |
| Reliability | ECE and reliability diagrams for horizon probabilities; coverage and region/interval size; risk–coverage curves; OOD AUROC / AUPRC |
| Efficiency | Trainable parameters, model size, peak memory, latency, throughput, training cost |

### G.4 Statistical protocol

- **Splits:** official splits where they exist. Elsewhere use temporal or grouped splits:
  - by mission and time for telemetry;
  - by sequence for SpotGEO;
  - by landmark date and by launch / parent event for re-entry.
- **Selection:** every hyperparameter, encoder choice and calibrator setting is chosen on
  validation. The test half is run **once** per frozen configuration. Configurations are
  committed (hash) before the test run.
- **Seeds:** ≥ 5 per deep arm; report mean ± SD and per-seed values.
- **Confidence intervals:** moving-block bootstrap over test time (e.g. 30-day blocks) for
  event-level metrics. Resampling events alone ignores false-alarm dependence.
- **Paired tests:** paired block-bootstrap of metric differences (primary), and Wilcoxon
  signed-rank across seeds × missions (secondary). Holm–Bonferroni correction over the
  pre-registered comparisons (H1: 1, H2: 2, H3: 2).
- **Pre-registration:** H1–H3, primary metric and margins are written down before the test run
  (e.g. an OSF entry, or a dated commit).

### G.5 Result-table templates (deliberately blank)

**Table T1 — Official ESA-ADB results, full channel sets (test half).**

| Model | Calibrator | M1 corr. F0.5 | M1 channel-aware F0.5 | M1 false alarms/yr | M2 corr. F0.5 | M2 channel-aware F0.5 | M2 false alarms/yr |
|---|---|---|---|---|---|---|---|
| Telemanom-ESA-Pruned (reproduced) | dynamic threshold | | | | | | |
| COSMOS AR(8) | conformal + BH | | | | | | |
| PatchTST (scratch) | conformal + BH | | | | | | |
| Foundation model, fine-tuned | conformal + BH | | | | | | |
| + telecommand gating | conformal + BH | | | | | | |

**Table T2 — Calibration (H2).**

| Model | Nominal FDR q | Realised false alarms/yr (CI) | p-value KS statistic on nominal spans | Channel-aware F0.5 |
|---|---|---|---|---|

**Table T3 — Telecommand fusion (H3).**

| Variant | Rare-nominal-event false positives | Anomaly recall | Corrected F0.5 | Δ vs none (CI) |
|---|---|---|---|---|

**Table T4 — Efficiency.**

| Model | Params | Size (MB) | Peak memory (GB) | Latency per telemetry-day | Throughput | Training GPU-h |
|---|---|---|---|---|---|---|

**What supports or rejects each hypothesis:**
- H1 is supported only if the foundation-model row beats the best non-pretrained row under the
  **same** calibrator, with the CI excluding 0.
- H2 needs realised false alarms within the nominal CI and higher channel-aware F0.5.
- H3 needs fewer rare-nominal false positives without the recall loss exceeding the margin.

---

## H. Verified literature review

### H.1 Core methodological literature

**Telemetry anomaly detection**
- ESA-ADB (Kotowski et al., DMLR 2026) [P] — see B.1. Core; base paper.
- Ruszczak, Kotowski, Evans, Nalepa, *The OPS-SAT benchmark for detecting anomalies in satellite
  telemetry*, Sci. Data 12:710, 2025, doi 10.1038/s41597-025-05035-3 [P].
  - Data: 30 baseline models on 2,123 fragments.
  - Results: FCNN F1 0.946; IForest 0.295; DeepSVDD 0.279 (Table 3).
  - Status: near-saturated (OpsSatNet, Sensors 2026, reaches 0.960 ± 0.009 [P via Europe PMC]).
- Hundman et al., KDD 2018, doi 10.1145/3219819.3219845 [P].
  - Results: event-level MSL precision 92.6 % / recall 69.4 %; SMAP 85.5 % / 85.5 % (Table 2).
- Wu & Keogh, IEEE TKDE 2021, doi 10.1109/TKDE.2021.3112126 [P] — benchmark flaws.
- Kim et al., AAAI 2022, doi 10.1609/aaai.v36i7.20680 [P] — point-adjustment inflation.
- Liu & Paparrizos, TSB-AD, NeurIPS 2024 D&B [P].
  - Finding: simple and statistical methods often win; MOMENT ranks 5th on univariate.
- Deep anomaly-detection methods with code:
  - Anomaly Transformer, ICLR 2022 [P arXiv 2110.02642; MIT];
  - TranAD, PVLDB 2022 [P DOI];
  - OmniAnomaly, KDD 2019;
  - USAD, KDD 2020;
  - GDN, AAAI 2021 (MIT);
  - PatchTST, ICLR 2023 (Apache-2.0);
  - TimesNet, ICLR 2023 (MIT).

  Their published scores mostly use point adjustment and test-informed thresholds, so they are
  not comparable with the official protocol.
- Foundation models:
  - MOMENT (ICML 2024, MIT): channel-independent, anomaly detection supported;
  - TimesFM (ICML 2024, Apache-2.0): forecasting;
  - MOIRAI (ICML 2024, **CC BY-NC** weights);
  - Chronos (TMLR, Apache-2.0);
  - Timer (ICML 2024, Apache-2.0): anomaly detection supported;
  - UniTS (NeurIPS 2024).

  Each verified from its arXiv / README / Hugging Face page [P]. None has been tested on ESA-ADB.
- 2026 telemetry papers in target venues, for an overlap check before submission:
  - Fejjari et al., Acta Astronautica, doi 10.1016/j.actaastro.2025.09.035 [M];
  - Dai et al., Adv. Space Res., doi 10.1016/j.asr.2025.10.088 [M];
  - Hong et al., IEEE TAES (Xplore 11495522) [M];
  - Cuéllar et al., Eng. Appl. AI 2024, doi 10.1016/j.engappai.2024.108083 [M];
  - JSSE 2026 OPS-SAT explainable ML, doi 10.1016/j.jsse.2026.09.003 [M];
  - systematic review, Nithya et al., IEEE Access 2026, doi 10.1109/ACCESS.2026.3745984 [M].

**Uncertainty, conformal prediction and explainability.** The guarantee classes are in D.6.

| Class | Papers |
|---|---|
| **Formal finite-sample guarantees** | Lei et al., JASA 2018 [P]; Angelopoulos & Bates, Found. Trends ML 16, 2023 [P]; Romano et al., NeurIPS 2019 [P]; Tibshirani et al., NeurIPS 2019 [P]; Barber et al., Ann. Statist. 2023 [P]; Bates et al., Ann. Statist. 2023 [P]; Angelopoulos et al., ICLR 2024 [P]; Candès et al., JRSSB 2023 [P] |
| **Long-run average coverage** | Gibbs & Candès, NeurIPS 2021 [P] |
| **Impossibility result** | Barber et al., Inf. Inference 2020 [P] |
| **Heuristic** | Deep ensembles (NeurIPS 2017) [P]; MC dropout (ICML 2016) [P] with the Foong et al. critique (NeurIPS 2020) [P]; evidential deep learning (NeurIPS 2018/2020) [P] with critiques (AAAI 2023, NeurIPS 2022) [P]; temperature scaling (ICML 2017) [P]; Ovadia et al. (NeurIPS 2019) [P] on calibration under shift |
| **OOD scores** | MSP (ICLR 2017) [P arXiv]; energy (NeurIPS 2020) [P]; Mahalanobis (NeurIPS 2018) [P]; OpenOOD (NeurIPS 2022 D&B) [P]; TS-OOD [AX 2502.15901] |
| **Selective prediction** | Geifman & El-Yaniv (NeurIPS 2017) [P]; SelectiveNet (ICML 2019) [P] |
| **Explainability** | Integrated Gradients (ICML 2017) [P]; SHAP (NeurIPS 2017) [P] with the Kumar et al. critique (ICML 2020) [P]; Jain & Wallace (NAACL 2019) [P]; Wiegreffe & Pinter (EMNLP 2019) [P]; Adebayo et al. (NeurIPS 2018) [P]; Ismail et al. (NeurIPS 2020) [P]; Turbé et al. (Nat. Mach. Intell. 2023) [P] |

**Fusion**
- Gated multimodal units: arXiv 2017; journal version in Neural Comput. Appl. 2020 [P].
- Tensor Fusion (EMNLP 2017) [P] and MulT (ACL 2019) [P].
- Missing-modality robustness:
  - Ma et al., CVPR 2022 [P]: transformers are sensitive to missing modalities;
  - SMIL, AAAI 2021 [P];
  - ModDrop, TPAMI 2016 [P].
- Trusted multi-view classification, ICLR 2021 [P] (evidential, so its uncertainty is heuristic).
- MultiSurv, Sci. Rep. 2021 [P DOI] — fusion-survival template.

**Re-entry and survival**
- Domain:
  - Salmaso et al. 2023 [P arXiv];
  - Jung et al., Adv. Space Res. 68(6), 2021 [M];
  - Peng & Bai, J. Astronaut. Sci. 2019/2020 [P];
  - Shorten et al., J. Spacecraft Rockets 60(6), 2023 [P arXiv] — TLE pitfalls;
  - Dutt et al., Adv. Space Res. 2023 [M].
- Methods:
  - DeepSurv (BMC Med. Res. Methodol. 2018) [P];
  - DeepHit (AAAI 2018) [P];
  - Dynamic-DeepHit (IEEE TBME 2020) [P];
  - Cox-Time (JMLR 2019) [P];
  - PC-Hazard (Lifetime Data Anal. 2021) [P];
  - SurvTRACE (ACM BCB 2022) [P arXiv].
- Evaluation pitfalls: Uno et al. 2011; Antolini et al. 2005; Graf et al. 1999; Blanche et al.
  2019; Sonabend et al. 2022 [P via Europe PMC].
- Tabular deep learning vs trees:
  - Grinsztajn et al. (NeurIPS 2022) [P];
  - FT-Transformer (NeurIPS 2021) [P];
  - TabPFN (Nature 637:319–326, 2025) [P] — no survival support seen;
  - XGBoost AFT (JCGS 2022) [P].
- Conformal survival: Candès et al. 2023 [P]; Gui et al., Biometrika [P arXiv]; Qi et al., ICML
  2024 [P].

### H.2 Vision literature and dataset fit

| Item | Verified facts | Fit |
|---|---|---|
| SpotGEO (CVPRW 2021) [P] | Real, unresolved, 5-frame, labelled; winner 1−F1 0.0517 (U-Net ensemble); second 0.0557 (no learning) | **Primary real test** |
| Frigate (Sci. Data 2025) [P] | Real LEO streak sequences, unlabelled, CC0 | Self-supervised / qualitative |
| Kruk et al., Nat. Astron. 7:262–268, 2023 [P] | HST trails, image-level labels, CC BY 4.0 | Partial (streaks, space-based observer) |
| DeepStreaks, MNRAS 486:4158, 2019 [P] | ZTF streak CNNs, 96–98 % true-positive rate | Partial |
| Rachith et al. 2026 [AX 2606.30286] | Real VST data: precision 0.783 → 0.990 with a classifier | Evidence that measured sim→real drops are publishable |
| Chen et al. [AX 2509.16771; "PASP accepted"] | > 99 % on simulation → recall 79.57 / precision 74.56 on real data | Same |
| SPARK 2021/22, SPEED+, SpaceSense-Bench [AX 2603.09320], TriM-SOD (Space: Sci. Tech., doi 10.34133/space.0299) | Resolved spacecraft | **Mismatch** — cite as contrast only |
| Ultralytics YOLO11 docs [P] | AGPL-3.0 | Baseline only; licence obligations if distributed |
| RT-DETR (CVPR 2024) [P]; SAHI (ICIP 2022) [P arXiv]; MAE (CVPR 2022) [P]; SimCLR (ICML 2020) [P arXiv]; MoCo (CVPR 2020) [P]; DINOv2 [P arXiv; venue not verified] | Method references | — |

### H.3 Amrita Vishwa Vidyapeetham publications

Affiliation was verified on the IEEE Xplore author record for each of the papers below.

| # | Paper | Amrita authors (as printed) | Venue / DOI | Relevance | Use |
|---|---|---|---|---|---|
| 1 | *Satellite Anomaly Prediction Using Time-Series Analysis of Telemetry Data* — P. S. Kumar, T. K. Ramesh, J. K. Kishore (URSC), A. Koithyar | Kumar, Ramesh — ECE, Amrita School of Engineering, Bengaluru | IEEE Access 2026, 10.1109/ACCESS.2026.3699365 | **Direct** | **Core related work**: closest institutional precedent; COSMOS-AI extends it to many channels and events, calibrated alarms and attribution |
| 2 | *Spectrum-Agnostic Space Debris Detection with Autoencoders* — Shakthi S, Lekshmi R. R | Both — EEE, Coimbatore | ICAUC 2026 (IEEE conf.), 10.1109/ICAUC68182.2026.11441314 | **Direct** (conference) | Vision related work (reconstruction-based debris detection) |
| 3 | *Sensor Anomaly Detection in Nuclear Power Plant Using Deep LSTM Denoising Autoencoder and Isolation Forest* — Gandhimathinathan A, Ananthakrishnan C G, Lavanya R, R Jehadeesan, P R Reddy | First three — ECE, Coimbatore | IEEE Sensors Letters 2024, 10.1109/LSENS.2024.3496540 | General method | **Core**: deep reconstruction-residual baseline for multichannel sensor anomaly detection |
| 4 | *Enhancing Early Exit Performance With Uncertainty-Aware Training in CNNs* — T. Kurian, S. Thangam | Both — CSE, Bengaluru | IEEE Access 2025, 10.1109/ACCESS.2025.3572415 | General method (uncertainty) | **Core**: uncertainty-aware efficient inference |
| 5 | *Multimodal Sentinel-1/2 Data Fusion With a Dual-Path ViT-KAN Pipeline for Flood Event Classification* — J. Anandakrishnan, A. Mohan, V. Venugopal, W. Jiang, M. A. A. Rahman, J. Khan | Anandakrishnan, Venugopal — Coimbatore | IEEE Access 2026, 10.1109/ACCESS.2026.3709031 | Direct (Earth observation, multimodal) | **Core**: multimodal fusion precedent |
| 6 | *A Deep Learning Model for Integrating Landsat-8 and Sentinel 2 … Drought Monitoring* — Aravinth J, Anand R | Both (record prints "Coimbatore … Haridwar"; check the PDF) | IEEE JSTARS 2026, 10.1109/JSTARS.2025.3637223 | Direct (Earth observation) | **Core**: multi-source satellite fusion + recurrent model |
| 7 | *ConvFormer-Lite …* — B. P. Kumar, Aravinth J, S. Rajendran, A. Mukundan | Kumar, Aravinth — ECE, Coimbatore | IEEE JSTARS 2026, 10.1109/JSTARS.2026.3686406 | Direct (Earth observation) | Lightweight CNN–transformer efficiency |
| 8 | *Saliency-Guided Feature Mining Network for Fine-Grained Scene Classification* — D. Revikumar, A. P. Byju, L. Bruzzone | First two — CSE, Amritapuri | IEEE GRSL 2026, 10.1109/LGRS.2025.3631832 | Direct (remote sensing) | Fine-grained small-object recognition |

Also found:
- **Affiliation verified but weak relevance (not counted):**
  - ship detection with YOLO-NAS, ICSC 2023;
  - space-debris clustering, ICEC 2024 (not deep learning);
  - neuro-symbolic XAI for satellite IoT, GSEACT 2026 (IoT data);
  - DA-DCC-IDS, IEEE Access 2026 (trees).
- **Affiliation unverified (publisher pages blocked):** five Taylor & Francis / SPIE / PeerJ / MDPI
  items. They can be added after you check the PDFs.

**Recommended six for the core related-work section:** #1, #3, #4, #5, #6, plus #2 for the
vision part.

---

## I. Step-by-step codebase implementation roadmap

### I.1 Structure

This follows the codebase's own pattern: one sub-package per concern, flat modules with
docstrings.

```
cosmos/
  health/ debris/ reentry/            unchanged (classical + numerical reference)
  ai/                                 NEW — learned models; imports from, never edits, the above
    __init__.py
    registry.py         model registry: config hash → checkpoint path, seed, git commit
    conformal.py        split / CQR / conformal p-values / BH / rolling + weighted calibration
                        (reentry/montecarlo.py is later refactored to import from here)
    ood.py              Mahalanobis / energy scores → conformal p-values; abstain flag
    explain.py          Integrated Gradients (captum) + Turbé-style faithfulness, stability
    stats.py            moving-block bootstrap, paired tests, Holm correction
    telemetry/
      esa_official.py   official split + telecommand loader (wraps kplabs-pl/ESA-ADB);
                        reuses health/realdata.py's safe unpickler
      encoders.py       TCN, PatchTST, foundation-model adapters (Chronos, MOMENT)
      forecaster.py     quantile head, nominal-only training loop, masking
      telecommand.py    event embedding + gated fusion + modality dropout
      alarms.py         residual → p-value → BH → events → channels / subsystems
      esa_metrics.py    thin adapter over the official metric code
      adapter.py        → health.monitor.MonitorResult (so hazard.py / pipeline.py work unchanged)
    vision/
      spotgeo.py        loader + official metric
      heatmap_detector.py, pretrain.py
    survival/
      spacetrack.py     rate-limited GP_History fetch + cache; landmark builder (CREATION_DATE ≤ τ)
      models.py         XGBoost-AFT, pycox Cox-Time / PC-Hazard, sequence hazard, physics offset
      metrics.py        Uno C, Antolini Ctd, IPCW integrated Brier score, td-AUC
  pipeline.py           + backend="classical" | "learned" (classical stays the default)
configs/                YAML per experiment (telemetry/E1…E12, vision/V1…, survival/R1…)
scripts/                train.py, evaluate.py, reproduce_esa_baselines.sh, fetch_spotgeo.py
experiments/            one folder per run: config.yaml, commit, seed, metrics.json, checkpoint
tests/                  test_verify.py, test_datasets.py (unchanged) + test_ai_*.py
```

### I.2 Phased plan

| Phase | Work | Depends on | Acceptance criterion |
|---|---|---|---|
| **0 Novelty and setup** (first) | Read the full Heydari et al. 2026 paper and the Allegrini & Pompei challenge subset; read Fejjari 2026 / Dai 2026; clone ESA-ADB; XPU smoke test of PatchTST + Chronos/MOMENT; pin the environment | — | Written novelty note; foundation-model forward/backward pass runs on XPU; `conda env export` committed |
| **1 Reproduce base** (E1) | Official pipeline: Telemanom-ESA(-Pruned), Windowed iForest, PCC | 0 | Lightweight-subset scores within tolerance of DMLR Table 4 (tolerance fixed in advance, e.g. ±0.03 or the paper's spread) |
| **2 Honest COSMOS baseline** (E2) | Run robust-z/AR(8) through official split and metrics; drop Mission 3 from comparisons; fix the README (DUST, mAP label) | 1 | Table T1 rows for COSMOS filled; old numbers marked "non-official" |
| **3 Deep encoders** (E3, E4) | `encoders.py`, `forecaster.py`; nominal-only training; ≥5 seeds | 1 | Validation curves; test run once per frozen config |
| **4 Calibration layer** (E5, E11) | `conformal.py`, `alarms.py`; apply to **all** models | 3 | Unit test: p-values uniform on exchangeable nulls; BH controls FDR on simulated nulls within binomial CI; Table T2 |
| **5 Telecommand fusion** (E6, E10) | `telecommand.py` | 4 | Table T3 |
| **6 Robustness and efficiency** (E7–E9, E12) | Transfer, ablations, profiling | 5 | Tables complete; latency/memory measured, not estimated |
| **7 Vision study** (V1–V7) | SpotGEO download (4.23 GB); simulator re-configured to SpotGEO geometry | 0 | Official-metric numbers; gap curve |
| **8 Re-entry study** (R1–R8, I1–I3) | Space-Track account; GP_History fetch within rate limits (< 30 requests/min, < 300/h [P]); landmarks | 0 | Leakage unit test passes (no feature timestamp > τ); survival tables |
| **9 Manuscript** | Section J outline | 1–6 | All numbers regenerated by one script from frozen configs |

### I.3 Tests

- **Unit tests:**
  - conformal coverage on exchangeable synthetic data;
  - BH FDR control;
  - p-value uniformity;
  - leakage guard (feature time ≤ τ; train statistics only);
  - metric adapter equals the official code on a fixture;
  - masking (masked targets never scored).
- **Integration tests:**
  - `run_pipeline(backend="learned")` on simulated telemetry returns a `MonitorResult` that
    `hazard.py` accepts;
  - a tiny SpotGEO fixture runs end to end.
- **Regression tests:** `test_verify.py` 20/20 and `test_datasets.py` must pass at every phase.

### I.4 Reproducibility and resources

- **Reproducibility:**
  - pinned conda environment;
  - seeds recorded; deterministic settings where XPU allows (report non-determinism if present);
  - data checksums (already implemented in `datasets.py`);
  - one command per table;
  - code + configs released with a Zenodo DOI.
- **Data redistribution:**
  - ESA-ADB: cite CC BY 3.0 IGO;
  - Space-Track: basic SSA data only, with citation;
  - do not redistribute Aerospace Corp. content.
- **Resources:**
  - GPU: 8 × Arc Pro B60, 24 GB each — sufficient for fine-tuning small and base foundation
    models. **Measure** actual memory and time in Phase 0 rather than estimating now.
  - CPU / RAM: 512 cores and ≈1 TB for TimeEval baselines and bootstraps.
  - Storage: `/data` has ≈179 GB free. ESA-ADB is already present (≈11.6 GB) and SpotGEO adds
    4.23 GB. Frigate (88.5 GB) is optional — take a subset.
  - Risk: some packages assume CUDA (e.g. fused attention kernels). Fall back to standard
    PyTorch attention on XPU.

---

## J. Journal readiness, limitations, risks and next actions

### J.1 Paper scope and outline

**Recommended scope:** one journal paper on the telemetry study (C1–C3). Vision and re-entry
become thesis chapters or follow-up papers. COSMOS-AI appears as the modular framework in the
discussion, not as a claimed contribution.

**Proposed title.** *Calibrated Deep Forecasting for Spacecraft Telemetry Anomaly Detection:
Conformal False-Alarm Control and Telecommand-Aware Fusion on the ESA Anomaly Benchmark.*

**Research problem.** Deep detectors for real spacecraft telemetry either do not beat simple
methods on full channel sets, or report numbers on protocols that inflate them. Operators also
lack a stated false-alarm rate and a reliable "which channel" answer.

**Research questions**
- RQ1 (H1): Does a pretrained channel-independent forecaster improve official full-channel
  detection over strong classical and deep baselines **at equal calibration**?
- RQ2 (H2): Do conformal FDR-controlled alarms deliver their nominal false-alarm rate on real
  telemetry and improve channel attribution?
- RQ3 (H3): Does telecommand-aware gating suppress rare-nominal-event false alarms without
  losing recall?

**Manuscript outline**
1. Introduction.
2. Related work: telemetry benchmarks; evaluation pitfalls; foundation models; conformal anomaly
   detection; Amrita precedents.
3. Data and official protocol.
4. Method: encoder, calibration, fusion, adapter.
5. Experimental design: E1–E12; statistics; pre-registration.
6. Results: T1–T4; ablations; transfer; efficiency.
7. Discussion: when deep learning helps, and when calibration does the work.
8. Limitations and threats to validity.
9. Reproducibility and ethics.
10. Conclusion.

**Comparison with existing work.**
- Official-protocol rows only in the main table (reproduced Telemanom-ESA, published DMLR
  numbers).
- Non-official numbers (Alcarria, PSTG lightweight, point-adjusted papers) go in a separate,
  labelled table and are never ranked against.

### J.2 Limitations and threats to validity

- **Benchmark coverage:** one benchmark with two usable missions; data anonymised, so no
  physics priors; annotation end times imprecise (author-stated).
- **Conformal guarantees:** they assume exchangeability, which serial dependence and drift
  violate. Realised calibration is measured; it is not assumed.
- **Sampling:** irregular sampling is resampled, so requirement R8 is not met.
- **Foundation-model contamination:** pretraining corpora are not fully known; check whether
  ESA-ADB could be included.
- **Selection bias:** guarded by validation-only selection and a single test run.
- **Compute fairness:** pretrained models consume external data. Report it, and include
  from-scratch deep baselines.
- **Supporting studies:**
  - Vision: one real benchmark (GEO-only).
  - Re-entry: no impact-location ground truth exists, so **no landing-location accuracy claim
    can be made**.

### J.3 Ethics and dual use

- Space-surveillance methods are dual-use. Frame the work as safety and operations.
- Expert Systems with Applications states it "no longer considers papers that contain
  applications to military/defense systems" [P].
- No personal data is used. Respect each dataset licence (several are non-commercial: MOIRAI
  weights, SPEED+, SpaceSense-Bench, MAE code).

### J.4 Venues

Requirements below were read from official journal pages [P] unless noted. Verify indexing
against your university's required list (Scopus / Web of Science / UGC-CARE) — that list was not
checked.

| Venue | Scope fit | OA / APC | Notes |
|---|---|---|---|
| **Engineering Applications of Artificial Intelligence** (Elsevier) | Strong — deep-learning applications, fault detection; must be "validated using public data sets"; the abstract must state both the AI contribution and the application | Hybrid; APC USD 3,040 | Example: Cuéllar et al. 2024 [M]. **Recommended first choice** for a deep-learning-primary paper |
| **IEEE Trans. Aerospace & Electronic Systems** | Strong — scope names spacecraft and telemetry | OA option USD 2,800 (2026) | $200 per page beyond 10 pages; Code Ocean / DataPort encouraged; Hong et al. 2026 example |
| **Acta Astronautica** | Strong | Hybrid; APC USD 3,750 | Fejjari et al. 2026 example [M] |
| **Advances in Space Research** | Strong (also re-entry) | Hybrid; APC USD 3,058 | Dai et al. 2026 example [M] |
| **Journal of Space Safety Engineering** | Good (safety angle) | Hybrid; APC USD 2,930 | OPS-SAT explainable-ML paper 2026 [M] |
| **IEEE Access** | Good, broad | Gold OA; APC USD 2,160 | < 20 pages recommended; SCIE / Scopus / WoS listed; Amrita precedent #1 |
| **Neural Computing and Applications** | Good | Hybrid; APC USD 3,190 | Data-availability statement mandatory; Scopus listed, WoS not in the page list |
| **Machine Learning: Science and Technology** (IOP) | Only if the ML contribution is strong | Gold OA; APC USD 3,350; 40 % discount for India-based authors | Research papers ≤ 8,500 words; data sharing required |
| Aerospace (MDPI) | Good | Gold OA; CHF 2,400 | Data-availability statement and code statement mandatory |

No acceptance is guaranteed.

### J.5 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| H1 fails (deep ≈ calibrated classical) | Medium | Still publishable as a rigorous negative result, with C2/C3 as contributions; the framing is pre-registered |
| Heydari et al. 2026 overlaps H2 | Unknown until read | Phase 0 read first; narrow to FDR channel attribution and lead with C3 |
| Telecommand semantics are anonymised (only IDs and priority) | Medium | Learned ID embeddings; the rule-based mask baseline isolates the gain |
| XPU incompatibility of foundation-model code | Low–medium | Phase 0 smoke test; fall back to the other foundation model or to plain-PyTorch attention |
| Exchangeability badly violated | Medium | Rolling / weighted calibration; report the realised gap honestly |

### J.6 Next actions, in order

1. **Read** Heydari et al. (IEEE Access 2026) and the Allegrini & Pompei challenge paper in full;
   write a one-page novelty note. *(Needs your institutional access to IEEE Xplore and
   ScienceDirect.)*
2. **Discuss this design with your guide**, especially the narrowing to one telemetry paper.
3. **Phase 0–1:** clone the ESA-ADB repository and reproduce the lightweight-subset baselines.
4. **Phase 2:** re-run the COSMOS monitor on the official protocol, so your first honest numbers
   exist before any deep-learning work.
5. Register a Space-Track account and download SpotGEO now (both are slow to obtain and needed
   later).
6. Re-verify every [M] and [AX] citation before it enters the manuscript.
