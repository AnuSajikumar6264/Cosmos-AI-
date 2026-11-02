# Candidate model survey for COSMOS-AI (state of 2026-10-27)

This document lists the models that are worth running on our three tasks, with versions,
licences, hardware needs and the published evidence behind them. It is a **pre-selection**. The
final choice for each task will be made from **our own measured results** on our own data, splits
and metrics (section 7), not from the leaderboard numbers quoted here.

- **(R)** Orbital lifetime / re-entry within a horizon, from a tabular object description (about
  20k–40k rows, roughly 10–20 numeric and categorical columns). Targets: log time-to-decay capped
  at the horizon (fully observed, so censoring is only administrative at the horizon), and
  survival-style targets.
- **(D)** Collision-risk forecasting from short CDM sequences (ESA Kelvins CAC). About 100
  features per CDM, 1–23 CDMs per event, target = final log10 collision probability.
- **(H)** Telemetry anomaly detection on NASA SMAP/MSL (telemanom) and ESA-ADB.

Nothing was installed for this survey. Facts come from PyPI JSON metadata, Hugging Face model
cards and API, GitHub READMEs, leaderboard CSVs and the papers. Source IDs in square brackets point
to the list in section 8. Where a fact could not be verified, the text says so.

## 0. How to read the tables

**Evidence type**

| Tag | Meaning |
|---|---|
| `dev` | Claim by the model's developers (paper, model card, README) |
| `LB-m` | Leaderboard where the maintainers run the models (TabArena/BeyondArena, TSB-AD). TabArena maintainers overlap with some model teams: TabArena's lead author co-authors the TabPFN-3.5 and Mitra-v1 reports [T5, T16b]. |
| `LB-s` | Leaderboard values submitted by the model developers (GIFT-Eval) |
| `ind` | Independent third-party study |
| `local` | Checked on this machine |

**Hardware column**

| Value | Meaning |
|---|---|
| CPU | Runs on CPU (as documented) |
| XPU-doc | The developer documents Intel XPU support |
| XPU-local | Already ran on an Intel XPU in this project |
| XPU-plausible | Pure PyTorch, and reading the deps/code found no CUDA-only import. **Not tested**; needs a smoke test. |
| XPU-no | Requires CUDA-only kernels or hardware, or the code rejects the `xpu` device |

**Our environment** (read on 2026-10-27, `.venv`) [L1]:

- Python 3.12.13, torch 2.13.0+xpu, triton-xpu 3.7.2
- numpy 2.3.5, pandas 3.0.3, scikit-learn 1.9.0, transformers 5.12.1, lightning 2.6.5
- Already installed: xgboost 3.4.1, lightgbm 4.7.0, chronos-forecasting 2.3.2

Several candidates pin older stacks (pandas < 3, torch < 2.12, numpy < 2) and would need their
own venv. This is flagged per row as **env-conflict**. Per the project notes, nothing goes into the
shared `sgl-xpu` env.

**Licence note.** Several top models have **non-commercial (NC)** weights. Academic,
non-commercial research use (a thesis, a journal paper) is generally what NC licences allow.
However, the Prior Labs FAQ says model outputs "can only be used for non-commercial evaluation and
research" [T3]. Timestral's licence withholds *all* rights from anyone whose company or employer
has more than USD 20 M monthly revenue, research included [F3]. If this work is done as an
employee of a company, check these licences before use.

---

## 1. Task R: tabular regression and survival (re-entry)

### 1.0 Candidate table for R (summary; details in 1.1–1.3)

Elo values are TabArena regression (13 datasets) / BeyondArena grouped regression (7) / BeyondArena
temporal regression (11) [T27, T28], all `LB-m`. A dash means not listed. Elo is relative to each
pool: RandomForest-default is 1000 in TabArena and XGBoost-default is 1000 in BeyondArena.

| Model | Version (date) | Licence (weights) | Hardware / XPU | Limits | Evidence: Elo TA / BA-grp / BA-tmp | URL |
|---|---|---|---|---|---|---|
| TabPFN-3.5 | tabpfn 9.2.0 (2026-10-12) | **NC** | CPU ≤ 5k rows by default; XPU-plausible | 1M rows, 20k cols | 2135 / 1133 / 1209 | [T2] |
| TabICLv2 | tabicl 2.2.0 (2026-09-02) | BSD-3 | CPU; **XPU-doc** | 300–100k rows, ≤ 2k feats | 1687 / 1036 / 848 | [T9] |
| Kumo-Tabular (L) | structured-data-models 0.2.1 (2026-10-14) | OpenMDW-1.1 | CPU; XPU-plausible | pretrained ≤ 60k rows, ≤ 100 cols | **2384** / 1279 / **1313** | [T12] |
| Mitra-v2 | HF 2026-09-03 (AutoGluon 1.6.3) | Apache-2.0 | reported recipe CUDA; env-conflict | not stated | 1977 / – / – | [T15] |
| LimiX-2 | HF 2026-09-15 | **NC** | CUDA builds documented; env-conflict | – | 2258 / 1120* / 1043* | [T18] |
| TabFM | tabfm 1.0.1 (2026-07-21) | **NC** | JAX/torch 2.12.1 pin; env-conflict | – | 1999 / 1079* / 980* | [T19] |
| TabDPT-1.3 | tabdpt 1.3.1 (2026-09-21) | Apache-2.0 | bf16 flash-attn default; unverified | retrieval context | 1665 / 767* / 854* | [T20] |
| TabPFNv2 | tabpfn (v2 weights) | Prior Labs Lic. (Apache + attribution) | as TabPFN | **10k rows, 500 cols** | 1198 (46 % imputed) | [T3] |
| CARTE | carte-ai 0.0.26 (2025-01-24) | MIT | needs PyG, fasttext | string-rich tables | – | [T31] |
| RealMLP | pytabkit 1.7.3 (2026-01-06) | Apache-2.0 | CPU; **XPU-no as packaged** | – | 1664 (t+e) / **1310** / 1206 | [T22] |
| TabM | tabm 0.0.3 (2025-08-14) | Apache-2.0 | PyTorch module, XPU-plausible | – | 1470 (t+e) / 1255 / 1070 | [T21] |
| FT-Transformer / ResNet | rtdl_revisiting_models 0.0.2 (2023-11-30) | Apache-2.0 | XPU-plausible | – | – | [T23] |
| LightGBM | 4.7.0 (2026-07-18), installed | MIT | CPU (GPU = source build) | – | 1508 (t+e) / 1269 / 1189 | [T25] |
| CatBoost | 1.2.10 (2026-02-18) | Apache-2.0 | CPU | Cox, SurvivalAft | 1513 (t+e) / 1234 / 1140 | [T26] |
| XGBoost | 3.4.1 (2026-08-15), installed | Apache-2.0 | CPU (SYCL = source build) | AFT, Cox | 1435 (t+e) / 1107 / 1145 | [T24] |
| scikit-survival (RSF, GBSA) | 0.28.0 (2026-07-05) | **GPL-3.0** | CPU | – | no tabular-board entry | [S1] |
| pycox (DeepHit, Cox-Time, DeepSurv) | 0.3.0 (2024-09-04) | BSD | XPU-plausible; low maintenance | – | methods papers [S8] | [S2] |
| torchsurv | 0.2.0 (2026-06-17) | MIT | XPU-plausible | losses and metrics | – | [S3] |
| SurvivalPFN | ckpt v0.1 (HF 2026-07-02) | Apache-2.0 | CPU / CUDA envs documented | – | `dev` only, 61 datasets | [S10] |
| SurvICL / SurvPFN / SIC | papers 2026 | code not found | – | – | `dev` only | [S9, S11, S12] |

`*` marks values where BeyondArena imputed part of the results. `t+e` means tuned and ensembled.

### 1.1 Tabular foundation models (in-context learners)

**TabPFN-3.5 / 3.5-Fast** (Prior Labs)
- Version: `tabpfn` 9.2.0 (2026-10-12); weights 2026-09-09 [T1, T4].
- Licence: code Apache-2.0; weights **NC** (TABPFN-3-5 License v1.0, login and acceptance required) [T2, T4].
- Size: checkpoint 876 MB (Fast: 334 MB) [T4].
- Python: ≥ 3.10 (3.10–3.14) [T1].
- Hardware: CPU is capped at **5,000 rows** by default (override with `TABPFN_ALLOW_CPU_LARGE_DATASET`) and is "slow". The `auto` device picks CUDA, then MPS, then CPU; it never picks XPU. An explicit `device="xpu"` passes `torch.device()` parsing, but this is **untested** (XPU-plausible) [T2, code read].
- Limits: up to 1,000,000 training rows and 20,000 columns (6,000 recommended) [T3].
- Evidence:
  - TabArena regression (13 datasets), Elo 2135 (+220/−118), rank 4 [T27, LB-m].
  - BeyondArena temporal regression (11 datasets) 1209, rank 3; grouped (7 datasets) 1133, rank 16 [T28, LB-m].
  - The report claims SOTA on TabArena [T5, dev].

**TabPFN-3** (Prior Labs)
- Version: same package; weights 2026-05 [T6].
- Licence: weights **NC** (TABPFN-3 License v1.0) [T3].
- Hardware: as for 3.5.
- Limits: 1 M rows, 2,000 columns [T3].
- Evidence: TabArena regression Elo 1802 [T27].

**TabPFN-2.5 / 2.6** (Prior Labs)
- Version: weights 2025-11 / 2026-03 [T7].
- Licence: weights **NC** (v1.1 / v1.0) [T3].
- Size: checkpoint about 41 MB [T7b].
- Hardware: as above (CPU limit 1,000 rows by default) [T2].
- Limits: the docs give 100,000 rows and 2,000 columns [T3]. The paper says "up to 50,000 data points and 2,000 features" [T7]; the two sources differ.
- Evidence: TabArena regression, TabPFN-2.6 Elo 1747; RealTabPFN-2.5 (default) 1553 [T27].

**TabPFNv2** (Prior Labs)
- Licence: Prior Labs License (Apache-2.0 plus attribution) [T2].
- Size: checkpoint 44 MB [T7b].
- Hardware: as above.
- Limits: **10,000 rows, 500 columns** [T3], which is below our row count.
- Evidence: TabArena regression Elo 1198, with 46 % of its results imputed [T27].

**TabICLv2** (Inria)
- Version: `tabicl` 2.2.0 (2026-09-02); checkpoints 2026-02-12 [T8, T11].
- Licence: **BSD-3-Clause** (code and weights) [T8, T11].
- Size: regressor checkpoint 114 MB [T11].
- Python: ≥ 3.10 (3.10–3.14) [T8].
- Hardware: CPU, plus CPU/disk offloading. **XPU-doc**: `device=None` auto-selects CUDA, then XPU, then MPS, then CPU [T9].
- Limits: benchmarked on 300–100,000 rows and ≤ 2,000 features; scales to about 500k rows with offloading, "though its accuracy may degrade" [T9].
- Evidence:
  - TabArena regression Elo 1687 (+215/−130), rank 18 [T27].
  - BeyondArena grouped regression 1036, temporal regression 848 (weak off-IID) [T28].
  - The ICML 2026 paper claims it beats tuned RealTabPFN-2.5 [T10, dev].

**Kumo-Tabular S/M/L** (NVIDIA)
- Version: `structured-data-models` 0.2.1 (2026-10-14); weights 2026-09-01 [T12, T14].
- Licence: code Apache-2.0; weights **OpenMDW-1.1** (permissive, commercial use allowed) [T12, T13].
- Size: 28M–215M parameters [T13]; checkpoints 114 / 250 / 863 MB [T12].
- Python: ≥ 3.11 [T14].
- Hardware: the card's example falls back to CPU. The library is "GPU-native" with optional cuDF. No triton or flash-attn import was found in the wheel: **XPU-plausible**, untested [T12, T14, code read].
- Limits: pretraining contexts went up to 60,000 rows and ≤ 100 columns [T13]. Our tables fit.
- Evidence:
  - TabArena regression Elo **2384** (+265/−183), rank 1 [T27].
  - BeyondArena temporal 1313 (rank 1), grouped 1279 (rank 2) [T28].
  - Released 2026-09. No paper was found in this survey; the evidence is the vendor blog [T13, dev] and TabArena.

**Mitra-v2** (Amazon)
- Version: weights 2026-09-03; used through AutoGluon 1.6.3 [T15, T17].
- Licence: Apache-2.0 [T15].
- Size: 76.7M parameters [T15].
- Python: AutoGluon needs 3.10–3.13 [T17].
- Hardware: the reported fine-tune and bagging recipe "requires a CUDA GPU". Plain AutoGluon inference is documented without that statement [T15]. **env-conflict**: AutoGluon pins pandas < 2.4 [T17].
- Limits: "10x longer context, 3x features" than v1, with no absolute numbers given [T15].
- Evidence: TabArena regression Elo 1977 [T27]. The report claims parity with TabFM [T16, dev].

**LimiX-2** (Stability AI and Tsinghua)
- Version: weights 2026-09-15 [T18].
- Licence: **NC** (StableAI LimiX NC v1.0) [T18].
- Size: checkpoint 1.63 GB [T18].
- Python: ≥ 3.12 [T18].
- Hardware: pins `torch==2.9.1`; the CUDA-matched build and flash-attn are "optional" but documented only for CUDA [T18]. **env-conflict**.
- Evidence: TabArena regression Elo 2258, rank 2 [T27].

**TabFM** (Google)
- Version: `tabfm` 1.0.1 (2026-07-21) [T19].
- Licence: code Apache-2.0; weights **NC** [T19].
- Size: 1.6B parameters, per the Mitra-v2 report [T16].
- Python: ≥ 3.11.
- Hardware: JAX (CPU or CUDA) or PyTorch "specifically torch==2.12.1+cpu or a GPU version" [T19]. **env-conflict**.
- Evidence: TabArena regression Elo 1999 [T27].

**TabDPT-1.3** (Layer 6)
- Version: `tabdpt` 1.3.1 (2026-09-21) [T20].
- Licence: Apache-2.0 [T20].
- Size: checkpoint 252 MB [T20].
- Python: ≥ 3.10 on PyPI; the HF card says 3.10/3.11 [T20].
- Hardware: flash attention in bf16 and compile are on by default; needs faiss-cpu. CPU/XPU are unverified [T20].
- Limits: retrieval-based context [T20].
- Evidence:
  - TabArena regression Elo 1665 [T27].
  - BeyondArena grouped 767, temporal 854, with imputed entries [T28].

**CARTE** (Inria)
- Version: `carte-ai` 0.0.26 (2025-01-24) [T31].
- Licence: MIT [T31].
- Python: ≥ 3.10.12 (3.10–3.12).
- Hardware: needs torch-geometric and fasttext [T31].
- Limits: aimed at string-rich tables [T31b].
- Evidence: not on the TabArena list read here.

Other open-licence TFMs on TabArena that are not shortlisted (regression Elo [T27]):

- Xiaomi-TabLDM 1779, Apache-2.0
- Nori-30M 1773, Apache-2.0
- TabSwift 1419
- iLTM, tuned and ensembled: 1460
- SAP-RPT-OSS 1279; "weights stated research-only"

These carry NC weights: EXAONE-Tabular (1892) and Causilo (2036).

### 1.2 Deep tabular models and GBDTs

**RealMLP**
- Version: `pytabkit` 1.7.3 (2026-01-06) [T22].
- Licence: Apache-2.0.
- Python: ≥ 3.9 (3.9–3.13).
- Hardware: CPU. **XPU-no as packaged**: `_allowed_device_names()` returns cpu/cuda/mps, and the Lightning device mapper raises an error for any other device (code read) [T22].
- Limits: none relevant.
- Evidence:
  - TabArena regression, tuned and ensembled: 1664; default: 1236 [T27].
  - BeyondArena grouped regression, tuned and ensembled: **1310, rank 1**; temporal: 1206 [T28].

**TabM**
- Version: `tabm` 0.0.3 (2025-08-14), or through pytabkit [T21].
- Licence: Apache-2.0.
- Python: ≥ 3.9.
- Hardware: the `tabm` package is a plain PyTorch module (you write the training loop), so it is **XPU-plausible** (this project already trains PyTorch models on XPU). The pytabkit wrapper has the same cpu/cuda/mps restriction as RealMLP.
- Evidence:
  - TabArena regression, tuned and ensembled: 1470; default: 1297 [T27].
  - BeyondArena grouped 1255, temporal 1070 [T28].
  - ICLR 2025 paper [T21, dev].

**FT-Transformer, ResNet**
- Version: `rtdl_revisiting_models` 0.0.2 (2023-11-30). The legacy `rtdl` 0.0.13 pins torch < 2 [T23].
- Licence: Apache-2.0.
- Python: ≥ 3.8.
- Hardware: plain PyTorch modules, XPU-plausible.
- Evidence: the NeurIPS 2021 paper [T23, dev]. The TabM paper reports MLP-family models beating attention-based ones [T21, dev]. Neither model is on the current TabArena list.

**LightGBM**
- Version: 4.7.0 (2026-07-18), installed [T25].
- Licence: MIT.
- Python: ≥ 3.10.
- Hardware: CPU. The GPU (OpenCL) build is source-only [T25].
- Survival support: none native.
- Evidence:
  - TabArena regression, tuned and ensembled: 1508; default: 1224 [T27].
  - BeyondArena grouped 1269 (rank 3), temporal 1189 [T28].

**CatBoost**
- Version: 1.2.10 (2026-02-18); cp312 wheels exist [T26].
- Licence: Apache-2.0.
- Hardware: CPU. The GPU mode was not evaluated here; there is no evidence of Intel GPU support.
- Survival support: native `Cox` and `SurvivalAft` objectives [T26].
- Evidence:
  - TabArena regression, tuned and ensembled: 1513; default: 1404 [T27].
  - BeyondArena grouped 1234, temporal 1140 [T28].

**XGBoost**
- Version: 3.4.1 (2026-08-15), installed [T24].
- Licence: Apache-2.0.
- Python: ≥ 3.12.
- Hardware: CPU. The GPU path is CUDA. A SYCL (Intel GPU) plugin exists but must be **built from source** with oneAPI [T24].
- Survival support: `survival:aft` with interval-censored labels, and `survival:cox` [T24].
- Evidence:
  - TabArena regression, tuned and ensembled: 1435; default: 1201 [T27].
  - BeyondArena grouped 1107, temporal 1145 [T28].

**Context from older and independent benchmarks**

- Grinsztajn et al. (2022, 45 datasets): tree models were still SOTA on about 10k-sample data
  [T30]. This predates the TFMs.
- TALENT (300+ datasets, v4 of 2025-11) [T29]:
  - GBDTs "remain very strong baselines";
  - pretrained models "now match or surpass them on many tasks";
  - ensembling helps both families.
- TabArena itself is small for regression: 13 datasets overall and 5 in the medium-size slice.
  The Elo 95 % CIs span hundreds of points (Kumo-Tabular +265/−183, TabICLv2 +215/−130) [T27].
- **The split type changes the ranking.** Under BeyondArena's grouped splits:
  - tuned and ensembled RealMLP beats every TFM, Kumo-Tabular included;
  - tuned LightGBM, TabM and CatBoost beat every TFM except the Kumo-Tabular sizes;
  - TabICLv2 falls to roughly the level of default XGBoost and LightGBM [T28].

  Our re-entry data have object families (launch, breakup) and an epoch axis, so grouped and
  temporal splits are the relevant ones.
- No public benchmark for ML orbital-lifetime regression from catalogue features was found in the
  sources checked. All evidence here is from general tabular benchmarks.

### 1.3 Survival models

**XGBoost AFT / Cox**
- Version: 3.4.1 (installed) [T24].
- Licence: Apache-2.0.
- Hardware: CPU.
- Maintenance: active.
- Notes: interval labels `[lower, upper]`. Objects still in orbit at the horizon are encoded as `[H, +inf)` [T24].

**CatBoost Cox / SurvivalAft**
- Version: 1.2.10 [T26].
- Licence: Apache-2.0.
- Hardware: CPU.
- Maintenance: active.

**scikit-survival** (RSF, ExtraSurvivalTrees, GBSA, CoxNet)
- Version: 0.28.0 (2026-07-05) [S1].
- Licence: **GPL-3.0-or-later** (copyleft if redistributed) [S1].
- Python: ≥ 3.11.
- Hardware: CPU.
- Maintenance: active.
- Compatibility: requires scikit-learn ≥ 1.9, < 1.10, which our 1.9.0 satisfies [S1]. The estimators listed exist in the API docs [S1].

**pycox** (DeepSurv/CoxPH, DeepHit, Cox-Time, Cox-CC, LogisticHazard, PMF, MTLR, PC-Hazard, BCESurv)
- Version: 0.3.0 (2024-09-04); depends on torchtuples 0.2.2 (2021-08-31) [S2].
- Licence: BSD.
- Hardware: plain PyTorch, XPU-plausible.
- Maintenance: **low** (one release in about 4 years).
- Compatibility with numpy 2 / pandas 3 is unverified.
- Model list from the README [S2]; methods from [S8].

**torchsurv** (Cox and Weibull losses, survival metrics)
- Version: 0.2.0 (2026-06-17) [S3].
- Licence: MIT.
- Python: classifiers up to 3.12; torch ≥ 2.8.
- Hardware: plain PyTorch, XPU-plausible.
- Maintenance: active.
- Use: losses and metrics for our own DeepSurv/DeepHit-style heads.

**SurvivalPFN** (survival foundation model)
- Version: checkpoint v0.1, HF 2026-07-02 [S10].
- Licence: Apache-2.0 (code and checkpoint) [S10].
- Size: checkpoint 75 MB.
- Hardware: separate CPU and CUDA conda envs are documented [S10].
- Maintenance: new.
- Evidence: the paper claims strong results on 61 datasets [S10, dev]. No independent benchmark found.

**SurvICL**
- Licence and code: **not found**. A BSD-3 HF repo `taltstidl/tabicl-survival` exists with an empty card; its relation to SurvICL is unverified [S9].
- Evidence: the paper (2026-10-15, Inria) claims it wins 82.9 % of pairwise integrated-Brier comparisons on 63 held-out datasets [S9, dev].

**SurvPFN and Survival In-Context (SIC)**
- Code was not found (SIC says "upon publication") [S11, S12].
- Evidence: workshop paper and preprint [dev].

**SurvFM** (RMST targets for any TFM)
- Method paper [S13].
- Hardware and licence depend on the TFM it wraps.

**Excluded on compatibility or maintenance**

- lifelines 0.30.3: pins pandas < 3 [S4].
- xgbse 0.3.3: pins xgboost < 3 and numpy < 2 [S6].
- auton-survival 0.1.0: torch < 2 [S5].
- pysurvival 0.1.2 (2019) [S7].

Our censoring is administrative only: every object either decays before the horizon H or is
known to survive past it. So the survival targets can also be expressed as:

- regression on `min(T, H)`, or the RMST up to H (the SurvFM idea [S13]);
- classification of `T < H` at several horizons.

The tabular foundation models above can handle both forms directly.

### 1.4 Recommendation for R (run these; ranked by expected value per effort)

1. **GBDT block: LightGBM 4.7.0, CatBoost 1.2.10, XGBoost 3.4.1**, all tuned (CPU).
   - Tuned, they stay near the top under grouped and temporal splits, where several TFMs drop
     [T28].
   - LightGBM and XGBoost are installed.
   - XGBoost-AFT and CatBoost-SurvivalAft/Cox cover the survival targets natively.
2. **RealMLP** (pytabkit 1.7.3, CPU only on our machine) **and/or TabM** (`tabm` 0.0.3, plain
   PyTorch on XPU). These are the best non-foundation deep models; tuned and ensembled RealMLP
   tops BeyondArena grouped regression [T28].
3. **TabICLv2** (`tabicl` 2.2.0, BSD-3). This is the only top-tier TFM with permissive weights
   **and** documented XPU support [T9]. Its off-IID weakness [T28] is exactly what our grouped
   split will test.
4. **Kumo-Tabular** (`structured-data-models` 0.2.1, OpenMDW-1.1).
   - It is first on TabArena regression and BeyondArena temporal, second on grouped [T27, T28], under a permissive licence.
   - Risks: XPU is untested; it was released 2026-09 and its evidence is the vendor blog plus TabArena only.
5. **TabPFN-3.5** (`tabpfn` 9.2.0), **only if the NC licence is acceptable** for the user's
   situation. It is strong on IID and temporal data [T27, T28]. CPU is capped at 5,000 rows, so
   XPU must pass a smoke test.

Survival arm, run alongside the regression targets:

- XGBoost-AFT and CatBoost-SurvivalAft;
- scikit-survival RSF and GBSA (GPL-3.0; fine for research code);
- one neural survival model: DeepHit or Cox-Time via pycox 0.3.0 if it imports cleanly in our
  stack, otherwise our own head with torchsurv 0.2.0 losses;
- SurvivalPFN as an **exploratory** foundation-model arm only.

**Excluded for R, and why**

- LimiX-2: NC weights; torch==2.9.1 and CUDA/flash-attn.
- TabFM: NC weights; 1.6B parameters; pinned torch 2.12.1.
- EXAONE-Tabular, Causilo: NC weights.
- Mitra-v2: Apache-2.0, but AutoGluon pins pandas < 2.4 and the reported recipe needs CUDA. Deferred to a separate env if time allows.
- TabPFNv2: its 10k-row and 500-column limits are below our data.
- TabDPT: weak under grouped and temporal splits [T28]; bf16 flash-attention default; faiss dependency.
- CARTE: built for string-heavy tables; last release 2025-01; needs fasttext and PyG.
- rtdl (legacy): torch < 2. FT-Transformer is optional only as a legacy baseline.
- pytorch-tabular: pandas < 3 and torchmetrics < 1.9 pins [T32].
- SurvICL, SurvPFN, SIC: no verified code or weights.

---

## 2. Task D: collision risk from CDM sequences (Kelvins CAC)

### 2.1 What the competition showed (Uriot et al., Astrodynamics 2022 [C1])

**Data**

- 15,321 events, 199,082 CDMs, 103 attributes.
- 13 CDMs per event on average (min 1, max 23).
- High-risk events: 30 above 1e-4, 131 above 1e-5, 515 above 1e-6.
- The train/test split was **not random**: operationally useful events were favoured for the test
  set.

**Metric**

- Score = MSE_HR / F2.
- The high-risk class is r ≥ −6 (log10 Pc).
- MSE is computed over true high-risk events only, with predictions clipped at −6.001.

**Results**

- The naive **latest-risk prediction (LRP) baseline** scores **0.694**. Only **12 of 97 teams**
  beat it.
- Top-3 scores: sesc 0.556, dietmarw 0.571, Magpies 0.585.

**Approaches**

- **sesc (1st place).**
  - Tried time-series features, a genetic-programming AutoML pipeline and random forests. These "performed radically better on the training set" than on test.
  - Won with a hand-made **cascade of thresholds on the latest risk**, plus rules on object type, span and miss distance.
- **Magpies (3rd place).** A **siamese Manhattan-LSTM** that flags "anomalous" low-risk events from 10 selected or derived features.
- **Survey of 22 teams.**
  - 65 % "framed the learning problem as a static one" (summaries or the last CDM).
  - 30 % probed the test set by trial and error.

**Post-competition study**

- Of 5,000 LightGBM models trained on the official split to predict the change from the last
  risk, only **one (0.02 %)** beat LRP on both train and test.
- On the official split, train and test loss were essentially uncorrelated.
- The authors also ran 10,000 simulated competitions on random stratified re-splits [C1]:
  - at test size 0.2, **67.55 %** of models beat LRP on 1/F2 in both sets;
  - the mean test gain on 1/F2 was only **1.12 %**;
  - few models ever beat LRP on MSE_HR.

  So the official split is unusually hard, and learned models add little beyond LRP even on
  random splits.

**Later work on the Kelvins data**

- Pinto et al. (2020) [C3], a Bayesian (MC-dropout) 2×256 LSTM:
  - predicts the **next CDM's** 52 features, a different task from the final risk;
  - next-step MSE 0.1419 against 0.2433 for persistence, on a random 15 % split, not the official one.
- Tok et al. (2026) [C4], a TCN-Transformer:
  - uses its **own** CDM set of 1,075 events, not Kelvins;
  - reports F2 0.79. Not comparable.
- Catulo et al. (2023) [C5]: a Bayesian HMM on the risk series alone, reported to beat the naive
  forecast on a Kelvins-derived split. Its table did not extract reliably, so no numbers are
  quoted here.

### 2.2 Candidates for D

**Candidate table for D (summary)**

| Model | Version (date) | Licence | Hardware / XPU | Limits | Evidence on CDMs | URL |
|---|---|---|---|---|---|---|
| LRP (latest risk) and rule cascade | own code | – | CPU | – | LRP 0.694; sesc rules 0.556 (official test) [C1, `ind`] | [C1] |
| GBDT on last-CDM + summaries (Δ target) | LightGBM 4.7.0 / XGBoost 3.4.1 / CatBoost 1.2.10 | MIT / Apache-2.0 | CPU | – | official split: 1 of 5,000 LightGBM models beat LRP; random splits: about 1 % mean 1/F2 gain [C1] | [T24–T26] |
| TabICLv2 / Kumo-Tabular / TabPFN-3.5 on the same features | see table 1.0 | BSD-3 / OpenMDW / NC | see table 1.0 | about 8–10k events fits | none (gap) | [T9, T12, T2] |
| GRU / LSTM with Δt and time-to-TCA (GRU-D-style) | own PyTorch | – | XPU-plausible | ≤ 23 steps | Manhattan-LSTM placed 3rd [C1]; next-CDM LSTM [C3] | [C6] |
| mTAN / SeFT-style continuous-time attention | own PyTorch | – | XPU-plausible | – | clinical data only (`dev`) | [C6] |
| Neural CDE | torchcde 0.2.5 (2021-10-17) | Apache-2.0 | CPU / XPU-plausible | – | none; unmaintained | [C6] |
| Raindrop | paper code | – | – | built for many sensors | none | [C6] |
| Time-series FMs (Chronos-2 etc.) | section 3 | section 3 | section 3 | regular-grid forecasting | none | [F1] |

**LRP and rule-cascade baselines**
- Version: our own code.
- Hardware: CPU.
- Evidence: the LRP baseline scores 0.694 on the official test set; sesc's rule cascade scores 0.556 [C1, ind].
- Notes: mandatory reference.

**GBDT on last-CDM plus sequence summaries, target Δ = r − r₋₂**
- Version: LightGBM 4.7.0 / XGBoost 3.4.1 / CatBoost 1.2.10.
- Licence: MIT / Apache-2.0.
- Hardware: CPU.
- Evidence: the CAC post-analysis (5,000 LightGBM models) [C1].
- Notes: this is the format 65 % of teams used.

**TabICLv2 / Kumo-Tabular / TabPFN-3.5 on the same features**
- Version and licence: see 1.1.
- Hardware: see 1.1.
- Limits: about 8–10k training events, inside all their row limits.
- Evidence: none on CAC (gap). General evidence only [T27].

**GRU / LSTM over the CDM sequence, with Δt and time-to-TCA as inputs (GRU-D-style masking)**
- Version: our PyTorch code.
- Hardware: XPU-plausible (plain PyTorch).
- Evidence:
  - Magpies' Manhattan-LSTM placed 3rd [C1].
  - Pinto et al.'s LSTM, on the next-CDM task [C3].
  - GRU-D [C6].
- Notes: about 23 steps maximum and few positives, so overfitting is the main risk.

**Small attention or set model with continuous-time encoding (mTAN- or SeFT-style)**
- Version: our PyTorch code.
- Hardware: XPU-plausible.
- Evidence: mTAN (ICLR 2021), SeFT (ICML 2020) on clinical irregular series [C6, dev]. No CDM evidence.

**Neural CDE**
- Version: `torchcde` 0.2.5 (2021-10-17); `torchdiffeq` 0.2.5 (2024-11-21) [C6].
- Licence: Apache-2.0 / MIT.
- Hardware: CPU/XPU-plausible.
- Evidence: paper only [C6].
- Notes: unmaintained. Overkill for ≤ 23 points.

**Raindrop** (graph-guided irregular series)
- Evidence: paper only [C6].
- Notes: designed for many asynchronous sensors, which does not match CDMs.

**Time-series FMs (Chronos-2 etc.)**
- Hardware: see section 3.
- Evidence: none on CDMs.
- Notes: 1–23 irregular points with a target at TCA. This is not the regular-grid forecasting these models are built for.

### 2.3 Recommendation for D

Run:

1. **LRP and the sesc-style rule cascade.** These are the bars to beat. Report the official CAC
   metric plus a **grouped and stratified repeated-split** evaluation (the paper's re-split
   analysis shows single-split rankings are unreliable [C1]).
2. **GBDT on engineered static features with a Δ-from-last-risk target** (LightGBM, XGBoost,
   CatBoost).
3. **A tabular FM on the same features** (TabICLv2 first; Kumo-Tabular or TabPFN-3.5 if the
   licence and XPU checks pass).
4. **A GRU sequence model** with time deltas.
5. **One small continuous-time attention model.**

Item 5 is the only "transformer for irregular time series" worth its cost here. Expect gains over
items 1–2 to be small or nil. On this dataset, learned models rarely beat LRP on the official
split, and gain about 1 % on random splits [C1].

**Excluded for D:**
- Time-series foundation models: wrong problem shape (section 2.2).
- Neural CDE (`torchcde`, unmaintained since 2021) and Raindrop: complexity with no CDM evidence.

---

## 3. Time-series foundation models (input to H; considered and rejected for D)

GIFT-Eval columns are **our recomputation** from the leaderboard's own result files [F11]:

- MASE and CRPS are each normalised by Seasonal Naive per configuration, then geometric-meaned
  over all **97 configurations**. Lower is better; Seasonal Naive = 1.000. This is the same method
  as the Space's `src/utils.py`.
- 156 models have complete results.
- Spot check: our values match the five models in Timestral's card table [F3].
- GIFT-Eval numbers are **submitted by model developers** (`LB-s`), with a self-declared
  "test-data leakage" flag.
- GIFT-Eval scores forecasting, **not** anomaly detection. It is weak evidence for task H.

**Candidate table: time-series FMs (summary; details below)**

| Model | Version (date) | Licence (weights) | Params | Hardware / XPU | Context, MV / covariates | Evidence | URL |
|---|---|---|---|---|---|---|---|
| Chronos-2 | chronos-forecasting 2.3.2 (2026-09-08); HF 2025-10-30 | Apache-2.0 | 120M | CPU; **XPU-local** (Bolt) | 8,192; MV and covariates | GIFT 0.698 / 0.485 | [F1] |
| TimesFM-2.5 | timesfm 3.0.2 (2026-09-09); HF 2025-09-02 | Apache-2.0 | 231M | XPU-plausible; XReg needs jax[cuda] | 16k; covariates via XReg | GIFT 0.705 / 0.490 | [F2] |
| TimesFM-3 | same; HF 2026-08-24 | **NC** | 331M | XPU-plausible | MV and covariates | GIFT 0.667 / 0.456 | [F2] |
| Timestral-209M / 28M | GitHub; HF 2026-10-14 | Mod-MIT, **revenue cap** | 209M / 28M | CPU; XPU-plausible | 32,768; MV and covariates | GIFT 0.658 / 0.446 (209M) | [F3] |
| TimeH2O-v1 | HF 2026-10-22, gated | **NC** | 109M | – | – | GIFT 0.654 / 0.441 | [F10] |
| t0-gamma | tfc-t0 0.6.0 (2026-10-13) | Apache-2.0 | 256M | XPU-plausible | MV and covariates | GIFT 0.661 / 0.452 | [F6] |
| Toto-2.0 (22m / 313m) | toto-2 2.0.0 (2026-06-04) | Apache-2.0 | 4M–2.5B | XPU-plausible; env-conflict | variable; MV | GIFT 0.703 / 0.481 (313m) | [F5] |
| TiRex-2 | tirex-2 0.4.0 (2026-10-20) | Apache-2.0 (+ NXAI deps) | 82.5M | CPU; Triton, XPU unverified | MV, covariates, streaming | GIFT 0.697 / 0.478 (ZS) | [F4] |
| Moirai-2 small | uni2ts 2.0.0 (2025-11-04) | **CC-BY-NC-4.0** | 11M | env-conflict (torch < 2.5) | – | GIFT 0.728 / 0.516 | [F7] |
| TTM-R2 / R3 | granite-tsfm 0.3.11 (2026-10-21) | Apache-2.0 / **CC-BY-NC-SA** | 0.8M / 1.4M | CPU; env-conflict | channel mix, exogenous | GIFT 1.020 / 0.873 (R2, leak flag) | [F8] |
| Sundial base | HF 2025-05-13 | Apache-2.0 | 128M | env-conflict (transformers 4.40.1) | 2,880; univariate | GIFT 0.750 / 0.559 | [F9] |
| TSPulse r1 (AD) | granite-tsfm 0.3.11; HF 2025-06-03 | Apache-2.0 | 1M | CPU; env-conflict | needs ≥ about 1.5–2k points | TSB-AD ZS 0.48 / 0.36 | [F14] |
| Time-RCD (AD) | GitHub; HF 2025-10-18 | Apache-2.0 | 37M | XPU-plausible | univariate and MV | TSB-AD 0.52 / 0.32 | [F15] |
| MOMENT-1-large (AD) | momentfm 0.1.4 (2025-03-19) | MIT | 346M | env-conflict | – | TSB-AD-U 0.38 (ZS); SMAP/MSL contamination | [F13] |
| Kumo-Anomaly | HF (release 2026-06-29) | Apache-2.0 | 8M | **XPU-no**, Ampere/Hopper only | – | none public | [F12] |
| CHARM | API only | weights not released | 63M | remote API | 8,192 | TSB-AD 0.68 / 0.54 (kNN) | [F16] |

GIFT values are MASE / CRPS relative to Seasonal Naive (lower is better; `LB-s`). TSB-AD values are
VUS-PR univariate / multivariate (`LB-m`).

**Chronos-2** (Amazon)
- Version: `chronos-forecasting` 2.3.2 (2026-09-08), installed; weights 2025-10-30 [F1].
- Licence: Apache-2.0.
- Size: 119.5M parameters.
- Python: ≥ 3.10.
- Hardware: CPU and GPU documented. **XPU-local** for Chronos-Bolt in this project (`device: xpu:2`, config.yaml in [A9]); Chronos-2 itself has not run here.
- Context: up to **8,192**.
- Multivariate and covariates: **Yes**, plus past and future covariates [F1].
- Anomaly-detection use: forecast residual or quantile exceedance (our existing `forecasting.py`).
- GIFT-Eval: MASE 0.698, CRPS 0.485.

**Chronos-Bolt base** (Amazon)
- Version: weights 2024-11-25 [F17].
- Licence: Apache-2.0.
- Size: 205M parameters.
- Hardware: XPU-local (Bolt-small ran on `xpu:2` in this project [A9]).
- Context: 2,048.
- Multivariate and covariates: no.
- Anomaly-detection use: residuals.
- GIFT-Eval: MASE 0.808, CRPS 0.574; leakage flag **Yes**.

**TimesFM-2.5** (Google)
- Version: `timesfm` 3.0.2 (2026-09-09); weights 2025-09-02 [F2].
- Licence: Apache-2.0.
- Size: 231M parameters.
- Python: ≥ 3.10.
- Hardware: `timesfm[torch]`, XPU-plausible. **XReg covariates need the `jax[cuda]` extra** [F2].
- Context: up to **16k**; horizon up to 1k.
- Multivariate and covariates: covariates via XReg.
- Anomaly-detection use: residuals.
- GIFT-Eval: MASE 0.705, CRPS 0.490.

**TimesFM-3** (Google)
- Version: same package; weights 2026-08-24 [F2].
- Licence: weights **NC** (non-commercial, non-production) [F2].
- Size: 331M parameters.
- Hardware: as for 2.5, XPU-plausible.
- Multivariate and covariates: native multivariate and covariates.
- Anomaly-detection use: residuals.
- GIFT-Eval: MASE 0.667, CRPS 0.456. The README claims "Rank #1 among all foundation models" [dev]. On our recomputation, several later or other entries score lower on both metrics (Timestral-209M/123M, TimeH2O-v1, t0-gamma).

**Timestral 5M / 28M / 123M / 209M** (Mistral AI)
- Version: install from GitHub; weights 2026-10-14 [F3].
- Licence: **Modified MIT, with no rights if the user's company or employer has more than USD 20 M monthly revenue** [F3].
- Size: 5M–209M parameters.
- Python: ≥ 3.10; torch ≥ 2.4.
- Hardware: CPU or GPU; "no GPU needed"; XPU-plausible.
- Context: **32,768**; horizon 1,024 per forward pass.
- Multivariate and covariates: Yes.
- Anomaly-detection use: residuals.
- GIFT-Eval: 209M scores MASE 0.658, CRPS 0.446; 28M scores 0.678 / 0.461.

**TimeH2O-v1** (H2O.ai)
- Version: weights 2026-10-22, gated [F10].
- Licence: **NC**.
- Size: 109M parameters.
- GIFT-Eval: MASE 0.654, CRPS 0.441.

**t0-gamma** (The Forecasting Company)
- Version: `tfc-t0` 0.6.0 (2026-10-13); weights 2026-10-12 [F6].
- Licence: Apache-2.0.
- Size: about 256M parameters.
- Python: ≥ 3.10.
- Hardware: `tfc-t0[torch]`, XPU-plausible.
- Multivariate and covariates: Yes.
- Anomaly-detection use: residuals.
- GIFT-Eval: MASE 0.661, CRPS 0.452.

**Toto-2.0 4m–2.5B** (Datadog)
- Version: `toto-2` 2.0.0 (2026-06-04); weights 2026-04 [F5].
- Licence: Apache-2.0.
- Size: 4M / 22M / 313M / 1B / 2.5B parameters.
- Python: ≥ 3.12.
- Hardware: CPU or CUDA in the examples; XPU-plausible. **env-conflict**: gluonts ≥ 0.16 pins pandas < 3 [F5].
- Context: variable (decoder-only).
- Multivariate and covariates: multivariate (alternating time and variate attention).
- Anomaly-detection use: residuals; built for observability metrics.
- GIFT-Eval: 313M scores MASE 0.703, CRPS 0.481; 22M scores 0.719 / 0.496.

**TiRex-2** (NX-AI)
- Version: `tirex-2` 0.4.0 (2026-10-20); weights 2026-06-16 [F4].
- Licence: weights Apache-2.0. The dependency `mlstm_kernels` uses the NXAI Community licence: a commercial licence is needed only above €100 M revenue **and** inside a commercial product [F4].
- Size: 82.5M parameters (38.4M univariate path).
- Python: ≥ 3.11.
- Hardware: a CPU env is documented. Needs `flashrnn` (Triton). XPU is unverified; installing it may pull CUDA `triton` over our `triton-xpu`.
- Multivariate and covariates: Yes, and **streaming** updates.
- Anomaly-detection use: residuals, streamed.
- GIFT-Eval: zero-shot variant MASE 0.697, CRPS 0.478.

**Moirai-2 small** (Salesforce)
- Version: `uni2ts` 2.0.0 (2025-11-04); weights 2025-08-06 [F7].
- Licence: **CC-BY-NC-4.0**.
- Size: 11.4M parameters.
- Hardware: **env-conflict**: torch < 2.5, numpy ~1.26 [F7].
- GIFT-Eval: MASE 0.728, CRPS 0.516.

**TTM-R2** (IBM)
- Version: `granite-tsfm` 0.3.11 (2026-10-21) [F8].
- Licence: Apache-2.0.
- Size: 0.8M parameters.
- Python: 3.11–3.13.
- Hardware: CPU. **env-conflict**: torch < 2.12, sklearn < 1.8 [F8].
- Multivariate and covariates: channel mixing and exogenous inputs (per the IBM card) [F8].
- Anomaly-detection use: residuals.
- GIFT-Eval: pretrained MASE 1.020, CRPS 0.873; leakage flag Yes.

**TTM-R3** (IBM)
- Version: weights 2026-02-09 [F8].
- Licence: **CC-BY-NC-SA-4.0**.
- Size: 1.4M parameters.
- Hardware: as TTM-R2.
- GIFT-Eval: pretrained MASE 0.724, CRPS 0.520.

**Sundial base** (Tsinghua)
- Version: weights 2025-05-13 [F9].
- Licence: Apache-2.0.
- Size: 128M parameters.
- Hardware: recommends transformers==4.40.1 and Python 3.10 (**env-conflict**: we have transformers 5.12.1).
- Context: 2,880.
- Multivariate and covariates: no.
- GIFT-Eval: MASE 0.750, CRPS 0.559.

Seasonal Naive, the reference: GIFT-Eval MASE 1.000, CRPS 1.000.

**Anomaly-detection-native foundation models** (evidence: TSB-AD VUS-PR, maintainer-run [A6])

**TSPulse r1** (IBM)
- Version: `granite-tsfm` 0.3.11; weights 2025-06-03 [F14].
- Licence: Apache-2.0.
- Size: 1.08M parameters.
- Hardware: "GPU-free inference" (CPU). **env-conflict**: torch < 2.12.
- Limits: anomaly detection needs series of about ≥ 1,536–2,048 points [F14].
- TSB-AD VUS-PR (univariate / multivariate): zero-shot 0.48 / 0.36; fine-tuned 0.55 / 0.39 [A6].

**Time-RCD** (THU)
- Version: install from GitHub; weights 2025-10-18, updated 2026-08 [F15].
- Licence: Apache-2.0.
- Size: 37M parameters.
- Python: ≥ 3.10 (classifiers up to 3.12).
- Hardware: deps are torch, einops and jaxtyping only; XPU-plausible.
- Supports univariate and multivariate scoring [F15].
- TSB-AD VUS-PR: 0.52 / 0.32 [A6].

**MOMENT-1-large** (CMU)
- Version: `momentfm` 0.1.4 (2025-03-19) [F13].
- Licence: MIT.
- Size: 346M parameters.
- Hardware: **env-conflict**: transformers==4.33.3, numpy==1.25.2 [F13].
- Reconstruction-based anomaly detection.
- **Pretraining used the TSB-UAD training splits, which include NASA MSL/SMAP** [F13].
- TSB-AD VUS-PR (univariate): zero-shot 0.38; fine-tuned 0.39 [A6].

**Kumo-Anomaly** (NVIDIA)
- Version: the card gives the release date as 2026-06-29; the HF repo was created 2026-04-21 [F12].
- Licence: Apache-2.0.
- Size: 8M parameters.
- Hardware: **XPU-no and CPU not documented**. Supported: NVIDIA Ampere and Hopper, with Transformer Engine [F12].
- Evidence: no TSB-AD entry; its own evaluation sets only.

**CHARM** (C3 AI)
- Version: SDK `c3-charm` 0.1.18 [F16].
- Licence: SDK Apache-2.0; **weights not released, API only** [F16].
- Size: 63M parameters.
- Hardware: remote API.
- TSB-AD VUS-PR: kNN variant 0.68 / 0.54; zero-shot 0.61 / 0.46 [A6].

---

## 4. Task H: telemetry anomaly detection (SMAP/MSL, ESA-ADB)

### 4.1 What the critiques establish (with numbers)

**Wu & Keogh (TKDE 2022) [A2]**

- NASA examples are largely trivial. In about half, the anomaly is "many orders of magnitude"
  different. Others are frozen signals, catchable by `diff(diff(TS)) == 0`. Only about 10 % are
  "mildly challenging".
- Unrealistic anomaly density: in D-2, M-1 and M-2, more than half the test data is one labelled
  region.
- Mislabelled ground truth: frozen snippets left unlabelled.
- Run-to-failure bias.

**Kim et al. (AAAI 2022) [A3]**

- Point-adjust (PA) lets a **random score** reach F1_PA **0.931 on MSL and 0.961 on SMAP**, above
  published methods.
- Without PA, the random score gets F1 0.190 / 0.227.
- Published deep methods score F1 0.19–0.22 on MSL and 0.23–0.33 on SMAP, comparable to "input
  itself as score" (0.239 / 0.229).
- They propose PA%K plus untrained-model baselines.

**Garg et al. (TNNLS 2021) [A4]**

- The composite event-and-point F-score (Fc1) separates detectors from random. With the
  best-F threshold:
  - MSL: random 0.324 against univariate AE (UAE) 0.713;
  - SMAP: random 0.290 against UAE 0.809.
- With the top-k threshold, Fc1:

  | Method | MSL | SMAP |
  |---|---|---|
  | UAE | 0.511 | 0.479 |
  | NASA LSTM (telemanom) | 0.472 | 0.428 |
  | PCA | 0.407 | 0.379 |
  | OmniAnomaly | 0.412 | 0.377 |

**Sarfraz et al. (ICML 2024) [A5]**

- They set SMAP/MSL aside as "potentially flawed", but in the appendix simple baselines beat SOTA
  on point-wise F1:

  | Method | MSL F1 | SMAP F1 |
  |---|---|---|
  | PCA error | 0.426 | 0.387 |
  | 1-NN distance | 0.404 | 0.352 |
  | USAD | 0.211 | 0.228 |
  | GDN | 0.217 | 0.252 |

- Caveat: their baselines are averaged per trace, while the SOTA numbers are copied from Kim et
  al., so the protocols are not identical.

**TSB-AD (NeurIPS 2024 D&B; leaderboard updated 2026-10-20) [A6]**

- It recommends VUS-PR as the metric, and finds "simpler architectures and statistical methods
  often yield better performance".
- The table itself shows the PA inflation. On TSB-AD-U:

  | Method | PA-F1 | VUS-PR |
  |---|---|---|
  | SR | 0.87 | 0.32 |
  | AnomalyTransformer | 0.53 | 0.12 (last) |

### 4.2 Results on SMAP/MSL and ESA-ADB with honest metrics

**telemanom original** (event-level precision/recall, F0.5) [A1]

| Dataset | Precision | Recall | F0.5 |
|---|---|---|---|
| SMAP | 85.5 % | 85.5 % | 0.71 |
| MSL | 92.6 % | 69.4 % | 0.69 |

The authors call the precision "overly optimistic" relative to deployment.

**TSB-AD per-series VUS-PR on its curated SMAP/MSL series** [A6, our mean over the published
per-series CSVs]

- Multivariate:
  - SMAP (25 series): KMeansAD 0.380, CNN 0.191, Donut 0.183, LSTMAD 0.163 … AnomalyTransformer 0.063.
  - MSL (14 series): KMeansAD 0.435, CNN 0.352, Donut 0.300 … AnomalyTransformer 0.116.
- Univariate:
  - SMAP (17 series): POLY 0.644, KMeansAD 0.632, MOMENT-FT 0.626 / ZS 0.621 … TimesFM 0.297, Chronos 0.186, AnomalyTransformer 0.094.
  - MSL (7 series): KShapeAD 0.554, POLY 0.542, MOMENT-FT 0.529 … TimesFM 0.321, Chronos 0.182, AnomalyTransformer 0.141.
- These are TSB-AD's curated versions, not the original telemanom split. MOMENT's pretraining saw
  the training portions of these datasets (see section 3).

**ESA-ADB official corrected event-wise F0.5** (arXiv v2, Table 2) [A7]

- Full channel sets: Mission 1 best 0.061 (Telemanom-ESA-Pruned); Mission 2 best 0.079 (PCC).
- Lightweight subsets: M1 0.786 (Telemanom-ESA-Pruned); M2 0.949 (Window iForest).
- This project's own results under the official protocol are in `outputs/benchmarks/SUMMARY.md`
  and the later literature is in `research_design/NOVELTY_NOTE.md` [A9]; they are not re-verified
  here. Two relevant ones:
  - Chronos-Bolt zero-shot was significantly *worse* than AR(8) on Mission 1 (Δ −0.052).
  - A parameter-free out-of-limit rule (nominal training range) scores 0.894 on full-set
    Mission 2, but 0.0007 on full-set Mission 1.

### 4.3 Candidate table for H (summary; the FM rows are detailed in section 3)

| Model | Version (date) | Licence | Hardware / XPU | Limits | Evidence with honest metrics | URL |
|---|---|---|---|---|---|---|
| PCA reconstruction error | pyod 3.6.7 (2026-10-12) or own | BSD-2 | CPU | – | MSL / SMAP F1 0.426 / 0.387 [A5]; overall TSB-AD-M VUS-PR 0.31 [A6] | [A8] |
| 1-NN / KMeansAD subsequence distance | own (TSB-AD pkg pins numpy < 2) | Apache-2.0 (TSB-AD) | CPU | – | KMeansAD top on TSB-AD SMAP/MSL-M (0.380 / 0.435) | [A6] |
| GlobalSTD / out-of-limit rules | own (ESA-ADB) | – | CPU | – | ESA-ADB reference baselines [A7] | [A7] |
| telemanom LSTM + NDT | khundman/telemanom (pins keras 2.3.1, tensorflow 2.0.1); ESA variant in ESA-ADB | see [A1] repo | original is TF/Keras (env-conflict); a PyTorch re-implementation would be XPU-plausible | – | event-level F0.5 SMAP 0.71, MSL 0.69 [A1]; Fc1 SMAP 0.428, MSL 0.472 [A4] | [A1] |
| AR(8) + calibrated alarms | own (`cosmos/ai/telemetry`) | – | CPU / XPU | – | project's ESA-ADB results [A9] | [A9] |
| Chronos-2 residual scoring | 2.3.2, installed | Apache-2.0 | XPU-local (Bolt) | 8,192 context | Chronos-Bolt below AR(8) on M1 [A9]; TSB-AD-U Chronos 0.27 | [F1] |
| Time-RCD | GitHub | Apache-2.0 | XPU-plausible | – | TSB-AD 0.52 / 0.32 | [F15] |
| TSPulse | granite-tsfm 0.3.11 | Apache-2.0 | CPU; env-conflict | ≥ about 1.5–2k points | TSB-AD 0.48 / 0.36 (ZS) | [F14] |
| TiRex-2 (streaming residuals) | 0.4.0 | Apache-2.0 | CPU; XPU unverified | – | forecasting only | [F4] |
| AnomalyTransformer | – | – | – | – | last on TSB-AD (0.12 / 0.12) | [A6] |

### 4.4 Recommendation for H

Run:

1. **Simple baselines**: PCA reconstruction error, KMeansAD / 1-NN subsequence distance, and the
   GlobalSTD / out-of-limit rules. Use our code or pyod 3.6.7 (BSD-2) [A8]; the TSB-AD package
   pins numpy < 2, so re-implement its KMeansAD.
   - They lead on SMAP/MSL under VUS-PR [A6], beat SOTA in [A5], and are the ESA-ADB references [A7].
   - CPU.
2. **telemanom-style LSTM**, plus our **AR(8) residual with calibrated alarms**: the
   domain-standard reference and the project's existing best. The original telemanom pins
   TF 2.0 / Keras 2.3, so use the ESA-ADB variant or a PyTorch re-implementation; the project's
   PyTorch models already train on XPU [A9].
3. **Chronos-2 forecast-residual / quantile-exceedance scoring**: Apache-2.0, installed, with
   multivariate and covariates and 8k context [F1]. Its sibling Bolt already ran on XPU here.
   TimesFM-2.5 (Apache-2.0) is the alternative if a second forecaster is wanted.
4. **One AD-native foundation model**:
   - **Time-RCD** (Apache-2.0, light deps, zero-shot, TSB-AD 0.52 / 0.32), or
   - **TSPulse** (Apache-2.0, 1M parameters, CPU, TSB-AD fine-tuned 0.55 / 0.39), which needs a separate venv because granite-tsfm pins torch < 2.12.
5. *(Optional)* **TiRex-2**: Apache-2.0 weights, streaming design suited to telemetry, CPU
   documented. XPU and the Triton dependency must be checked first.

**Metric protocol for H**

- Never report point-adjusted F1 as a headline.
- Report VUS-PR, plus event-wise or composite F scores (Fc1 [A4]), plus ESA-ADB's corrected
  event-wise F0.5 [A7].
- Include random-score and untrained-model floors [A3].
- Report SMAP/MSL per-channel results with the Wu & Keogh caveats [A2].

**Excluded for H, and why**

| Model | Reason |
|---|---|
| Kumo-Anomaly | Ampere/Hopper and Transformer Engine, **no CPU/XPU path** |
| CHARM | API-only, no weights; data would leave the premises |
| AnomalyTransformer and other PA-dependent methods | Last on VUS-PR [A6] |
| TimesFM-3, Moirai-2, TTM-R3, TimeH2O | NC licences |
| Timestral | Revenue-cap clause; strongest GIFT-Eval numbers but forecasting-only evidence |
| MOMENT | Hard dependency pins, and pretraining included the SMAP/MSL training splits |
| Toto-2.0 | Optional only: gluonts forces pandas < 3, so a separate venv is needed |
| Sundial | Pinned old transformers |

---

## 5. Graph models: is a graph over catalogue objects feasible?

**PyTorch Geometric**
- `torch-geometric` 2.8.1 (2026-10-12), MIT, Python ≥ 3.10 [G1].
- The core installs with "no external library required except for PyTorch". The optional
  extensions (pyg-lib, torch-scatter, torch-sparse) ship CPU, CUDA and ROCm kernels only.
- The CHANGELOG has **XPU** entries in 2.4.0–2.6.0: XPU env-setup docs, XPU in the basic GNN
  examples, multi-XPU training fixes [G1].
- Graph-transformer layers exist (`GPSConv`, `TransformerConv`, `SGFormer`, `Polynormer`) [G1].
- Verdict: **technically feasible on XPU without the optional extensions** (XPU-doc for the
  basics).

**DGL**
- PyPI 2.2.1 (2024-05-13); latest GitHub release v2.4.0 (2024-09-03); no XPU backend.
- Verdict: **exclude** [G2].

**Data side**

- Edges can come from shared international designator (same launch, `yyyy-nnn`) and from
  breakup parentage. A complex-network view of the resident-object population already exists in
  the literature (arXiv 2310.14795) [G3].
- The main risk is **leakage**: transductive message passing across train and test members of
  the same family. Splits must be grouped by family, as in section 1.
- The cheaper first step is family-level aggregate features fed to the GBDT/TFM arm. A GNN is
  worth trying only if those aggregates help on the grouped split.
- No published evidence was found that a GNN over catalogue objects improves lifetime prediction.

---

## 6. Hardware and licence flags

**Cannot run on CPU or Intel XPU here (as published)**

- Kumo-Anomaly (Ampere/Hopper and Transformer Engine) [F12].
- CHARM (no weights) [F16].
- Mitra-v2's reported fine-tuning recipe ("requires a CUDA GPU") [T15].
- LimiX-2 is documented only with CUDA builds of torch 2.9.1 and flash-attn [T18].
- TimesFM-2.5 XReg covariates (the `jax[cuda]` extra) [F2].
- DGL (no XPU) [G2].
- GPU modes of the GBDT pip wheels are CUDA-only. The XGBoost SYCL plugin and the LightGBM OpenCL
  build need source builds [T24, T25]. Their CPU modes are fine at our scale.

**CPU only on our machine (the XPU device is rejected)**

- pytabkit RealMLP/TabM wrappers (device whitelist cpu/cuda/mps) [T22].
- scikit-survival [S1].

**XPU-plausible but untested (smoke test needed)**

- TabPFN 9.2 (never auto-selected; CPU capped at 5,000 rows by default) [T2].
- Kumo-Tabular / structured-data-models [T14].
- TabDPT (bf16 flash-attention default) [T20].
- Timestral, TimesFM (torch), Toto-2, t0, Time-RCD [F2–F6, F15].
- SurvivalPFN [S10].
- pycox and torchsurv [S2, S3].
- TabM, rtdl modules, and our own GRU and attention models.
- TiRex-2: Triton kernels; risk of a `triton` / `triton-xpu` clash [F4].

**XPU documented or already used here**

- TabICL (XPU-doc) [T9].
- PyG core (XPU-doc) [G1].
- Chronos family (XPU-local, Bolt) [A9].

**Non-commercial or restricted weights**

- TabPFN-2.5 / 2.6 / 3 / 3.5, LimiX-2, TabFM, EXAONE-Tabular, Causilo.
- TimesFM-3, Moirai-2, TTM-R3, TimeH2O.
- Timestral (revenue cap).
- SAP-RPT-OSS ("research-only").

**Copyleft:** scikit-survival (GPL-3.0).

**Need a separate venv** (conflict with pandas 3 / torch 2.13 / numpy 2 / transformers 5)

- AutoGluon (Mitra, TabArena tooling) [T17].
- tabpfn-client (pandas ≤ 2.3.3) [T1b].
- granite-tsfm (TTM, TSPulse) [F8].
- uni2ts [F7].
- toto-2 via gluonts [F5].
- momentfm [F13].
- TSB-AD package [A6].
- Original telemanom (TF 2.0.1 / Keras 2.3.1) [A1].
- Sundial (transformers 4.40.1) [F9].
- lifelines, xgbse, pytorch-tabular, LimiX-2, TabFM [S4, S6, T32, T18, T19].

---

## 7. How the final choice will be made

The shortlists above are hypotheses. The selected model per task will be the one that wins **on
our data**, under a protocol fixed before any test scoring, in the spirit of the project's
pre-registration practice:

1. **Smoke test** each shortlisted candidate in its own venv.
   - Fit and predict on about 1k rows (or one telemetry channel) on CPU and on one free XPU.
   - Compare XPU outputs with CPU outputs to tolerance.
   - Record wall time and peak memory.
   - Drop candidates that fail, or whose licence the user cannot accept.
2. **Splits.**
   - R: grouped by launch/breakup family, plus a temporal (epoch) hold-out.
   - D: the official CAC split plus repeated stratified, grouped re-splits.
   - H: official ESA-ADB halves and TSB-AD-style per-series scoring.
3. **Metrics.**
   - R: MAE/RMSE on log-time capped at H; C-index and integrated Brier for survival; calibration.
   - D: MSE_HR / F2 plus the separate F2 and MSE_HR components.
   - H: VUS-PR, Fc1 or event-wise F, ESA-ADB corrected event-wise F0.5, and false alarms per year.
4. **Same budget per family.** Equal tuning budgets for trees and deep models. Foundation models
   at default settings, plus at most the documented ensembling options.
5. **Report against the floor.** Every result is reported against the naive or simple baselines
   (LRP, AR(8), PCA, random score), with paired bootstrap CIs and multiple seeds.

---

## 8. Sources (accessed 2026-10-27)

**Local**

- [L1] `.venv` `pip list` and `python -V`. Read-only, on this machine.
- [A9] Project files, not re-verified here:
  - `outputs/benchmarks/SUMMARY.md`
  - `research_design/NOVELTY_NOTE.md`
  - `outputs/experiments/20261024-093643_esa_m1_target_chronos-bolt-small_s0_s0/config.yaml` (`device: xpu:2`)

**Tabular**

- [T1] https://pypi.org/pypi/tabpfn/json
- [T1b] https://pypi.org/pypi/tabpfn-client/json
- [T2] https://github.com/PriorLabs/TabPFN (README). Device logic read from the `tabpfn/utils.py` (`infer_devices`) and `validation.py` files in the PyPI 9.2.0 wheel.
- [T3] https://docs.priorlabs.ai/models (limits per checkpoint and licence FAQ)
- [T4] https://huggingface.co/Prior-Labs/tabpfn_3_5
- [T5] https://arxiv.org/abs/2609.17895 (TabPFN-3.5 report)
- [T6] https://arxiv.org/abs/2605.13986 (TabPFN-3 report)
- [T7] https://arxiv.org/abs/2511.08667 (TabPFN-2.5 report)
- [T7b] https://huggingface.co/Prior-Labs/tabpfn_2_5 and https://huggingface.co/Prior-Labs/TabPFN-v2-reg (file sizes)
- [T8] https://pypi.org/pypi/tabicl/json
- [T9] https://github.com/soda-inria/tabicl (README)
- [T10] https://arxiv.org/abs/2602.11139 (TabICLv2)
- [T11] https://huggingface.co/jingang/TabICL
- [T12] https://huggingface.co/nvidia/Kumo-Tabular
- [T13] https://huggingface.co/blog/nvidia/kumo-tabular
- [T14] https://pypi.org/pypi/structured-data-models/json and https://github.com/NVIDIA/structured-data-models. The 0.2.1 wheel was inspected for triton, flash-attn and cuDF imports.
- [T15] https://huggingface.co/autogluon/mitra-classifier-2 and https://huggingface.co/autogluon/mitra-regressor-2
- [T16] https://arxiv.org/abs/2609.04540 (Mitra-v2)
- [T16b] https://arxiv.org/abs/2510.21204 (Mitra-v1)
- [T17] https://pypi.org/pypi/autogluon.tabular/json
- [T18] https://huggingface.co/stable-ai/LimiX-2 and https://arxiv.org/abs/2609.17488
- [T19] https://github.com/google-research/tabfm (README) and https://pypi.org/pypi/tabfm/json
- [T20] https://pypi.org/pypi/tabdpt/json and https://huggingface.co/Layer6/TabDPT
- [T21] https://pypi.org/pypi/tabm/json and https://arxiv.org/abs/2410.24210
- [T22] https://pypi.org/pypi/pytabkit/json and https://arxiv.org/abs/2407.04491. Device whitelist read in the 1.7.3 wheel: `models/sklearn/sklearn_interfaces.py` and `models/alg_interfaces/nn_interfaces.py`.
- [T23] https://pypi.org/pypi/rtdl_revisiting_models/json, https://pypi.org/pypi/rtdl/json and https://arxiv.org/abs/2106.11959
- [T24] https://pypi.org/pypi/xgboost/json, https://github.com/dmlc/xgboost/blob/master/plugin/sycl/README.md and https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
- [T25] https://pypi.org/pypi/lightgbm/json and https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
- [T26] https://pypi.org/pypi/catboost/json and https://catboost.ai/docs/en/concepts/loss-functions-regression
- [T27] TabArena leaderboard https://huggingface.co/spaces/TabArena/leaderboard. Regression CSV: `data/entrants_models/imputation_yes/splits_all/tasks_regression/datasets_all/website_leaderboard.csv` (13 datasets). Paper: https://arxiv.org/abs/2506.16791.
- [T28] BeyondArena: https://arxiv.org/abs/2606.30410. CSVs `data_beyondarena/subsets/split_{grouped,temporal}/size_all/features_all/tasks_regression/website_leaderboard.csv` in the same Space (7 and 11 datasets).
- [T29] https://arxiv.org/abs/2407.00956 (TALENT)
- [T30] https://arxiv.org/abs/2207.08815 (Grinsztajn et al.)
- [T31] https://pypi.org/pypi/carte-ai/json
- [T31b] https://arxiv.org/abs/2402.16785 (CARTE)
- [T32] https://pypi.org/pypi/pytorch-tabular/json

**Survival**

- [S1] https://pypi.org/pypi/scikit-survival/json and https://scikit-survival.readthedocs.io/en/stable/api/index.html
- [S2] https://pypi.org/pypi/pycox/json, https://pypi.org/pypi/torchtuples/json and https://github.com/havakv/pycox
- [S3] https://pypi.org/pypi/torchsurv/json
- [S4] https://pypi.org/pypi/lifelines/json
- [S5] https://pypi.org/pypi/auton-survival/json
- [S6] https://pypi.org/pypi/xgbse/json
- [S7] https://pypi.org/pypi/pysurvival/json
- [S8] Methods:
  - DeepSurv: https://arxiv.org/abs/1606.00931
  - DeepHit: https://ojs.aaai.org/index.php/AAAI/article/view/11842
  - Cox-Time: https://arxiv.org/abs/1907.00825
- [S9] https://arxiv.org/abs/2610.22242 (SurvICL) and https://huggingface.co/taltstidl/tabicl-survival
- [S10] https://arxiv.org/abs/2605.15488, https://github.com/rgklab/SurvivalPFN and https://huggingface.co/shi-ang/SurvivalPFN
- [S11] https://arxiv.org/abs/2606.04564 (SurvPFN)
- [S12] https://arxiv.org/abs/2603.29475 (Survival In-Context)
- [S13] https://arxiv.org/abs/2607.09577 (SurvFM)

**CDM / Kelvins**

- [C1] https://arxiv.org/abs/2008.03069 (Uriot et al., Astrodynamics 6, 2022)
- [C2] https://kelvins.esa.int/collision-avoidance-challenge/
- [C3] https://arxiv.org/abs/2012.12450 (Pinto et al.)
- [C4] https://arxiv.org/abs/2609.13191 (Tok et al.)
- [C5] https://arxiv.org/abs/2311.10633 (Catulo et al.)
- [C6] Irregular-series models and packages:
  - mTAN: https://arxiv.org/abs/2101.10318
  - GRU-D: https://arxiv.org/abs/1606.01865
  - SeFT: https://arxiv.org/abs/1909.12064
  - Raindrop: https://arxiv.org/abs/2110.05357
  - Neural CDE: https://arxiv.org/abs/2005.08926
  - https://pypi.org/pypi/torchcde/json and https://pypi.org/pypi/torchdiffeq/json

**Time-series foundation models**

- [F1] https://pypi.org/pypi/chronos-forecasting/json, https://huggingface.co/amazon/chronos-2 and https://arxiv.org/abs/2510.15821
- [F2] https://pypi.org/pypi/timesfm/json, https://github.com/google-research/timesfm (README), https://huggingface.co/google/timesfm-3.0-pytorch and https://huggingface.co/google/timesfm-2.5-200m-pytorch
- [F3] https://huggingface.co/mistralai/Timestral-209M (card and LICENSE) and https://github.com/mistralai/timestral
- [F4] TiRex-2 and its dependencies:
  - https://pypi.org/pypi/tirex-2/json
  - https://huggingface.co/NX-AI/TiRex-2
  - https://arxiv.org/abs/2607.01204
  - https://pypi.org/pypi/mlstm_kernels/json (licence text)
  - https://pypi.org/pypi/flashrnn/json
- [F5] Toto-2.0 and its dependencies:
  - https://huggingface.co/Datadog/Toto-2.0-313m
  - https://pypi.org/pypi/toto-2/json
  - https://pypi.org/pypi/gluonts/json
  - https://arxiv.org/abs/2605.20119
- [F6] https://huggingface.co/theforecastingcompany/t0-gamma and https://pypi.org/pypi/tfc-t0/json
- [F7] https://huggingface.co/Salesforce/moirai-2.0-R-small and https://pypi.org/pypi/uni2ts/json
- [F8] https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2, https://huggingface.co/ibm-research/ttm-r3 and https://pypi.org/pypi/granite-tsfm/json
- [F9] https://huggingface.co/thuml/sundial-base-128m
- [F10] https://huggingface.co/h2oai/timeh2o-v1
- [F11] GIFT-Eval:
  - Space: https://huggingface.co/spaces/Salesforce/GIFT-Eval (`results/*/all_results.csv`, `config.json`, `src/utils.py`)
  - Paper: https://arxiv.org/abs/2410.10393
  - TIME benchmark: https://arxiv.org/abs/2602.12147
- [F12] https://huggingface.co/nvidia/Kumo-Anomaly
- [F13] https://huggingface.co/AutonLab/MOMENT-1-large, https://pypi.org/pypi/momentfm/json and https://arxiv.org/abs/2402.03885 (Time Series Pile includes the TSB-UAD training splits)
- [F14] https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1 and https://arxiv.org/abs/2505.13033
- [F15] https://huggingface.co/thu-sail-lab/Time-RCD, https://github.com/thu-sail-lab/Time-RCD (pyproject.toml) and https://arxiv.org/abs/2509.21190
- [F16] https://huggingface.co/c3aiia3c/CHARM and https://pypi.org/pypi/c3-charm/json
- [F17] https://huggingface.co/amazon/chronos-bolt-base

**Anomaly detection**

- [A1] https://arxiv.org/abs/1802.04431 (Hundman et al., KDD 2018) and https://github.com/khundman/telemanom
- [A2] https://arxiv.org/abs/2009.13807 (Wu & Keogh)
- [A3] https://arxiv.org/abs/2109.05257 (Kim et al.)
- [A4] https://arxiv.org/abs/2109.11428 (Garg et al.)
- [A5] https://arxiv.org/abs/2405.02678 (Sarfraz et al.)
- [A6] TSB-AD:
  - Leaderboard: https://thedatumorg.github.io/TSB-AD/ (frames under `static/leaderboard/`)
  - Per-series results: https://github.com/TheDatumOrg/TSB-AD/tree/main/benchmark_exp/benchmark_eval_results
  - Package: https://pypi.org/pypi/TSB-AD/json
- [A7] https://arxiv.org/abs/2406.17826 (ESA-ADB, v2, Table 2) and https://github.com/kplabs-pl/ESA-ADB
- [A8] https://pypi.org/pypi/pyod/json

**Graph**

- [G1] PyTorch Geometric:
  - https://pypi.org/pypi/torch-geometric/json
  - https://github.com/pyg-team/pytorch_geometric/blob/master/CHANGELOG.md
  - https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  - https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
- [G2] https://pypi.org/pypi/dgl/json and https://github.com/dmlc/dgl/releases
- [G3] https://arxiv.org/abs/2310.14795

---

## 7. Model set actually used (fixed 2026-10-27, after the user asked for models published 2022–2026)

### Learned-expert candidates: re-entry (R) and collision risk (D)

The expert is the candidate with the lowest pinball loss on the tuning split.

| Model | Published | Package / weights | Licence (code / weights) | Hardware here | Notes |
|---|---|---|---|---|---|
| TabM | Gorishniy et al., ICLR 2025 | `tabm` 0.0.3 | Apache-2.0 | Intel XPU, fp32 | quantile head (non-crossing), the paper's search space, 12 trials as for every tuned learner |
| TabICLv2 | Qu et al., 2026 (ICML '26) | `tabicl` 2.2.0, `jingang/TabICL` | BSD-3 | CPU, fp32 (mixed precision on XPU gave NaN) | default inference, no search; full training set as context |
| Kumo-Tabular "large" | NVIDIA, 2026 | `structured-data-models` 0.2.1, `nvidia/Kumo-Tabular` | Apache-2.0 / OpenMDW-1.1 | Intel XPU, fp32 | 999 quantiles, 7 read off; default inference, no search. Context capped at 16,000 training rows, drawn per seed (device memory). Attention split along queries on XPU, which is exact. |

### Reference baselines

These are scored and reported but are not eligible as the expert:
- XGBoost 3.4.1;
- LightGBM 4.7.0;
- CatBoost 1.2.10;
- a plain quantile MLP;
- for R only, XGBoost-AFT, the censoring-aware baseline.

### Telemetry (H)

The detectors are:
- Chronos-2 (Amazon, 2025; `chronos-forecasting` 2.3.2, `amazon/chronos-2`, Apache-2.0). It is a
  zero-shot forecaster on XPU, scored by its forecast residuals; its training-series scores serve
  as an out-of-sample calibration set.
- The published telemanom scores (Hundman et al. 2018) as the domain reference.
- Ridge autoregression, PCA and kNN windows as classic baselines.
- A random-score floor and an always-on reference.

### Considered and not used

| Model | Reason |
|---|---|
| RealMLP (NeurIPS 2024) | The `pytabkit` implementation has no quantile/pinball loss, so it cannot produce the seven quantiles every expert must output. Re-implementing it would not be the published model. |
| TabPFN-3.5 (2026) | Weights are licensed for non-commercial use only. Can be added if the authors' use allows it. |
| LimiX-2, TabFM, Mitra-v2, TabDPT | See section 1.4 (non-commercial weights, environment pins, or weak off-IID evidence). |
| Time-RCD, TSPulse, MOMENT | See section 4.4 (environment pins; MOMENT's pretraining data include SMAP/MSL). |

### Deviation

The selection pool was changed after the first full re-entry run had been scored. See
`docs/REVIEW_AND_FIXES.md` (deviation of 2026-10-27). That run is kept and reported.
