# COSMOS baseline report (before COSMOS-AI changes)

Recorded 2026-10-24, before any file of the existing project was modified.

## Repository state

- Project root: `/data/anupa/COSMOS_work/Final_Project`.
- **Not a git repository** (no branch and no tracked changes to inspect). The original source,
  tests, notebook, README and research-design documents were snapshotted to
  `outputs/baseline/cosmos_baseline_src_2026-10-24.tar.gz` before any change.
  `tests/test_ai.py::test_pipeline_backward_compatible` runs the snapshotted original package
  in a subprocess and checks that `run_pipeline()` is unchanged.
- Approved methodology: `research_design/COSMOS_AI_Research_Design.md` (Candidate 4,
  calibrated telecommand-aware deep forecasting on ESA-ADB; base paper Kotowski et al., DMLR
  2026). The audit is in `research_design/A_existing_project_audit.md`.

## Existing tests (environment: `sgl-xpu` Python 3.12.13, numpy 2.3.5, pandas 3.0.3)

| Suite | Result | Log |
|---|---|---|
| `tests/test_verify.py` (physics and numerics) | **20 / 20 pass**, 6 s | `test_verify_before.log` |
| `tests/test_datasets.py` (real-data layer) | **20 pass, 1 error** | `test_datasets_before.log`, `test_datasets_before_per_test.log` |

The one error is pre-existing and environmental. `test_safe_unpickler` pickles a DataFrame of
strings; under pandas 3 with pyarrow this produces an `ArrowStringArray`, which the project's
allow-list unpickler refuses (`UnpicklingError: refusing to load
pandas.arrays.ArrowStringArray`). The real ESA archives contain no such type and load
correctly. The allow-list was not widened, because doing so would weaken a security control
in order to make a test pass. When run as a script, the error aborts the remainder of
`test_datasets.py`; run per test, every other check passes.

A second pre-existing defect surfaced later. `cosmos.health.simulate_telemetry` raises
`ValueError: scale < 0` for the `"stuck"` fault: its noise multiplier 0.6 gives the negative
scale σ·(0.6 − 1). The simulator was not changed. The COSMOS-AI synthetic benchmark excludes
`stuck`, and the original evaluation in the audit also reports no `stuck` results.

## Existing benchmark results (from the audit, section A5; not re-run unless stated)

| Result | Value | Condition |
|---|---|---|
| Simulated health F1 | 0.824–0.829 (P 1.0, R ≈ 0.70) | synthetic, 3 seed sets |
| ESA-ADB "simulator monitor" F1 | 1.0 / 1.0 / 0.056 | COSMOS protocol, degenerate (alarm duty 1.0) |
| ESA-ADB AR-residual variant F1 | 0.344 / 0.069 / 0.528 | COSMOS protocol, not comparable with published results |
| SMAP/MSL event F1 | 0.472 → 0.572 | variant chosen on test |
| TinyYOLO "mAP@50" | 0.947 | centre-distance, class-agnostic AP on 12 synthetic images |
| Physics checks | 20 / 20 | `test_verify.py` |

Re-run under the official ESA-ADB protocol, the existing COSMOS monitor scores corrected
event-wise F0.5 = **0.004** (Mission 1, full channel set) and **0.047** (Mission 2), with 3,278
and 3,576 false alarms per year (`outputs/benchmarks/telemetry_esa_adb/results.csv`).

## Compute

| Resource | Measured |
|---|---|
| CPU | 512 cores |
| RAM | 1.0 TiB (≈ 890 GiB available) |
| GPU | 8 × Intel Arc Pro B60 (23.9 GiB each), PyTorch 2.13.0+xpu; XPU 3 was in use by another user's job (2 GiB free) and was left alone throughout |
| fp32 matmul (XPU 0) | ≈ 6.2 TFLOPS measured |
| Disk | `/data` 168 GB free at start; `/` 17 GB free (caches redirected to `/data`) |

## Dependencies

- Base `python3` (miniforge, 3.13) lacks numpy and torch, so all work uses `sgl-xpu`.
- A project venv (`.venv`, `--system-site-packages` on `sgl-xpu`) was created, so the shared
  `sgl-xpu` environment is unchanged. Added to it: `chronos-forecasting 2.3.2`,
  `ultralytics 8.4.185`, `portion 2.6.3` (needed by the official ESA-ADB metrics), `pypdf`,
  `nbformat`, `nbclient`, `nbconvert`, `pytest`. The full list is in `venv_freeze_2026-10-24.txt`.
- Not installed: `lifelines`, because it pins pandas < 3. Survival metrics were implemented in
  `cosmos/ai/evaluation.py` and checked against hand cases. Also not installed: `pycox`,
  `captum` (Integrated Gradients is implemented directly) and `momentfm` (pins old numpy and
  transformers).

## Dataset availability at the start

| Dataset | State at start |
|---|---|
| ESA-ADB v1.0 (3 archives, 11.6 GB) | on disk, prepared at the COSMOS 5/2-min grid only |
| NASA SMAP/MSL | on disk |
| CelesTrak SATCAT + GP (2026-08-16) | on disk |
| GHS-POP, Natural Earth | on disk |
| COSMOS synthetic YOLO set | 48 images (36 train / 12 val, val used as test) |
| OPSSAT-AD, SpotGEO, SPARK, SpaceSense-Bench, DISCOS, Space-Track | absent |

## Limitations at the start (audit A6, abbreviated)

- No deep temporal model.
- No official-protocol ESA-ADB evaluation.
- No validation split anywhere.
- Subsystem attribution no better than a majority baseline.
- No real imagery.
- A detector metric mislabelled as mAP@50.
- No element-set history.
- Leakage risk in the SATCAT orbit fields.
- Single seeds.
