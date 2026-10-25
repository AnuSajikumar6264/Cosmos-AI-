"""Checks for COSMOS-AI — preprocessing, splits, metrics, models, integration.

Run it directly (no pytest needed), or with pytest:

    python tests/test_ai.py
    pytest tests/test_ai.py

As in ``test_verify.py`` and ``test_datasets.py``, every expected value comes from outside the
code under test: a literal port of the official code, a hand-built case, an analytic result,
or the original (unmodified) COSMOS code.  Checks that need a large dataset or a trained model
run only if it is present and are reported as SKIP otherwise, never as PASS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = []
UNDER_PYTEST = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def check(name, expected, measured, ok, note=""):
    RESULTS.append({"check": name, "expected": expected, "measured": measured,
                    "status": "PASS" if ok else "FAIL", "note": note})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         expected {expected}   "
          f"measured {measured}" + (f"   ({note})" if note else ""))
    if UNDER_PYTEST:
        assert ok, f"{name}: expected {expected}, measured {measured}"
    return ok


def skip(name, why):
    RESULTS.append({"check": name, "expected": "-", "measured": "-", "status": "SKIP",
                    "note": why})
    print(f"  [SKIP] {name}   ({why})")


def _has(*parts):
    return os.path.exists(os.path.join(ROOT, *parts))


# ============================================================================ preprocessing
def test_zoh_matches_official_port():
    from cosmos.ai.data import preprocess_telemetry as P
    rng = np.random.default_rng(0)
    for mission, name, day in ((1, "channel_20", "2003-03-01"), (1, "channel_5", "2003-03-01"),
                               (2, "channel_30", "2000-03-01")):   # inside each training half
        t = pd.Timestamp(day) + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 86400, 4000)), "s")
        df = pd.DataFrame({name: np.cumsum(rng.normal(size=4000)).astype(np.float32)}, index=t)
        lab = pd.DataFrame({"ID": ["a", "b"], "Channel": [name, name],
                            "StartTime": [t[500], t[2100]], "EndTime": [t[540], t[2100]],
                            "Category": ["Anomaly", "Rare Event"]})
        proto = P.ESA_PROTOCOL[mission]
        rows = [(r.StartTime.value, r.EndTime.value, 1 if r.Category == "Anomaly" else 2)
                for r in lab.itertuples()]
        first, gv, gl = P.channel_grids(df, rows, proto)["train"]
        ref = P.official_reference_channel(df, lab, mission, "train")
        ok = len(ref) == len(gv) and np.allclose(ref.value.to_numpy(float), gv, atol=1e-5) \
            and np.array_equal(ref.label.to_numpy(int), gl.astype(int))
        check(f"vectorised ZOH = official pandas port (M{mission} {name})", "identical",
              "identical" if ok else f"{len(ref)} vs {len(gv)} points", ok,
              "values, labels and the 'restore annotated samples' step")


def test_telecommand_encoding():
    from cosmos.ai.data.preprocess_telemetry import _telecommand_indicator
    rule = 30 * 10 ** 9
    ex = np.array([95, 100, 150, 400], dtype=np.int64) * 10 ** 9      # seconds -> ns
    ind = _telecommand_indicator(ex, 0, 20, rule)
    # executions in (g - 30 s, g]: g = 120 s (95, 100 s), 150 s (150 s), 420 s (400 s)
    expect = np.zeros(20, np.uint8)
    expect[[4, 5, 14]] = 1
    check("telecommand indicator = 1 iff an execution lies in (g - rule, g]",
          expect.nonzero()[0].tolist(), ind.nonzero()[0].tolist(), np.array_equal(ind, expect))


def test_esa_processed():
    from cosmos.ai.data.preprocess_telemetry import (ESA_PROTOCOL, load_esa_split,
                                                     official_test_labels)
    expected = {1: (7_364_161, 91), 2: (3_062_401, 362)}
    for m in (1, 2):
        if not _has("data", "feature1_health", "esa_adb", "processed", f"mission{m}", "meta.json"):
            skip(f"ESA mission {m} official grid", "run: python -m cosmos.ai.data.preprocess_telemetry")
            continue
        te = load_esa_split(m, "test")
        lab = official_test_labels(m, te)
        ev = lab[lab.Channel.isin(ESA_PROTOCOL[m]["target"])].ID.nunique()
        nan = bool(np.isnan(np.asarray(te.X[:200000])).any())
        check(f"ESA mission {m}: test grid length / target events / no NaN",
              f"{expected[m][0]:,} / {expected[m][1]} / False", f"{te.n:,} / {ev} / {nan}",
              te.n == expected[m][0] and ev == expected[m][1] and not nan,
              "events: test-range events on target channels (M1 91 per DMLR Table 21)")


def test_opssat_tensors():
    if not _has("data", "feature1_health", "opssat_ad", "raw", "segments.csv"):
        skip("OPSSAT-AD tensors", "dataset not downloaded")
        return
    from cosmos.ai.data.preprocess_telemetry import load_opssat_ad, opssat_tensors
    d = load_opssat_ad()
    X, mask, ids = opssat_tensors(d["segments"], 256)
    # interpolation may not leave each fragment's own z-scored range (no cross-fragment info)
    out = 0
    for k, (sid, g) in enumerate(d["segments"].groupby("segment", sort=True)):
        v = g.value.to_numpy(np.float64)
        z = (v - v.mean()) / (v.std() + 1e-12)
        out += int(X[k, 0].min() < z.min() - 1e-4 or X[k, 0].max() > z.max() + 1e-4)
    check("OPSSAT-AD: 2,123 fragments -> (2123, 3, 256), each inside its own z-range",
          "(2123, 3, 256), 0 outside", f"{X.shape}, {out} outside", X.shape == (2123, 3, 256)
          and out == 0)
    f = d["features"]
    check("OPSSAT-AD official split sizes", "1594 train / 529 test",
          f"{int(f.train.sum())} train / {int((f.train == 0).sum())} test",
          int(f.train.sum()) == 1594 and int((f.train == 0).sum()) == 529)


# ============================================================================ splits / leakage
def test_splits():
    from cosmos.ai.data.preprocess_catalogue import group_split
    g = pd.Series([f"2020-{i:03d}" for i in range(500)] * 3)
    s1, s2 = group_split(g), group_split(g)
    per = pd.DataFrame({"g": g, "s": s1}).groupby("g").s.nunique()
    check("group split: deterministic and every group in exactly one split", "True / 1",
          f"{np.array_equal(s1, s2)} / {per.max()}", np.array_equal(s1, s2) and per.max() == 1)
    if _has("data", "feature2_debris", "spotgeo", "raw", "extracted", "SpotGEOv2", "train"):
        from cosmos.ai.vision.spotgeo import train_val_split
        a, b = train_val_split()
        c, d = train_val_split()
        check("SpotGEO train/val split by sequence: disjoint, deterministic, ~10 % val",
              "disjoint, equal, 0.07-0.13", f"{not set(a) & set(b)}, {a == c and b == d}, "
              f"{len(b) / (len(a) + len(b)):.3f}",
              not set(a) & set(b) and a == c and b == d and 0.07 < len(b) / (len(a) + len(b)) < 0.13)
    else:
        skip("SpotGEO split", "dataset not extracted")
    p = os.path.join(ROOT, "data", "feature2_debris", "cosmos_synthetic", "yolo_v1", "scenes.jsonl")
    if os.path.isfile(p):
        recs = [json.loads(l) for l in open(p)]
        seeds = {s: {r["seed"] for r in recs if r["split"] == s} for s in ("train", "val", "test")}
        ok = not (seeds["train"] & seeds["val"] or seeds["train"] & seeds["test"]
                  or seeds["val"] & seeds["test"])
        check("COSMOS synthetic vision: no scene seed shared across splits", "disjoint",
              "disjoint" if ok else "overlap", ok)
    else:
        skip("COSMOS synthetic vision split", "dataset not generated")


def test_landmark_leakage():
    p = os.path.join(ROOT, "data", "feature3_reentry", "derived",
                     "landmark_2026-08-16_to_2026-10-23.csv")
    if not os.path.isfile(p):
        skip("re-entry landmark leakage report", "landmark table not built")
        return
    from cosmos.ai.data.preprocess_catalogue import leakage_report
    rep = leakage_report(pd.read_csv(p))
    check("re-entry landmark: every leakage rule holds", "all True", rep, all(rep.values()),
          "GP epoch <= tau day, events after tau, launches in one split, constant censoring")


# ============================================================================ metrics
def test_official_esa_metrics():
    from cosmos.ai.telemetry.esa_eval import compress_binary, _quiet
    from cosmos.ai.telemetry.esa_metrics import ESAScores
    # the official module's own example (ESA_ADB_metrics.py __main__)
    fr = [pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-15")]
    yt = pd.DataFrame([["id_0", pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-02"), "Anomaly"],
                       ["id_1", pd.Timestamp("2015-01-04"), pd.Timestamp("2015-01-05"), "Anomaly"],
                       ["id_2", pd.Timestamp("2015-01-07"), pd.Timestamp("2015-01-08"), "Anomaly"]],
                      columns=["ID", "StartTime", "EndTime", "Category"])
    yp = [[pd.Timestamp("2015-01-01"), 0], [pd.Timestamp("2015-01-04"), 1],
          [pd.Timestamp("2015-01-09"), 0]]
    r = _quiet(ESAScores(betas=0.5, full_range=fr).score, yt, yp)
    # by hand: the alarm [Jan 4, Jan 9) overlaps events 2 and 3 (recall 2/3) and is matched,
    # so precision before correction is 1; it covers 3 nominal days (Jan 5-7, Jan 8-9) out of
    # 14 - 3 = 11, so TNR = 8/11 and P = 8/11; F0.5 = 1.25 P R / (0.25 P + R)
    P, R = 8 / 11, 2 / 3
    f = 1.25 * P * R / (0.25 * P + R)
    check("official corrected event-wise F0.5 on a hand-solved case", f"{f:.4f}",
          f"{r['EW_F_0.50']:.4f}", abs(r["EW_F_0.50"] - f) < 1e-9)
    # compressing a long binary series to its change points must not change any score
    rng = np.random.default_rng(1)
    ts = pd.date_range("2015-01-01", periods=5000, freq="30s")
    y = (rng.random(5000) < 0.01).astype(int)
    y = np.convolve(y, np.ones(20), "same").clip(0, 1).astype(int)
    lab = pd.DataFrame({"ID": ["e1", "e2"], "StartTime": [ts[1000], ts[3000]],
                        "EndTime": [ts[1100], ts[3050]], "Category": ["Anomaly", "Rare Event"]})
    full = np.empty((5000, 2), dtype=object)
    full[:, 0], full[:, 1] = list(ts), y
    a = _quiet(ESAScores(betas=0.5).score, lab, full)
    b = _quiet(ESAScores(betas=0.5).score, lab, compress_binary(ts, y))
    same = all(abs(a[k] - b[k]) < 1e-12 for k in a)
    check("change-point compression leaves every official score unchanged", "identical",
          "identical" if same else str({k: (a[k], b[k]) for k in a}), same)


def test_conformal_and_bh():
    from cosmos.ai.uncertainty import bh_reject, conformal_pvalues, simes
    rng = np.random.default_rng(0)
    # super-uniformity averaged over calibration draws: P(p <= a) <= a
    hits = []
    for _ in range(400):
        cal = np.sort(rng.normal(size=199))
        hits.append((conformal_pvalues(cal, rng.normal(size=500)) <= 0.05).mean())
    m = float(np.mean(hits))
    se = float(np.std(hits) / np.sqrt(len(hits)))
    check("conformal p-values are valid under exchangeability", "P(p <= 0.05) <= 0.05 + 3 SE",
          f"{m:.4f} (SE {se:.4f})", m <= 0.05 + 3 * se)
    P = rng.uniform(size=(20000, 40))
    sig = np.zeros_like(P, bool)
    sig[:, :4] = True
    P[:, :4] = rng.uniform(size=(20000, 4)) * 1e-3
    rej = bh_reject(P, 0.1)
    fdr = float(((rej & ~sig).sum(1) / np.maximum(rej.sum(1), 1)).mean())
    check("Benjamini-Hochberg controls FDR at q m0 / m", "<= 0.09 + 0.005", f"{fdr:.4f}",
          fdr <= 0.09 + 0.005, "independent nulls: FDR = q m0 / m exactly in expectation")
    check("BH rejects something exactly when Simes <= q", "agree",
          "agree" if np.array_equal(rej.any(1), simes(P) <= 0.1) else "disagree",
          np.array_equal(rej.any(1), simes(P) <= 0.1))


def test_survival_and_classification():
    from cosmos.ai.evaluation import harrell_c, ipcw_brier, brier_score
    # times 1..4 all observed, risk perfectly anti-ordered with time -> C = 1; reversed -> 0
    t, e = np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1])
    check("Harrell C on perfectly ordered / reversed risks", "1.0 / 0.0",
          f"{harrell_c(t, e, -t):.1f} / {harrell_c(t, e, t):.1f}",
          harrell_c(t, e, -t) == 1.0 and harrell_c(t, e, t) == 0.0)
    rng = np.random.default_rng(0)
    t = rng.uniform(1, 10, 300)
    e = np.ones(300, int)
    s = rng.uniform(size=300)
    b1 = ipcw_brier(t, e, s, 5.0)
    b2 = brier_score((t > 5).astype(float), s)
    check("IPCW Brier equals the plain Brier score without censoring", f"{b2:.6f}",
          f"{b1:.6f}", abs(b1 - b2) < 1e-9)


def test_detection_metrics():
    from cosmos.ai.vision.metrics import coco_map
    from cosmos.ai.vision.spotgeo_validation import score_frame
    t = [np.array([[0, 50, 50, 10, 10], [1, 100, 100, 8, 8]], float)]
    perfect = [np.array([[50, 50, 10, 10, 0.9, 0], [100, 100, 8, 8, 0.8, 1]], float)]
    shifted = [np.array([[52, 50, 10, 10, 0.9, 0], [100, 100, 8, 8, 0.8, 1]], float)]
    a, b = coco_map(perfect, t, 2), coco_map(shifted, t, 2)
    # 2 px shift of a 10 px box: IoU = 80 / 120 = 0.667 -> AP 1 at IoU 0.50..0.65, 0 above
    exp_b = (1.0 + 4 / 10) / 2
    check("COCO mAP50-95: perfect boxes = 1; a 2 px shift scores 4/10 of thresholds",
          f"1.000 / {exp_b:.3f}", f"{a['map50_95']:.3f} / {b['map50_95']:.3f}",
          abs(a["map50_95"] - 1) < 1e-9 and abs(b["map50_95"] - exp_b) < 1e-9)
    # SpotGEO page example structure: 2 within tau, 1 outside tau, 1 unmatched prediction
    X = np.array([[0.0, 0.0], [20.0, 2.0], [60.0, 0.0], [200.0, 200.0]])
    Y = np.array([[1.0, 0.0], [20.0, 0.0], [40.0, 0.0]])
    TP, FN, FP, sse = score_frame(X, Y)
    check("official SpotGEO score_frame: TP / FN / FP on the scoring-page example layout",
          "2 / 1 / 2", f"{TP} / {FN} / {FP}", (TP, FN, FP) == (2, 1, 2))


def test_configs_match_code():
    import yaml
    from cosmos.ai.telemetry import alarms, forecasting
    from cosmos.ai import surrogate
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "telemetry", "esa_adb.yaml")))
    d = forecasting.ForecastConfig()
    ok = all(getattr(d, k) == v for k, v in cfg["forecast"].items())
    ok &= cfg["calibrators"]["grids"] == {k: list(v) for k, v in alarms.GRIDS.items()}
    ok &= cfg["calibrators"]["persistence_k"] == alarms.K_GRID
    ok &= cfg["nonconformity_delta"] == forecasting.DELTA
    ok &= all(cfg["periods_inside_training_half"][f"mission{m}"] == {k: list(v) for k, v in
              forecasting.PERIODS[m].items()} for m in (1, 2))
    sc = yaml.safe_load(open(os.path.join(ROOT, "configs", "reentry", "surrogate.yaml")))
    ok &= {k: list(v) for k, v in surrogate.RANGES.items()} == {k: [float(x) for x in v] for k, v in
                                                               sc["input_ranges"].items()}
    check("configs/*.yaml state exactly the values the code runs with", "identical",
          "identical" if ok else "diverged", bool(ok))


# ============================================================================ models
def test_models():
    import torch
    from cosmos.ai.telemetry.models import (GatedCovariateFusion, PatchTSTForecaster,
                                            TCNForecaster, build_classifier)
    from cosmos.ai.vision.heatmap import build
    torch.manual_seed(0)
    tcn = TCNForecaster(5, horizon=16, d_model=16, levels=7).eval()
    x = torch.randn(2, 1, 600)
    q, _ = tcn(x, torch.tensor([0, 1]))
    check("TCN output shape (B, T, H, Q)", "(2, 600, 16, 7)", tuple(q.shape),
          tuple(q.shape) == (2, 600, 16, 7))
    x2 = x.clone()
    x2[..., 400:] += 3.0
    with torch.no_grad():
        d = (tcn(x2, torch.tensor([0, 1]))[0][:, :400] - q[:, :400]).abs().max().item()
    check("TCN is causal: future inputs never change past forecasts", "0", f"{d:.1e}", d == 0.0)
    check("TCN receptive field fits the 512-sample context", "<= 512", tcn.receptive_field,
          tcn.receptive_field <= 512)
    pt = PatchTSTForecaster(5, context=512, horizon=16, d_model=32, n_heads=4, n_layers=1).eval()
    q2, _ = pt(torch.randn(3, 1, 512), torch.tensor([0, 1, 2]))
    check("PatchTST output shape (B, H, Q) and monotone quantiles", "(3, 16, 7), monotone",
          f"{tuple(q2.shape)}, {bool((q2.diff(dim=-1) >= 0).all())}",
          tuple(q2.shape) == (3, 16, 7) and bool((q2.diff(dim=-1) >= 0).all()))
    g = GatedCovariateFusion(8, 4)
    with torch.no_grad():
        g.gate.bias.fill_(-1e4)
        h = torch.randn(5, 8)
        out, gate = g(h, torch.randn(5, 4))
    check("closed gate recovers the telemetry-only representation", "max |h' - h| = 0",
          f"{(out - h).abs().max().item():.1e}", (out - h).abs().max().item() < 1e-6)
    for kind in ("fcnn", "cnn", "transformer", "hybrid"):
        m = build_classifier(kind, **({"n_features": 18} if kind == "fcnn" else
                                      {"in_ch": 3, "n_features": 18} if kind == "hybrid" else
                                      {"in_ch": 3})).eval()
        o = m(torch.randn(4, 3, 256), torch.randn(4, 18), torch.ones(4, 256))
        check(f"{kind} classifier returns one logit per fragment", "(4,)", tuple(o.shape),
              tuple(o.shape) == (4,))
    hm = build("unet", 10, 5).eval()
    o = hm(torch.randn(1, 10, 96, 128))
    check("heat-map U-Net: full-resolution heat and offset maps", "(1,5,96,128) / (1,10,96,128)",
          f"{tuple(o['heat'].shape)} / {tuple(o['offset'].shape)}",
          tuple(o["heat"].shape) == (1, 5, 96, 128) and tuple(o["offset"].shape) == (1, 10, 96, 128))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        torch.save({"state_dict": pt.state_dict()}, p)
        pt2 = PatchTSTForecaster(5, context=512, horizon=16, d_model=32, n_heads=4, n_layers=1)
        pt2.load_state_dict(torch.load(p)["state_dict"])
        pt2.eval()
        xx = torch.randn(2, 1, 512)
        same = torch.equal(pt(xx, torch.tensor([0, 1]))[0], pt2(xx, torch.tensor([0, 1]))[0])
    check("checkpoint round trip reproduces the forecasts exactly", "identical",
          "identical" if same else "different", same)


def test_explain():
    import torch
    from cosmos.ai.explain import completeness_gap, integrated_gradients
    w = torch.tensor([1.0, -2.0, 0.5])
    f = lambda x: (x * w).sum(-1)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    a = integrated_gradients(f, x, steps=16)
    check("Integrated Gradients on a linear model = w * x (completeness)", "[1, -4, 1.5]",
          a.numpy().round(4).tolist(), torch.allclose(a, x * w, atol=1e-5)
          and completeness_gap(f, x, a) < 1e-5)


# ============================================================================ integration
def _original_pipeline_summary():
    """Run the UNMODIFIED baseline COSMOS (from outputs/baseline) in a subprocess."""
    tar = os.path.join(ROOT, "outputs", "baseline", "cosmos_baseline_src_2026-10-24.tar.gz")
    if not os.path.isfile(tar):
        return None
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(tar) as t:
            t.extractall(d, members=[m for m in t.getmembers() if m.name.startswith("cosmos/")],
                         filter="data")
        os.symlink(os.path.join(ROOT, "data"), os.path.join(d, "data"))
        code = ("import json, cosmos; r = cosmos.run_pipeline(faults={'Propulsion System': "
                "'leak'}, n_samples=150, seed=3); s = r.summary(); "
                "print(json.dumps([str(v) for v in s.value]))")
        out = subprocess.run([sys.executable, "-c", code], cwd=d, capture_output=True,
                             text=True, timeout=900)
        return json.loads(out.stdout.strip().splitlines()[-1])


def test_pipeline_backward_compatible():
    import cosmos
    old = _original_pipeline_summary()
    if old is None:
        skip("run_pipeline() unchanged against the baseline code", "baseline snapshot missing")
        return
    r = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, n_samples=150, seed=3)
    new = [str(v) for v in r.summary().value]
    check("run_pipeline() defaults reproduce the ORIGINAL code's summary exactly",
          "identical 20-row summary", "identical" if new == old else
          f"{sum(a != b for a, b in zip(new, old))} rows differ", new == old,
          "original package extracted from outputs/baseline and run in a subprocess")


def test_learned_backends():
    import cosmos
    from cosmos.ai.fusion import Evidence, decide
    if _has("data", "feature1_health", "cosmos_synthetic", "learned_monitor_forecaster.pt"):
        r = cosmos.run_pipeline(faults={"Propulsion System": "leak"}, n_samples=150, seed=3,
                                backend="learned")
        ok = r.monitor.method.startswith("learned") and 0 <= r.monitor.anomaly.max() <= 1 \
            and r.hazard.breakup_type in ("explosion", "aerodynamic")
        check("learned telemetry backend feeds the unchanged hazard / break-up chain",
              "MonitorResult -> BreakupHazard", f"{r.monitor.method} -> {r.hazard.breakup_type}",
              ok)
    else:
        skip("learned telemetry backend", "train it: python -m cosmos.ai.telemetry.learned_monitor train")
    if _has("data", "feature3_reentry", "simulated", "propagator_surrogate_v1", "checkpoints"):
        r = cosmos.run_pipeline(n_samples=200, seed=3, footprint_backend="surrogate")
        frac = getattr(r.footprint, "surrogate_fraction", float("nan"))
        check("surrogate footprint runs inside the pipeline (with reference fallback)",
              "impacts > 0, surrogate fraction in [0, 1]",
              f"{len(r.footprint.frame)} impacts, fraction {frac:.2f}",
              len(r.footprint.frame) > 0 and 0 <= frac <= 1)
    else:
        skip("surrogate footprint", "train it: python -m cosmos.ai.surrogate train")
    a = Evidence("health", "m1", "c", 0.6, "validated-real")
    b = Evidence("reentry", "m2", "c", 0.5, "validated-real")
    c = Evidence("debris", "m3", "c", 0.9, "not-run")
    d1 = decide([a, b, c], "c")
    d2 = decide([c], "c")
    d3 = decide([a, Evidence("x", "m4", "c", 0.0, "validated-real")], "c", conflict=0.5)
    ok = (not d1.abstain and abs(d1.probability - 0.8) < 1e-12 and d1.used == ["m1", "m2"]
          and d2.abstain and d3.abstain)
    check("decision layer: noisy-OR of validated items, abstains when evidence is "
          "insufficient or conflicting", "0.80 / abstain / abstain",
          f"{d1.probability:.2f} / {d2.abstain} / {d3.abstain}", ok)


# ============================================================================ OOF protocol
def test_oof_protocol():
    from cosmos.ai.telemetry import oof
    from cosmos.ai.telemetry.event_bootstrap import _blockify, _event_table, _ewf, _sample
    from cosmos.ai.telemetry.oof_benchmark import aci_hot, check_freeze
    f = oof.fold_bounds(1003, 5)
    ok = f[0][0] == 0 and f[-1][1] == 1003 and all(a[1] == b[0] for a, b in zip(f, f[1:]))
    check("OOF folds tile the training half contiguously", "0..1003, no gaps",
          f"{f[0][0]}..{f[-1][1]}", ok)
    # AR fit on an AR(1) series recovers its coefficient; persistence fallback when unusable
    rng = np.random.default_rng(0)
    z = np.zeros(20000)
    for t in range(1, z.size):
        z[t] = 0.9 * z[t - 1] + rng.normal()
    w = oof._ar_fit(z, np.ones(z.size, bool), rng)
    w0 = oof._ar_fit(z, np.zeros(z.size, bool), rng)
    check("ridge AR(8) recovers an AR(1) coefficient; no usable samples -> persistence",
          "0.9 / [1, 0, ...]", f"{w[0]:.3f} / {w0[:2]}", abs(w[0] - 0.9) < 0.02 and w0[0] == 1)
    # ACI tracks its target point-level false-alarm rate even under drift
    pool = [np.sort(rng.normal(size=5000))]
    S = rng.normal(size=(100000, 1)) + np.linspace(0, 3, 100000)[:, None]
    far = aci_hot(S, pool, 0.01).mean()
    check("ACI keeps the point-level alarm rate at alpha = 0.01 under drift", "0.010",
          f"{far:.4f}", abs(far - 0.01) < 0.001)
    # event-wise F0.5 re-implementation on a hand-made case: 1 TP event, 1 FP run
    ts = pd.date_range("2007-01-01", periods=100000, freq="30s")
    g = np.zeros(len(ts), bool)
    g[1000:1100] = True
    g[50000:50010] = True
    lab = pd.DataFrame({"ID": ["a"], "Channel": ["channel_1"], "StartTime": [ts[1050]],
                        "EndTime": [ts[1200]], "Category": ["Anomaly"]})
    t = _blockify(_event_table(1, g, ts, lab))
    v = _ewf(t, _sample(t, np.arange(t["nb"])))
    prec = 0.5 * (1 - 60 / (len(ts) - 151))
    exp = 1.25 * prec / (0.25 * prec + 1)
    check("bootstrap EW F0.5 = hand computation (TP 1, FP 1, TNR-corrected)",
          f"{exp:.4f}", f"{v:.4f}", abs(v - exp) < 1e-9)
    if _has("research_design", "preregistration", "freeze.json"):
        try:
            fid = check_freeze()["freeze_id"]
            check("pre-registration freeze: every frozen file unchanged", "match", fid, True)
        except SystemExit as e:
            check("pre-registration freeze: every frozen file unchanged", "match", str(e), False)


def test_event_model_causal():
    import torch
    from cosmos.ai.telemetry.event_model import EventConfig, build, features
    cfg = EventConfig(levels=5, d_model=16, window=256)
    m = build(cfg, 4, 7).eval()
    x = torch.randn(1, 4, 7, 300)
    x2 = x.clone()
    x2[..., 200:] += 5.0
    with torch.no_grad():
        a, b = m(x), m(x2)
    before = float((a[..., :200] - b[..., :200]).abs().max())
    after = float((a[..., 200:] - b[..., 200:]).abs().max())
    check("event TCN is causal across channels (a change at t=200 leaves t<200 unchanged)",
          "0 before, > 0 after", f"{before:.1e} / {after:.2f}", before == 0.0 and after > 0)
    Z = np.random.default_rng(1).normal(size=(500, 3))
    S = np.abs(Z)
    F1 = features(Z, S, 30)
    Z2, S2 = Z.copy(), S.copy()
    Z2[400:] += 10
    S2[400:] += 10
    F2 = features(Z2, S2, 30)
    check("event features are causal (trailing means only)", "identical before t=400",
          f"max diff {np.abs(F1[:400] - F2[:400]).max():.1e}",
          np.abs(F1[:400] - F2[:400]).max() == 0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        print(f"\n{fn.__name__.removeprefix('test_').upper()}")
        try:
            fn()
        except Exception as exc:                              # report, keep going
            RESULTS.append({"check": fn.__name__, "status": "ERROR", "note": repr(exc)[:300]})
            print(f"  [ERROR] {fn.__name__}: {exc!r}")
    df = pd.DataFrame(RESULTS)
    n = df.status.value_counts().to_dict()
    print("\n" + "=" * 74 + f"\n{n.get('PASS', 0)} passed, {n.get('FAIL', 0)} failed, "
          f"{n.get('SKIP', 0)} skipped, {n.get('ERROR', 0)} errors")
    return df


if __name__ == "__main__":
    df = main()
    sys.exit(0 if not (df.status.isin(["FAIL", "ERROR"])).any() else 1)
