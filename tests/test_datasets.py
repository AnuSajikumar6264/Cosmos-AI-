"""Checks for the real-data layer — loaders, metrics and the datasets themselves.

Run it directly (no pytest needed):

    python tests/test_datasets.py

As in ``test_verify.py``, every expected value comes from outside the code: a case built by
hand whose answer is known, an analytic result, or a published figure about the dataset.
Checks that need a large dataset run only if it is present (``python -m cosmos.datasets
status`` lists them); they are reported as SKIP otherwise, never as PASS.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmos import datasets
from cosmos.health import realdata as R
from cosmos.health.monitor import _ewma

RESULTS = []


def check(name, expected, measured, ok, note=""):
    RESULTS.append({"check": name, "expected": expected, "measured": measured,
                    "status": "PASS" if ok else "FAIL", "note": note})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         expected {expected}   "
          f"measured {measured}" + (f"   ({note})" if note else ""))
    return ok


def skip(name, why):
    RESULTS.append({"check": name, "expected": "-", "measured": "-", "status": "SKIP",
                    "note": why})
    print(f"  [SKIP] {name}   ({why})")


# ----------------------------------------------------------------- metrics, by construction
def test_event_metrics():
    # events at [10,20] and [50,60]; alarms hit the first, miss the second, plus one stray
    events = [(10, 20), (50, 60)]
    alarms = [(15, 18), (30, 33)]
    m = R.event_metrics(alarms, events)
    check("event-wise recall on a hand-built case", "0.500", f"{m['recall']:.3f}",
          abs(m["recall"] - 0.5) < 1e-12, "one of two events overlapped")
    check("event-wise precision on a hand-built case", "0.500", f"{m['precision']:.3f}",
          abs(m["precision"] - 0.5) < 1e-12, "one of two alarms is false")
    # two alarms 5 samples apart form one episode when the gap is 10, two when it is 3
    ep10 = R.episodes([(0, 4), (10, 12)], merge_gap=10)
    ep3 = R.episodes([(0, 4), (10, 12)], merge_gap=3)
    check("alarm episodes merge only within the gap", "1 then 2", f"{len(ep10)} then {len(ep3)}",
          len(ep10) == 1 and len(ep3) == 2)
    # merging must never turn an event BETWEEN two alarms into a detection
    m2 = R.event_metrics([(0, 4), (20, 24)], [(10, 12)], merge_gap=100)
    check("an event between two merged alarms is not 'detected'", "recall 0.0",
          f"recall {m2['recall']:.1f}", m2["recall"] == 0.0)
    seg = R.alarm_segments(np.r_[np.zeros(5), np.ones(4), np.zeros(5), np.ones(6)], 0.5, 5)
    check("persistence: a 4-sample run does not alarm, a 6-sample run does", "[(14, 19)]",
          str(seg), seg == [(14, 19)])


def test_scoring():
    a = np.random.default_rng(1).normal(size=20_000)
    d = float(np.max(np.abs(R.ewma(a, 0.08) - _ewma(a, 0.08))))
    check("vectorised EWMA equals the monitor's loop", "0", f"{d:.1e}", d < 1e-12)

    # AR(1) with phi = 0.9 and unit innovations: one-step residuals have unit spread
    rng = np.random.default_rng(2)
    x = np.zeros(60_000)
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    e = R.ar_residuals(x[:, None], 40_000)[40_000:, 0]
    s = float(np.nanstd(e))
    check("AR forecaster recovers the innovation spread of an AR(1)", "1.00 +/- 0.03",
          f"{s:.3f}", abs(s - 1.0) < 0.03, "residual std of x_t = 0.9 x_t-1 + N(0,1)")

    # a slow drift: the frozen baseline alarms on it, the trailing baseline does not
    n, cad = 60 * 24 * 12, 300.0                     # 60 days at 5 min
    t = np.arange(n)
    y = (rng.normal(0, 1, n) + 20.0 * t / n)[:, None]
    fit = R.fit_baseline(y[: n // 5])
    z_fixed = R.channel_z(y, fit)[-1000:, 0].mean()
    med, sig = R.rolling_baseline(y, cad, 30, 1, fit=fit)
    z_roll = R.channel_z(y, fit, 6.0, med, sig)[-1000:, 0].mean()
    check("trailing baseline absorbs a slow drift a frozen one alarms on",
          "frozen |z| > 6, trailing |z| < 2", f"{z_fixed:.1f} vs {z_roll:.1f}",
          z_fixed > 6 and z_roll < 2)


def test_safe_unpickler():
    # the shape of an ESA channel file: float32 values on an irregular datetime index
    idx = pd.DatetimeIndex(pd.to_datetime(["2000-01-01 00:00:02.805", "2000-01-01 00:01:32.805",
                                           "2000-01-01 00:03:02.805"]), name="datetime")
    df = pd.DataFrame({"channel_1": np.array([0.1, 0.2, 0.3], np.float32)}, index=idx)
    back = R.safe_read_pickle(pickle.dumps(df))
    check("safe unpickler reads a telemetry DataFrame", "identical", "identical"
          if back.equals(df) else "different", back.equals(df))

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo should-never-run",))
    try:
        R.safe_read_pickle(pickle.dumps(Evil()))
        refused = False
    except pickle.UnpicklingError:
        refused = True
    check("safe unpickler refuses a payload that would run a command", "refused",
          "refused" if refused else "LOADED", refused)


# ------------------------------------------------------------------- loaders, by construction
def test_imagery_conversion():
    from PIL import Image
    from cosmos.debris import realimagery as RI
    with tempfile.TemporaryDirectory() as tmp:
        src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
        os.makedirs(os.path.join(src, "train"))
        Image.new("RGB", (200, 100)).save(os.path.join(src, "train", "a.png"))
        # SPARK order [row_min, col_min, row_max, col_max]: rows 20-60, cols 40-120
        pd.DataFrame({"filename": ["a.png"], "class": ["debris"],
                      "bbox": ["[20, 40, 60, 120]"]}).to_csv(os.path.join(src, "train.csv"),
                                                              index=False)
        info = RI.convert_to_yolo("x", root=src, out=out, size=64, val_frac=0.0)
        lab = open(os.path.join(out, "train", "labels", "a.txt")).read().split()
        got = [round(float(v), 4) for v in lab[1:]]
        expect = [0.4, 0.4, 0.4, 0.4]               # cx=80/200 cy=40/100 w=80/200 h=40/100
        check("SPARK-style box converts to normalised YOLO", str(expect), str(got),
              got == expect and info["classes"] == ["debris"])
        coco = {"images": [{"id": 1, "file_name": "a.png"}],
                "categories": [{"id": 7, "name": "sat"}],
                "annotations": [{"image_id": 1, "category_id": 7, "bbox": [40, 20, 80, 40]}]}
        jp = os.path.join(tmp, "instances.json")
        json.dump(coco, open(jp, "w"))
        c = RI.read_coco_labels(jp).iloc[0]
        check("COCO box [x, y, w, h] reads as corners", "(40, 20, 120, 60)",
              str((c.x0, c.y0, c.x1, c.y1)), (c.x0, c.y0, c.x1, c.y1) == (40, 20, 120, 60))


def test_discos_parsing():
    from cosmos.reentry import discos as D
    pages = [{"data": [{"id": "1", "type": "fragmentation",
                        "attributes": {"eventType": "Propulsion related", "epoch": "2001"}},
                       {"id": "2", "type": "fragmentation",
                        "attributes": {"eventType": "Battery", "epoch": "2002"}}],
              "links": {"next": "/api/fragmentations?page[number]=2"}},
             {"data": [{"id": "3", "type": "fragmentation",
                        "attributes": {"eventType": "Collision", "epoch": "2009"}}],
              "links": {"next": None}}]
    real_get = D._get
    D._get = lambda url, token: pages.pop(0)
    try:
        recs = D.fetch("fragmentations", token="test", cache=False)
    finally:
        D._get = real_get
    check("DISCOS client follows JSON:API pagination", "3 records over 2 pages",
          f"{len(recs)} records", len(recs) == 3)
    t = D.fragmentation_causes(D.to_frame(recs))
    subs = dict(zip(t.event_type, t.subsystem))
    check("fragmentation causes map onto coupling subsystems",
          "propulsion->Propulsion System, battery->Power System, collision->external",
          str(subs), subs.get("Propulsion related") == "Propulsion System"
          and subs.get("Battery") == "Power System" and pd.isna(subs.get("Collision")))


def test_cell_area():
    from cosmos.reentry.population import R_MEAN_KM, cell_area_km2
    lat = np.linspace(90, -90, 721)
    total = float(cell_area_km2(lat, 0.25).sum() * 1440)
    sphere = 4 * np.pi * R_MEAN_KM ** 2
    check("grid cell areas sum to the sphere's area", f"{sphere / 1e6:.2f} M km2",
          f"{total / 1e6:.2f} M km2", abs(total - sphere) / sphere < 1e-9)


# ----------------------------------------------------------------------- the real datasets
def test_real_datasets():
    reg = datasets.REGISTRY
    if reg["smap_msl"].present():
        ch = R.load_smap_msl()
        sc = pd.Series([c.spacecraft for c in ch]).value_counts().to_dict()
        n_seq = {s: sum(len(c.anomalies) for c in ch if c.spacecraft == s) for s in sc}
        check("SMAP/MSL release: channels per spacecraft", "SMAP 55, MSL 27", str(sc),
              sc == {"SMAP": 55, "MSL": 27}, "Hundman et al. 2018")
        check("SMAP/MSL release: labelled anomaly sequences", "SMAP 69, MSL 36", str(n_seq),
              n_seq == {"SMAP": 69, "MSL": 36}, "Hundman et al. 2018")
    else:
        skip("SMAP/MSL release counts", "smap_msl not present")

    prep = os.path.join(reg["esa_ad"].path, "prepared", "mission1.npz")
    if os.path.isfile(prep):
        for m, (nch, nsub) in {1: (76, 4), 2: (100, 5), 3: (48, 3)}.items():
            ms = R.load_esa_mission(m)
            ok = ms.X.shape[1] == nch and len(set(ms.subsystems)) == nsub \
                and ms.t[0] == pd.Timestamp("2000-01-01")
            check(f"ESA mission {m}: channels / subsystems / start after conversion",
                  f"{nch} / {nsub} / 2000-01-01",
                  f"{ms.X.shape[1]} / {len(set(ms.subsystems))} / {ms.t[0].date()}", ok,
                  "the archive's own channels.csv")
    else:
        skip("ESA missions after conversion", "run: python -m cosmos.datasets prepare esa_ad")

    grid = os.path.join(datasets.REGISTRY["natural_earth"].path, "..",
                        "population_grid_0p25deg.npz")
    if os.path.isfile(grid):
        from cosmos.reentry.population import cell_area_km2, load_population_grid
        g = load_population_grid()
        total = float(g["people"].sum())
        check("GHS-POP 2020 world population", "7.84 billion +/- 1 %",
              f"{total / 1e9:.3f} billion", abs(total / 7.84e9 - 1) < 0.01,
              "UN World Population Prospects: 7.84 billion in 2020")
        area = cell_area_km2(np.linspace(90, -90, g["land_fraction"].shape[0] + 1), 0.25)
        lf = float((g["land_fraction"] * area[:, None]).sum() / (area.sum() * 1440))
        check("land fraction of Earth's surface", "0.29 +/- 0.02", f"{lf:.3f}",
              abs(lf - 0.29) < 0.02, "about 29 % of the surface is land")
    else:
        skip("population grid", "run: python -m cosmos.datasets prepare natural_earth")


def main():
    print(__doc__.splitlines()[0])
    for fn in (test_event_metrics, test_scoring, test_safe_unpickler, test_imagery_conversion,
               test_discos_parsing, test_cell_area, test_real_datasets):
        print(f"\n{fn.__name__.replace('test_', '').upper()}")
        fn()
    df = pd.DataFrame(RESULTS)
    n_pass = int((df.status == "PASS").sum())
    n_run = int((df.status != "SKIP").sum())
    print(f"\n{'=' * 74}\n{n_pass} / {n_run} checks passed"
          + (f"   ({int((df.status == 'SKIP').sum())} skipped: data not present)"
             if (df.status == "SKIP").any() else ""))
    if n_pass != n_run:
        print(df[df.status == "FAIL"].to_string(index=False))
    return df


if __name__ == "__main__":
    df = main()
    sys.exit(0 if not (df.status == "FAIL").any() else 1)
