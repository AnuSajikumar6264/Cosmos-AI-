"""Verification suite — every check has an answer that does not come from this code.

Run it directly (no pytest needed):

    python tests/test_verify.py

The point of each check is that the expected value is derived independently: a published
table, a closed-form solution, an analytic limit, or a property the physics must satisfy.
A test that only compares the code against itself proves nothing.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmos.common import (G0, MU_EARTH, OMEGA_EARTH, R_EARTH, ecef_to_geodetic,
                           geodetic_to_ecef, great_circle_km)
from cosmos.debris.detect import check_shift_sign
from cosmos.reentry.atmosphere import US76, geometric
from cosmos.reentry.breakup import sample_cloud, size_law_check
from cosmos.reentry.drag import CD_FREE_MOLECULAR, DragModel, cd_of
from cosmos.reentry.propagator import DebrisObject, State, entry_state, propagate

RESULTS = []


def check(name: str, expected, measured, ok: bool, note: str = ""):
    RESULTS.append({"check": name, "expected": expected, "measured": measured,
                    "pass": ok, "note": note})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n"
          f"         expected {expected}   measured {measured}"
          + (f"   ({note})" if note else ""))
    return ok


# --------------------------------------------------------------------------- atmosphere
def test_atmosphere():
    """US76 densities against the published table, indexed by GEOPOTENTIAL altitude."""
    ref = {0: 1.2250, 11000: 0.36392, 20000: 0.088035, 32000: 0.013225,
           47000: 0.0014275, 51000: 0.00086160, 71000: 6.4211e-5, 84852: 6.9578e-6}
    worst = 0.0
    for h, rho in ref.items():
        got = float(US76.density(float(geometric(h))))
        worst = max(worst, abs(got - rho) / rho * 100)
    check("US76 density vs published table", "< 0.01 %", f"{worst:.4f} %", worst < 0.01,
          "layer boundaries are geopotential, so the table must be indexed through the inverse")

    tref = {0: 288.15, 11000: 216.65, 32000: 228.65, 47000: 270.65, 71000: 214.65}
    dt = max(abs(float(US76.temperature(float(geometric(h)))) - t) for h, t in tref.items())
    check("US76 temperature vs published table", "0 K", f"{dt:.3f} K", dt < 0.01)

    # hydrostatic consistency: dp/dz must equal -rho g
    z = 30_000.0
    dz = 50.0
    dpdz = (float(US76.pressure(z + dz)) - float(US76.pressure(z - dz))) / (2 * dz)
    expect = -float(US76.density(z)) * G0
    err = abs(dpdz - expect) / abs(expect) * 100
    check("hydrostatic balance dp/dz = -rho g", "< 1 %", f"{err:.3f} %", err < 1.0)


# ---------------------------------------------------------------------------- geodesy
def test_geodesy():
    lat, lon, alt = 12.34, 77.65, 543_210.0
    la, lo, al = ecef_to_geodetic(*geodetic_to_ecef(lat, lon, alt))
    err = max(abs(la - lat), abs(lo - lon)) * 111_000.0 + abs(al - alt)
    check("WGS-84 geodetic round trip", "< 1e-6 m", f"{err:.2e} m", err < 1e-6)

    d = great_circle_km(0.0, 0.0, 0.0, 90.0)
    expect = 2 * np.pi * 6371.0 / 4
    check("great-circle quarter of the equator", f"{expect:.1f} km", f"{float(d):.1f} km",
          abs(float(d) - expect) < 1.0)


# ------------------------------------------------------------------------------- drag
def test_drag():
    cd_free = cd_of(20.0, 1e6, 1.2, 0.8)
    check("C_D at Kn -> infinity reaches the free-molecular limit",
          f"{CD_FREE_MOLECULAR}", f"{float(cd_free):.3f}",
          abs(float(cd_free) - CD_FREE_MOLECULAR) < 0.02)

    dm = DragModel("sphere", 0.5)
    cd_cont = float(dm.cd(mach=20.0, knudsen=1e-6))
    check("C_D at Kn -> 0, Mach 20 reaches the continuum hypersonic value",
          f"{dm.cd_hyper}", f"{cd_cont:.3f}", abs(cd_cont - dm.cd_hyper) < 0.05)

    transonic = float(dm.cd(mach=1.25, knudsen=1e-6))
    check("C_D peaks in the transonic regime above the hypersonic value",
          f"> {dm.cd_hyper}", f"{transonic:.3f}", transonic > dm.cd_hyper,
          "a sphere peaks near 1.0 against 0.92 hypersonic")


# -------------------------------------------------------------------------- propagator
def test_propagator():
    # terminal velocity: at impact, drag must balance gravity
    # a LOW ballistic coefficient is essential: B = 50/(0.47*0.05) = 2100 kg/m^2 never
    # reaches equilibrium inside 20 km of fall, so the "terminal velocity" it is compared
    # against is not the quantity being measured.  B ~ 11 asymptotes within a few hundred m.
    obj = DebrisObject("light sphere", mass=5.0, area=1.0, shape="sphere",
                       material="titanium", demisable=False)
    tr = propagate(entry_state(20_000.0, 1.0, -90.0), obj, rtol=1e-9)
    v = float(tr.speed[-1])
    rho = float(US76.density(0.0))
    kn = float(US76.knudsen(0.0, obj.drag.length))
    cd = float(cd_of(v / float(US76.sound_speed(0.0)), kn, obj.drag.cd_hyper, obj.drag.cd_sub))
    ratio = (0.5 * rho * v * v * cd * obj.area / obj.mass) / G0
    check("terminal velocity: drag / gravity at impact", "1.00", f"{ratio:.4f}",
          abs(ratio - 1.0) < 0.02, "a vertical drop must reach force balance")

    # Coriolis: an eastward deflection whose magnitude is bounded by the free-fall value
    dense = DebrisObject("dense", mass=5000.0, area=0.01, shape="sphere",
                         material="titanium", demisable=False)
    tr2 = propagate(entry_state(30_000.0, 0.001, -90.0, 0.0, 45.0, 0.0), dense, rtol=1e-9)
    tfall = float(tr2.t[-1])
    analytic = OMEGA_EARTH * np.cos(np.radians(45.0)) * G0 * tfall ** 3 / 3.0
    east_m = float(great_circle_km(45.0, 0.0, 45.0, tr2.lon[-1])) * 1000.0
    check("Coriolis deflection is eastward and below the free-fall value",
          f"0 < d < {analytic:.0f} m", f"{east_m:.0f} m",
          0.0 < east_m < analytic,
          "drag must REDUCE the deflection, so agreement would indicate a bug")

    # step-size convergence
    pts = []
    for rtol in (1e-6, 1e-9, 1e-12):
        t = propagate(entry_state(120e3, 7500.0, -2.0),
                      DebrisObject("tank", mass=200.0, area=0.15, shape="cylinder",
                                   material="stainless_steel", demisable=False), rtol=rtol)
        pts.append((float(t.lat[-1]), float(t.lon[-1])))
    move = float(great_circle_km(*pts[0], *pts[-1])) * 1000.0
    check("impact point converges as rtol tightens 1e-6 -> 1e-12", "< 1 m",
          f"{move:.3f} m", move < 1.0)

    # energy: a vacuum orbit must not decay
    class Vacuum:
        def state(self, z):
            z = np.atleast_1d(np.asarray(z, float))
            return np.zeros_like(z) + 1e-30, np.zeros_like(z) + 250.0, \
                np.zeros_like(z), np.zeros_like(z) + 300.0

        def knudsen(self, z, length):
            return np.atleast_1d(np.asarray(z, float)) * 0 + 1e6

        def mean_free_path(self, z):
            return np.atleast_1d(np.asarray(z, float)) * 0 + 1e6

    rc = R_EARTH + 800e3
    vc = np.sqrt(MU_EARTH / rc)
    sat = DebrisObject("sat", mass=1000.0, area=1.0, shape="sphere", demisable=False)
    tr3 = propagate(State(np.array([rc, 0.0, 0.0]),
                          np.array([0.0, vc - OMEGA_EARTH * rc, 0.0])),
                    sat, atmo=Vacuum(), rtol=1e-11, t_max=6000.0)
    alt_km = tr3.alt / 1e3
    j2_swing = float(alt_km.max() - alt_km.min())
    check("J2 makes a circular orbit oscillate in altitude", "15-25 km",
          f"{j2_swing:.1f} km", 15.0 < j2_swing < 25.0,
          "an 800 km circular orbit swings ~19 km peak-to-peak under J2")
    check("vacuum orbit does not decay", "no impact", tr3.outcome,
          tr3.outcome != "impact")


# ---------------------------------------------------------------------------- break-up
def test_breakup():
    cloud = sample_cloud(1800.0, "explosion", rng=7)
    check("break-up conserves the parent mass", "1.000",
          f"{cloud.mass_closure:.4f}", abs(cloud.mass_closure - 1.0) < 0.01)

    law = size_law_check(200_000, rng=3)
    # compare only bins with enough expected counts to be statistically meaningful
    ok_bins = law.expected_count >= 200
    rel = np.abs(law.sampled_fraction - law.analytic_fraction) / law.analytic_fraction
    worst = float(rel[ok_bins].max())
    check("sampled size distribution follows the truncated power law", "< 5 %",
          f"{worst * 100:.2f} %", worst < 0.05,
          f"{int(ok_bins.sum())} bins with >= 200 expected counts")


# --------------------------------------------------------------------------- detection
def test_detection():
    r = check_shift_sign()
    check("registration recovers a known shift with the right SIGN", "< 0.5 px",
          f"{r['correction_error_px']:.3f} px", r["correction_error_px"] < 0.5,
          "getting this backwards is invisible at small jitter and fatal at large")
    check("residual misalignment after registration", "< 0.3 px",
          f"{r['residual_after_registration_px']:.3f} px",
          r["residual_after_registration_px"] < 0.3)


# ------------------------------------------------------------------------------ zones
def test_conformal():
    """A conformal zone must reach its nominal coverage on data it did not see."""
    from cosmos.reentry.montecarlo import conformal_zone, gaussian_ellipse_zone
    rng = np.random.default_rng(0)
    # deliberately skewed, heavy-tailed 2-D sample — the case an ellipse gets wrong
    n = 4000
    xy = np.column_stack([rng.gamma(2.0, 60.0, n), rng.normal(0.0, 30.0, n)])
    build, test = xy[: n // 2], xy[n // 2:]
    for alpha in (0.10, 0.05):
        z = conformal_zone(build, alpha, "density", seed=1)
        cov = float(np.mean(z["contains"](test)))
        check(f"conformal density zone coverage at nominal {1 - alpha:.0%}",
              f">= {1 - alpha - 0.03:.2f}", f"{cov:.4f}", cov >= 1 - alpha - 0.03)
    g = gaussian_ellipse_zone(build, 0.05)
    inv = np.linalg.pinv(g["cov"])
    d = np.einsum("ij,jk,ik->i", test - g["mu"], inv, test - g["mu"])
    gcov = float(np.mean(d <= g["k2"]))
    check("Gaussian ellipse under-covers on a skewed sample", "< 0.95", f"{gcov:.4f}",
          gcov < 0.95, "this is the motivation for using a conformal zone at all")


def main():
    print(__doc__.splitlines()[0])
    for fn in (test_atmosphere, test_geodesy, test_drag, test_propagator, test_breakup,
               test_detection, test_conformal):
        print(f"\n{fn.__name__.replace('test_', '').upper()}")
        fn()
    import pandas as pd
    df = pd.DataFrame(RESULTS)
    n_pass = int(df["pass"].sum())
    print(f"\n{'=' * 74}\n{n_pass} / {len(df)} checks passed")
    if n_pass != len(df):
        print(df[~df["pass"]].to_string(index=False))
    return df


if __name__ == "__main__":
    df = main()
    sys.exit(0 if bool(df["pass"].all()) else 1)
