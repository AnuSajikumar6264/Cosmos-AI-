"""Physics expert: orbit-averaged drag decay with NRLMSIS 2.1 densities.

For each object the semi-major axis ``a`` and eccentricity ``e`` are integrated forward from
the orbit at ``t0`` with King-Hele's orbit-averaged drag equations for an atmosphere that is
locally exponential about perigee (King-Hele 1987, *Satellite Orbits in an Atmosphere*;
Vallado 2013, sec. 9.6):

    Delta a / rev = -2 pi delta a^2 rho_p e^{-c} [I0 + 2e I1 + 3/4 e^2 (I0 + I2) + e^3/4 (3 I1 + I3)]
    Delta e / rev = -2 pi delta a   rho_p e^{-c} [I1 + e/2 (I0 + I2) - e^2/8 (5 I1 - I3)
                                                  - e^3/16 (5 I0 + 4 I2 - I4)]

with ``c = a e / H``, ``I_n`` modified Bessel functions of argument ``c``, ``rho_p`` and ``H``
the density and density scale height at perigee, and ``delta = F B``: ballistic coefficient
``B = C_D A/m`` (C_D = 2.2) times the atmospheric-rotation factor
``F = (1 - r_p omega_E cos i / v_p)^2``.  The series is truncated at e^3 and is valid only for
small eccentricity (its perigee rate is 0.68 of the exact value at e = 0.2 and changes sign
near e = 0.3), so for ``e > E_SWITCH`` the same orbit averages are integrated exactly over the
eccentric anomaly E (trapezoid rule on [0, pi], spectrally accurate for the periodic
integrand):

    Delta a / rev = -delta rho_p a^2 2 int_0^pi exp(c (cos E - 1)) (1 + e cos E)^{3/2} (1 - e cos E)^{-1/2} dE
    Delta r_p / rev = -delta rho_p a^2 (1 - e) 2 int_0^pi exp(c (cos E - 1)) ((1 + e cos E)/(1 - e cos E))^{1/2} (1 - cos E) dE

(King-Hele 1987, eqs. 3.21-3.22, with the exponential atmosphere about perigee); the second
integrand is non-negative, so drag can only lower the perigee.  Re-entry is declared when the
perigee height falls below 120 km; integration stops at the horizon cap.

Densities come from a table of NRLMSIS 2.1 (pymsis) values, averaged over latitude, local
time and season, as a function of altitude, F10.7 and Ap.  Solar activity during the
propagation is unknown at ``t0``; the expert holds the trailing 81-day F10.7 and Ap at ``t0``
constant ("persistence"), which is the information available at prediction time.  An
``oracle`` option replays the realised daily F10.7 for diagnosis only.

Area-to-mass ratio, the dominant uncertainty, is sampled per object:

* design value (mass and dimensions known): log-normal around the design A/m, 0.25 dex;
* no design value, intact object (payload or rocket stage): a design A/m drawn from the
  training-set distribution of design A/m of that class, then the same 0.25 dex spread;
* no design value, fragment or component: characteristic length ``L_c`` drawn from the
  training-set distribution of radar-derived ``L_c = sqrt(4 RCS / pi)`` for the class, and
  A/m from the NASA Standard Breakup Model distribution (Johnson et al. 2001, Adv. Space Res.
  28:1377), rocket-body or spacecraft branch from the fragment's parent.

An object's own radar cross-section is not used: in the catalogue snapshot it is missing far
more often for objects that decayed soon after ``t0`` (it was never published for them), so
its availability carries outcome information.  The class pools are built from training
objects only; the ``L_c`` pool is biased towards objects that lived long enough to be
measured (a population-level bias shared by every prediction, corrected by the learned
experts and the gate).

Random draws are reproducible per object: each object has its own generator seeded with
(seed, md5 of its catalogue key), and class-pool values are drawn through the pool's empirical
quantile function, so adding or removing other objects (or a few pool entries) does not
re-randomise an object's predictive distribution.

The result is a predictive distribution: Monte-Carlo samples of the capped time to re-entry.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd
from scipy.special import ive

from ...common import feature_dir

MU = 398600.4418                     # km^3/s^2
RE = 6378.137                        # km
OMEGA_E = 7.2921159e-5               # rad/s
CD = 2.2
H_REENTRY = 120.0                    # km
CAP_DAYS = 1826.25                   # 5 years
E_SWITCH = 0.05                      # exact orbit averages above this eccentricity
_NE = 257                            # quadrature nodes on [0, pi] (rel. error < 1e-12 up to c = 3000)
_EN = np.linspace(0.0, np.pi, _NE)
_WE = np.full(_NE, np.pi / (_NE - 1))
_WE[[0, -1]] /= 2
TABLE_H = np.concatenate([np.arange(100, 400, 5.0), np.arange(400, 1000, 10.0),
                          np.arange(1000, 2601, 25.0)])
TABLE_F = np.array([60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 230, 260, 300.0])
TABLE_AP = np.array([0, 4, 8, 15, 25, 40, 70.0])


# ------------------------------------------------------------------------------ density
def density_table(path: str | None = None) -> dict:
    """log density table [kg/m^3] on (h, F10.7, Ap), NRLMSIS 2.1, globally averaged."""
    path = path or os.path.join(feature_dir("reentry", "processed"), "msis21_density_table.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from pymsis import msis
    lats = np.array([-60, -40, -20, 0, 20, 40, 60.0])
    wlat = np.cos(np.deg2rad(lats))
    lons = np.arange(0, 360, 45.0)                       # 8 local times at fixed UT
    dates = np.array(["2010-01-15", "2010-04-15", "2010-07-15", "2010-10-15"], dtype="datetime64[m]")
    logrho = np.zeros((TABLE_H.size, TABLE_F.size, TABLE_AP.size))
    t0 = time.time()
    for i, f in enumerate(TABLE_F):
        for j, ap in enumerate(TABLE_AP):
            nd = dates.size
            out = msis.run(dates, lons, lats, TABLE_H, np.full(nd, f), np.full(nd, f),
                           np.full((nd, 7), ap), version=2.1)
            rho = np.asarray(out)[..., 0]                 # (date, lon, lat, alt)
            rho = np.average(rho.mean(axis=(0, 1)), axis=0, weights=wlat)   # -> (alt,)
            logrho[:, i, j] = np.log(rho)
    np.savez(path, h=TABLE_H, f=TABLE_F, ap=TABLE_AP, logrho=logrho,
             note="NRLMSIS 2.1 via pymsis; mean over 4 seasons x 8 longitudes x 7 latitudes (cos-weighted)")
    print(f"MSIS table built in {time.time() - t0:.0f} s -> {path}")
    return {"h": TABLE_H, "f": TABLE_F, "ap": TABLE_AP, "logrho": logrho}


class Density:
    """Trilinear interpolation of log density; returns rho [kg/m^3] and scale height H [km]."""

    def __init__(self, table: dict | None = None):
        t = table or density_table()
        self.h, self.f, self.ap, self.lr = t["h"], t["f"], t["ap"], t["logrho"]

    @staticmethod
    def _locate(grid, x):
        x = np.clip(x, grid[0], grid[-1])
        i = np.clip(np.searchsorted(grid, x) - 1, 0, grid.size - 2)
        w = (x - grid[i]) / (grid[i + 1] - grid[i])
        return i, w

    def __call__(self, h, f107, ap):
        ih, wh = self._locate(self.h, h)
        jf, wf = self._locate(self.f, f107)
        ka, wa = self._locate(self.ap, ap)
        lr = self.lr

        def at(ii):
            v00 = lr[ii, jf, ka] * (1 - wa) + lr[ii, jf, ka + 1] * wa
            v10 = lr[ii, jf + 1, ka] * (1 - wa) + lr[ii, jf + 1, ka + 1] * wa
            return v00 * (1 - wf) + v10 * wf
        l0, l1 = at(ih), at(ih + 1)
        logr = l0 * (1 - wh) + l1 * wh
        dh = self.h[ih + 1] - self.h[ih]
        H = np.where(l0 > l1, dh / np.maximum(l0 - l1, 1e-9), 1000.0)
        return np.exp(logr), H


# ------------------------------------------------------------------------------ A/m (SBM)
def _piece(x, lo, hi, vlo, vhi, mid):
    return np.where(x <= lo, vlo, np.where(x >= hi, vhi, mid(x)))


def sbm_log_am(lc: np.ndarray, rocket: np.ndarray, u: np.ndarray, zn: np.ndarray) -> np.ndarray:
    """log10(A/m [m^2/kg]) from the NASA SBM mixture for L_c > 11 cm (Johnson et al. 2001);
    ``u`` (uniform) picks the mixture component and ``zn`` (standard normal) the value.

    Coefficients checked against ESA's NASA-breakup-model-cpp (commit ddd96572); two defects
    of that implementation are not reproduced: the rocket-body sigma_2 middle branch is
    0.28 - 0.1636 (lambda + 1) (the code has -0.28), and the two components are sampled as a
    mixture (probability alpha), not as a weighted sum of two normal draws.  Below 11 cm the
    single-normal small-fragment law is used.
    """
    lam = np.log10(np.maximum(lc, 1e-3))
    rb = np.asarray(rocket, bool)
    a_rb = _piece(lam, -1.4, 0.0, 1.0, 0.5, lambda x: 1.0 - 0.3571 * (x + 1.4))
    a_sc = _piece(lam, -1.95, 0.55, 0.0, 1.0, lambda x: 0.3 + 0.4 * (x + 1.2))
    m1_rb = _piece(lam, -0.5, 0.0, -0.45, -0.9, lambda x: -0.45 - 0.9 * (x + 0.5))
    m1_sc = _piece(lam, -1.1, 0.0, -0.6, -0.95, lambda x: -0.6 - 0.318 * (x + 1.1))
    s1_rb = np.full_like(lam, 0.55)
    s1_sc = _piece(lam, -1.3, -0.3, 0.1, 0.3, lambda x: 0.1 + 0.2 * (x + 1.3))
    m2_rb = np.full_like(lam, -0.9)
    m2_sc = _piece(lam, -0.7, -0.1, -1.2, -2.0, lambda x: -1.2 - 1.333 * (x + 0.7))
    s2_rb = _piece(lam, -1.0, 0.1, 0.28, 0.1, lambda x: 0.28 - 0.1636 * (x + 1.0))
    s2_sc = _piece(lam, -0.5, -0.3, 0.5, 0.3, lambda x: 0.5 - (x + 0.5))
    alpha = np.where(rb, a_rb, a_sc)
    first = u < alpha
    mu = np.where(first, np.where(rb, m1_rb, m1_sc), np.where(rb, m2_rb, m2_sc))
    sd = np.where(first, np.where(rb, s1_rb, s1_sc), np.where(rb, s2_rb, s2_sc))
    big = mu + sd * zn
    mu_s = _piece(lam, -1.75, -1.25, -0.3, -1.0, lambda x: -0.3 - 1.4 * (x + 1.75))
    sd_s = np.where(lam <= -3.5, 0.2, 0.2 + 0.1333 * (lam + 3.5))
    small = mu_s + sd_s * zn                      # one law applies per draw, so zn is shared
    return np.where(lc < 0.11, small, big)


def _object_keys(df: pd.DataFrame) -> np.ndarray:
    """Stable 64-bit key per object (GCAT JCAT, else NORAD number, else row position)."""
    if "JCAT" in df:
        ids = df["JCAT"].astype(str)
    elif "norad" in df:
        ids = df["norad"].astype(str)
    else:
        ids = pd.Series(np.arange(len(df)).astype(str))
    return np.array([int(hashlib.md5(s.encode()).hexdigest()[:16], 16) for s in ids], dtype=np.uint64)


def _draws(df: pd.DataFrame, n: int, seed: int) -> tuple:
    """Per-object uniform and normal draws (each object has its own generator)."""
    m = len(df)
    u_pool, u_mix, zn = np.empty((m, n)), np.empty((m, n)), np.empty((m, n))
    for i, k in enumerate(_object_keys(df)):
        g = np.random.default_rng([seed, int(k)])
        u_pool[i], u_mix[i], zn[i] = g.random(n), g.random(n), g.standard_normal(n)
    return u_pool, u_mix, zn


def sample_log_am(df: pd.DataFrame, n: int, seed: int, prior: dict) -> tuple:
    """(n_obj, n) samples of log10(A/m) and the source used per object."""
    m = len(df)
    u_pool, u_mix, zn = _draws(df, n, seed)
    out = np.empty((m, n))
    src = np.empty(m, dtype=object)
    design = np.log10(df["am_design"].to_numpy(float))
    types = df["type1"].astype(str).to_numpy()
    parent_rb = df["parent_is_rocket"].to_numpy(bool) if "parent_is_rocket" in df else df["type1"].eq("R").to_numpy()
    rocket = np.where(types == "R", True, parent_rb)
    has_d = np.isfinite(design)
    out[has_d] = design[has_d, None] + 0.25 * zn[has_d]
    src[has_d] = "design"
    intact = ~has_d & np.isin(types, list(prior["am"]))
    for t in np.unique(types[intact]):
        sel = intact & (types == t)
        out[sel] = np.quantile(prior["am"][t], u_pool[sel]) + 0.25 * zn[sel]
        src[sel] = "class_prior_design"
    rest = ~has_d & ~intact
    for t in np.unique(types[rest]):
        sel = rest & (types == t)
        lcs = np.quantile(prior["lc"].get(t, prior["lc"]["all"]), u_pool[sel])
        out[sel] = sbm_log_am(lcs, np.repeat(rocket[sel, None], n, 1), u_mix[sel], zn[sel])
        src[sel] = "class_prior_sbm"
    return out, src


# ------------------------------------------------------------------------------ propagation
def decay_times(hp, ha, inc, log_am, f107, ap, cap_days=CAP_DAYS, dens: Density | None = None,
                f107_path=None, max_steps: int = 20000, ap_path=None, stop_at_cap: bool = False,
                return_state: bool = False):
    """Time [days] until perigee < 120 km, capped at ``cap_days`` (returned as cap if not reached).

    All inputs are flat arrays of equal length (one element per object-sample); ``cap_days`` may
    be a scalar or such an array.  ``f107_path`` / ``ap_path`` (optional) are callables
    (element index, elapsed days) -> F10.7 / Ap along the trajectory (realised history, or the
    oracle diagnostic).  ``stop_at_cap`` shortens the last step so that the integration ends
    exactly at the cap (needed when the state at the cap is used); ``return_state`` also returns
    the semi-major axis and eccentricity at the end and a re-entered flag.
    """
    dens = dens or Density()
    a = RE + (np.asarray(hp, float) + np.asarray(ha, float)) / 2
    e = np.clip((np.asarray(ha, float) - np.asarray(hp, float)) / (2 * a), 0.0, 0.95)
    inc = np.deg2rad(np.asarray(inc, float))
    B = CD * 10.0 ** np.asarray(log_am, float) * 1e-6            # m^2/kg -> km^2/kg
    f107 = np.asarray(f107, float)
    ap = np.asarray(ap, float)
    cap = np.broadcast_to(np.asarray(cap_days, float), a.shape)
    t = np.zeros_like(a)
    done = (a * (1 - e) - RE) < H_REENTRY
    reentered = done.copy()
    out = cap.copy()
    out[done] = 0.0
    active = np.flatnonzero(~done & (cap > 0))
    for _ in range(max_steps):
        if active.size == 0:
            break
        aa, ee = a[active], e[active]
        rp = aa * (1 - ee)
        hp_ = rp - RE
        ff = f107[active] if f107_path is None else f107_path(active, t[active])
        rho, H = dens(hp_, ff, ap[active] if ap_path is None else ap_path(active, t[active]))
        rho_km = rho * 1e9                                        # kg/m^3 -> kg/km^3
        vp = np.sqrt(MU * (1 + ee) / rp)                           # km/s at perigee
        F = (1 - rp * OMEGA_E * np.cos(inc[active]) / vp) ** 2
        delta = F * B[active]
        c = aa * ee / H
        i0, i1, i2, i3, i4 = (ive(k, c) for k in range(5))       # e^{-c} I_k(c)
        k = 2 * np.pi * delta * rho_km
        da = -k * aa ** 2 * (i0 + 2 * ee * i1 + 0.75 * ee ** 2 * (i0 + i2) + 0.25 * ee ** 3 * (3 * i1 + i3))
        de = -k * aa * (i1 + 0.5 * ee * (i0 + i2) - 0.125 * ee ** 2 * (5 * i1 - i3)
                        - ee ** 3 / 16 * (5 * i0 + 4 * i2 - i4))
        de = np.where(ee > 1e-6, de, 0.0)
        hi_e = np.flatnonzero(ee > E_SWITCH)
        if hi_e.size:
            ia, ip = _orbit_averages(ee[hi_e], c[hi_e])
            kk = delta[hi_e] * rho_km[hi_e]
            da[hi_e] = -kk * aa[hi_e] ** 2 * ia
            # Delta r_p = (1 - e) Delta a - a Delta e = -kk a^2 ip
            de[hi_e] = ((1 - ee[hi_e]) * da[hi_e] + kk * aa[hi_e] ** 2 * ip) / aa[hi_e]
        period_d = 2 * np.pi * np.sqrt(aa ** 3 / MU) / 86400.0
        # revolutions per step: perigee may drop by at most 2% of its height above 120 km (and
        # >= 1 km), and one step is at most 10 days
        dhp = -(da * (1 - ee) - aa * de)                           # perigee drop per rev [km] (>= 0)
        if (dhp < -1e-9 * np.maximum(np.abs(da), 1e-300)).any():
            raise FloatingPointError("drag raised the perigee: orbit-average formulas used outside their range")
        room = np.maximum(0.02 * (hp_ - H_REENTRY), 1.0)
        nrev = np.clip(room / np.maximum(dhp, 1e-12), 1.0, 10.0 / period_d)
        if stop_at_cap:
            nrev = np.minimum(nrev, (cap[active] - t[active]) / period_d)
        aa = aa + da * nrev
        ee = np.clip(ee + de * nrev, 0.0, 0.95)
        tt = t[active] + nrev * period_d
        a[active], e[active], t[active] = aa, ee, tt
        down = (aa * (1 - ee) - RE) < H_REENTRY
        fin = down | (tt >= cap[active] - (1e-9 if stop_at_cap else 0.0))
        out[active[fin]] = np.minimum(tt[fin], cap[active[fin]])
        reentered[active[down]] = True
        active = active[~fin]
    if return_state:
        return out, a, e, reentered
    return out


def _orbit_averages(e: np.ndarray, c: np.ndarray) -> tuple:
    """Exact drag orbit averages for an exponential atmosphere about perigee:
    Delta a = -delta rho_p a^2 ia, Delta r_p = -delta rho_p a^2 ip (see the module docstring)."""
    ia, ip = np.empty(e.size), np.empty(e.size)
    cos_e = np.cos(_EN)[None, :]
    for s in range(0, e.size, 8192):                               # bounded memory
        ee, cc = e[s:s + 8192, None], c[s:s + 8192, None]
        x = ee * cos_e
        w = np.exp(cc * (cos_e - 1.0)) * _WE[None, :]
        root = np.sqrt((1 + x) / (1 - x))
        ia[s:s + 8192] = 2 * np.sum(w * (1 + x) * root, axis=1)
        ip[s:s + 8192] = 2 * (1 - ee[:, 0]) * np.sum(w * root * (1 - cos_e), axis=1)
    return ia, ip


def class_prior_from(train: pd.DataFrame, min_n: int = 50) -> dict:
    """Training-set pools per object class: radar-derived characteristic length ``L_c`` [m]
    (for fragments and components without design values) and design log10 A/m (for intact
    payloads and rocket stages without design values).  Pools are sorted, for quantile draws."""
    lc = np.sqrt(4 * train["rcs_m2"] / np.pi)
    pr = {"lc": {"all": np.sort(lc.dropna().to_numpy())}, "am": {}}
    for t, s in lc.groupby(train["type1"]):
        s = s.dropna()
        if len(s) >= min_n:
            pr["lc"][t] = np.sort(s.to_numpy())
    am = np.log10(train["am_design"].astype(float))
    for t in ("P", "R"):
        s = am[train["type1"].eq(t) & np.isfinite(am)]
        if len(s) >= min_n:
            pr["am"][t] = np.sort(s.to_numpy())
    return pr


def daily_space_weather(sw: pd.DataFrame, end) -> tuple:
    """Trailing 81-day F10.7 and Ap (observed rows) for every day up to the day before ``end``,
    gaps forward-filled; returns (first day, f107 array, ap array)."""
    days = pd.date_range(sw.index[0], pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    s = sw[["f107_last81", "ap_last81"]].reindex(days).ffill().bfill()
    return days[0], s["f107_last81"].to_numpy(), s["ap_last81"].to_numpy()


def history_posterior(d: pd.DataFrame, prior: dict, as_of, sw: pd.DataFrame, n_out: int = 32, n_prior: int = 128,
                      seed: int = 0, min_ess: float = 4.0, n_jobs: int | None = None, chunk: int = 500) -> tuple:
    """Area-to-mass samples conditioned on an object's catalogue history (twin state estimation).

    Objects in orbit at ``as_of`` have survived since their canonical-orbit epoch ``t0_canonical``,
    and their orbit has evolved from (``perigee0``, ``apogee0``) to the current (``perigee``,
    ``apogee``).  ``n_prior`` prior samples of log10(A/m) (:func:`sample_log_am`) are propagated
    from the canonical orbit to ``as_of`` with the realised trailing space weather.  A sample that
    would have re-entered before ``as_of`` gets weight 0; the others are weighted by a Gaussian
    likelihood of the current perigee and apogee with sigma = 10 km + 0.3 x the observed change.
    Fallbacks: fewer than ``min_ess`` effective samples -> survival alone (weight 1 for every
    surviving sample); no survivor -> the 8 samples that survived longest.  ``n_out`` samples
    are drawn by systematic resampling with each object's own generator.  Objects without a
    usable history (no canonical epoch before ``as_of``) keep their prior samples.

    Objects are independent (own random streams), so they are processed in chunks on ``n_jobs``
    processes (default: environment variable COSMOS_PHYSICS_JOBS, else 16) with identical results.

    Returns (log_am (n_obj, n_out), info DataFrame with history_days, kind, ess)."""
    n_jobs = int(os.environ.get("COSMOS_PHYSICS_JOBS", "16")) if n_jobs is None else n_jobs
    if n_jobs > 1 and len(d) > chunk:
        from concurrent.futures import ProcessPoolExecutor
        parts = [d.iloc[i:i + chunk] for i in range(0, len(d), chunk)]
        with ProcessPoolExecutor(n_jobs) as ex:
            res = list(ex.map(history_posterior, parts, *[[x] * len(parts) for x in (prior, as_of, sw, n_out, n_prior, seed, min_ess)],
                              [1] * len(parts)))
        return np.concatenate([r[0] for r in res]), pd.concat([r[1] for r in res], ignore_index=True)
    m = len(d)
    as_of = pd.Timestamp(as_of)
    log_am0, _ = sample_log_am(d, n_prior, seed, prior)
    t0 = pd.to_datetime(d["t0_canonical"]).reset_index(drop=True)
    dt = ((as_of - t0).dt.total_seconds() / 86400.0).to_numpy()
    hp0, ha0 = d["perigee0"].to_numpy(float), d["apogee0"].to_numpy(float)
    hp1, ha1 = d["perigee"].to_numpy(float), d["apogee"].to_numpy(float)
    ok = np.isfinite(dt) & (dt > 0) & np.isfinite(hp0) & np.isfinite(ha0) & (hp0 > H_REENTRY)
    out = log_am0[:, :n_out].copy()
    kind = np.where(ok, "", "prior").astype(object)
    ess = np.full(m, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size:
        day0, f_d, ap_d = daily_space_weather(sw, as_of)
        start = np.repeat(((t0.iloc[idx] - day0).dt.days.to_numpy() - 1), n_prior)

        def at(arr):
            return lambda act, el: arr[np.clip(start[act] + el.astype(int), 0, arr.size - 1)]
        rep = lambda x: np.repeat(x[idx], n_prior)  # noqa: E731
        T, a, e, down = decay_times(rep(hp0), rep(ha0), rep(d["inc"].to_numpy(float)), log_am0[idx].ravel(),
                                    f_d[np.clip(start, 0, f_d.size - 1)], ap_d[np.clip(start, 0, ap_d.size - 1)],
                                    cap_days=rep(dt), f107_path=at(f_d), ap_path=at(ap_d), stop_at_cap=True,
                                    return_state=True)
        hp_p, ha_p = a * (1 - e) - RE, a * (1 + e) - RE
        sp = 10.0 + 0.3 * np.abs(rep(hp0) - rep(hp1))
        sa = 10.0 + 0.3 * np.abs(rep(ha0) - rep(ha1))
        ll = np.where(down, -np.inf, -0.5 * (((hp_p - rep(hp1)) / sp) ** 2 + ((ha_p - rep(ha1)) / sa) ** 2))
        ll, T, surv = ll.reshape(idx.size, n_prior), T.reshape(idx.size, n_prior), ~down.reshape(idx.size, n_prior)
        keys = _object_keys(d)
        for j, i in enumerate(idx):
            w = np.exp(ll[j] - ll[j].max()) if surv[j].any() else np.zeros(n_prior)
            ess[i] = w.sum() ** 2 / (w ** 2).sum() if w.sum() > 0 else 0.0
            if ess[i] >= min_ess:
                kind[i] = "history"
            elif surv[j].any():
                w, kind[i] = surv[j].astype(float), "survival_only"
            else:
                w = np.zeros(n_prior)
                w[np.argsort(T[j])[-8:]] = 1.0
                kind[i] = "longest_survivors"
            g = np.random.default_rng([seed + 7, int(keys[i])])
            u = (g.random() + np.arange(n_out)) / n_out
            pick = np.minimum(np.searchsorted(np.cumsum(w / w.sum()), u), n_prior - 1)
            out[i] = log_am0[i, pick]
    info = pd.DataFrame({"history_days": dt, "kind": kind, "ess": ess})
    return out, info


def predict(df: pd.DataFrame, prior: dict, n_samples: int = 32, seed: int = 0,
            cap_days: float = CAP_DAYS, oracle_f107=None, log_am: np.ndarray | None = None) -> dict:
    """Monte-Carlo physics predictive distribution for every row of ``df`` (``prior`` from
    :func:`class_prior_from` on the training split).  ``log_am`` optionally supplies the
    area-to-mass samples (e.g. from :func:`history_posterior`) instead of the prior draws."""
    if log_am is None:
        log_am, src = sample_log_am(df, n_samples, seed, prior)
    else:
        _, src = sample_log_am(df, 1, seed, prior)
        n_samples = log_am.shape[1]
    rep = lambda c: np.repeat(df[c].to_numpy(float), n_samples)  # noqa: E731
    f107 = np.nan_to_num(rep("f107_last81"), nan=150.0)
    ap = np.nan_to_num(rep("ap_last81"), nan=12.0)
    t0 = time.time()
    T = decay_times(rep("perigee"), rep("apogee"), rep("inc"), log_am.ravel(), f107, ap, cap_days,
                    f107_path=oracle_f107).reshape(len(df), n_samples)
    return {"samples_days": T, "am_source": src, "seconds": time.time() - t0, "log_am": log_am}
