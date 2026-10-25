"""FEATURE 3 (e) — Monte Carlo dispersion and conformal impact zones.

A single predicted impact point is meaningless: fragments from one break-up span four
decades of ballistic coefficient and land thousands of kilometres apart.  So the output is a
region, and the region needs a guarantee.

The usual choice — a Gaussian covariance ellipse — assumes a shape the dispersion does not
have.  Fragment footprints are skewed downrange and heavier in the tail than an ellipse
allows, so a nominal 99 % ellipse does not contain 99 % of the impacts.  A conformal zone
instead calibrates a threshold on held-out samples, which gives finite-sample coverage under
nothing stronger than exchangeability.

    fp = monte_carlo_footprint(hazard, cloud)
    zone_comparison(fp)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import R_EARTH, enu_basis, geodetic_to_ecef, great_circle_km, rng_of
from .atmosphere import US76
from .drag import SHAPES
from .propagator import WindField, entry_state, propagate_ensemble


@dataclass
class DispersionConfig:
    """What gets randomised, and by how much."""
    n_samples: int = 1500
    bc_sigma_log10: float = 0.20         # ballistic-coefficient spread
    density_sigma_log10: float = 0.08    # atmospheric density uncertainty
    gamma_sigma_deg: float = 0.15        # flight-path angle
    wind_sigma_ms: float = 12.0
    dv_scale: float = 1.0                # multiplier on the break-up delta-v
    seed: int = 0


@dataclass
class Footprint:
    """Impact points and their summary statistics."""
    frame: pd.DataFrame
    entry_lat: float
    entry_lon: float
    config: DispersionConfig = field(default=None)

    @property
    def latlon(self) -> np.ndarray:
        return self.frame[["lat", "lon"]].values

    def downrange_crossrange_km(self) -> np.ndarray:
        """Impacts in a local downrange / crossrange frame [km], for zone fitting.

        Working in kilometres on a local tangent plane keeps the geometry Euclidean, which is
        what both the ellipse and the density estimate assume.
        """
        lat0, lon0 = float(self.frame.lat.mean()), float(self.frame.lon.mean())
        dlat = np.radians(self.frame.lat.values - lat0)
        dlon = np.radians(self.frame.lon.values - lon0)
        north = dlat * (R_EARTH / 1e3)
        east = dlon * (R_EARTH / 1e3) * np.cos(np.radians(lat0))
        return np.column_stack([east, north])

    def summary(self) -> dict:
        f = self.frame
        dr = f.downrange_km.values
        return {"n_impacts": len(f), "downrange_mean_km": round(float(dr.mean()), 1),
                "downrange_std_km": round(float(dr.std()), 1),
                "downrange_min_km": round(float(dr.min()), 1),
                "downrange_max_km": round(float(dr.max()), 1),
                "downrange_skew": round(float(_skew(dr)), 3),
                "cep50_km": round(float(np.median(_radius(f))), 1),
                "r95_km": round(float(np.percentile(_radius(f), 95)), 1),
                "impact_speed_median_ms": round(float(f.speed_ms.median()), 1),
                "max_energy_J": round(float(f.energy_J.max()), 0),
                "casualty_capable": int((f.energy_J > 15.0).sum())}


def _radius(f: pd.DataFrame) -> np.ndarray:
    return great_circle_km(f.lat.mean(), f.lon.mean(), f.lat.values, f.lon.values)


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    s = x.std()
    return 0.0 if s == 0 else float(np.mean((x - x.mean()) ** 3) / s ** 3)


def monte_carlo_footprint(hazard, cloud, config: DispersionConfig | None = None,
                          entry_speed_ms: float = 7600.0, flight_path_deg: float = -1.4,
                          azimuth_deg: float = 140.0, lat_deg: float = 13.72,
                          lon_deg: float = 80.23, atmo=US76) -> Footprint:
    """Propagate a dispersed fragment population from a break-up state to the ground.

    The break-up altitude, the ballistic-coefficient spread and the delta-v scale all come
    from the health-derived :class:`~cosmos.health.hazard.BreakupHazard`, which is what makes
    the footprint depend on *which* subsystem failed.
    """
    cfg = config or DispersionConfig()
    rng = rng_of(cfg.seed)
    n = cfg.n_samples

    frag = cloud.frame.sample(n=n, replace=True, random_state=cfg.seed).reset_index(drop=True)
    # ballistic coefficient: the cloud's own spread, widened by the health-driven sigma
    bc = frag.bc.values * 10.0 ** rng.normal(0.0, hazard.bc_sigma_log10, n)
    mass = frag.mass_kg.values
    cd_h = np.full(n, SHAPES["tumbling"]["cd_hyper"])
    cd_s = np.full(n, SHAPES["tumbling"]["cd_sub"])
    area = mass / (cd_h * np.maximum(bc, 1e-6))
    length = np.sqrt(np.maximum(area, 1e-9))

    st = entry_state(hazard.altitude_km * 1e3, entry_speed_ms,
                     flight_path_deg + rng.normal(0.0, cfg.gamma_sigma_deg),
                     azimuth_deg, lat_deg, lon_deg)
    R = np.tile(st.r, (n, 1))
    V = np.tile(st.v, (n, 1))

    # break-up delta-v, isotropic
    dv = frag.dv_ms.values * cfg.dv_scale
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    V = V + dv[:, None] * u

    # per-sample flight-path perturbation and atmospheric density multiplier
    dens = 10.0 ** rng.normal(0.0, cfg.density_sigma_log10, n)

    res = propagate_ensemble(R, V, mass, area, cd_h, cd_s, length, atmo=atmo,
                             density_scale=dens, dt=0.5,
                             entry_lat=lat_deg, entry_lon=lon_deg)
    df = res.frame()
    df = df[df.impacted].reset_index(drop=True)
    df["bc"] = bc[:len(df)] if len(df) == n else df.bc
    return Footprint(df, lat_deg, lon_deg, cfg)


# ------------------------------------------------------------------------ prediction zones
def gaussian_ellipse_zone(xy: np.ndarray, alpha: float = 0.05) -> dict:
    """Classical covariance ellipse at nominal coverage ``1 - alpha``.

    The radius uses the exact chi-square quantile for two degrees of freedom,
    ``-2 ln(alpha)``, so any under-coverage is the fault of the Gaussian assumption and not
    of a table lookup.
    """
    mu = xy.mean(axis=0)
    cov = np.cov(xy.T)
    k2 = -2.0 * np.log(alpha)                      # chi-square(2) quantile
    return {"kind": "gaussian_ellipse", "mu": mu, "cov": cov, "k2": k2,
            "area_km2": float(np.pi * np.sqrt(max(np.linalg.det(cov), 1e-12)) * k2)}


def _mahalanobis(xy: np.ndarray, mu: np.ndarray, inv: np.ndarray) -> np.ndarray:
    d = xy - mu
    return np.einsum("ij,jk,ik->i", d, inv, d)


def _loo_kde_score(xy: np.ndarray, bw: float | None = None) -> tuple:
    """Leave-one-out kernel density at each point, and the bandwidth used.

    Leaving the point itself out of its own density estimate matters: including it adds a
    constant kernel peak to every score, which makes calibration points look denser than
    test points and costs real coverage.
    """
    n = xy.shape[0]
    if bw is None:                                  # Scott's rule
        bw = n ** (-1.0 / 6.0) * np.sqrt(np.mean(np.var(xy, axis=0)))
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    k = np.exp(-0.5 * d2 / bw ** 2)
    np.fill_diagonal(k, 0.0)                        # leave-one-out
    return k.sum(1) / ((n - 1) * 2 * np.pi * bw ** 2), bw


def conformal_zone(xy: np.ndarray, alpha: float = 0.05, method: str = "density",
                   calib_frac: float = 0.5, seed=0) -> dict:
    """Split-conformal prediction region with finite-sample coverage ``>= 1 - alpha``.

    ``method="ellipse"`` scores points by Mahalanobis distance, so the region stays an
    ellipse but its size is calibrated rather than assumed.  ``method="density"`` scores by
    (leave-one-out) kernel density, which lets the region take the shape of the dispersion.

    The quantile index is ``ceil((n + 1)(1 - alpha))``.  That ``+1`` is what makes the
    guarantee hold for a finite sample instead of only asymptotically.
    """
    rng = rng_of(seed)
    n = xy.shape[0]
    idx = rng.permutation(n)
    n_cal = int(round(calib_frac * n))
    cal, fit = xy[idx[:n_cal]], xy[idx[n_cal:]]

    if method == "ellipse":
        mu, cov = fit.mean(axis=0), np.cov(fit.T)
        inv = np.linalg.pinv(cov)
        s_cal = _mahalanobis(cal, mu, inv)
        q_idx = int(np.ceil((n_cal + 1) * (1 - alpha))) - 1
        q = np.sort(s_cal)[min(q_idx, n_cal - 1)]
        area = float(np.pi * np.sqrt(max(np.linalg.det(cov), 1e-12)) * q)
        return {"kind": "conformal_ellipse", "mu": mu, "cov": cov, "q": float(q),
                "area_km2": area, "n_cal": n_cal,
                "contains": lambda p: _mahalanobis(np.atleast_2d(p), mu, inv) <= q}

    # Density (highest-density region), estimated in WHITENED coordinates.
    #
    # An impact footprint is extremely anisotropic: fragments spread hundreds of kilometres
    # downrange and only tens across it.  A single isotropic bandwidth then has to be wide
    # enough for the long axis, which over-smooths the short one and inflates the region by
    # more than an order of magnitude — measured at 25x the area of the Gaussian ellipse
    # before this was fixed.  Whitening by the covariance makes the bandwidth adapt to the
    # shape, and the Jacobian |det L| carries the area back to real kilometres.
    mu = fit.mean(axis=0)
    cov = np.cov(fit.T) + np.eye(2) * 1e-9
    L = np.linalg.cholesky(cov)
    Linv = np.linalg.inv(L)
    det_L = float(abs(np.linalg.det(L)))

    def white(p):
        return (np.atleast_2d(p) - mu) @ Linv.T

    zfit = white(fit)
    _dens_fit, bw = _loo_kde_score(zfit)

    def density_z(p):
        z = white(p)
        d2 = ((z[:, None, :] - zfit[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / bw ** 2).sum(1) / (zfit.shape[0] * 2 * np.pi * bw ** 2)

    s_cal = density_z(cal)
    q_idx = int(np.ceil((n_cal + 1) * alpha)) - 1               # LOW quantile of density
    q = np.sort(s_cal)[max(min(q_idx, n_cal - 1), 0)]
    # area by Monte Carlo in whitened space, then scaled back by the Jacobian
    lo, hi = zfit.min(0) - 4 * bw, zfit.max(0) + 4 * bw
    gz = rng.uniform(lo, hi, size=(20000, 2))
    d2g = ((gz[:, None, :] - zfit[None, :, :]) ** 2).sum(-1)
    dg = np.exp(-0.5 * d2g / bw ** 2).sum(1) / (zfit.shape[0] * 2 * np.pi * bw ** 2)
    frac = float(np.mean(dg >= q))
    area = frac * float(np.prod(hi - lo)) * det_L
    return {"kind": "conformal_density", "q": float(q), "bw": float(bw), "area_km2": area,
            "n_cal": n_cal, "contains": lambda p: density_z(p) >= q}


def zone_comparison(fp: Footprint, alphas=(0.10, 0.05, 0.01), n_splits: int = 12,
                    seed=0) -> pd.DataFrame:
    """Measure the coverage each zone actually achieves, averaged over random splits.

    For each split the zone is built on half the impacts and its coverage measured on the
    other half, so the number reported is out-of-sample.
    """
    xy = fp.downrange_crossrange_km()
    rng = rng_of(seed)
    rows = []
    for alpha in alphas:
        acc = {k: [] for k in ("gauss_cov", "gauss_area", "ell_cov", "ell_area",
                               "dens_cov", "dens_area")}
        for s in range(n_splits):
            perm = rng.permutation(xy.shape[0])
            half = xy.shape[0] // 2
            build, test = xy[perm[:half]], xy[perm[half:]]

            g = gaussian_ellipse_zone(build, alpha)
            inv = np.linalg.pinv(g["cov"])
            acc["gauss_cov"].append(float(np.mean(
                _mahalanobis(test, g["mu"], inv) <= g["k2"])))
            acc["gauss_area"].append(g["area_km2"])

            for meth, tag in (("ellipse", "ell"), ("density", "dens")):
                z = conformal_zone(build, alpha, meth, seed=int(rng.integers(1 << 30)))
                acc[f"{tag}_cov"].append(float(np.mean(z["contains"](test))))
                acc[f"{tag}_area"].append(z["area_km2"])
        rows.append({"nominal": 1 - alpha,
                     "gaussian_ellipse_coverage": round(np.mean(acc["gauss_cov"]), 4),
                     "gaussian_ellipse_area_km2": round(np.mean(acc["gauss_area"]), 0),
                     "conformal_ellipse_coverage": round(np.mean(acc["ell_cov"]), 4),
                     "conformal_ellipse_area_km2": round(np.mean(acc["ell_area"]), 0),
                     "conformal_density_coverage": round(np.mean(acc["dens_cov"]), 4),
                     "conformal_density_area_km2": round(np.mean(acc["dens_area"]), 0)})
    df = pd.DataFrame(rows)
    df["gaussian_shortfall_pts"] = ((df.nominal - df.gaussian_ellipse_coverage) * 100).round(2)
    df["density_area_saving_pct"] = ((1 - df.conformal_density_area_km2 /
                                     df.gaussian_ellipse_area_km2) * 100).round(1)
    return df


def ballistic_sweep(hazard, bc_values=(1.0, 10.0, 100.0, 500.0, 2000.0, 8000.0),
                    entry_speed_ms: float = 7600.0, flight_path_deg: float = -1.4,
                    lat_deg: float = 13.72, lon_deg: float = 80.23) -> pd.DataFrame:
    """One entry state, several ballistic coefficients — the footprint-width experiment."""
    st = entry_state(hazard.altitude_km * 1e3, entry_speed_ms, flight_path_deg,
                     140.0, lat_deg, lon_deg)
    n = len(bc_values)
    bc = np.asarray(bc_values, float)
    mass = np.full(n, 200.0)
    cd_h = np.full(n, SHAPES["tumbling"]["cd_hyper"])
    cd_s = np.full(n, SHAPES["tumbling"]["cd_sub"])
    area = mass / (cd_h * bc)
    res = propagate_ensemble(np.tile(st.r, (n, 1)), np.tile(st.v, (n, 1)), mass, area,
                             cd_h, cd_s, np.sqrt(area), dt=0.5,
                             entry_lat=lat_deg, entry_lon=lon_deg)
    df = res.frame()
    df["ballistic_coefficient"] = bc
    return df[["ballistic_coefficient", "downrange_km", "fall_time_s", "speed_ms",
               "lat", "lon", "impacted"]]
