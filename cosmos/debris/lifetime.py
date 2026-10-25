"""FEATURE 2 (e) — how long debris stays up, and the collision risk while it is there.

Two things are needed to link feature 2 to feature 3: an estimate of *when* an object in
orbit will come down, and a measure of how dangerous it is in the meantime.

The lifetime model is physics, not a fit: it integrates the energy lost to drag per orbit
using the US-76 density at perigee.  Its one free parameter is the area-to-mass ratio, which
:func:`calibrate_area_to_mass` fits against the real measured decay times from the
catalogue — accounting for censoring, because a third of the fragments have not come down
yet and discarding them biases the answer badly.

    orbital_lifetime(480.0, 0.02)        # years
    calibrate_area_to_mass(clouds)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..common import MU_EARTH, R_EARTH
from ..reentry.atmosphere import US76

#: reference cross-sectional area to mass ratio for catalogue debris [m^2/kg]
AM_DEFAULT = 0.02
#: energy per unit mass above which a collision is catastrophic (fragments both objects)
CATASTROPHIC_J_PER_G = 40.0


def orbital_lifetime(altitude_km: float | np.ndarray, area_to_mass: float | np.ndarray =
                     AM_DEFAULT, cd: float = 2.2) -> np.ndarray:
    """Years for a circular orbit to decay to 120 km, by orbit-averaged energy loss.

    Drag removes energy at a rate set by the density at the current altitude, and the
    semi-major axis shrinks accordingly.  Integrating ``da/dt`` downward in 5 km steps is
    accurate enough for a lifetime estimate and needs no fitted coefficient.
    """
    alt = np.atleast_1d(np.asarray(altitude_km, float))
    am = np.atleast_1d(np.asarray(area_to_mass, float))
    alt, am = np.broadcast_arrays(alt, am)
    out = np.zeros(alt.shape)
    step = 5.0e3
    for i in np.ndindex(alt.shape):
        a_m = float(alt[i]) * 1e3
        yrs = 0.0
        while a_m > 120e3 and yrs < 1e5:
            r = R_EARTH + a_m
            rho = float(np.asarray(US76.density(a_m)).reshape(-1)[0])
            v = np.sqrt(MU_EARTH / r)
            # da/dt = -(cd * A/m) * rho * v * r   for a near-circular orbit
            da_dt = -cd * float(am[i]) * rho * v * r
            if da_dt >= -1e-14:
                yrs = np.inf
                break
            yrs += (step / abs(da_dt)) / (365.25 * 86400.0)
            a_m -= step
        out[i] = yrs
    return out if out.size > 1 else float(out.reshape(-1)[0])


def lifetime_profile(alt_km=None, area_to_mass: float = AM_DEFAULT) -> pd.DataFrame:
    """Lifetime against altitude, for the notebook figure."""
    alt_km = np.arange(300.0, 1201.0, 50.0) if alt_km is None else np.asarray(alt_km, float)
    return pd.DataFrame({"alt_km": alt_km,
                         "lifetime_years": np.atleast_1d(orbital_lifetime(alt_km,
                                                                         area_to_mass))})


# -------------------------------------------------------------- calibration on real data
def _censored_loss(am: float, alt: np.ndarray, t: np.ndarray, obs: np.ndarray) -> float:
    """Squared error in log-lifetime, treating still-in-orbit fragments as lower bounds.

    A fragment that has not decayed tells us its lifetime is *at least* its current age.
    Scoring it as if it had decayed at that age is what biases a naive fit, so censored rows
    only contribute when the model predicts a lifetime SHORTER than the observed age.
    """
    # The 7,711 fragments share only a handful of distinct break-up altitudes, and
    # orbital_lifetime integrates downward in 5 km steps.  Evaluating it per fragment turns a
    # 60-point grid search into ~90 million integration steps; per UNIQUE altitude it is a
    # few hundred.
    ualt, inv = np.unique(alt, return_inverse=True)
    pred = np.atleast_1d(orbital_lifetime(ualt, am))[inv]
    lp, lt = np.log10(np.maximum(pred, 1e-6)), np.log10(np.maximum(t, 1e-6))
    resid = np.where(obs, lp - lt, np.minimum(lp - lt, 0.0))
    return float(np.mean(resid ** 2))


def calibrate_area_to_mass(clouds: pd.DataFrame, grid=None) -> dict:
    """Fit the single A/M parameter against real measured decay times.

    Returns both the censoring-aware fit and the naive one that throws the survivors away,
    so the size of the bias is visible rather than assumed.
    """
    # the grid must bracket the optimum: with an upper bound of 0.126 both fits pinned at the
    # boundary and reported an identical, meaningless "bias factor" of exactly 1.0
    grid = np.logspace(-2.6, 0.3, 140) if grid is None else np.asarray(grid, float)
    alt = clouds.event_altitude_km.values.astype(float)
    t = clouds.lifetime_years.values.astype(float)
    obs = clouds.observed.values.astype(bool)

    losses = [_censored_loss(a, alt, t, obs) for a in grid]
    best = float(grid[int(np.argmin(losses))])

    m = obs
    losses_n = [_censored_loss(a, alt[m], t[m], obs[m]) for a in grid]
    naive = float(grid[int(np.argmin(losses_n))])

    at_edge = bool(np.argmin(losses) in (0, len(grid) - 1))
    return {"area_to_mass_censored": round(best, 5), "optimum_at_grid_edge": at_edge,
            "area_to_mass_naive": round(naive, 5),
            "bias_factor": round(naive / best, 3),
            "n_fragments": int(len(clouds)), "n_observed": int(obs.sum()),
            "n_censored": int((~obs).sum()),
            "rms_log10_error": round(np.sqrt(_censored_loss(best, alt, t, obs)), 4),
            "grid": grid, "losses": np.array(losses)}


@lru_cache(maxsize=4096)
def _lifetime_cached(alt_km: float, area_to_mass: float, cd: float = 2.2) -> float:
    return float(np.atleast_1d(orbital_lifetime(alt_km, area_to_mass, cd))[0])


def validate_lifetimes(clouds: pd.DataFrame, area_to_mass: float) -> pd.DataFrame:
    """Modelled against measured median lifetime, per break-up event."""
    from .catalogue import kaplan_meier
    rows = []
    for name, g in clouds.groupby("event", sort=False):
        km = kaplan_meier(g.lifetime_years.values, g.observed.values)
        pred = float(np.atleast_1d(orbital_lifetime(float(g.event_altitude_km.iloc[0]),
                                                    area_to_mass))[0])
        rows.append({"event": name, "altitude_km": float(g.event_altitude_km.iloc[0]),
                     "fragments": len(g), "percent_decayed": round(100 * g.observed.mean(), 1),
                     "measured_median_years": round(km["median"], 3),
                     "modelled_years": round(pred, 3),
                     "ratio": round(pred / km["median"], 3) if km["median"] == km["median"]
                     else float("nan")})
    return pd.DataFrame(rows).sort_values("altitude_km").reset_index(drop=True)


# ------------------------------------------------------------------ collision environment
def collision_risk(density: pd.DataFrame, mean_mass_kg: float = 150.0,
                   cross_section_m2: float = 1.0, rel_speed_ms: float = 10000.0) -> pd.DataFrame:
    """Per-shell collision rate and a criticality index.

    Rate is the particle-in-a-box estimate ``n * sigma * v``.  Criticality compares the rate
    at which collisions *create* debris against the rate at which drag *removes* it: above
    one, a shell makes debris faster than it clears, which is the Kessler condition.
    """
    df = density.copy()
    sigma_km2 = cross_section_m2 * 1e-6
    v_km_yr = rel_speed_ms * 1e-3 * 3600.0 * 24.0 * 365.25
    df["collision_rate_per_year"] = df.density_per_km3 * sigma_km2 * v_km_yr
    df["mean_years_between_collisions"] = 1.0 / np.maximum(df.collision_rate_per_year, 1e-30)
    lt = np.array([_lifetime_cached(float(a), AM_DEFAULT) for a in df.alt_mid_km])
    df["debris_lifetime_years"] = lt
    # fragments produced per collision, from the 40 J/g catastrophic threshold
    frags = 0.1 * (mean_mass_kg * 1000.0 / CATASTROPHIC_J_PER_G) ** 0.75
    df["fragments_per_collision"] = round(float(frags), 1)
    # criticality = debris production rate / removal rate.  With N objects in the shell,
    # production = (per-object collision rate) x N x fragments-per-collision and removal
    # = N / lifetime, so N cancels.  Dividing by the count again drove every shell to ~1e-5.
    df["criticality"] = df.collision_rate_per_year * frags * df.debris_lifetime_years
    df["stability"] = np.where(df.criticality > 1.0, "SUPERCRITICAL", "stable")
    return df[["alt_mid_km", "count", "density_per_km3", "collision_rate_per_year",
               "mean_years_between_collisions", "debris_lifetime_years", "criticality",
               "stability"]]


def total_leo_collision_rate(risk: pd.DataFrame) -> dict:
    """Aggregate the per-shell rates into one LEO-wide expectation."""
    tot = float((risk.collision_rate_per_year * risk["count"]).sum())
    return {"collisions_per_year": round(tot, 5),
            "years_between_collisions": round(1.0 / max(tot, 1e-30), 1),
            "worst_shell_km": int(risk.loc[risk.criticality.idxmax(), "alt_mid_km"]),
            "max_criticality": round(float(risk.criticality.max()), 3),
            "supercritical_shells": int((risk.criticality > 1.0).sum())}
