"""FEATURE 3 (d) — 3-DOF re-entry propagation in the rotating ECEF frame.

Integrating in the Earth-fixed frame makes the Coriolis and centrifugal terms explicit,
which is what lets them be checked against closed-form answers instead of assumed.  Mass
loss from aerodynamic heating feeds back into the ballistic coefficient every step, so a
demising object flies a genuinely different trajectory rather than a rescaled one.

Two integrators share the same right-hand side:

``propagate``           one object, adaptive Dormand-Prince 5(4), stores the trajectory.
``propagate_ensemble``  thousands of fragments at once, vectorised fixed-step RK4.

    obj = DebrisObject("tank", mass=200, area=0.15, shape="cylinder")
    tr = propagate(entry_state(120e3, 7500.0, -1.5), obj)
    tr.summary()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import (G0, OMEGA_EARTH, ecef_to_geodetic, enu_basis, geodetic_altitude,
                      geodetic_to_ecef, gravity_ecef, gravity_ecef_vec, great_circle_km,
                      rng_of)
from .atmosphere import US76
from .drag import DragModel, cd_of

OMEGA_VEC = np.array([0.0, 0.0, OMEGA_EARTH])

#: material properties: melt temperature [K], heat of ablation [J/kg], emissivity
MATERIALS = {
    "aluminium":       dict(t_melt=933.0, h_abl=1.10e6, eps=0.14, density=2700.0),
    "stainless_steel": dict(t_melt=1700.0, h_abl=1.35e6, eps=0.35, density=7900.0),
    "titanium":        dict(t_melt=1941.0, h_abl=1.60e6, eps=0.30, density=4500.0),
    "inconel":         dict(t_melt=1610.0, h_abl=1.45e6, eps=0.38, density=8400.0),
    "cfrp":            dict(t_melt=800.0, h_abl=2.20e6, eps=0.85, density=1600.0),
    "mli":             dict(t_melt=600.0, h_abl=1.00e6, eps=0.80, density=100.0),
}


@dataclass
class DebrisObject:
    """One falling object."""
    name: str = "fragment"
    mass: float = 100.0                 # kg
    area: float = 0.5                   # m^2 reference cross-section
    shape: str = "tumbling"
    material: str = "aluminium"
    demisable: bool = True

    def __post_init__(self):
        if self.material not in MATERIALS:
            raise KeyError(f"unknown material {self.material!r}")
        self.drag = DragModel(self.shape, length=float(np.sqrt(max(self.area, 1e-6))))

    @property
    def ballistic_coefficient(self) -> float:
        """B = m / (C_D A) using the hypersonic C_D [kg/m^2]."""
        return self.mass / (self.drag.cd_hyper * self.area)


@dataclass
class State:
    """Entry state as ECEF position and velocity."""
    r: np.ndarray
    v: np.ndarray

    @property
    def altitude(self) -> float:
        return geodetic_altitude(self.r)


def entry_state(altitude_m: float = 120e3, speed_ms: float = 7500.0,
                flight_path_deg: float = -1.5, azimuth_deg: float = 140.0,
                lat_deg: float = 13.72, lon_deg: float = 80.23) -> State:
    """Build an entry state from the usual flight-mechanics quantities.

    Defaults place the break-up over Sriharikota, heading south-east.
    """
    r = geodetic_to_ecef(lat_deg, lon_deg, altitude_m)
    east, north, up = enu_basis(lat_deg, lon_deg)
    fp, az = np.radians(flight_path_deg), np.radians(azimuth_deg)
    horiz = np.cos(fp) * (np.sin(az) * east + np.cos(az) * north)
    v = speed_ms * (horiz + np.sin(fp) * up)
    return State(r, v)


@dataclass
class WindField:
    """Constant horizontal wind in the local East-North frame [m/s]."""
    zonal: float = 0.0
    meridional: float = 0.0

    def ecef(self, lat_deg, lon_deg):
        east, north, _up = enu_basis(lat_deg, lon_deg)
        return self.zonal * east + self.meridional * north


def _scalar(x) -> float:
    """Accept a float, a 0-d array or a size-1 array from any atmosphere model."""
    return float(np.asarray(x, float).reshape(-1)[0])


def _accel(r, v, obj: DebrisObject, mass: float, atmo, wind: WindField | None):
    """Right-hand side: gravity + Coriolis + centrifugal + drag, and the heating rate."""
    lat, lon, alt = ecef_to_geodetic(*r)
    alt = float(alt)
    rho, _t, _p, a_snd = atmo.state(max(alt, 0.0))
    rho, a_snd = _scalar(rho), _scalar(a_snd)
    v_rel = v.copy()
    if wind is not None and alt < 120e3:
        v_rel = v - wind.ecef(lat, lon)
    sp = float(np.linalg.norm(v_rel))
    mach = sp / max(a_snd, 1e-9)
    kn = _scalar(atmo.knudsen(max(alt, 0.0), obj.drag.length))
    cd = _scalar(cd_of(mach, kn, obj.drag.cd_hyper, obj.drag.cd_sub))

    q = 0.5 * rho * sp * sp                                         # dynamic pressure
    a_drag = -(q * cd * obj.area / max(mass, 1e-9)) * (v_rel / max(sp, 1e-9))
    a_grav = gravity_ecef(r)
    a_cor = -2.0 * np.cross(OMEGA_VEC, v)
    a_cen = -np.cross(OMEGA_VEC, np.cross(OMEGA_VEC, r))

    # Sutton-Graves stagnation-point convective heating [W/m^2]
    r_nose = max(np.sqrt(obj.area / np.pi), 0.02)
    qdot = 1.7415e-4 * np.sqrt(rho / r_nose) * sp ** 3
    return a_grav + a_cor + a_cen + a_drag, qdot, q, cd, alt, sp


def _mass_loss(qdot: float, obj: DebrisObject, mass: float, dt: float) -> float:
    """Lumped-mass ablation: net heat above re-radiation removes mass."""
    if not obj.demisable or mass <= 0.02 * obj.mass:
        return 0.0
    m = MATERIALS[obj.material]
    from ..common import SIGMA_SB
    q_rerad = m["eps"] * SIGMA_SB * m["t_melt"] ** 4
    net = max(qdot - q_rerad, 0.0)
    return net * obj.area * dt / m["h_abl"]


# Dormand-Prince 5(4) coefficients
_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_B4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


@dataclass
class Trajectory:
    """Result of one propagation."""
    name: str
    outcome: str                     # "impact" | "demised" | "timeout"
    t: np.ndarray = field(repr=False, default=None)
    alt: np.ndarray = field(repr=False, default=None)
    lat: np.ndarray = field(repr=False, default=None)
    lon: np.ndarray = field(repr=False, default=None)
    speed: np.ndarray = field(repr=False, default=None)
    mass: np.ndarray = field(repr=False, default=None)
    qdot: np.ndarray = field(repr=False, default=None)
    q: np.ndarray = field(repr=False, default=None)
    accel_g: np.ndarray = field(repr=False, default=None)
    entry_lat: float = 0.0
    entry_lon: float = 0.0
    ballistic_coefficient: float = 0.0

    @property
    def downrange_km(self) -> float:
        return float(great_circle_km(self.entry_lat, self.entry_lon, self.lat[-1],
                                     self.lon[-1]))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"t_s": self.t, "alt_km": self.alt / 1e3, "lat": self.lat,
                             "lon": self.lon, "speed_ms": self.speed, "mass_kg": self.mass,
                             "q_kPa": self.q / 1e3, "qdot_kW_m2": self.qdot / 1e3,
                             "decel_g": self.accel_g})

    def summary(self) -> dict:
        return {"object": self.name, "outcome": self.outcome,
                "B_kg_m2": round(self.ballistic_coefficient, 1),
                "fall_time_s": round(float(self.t[-1]), 1),
                "downrange_km": round(self.downrange_km, 1),
                "impact_lat": round(float(self.lat[-1]), 4),
                "impact_lon": round(float(self.lon[-1]), 4),
                "impact_speed_ms": round(float(self.speed[-1]), 1),
                "mass_remaining_frac": round(float(self.mass[-1] / self.mass[0]), 3),
                "peak_decel_g": round(float(np.nanmax(self.accel_g)), 2),
                "peak_q_kPa": round(float(np.nanmax(self.q)) / 1e3, 1),
                "peak_heat_MW_m2": round(float(np.nanmax(self.qdot)) / 1e6, 2),
                "impact_energy_J": round(0.5 * float(self.mass[-1]) *
                                         float(self.speed[-1]) ** 2, 1)}


def propagate(state: State, obj: DebrisObject, atmo=US76, wind: WindField | None = None,
              rtol: float = 1e-7, dt0: float = 0.5, t_max: float = 20000.0,
              store: bool = True) -> Trajectory:
    """Adaptive Dormand-Prince 5(4) propagation of a single object to the ground."""
    r, v, m, t, dt = state.r.astype(float), state.v.astype(float), float(obj.mass), 0.0, dt0
    lat0, lon0, _ = ecef_to_geodetic(*r)
    rec = {k: [] for k in ("t", "alt", "lat", "lon", "speed", "mass", "qdot", "q", "g")}
    outcome = "timeout"

    def rhs(rr, vv, mm):
        a, qd, q, _cd, alt, sp = _accel(rr, vv, obj, mm, atmo, wind)
        return a, qd, q, alt, sp

    def rk4(rr, vv, mm, h):
        """One classical RK4 step, used only to refine the ground crossing."""
        a1 = rhs(rr, vv, mm)[0]
        a2 = rhs(rr + 0.5 * h * vv, vv + 0.5 * h * a1, mm)[0]
        a3 = rhs(rr + 0.5 * h * (vv + 0.5 * h * a1), vv + 0.5 * h * a2, mm)[0]
        a4 = rhs(rr + h * (vv + 0.5 * h * a2), vv + h * a3, mm)[0]
        return (rr + h * (vv + h * (a1 + a2 + a3) / 6.0),
                vv + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0)

    while t < t_max:
        a0, qd0, q0, alt0, sp0 = rhs(r, v, m)
        if store:
            la, lo, _ = ecef_to_geodetic(*r)
            rec["t"].append(t); rec["alt"].append(alt0); rec["lat"].append(float(la))
            rec["lon"].append(float(lo)); rec["speed"].append(sp0); rec["mass"].append(m)
            rec["qdot"].append(qd0); rec["q"].append(q0)
            rec["g"].append(float(np.linalg.norm(a0 - gravity_ecef(r))) / G0)
        if alt0 <= 0.0:
            outcome = "impact"
            break
        if m <= 0.02 * obj.mass:
            outcome = "demised"
            break

        r_prev, v_prev, t_prev = r.copy(), v.copy(), t

        # one adaptive step
        for _ in range(40):
            ks_r, ks_v = [], []
            for i in range(7):
                rr = r + dt * sum(c * ks_r[j] for j, c in enumerate(_A[i])) if i else r.copy()
                vv = v + dt * sum(c * ks_v[j] for j, c in enumerate(_A[i])) if i else v.copy()
                acc, _qd, _q, _al, _sp = rhs(rr, vv, m)
                ks_r.append(vv)
                ks_v.append(acc)
            KR, KV = np.array(ks_r), np.array(ks_v)
            r5, v5 = r + dt * _B5 @ KR, v + dt * _B5 @ KV
            r4, v4 = r + dt * _B4 @ KR, v + dt * _B4 @ KV
            err = max(np.linalg.norm(r5 - r4) / max(np.linalg.norm(r5), 1.0),
                      np.linalg.norm(v5 - v4) / max(np.linalg.norm(v5), 1.0))
            if err <= rtol or dt <= 1e-4:
                break
            dt *= max(0.2, 0.9 * (rtol / max(err, 1e-18)) ** 0.2)
        dt_used = dt
        m = max(m - _mass_loss(qd0, obj, m, dt_used), 0.0)

        # refine the ground crossing by bisecting the step that crossed it
        if geodetic_altitude(r5) <= 0.0:
            lo_h, hi_h = 0.0, dt_used
            for _ in range(50):
                mid = 0.5 * (lo_h + hi_h)
                r_try, _v_try = rk4(r_prev, v_prev, m, mid)
                if geodetic_altitude(r_try) > 0.0:
                    lo_h = mid
                else:
                    hi_h = mid
            r, v = rk4(r_prev, v_prev, m, 0.5 * (lo_h + hi_h))
            t = t_prev + 0.5 * (lo_h + hi_h)
            if store:
                la, lo, al = ecef_to_geodetic(*r)
                rec["t"].append(t); rec["alt"].append(max(float(al), 0.0))
                rec["lat"].append(float(la)); rec["lon"].append(float(lo))
                rec["speed"].append(float(np.linalg.norm(v))); rec["mass"].append(m)
                rec["qdot"].append(rec["qdot"][-1]); rec["q"].append(rec["q"][-1])
                rec["g"].append(rec["g"][-1])
            outcome = "impact"
            break

        r, v, t = r5, v5, t + dt_used
        dt = min(dt_used * min(4.0, 0.9 * (rtol / max(err, 1e-18)) ** 0.2), 5.0)

    arr = {k: np.array(vals, float) for k, vals in rec.items()}
    return Trajectory(obj.name, outcome, arr["t"], arr["alt"], arr["lat"], arr["lon"],
                      arr["speed"], arr["mass"], arr["qdot"], arr["q"], arr["g"],
                      float(lat0), float(lon0), obj.ballistic_coefficient)


# ----------------------------------------------------------------- vectorised ensemble
@dataclass
class EnsembleResult:
    """Impact conditions for a whole population of fragments."""
    lat: np.ndarray
    lon: np.ndarray
    speed: np.ndarray
    mass: np.ndarray
    fall_time: np.ndarray
    impacted: np.ndarray
    bc: np.ndarray
    entry_lat: float
    entry_lon: float

    @property
    def n(self) -> int:
        return int(self.impacted.sum())

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"lat": self.lat, "lon": self.lon, "speed_ms": self.speed,
                             "mass_kg": self.mass, "fall_time_s": self.fall_time,
                             "bc": self.bc, "impacted": self.impacted,
                             "energy_J": 0.5 * self.mass * self.speed ** 2,
                             "downrange_km": great_circle_km(self.entry_lat, self.entry_lon,
                                                             self.lat, self.lon)})


def propagate_ensemble(states_r: np.ndarray, states_v: np.ndarray, mass: np.ndarray,
                       area: np.ndarray, cd_hyper: np.ndarray, cd_sub: np.ndarray,
                       length: np.ndarray, atmo=US76, density_scale: np.ndarray | None = None,
                       dt: float = 0.5, t_max: float = 20000.0,
                       entry_lat: float = 0.0, entry_lon: float = 0.0) -> EnsembleResult:
    """Fixed-step RK4 propagation of ``N`` fragments simultaneously.

    Every fragment carries its own mass, area, drag law and density multiplier.  Fragments
    drop out of the integration as they reach the ground, so the cost falls as the run
    proceeds.  No mass loss here: the Monte Carlo already samples the ballistic coefficient
    spread that demise would produce, and keeping the ensemble mass fixed makes the
    dispersion attributable to one cause at a time.
    """
    R, V = states_r.astype(float).copy(), states_v.astype(float).copy()
    n = R.shape[0]
    mass, area = np.asarray(mass, float), np.asarray(area, float)
    cdh, cds = np.asarray(cd_hyper, float), np.asarray(cd_sub, float)
    ln = np.asarray(length, float)
    dsc = np.ones(n) if density_scale is None else np.asarray(density_scale, float)

    alive = np.ones(n, bool)
    out_lat = np.full(n, np.nan)
    out_lon = np.full(n, np.nan)
    out_sp = np.full(n, np.nan)
    out_t = np.full(n, np.nan)
    t = 0.0

    def acc(Rx, Vx, idx):
        lat, lon, alt = ecef_to_geodetic(Rx[:, 0], Rx[:, 1], Rx[:, 2])
        alt = np.maximum(alt, 0.0)
        rho, _tt, _pp, snd = atmo.state(alt)
        rho = np.atleast_1d(rho) * dsc[idx]
        snd = np.atleast_1d(snd)
        sp = np.linalg.norm(Vx, axis=1)
        kn = np.atleast_1d(atmo.mean_free_path(alt)) / np.maximum(ln[idx], 1e-6)
        cd = cd_of(sp / np.maximum(snd, 1e-9), kn, cdh[idx], cds[idx])
        adrag = -((0.5 * rho * sp * sp * cd * area[idx] / np.maximum(mass[idx], 1e-9))[:, None]
                  * Vx / np.maximum(sp, 1e-9)[:, None])
        agrav = gravity_ecef_vec(Rx)
        acor = -2.0 * np.cross(np.broadcast_to(OMEGA_VEC, Vx.shape), Vx)
        acen = -np.cross(np.broadcast_to(OMEGA_VEC, Rx.shape),
                         np.cross(np.broadcast_to(OMEGA_VEC, Rx.shape), Rx))
        return agrav + acor + acen + adrag, alt

    while t < t_max and alive.any():
        idx = np.flatnonzero(alive)
        r0, v0 = R[idx], V[idx]
        a1, alt0 = acc(r0, v0, idx)
        a2, _ = acc(r0 + 0.5 * dt * v0, v0 + 0.5 * dt * a1, idx)
        a3, _ = acc(r0 + 0.5 * dt * (v0 + 0.5 * dt * a1), v0 + 0.5 * dt * a2, idx)
        a4, _ = acc(r0 + dt * (v0 + 0.5 * dt * a2), v0 + dt * a3, idx)
        r_new = r0 + dt * (v0 + dt * (a1 + a2 + a3) / 6.0)
        v_new = v0 + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0

        _la, _lo, alt_new = ecef_to_geodetic(r_new[:, 0], r_new[:, 1], r_new[:, 2])
        # A break-up delta-v can put a fragment on a rising or escaping trajectory.  Left in
        # the integration its radius grows without bound until r**3 overflows, and the NaNs
        # then poison the whole vectorised step, so those samples are retired here instead.
        bad = ~np.isfinite(alt_new) | (alt_new > 3.0e6) | ~np.isfinite(r_new).all(axis=1)
        if bad.any():
            alive[idx[bad]] = False
        hit = (alt_new <= 0.0) & ~bad
        if hit.any():
            # linear interpolation in altitude to place the impact point
            f = alt0[hit] / np.maximum(alt0[hit] - alt_new[hit], 1e-9)
            r_hit = r0[hit] + f[:, None] * (r_new[hit] - r0[hit])
            la, lo, _ = ecef_to_geodetic(r_hit[:, 0], r_hit[:, 1], r_hit[:, 2])
            gi = idx[hit]
            out_lat[gi], out_lon[gi] = la, lo
            out_sp[gi] = np.linalg.norm(v0[hit] + f[:, None] * (v_new[hit] - v0[hit]), axis=1)
            out_t[gi] = t + f * dt
            alive[gi] = False
        keep = ~hit
        if keep.any():
            gi = idx[keep]
            R[gi], V[gi] = r_new[keep], v_new[keep]
        t += dt

    bc = mass / (cdh * np.maximum(area, 1e-12))
    return EnsembleResult(out_lat, out_lon, out_sp, mass, out_t, ~np.isnan(out_lat), bc,
                          entry_lat, entry_lon)
