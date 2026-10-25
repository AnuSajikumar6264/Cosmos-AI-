"""Shared constants, geodesy and small helpers used by all three features."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- physical constants
MU_EARTH = 3.986004418e14        # m^3 / s^2   geocentric gravitational constant
R_EARTH = 6378137.0              # m           WGS-84 semi-major axis
F_EARTH = 1.0 / 298.257223563    #             WGS-84 flattening
E2_EARTH = F_EARTH * (2 - F_EARTH)
OMEGA_EARTH = 7.2921159e-5       # rad / s     Earth rotation rate
J2_EARTH = 1.08262668e-3         #             second zonal harmonic
G0 = 9.80665                     # m / s^2     standard gravity
R_AIR = 287.05287                # J / (kg K)  specific gas constant, dry air
GAMMA_AIR = 1.4                  #             ratio of specific heats
SIGMA_SB = 5.670374419e-8        # W / (m^2 K^4)
KB = 1.380649e-23                # J / K
M_AIR = 28.9644e-3               # kg / mol
NA = 6.02214076e23               # 1 / mol

#: kinetic energy above which a falling fragment is treated as casualty-capable [J]
LETHAL_ENERGY_J = 15.0
#: accepted international gate on expected casualties per re-entry
EC_THRESHOLD = 1e-4


def package_root() -> str:
    """The ``Final_Project`` directory (parent of the ``cosmos`` package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_root() -> str:
    """``Final_Project/data`` — datasets are organised one folder per feature."""
    return os.path.join(package_root(), "data")


#: dataset layout, one branch per feature of the project
DATA_LAYOUT = {
    "health":  os.path.join("feature1_health"),
    "debris":  os.path.join("feature2_debris"),
    "reentry": os.path.join("feature3_reentry"),
}


def feature_dir(feature: str, *parts) -> str:
    """Path inside a feature's dataset folder, created on demand.

    >>> feature_dir("debris", "celestrak")      # .../data/feature2_debris/celestrak
    """
    if feature not in DATA_LAYOUT:
        raise KeyError(f"unknown feature {feature!r}; expected {sorted(DATA_LAYOUT)}")
    d = os.path.join(data_root(), DATA_LAYOUT[feature], *parts)
    os.makedirs(d, exist_ok=True)
    return d


def catalogue_dir() -> str:
    """Where the CelesTrak CSVs live. Falls back to the parent project if not copied yet."""
    local = os.path.join(data_root(), DATA_LAYOUT["debris"], "celestrak")
    if os.path.isfile(os.path.join(local, "satcat.csv")):
        return local
    parent = os.path.join(os.path.dirname(package_root()), "data", "celestrak")
    return parent if os.path.isfile(os.path.join(parent, "satcat.csv")) else local


def output_dir(*parts) -> str:
    d = os.path.join(package_root(), "outputs", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def rng_of(seed) -> np.random.Generator:
    """Accept ``None``, an int seed, or an existing Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ------------------------------------------------------------------------- geodesy
def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """WGS-84 geodetic -> Earth-Centred Earth-Fixed cartesian [m]."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    n = R_EARTH / np.sqrt(1.0 - E2_EARTH * sl * sl)
    return np.array([(n + alt_m) * cl * np.cos(lon),
                     (n + alt_m) * cl * np.sin(lon),
                     (n * (1.0 - E2_EARTH) + alt_m) * sl])


def ecef_to_geodetic(x, y, z):
    """ECEF -> (lat_deg, lon_deg, alt_m) using Bowring's iteration (mm accurate)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2_EARTH))
    for _ in range(6):                       # converges to < 1 mm in 3-4 passes
        sl = np.sin(lat)
        n = R_EARTH / np.sqrt(1.0 - E2_EARTH * sl * sl)
        lat = np.arctan2(z + E2_EARTH * n * sl, p)
    sl = np.sin(lat)
    n = R_EARTH / np.sqrt(1.0 - E2_EARTH * sl * sl)
    alt = p / np.maximum(np.cos(lat), 1e-12) - n
    return np.degrees(lat), np.degrees(lon), alt


def geodetic_altitude(r_ecef) -> float:
    """Altitude above the WGS-84 ellipsoid for one ECEF position [m]."""
    return float(ecef_to_geodetic(*np.asarray(r_ecef, float))[2])


def enu_basis(lat_deg, lon_deg):
    """East / North / Up unit vectors at a geodetic point, expressed in ECEF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sla, cla, slo, clo = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    east = np.array([-slo, clo, 0.0])
    north = np.array([-sla * clo, -sla * slo, cla])
    up = np.array([cla * clo, cla * slo, sla])
    return east, north, up


def great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance on a spherical Earth [km] (vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gravity_ecef(r):
    """Point-mass + J2 gravitational acceleration in ECEF [m/s^2]."""
    x, y, z = r
    rn = np.sqrt(x * x + y * y + z * z)
    pre = -MU_EARTH / rn ** 3
    zr2 = (z / rn) ** 2
    k = 1.5 * J2_EARTH * (R_EARTH / rn) ** 2
    return np.array([pre * x * (1.0 - k * (5.0 * zr2 - 1.0)),
                     pre * y * (1.0 - k * (5.0 * zr2 - 1.0)),
                     pre * z * (1.0 - k * (5.0 * zr2 - 3.0))])


def gravity_ecef_vec(R):
    """Vectorised J2 gravity for an (N, 3) array of ECEF positions."""
    rn = np.linalg.norm(R, axis=1)
    pre = -MU_EARTH / rn ** 3
    zr2 = (R[:, 2] / rn) ** 2
    k = 1.5 * J2_EARTH * (R_EARTH / rn) ** 2
    out = np.empty_like(R)
    out[:, 0] = pre * R[:, 0] * (1.0 - k * (5.0 * zr2 - 1.0))
    out[:, 1] = pre * R[:, 1] * (1.0 - k * (5.0 * zr2 - 1.0))
    out[:, 2] = pre * R[:, 2] * (1.0 - k * (5.0 * zr2 - 3.0))
    return out


@dataclass
class Table:
    """Tiny helper so notebook cells can print a dict of scalars as a neat table."""
    rows: dict

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({"quantity": list(self.rows), "value": list(self.rows.values())})

    def __repr__(self):
        w = max(len(k) for k in self.rows) if self.rows else 0
        return "\n".join(f"{k:<{w}}  {v}" for k, v in self.rows.items())
