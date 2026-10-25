"""FEATURE 3 (a) — US Standard Atmosphere 1976.

Closed-form below 86 km geopotential altitude (seven gradient/isothermal layers), and a
smooth log-density table above it out to 1000 km.  Vectorised throughout, because the
Monte Carlo propagator asks for thousands of densities per step.

    US76.density(50_000)      # kg/m^3
    US76.state(80_000)        # (rho, T, p, sound speed)
"""
from __future__ import annotations

import numpy as np

from ..common import G0, GAMMA_AIR, KB, M_AIR, NA, R_AIR, R_EARTH

# base geopotential altitude [m], base temperature [K], lapse rate [K/m]
_LAYER_H = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0])
_LAYER_T = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
_LAYER_L = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])

# base pressures, derived once by integrating the hydrostatic equation upward
def _base_pressures():
    p = [101325.0]
    for i in range(len(_LAYER_H) - 1):
        dh = _LAYER_H[i + 1] - _LAYER_H[i]
        t0, lam = _LAYER_T[i], _LAYER_L[i]
        if lam == 0.0:
            p.append(p[i] * np.exp(-G0 * dh / (R_AIR * t0)))
        else:
            p.append(p[i] * (1.0 + lam * dh / t0) ** (-G0 / (R_AIR * lam)))
    return np.array(p)


_LAYER_P = _base_pressures()

# upper atmosphere: log10(density) sampled on a 2 km grid from 86 to 1000 km.  Values follow
# the US76 upper tables; between nodes we interpolate in log density, which is where the
# profile is closest to linear.
_UP_H = np.array([86, 90, 95, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300, 350,
                  400, 450, 500, 600, 700, 800, 900, 1000], float) * 1e3
_UP_RHO = np.array([6.958e-6, 3.416e-6, 1.393e-6, 5.604e-7, 9.708e-8, 2.222e-8, 8.152e-9,
                    3.831e-9, 2.076e-9, 1.233e-9, 5.194e-10, 2.541e-10, 6.073e-11,
                    1.916e-11, 7.014e-12, 2.803e-12, 1.184e-12, 5.215e-13, 1.137e-13,
                    3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])
_UP_T = np.array([186.9, 186.9, 186.9, 195.1, 240.0, 360.0, 469.3, 559.6, 634.4, 696.3,
                  790.1, 854.6, 941.3, 976.0, 991.1, 995.8, 999.0, 999.2, 999.9, 1000.0,
                  1000.0, 1000.0, 1000.0])
_UP_LOGRHO = np.log(_UP_RHO)


def geopotential(z):
    """Geometric altitude [m] -> geopotential altitude [m]."""
    z = np.asarray(z, float)
    return R_EARTH * z / (R_EARTH + z)


def geometric(h):
    """Geopotential altitude [m] -> geometric altitude [m].

    The US76 layer boundaries (11, 20, 32, 47, 51, 71, 84.852 km) are *geopotential*
    values, so published tables must be indexed through this inverse before they can be
    compared against a geometric-altitude query.
    """
    h = np.asarray(h, float)
    return R_EARTH * h / (R_EARTH - h)


class _US76:
    """Callable US Standard Atmosphere 1976."""

    name = "US Standard Atmosphere 1976"

    def state(self, z):
        """Return ``(density, temperature, pressure, sound_speed)`` at geometric altitude."""
        z = np.atleast_1d(np.asarray(z, float))
        h = geopotential(z)
        rho = np.empty_like(z)
        temp = np.empty_like(z)
        pres = np.empty_like(z)

        low = h <= _LAYER_H[-1]
        if np.any(low):
            hl = h[low]
            i = np.clip(np.searchsorted(_LAYER_H, hl, side="right") - 1, 0, len(_LAYER_H) - 2)
            t0, p0, lam, h0 = _LAYER_T[i], _LAYER_P[i], _LAYER_L[i], _LAYER_H[i]
            dh = hl - h0
            t = t0 + lam * dh
            iso = lam == 0.0
            p = np.where(iso,
                         p0 * np.exp(-G0 * np.where(iso, dh, 0.0) / (R_AIR * t0)),
                         p0 * (t / t0) ** (-G0 / (R_AIR * np.where(iso, 1.0, lam))))
            temp[low], pres[low] = t, p
            rho[low] = p / (R_AIR * t)

        high = ~low
        if np.any(high):
            zh = np.clip(z[high], _UP_H[0], _UP_H[-1])
            temp[high] = np.interp(zh, _UP_H, _UP_T)
            rho[high] = np.exp(np.interp(zh, _UP_H, _UP_LOGRHO))
            pres[high] = rho[high] * R_AIR * temp[high]

        rho = np.maximum(rho, 1e-20)
        a = np.sqrt(GAMMA_AIR * R_AIR * np.maximum(temp, 1.0))
        if rho.size == 1:
            return float(rho[0]), float(temp[0]), float(pres[0]), float(a[0])
        return rho, temp, pres, a

    def density(self, z):
        return self.state(z)[0]

    def temperature(self, z):
        return self.state(z)[1]

    def pressure(self, z):
        return self.state(z)[2]

    def sound_speed(self, z):
        return self.state(z)[3]

    def viscosity(self, z):
        """Sutherland's law dynamic viscosity [Pa s]."""
        t = np.asarray(self.temperature(z), float)
        return 1.458e-6 * t ** 1.5 / (t + 110.4)

    def mean_free_path(self, z):
        """Molecular mean free path [m], needed for the Knudsen number."""
        rho, t, _p, _a = self.state(z)
        n = np.asarray(rho, float) / M_AIR * NA
        d = 3.65e-10                                   # effective collision diameter of air
        return 1.0 / (np.sqrt(2.0) * np.pi * d * d * np.maximum(n, 1e-30))

    def knudsen(self, z, length):
        return np.asarray(self.mean_free_path(z), float) / max(float(length), 1e-6)

    def scale_height(self, z, dz: float = 100.0):
        """Local density scale height [m], from a centred finite difference."""
        z = np.asarray(z, float)
        r1, r2 = self.density(z - dz), self.density(z + dz)
        return -2.0 * dz / np.log(np.asarray(r2, float) / np.asarray(r1, float))


US76 = _US76()


def profile_frame(z_km=None):
    """Density / temperature / pressure table, for plotting in the notebook."""
    import pandas as pd
    if z_km is None:
        z_km = np.concatenate([np.arange(0, 90, 2.0), np.arange(90, 1001, 10.0)])
    z = np.asarray(z_km, float) * 1e3
    rho, t, p, a = US76.state(z)
    return pd.DataFrame({"alt_km": z_km, "density": rho, "temperature_K": t,
                        "pressure_Pa": p, "sound_speed_ms": a})
