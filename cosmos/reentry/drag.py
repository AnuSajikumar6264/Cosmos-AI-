"""FEATURE 3 (b) — Mach- and Knudsen-dependent drag coefficient.

A re-entering fragment crosses three flow regimes in its first minute: free molecular high
up (Knudsen >> 1), transitional, then continuum.  Holding C_D fixed is the single most
common way to misplace an impact point by hundreds of kilometres, so C_D here depends on
both Mach and Knudsen number.

    dm = DragModel("cylinder", length=0.5)
    dm.cd(mach=8.0, knudsen=1e-3)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: continuum hypersonic drag coefficient by shape, plus the subsonic value
SHAPES = {
    "sphere":      dict(cd_hyper=0.92, cd_sub=0.47, ratio=1.00),
    "cylinder":    dict(cd_hyper=1.20, cd_sub=0.82, ratio=1.00),
    "box":         dict(cd_hyper=1.30, cd_sub=1.05, ratio=1.00),
    "flat_plate":  dict(cd_hyper=1.65, cd_sub=1.28, ratio=1.00),
    "tumbling":    dict(cd_hyper=1.35, cd_sub=0.95, ratio=1.00),
    "cone":        dict(cd_hyper=1.05, cd_sub=0.60, ratio=1.00),
}
#: free-molecular limit: a diffusely reflecting convex body tends to C_D ~ 2.0-2.2
CD_FREE_MOLECULAR = 2.10


def cd_continuum(mach, cd_hyper: float, cd_sub: float):
    """Continuum C_D against Mach number, with the transonic rise near Mach 1."""
    m = np.asarray(mach, float)
    # smooth subsonic -> hypersonic blend
    base = cd_sub + (cd_hyper - cd_sub) / (1.0 + np.exp(-(m - 1.6) / 0.55))
    # transonic bump: real bodies overshoot their hypersonic value around Mach 1.1-1.4.
    # For a sphere the peak sits near C_D ~ 1.0 against 0.92 hypersonic, so the amplitude
    # has to be large enough for the maximum to exceed cd_hyper, not merely cd_sub.
    bump = 0.42 * cd_hyper * np.exp(-((m - 1.25) / 0.45) ** 2)
    return base + bump


def rarefaction_bridge(knudsen):
    """Weight in ``[0, 1]`` for the free-molecular limit (0 = continuum, 1 = free stream).

    A logistic in log10(Kn) centred at Kn = 1, which reproduces the usual bridging
    functions closely enough for trajectory work and stays differentiable.
    """
    kn = np.maximum(np.asarray(knudsen, float), 1e-30)
    return 1.0 / (1.0 + np.exp(-(np.log10(kn) - 0.0) / 0.45))


def cd_of(mach, knudsen, cd_hyper: float, cd_sub: float):
    """Blended drag coefficient across all three flow regimes."""
    w = rarefaction_bridge(knudsen)
    return (1.0 - w) * cd_continuum(mach, cd_hyper, cd_sub) + w * CD_FREE_MOLECULAR


@dataclass
class DragModel:
    """Drag law for one object.

    Parameters
    ----------
    shape
        Key into :data:`SHAPES`.
    length
        Characteristic length [m] used for the Knudsen number.
    """
    shape: str = "tumbling"
    length: float = 0.5

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise KeyError(f"unknown shape {self.shape!r}; expected {sorted(SHAPES)}")
        s = SHAPES[self.shape]
        self.cd_hyper = s["cd_hyper"]
        self.cd_sub = s["cd_sub"]

    def cd(self, mach, knudsen):
        return cd_of(mach, knudsen, self.cd_hyper, self.cd_sub)

    def cd_at(self, altitude_m, speed_ms, atmosphere):
        """Convenience: compute Mach and Knudsen from the atmosphere, then C_D."""
        _rho, _t, _p, a = atmosphere.state(altitude_m)
        kn = atmosphere.knudsen(altitude_m, self.length)
        return self.cd(np.asarray(speed_ms, float) / np.maximum(a, 1e-9), kn)


def sweep_frame(shape: str = "cylinder", length: float = 0.5):
    """C_D against Mach at several Knudsen numbers, for the notebook figure."""
    import pandas as pd
    dm = DragModel(shape, length)
    mach = np.linspace(0.1, 25.0, 300)
    out = {"mach": mach}
    for kn, lab in [(1e-5, "continuum (Kn 1e-5)"), (1e-1, "transitional (Kn 0.1)"),
                    (1e1, "free molecular (Kn 10)")]:
        out[lab] = dm.cd(mach, kn)
    return pd.DataFrame(out)
