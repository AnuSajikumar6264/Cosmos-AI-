"""FEATURE 3 (c) — NASA Standard Breakup Model (Johnson et al., 2001).

Samples a fragment cloud from a parent object: characteristic length from a power law,
area-to-mass from the bimodal log-normal, mass from A/M and area, and an ejection delta-v.

One correction matters and is applied here.  The size law and the area-to-mass law are
sampled independently, so nothing in them enforces conservation of the parent's mass: for a
1800 kg spacecraft the raw draw overshoots by more than an order of magnitude, because a few
large low-A/M fragments each come out at hundreds of kilograms.  :func:`sample_cloud`
therefore enforces the budget in both directions — it truncates the over-massive tail, and
issues explicit parent remnants when the sample falls short — so mass closure lands at 1.0.

    cloud = sample_cloud(parent_mass=1800.0, kind="explosion")
    cloud.summary()
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common import rng_of

#: minimum characteristic length tracked [m] — 5 cm, roughly the radar catalogue limit
LC_MIN = 0.05
#: delta-v scale [m/s] by break-up character
DV_SCALE = {"explosion": 0.20, "collision": 0.90, "aerodynamic": 0.06}


def n_fragments(parent_mass: float, lc_min: float = LC_MIN, kind: str = "explosion",
                collision_mass: float = 0.0, v_rel: float = 10.0) -> int:
    """Expected number of fragments larger than ``lc_min``.

    Explosion:  N(>Lc) = 6 * s * Lc^-1.6   with a scale factor s tied to parent mass.
    Collision:  N(>Lc) = 0.1 * M_eff^0.75 * Lc^-1.71,  M_eff from the NASA formulation.
    """
    if kind == "collision":
        m_eff = collision_mass + parent_mass if v_rel >= 1.0 else \
            collision_mass * (v_rel ** 2) / 2.0
        return int(max(1, 0.1 * m_eff ** 0.75 * lc_min ** -1.71))
    s = max(parent_mass / 1000.0, 0.05)
    return int(max(1, 6.0 * s * lc_min ** -1.6))


def sample_lc(n: int, lc_min: float = LC_MIN, slope: float = 1.6, rng=None) -> np.ndarray:
    """Inverse-transform sample of the size power law, truncated at 4 m."""
    rng = rng_of(rng)
    u = rng.random(n)
    lc_max = 4.0
    a, b = lc_min ** -slope, lc_max ** -slope
    return (a - u * (a - b)) ** (-1.0 / slope)


def sample_area_to_mass(lc: np.ndarray, spacecraft: bool = True, rng=None) -> np.ndarray:
    """Bimodal log-normal A/M [m^2/kg] conditioned on characteristic length."""
    rng = rng_of(rng)
    lc = np.asarray(lc, float)
    lam = np.log10(np.maximum(lc, 1e-4))
    if spacecraft:
        # two populations: dense structural fragments and light multi-layer insulation
        alpha = np.clip(0.5 + 0.5 * np.tanh((lam + 1.05) / 0.35), 0.0, 1.0)
        mu1 = np.where(lam <= -1.75, -0.45, np.where(lam >= -1.25, -0.95 - 1.05 * (lam + 1.25),
                                                    -0.45 - 1.0 * (lam + 1.75)))
        s1 = np.where(lam <= -1.3, 0.55, np.where(lam >= -0.5, 0.28, 0.55 - 0.3375 * (lam + 1.3)))
        mu2 = np.where(lam <= -3.5, -0.9, -0.9 - 1.15 * (lam + 3.5))
        s2 = 0.55
    else:                                     # rocket body
        alpha = np.clip(1.0 - 0.3571 * (lam + 1.4), 0.0, 1.0)
        mu1 = np.where(lam <= -0.5, -0.45, -0.45 - 0.9 * (lam + 0.5))
        s1 = 0.55
        mu2 = np.where(lam <= -1.0, -1.15, -1.15 - 0.8 * (lam + 1.0))
        s2 = 0.30
    pick = rng.random(lc.size) < alpha
    mu = np.where(pick, mu1, mu2)
    sig = np.where(pick, s1, s2)
    return 10.0 ** rng.normal(mu, sig)


def sample_dv(am: np.ndarray, kind: str = "explosion", rng=None) -> np.ndarray:
    """Ejection speed magnitude [m/s]; light, high-A/M fragments get the biggest kick."""
    rng = rng_of(rng)
    am = np.asarray(am, float)
    mu = 0.9 * np.log10(np.maximum(am, 1e-6)) + np.log10(DV_SCALE.get(kind, 0.2)) + 1.85
    return 10.0 ** rng.normal(mu, 0.40)


@dataclass
class FragmentCloud:
    """Sampled fragments as a tidy frame, plus mass bookkeeping."""
    frame: pd.DataFrame
    parent_mass: float
    kind: str
    n_remnants: int = 0
    n_dropped: int = 0          # fragments discarded to respect the parent's mass budget

    @property
    def mass_accounted(self) -> float:
        return float(self.frame["mass_kg"].sum())

    @property
    def mass_closure(self) -> float:
        return self.mass_accounted / self.parent_mass

    def summary(self) -> dict:
        f = self.frame
        return {"kind": self.kind, "parent_mass_kg": self.parent_mass,
                "n_fragments": len(f), "n_remnants": self.n_remnants,
                "n_dropped_over_budget": self.n_dropped,
                "mass_closure": round(self.mass_closure, 4),
                "lc_cm_min": round(float(f.lc_m.min() * 100), 1),
                "lc_cm_max": round(float(f.lc_m.max() * 100), 1),
                "mass_kg_median": round(float(f.mass_kg.median()), 4),
                "bc_min": round(float(f.bc.min()), 2), "bc_max": round(float(f.bc.max()), 1),
                "dv_ms_median": round(float(f.dv_ms.median()), 2),
                "casualty_capable": int((f.mass_kg > 1.0).sum())}


def sample_cloud(parent_mass: float = 1800.0, kind: str = "explosion",
                 spacecraft: bool = True, lc_min: float = LC_MIN, max_fragments: int = 1200,
                 close_mass: bool = True, mass_budget: float = 0.92,
                 rng=None) -> FragmentCloud:
    """Sample a fragment cloud whose total mass equals the parent's.

    ``mass_budget`` is the share of the parent mass allotted to sampled fragments; the
    remainder is issued as large, dense remnants so that closure lands at 1.0.
    """
    rng = rng_of(rng)
    n_want = n_fragments(parent_mass, lc_min, kind)
    n = int(min(n_want, max_fragments))

    lc = sample_lc(n, lc_min, 1.6 if kind != "collision" else 1.71, rng)
    am = sample_area_to_mass(lc, spacecraft, rng)
    area = 0.540424 * lc ** 2                    # NASA average cross-section relation
    mass = area / np.maximum(am, 1e-6)
    dv = sample_dv(am, kind, rng)

    # a fragment cannot outweigh its parent
    mass = np.minimum(mass, parent_mass * 0.35)
    cd0 = 1.35
    bc = mass / (cd0 * np.maximum(area, 1e-8))

    df = pd.DataFrame({"lc_m": lc, "area_m2": area, "mass_kg": mass, "am_m2kg": am,
                       "dv_ms": dv, "bc": bc, "is_remnant": False})

    # Enforce the mass budget.  The published size and area-to-mass laws are sampled
    # independently, so nothing stops their product from exceeding the parent: with a 1800 kg
    # parent this over-shoots by more than an order of magnitude, because a handful of large
    # low-A/M draws each come out at hundreds of kilograms.  Fragments are kept from the
    # smallest upward — those are the numerous ones the size law is really about — until the
    # budget is spent, and whatever is left over is issued as parent remnants below.
    budget = mass_budget * parent_mass
    df = df.sort_values("mass_kg").reset_index(drop=True)
    keep = df.mass_kg.cumsum() <= budget
    n_dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)

    n_rem = 0
    if close_mass:
        missing = parent_mass - df.mass_kg.sum()
        # remnants: a few large, low-A/M pieces of the primary structure
        while missing > 0.02 * parent_mass and n_rem < 12:
            m = min(missing, parent_mass * rng.uniform(0.06, 0.22))
            am_r = 10.0 ** rng.normal(-1.35, 0.18)          # dense: ~0.045 m^2/kg
            a_r = m * am_r
            df.loc[len(df)] = {"lc_m": float(np.sqrt(a_r / 0.540424)), "area_m2": a_r,
                               "mass_kg": m, "am_m2kg": am_r,
                               "dv_ms": float(10 ** rng.normal(-0.4, 0.3)),
                               "bc": m / (cd0 * a_r), "is_remnant": True}
            missing -= m
            n_rem += 1
    out = FragmentCloud(df.reset_index(drop=True), float(parent_mass), kind, n_rem)
    out.n_dropped = n_dropped
    return out


def size_law_check(n_samples: int = 200_000, rng=None) -> pd.DataFrame:
    """Sampled cumulative size distribution against the analytic power law.

    ``n_samples`` is a statistical choice, not a physical one: this checks the SAMPLER, so it
    needs enough draws that the rarest bin is not dominated by counting noise.  A realistic
    cloud of ~1900 fragments puts only about nine of them above one metre, which is a 33 %
    relative error on that bin from sampling alone.
    """
    rng = rng_of(rng)
    lc = sample_lc(int(n_samples), LC_MIN, 1.6, rng)
    edges = np.logspace(np.log10(LC_MIN), np.log10(1.0), 22)
    emp = np.array([(lc >= e).sum() / lc.size for e in edges])
    # survival function of the TRUNCATED power law on [lc_min, lc_max]; dropping the
    # lc_max term overstates the analytic fraction by ~10 % at 1 m and far more above it
    b_, lc_max = 1.6, 4.0
    ana = ((edges ** -b_ - lc_max ** -b_) / (LC_MIN ** -b_ - lc_max ** -b_))
    return pd.DataFrame({"lc_m": edges, "sampled_fraction": emp, "analytic_fraction": ana,
                         "expected_count": ana * lc.size})
