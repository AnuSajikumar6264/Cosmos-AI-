"""Evidence-aware decision layer, and the learned back-ends of the COSMOS pipeline.

No paired telemetry + imagery + re-entry observations of one object exist (ESA-ADB is
anonymised, SpotGEO has no object identities), so COSMOS-AI does NOT train a joint model
across features.  Instead each component emits an :class:`Evidence` item with its own
calibrated probability, its validation status and whether its input was in-domain, and
:func:`decide` combines only items that refer to the same object and time window:

* items from models whose card status is not ``validated-*`` are reported, never combined;
* out-of-domain items (an OOD flag, or a surrogate outside its training ranges) abstain;
* compatible items are combined by noisy-OR of their probabilities — an explicit
  independence assumption, stated in the output — and the decision abstains when no item
  remains or when items disagree by more than ``conflict``.

:func:`surrogate_footprint` is the Monte Carlo footprint with the propagator replaced by the
neural surrogate, falling back to the reference propagator for samples the ensemble flags.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Evidence:
    feature: str                    # health | debris | reentry
    model: str                      # registry card name
    claim: str                      # e.g. "anomalous telemetry", "re-entry within 68 d"
    probability: float              # calibrated P(claim)
    status: str                     # card status (validated-real, validated-synthetic, ...)
    in_domain: bool = True
    object_id: str | None = None
    window: tuple | None = None     # (start, end) the evidence refers to
    details: dict = field(default_factory=dict)


@dataclass
class Decision:
    claim: str
    probability: float | None
    abstain: bool
    reason: str
    used: list
    reported_only: list
    assumptions: list


def _overlap(a, b) -> bool:
    if a is None or b is None:
        return True
    return a[0] <= b[1] and b[0] <= a[1]


def decide(items: list, claim: str, object_id: str | None = None, window=None,
           conflict: float = 0.6, min_items: int = 1) -> Decision:
    used, reported, why = [], [], []
    for it in items:
        if it.claim != claim:
            reported.append((it.model, "different claim"))
        elif object_id is not None and it.object_id not in (None, object_id):
            reported.append((it.model, "different object"))
        elif not _overlap(it.window, window):
            reported.append((it.model, "outside the time window"))
        elif not it.status.startswith("validated"):
            reported.append((it.model, f"status {it.status}"))
        elif not it.in_domain:
            reported.append((it.model, "input out of domain"))
        else:
            used.append(it)
    if len(used) < min_items:
        return Decision(claim, None, True, "insufficient compatible evidence", [],
                        reported, [])
    ps = np.array([u.probability for u in used])
    if ps.max() - ps.min() > conflict:
        return Decision(claim, None, True, f"conflicting evidence (spread {ps.max() - ps.min():.2f})",
                        [u.model for u in used], reported, [])
    p = 1.0 - np.prod(1.0 - ps)
    syn = any(u.status == "validated-synthetic" for u in used)
    assumptions = ["noisy-OR: the evidence items are treated as conditionally independent"]
    if syn:
        assumptions.append("at least one item is validated on synthetic data only")
    return Decision(claim, float(p), False, "combined", [u.model for u in used], reported,
                    assumptions)


def evidence_from_monitor(res, model: str, status: str, object_id=None) -> Evidence:
    """Telemetry evidence: the final anomaly score of a MonitorResult as P(anomalous).

    For :class:`~cosmos.ai.telemetry.learned_monitor.LearnedHealthMonitor` the score is a
    monotone map of a conformal (Simes) p-value; P(anomalous) := score is a reporting
    convention, not a calibrated posterior, and is labelled as such in ``details``.
    """
    return Evidence("health", model, "anomalous telemetry", float(res.anomaly[-1]), status,
                    True, object_id, None,
                    {"first_alarm_index": int(res.detected_index), "method": res.method,
                     "probability_semantics": "score in [0, 1], 0.5 = alarm level"})


# ============================================================================ surrogate footprint
def _destination(lat0, lon0, az_deg, along_km, cross_km):
    """Lat/lon of the point ``along`` km along heading ``az`` and ``cross`` km to its right."""
    R = 6371.0
    d = np.hypot(along_km, cross_km) / R
    brg = np.radians(az_deg) + np.arctan2(cross_km, along_km)
    p1, l1 = np.radians(lat0), np.radians(lon0)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(brg))
    l2 = l1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def surrogate_footprint(hazard, cloud, config=None, surrogate=None, entry_speed_ms=7600.0,
                        flight_path_deg=-1.4, azimuth_deg=140.0, lat_deg=13.72,
                        lon_deg=80.23, fallback: str = "range",
                        max_std_km: float | None = None):
    """:func:`cosmos.reentry.monte_carlo_footprint` with the surrogate in place of the
    propagator.

    The random draws replicate the reference sampler call for call (ballistic-coefficient
    noise, ONE flight-path perturbation for the ensemble, isotropic delta-v directions,
    density multipliers), so for the same seed the surrogate sees exactly the inputs the
    propagator would.  ``fallback`` decides which samples go to the reference propagator:

    ``"none"``    every sample on the surrogate (fastest);
    ``"range"``   samples outside the surrogate's training ranges (default);
    ``"spread"``  also samples whose ensemble spread exceeds ``max_std_km`` (default: the 95th
                  percentile of the spread on the validation split).  The flagged samples are
                  the slow, low-ballistic-coefficient ones, so this mode is accurate but not
                  faster than the reference.
    """
    from ..common import enu_basis, great_circle_km, rng_of
    from ..reentry.drag import SHAPES
    from ..reentry.montecarlo import DispersionConfig, Footprint
    from ..reentry.propagator import entry_state
    from .surrogate import RANGES, Surrogate, simulate
    cfg = config or DispersionConfig()
    S = surrogate or Surrogate()
    rng = rng_of(cfg.seed)
    n = cfg.n_samples
    frag = cloud.frame.sample(n=n, replace=True, random_state=cfg.seed).reset_index(drop=True)
    bc = frag.bc.values * 10.0 ** rng.normal(0.0, hazard.bc_sigma_log10, n)
    mass = frag.mass_kg.values
    gamma = flight_path_deg + rng.normal(0.0, cfg.gamma_sigma_deg)      # one draw, as reference
    st = entry_state(hazard.altitude_km * 1e3, entry_speed_ms, gamma, azimuth_deg, lat_deg,
                     lon_deg)
    dv = frag.dv_ms.values * cfg.dv_scale
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    dens = 10.0 ** rng.normal(0.0, cfg.density_sigma_log10, n)
    east, north, up = enu_basis(lat_deg, lon_deg)                       # ECEF delta-v -> ENU
    dvv = dv[:, None] * u
    d = pd.DataFrame({"alt_km": np.full(n, hazard.altitude_km), "speed_ms": entry_speed_ms,
                      "gamma_deg": gamma, "azimuth_deg": azimuth_deg, "lat_deg": lat_deg,
                      "log_bc": np.log10(np.maximum(bc, 1e-12)),
                      "log_mass": np.log10(np.maximum(mass, 1e-12)),
                      "log_density": np.log10(dens),
                      "dv_e": dvv @ east, "dv_n": dvv @ north, "dv_u": dvv @ up})
    inside = np.ones(n, bool)
    for k, (lo, hi) in RANGES.items():
        if k in d and k != "azimuth_deg":
            inside &= (d[k] >= lo) & (d[k] <= hi)
    inside &= dv <= RANGES["dv_ms"][1]
    mean, std, p_imp = S.predict(d)
    spread = np.hypot(std[:, 0], std[:, 1])
    if fallback == "none":
        use = np.ones(n, bool)
    elif fallback == "range":
        use = inside
    elif fallback == "spread":
        max_std_km = S.spread_threshold() if max_std_km is None else max_std_km
        use = inside & (spread <= max_std_km)
    else:
        raise ValueError(fallback)
    along, cross = mean[:, 0].copy(), mean[:, 1].copy()
    fall, speed = 10 ** mean[:, 2], 10 ** mean[:, 3]
    impacted = p_imp > 0.5
    if (~use).any():                                     # reference propagator fallback
        ref = simulate(d[~use])
        along[~use], cross[~use] = ref.downrange_km.values, ref.crossrange_km.values
        fall[~use] = 10 ** ref.log_fall_time.values
        speed[~use] = 10 ** ref.log_speed.values
        impacted[~use] = ref.impacted.values
    lat, lon = _destination(lat_deg, lon_deg, azimuth_deg, along, cross)
    df = pd.DataFrame({"lat": lat, "lon": lon, "speed_ms": speed, "mass_kg": mass,
                       "fall_time_s": fall, "bc": bc, "impacted": impacted,
                       "energy_J": 0.5 * mass * speed ** 2,
                       "downrange_km": great_circle_km(lat_deg, lon_deg, lat, lon),
                       "surrogate_used": use, "ensemble_spread_km": spread})
    fp = Footprint(df[df.impacted].reset_index(drop=True), lat_deg, lon_deg, cfg)
    fp.surrogate_fraction = float(use.mean())
    fp.inside_training_ranges = float(inside.mean())
    fp.fallback = fallback
    fp.spread_threshold_km = float(max_std_km) if max_std_km is not None else float("nan")
    return fp


def summarise_decision(dec: Decision) -> dict:
    return asdict(dec)
