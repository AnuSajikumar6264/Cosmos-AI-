"""The end-to-end chain that makes the three features one system.

    telemetry -> health -> hazard -> break-up cloud -> re-entry -> footprint -> risk

Feature 1 does not merely raise an alarm: the *type* of failure selects the break-up physics
that feature 3 propagates.  So the same anomaly score produces a different landing footprint
depending on which subsystem failed, which is the claim the whole project rests on.

    res = run_pipeline(faults={"Propulsion System": "leak"})
    res.summary()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .health.hazard import BreakupHazard, assess_hazard
from .health.monitor import HealthMonitor, MonitorResult, SubsystemHealth, detection_delay
from .health.telemetry import TelemetrySeries, simulate_telemetry
from .reentry.breakup import FragmentCloud, sample_cloud
from .reentry.montecarlo import (DispersionConfig, Footprint, monte_carlo_footprint,
                                 zone_comparison)
from .reentry.risk import PopulationModel, assess_footprint


@dataclass
class PipelineResult:
    """Everything the chain produced, so the notebook can show any stage."""
    telemetry: TelemetrySeries
    monitor: MonitorResult
    health: SubsystemHealth
    hazard: BreakupHazard
    cloud: FragmentCloud
    footprint: Footprint
    risk: dict
    label: str = "scenario"
    zones: pd.DataFrame = field(default=None)

    def summary(self) -> pd.DataFrame:
        """One tidy table walking the chain from telemetry to casualties."""
        d = detection_delay(self.telemetry, self.monitor)
        rows = [
            ("1 Health", "telemetry samples", self.telemetry.n),
            ("1 Health", "injected fault", self.telemetry.summary()["faults"]),
            ("1 Health", "anomaly score (final)", round(float(self.monitor.anomaly[-1]), 3)),
            ("1 Health", "worst subsystem", self.health.worst()[0]),
            ("1 Health", "health band", self.health.band()),
            ("1 Health", "detection delay (min)", d.get("delay_minutes", "n/a")),
            ("1 -> 3 coupling", "break-up type", self.hazard.breakup_type),
            ("1 -> 3 coupling", "break-up altitude (km)", round(self.hazard.altitude_km, 1)),
            ("1 -> 3 coupling", "P(fragmentation)", round(self.hazard.p_fragmentation, 3)),
            ("1 -> 3 coupling", "footprint sigma log10(B)", round(self.hazard.bc_sigma_log10, 3)),
            ("3 Re-entry", "fragments generated", len(self.cloud.frame)),
            ("3 Re-entry", "mass closure", round(self.cloud.mass_closure, 3)),
            ("3 Re-entry", "impacts propagated", len(self.footprint.frame)),
            ("3 Re-entry", "downrange mean (km)",
             self.footprint.summary()["downrange_mean_km"]),
            ("3 Re-entry", "downrange spread (km)",
             round(self.footprint.summary()["downrange_max_km"]
                   - self.footprint.summary()["downrange_min_km"], 1)),
            ("3 Re-entry", "CEP50 (km)", self.footprint.summary()["cep50_km"]),
            ("3 Risk", "expected casualties", f"{self.risk['expected_casualties']:.3e}"),
            ("3 Risk", "risk band", self.risk["risk_band"]),
            ("3 Risk", "land impact probability", self.risk["land_impact_probability"]),
            ("3 Risk", "nearest city", f"{self.risk['nearest_city']} "
                                       f"({self.risk['nearest_distance_km']:.0f} km)"),
        ]
        return pd.DataFrame(rows, columns=["stage", "quantity", "value"])


def run_pipeline(faults: dict | None = None, anomaly_override: float | None = None,
                 parent_mass: float = 1800.0, n_samples: int = 1200, hours: float = 24.0,
                 population: PopulationModel | None = None, with_zones: bool = False,
                 azimuth_deg: float = 140.0, lat_deg: float = 13.72, lon_deg: float = 80.23,
                 label: str = "scenario", seed: int = 7, backend: str = "classical",
                 footprint_backend: str = "reference") -> PipelineResult:
    """Run telemetry -> health -> hazard -> break-up -> footprint -> risk.

    Parameters
    ----------
    faults
        ``{subsystem: fault_name}`` passed to the telemetry simulator.
    anomaly_override
        Force the anomaly score instead of using the monitor's own value; useful for
        comparing failure modes at *identical* score, which is what isolates the coupling.
    with_zones
        Also compute the conformal-versus-Gaussian zone comparison (slower).
    backend
        ``"classical"`` (default, the original monitor) or ``"learned"``
        (:class:`cosmos.ai.telemetry.learned_monitor.LearnedHealthMonitor`, same interface).
    footprint_backend
        ``"reference"`` (default, the propagator) or ``"surrogate"``
        (:func:`cosmos.ai.fusion.surrogate_footprint`; falls back to the propagator for
        samples the surrogate ensemble flags).  Both learned options need trained models.
    """
    ts = simulate_telemetry(hours=hours, faults=faults, seed=seed)
    if backend == "learned":
        from .ai.telemetry.learned_monitor import LearnedHealthMonitor
        mon = LearnedHealthMonitor().fit(ts).score(ts)
    elif backend == "classical":
        mon = HealthMonitor().fit(ts).score(ts)
    else:
        raise ValueError(f"backend must be 'classical' or 'learned', not {backend!r}")
    health = mon.health_at(-1)
    score = float(mon.anomaly[-1]) if anomaly_override is None else float(anomaly_override)
    hz = assess_hazard(score, health)
    cloud = sample_cloud(parent_mass=parent_mass, kind=hz.breakup_type, rng=seed)
    if footprint_backend == "surrogate":
        from .ai.fusion import surrogate_footprint
        fp = surrogate_footprint(hz, cloud, DispersionConfig(n_samples=n_samples, seed=seed),
                                 azimuth_deg=azimuth_deg, lat_deg=lat_deg, lon_deg=lon_deg)
    elif footprint_backend == "reference":
        fp = monte_carlo_footprint(hz, cloud, DispersionConfig(n_samples=n_samples, seed=seed),
                                   azimuth_deg=azimuth_deg, lat_deg=lat_deg, lon_deg=lon_deg)
    else:
        raise ValueError(f"footprint_backend must be 'reference' or 'surrogate', "
                         f"not {footprint_backend!r}")
    risk = assess_footprint(fp, population)
    zones = zone_comparison(fp, n_splits=8, seed=seed) if with_zones else None
    return PipelineResult(ts, mon, health, hz, cloud, fp, risk, label, zones)


#: re-entry corridors for the ground-risk comparison: (entry latitude, longitude, azimuth).
#: The two Sriharikota headings are the pipeline's default entry point; the other two put
#: the same entry geometry over open ocean and over a densely populated continent.
CORRIDORS = {
    "Mid-Pacific (E)":          (0.0, -150.0, 90.0),
    "Bay of Bengal (SE)":       (13.72, 80.23, 140.0),
    "Peninsular India (NW)":    (13.72, 80.23, 320.0),
    "Central Europe (E)":       (50.0, 2.0, 90.0),
}


def corridor_risk(corridors: dict | None = None, population: PopulationModel | None = None,
                  faults: dict | None = None, n_samples: int = 1200, seed: int = 7,
                  survival_fraction: float | None = None) -> pd.DataFrame:
    """The same failing spacecraft over several re-entry corridors.

    Ground risk is a property of the corridor as much as of the spacecraft; this is the
    quantitative argument for corridor selection in a controlled de-orbit.
    """
    from .reentry.risk import assess_footprint
    corridors = corridors or CORRIDORS
    faults = {"Propulsion System": "leak"} if faults is None else faults
    rows = []
    for name, (lat, lon, az) in corridors.items():
        r = run_pipeline(faults=faults, n_samples=n_samples, lat_deg=lat, lon_deg=lon,
                         azimuth_deg=az, population=population, seed=seed)
        risk = assess_footprint(r.footprint, population, n_fragments=len(r.cloud.frame),
                                survival_fraction=survival_fraction)
        rows.append({"corridor": name, "entry_lat": lat, "entry_lon": lon, "azimuth": az,
                     "expected_casualties": risk["expected_casualties"],
                     "risk_band": risk["risk_band"],
                     "expected_casualties_event": risk["expected_casualties_event"],
                     "land_impact_probability": risk["land_impact_probability"],
                     "mean_population_density": risk["mean_population_density"],
                     "population_model": risk["population_model"]})
    return pd.DataFrame(rows)


def compare_failure_modes(modes: dict | None = None, anomaly: float = 0.85,
                          parent_mass: float = 1800.0, n_samples: int = 700,
                          seed: int = 7) -> tuple:
    """Same anomaly score, different failed subsystem — the central experiment.

    Returns ``(table, results)``.  Holding the anomaly score fixed is the point: any
    difference in the footprint then comes from the failure MODE, not from severity.
    """
    if modes is None:
        modes = {"nominal": {},
                 "thermal": {"Thermal Control": "overheat"},
                 "structure": {"Structure": "degradation"},
                 "propulsion": {"Propulsion System": "leak"},
                 "comms lost": {"Communication": "dropout"}}
    rows, out = [], {}
    for name, faults in modes.items():
        a = 0.05 if name == "nominal" else anomaly
        r = run_pipeline(faults=faults, anomaly_override=a, parent_mass=parent_mass,
                         n_samples=n_samples, label=name, seed=seed)
        out[name] = r
        s = r.footprint.summary()
        rows.append({"scenario": name, "anomaly_score": a,
                     "driver": r.hazard.driver, "breakup_type": r.hazard.breakup_type,
                     "breakup_alt_km": round(r.hazard.altitude_km, 1),
                     "P_fragmentation": round(r.hazard.p_fragmentation, 3),
                     "sigma_logB": round(r.hazard.bc_sigma_log10, 3),
                     "state_sigma_m": round(r.hazard.state_sigma_m, 0),
                     "downrange_mean_km": s["downrange_mean_km"],
                     "downrange_spread_km": round(s["downrange_max_km"]
                                                  - s["downrange_min_km"], 1),
                     "cep50_km": s["cep50_km"],
                     "expected_casualties": r.risk["expected_casualties"],
                     "risk_band": r.risk["risk_band"]})
    return pd.DataFrame(rows), out
