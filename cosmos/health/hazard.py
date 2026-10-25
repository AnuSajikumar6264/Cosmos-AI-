"""FEATURE 1 (c) — the coupling: health state -> break-up physics.

This is the module that joins the three features.  A health monitor normally stops at an
alarm.  Here the *type* of failure selects the break-up physics that feature 3 then
propagates, so a propulsion fault and a thermal fault produce genuinely different landing
footprints from the same anomaly score.

    hz = assess_hazard(0.85, SubsystemHealth({... "Propulsion System": 0.9 ...}))
    hz.summary()
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .monitor import SUBSYSTEMS, SubsystemHealth

#: per-subsystem influence on the *probability* that degradation ends in fragmentation.
#: Propulsion dominates: residual propellant and pressure vessels are what actually burst.
FRAG_WEIGHT = {"Propulsion System": 3.20, "Power System": 1.05, "Thermal Control": 1.35,
               "Attitude Control": 0.95, "Structure": 1.60, "Payload": 0.35,
               "Communication": 0.12, "Data Storage Unit": 0.10}

#: nominal break-up altitude [km] associated with each failure mode.  A pressure-vessel
#: burst happens high; a structure that slowly loses integrity survives to denser air.
BREAKUP_ALT_KM = {"Propulsion System": 92.0, "Power System": 80.0, "Thermal Control": 74.0,
                  "Attitude Control": 84.0, "Structure": 68.0, "Payload": 78.0,
                  "Communication": 80.0, "Data Storage Unit": 80.0}

#: break-up character, which sets the NASA-SBM branch and the delta-v scale
BREAKUP_TYPE = {"Propulsion System": "explosion", "Power System": "explosion",
                "Thermal Control": "aerodynamic", "Attitude Control": "aerodynamic",
                "Structure": "aerodynamic", "Payload": "aerodynamic",
                "Communication": "aerodynamic", "Data Storage Unit": "aerodynamic"}

#: subsystems whose loss makes the object hard to TRACK rather than likely to break up
OBSERVABILITY = {"Communication": 1.0, "Data Storage Unit": 0.75, "Attitude Control": 0.35}

LOGIT_INTERCEPT = -5.2      # calibrated so a nominal satellite reads ~1 %, not ~5 %
LOGIT_ANOMALY = 2.10


@dataclass
class BreakupHazard:
    """Everything feature 3 needs in order to start propagating."""
    anomaly_score: float
    driver: str                     # subsystem judged responsible
    breakup_type: str               # "explosion" | "aerodynamic"
    altitude_km: float
    p_fragmentation: float
    bc_sigma_log10: float           # spread of ballistic coefficient -> footprint width
    state_sigma_m: float            # entry-state uncertainty -> how well we can track it
    severity: float
    band: str

    def summary(self) -> dict:
        return {"anomaly_score": round(self.anomaly_score, 3), "driver": self.driver,
                "breakup_type": self.breakup_type,
                "breakup_altitude_km": round(self.altitude_km, 1),
                "p_fragmentation": round(self.p_fragmentation, 3),
                "bc_sigma_log10": round(self.bc_sigma_log10, 3),
                "state_sigma_m": round(self.state_sigma_m, 0),
                "severity": round(self.severity, 3), "band": self.band}


def assess_hazard(anomaly_score: float, health: SubsystemHealth) -> BreakupHazard:
    """Map an anomaly score plus subsystem health onto break-up physics."""
    a = float(np.clip(anomaly_score, 0.0, 1.0))
    driver, dval = health.worst()

    # weighted severity across all subsystems, dominated by the worst weighted one
    weighted = {k: FRAG_WEIGHT[k] * health[k] for k in SUBSYSTEMS}
    sev = float(np.clip(0.65 * max(weighted.values()) / 3.20 + 0.35 * health.mean(), 0.0, 1.0))

    logit = LOGIT_INTERCEPT + LOGIT_ANOMALY * a + 3.05 * sum(weighted.values()) / 3.20
    p = float(1.0 / (1.0 + np.exp(-logit)))

    # break-up altitude: the driver sets the nominal value, severity pulls it lower
    alt = BREAKUP_ALT_KM[driver] - 14.0 * sev

    # footprint width: more severe damage means a wider spread of fragment shapes
    bc_sigma = 0.16 + 0.13 * sev

    # tracking uncertainty: losing comms or storage blinds us without endangering the object
    obs = sum(OBSERVABILITY.get(k, 0.0) * health[k] for k in SUBSYSTEMS)
    state_sigma = 250.0 + 5200.0 * obs + 900.0 * a

    return BreakupHazard(a, driver, BREAKUP_TYPE[driver], alt, p, bc_sigma, state_sigma,
                         sev, health.band())


def scenario_table(scenarios: dict | None = None) -> pd.DataFrame:
    """Run a set of named health scenarios through the coupling and tabulate them.

    This is the experiment behind the two "Health" rows of the results slide: the same
    anomaly score gives a different break-up altitude, probability and footprint width
    depending on *which* subsystem failed.
    """
    if scenarios is None:
        scenarios = {
            "all nominal":            (0.05, {}),
            "thermal critical":       (0.85, {"Thermal Control": 0.90}),
            "power critical":         (0.85, {"Power System": 0.88}),
            "attitude critical":      (0.85, {"Attitude Control": 0.92}),
            "comms + storage lost":   (0.85, {"Communication": 0.90,
                                              "Data Storage Unit": 0.85}),
            "structure compromised":  (0.85, {"Structure": 0.88}),
            "propulsion critical":    (0.85, {"Propulsion System": 0.90}),
            "propulsion + attitude":  (0.90, {"Propulsion System": 0.85,
                                              "Attitude Control": 0.88}),
        }
    rows = []
    for label, (a, over) in scenarios.items():
        h = SubsystemHealth({**{k: 0.05 for k in SUBSYSTEMS}, **over})
        s = assess_hazard(a, h).summary()
        s["scenario"] = label
        rows.append(s)
    cols = ["scenario", "anomaly_score", "driver", "breakup_type", "breakup_altitude_km",
            "p_fragmentation", "bc_sigma_log10", "state_sigma_m", "band"]
    return pd.DataFrame(rows)[cols]
