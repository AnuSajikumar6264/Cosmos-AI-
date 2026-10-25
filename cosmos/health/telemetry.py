"""FEATURE 1 (a) — telemetry for a newly launched satellite.

Generates a realistic multi-channel telemetry stream for eight subsystems, with optional
faults injected on named subsystems.  Every channel has a physical baseline, orbital
periodicity (eclipse cycling), sensor noise and slow drift, so an anomaly detector has
something non-trivial to work on.

    ts = simulate_telemetry(hours=24, faults={"Propulsion System": "leak"})
    ts.frame.head()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import rng_of

#: the eight monitored subsystems, in the order used everywhere in the project
SUBSYSTEMS = ["Power System", "Thermal Control", "Attitude Control", "Propulsion System",
              "Communication", "Payload", "Data Storage Unit", "Structure"]

#: channel definition per subsystem: (name, unit, baseline, orbital amplitude, noise sigma)
CHANNELS = {
    "Power System":      [("bus_voltage", "V", 28.0, 0.45, 0.06),
                          ("solar_current", "A", 12.0, 4.00, 0.15)],
    "Thermal Control":   [("radiator_temp", "degC", 12.0, 9.00, 0.35),
                          ("heater_duty", "%", 22.0, 8.00, 0.90)],
    "Attitude Control":  [("gyro_bias", "deg/h", 0.10, 0.03, 0.010),
                          ("wheel_rpm", "rpm", 3200.0, 120.0, 9.0)],
    "Propulsion System": [("tank_pressure", "bar", 18.0, 0.10, 0.035),
                          ("valve_temp", "degC", 24.0, 3.00, 0.30)],
    "Communication":     [("downlink_snr", "dB", 14.5, 1.20, 0.22),
                          ("tx_power", "W", 9.0, 0.30, 0.08)],
    "Payload":           [("detector_temp", "degC", -35.0, 2.50, 0.30),
                          ("frame_rate", "fps", 8.0, 0.15, 0.05)],
    "Data Storage Unit": [("buffer_fill", "%", 45.0, 18.0, 1.10),
                          ("write_errors", "count/min", 0.30, 0.10, 0.12)],
    "Structure":         [("panel_strain", "ustrain", 120.0, 25.0, 2.20),
                          ("vibration_rms", "mg", 3.5, 0.60, 0.18)],
}

#: fault signatures — how each named fault deforms that subsystem's channels
FAULTS = {
    "degradation": dict(shape="ramp", gain=(-0.5, -0.5), noise=1.6),
    "leak":        dict(shape="ramp", gain=(-3.0, +1.2), noise=2.2),
    "overheat":    dict(shape="ramp", gain=(+4.0, +3.0), noise=2.0),
    "stuck":       dict(shape="step", gain=(-2.5, -3.5), noise=0.6),
    "dropout":     dict(shape="step", gain=(-4.0, -2.0), noise=3.0),
    "saturation":  dict(shape="ramp", gain=(+3.5, +2.5), noise=1.4),
}
ORBIT_PERIOD_S = 95.0 * 60.0          # ~550 km circular orbit


@dataclass
class TelemetrySeries:
    """A telemetry run: the sampled frame plus the ground truth about injected faults."""
    frame: pd.DataFrame
    channel_map: dict = field(default_factory=dict)   # subsystem -> [column names]
    faults: dict = field(default_factory=dict)        # subsystem -> fault name
    fault_onset_index: int = -1
    dt_s: float = 30.0

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def channels(self) -> list:
        return [c for c in self.frame.columns if c != "t_s"]

    def subsystem_frame(self, name: str) -> pd.DataFrame:
        return self.frame[self.channel_map[name]]

    def summary(self) -> dict:
        return {"samples": self.n, "duration_h": round(self.n * self.dt_s / 3600.0, 2),
                "channels": len(self.channels), "subsystems": len(self.channel_map),
                "faults": ", ".join(f"{k} ({v})" for k, v in self.faults.items()) or "none",
                "fault_onset_index": self.fault_onset_index}


def simulate_telemetry(hours: float = 24.0, dt_s: float = 30.0, faults: dict | None = None,
                       fault_onset_frac: float = 0.55, seed=7) -> TelemetrySeries:
    """Simulate the satellite's telemetry.

    Parameters
    ----------
    hours, dt_s
        Duration and sampling interval.
    faults
        ``{subsystem_name: fault_name}``, where fault names come from :data:`FAULTS`.
    fault_onset_frac
        Where in the run the faults begin, as a fraction of total samples.
    """
    rng = rng_of(seed)
    faults = dict(faults or {})
    for sub, fl in faults.items():
        if sub not in CHANNELS:
            raise KeyError(f"unknown subsystem {sub!r}; expected one of {SUBSYSTEMS}")
        if fl not in FAULTS:
            raise KeyError(f"unknown fault {fl!r}; expected one of {sorted(FAULTS)}")

    n = int(round(hours * 3600.0 / dt_s))
    t = np.arange(n) * dt_s
    onset = int(round(fault_onset_frac * n)) if faults else -1
    phase = 2 * np.pi * t / ORBIT_PERIOD_S

    cols, cmap = {"t_s": t}, {}
    for sub in SUBSYSTEMS:
        names = []
        fl = FAULTS.get(faults.get(sub, ""), None)
        for k, (cname, _unit, base, amp, sig) in enumerate(CHANNELS[sub]):
            col = f"{cname}"
            signal = base + amp * np.sin(phase + 0.7 * k)
            signal = signal + base * 2e-6 * t / 3600.0                 # slow ageing drift
            signal = signal + rng.normal(0.0, sig, n)
            if fl is not None:
                sev = np.zeros(n)
                if fl["shape"] == "ramp":
                    sev[onset:] = np.linspace(0.0, 1.0, n - onset) ** 1.3
                else:                                                   # step
                    sev[onset:] = 1.0
                signal = signal + fl["gain"][k] * sig * 6.0 * sev
                signal = signal + rng.normal(0.0, sig * (fl["noise"] - 1.0), n) * sev
            cols[col] = signal
            names.append(col)
        cmap[sub] = names
    return TelemetrySeries(pd.DataFrame(cols), cmap, faults, onset, dt_s)


def channel_units() -> pd.DataFrame:
    """Reference table of every channel, for the notebook."""
    rows = [{"subsystem": s, "channel": c, "unit": u, "baseline": b}
            for s, chs in CHANNELS.items() for c, u, b, _a, _n in chs]
    return pd.DataFrame(rows)
