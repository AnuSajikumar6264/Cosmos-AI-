"""FEATURE 1 — satellite health monitoring, and its coupling to break-up physics.

Simulated telemetry is in :mod:`.telemetry`; real telemetry (NASA SMAP / MSL and the ESA
Anomaly Dataset) is in :mod:`.realdata`.
"""
from .hazard import BreakupHazard, assess_hazard, scenario_table
from .monitor import (BANDS, HealthMonitor, MonitorResult, SubsystemHealth, band_of,
                      detection_delay)
from .realdata import (ESAMission, RealChannel, esa_ablation, evaluate_esa,
                       evaluate_esa_mission, evaluate_smap_msl, load_esa_mission,
                       load_smap_msl, prepare_esa_mission)
from .telemetry import (CHANNELS, FAULTS, SUBSYSTEMS, TelemetrySeries, channel_units,
                        simulate_telemetry)

__all__ = ["SUBSYSTEMS", "CHANNELS", "FAULTS", "TelemetrySeries", "simulate_telemetry",
           "channel_units", "HealthMonitor", "MonitorResult", "SubsystemHealth", "BANDS",
           "band_of", "detection_delay", "BreakupHazard", "assess_hazard", "scenario_table",
           "RealChannel", "ESAMission", "load_smap_msl", "evaluate_smap_msl",
           "prepare_esa_mission", "load_esa_mission", "evaluate_esa_mission", "evaluate_esa",
           "esa_ablation"]
