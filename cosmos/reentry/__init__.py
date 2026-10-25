"""FEATURE 3 — re-entry of discarded parts into the atmosphere."""
from .atmosphere import US76, geometric, geopotential, profile_frame
from .breakup import FragmentCloud, sample_cloud, size_law_check
from .drag import CD_FREE_MOLECULAR, SHAPES, DragModel, cd_of, sweep_frame
from .montecarlo import (DispersionConfig, Footprint, ballistic_sweep, conformal_zone,
                         gaussian_ellipse_zone, monte_carlo_footprint, zone_comparison)
from .propagator import (MATERIALS, DebrisObject, EnsembleResult, State, Trajectory,
                         WindField, entry_state, propagate, propagate_ensemble)
from .population import load_population_grid, prepare_population
from .risk import (PopulationModel, assess_footprint, casualty_area, estimate_survival,
                   population_along_track, scenario_risk)

__all__ = ["US76", "profile_frame", "geopotential", "geometric", "DragModel", "SHAPES",
           "cd_of", "sweep_frame", "CD_FREE_MOLECULAR", "sample_cloud", "FragmentCloud",
           "size_law_check", "DebrisObject", "State", "entry_state", "propagate",
           "propagate_ensemble", "Trajectory", "EnsembleResult", "WindField", "MATERIALS",
           "monte_carlo_footprint", "DispersionConfig", "Footprint", "zone_comparison",
           "conformal_zone", "gaussian_ellipse_zone", "ballistic_sweep", "PopulationModel",
           "assess_footprint", "casualty_area", "scenario_risk", "population_along_track",
           "estimate_survival", "prepare_population", "load_population_grid"]
