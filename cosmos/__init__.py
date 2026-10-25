"""COSMOS — a digital twin joining three problems that are usually studied apart.

FEATURE 1  satellite health monitoring        :mod:`cosmos.health`
FEATURE 2  debris generated in space          :mod:`cosmos.debris`
FEATURE 3  re-entry into the atmosphere       :mod:`cosmos.reentry`

The three are one chain, not three tools: the satellite's health decides HOW it breaks up,
which sets the fragment cloud that re-enters, and the objects catalogued in orbit are the
same objects that later come down.  :func:`cosmos.pipeline.run_pipeline` runs it end to end.
"""
from . import common, datasets, debris, health, reentry
from .common import Table, data_root, feature_dir, output_dir
from .pipeline import (CORRIDORS, PipelineResult, compare_failure_modes, corridor_risk,
                       run_pipeline)

__version__ = "1.1.0"
__all__ = ["health", "debris", "reentry", "common", "datasets", "run_pipeline",
           "compare_failure_modes", "corridor_risk", "CORRIDORS", "PipelineResult",
           "feature_dir", "data_root", "output_dir", "Table"]
