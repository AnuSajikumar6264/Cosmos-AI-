"""Object state of the COSMOS-AI digital twin.

An :class:`ObjectTwin` is the twin of one catalogued object at one time ``as_of``.  It holds
what is known about the object at that time (identity, orbit, physical properties, operational
status, debris family, space weather) and the outputs that the modules attach to it.  Module
outputs always carry their provenance (module, model, validation status) and their own
uncertainty, so that a consumer can see which part of the twin is validated on real data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModuleOutput:
    module: str                       # health | debris | reentry | coupling
    model: str
    validation: str                   # e.g. "validated-real (GCAT temporal test 2016-2021)"
    values: dict
    uncertainty: dict = field(default_factory=dict)
    abstain: bool = False
    reason: str = ""


@dataclass
class ObjectTwin:
    object_id: str                    # GCAT JCAT / NORAD number
    name: str
    object_class: str                 # P payload, R rocket stage, C component, D fragment
    as_of: str
    orbit: dict                       # perigee, apogee, inc (km, deg) and orbit epoch
    physical: dict                    # mass, dimensions, area-to-mass source, RCS
    operational_status: str | None    # catalogue health proxy for payloads ('+', 'P', '-', ...)
    family: dict                      # launch / breakup family and causal sibling statistics
    space_weather: dict
    outputs: list = field(default_factory=list)

    def attach(self, out: ModuleOutput) -> None:
        self.outputs.append(out)

    def regime(self) -> str:
        """Re-entry regime implied by the health evidence (coupling C3)."""
        if self.object_class != "P":
            return "natural_decay"
        if self.operational_status in ("+", "P", "B", "S", "X"):
            return "operator_controlled"
        if self.operational_status == "-":
            return "natural_decay"
        return "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime()
        return d
