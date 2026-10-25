"""FEATURE 3 (f) — who is underneath: population exposure and expected casualties.

Turns an impact footprint into the number a licensing authority actually asks for: the
expected number of casualties per re-entry, against the accepted 1-in-10,000 gate.

Two population models share one interface:

real      GHS-POP 2020 (JRC, about 1 km) aggregated to 0.25 degrees, with a Natural Earth land
          mask — see :mod:`.population`.  Used by default whenever it has been prepared.
coarse    a hand-authored 10-degree land grid plus 51 metropolitan centres.  It gets the
          land/ocean and city/countryside contrast right, which sets the order of magnitude,
          and is kept so the earlier results can be reproduced: ``PopulationModel.coarse()``.

    assess_footprint(fp)                                   # real raster if prepared
    assess_footprint(fp, PopulationModel.coarse())         # the hand-authored grid
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common import EC_THRESHOLD, LETHAL_ENERGY_J, great_circle_km

#: mean projected area of a standing person [m^2], the FAA convention
PERSON_AREA_M2 = 0.36

#: land fraction on a coarse 10-degree grid, latitude 90N -> 90S, longitude 180W -> 180E.
#: Hand-authored: 0 = open ocean, 1 = continental interior.
_LAND = [
    "000000000000000000000000000000000000",
    "000000011111111111111000000000011000",
    "000011111111111111111111100000111110",
    "000111111111111111111111111000111111",
    "001111111111111111111111111100011111",
    "000011111111111111111111111100001110",
    "000001111100011111111111111000000100",
    "000000111000001111111111100000000000",
    "000000011000000111111110000000000000",
    "000000001100000011111100000000000000",
    "000000000110000001111000000000110000",
    "000000000011000000110000000001111000",
    "000000000001100000000000000011111000",
    "000000000000110000000000000001111000",
    "000000000000010000000000000000110000",
    "000000000000000000000000000000000000",
    "000000000000000000000000000000000000",
    "111111111111111111111111111111111111",
]

#: metropolitan centres: (name, lat, lon, population, radius_km)
_CITIES = [
    ("Tokyo", 35.69, 139.69, 37_400_000, 60), ("Delhi", 28.70, 77.10, 32_900_000, 55),
    ("Shanghai", 31.23, 121.47, 29_200_000, 50), ("Dhaka", 23.81, 90.41, 23_200_000, 30),
    ("Cairo", 30.04, 31.24, 22_200_000, 40), ("Mumbai", 19.08, 72.88, 21_300_000, 35),
    ("Beijing", 39.90, 116.41, 21_800_000, 50), ("Osaka", 34.69, 135.50, 19_000_000, 40),
    ("Karachi", 24.86, 67.00, 17_200_000, 35), ("Chongqing", 29.56, 106.55, 17_000_000, 55),
    ("Istanbul", 41.01, 28.98, 15_800_000, 40), ("Buenos Aires", -34.60, -58.38, 15_400_000, 40),
    ("Kolkata", 22.57, 88.36, 15_300_000, 35), ("Lagos", 6.52, 3.38, 15_400_000, 35),
    ("Manila", 14.60, 120.98, 14_400_000, 30), ("Sao Paulo", -23.55, -46.63, 22_400_000, 45),
    ("Mexico City", 19.43, -99.13, 22_100_000, 45), ("Guangzhou", 23.13, 113.26, 13_600_000, 45),
    ("Moscow", 55.76, 37.62, 12_600_000, 40), ("Jakarta", -6.21, 106.85, 11_100_000, 35),
    ("Chennai", 13.08, 80.27, 11_500_000, 35), ("Bengaluru", 12.97, 77.59, 13_200_000, 35),
    ("Hyderabad", 17.39, 78.49, 10_500_000, 30), ("New York", 40.71, -74.01, 18_800_000, 45),
    ("Los Angeles", 34.05, -118.24, 12_500_000, 45), ("London", 51.51, -0.13, 9_600_000, 35),
    ("Paris", 48.86, 2.35, 11_200_000, 35), ("Lima", -12.05, -77.04, 11_000_000, 30),
    ("Bangkok", 13.76, 100.50, 10_700_000, 35), ("Tehran", 35.69, 51.39, 9_500_000, 30),
    ("Seoul", 37.57, 126.98, 9_900_000, 35), ("Nagoya", 35.18, 136.91, 9_100_000, 35),
    ("Ho Chi Minh City", 10.82, 106.63, 9_300_000, 30), ("Luanda", -8.84, 13.23, 9_300_000, 25),
    ("Kinshasa", -4.44, 15.27, 17_000_000, 30), ("Rio de Janeiro", -22.91, -43.17, 13_600_000, 35),
    ("Chicago", 41.88, -87.63, 8_900_000, 40), ("Johannesburg", -26.20, 28.05, 6_000_000, 35),
    ("Sydney", -33.87, 151.21, 5_300_000, 35), ("Nairobi", -1.29, 36.82, 5_100_000, 25),
    ("Riyadh", 24.71, 46.68, 7_700_000, 30), ("Baghdad", 33.31, 44.37, 7_700_000, 25),
    ("Singapore", 1.35, 103.82, 6_000_000, 20), ("Madrid", 40.42, -3.70, 6_700_000, 30),
    ("Toronto", 43.65, -79.38, 6_400_000, 35), ("Berlin", 52.52, 13.40, 3_700_000, 25),
    ("Colombo", 6.93, 79.86, 5_600_000, 20), ("Ahmedabad", 23.03, 72.58, 8_400_000, 30),
    ("Pune", 18.52, 73.86, 6_800_000, 25), ("Surat", 21.17, 72.83, 7_500_000, 25),
    ("Jaipur", 26.91, 75.79, 4_100_000, 22),
]
#: mean rural population density on land [people / km^2]
RURAL_DENSITY = 22.0


@dataclass
class PopulationModel:
    """Population density lookup [people / km^2]."""
    land: list = field(default_factory=lambda: list(_LAND))
    cities: list = field(default_factory=lambda: list(_CITIES))
    rural_density: float = RURAL_DENSITY
    grid: np.ndarray = None            # optional real raster, shape (nlat, nlon)
    land_grid: np.ndarray = None       # optional real land fraction on the same grid
    source: str = "coarse hand-authored grid"

    @classmethod
    def from_grid(cls, grid: np.ndarray, rural_density: float = RURAL_DENSITY,
                  land_grid: np.ndarray | None = None, source: str = "user raster"):
        """Use a real gridded raster instead of the coarse land model.

        ``grid[i, j]`` is people per km^2, with ``i`` running 90N -> 90S and ``j`` 180W ->
        180E.  Nothing else in the project changes.
        """
        return cls(grid=np.asarray(grid, float), rural_density=rural_density,
                   land_grid=None if land_grid is None else np.asarray(land_grid, float),
                   source=source)

    @classmethod
    def real(cls, path: str | None = None):
        """GHS-POP 2020 + Natural Earth land, as prepared by :func:`.population.prepare_population`."""
        from .population import load_population_grid
        g = load_population_grid(path)
        return cls.from_grid(g["density"], land_grid=g["land_fraction"], source=str(g["source"]))

    @classmethod
    def coarse(cls):
        """The hand-authored 10-degree grid with 51 metropolitan centres."""
        return cls()

    @classmethod
    def best_available(cls):
        """The real raster if it has been prepared, otherwise the coarse grid."""
        try:
            return cls.real()
        except FileNotFoundError:
            return cls.coarse()

    @staticmethod
    def _cell(grid, lat, lon):
        nlat, nlon = grid.shape
        i = np.clip(((90.0 - lat) / (180.0 / nlat)).astype(int), 0, nlat - 1)
        j = np.clip(((lon + 180.0) / (360.0 / nlon)).astype(int), 0, nlon - 1)
        return grid[i, j]

    def land_fraction(self, lat, lon) -> np.ndarray:
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = ((np.atleast_1d(np.asarray(lon, float)) + 180.0) % 360.0) - 180.0
        if self.land_grid is not None:
            return self._cell(self.land_grid, lat, lon)
        i = np.clip(((90.0 - lat) / 10.0).astype(int), 0, len(self.land) - 1)
        j = np.clip(((lon + 180.0) / 10.0).astype(int), 0, len(self.land[0]) - 1)
        return np.array([float(self.land[a][b]) for a, b in zip(i, j)])

    def density(self, lat, lon) -> np.ndarray:
        """People per km^2 at each point."""
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = ((np.atleast_1d(np.asarray(lon, float)) + 180.0) % 360.0) - 180.0
        if self.grid is not None:
            return self._cell(self.grid, lat, lon)
        dens = self.land_fraction(lat, lon) * self.rural_density
        for name, clat, clon, pop, rad in self.cities:
            d = great_circle_km(lat, lon, clat, clon)
            inside = d < rad
            if np.any(inside):
                city_density = pop / (np.pi * rad ** 2)
                dens = np.where(inside, np.maximum(dens, city_density), dens)
        return dens

    def nearest_city(self, lat: float, lon: float) -> dict:
        d = [(great_circle_km(lat, lon, c[1], c[2]), c) for c in self.cities]
        dist, c = min(d, key=lambda v: float(v[0]))
        return {"city": c[0], "distance_km": round(float(dist), 1), "population": c[3]}


def casualty_area(debris_area_m2) -> np.ndarray:
    """FAA casualty area: ``(sqrt(A_debris) + sqrt(A_person))^2`` [m^2]."""
    a = np.atleast_1d(np.asarray(debris_area_m2, float))
    return (np.sqrt(np.maximum(a, 0.0)) + np.sqrt(PERSON_AREA_M2)) ** 2


def estimate_survival(cloud, hazard, n: int = 120, material: str = "aluminium",
                      seed: int = 0, entry_speed_ms: float = 7600.0,
                      flight_path_deg: float = -1.4) -> float:
    """Fraction of fragments that reach the ground, from the adaptive propagator.

    The vectorised ensemble that builds the footprint keeps mass fixed; the adaptive
    integrator models stagnation heating and ablation, so a random subsample is flown with
    it to measure how many fragments survive to impact.
    """
    from .propagator import DebrisObject, entry_state, propagate
    sub = cloud.frame.sample(min(n, len(cloud.frame)), random_state=seed)
    st = entry_state(hazard.altitude_km * 1e3, entry_speed_ms, flight_path_deg)
    hit = [propagate(st, DebrisObject("fragment", mass=f.mass_kg,
                                      area=f.mass_kg / max(f.bc, 1e-6), shape="tumbling",
                                      material=material), store=False).outcome == "impact"
           for f in sub.itertuples()]
    return float(np.mean(hit))


def assess_footprint(fp, population: PopulationModel | None = None,
                     n_fragments: int | None = None,
                     survival_fraction: float | None = None) -> dict:
    """Expected casualties for a whole footprint, and where the risk sits.

    Only fragments above the 15 J lethality threshold contribute — below it an impact is not
    casualty-capable.

    ``expected_casualties`` is the quantity reported so far: the mean over Monte Carlo
    samples with a 1 m^2 debris footprint, i.e. the expectation for ONE fragment.  With
    ``n_fragments`` (and optionally ``survival_fraction``), ``expected_casualties_event`` is
    also given: the total for the whole re-entry, summing each surviving fragment's own
    casualty area — the number the 1e-4 convention is meant for.
    """
    pop = population or PopulationModel.best_available()
    f = fp.frame
    lethal = f.energy_J.values > LETHAL_ENERGY_J
    dens = pop.density(f.lat.values, f.lon.values)              # people / km^2
    land = pop.land_fraction(f.lat.values, f.lon.values)
    # each Monte Carlo sample represents 1/N of the probability mass
    ca_km2 = casualty_area(f.mass_kg.values * 0.0 + 1.0) / 1e6   # ~1 m^2 debris footprint
    ec_per = dens * ca_km2 * lethal
    ec = float(np.mean(ec_per))
    centre = (float(f.lat.mean()), float(f.lon.mean()))
    band = _band(ec)
    event = {}
    if n_fragments:
        area = f.mass_kg.values / np.maximum(f.bc.values, 1e-6)       # each fragment's own
        per = float(np.mean(dens * casualty_area(area) / 1e6 * lethal))
        surv = 1.0 if survival_fraction is None else float(survival_fraction)
        ec_event = n_fragments * surv * per
        event = {"expected_casualties_event": ec_event, "risk_band_event": _band(ec_event),
                 "fragments": int(n_fragments), "survival_fraction": round(surv, 3)}
    return {"expected_casualties": ec, "risk_band": band, "population_model": pop.source,
            **event,
            "threshold": EC_THRESHOLD, "exceeds_threshold": bool(ec > EC_THRESHOLD),
            "land_impact_probability": round(float(land.mean()), 4),
            "casualty_capable_fraction": round(float(lethal.mean()), 4),
            "mean_population_density": round(float(dens.mean()), 2),
            "max_population_density": round(float(dens.max()), 1),
            "centre_lat": round(centre[0], 3), "centre_lon": round(centre[1], 3),
            **{f"nearest_{k}": v for k, v in pop.nearest_city(*centre).items()}}


def _band(ec: float) -> str:
    return ("NEGLIGIBLE" if ec < 1e-7 else "ACCEPTABLE" if ec < EC_THRESHOLD
            else "ELEVATED" if ec < 10 * EC_THRESHOLD else "HIGH")


def scenario_risk(footprints: dict, population: PopulationModel | None = None) -> pd.DataFrame:
    """Compare several named footprints in one table."""
    rows = []
    for name, fp in footprints.items():
        r = assess_footprint(fp, population)
        rows.append({"scenario": name, "expected_casualties": r["expected_casualties"],
                     "risk_band": r["risk_band"],
                     "land_probability": r["land_impact_probability"],
                     "nearest_city": r["nearest_city"],
                     "distance_km": r["nearest_distance_km"]})
    return pd.DataFrame(rows)


def population_along_track(lat0: float, lon0: float, lat1: float, lon1: float,
                           n: int = 200, population: PopulationModel | None = None
                           ) -> pd.DataFrame:
    """Population density sampled along a ground track, for the notebook figure."""
    pop = population or PopulationModel.best_available()
    lat = np.linspace(lat0, lat1, n)
    lon = np.linspace(lon0, lon1, n)
    return pd.DataFrame({"lat": lat, "lon": lon, "density": pop.density(lat, lon),
                         "land_fraction": pop.land_fraction(lat, lon)})
