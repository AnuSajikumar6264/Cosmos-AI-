"""FEATURE 3 (g) — a real population raster and land mask for the ground-risk stage.

The hand-authored grid in :mod:`.risk` gets the land/ocean and city/countryside contrast
right but is accurate only to about an order of magnitude.  This module replaces it with:

GHS-POP 2020   JRC Global Human Settlement Layer, population, epoch 2020, 30 arc-second
               (about 1 km), WGS-84, release R2023A.  CC BY 4.0.
Natural Earth  1:50 m land polygons, public domain — GHS-POP stores ocean as zero, so it
               cannot tell an empty desert from open sea on its own.

:func:`prepare_population` aggregates both onto a 0.25 degree grid once (a few minutes)
and writes a small ``.npz``; :func:`load_population_grid` and
:meth:`PopulationModel.best_available` read it.

    prepare_population()
    pop = PopulationModel.best_available()
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile

import numpy as np

from ..common import feature_dir

GHS_ZIP = "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip"
GHS_TIF = "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
NE_LAND = "ne_50m_land.geojson"
GRID_FILE = "population_grid_0p25deg.npz"
R_MEAN_KM = 6371.0088                    # IUGG mean Earth radius


def population_dir() -> str:
    return feature_dir("reentry", "external")


def cell_area_km2(lat_edges: np.ndarray, dlon_deg: float) -> np.ndarray:
    """Area of each latitude band of a regular grid on a sphere [km^2], north to south."""
    s = np.sin(np.radians(lat_edges))
    return R_MEAN_KM ** 2 * np.radians(dlon_deg) * np.abs(s[:-1] - s[1:])


def _aggregate_ghs(tif_path: str, res: float) -> np.ndarray:
    """Sum GHS-POP people per pixel into ``res``-degree cells.

    The raster is decoded one 256 x 256 tile at a time, so memory stays small (the full
    image would be 7.4 GB as float64).
    """
    import tifffile
    nlat, nlon = int(round(180 / res)), int(round(360 / res))
    out = np.zeros(nlat * nlon)
    with tifffile.TiffFile(tif_path) as t:
        page = t.pages[0]
        sx, sy = page.tags["ModelPixelScaleTag"].value[:2]
        x0, y0 = page.tags["ModelTiepointTag"].value[3:5]
        h, w = page.shape
        for seg, idx, _shape in page.segments():
            if seg is None:
                continue
            tile = np.nan_to_num(seg[0, :, :, 0].astype(np.float64))   # ocean = 0
            r0, c0 = idx[2], idx[3]
            rows = min(tile.shape[0], h - r0)
            cols = min(tile.shape[1], w - c0)
            tile = tile[:rows, :cols]
            lat_c = y0 - (r0 + np.arange(rows) + 0.5) * sy
            lon_c = x0 + (c0 + np.arange(cols) + 0.5) * sx
            i = np.clip(((90.0 - lat_c) / res).astype(int), 0, nlat - 1)
            j = np.clip(((lon_c + 180.0) / res).astype(int), 0, nlon - 1)
            out += np.bincount((i[:, None] * nlon + j[None, :]).ravel(), weights=tile.ravel(),
                               minlength=nlat * nlon)
    return out.reshape(nlat, nlon)


def _land_fraction(geojson_path: str, res: float, oversample: int = 10) -> np.ndarray:
    """Rasterise Natural Earth land polygons and average them into ``res``-degree cells."""
    from PIL import Image, ImageDraw
    nlat, nlon = int(round(180 / res)), int(round(360 / res))
    W, H = nlon * oversample, nlat * oversample
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    px = lambda ring: [((lon + 180.0) / 360.0 * W, (90.0 - lat) / 180.0 * H)   # noqa: E731
                       for lon, lat in ring]
    with open(geojson_path) as f:
        gj = json.load(f)
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            draw.polygon(px(poly[0]), fill=255)              # exterior ring
            for hole in poly[1:]:
                draw.polygon(px(hole), fill=0)               # lakes inside the land
    a = np.asarray(img, np.float64) / 255.0
    return a.reshape(nlat, oversample, nlon, oversample).mean(axis=(1, 3))


def prepare_population(res: float = 0.25, verbose: bool = True) -> str:
    """Build ``population_grid_0p25deg.npz`` from the GHS-POP archive and Natural Earth."""
    d = population_dir()
    zip_path = os.path.join(d, "ghs_pop", GHS_ZIP)
    ne_path = os.path.join(d, "natural_earth", NE_LAND)
    for p in (zip_path, ne_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} not found — run: python -m cosmos.datasets fetch "
                                    "ghs_pop natural_earth")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extract(GHS_TIF, tmp)
        people = _aggregate_ghs(os.path.join(tmp, GHS_TIF), res)
    land = _land_fraction(ne_path, res)
    nlat, nlon = people.shape
    lat_edges = np.linspace(90.0, -90.0, nlat + 1)
    area = cell_area_km2(lat_edges, 360.0 / nlon)[:, None] * np.ones((1, nlon))
    density = people / area
    out = os.path.join(d, GRID_FILE)
    np.savez_compressed(out, density=density.astype(np.float32),
                        land_fraction=land.astype(np.float32),
                        people=people.astype(np.float64), res_deg=res,
                        source="GHS-POP E2020 R2023A 30ss (JRC, CC BY 4.0); "
                               "Natural Earth 1:50m land (public domain)")
    if verbose:
        print(f"  wrote {out}: total population {people.sum() / 1e9:.3f} billion, "
              f"land fraction of the sphere {np.sum(land * area) / area.sum():.3f}")
    return out


def load_population_grid(path: str | None = None) -> dict:
    """The prepared grid: ``density`` [people/km^2], ``land_fraction`` [0-1], ``people``."""
    path = path or os.path.join(population_dir(), GRID_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found — run prepare_population()")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
