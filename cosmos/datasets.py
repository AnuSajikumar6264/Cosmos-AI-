"""Every dataset the project uses: where it comes from, where it lives, and whether it is here.

    python -m cosmos.datasets status                 # what is present, what is missing
    python -m cosmos.datasets fetch smap_msl esa_ad  # download public data (with checksums)
    python -m cosmos.datasets prepare esa_ad         # one-off conversions to plain arrays

Public datasets are downloaded and verified here.  Two need personal access and are never
downloaded by the project: ESA DISCOS (a token, read by :mod:`cosmos.reentry.discos`) and the
SPARK / DUST image sets (placed by hand, converted by :mod:`cosmos.debris.realimagery`).
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass

from .common import catalogue_dir, feature_dir


@dataclass
class Dataset:
    name: str
    feature: str
    real: bool
    description: str
    source: str
    licence: str
    folder: tuple                         # (feature, *parts) under data/
    required: tuple = ()                  # files that must exist for the dataset to be "here"
    downloads: tuple = ()                 # (url, relative path, md5 or None)
    access: str = "public"                # public | token | request | generated
    prepare: str | None = None            # dotted function that prepares it
    prepared: tuple = ()                  # files that prepare() writes

    @property
    def path(self) -> str:
        return feature_dir(*self.folder)

    def present(self) -> bool:
        return all(os.path.exists(os.path.join(self.path, f)) for f in self.required)

    def is_prepared(self) -> bool:
        return all(os.path.exists(os.path.join(self.path, f)) for f in self.prepared)


_SMAP = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
_SMAP_MIRROR = "https://github.com/imperial-qore/TranAD/raw/main/data/SMAP_MSL"
_ZENODO = "https://zenodo.org/api/records/12528696/files"
_GHS = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
        "GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip")
_NE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
       "ne_50m_land.geojson")

REGISTRY = {d.name: d for d in [
    Dataset("celestrak_satcat", "debris", True,
            "CelesTrak SATCAT: 70,288 catalogued objects (35,834 debris) + GP element sets",
            "https://celestrak.org (T.S. Kelso)", "public, no credentials",
            ("debris", "celestrak"), required=("satcat.csv",)),
    Dataset("measured_lifetimes", "reentry", True,
            "Measured orbital lifetimes (decay date - break-up date), 7,711 fragments",
            "derived from celestrak_satcat by cosmos.debris.export_derived",
            "as celestrak_satcat", ("reentry", "derived"),
            required=("measured_lifetimes.csv",)),
    Dataset("smap_msl", "health", True,
            "NASA SMAP / MSL telemetry, 82 channels, 105 labelled anomaly windows",
            "Hundman et al., KDD 2018 (Telemanom); labels from the official repository, arrays "
            "from a byte-identical mirror (the original S3 bucket now returns 403)",
            "NASA/JPL public release", ("health", "external", "smap_msl"),
            required=("labeled_anomalies.csv", "train", "test"),
            downloads=((_SMAP, "labeled_anomalies.csv", "c54f6c09f44410763e33aac7329d769a"),)),
    Dataset("esa_ad", "health", True,
            "ESA Anomaly Dataset (ESA-ADB): 3 missions, 224 channels, labelled by ESA operators",
            "Zenodo record 12528696, doi:10.5281/zenodo.12528696 (Kotowski, De Canio, "
            "Haskamp et al., 2024)", "see the Zenodo record",
            ("health", "external", "esa_ad"),
            required=("ESA-Mission1.zip", "ESA-Mission2.zip", "ESA-Mission3.zip"),
            downloads=tuple((f"{_ZENODO}/ESA-Mission{m}.zip/content", f"ESA-Mission{m}.zip",
                             md5) for m, md5 in ((1, "80750189d171f5f398fb3d96c49df12b"),
                                                 (2, "0b7505b7f0731ca037ee889ca2a520ce"),
                                                 (3, "34b8a8857e2242fb1f00b368b7421595"))),
            prepare="cosmos.health.realdata:prepare_esa_mission",
            prepared=tuple(f"prepared/mission{m}.npz" for m in (1, 2, 3))),
    Dataset("ghs_pop", "reentry", True,
            "GHS-POP 2020 population, 30 arc-second (about 1 km) WGS-84, release R2023A",
            "European Commission JRC, Global Human Settlement Layer", "CC BY 4.0",
            ("reentry", "external", "ghs_pop"),
            required=("GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip",),
            downloads=((_GHS, "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip",
                        "a6b110f027c59b726d504896b4e8de43"),)),
    Dataset("natural_earth", "reentry", True,
            "Natural Earth 1:50m land polygons (the land mask GHS-POP lacks)",
            "naturalearthdata.com", "public domain", ("reentry", "external", "natural_earth"),
            required=("ne_50m_land.geojson",),
            downloads=((_NE, "ne_50m_land.geojson", None),),
            prepare="cosmos.reentry.population:prepare_population",
            prepared=("../population_grid_0p25deg.npz",)),
    Dataset("discos", "reentry", True,
            "ESA DISCOS fragmentation events, re-entries and object properties",
            "ESA Space Debris Office, DISCOSweb API v2", "ESA terms of use; personal token",
            ("reentry", "external", "discos"), required=("fragmentations.json",),
            access="token"),
    Dataset("spark", "debris", True, "SPARK space-object images with boxes and classes",
            "SnT, University of Luxembourg", "by request", ("debris", "external", "spark"),
            required=(), access="request"),
    Dataset("dust", "debris", True, "DUST space-object images with boxes and classes",
            "see the dataset's own licence", "by request", ("debris", "external", "dust"),
            required=(), access="request"),
    Dataset("simulated_telemetry", "health", False,
            "8-subsystem, 16-channel telemetry with 6 injectable fault signatures",
            "generated by cosmos.health.simulate_telemetry", "this project",
            ("health", "simulated"), access="generated"),
    Dataset("frame_pairs", "debris", False,
            "Synthetic star-field frame pairs, YOLO-format export (48 scenes, 381 boxes)",
            "generated by cosmos.debris.export_yolo_dataset", "this project",
            ("debris", "vision", "yolo_dataset"), required=("data.yaml",), access="generated"),
]}


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def status() -> "pd.DataFrame":
    """One row per dataset: present, prepared, real or simulated, how it is obtained."""
    import pandas as pd
    rows = []
    for d in REGISTRY.values():
        if d.name == "celestrak_satcat":
            here = os.path.isfile(os.path.join(catalogue_dir(), "satcat.csv"))
        elif d.access == "request":
            here = any(f for f in os.listdir(d.path) if not f.startswith("."))
        elif not d.required:
            here = True
        else:
            here = d.present()
        rows.append({"dataset": d.name, "feature": d.feature,
                     "type": "real" if d.real else "simulated", "access": d.access,
                     "present": here, "prepared": d.is_prepared() if d.prepared else None,
                     "description": d.description})
    return pd.DataFrame(rows)


def fetch(name: str, force: bool = False, verbose: bool = True) -> list:
    """Download a public dataset's files and verify their checksums."""
    d = REGISTRY[name]
    if d.access == "token":
        from .reentry import discos
        discos.fetch("fragmentations")
        discos.fetch("reentries")
        return [discos.discos_dir()]
    if d.access == "request":
        raise PermissionError(f"{name} is distributed by request; place the archives in "
                              f"{d.path} and run cosmos.debris.realimagery.convert_to_yolo")
    if name == "smap_msl":
        return _fetch_smap_msl(d, force, verbose)
    out = []
    for url, rel, md5 in d.downloads:
        dst = os.path.join(d.path, rel)
        if force or not os.path.isfile(dst):
            if verbose:
                print(f"  downloading {rel} ...")
            urllib.request.urlretrieve(url, dst)
        if md5 and _md5(dst) != md5:
            raise IOError(f"{dst}: checksum mismatch (expected {md5})")
        out.append(dst)
    return out


def _fetch_smap_msl(d: Dataset, force: bool, verbose: bool) -> list:
    import pandas as pd
    url, rel, md5 = d.downloads[0]
    lab = os.path.join(d.path, rel)
    if force or not os.path.isfile(lab):
        urllib.request.urlretrieve(url, lab)
    if _md5(lab) != md5:
        raise IOError(f"{lab}: checksum mismatch")
    chans = pd.read_csv(lab).chan_id
    for split in ("train", "test"):
        os.makedirs(os.path.join(d.path, split), exist_ok=True)
        for c in chans:
            dst = os.path.join(d.path, split, f"{c}.npy")
            if force or not os.path.isfile(dst):
                urllib.request.urlretrieve(f"{_SMAP_MIRROR}/{split}/{c}.npy", dst)
    if verbose:
        print(f"  smap_msl: {len(chans)} channels in {d.path}")
    return [d.path]


def prepare(name: str) -> list:
    """Run a dataset's one-off conversion (ESA pickles -> npz, rasters -> 0.25 deg grid)."""
    import importlib
    d = REGISTRY[name]
    if not d.prepare:
        return []
    mod, fn = d.prepare.split(":")
    f = getattr(importlib.import_module(mod), fn)
    if name == "esa_ad":
        return [f(m) for m in (1, 2, 3)]
    return [f()]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_colwidth", 70):
            print(status().to_string(index=False))
        return 0
    if cmd in ("fetch", "prepare"):
        for name in argv[1:] or [n for n, d in REGISTRY.items() if d.downloads]:
            (fetch if cmd == "fetch" else prepare)(name)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
