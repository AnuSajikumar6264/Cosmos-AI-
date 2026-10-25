"""Reproducible acquisition of every external dataset COSMOS-AI uses.

    python -m cosmos.ai.data.download status
    python -m cosmos.ai.data.download opssat_ad esa_adb spotgeo celestrak

Rules this module enforces, because each one has bitten a benchmark somewhere:

* **Raw data is immutable.**  Files land in ``<dataset>/raw/``; nothing in COSMOS-AI writes
  there except this module, and an existing file whose checksum matches is never fetched
  again.  Preprocessed data goes to ``<dataset>/processed/``.
* **Every file is verified.**  Against the publisher's checksum when one is published
  (Zenodo MD5), otherwise a SHA-256 is computed at download and recorded, so a later copy can
  be checked against it.
* **Downloads resume.**  A partial ``.part`` file is continued with an HTTP Range request.
* **Every dataset gets a ``MANIFEST.json``**: source URL, DOI, licence, version, retrieval
  time, and one entry per file with size and checksums.  Datasets that could not be obtained
  get a manifest too, stating exactly why.
* **Credentials never touch the repository.**  Nothing here needs one; sources that do
  (Space-Track, DISCOSweb, SPARK by request) are recorded as unavailable instead.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile

from ...common import feature_dir

USER_AGENT = "COSMOS-AI dataset fetcher (research; contact via project README)"


# ============================================================================ low level
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def file_hashes(path: str) -> dict:
    """MD5 and SHA-256 of a file, read in 16 MB blocks."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            md5.update(b)
            sha.update(b)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def _session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def fetch_url(url: str, dst: str, md5: str | None = None, size: int | None = None,
              resume: bool = True, retries: int = 5, timeout: float = 120.0,
              verbose: bool = True) -> dict:
    """Download ``url`` to ``dst`` (resumable), verify, and return a manifest entry.

    If ``dst`` already exists and matches ``md5`` (or, without one, ``size``), nothing is
    downloaded and the entry says ``"status": "cached"``.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst):
        h = file_hashes(dst)
        if (md5 and h["md5"] == md5) or (not md5 and (size is None
                                                      or os.path.getsize(dst) == size)):
            return {"url": url, "bytes": os.path.getsize(dst), **h, "status": "cached",
                    "md5_verified": bool(md5)}
        if verbose:
            print(f"  {os.path.basename(dst)}: checksum mismatch on disk, re-downloading")
        os.remove(dst)
    part = dst + ".part"
    sess = _session()
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(part) if (resume and os.path.isfile(part)) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with sess.get(url, stream=True, timeout=timeout, headers=headers,
                          allow_redirects=True) as r:
                if r.status_code == 416:                     # already complete
                    break
                r.raise_for_status()
                mode = "ab" if (have and r.status_code == 206) else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
                done, t0, last = (have if mode == "ab" else 0), time.time(), 0.0
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if verbose and total > (50 << 20) and time.time() - last > 30:
                            last = time.time()
                            print(f"    {os.path.basename(dst)}: {done / 2**20:,.0f} / "
                                  f"{total / 2**20:,.0f} MB "
                                  f"({done / 2**20 / max(time.time() - t0, 1e-9):.1f} MB/s)")
            break
        except Exception as exc:                              # network hiccup: resume
            if attempt == retries:
                raise
            if verbose:
                print(f"    retry {attempt}/{retries} after {type(exc).__name__}: {exc}")
            time.sleep(min(60, 5 * attempt))
    h = file_hashes(part)
    if md5 and h["md5"] != md5:
        os.replace(part, part + ".bad")
        raise IOError(f"{dst}: MD5 {h['md5']} does not match the published {md5}")
    if size is not None and os.path.getsize(part) != size:
        raise IOError(f"{dst}: {os.path.getsize(part)} bytes, expected {size}")
    os.replace(part, dst)
    return {"url": url, "bytes": os.path.getsize(dst), **h, "status": "downloaded",
            "md5_verified": bool(md5)}


def safe_extract(archive: str, out_dir: str) -> list:
    """Extract a tar or zip archive, refusing absolute paths, ``..`` and links."""
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.realpath(out_dir)
    names = []

    def _ok(name):
        p = os.path.realpath(os.path.join(root, name))
        return p == root or p.startswith(root + os.sep)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if not _ok(m.filename):
                    raise IOError(f"unsafe path in {archive}: {m.filename}")
                names.append(m.filename)
            z.extractall(root)
    else:
        with tarfile.open(archive) as t:
            members = []
            for m in t.getmembers():
                if not _ok(m.name) or m.issym() or m.islnk() or m.isdev():
                    continue                                  # skip, never follow links
                members.append(m)
                names.append(m.name)
            t.extractall(root, members=members, filter="data")
    return names


# ============================================================================ manifests
def dataset_root(feature: str, name: str) -> str:
    return feature_dir(feature, name)


def write_manifest(folder: str, meta: dict, files: list | None = None) -> str:
    """Write (or update) ``MANIFEST.json``; file entries are merged by path."""
    path = os.path.join(folder, "MANIFEST.json")
    old = {}
    if os.path.isfile(path):
        with open(path) as f:
            old = json.load(f)
    merged = {e["path"]: e for e in old.get("files", [])}
    for e in files or []:
        merged[e["path"]] = e
    out = {**old, **meta, "manifest_written_utc": utc_now(),
           "files": sorted(merged.values(), key=lambda e: e["path"])}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return path


def read_manifest(folder: str) -> dict:
    path = os.path.join(folder, "MANIFEST.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ============================================================================ Zenodo / GitHub
def zenodo_record(record_id: int | str) -> dict:
    """Zenodo API record (follows redirects to the newest version of a concept record)."""
    r = _session().get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_zenodo(record_id, folder: str, keys=None, verbose: bool = True) -> tuple:
    """Download every (or the listed) file of a Zenodo record into ``folder/raw``."""
    rec = zenodo_record(record_id)
    entries = []
    for f in rec["files"]:
        if keys and f["key"] not in keys:
            continue
        algo, _, digest = f["checksum"].partition(":")
        if verbose:
            print(f"  zenodo {rec['id']}: {f['key']} ({f['size'] / 2**20:,.1f} MB)")
        e = fetch_url(f["links"]["self"], os.path.join(folder, "raw", f["key"]),
                      md5=digest if algo == "md5" else None, size=f["size"], verbose=verbose)
        entries.append({"path": f"raw/{f['key']}", **e, "published_checksum": f["checksum"]})
    return rec, entries


def fetch_github_snapshot(repo: str, sha: str, folder: str, verbose: bool = True) -> dict:
    """Download a pinned commit of a GitHub repository and extract it for reading.

    The archive is kept under ``raw/``; the extracted tree is reference material that
    COSMOS-AI reads (scripts, metric definitions) but never executes in place.
    """
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    name = f"{repo.replace('/', '__')}-{sha[:12]}.tar.gz"
    dst = os.path.join(folder, "raw", name)
    e = fetch_url(url, dst, verbose=verbose)
    out = os.path.join(folder, "raw", "official_repo")
    marker = os.path.join(out, ".commit")
    if not (os.path.isfile(marker) and open(marker).read().strip() == sha):
        if os.path.isdir(out):
            shutil.rmtree(out)
        safe_extract(dst, out)
        with open(marker, "w") as f:
            f.write(sha)
    return {"path": f"raw/{name}", **e, "repo": repo, "commit": sha,
            "extracted_to": "raw/official_repo"}


def github_head(repo: str, branch: str = "main") -> str:
    r = _session().get(f"https://api.github.com/repos/{repo}/commits/{branch}", timeout=60)
    r.raise_for_status()
    return r.json()["sha"]


# ============================================================================ datasets
def fetch_opssat_ad(verbose: bool = True) -> str:
    """OPSSAT-AD (Ruszczak et al., Sci. Data 12:710, 2025): data from Zenodo, code from GitHub."""
    folder = dataset_root("health", "opssat_ad")
    rec, files = fetch_zenodo(12588358, folder, verbose=verbose)
    sha = github_head("kplabs-pl/OPS-SAT-AD")
    files.append(fetch_github_snapshot("kplabs-pl/OPS-SAT-AD", sha, folder, verbose))
    m = rec["metadata"]
    write_manifest(folder, {
        "dataset": "opssat_ad", "class": "REAL", "status": "downloaded",
        "title": m.get("title"), "zenodo_record": rec["id"],
        "zenodo_requested_record": 12588358, "doi": rec.get("doi"),
        "paper_doi": "10.1038/s41597-025-05035-3",
        "licence": (m.get("license") or {}).get("id"), "code_licence": "MIT",
        "publication_date": m.get("publication_date"),
        "creators": [c["name"] for c in m.get("creators", [])],
        "retrieved_utc": utc_now(),
        "source_urls": [f"https://zenodo.org/records/{rec['id']}",
                        "https://github.com/kplabs-pl/OPS-SAT-AD"],
        "notes": ["Zenodo record 12588358 resolves to its newest version; the version "
                  "actually downloaded is the zenodo_record field.",
                  "segments.csv = raw telemetry fragments; dataset.csv = the 18 handcrafted "
                  "features with the official train/test assignment."],
    }, files)
    return folder


#: ESA-AD archives were downloaded earlier by cosmos.datasets into external/esa_ad; they are
#: linked, not copied (11 GB), and re-verified against the Zenodo MD5s
ESA_ZENODO = 12528696
ESA_ZENODO_LATEST = 15237121


def fetch_esa_adb(verbose: bool = True) -> str:
    """ESA-ADB: link the existing archives, verify them, snapshot the official benchmark code."""
    folder = dataset_root("health", "esa_adb")
    src = feature_dir("health", "external", "esa_ad")
    rec = zenodo_record(ESA_ZENODO)
    published = {f["key"]: f for f in rec["files"]}
    files = []
    os.makedirs(os.path.join(folder, "raw"), exist_ok=True)
    for key, f in sorted(published.items()):
        local = os.path.join(src, key)
        link = os.path.join(folder, "raw", key)
        if os.path.isfile(local):
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(local, os.path.dirname(link)), link)
            e = fetch_url(f["links"]["self"], local, md5=f["checksum"].split(":")[1],
                          size=f["size"], verbose=verbose)   # verifies; downloads only if bad
        else:
            e = fetch_url(f["links"]["self"], link, md5=f["checksum"].split(":")[1],
                          size=f["size"], verbose=verbose)
        files.append({"path": f"raw/{key}", **e, "published_checksum": f["checksum"],
                      "stored_at": os.path.relpath(local if os.path.isfile(local) else link,
                                                   folder)})
    sha = github_head("kplabs-pl/ESA-ADB")
    files.append(fetch_github_snapshot("kplabs-pl/ESA-ADB", sha, folder, verbose))
    files += check_esa_revision(folder, verbose=verbose)
    m = rec["metadata"]
    write_manifest(folder, {
        "dataset": "esa_adb", "class": "REAL", "status": "downloaded",
        "versions": {str(ESA_ZENODO): "v1.0, 2024-06-25 (raw/*.zip; used by the original "
                                      "COSMOS code)",
                     str(ESA_ZENODO_LATEST): "2025-04-17 revision: identical to v1.0 except "
                                             "channels.csv (adds 'Categorical'); the changed "
                                             "members are in raw/zenodo_15237121/members/"},
        "title": m.get("title"), "zenodo_record": rec["id"], "doi": rec.get("doi"),
        "paper": "Kotowski et al., DMLR 3(23), 2026; arXiv 2406.17826",
        "licence": (m.get("license") or {}).get("id") or
        "not stated in the Zenodo record; CC BY 3.0 IGO per the DMLR paper",
        "code_licence": "MIT", "publication_date": m.get("publication_date"),
        "retrieved_utc": utc_now(),
        "source_urls": [f"https://zenodo.org/records/{rec['id']}",
                        "https://github.com/kplabs-pl/ESA-ADB"],
        "notes": ["Archives physically live in data/feature1_health/external/esa_ad (fetched "
                  "by cosmos.datasets); raw/ holds relative symlinks so nothing is duplicated.",
                  "Mission 3 is excluded from algorithm comparison by the benchmark authors."],
    }, files)
    return folder


class HTTPRangeFile:
    """Read-only, seekable view of a remote file through HTTP Range requests.

    Enough for :mod:`zipfile` to read a remote archive's central directory (the CRC-32 and
    size of every member) without downloading the archive.
    """

    def __init__(self, url: str):
        self.s = _session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos, self.requests = r.url, int(r.headers["Content-Length"]), 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        self.pos += len(r.content)
        self.requests += 1
        return r.content


def _remote_member(f: HTTPRangeFile, info) -> bytes:
    """One member's bytes; Deflate64 (method 9) is decoded by Info-ZIP ``funzip``."""
    import struct
    import subprocess
    import zlib
    f.seek(info.header_offset)
    hdr = f.read(30)
    fn_len, ex_len = struct.unpack("<HH", hdr[26:30])
    f.seek(info.header_offset)
    blob = f.read(30 + fn_len + ex_len + info.compress_size)
    data = subprocess.run(["funzip"], input=blob, capture_output=True, check=True).stdout
    if (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC or len(data) != info.file_size:
        raise IOError(f"{info.filename}: CRC/size mismatch after decoding")
    return data


def check_esa_revision(folder: str, verbose: bool = True) -> list:
    """Compare the newest ESA-AD release with the local v1.0 member by member.

    Reads the remote central directories (a few Range requests), then downloads only the
    members whose CRC-32 differs.  For the 2025-04-17 revision that is ``channels.csv`` alone.
    """
    rec = zenodo_record(ESA_ZENODO_LATEST)
    src = feature_dir("health", "external", "esa_ad")
    out, changed_all = [], {}
    for f in rec["files"]:
        if f["key"] not in ("ESA-Mission1.zip", "ESA-Mission2.zip"):
            continue                                  # Mission 3: excluded from benchmarking
        rf = HTTPRangeFile(f["links"]["self"])
        zr = zipfile.ZipFile(rf)
        zl = zipfile.ZipFile(os.path.join(src, f["key"]))
        local = {i.filename: (i.CRC, i.file_size) for i in zl.infolist()}
        changed = [i for i in zr.infolist() if local.get(i.filename) != (i.CRC, i.file_size)]
        changed_all[f["key"]] = [i.filename for i in changed]
        for i in changed:
            rel = f"raw/zenodo_{ESA_ZENODO_LATEST}/members/" + i.filename.replace("/", "_", 1)
            dst = os.path.join(folder, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.isfile(dst):
                with open(dst, "wb") as fh:
                    fh.write(_remote_member(rf, i))
            out.append({"path": rel, "bytes": os.path.getsize(dst), **file_hashes(dst),
                        "status": "member fetched by HTTP Range; CRC-32 verified",
                        "url": f["links"]["self"], "zenodo_record": ESA_ZENODO_LATEST})
        if verbose:
            print(f"  revision {ESA_ZENODO_LATEST} {f['key']}: {len(changed)} changed member(s) "
                  f"{changed_all[f['key']]} ({rf.requests} range requests)")
    return out


def fetch_spotgeo(verbose: bool = True) -> str:
    """SpotGEO (Chen et al., CVPR Workshops 2021): real unresolved GEO targets, CC BY 4.0."""
    folder = dataset_root("debris", "spotgeo")
    rec, files = fetch_zenodo(4432143, folder, verbose=verbose)
    # the official starter kit holds validation.py, the scorer run on the Kelvins server
    kit, kit_files = fetch_zenodo(3874368, os.path.join(folder, "starter_kit"),
                                  verbose=verbose)
    files += [{**e, "path": "starter_kit/" + e["path"], "zenodo_record": kit["id"]}
              for e in kit_files]
    m = rec["metadata"]
    write_manifest(folder, {
        "dataset": "spotgeo", "class": "REAL", "status": "downloaded",
        "title": m.get("title"), "zenodo_record": rec["id"], "doi": rec.get("doi"),
        "licence": (m.get("license") or {}).get("id"),
        "publication_date": m.get("publication_date"),
        "creators": [c["name"] for c in m.get("creators", [])],
        "retrieved_utc": utc_now(),
        "source_urls": [f"https://zenodo.org/records/{rec['id']}"],
        "starter_kit": {"zenodo_record": 3874368, "licence": "cc-by-4.0",
                        "contents": "validation.py (official scorer), metric PDF, notebook"},
        "notes": ["Selected by the COSMOS-AI methodology as the only public, real, labelled "
                  "benchmark of unresolved multi-frame targets (research_design section F).",
                  "Version 2 adds the test annotations used for the final challenge score "
                  "and a few train annotations found missing during the challenge."],
    }, files)
    return folder


#: CelesTrak products used by Feature 2/3.  Space weather is needed for re-entry covariates.
CELESTRAK = {
    "satcat.csv": "https://celestrak.org/pub/satcat.csv",
    "gp-active.csv": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv",
    "gp-last-30-days.csv":
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=last-30-days&FORMAT=csv",
    "gp-cosmos-1408-debris.csv":
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=csv",
    "gp-cosmos-2251-debris.csv":
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=csv",
    "gp-fengyun-1c-debris.csv":
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=csv",
    "gp-iridium-33-debris.csv":
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=csv",
    "SW-All.csv": "https://celestrak.org/SpaceData/SW-All.csv",
}


def fetch_celestrak(date: str | None = None, verbose: bool = True) -> str:
    """A dated CelesTrak snapshot under ``celestrak/snapshots/<date>/``.

    The catalogue changes daily (decays, new launches, re-identifications), so every
    snapshot is kept with its retrieval date and never overwritten.  The files already in
    ``celestrak/`` (downloaded 2026-08) are registered as the earlier snapshot.
    """
    folder = feature_dir("debris", "celestrak")
    date = date or _dt.date.today().isoformat()
    files = []
    for name, url in CELESTRAK.items():
        dst = os.path.join(folder, "snapshots", date, name)
        if os.path.isfile(dst):
            e = {"url": url, "bytes": os.path.getsize(dst), **file_hashes(dst),
                 "status": "cached"}
        else:
            e = fetch_url(url, dst, verbose=verbose)
            time.sleep(2.0)                                   # be polite to CelesTrak
        files.append({"path": f"snapshots/{date}/{name}", **e, "retrieved_utc": utc_now()})
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if name.endswith(".csv") and os.path.isfile(p):
            files.append({"path": name, "bytes": os.path.getsize(p), **file_hashes(p),
                          "status": "pre-existing", "retrieved_utc": "2026-08 (see data/README.md)"})
    write_manifest(folder, {
        "dataset": "celestrak", "class": "REAL", "status": "downloaded",
        "source": "CelesTrak (T.S. Kelso), https://celestrak.org",
        "licence": "CelesTrak usage policy; public, no credentials; no explicit licence",
        "field_definitions": "https://celestrak.org/satcat/satcat-format.php ; "
                             "https://celestrak.org/NORAD/documentation/gp-data-formats.php ; "
                             "https://celestrak.org/SpaceData/SpaceWx-format.php",
        "notes": ["satcat.csv in this folder is the 2026-08 snapshot used by the original "
                  "COSMOS code; dated snapshots live in snapshots/<retrieval date>/.",
                  "gp-*.csv contain only objects still in orbit at retrieval.",
                  "SW-All.csv mixes observed and predicted rows (PRD/PRM): only observed rows "
                  "may be used as features."],
    }, files)
    return folder


def record_unavailable(feature: str, name: str, meta: dict) -> str:
    """Write a manifest for a dataset that could not be obtained, saying exactly why."""
    folder = dataset_root(feature, name)
    write_manifest(folder, {"dataset": name, "status": "not downloaded",
                            "checked_utc": utc_now(), **meta}, [])
    return folder


FETCHERS = {"opssat_ad": fetch_opssat_ad, "esa_adb": fetch_esa_adb, "spotgeo": fetch_spotgeo,
            "celestrak": fetch_celestrak}

#: (feature, folder) of every dataset COSMOS-AI tracks, for :func:`status`
TRACKED = {
    "opssat_ad": ("health", "opssat_ad"), "esa_adb": ("health", "esa_adb"),
    "cosmos_synthetic_telemetry": ("health", "cosmos_synthetic"),
    "spotgeo": ("debris", "spotgeo"), "spark": ("debris", "spark"),
    "spacesense_bench": ("debris", "spacesense_bench"),
    "cosmos_synthetic_vision": ("debris", "cosmos_synthetic"),
    "celestrak": ("debris", "celestrak"),
    "reentry_historical": ("reentry", "historical"),
    "reentry_simulated": ("reentry", "simulated"),
}


def status() -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for name, (feat, sub) in TRACKED.items():
        folder = feature_dir(feat, sub)
        m = read_manifest(folder)
        fs = m.get("files", [])
        for f in fs:                                          # generated entries: size on disk
            if "bytes" not in f and os.path.exists(os.path.join(folder, f["path"])):
                f["bytes"] = os.path.getsize(os.path.join(folder, f["path"]))
        rows.append({"dataset": name, "status": m.get("status", "no manifest"),
                     "class": m.get("class", ""), "files": len(fs),
                     "MB": round(sum(f.get("bytes", 0) for f in fs) / 2**20, 1),
                     "licence": m.get("licence", "")})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "status":
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(status().to_string(index=False))
        return 0
    for name in argv:
        print(f"== {name}")
        print("  ->", FETCHERS[name]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
