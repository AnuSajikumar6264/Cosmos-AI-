"""Compile data/DATASET_MANIFEST.csv from every dataset's MANIFEST.json (one row per file).

    python scripts/build_dataset_manifest.py

Each dataset folder written by cosmos.ai.data.download carries a MANIFEST.json with its
source, licence, class (REAL / SYNTHETIC / SIM), retrieval time and per-file checksums.  This
script only collects them; it never downloads or changes data.
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["dataset", "feature_folder", "class", "status", "path", "bytes", "sha256", "md5", "retrieved_utc",
          "url_or_source", "licence", "version_or_record", "notes"]


def main():
    rows = []
    for mf in sorted(glob.glob(os.path.join(ROOT, "data", "*", "*", "MANIFEST.json")) +
                     glob.glob(os.path.join(ROOT, "data", "*", "*", "*", "MANIFEST.json"))):
        m = json.load(open(mf))
        folder = os.path.relpath(os.path.dirname(mf), os.path.join(ROOT, "data"))
        version = m.get("zenodo_record") or m.get("release_on_retrieval") or m.get("version") or m.get("commit") or ""
        base = {"dataset": m.get("dataset", folder), "feature_folder": folder, "class": m.get("class", ""),
                "status": m.get("status", ""), "licence": m.get("licence", ""),
                "version_or_record": json.dumps(version) if isinstance(version, dict) else version,
                "notes": " | ".join(m.get("notes", [])) if isinstance(m.get("notes"), list) else m.get("notes", "")}
        files = m.get("files", [])
        if not files:
            rows.append({**base, "path": "", "url_or_source": m.get("source", "")})
        for f in files:
            rows.append({**base, "path": f.get("path", ""), "bytes": f.get("bytes", ""), "sha256": f.get("sha256", ""),
                         "md5": f.get("md5", ""), "retrieved_utc": f.get("retrieved_utc", m.get("retrieved_utc", "")),
                         "url_or_source": f.get("url", m.get("source", ""))})
    out = os.path.join(ROOT, "data", "DATASET_MANIFEST.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
