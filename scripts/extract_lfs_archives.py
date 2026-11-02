"""Extract the dataset archives carried in Git LFS, without re-downloading them.

    python scripts/extract_lfs_archives.py           # extract what is missing
    python scripts/extract_lfs_archives.py --force   # re-extract even if present
    python scripts/extract_lfs_archives.py --check   # report only, extract nothing

Some dataset archives are committed to this repository through Git LFS, but the tree they
extract to is not: it is byte-for-byte derivable from the archive, so committing both would
spend LFS quota twice for the same bytes.  The loaders read the extracted tree, so a fresh
clone needs this one step:

    git lfs pull && python scripts/extract_lfs_archives.py

Extraction uses ``cosmos.ai.data.download.safe_extract``, the same path the downloader uses,
so the result is identical to a freshly downloaded dataset.  Nothing is fetched from the
network and no manifest is rewritten.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cosmos.ai.data.download import safe_extract  # noqa: E402

# (archive relative to the repository root, destination directory, a path that proves
#  extraction already happened)
ARCHIVES = [
    ("data/feature2_debris/kelvins_cac/raw/Collision Avoidance Challenge - Dataset.zip",
     "data/feature2_debris/kelvins_cac/raw/extracted",
     "data/feature2_debris/kelvins_cac/raw/extracted/Collision Avoidance Challenge - Dataset/"
     "kelvins_competition_data/train_data.csv"),
]

POINTER = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: str) -> bool:
    """True if the file is still an unresolved LFS pointer rather than the real archive."""
    if os.path.getsize(path) > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(len(POINTER)) == POINTER


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-extract archives already extracted")
    ap.add_argument("--check", action="store_true", help="report status without extracting")
    a = ap.parse_args(argv)

    rc = 0
    for rel_archive, rel_dest, rel_witness in ARCHIVES:
        archive = os.path.join(ROOT, rel_archive)
        dest = os.path.join(ROOT, rel_dest)
        witness = os.path.join(ROOT, rel_witness)
        name = os.path.basename(rel_archive)

        if not os.path.isfile(archive):
            print(f"MISSING  {name}: not present — run 'git lfs pull'")
            rc = 1
            continue
        if is_lfs_pointer(archive):
            print(f"POINTER  {name}: only the LFS pointer is here — run 'git lfs pull'")
            rc = 1
            continue
        if os.path.exists(witness) and not a.force:
            print(f"OK       {name}: already extracted to {rel_dest}")
            continue
        if a.check:
            print(f"TODO     {name}: would extract to {rel_dest}")
            rc = 1
            continue

        print(f"EXTRACT  {name} -> {rel_dest}")
        members = safe_extract(archive, dest)
        print(f"         {len(members)} members")
        if not os.path.exists(witness):
            print(f"FAILED   {name}: expected {rel_witness} after extraction")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
