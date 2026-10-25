"""Write research_design/preregistration/freeze.json: SHA-256 of every file the confirmatory
analysis depends on.  ``oof_benchmark.test`` refuses to run if any of them has changed.

    python scripts/freeze_prereg.py            # refuses to overwrite an existing freeze
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["research_design/preregistration/PREREG_TELEMETRY_V1.md",
         "scripts/prereg_analysis.py",
         "cosmos/ai/experiment.py", "cosmos/ai/uncertainty.py", "cosmos/ai/evaluation.py",
         "cosmos/ai/data/preprocess_telemetry.py"] + [
    f"cosmos/ai/telemetry/{f}.py" for f in
    ("oof", "event_model", "event_gbt", "oof_benchmark", "event_bootstrap", "alarms",
     "esa_eval", "forecasting", "models")]


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = os.path.join(ROOT, "research_design", "preregistration", "freeze.json")
    if os.path.exists(out) and "--force" not in sys.argv:
        raise SystemExit(f"{out} exists; a new freeze needs a logged deviation and --force")
    files = FILES + sorted(os.path.relpath(p, ROOT) for p in glob.glob(
        os.path.join(ROOT, "cosmos/ai/telemetry/esa_metrics/**/*.py"), recursive=True))
    hashes = {p: sha(p) for p in files}
    fid = hashlib.sha256("".join(hashes[p] for p in sorted(hashes)).encode()).hexdigest()[:12]
    rec = {"freeze_id": fid, "written": dt.datetime.now().isoformat(timespec="seconds"),
           "document": FILES[0], "sha256": hashes}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"freeze {fid}: {len(hashes)} files -> {out}")


if __name__ == "__main__":
    main()
