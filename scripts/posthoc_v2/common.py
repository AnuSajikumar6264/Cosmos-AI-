"""Shared helpers for the post-hoc V2 analyses (research_design/preregistration/POSTHOC_PLAN_V2.md).

The frozen modules of the pre-registered study are imported, never edited. Outputs go to
outputs/benchmarks/telemetry_esa_adb/posthoc_v2/ (small JSON files and packed alarm arrays);
large intermediate arrays go to /tmp/posthoc_v2/ (RAM-backed, deleted after use).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "outputs", "benchmarks", "telemetry_esa_adb", "posthoc_v2")
# scratch for large intermediate arrays; /tmp is RAM-backed and is cleared on reboot, so a rerun
# can point it elsewhere (POSTHOC_TMP)
TMP = os.environ.get("POSTHOC_TMP", "/tmp/posthoc_v2")
PLAN = os.path.join(ROOT, "research_design", "preregistration", "POSTHOC_PLAN_V2.md")
PLAN_FREEZE = os.path.join(ROOT, "research_design", "preregistration", "posthoc_v2_freeze.json")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_plan() -> dict:
    """Refuse to run if the plan changed after it was frozen."""
    fr = json.load(open(PLAN_FREEZE))
    if sha256(PLAN) != fr["sha256"]:
        raise SystemExit("POSTHOC_PLAN_V2.md changed after its freeze; refusing to run")
    return fr


def provenance(script: str) -> dict:
    fr = check_plan()
    return {"script": os.path.relpath(script, ROOT), "script_sha256": sha256(script),
            "plan_sha256": fr["sha256"], "run_at": datetime.datetime.now().isoformat(timespec="seconds")}


def out_path(*parts) -> str:
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_json(obj, *parts) -> str:
    p = out_path(*parts)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    return p
