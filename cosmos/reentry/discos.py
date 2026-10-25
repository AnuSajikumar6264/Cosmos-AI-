"""FEATURE 3 (h) — ESA DISCOS: real fragmentation events, re-entries and object properties.

DISCOS (Database and Information System Characterising Objects in Space) is ESA's
reference catalogue of objects, launches, fragmentation events and re-entries, served by
DISCOSweb at https://discosweb.esoc.esa.int.  It needs a personal access token from the
ESA Space Debris User Portal.  The token is read from the ``DISCOS_TOKEN`` environment
variable or from ``~/.config/cosmos/discos_token`` — never from the repository.

Responses follow JSON:API (``data`` / ``attributes`` / ``links.next``).  Every page fetched is
cached under ``data/feature3_reentry/external/discos/``, so after one download the project
runs offline and the analysis is reproducible from the cache alone.

    frag = load_fragmentations()             # one row per fragmentation event
    fragmentation_causes(frag)               # event types mapped onto the coupling's subsystems
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from ..common import feature_dir

BASE_URL = "https://discosweb.esoc.esa.int/api"
API_VERSION = "2"
TOKEN_FILE = os.path.join("~", ".config", "cosmos", "discos_token")
PAGE_SIZE = 100                                # the API's maximum page size

#: how DISCOS fragmentation event types map onto the subsystem whose failure causes them.
#: Matching is by lower-case substring, so it tolerates small differences in wording.
#: ``None`` marks causes that are not a health failure of the spacecraft itself.
CAUSE_TO_SUBSYSTEM = [
    ("propulsion", "Propulsion System"),
    ("battery", "Power System"),
    ("electrical", "Power System"),
    ("aerodynamic", "Structure"),
    ("collision", None),
    ("impact", None),
    ("deliberate", None),
    ("anomal", "unknown"),
    ("accidental", "unknown"),
    ("unknown", "unknown"),
]


def discos_dir() -> str:
    return feature_dir("reentry", "external", "discos")


def read_token() -> str | None:
    tok = os.environ.get("DISCOS_TOKEN")
    if tok:
        return tok.strip()
    path = os.path.expanduser(TOKEN_FILE)
    if os.path.isfile(path):
        with open(path) as f:
            return f.read().strip() or None
    return None


def _get(url: str, token: str, retries: int = 6) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "DiscosWeb-Api-Version": API_VERSION,
                                               "Accept": "application/vnd.api+json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:                          # rate limited: wait as told, retry
                wait = float(e.headers.get("Retry-After", 2 ** attempt * 5))
                time.sleep(min(wait, 120))
                continue
            if e.code == 401:
                raise PermissionError("DISCOSweb rejected the token (401)") from e
            raise
    raise RuntimeError(f"DISCOSweb kept rate-limiting {url}")


def fetch(resource: str, params: dict | None = None, max_pages: int | None = None,
          token: str | None = None, cache: str | None = None, refresh: bool = False) -> list:
    """All records of one resource (``fragmentations``, ``reentries``, ``objects`` ...).

    Returns JSON:API resource objects.  With a cache name the result is stored, and reused
    unless ``refresh`` is set.
    """
    path = os.path.join(discos_dir(), f"{cache or resource}.json")
    if cache is not False and not refresh and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    token = token or read_token()
    if not token:
        raise PermissionError("no DISCOS token: set DISCOS_TOKEN or write it to "
                              f"{TOKEN_FILE} (chmod 600)")
    q = {"page[size]": PAGE_SIZE, "page[number]": 1, **(params or {})}
    url = f"{BASE_URL}/{resource}?{urllib.parse.urlencode(q)}"
    out, pages = [], 0
    while url and (max_pages is None or pages < max_pages):
        page = _get(url, token)
        out.extend(page.get("data", []))
        nxt = (page.get("links") or {}).get("next")
        url = urllib.parse.urljoin(BASE_URL + "/", nxt) if nxt else None
        pages += 1
    if cache is not False:
        with open(path, "w") as f:
            json.dump(out, f)
    return out


def to_frame(records: list) -> pd.DataFrame:
    """Flatten JSON:API resource objects: ``id``, ``type`` and every attribute as a column."""
    return pd.DataFrame([{"id": r.get("id"), "type": r.get("type"), **(r.get("attributes") or {})}
                         for r in records])


def load_fragmentations(refresh: bool = False) -> pd.DataFrame:
    return to_frame(fetch("fragmentations", refresh=refresh))


def load_reentries(refresh: bool = False) -> pd.DataFrame:
    return to_frame(fetch("reentries", refresh=refresh))


def load_objects(object_class: str | None = None, refresh: bool = False) -> pd.DataFrame:
    params = {"filter": f"eq(objectClass,'{object_class}')"} if object_class else None
    name = f"objects_{object_class.replace(' ', '_').lower()}" if object_class else "objects"
    return to_frame(fetch("objects", params, cache=name, refresh=refresh))


def cause_subsystem(event_type) -> str | None:
    """The coupling subsystem behind a DISCOS fragmentation event type (see the table)."""
    t = str(event_type or "").lower()
    for key, sub in CAUSE_TO_SUBSYSTEM:
        if key in t:
            return sub
    return "unknown"


def fragmentation_causes(frag: pd.DataFrame, type_column: str | None = None) -> pd.DataFrame:
    """Share of real fragmentation events per cause, next to the coupling's weight for it.

    This is the check the coupling was missing: its weights (:data:`.hazard.FRAG_WEIGHT`)
    were reasoned from physics; here they are set against what actually broke up.
    """
    from ..health.hazard import FRAG_WEIGHT
    col = type_column or next((c for c in frag.columns if "type" in c.lower() and c != "type"),
                              None)
    if col is None:
        raise KeyError(f"no event-type column among {list(frag.columns)}")
    sub = frag[col].map(cause_subsystem)
    t = (pd.DataFrame({"event_type": frag[col].fillna("(blank)"), "subsystem": sub})
         .groupby(["subsystem", "event_type"], dropna=False).size().rename("events")
         .reset_index())
    t["share"] = (t.events / t.events.sum()).round(4)
    t["coupling_weight"] = t.subsystem.map(FRAG_WEIGHT)
    return t.sort_values("events", ascending=False).reset_index(drop=True)
