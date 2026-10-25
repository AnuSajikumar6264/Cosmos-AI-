"""Telemetry preprocessing for COSMOS-AI: ESA-ADB under the OFFICIAL protocol, and OPSSAT-AD.

ESA-ADB (Kotowski et al., DMLR 2026)
------------------------------------
The original COSMOS loader (:func:`cosmos.health.realdata.prepare_esa_mission`) resamples to
its own 5/2-minute medians and splits at 20 % — fine for the classical monitor, but not
comparable with any published result.  :func:`prepare_esa_official` re-implements the
benchmark's own preprocessing scripts (``notebooks/data-prep/Mission{1,2}_*_prep_from_raw.py``
in kplabs-pl/ESA-ADB, commit 67194d3) so that every COSMOS-AI number is on the official grid:

* zero-order-hold resampling to 30 s (Mission 1) or 18 s (Mission 2), grid aligned to the
  epoch, first grid point initialised with the first sample;
* first differences of the monotonic channels (M1 4-11, M2 29-46), taken on the whole record
  before the split; M2 status strings are factorised, also before the split;
* per-channel labels 1 = anomaly, 2 = rare nominal event, 3 = gap (M1 only), written in
  ``labels.csv`` row order, and annotated samples "restored" when a resampling bin would
  otherwise hide them;
* telecommands of priority >= 3, encoded so that each execution survives resampling as one
  grid sample (vectorised; equivalent to the official peak encoding, see
  :func:`_telecommand_indicator`);
* train = samples up to the split date, test = samples strictly after it, each resampled on
  its own and joined on one grid with forward- then backward-fill.

The official scripts write CSVs of tens of GB; here each split becomes ``.npy`` arrays that
can be memory-mapped.  :func:`official_reference_channel` is a literal port of the official
pandas code for one channel; the unit tests check the vectorised path against it.

OPSSAT-AD (Ruszczak et al., Sci. Data 2025)
-------------------------------------------
:func:`load_opssat_ad` returns the 2,123 labelled fragments with the official ``train`` flag
untouched, plus fixed-length tensors for the deep models (:func:`opssat_tensors`).
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...common import feature_dir

# ============================================================================ protocol
M1_TARGET = ([f"channel_{i}" for i in range(12, 53)] + [f"channel_{i}" for i in range(57, 67)]
             + [f"channel_{i}" for i in range(70, 77)])
M2_TARGET = ([f"channel_{i}" for i in range(9, 29)] + ["channel_58", "channel_59"]
             + [f"channel_{i}" for i in range(70, 92)] + ["channel_96", "channel_97",
                                                          "channel_98"])

#: the official protocol per mission (values read from the benchmark's scripts)
ESA_PROTOCOL = {
    1: dict(rule_s=30, train_end="2007-01-01", val_start="2006-10-01", test_after="2007-01-01",
            monotonic=(4, 11), gap_code_for_other=True, factorize_strings=False,
            restore_codes=(1, 2), tc_min_priority=3, target=M1_TARGET,
            lightweight=[f"channel_{i}" for i in range(41, 47)]),
    2: dict(rule_s=18, train_end="2001-10-01", val_start="2001-07-01", test_after="2001-10-01",
            monotonic=(29, 46), gap_code_for_other=False, factorize_strings=True,
            restore_codes=(1, 2, 3, 4), tc_min_priority=3, target=M2_TARGET,
            lightweight=[f"channel_{i}" for i in range(18, 29)]),
}
assert len(M1_TARGET) == 58 and len(M2_TARGET) == 47

LABEL_NOMINAL, LABEL_ANOMALY, LABEL_RARE, LABEL_GAP = 0, 1, 2, 3
NS = 1_000_000_000


def esa_adb_dir(*parts) -> str:
    return os.path.join(feature_dir("health", "esa_adb"), *parts)


def _raw_zip(mission: int) -> str:
    p = esa_adb_dir("raw", f"ESA-Mission{mission}.zip")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing — run: python -m cosmos.ai.data.download esa_adb")
    return os.path.realpath(p)


def _read_member_pickle(zip_path: str, member: str) -> pd.DataFrame:
    """One channel / telecommand pickle, through the project's allow-list unpickler."""
    from ...health.realdata import safe_read_pickle
    with zipfile.ZipFile(zip_path) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read(member)))
        return safe_read_pickle(inner.read(inner.namelist()[0]))


def _as_ns(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8


def load_labels(mission: int) -> pd.DataFrame:
    """labels.csv joined with anomaly_types.csv, timestamps tz-naive (as the official code)."""
    with zipfile.ZipFile(_raw_zip(mission)) as z:
        base = f"ESA-Mission{mission}"
        lab = pd.read_csv(z.open(f"{base}/labels.csv"))
        types = pd.read_csv(z.open(f"{base}/anomaly_types.csv"))
    for c in ("StartTime", "EndTime"):
        lab[c] = pd.to_datetime(lab[c], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    return lab.merge(types, on="ID", how="left")


def load_channels_table(mission: int) -> pd.DataFrame:
    with zipfile.ZipFile(_raw_zip(mission)) as z:
        return pd.read_csv(z.open(f"ESA-Mission{mission}/channels.csv"))


def load_telecommands_table(mission: int) -> pd.DataFrame:
    with zipfile.ZipFile(_raw_zip(mission)) as z:
        return pd.read_csv(z.open(f"ESA-Mission{mission}/telecommands.csv"))


# ============================================================================ one channel
def _channel_labels(ts: np.ndarray, rows: list) -> np.ndarray:
    """Per-sample label, written row by row in labels.csv order (later rows overwrite)."""
    lab = np.zeros(ts.size, np.uint8)
    for start_ns, end_ns, code in rows:
        i0 = np.searchsorted(ts, start_ns, side="left")
        i1 = np.searchsorted(ts, end_ns, side="right")
        lab[i0:i1] = code
    return lab


def _zoh(ts: np.ndarray, vals: np.ndarray, labs: np.ndarray, rule_ns: int,
         restore_codes=(1, 2)) -> tuple:
    """Zero-order hold onto an epoch-aligned grid, with the official 'restore' step.

    Returns ``(first_grid_ns, values, labels)``.
    """
    first = ts[0] - (ts[0] % rule_ns)                       # Timestamp.floor
    last = ts[-1] + ((-ts[-1]) % rule_ns)                    # Timestamp.ceil
    n = int((last - first) // rule_ns) + 1
    grid = first + np.arange(n, dtype=np.int64) * rule_ns
    idx = np.searchsorted(ts, grid, side="right") - 1       # last sample <= grid point
    idx[0] = 0                                               # iloc[0] = first sample
    idx = np.maximum(idx, 0)
    gv, gl = vals[idx].copy(), labs[idx].copy()

    # restore annotated samples a bin would otherwise hide: bins with >1 sample whose LAST
    # sample is nominal but which contain an annotated one -> grid point (bin + rule) gets the
    # last annotated sample of the bin
    if labs.any():
        b = ts - (ts % rule_ns)
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        ends = np.r_[starts[1:], ts.size]
        ann = np.isin(labs, restore_codes)
        pos = np.where(ann, np.arange(ts.size), -1)
        last_ann = np.maximum.reduceat(pos, starts)
        sel = ((ends - starts) > 1) & (labs[ends - 1] == LABEL_NOMINAL) & (last_ann >= 0)
        g = (b[starts[sel]] + rule_ns - first) // rule_ns
        ok = g < n
        gv[g[ok]] = vals[last_ann[sel][ok]]
        gl[g[ok]] = labs[last_ann[sel][ok]]
    return int(first), gv, gl


def _process_channel(args) -> dict:
    """Worker: one channel pickle -> its train and test grids (official steps, vectorised)."""
    zip_path, mission, name, label_rows, proto = args
    df = _read_member_pickle(zip_path, f"ESA-Mission{mission}/channels/{name}.zip")
    return channel_grids(df, label_rows, proto)


def channel_grids(df: pd.DataFrame, label_rows: list, proto: dict) -> dict:
    """The official per-channel steps on one channel frame, vectorised."""
    name = df.columns[0]
    s = df.iloc[:, 0]
    ts = _as_ns(s.index)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    raw = s.to_numpy()[order]
    is_string = raw.dtype == object or pd.api.types.is_string_dtype(s.dtype)
    k = int(name.split("_")[1])
    lo, hi = proto["monotonic"]
    if lo <= k <= hi:                                        # monotonic: derivative
        v = raw.astype(np.float64)
        vals = np.diff(v, append=v[-1])
    elif is_string and proto["factorize_strings"]:
        vals = pd.factorize(raw)[0].astype(np.float64)
    else:
        vals = raw.astype(np.float64)
    labs = _channel_labels(ts, label_rows)
    rule_ns = proto["rule_s"] * NS
    out = {"name": name, "categorical": bool(is_string)}
    split_ns = pd.Timestamp(proto["train_end"]).value
    test_ns = pd.Timestamp(proto["test_after"]).value
    for part, m in (("train", ts <= split_ns), ("test", ts > test_ns)):
        if m.sum() == 0:
            out[part] = None
            continue
        out[part] = _zoh(ts[m], vals[m].astype(np.float32), labs[m], rule_ns,
                         proto["restore_codes"])
    return out


def _telecommand_indicator(exec_ns: np.ndarray, first: int, n: int, rule_ns: int) -> np.ndarray:
    """1 at grid points g with an execution in (g - rule, g].

    The official ``encode_telecommands`` inserts zeros one rule before / after an execution
    when no other sample is that close, then zero-order-holds.  The held value at g is then 1
    exactly when the latest sample <= g is an execution no more than one rule old, i.e. when
    some execution falls in (g - rule, g].
    """
    out = np.zeros(n, np.uint8)
    if exec_ns.size == 0:
        return out
    g = (exec_ns - first + rule_ns - 1) // rule_ns           # first grid point >= execution
    g = g[(g >= 0) & (g < n)]
    out[g] = 1
    return out


def _process_telecommand(args) -> dict:
    zip_path, mission, name = args
    df = _read_member_pickle(zip_path, f"ESA-Mission{mission}/telecommands/{name}.zip")
    ts = np.unique(_as_ns(df.index))                         # official: drop duplicates
    return {"name": name, "exec_ns": ts}


# ============================================================================ a mission
@dataclass
class ESASplit:
    """One split of one mission on the official grid (arrays may be memory-mapped)."""
    mission: int
    part: str
    t0_ns: int
    rule_s: int
    X: np.ndarray                  # (T, C) float32 channel values
    L: np.ndarray                  # (T, C) uint8 per-channel labels (official codes)
    TC: np.ndarray                 # (T, K) uint8 telecommand executions
    channels: list
    subsystems: list
    categorical: np.ndarray
    target: np.ndarray             # (C,) bool
    lightweight: np.ndarray        # (C,) bool
    telecommands: list

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(pd.Timestamp(self.t0_ns), periods=self.n,
                             freq=pd.Timedelta(seconds=self.rule_s))

    def index_of(self, when) -> int:
        """Grid index of the first grid point >= ``when``."""
        ns = pd.Timestamp(when).value
        return int(np.clip(-(-(ns - self.t0_ns) // (self.rule_s * NS)), 0, self.n))

    def channel_index(self, names) -> np.ndarray:
        pos = {c: i for i, c in enumerate(self.channels)}
        return np.array([pos[c] for c in names])


def processed_dir(mission: int) -> str:
    return esa_adb_dir("processed", f"mission{mission}")


def prepare_esa_official(mission: int, workers: int = 32, verbose: bool = True) -> str:
    """Write the official train/test grids of one mission to ``processed/mission{m}/``."""
    from concurrent.futures import ProcessPoolExecutor
    proto = ESA_PROTOCOL[mission]
    zip_path = _raw_zip(mission)
    chans = load_channels_table(mission)
    labels = load_labels(mission)
    tcs = load_telecommands_table(mission)
    tc_names = sorted(tcs.loc[tcs.Priority >= proto["tc_min_priority"], "Telecommand"])

    def code(cat):
        if cat == "Anomaly":
            return LABEL_ANOMALY
        if cat == "Rare Event":
            return LABEL_RARE
        return LABEL_GAP if proto["gap_code_for_other"] else LABEL_NOMINAL

    rows = {c: [] for c in chans.Channel}
    for r in labels.itertuples():
        if r.Channel in rows:
            rows[r.Channel].append((r.StartTime.value, r.EndTime.value, code(r.Category)))
    jobs = [(zip_path, mission, c, rows[c], proto) for c in chans.Channel]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_process_channel, jobs))
        tres = list(ex.map(_process_telecommand, [(zip_path, mission, t) for t in tc_names]))
    if verbose:
        print(f"  mission {mission}: {len(results)} channels, {len(tres)} telecommands read")

    rule_ns = proto["rule_s"] * NS
    out_dir = processed_dir(mission)
    os.makedirs(out_dir, exist_ok=True)
    split_ns = pd.Timestamp(proto["train_end"]).value
    test_ns = pd.Timestamp(proto["test_after"]).value
    summary = {}
    for part in ("train", "test"):
        segs = [r[part] for r in results]
        firsts, lasts = [], []
        for s in segs:
            if s is not None:
                firsts.append(s[0])
                lasts.append(s[0] + (len(s[1]) - 1) * rule_ns)
        # telecommands count towards the time range, as in find_full_time_range
        for t in tres:
            e = t["exec_ns"]
            e = e[e <= split_ns] if part == "train" else e[e > test_ns]
            if e.size:
                firsts.append((e[0] - rule_ns) - ((e[0] - rule_ns) % rule_ns))
                lasts.append((e[-1] + rule_ns) + ((-(e[-1] + rule_ns)) % rule_ns))
        t0, t1 = min(firsts), max(lasts)
        n = int((t1 - t0) // rule_ns) + 1
        C = len(results)
        X = np.lib.format.open_memmap(os.path.join(out_dir, f"{part}_X.npy"), "w+",
                                      np.float32, (n, C))
        L = np.lib.format.open_memmap(os.path.join(out_dir, f"{part}_L.npy"), "w+",
                                      np.uint8, (n, C))
        for j, s in enumerate(segs):
            if s is None:
                X[:, j] = 0.0
                L[:, j] = 0
                continue
            first, gv, gl = s
            a = int((first - t0) // rule_ns)
            b = a + len(gv)
            X[a:b, j] = gv
            L[a:b, j] = gl
            X[:a, j], L[:a, j] = gv[0], gl[0]                # bfill
            X[b:, j], L[b:, j] = gv[-1], gl[-1]              # ffill
        X.flush()
        L.flush()
        TC = np.zeros((n, len(tres)), np.uint8)
        for k, t in enumerate(tres):
            e = t["exec_ns"]
            e = e[e <= split_ns] if part == "train" else e[e > test_ns]
            TC[:, k] = _telecommand_indicator(e, t0, n, rule_ns)
        np.save(os.path.join(out_dir, f"{part}_TC.npy"), TC)
        summary[part] = {"t0": str(pd.Timestamp(t0)), "t_end": str(pd.Timestamp(t1)), "n": n,
                         "nan_values": int(np.isnan(X).sum()),
                         "labelled_fraction": float((L[:, :] > 0).mean()),
                         "telecommand_executions": int(TC.sum())}
        del X, L
    meta = {"mission": mission, "protocol": {k: v for k, v in proto.items()
                                             if k not in ("target", "lightweight")},
            "rule_s": proto["rule_s"],
            "channels": [r["name"] for r in results],
            "subsystems": chans.Subsystem.tolist(),
            "groups": chans.Group.tolist(),
            "categorical": [r["categorical"] for r in results],
            "target": [c in proto["target"] for c in chans.Channel],
            "lightweight": [c in proto["lightweight"] for c in chans.Channel],
            "telecommands": tc_names, "splits": summary,
            "t0_ns": {p: int(pd.Timestamp(summary[p]["t0"]).value) for p in summary},
            "source": "kplabs-pl/ESA-ADB official prep scripts @67194d3, re-implemented "
                      "(cosmos.ai.data.preprocess_telemetry)"}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    labels.to_csv(os.path.join(out_dir, "labels_with_types.csv"), index=False)
    if verbose:
        for p, s in summary.items():
            print(f"  {p}: {s['n']:,} x {len(results)} from {s['t0']} to {s['t_end']}")
    return out_dir


def load_esa_split(mission: int, part: str = "test", mmap: bool = True) -> ESASplit:
    d = processed_dir(mission)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    mode = "r" if mmap else None
    return ESASplit(mission, part, meta["t0_ns"][part], meta["rule_s"],
                    np.load(os.path.join(d, f"{part}_X.npy"), mmap_mode=mode),
                    np.load(os.path.join(d, f"{part}_L.npy"), mmap_mode=mode),
                    np.load(os.path.join(d, f"{part}_TC.npy")),
                    meta["channels"], meta["subsystems"], np.array(meta["categorical"]),
                    np.array(meta["target"]), np.array(meta["lightweight"]),
                    meta["telecommands"])


def official_test_labels(mission: int, split: ESASplit | None = None) -> pd.DataFrame:
    """Labels exactly as TimeEval passes them to the metrics: restricted to the test range."""
    split = split or load_esa_split(mission, "test")
    lab = pd.read_csv(os.path.join(processed_dir(mission), "labels_with_types.csv"),
                      parse_dates=["StartTime", "EndTime"])
    ts = split.timestamps()
    lab = lab[(lab.StartTime >= ts[0]) & (lab.EndTime <= ts[-1])]
    return lab[["ID", "Channel", "StartTime", "EndTime", "Category", "Dimensionality",
                "Locality", "Length"]].reset_index(drop=True)


# ============================================================================ reference
def official_reference_channel(df: pd.DataFrame, labels: pd.DataFrame, mission: int,
                               part: str = "train") -> pd.DataFrame:
    """Literal port of the official per-channel pandas code (slow; for verification only)."""
    proto = ESA_PROTOCOL[mission]
    rule = pd.Timedelta(seconds=proto["rule_s"])
    param = df.columns[0]
    param_df = df.copy()
    param_df["label"] = np.uint8(0)
    param_df = param_df.rename(columns={param: "value"})
    lo, hi = proto["monotonic"]
    if lo <= int(param.split("_")[1]) <= hi:
        param_df.value = np.diff(param_df.value, append=param_df.value.iloc[-1])
    if proto["factorize_strings"] and param_df["value"].dtype == "O":
        param_df["value"] = pd.factorize(param_df["value"])[0]
    annotated = False
    for _, row in labels.iterrows():
        if row["Channel"] == param:
            cat = row["Category"]
            v = (1 if cat == "Anomaly" else 2 if cat == "Rare Event" else
                 3 if proto["gap_code_for_other"] else 0)
            param_df.loc[row["StartTime"]:row["EndTime"], "label"] = v
            annotated = True
    if part == "train":
        param_df = param_df[param_df.index <= pd.Timestamp(proto["train_end"])].copy()
    else:
        param_df = param_df[param_df.index > pd.Timestamp(proto["test_after"])].copy()
    first = pd.Timestamp(param_df.index[0]).floor(freq=rule)
    last = pd.Timestamp(param_df.index[-1]).ceil(freq=rule)
    out = param_df.reindex(pd.date_range(first, last, freq=rule), method="ffill")
    out.iloc[0] = param_df.iloc[0]
    if annotated:
        grouper = param_df.groupby(pd.Grouper(freq=rule))
        for timestamp, group in grouper.indices.items():
            if len(group) <= 1:
                continue
            org = param_df.iloc[group]
            if org.label.values[-1] != 0:
                continue
            is_ann = org.label.isin(proto["restore_codes"])
            if is_ann.any():
                out.loc[timestamp + rule] = org[is_ann].iloc[-1]
    return out


# ============================================================================ OPSSAT-AD
OPSSAT_FEATURES = ["mean", "var", "std", "len", "duration", "len_weighted", "gaps_squared",
                   "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration",
                   "var_div_len", "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
                   "kurtosis", "skew"]
OPSSAT_SEED = 2137                     # the official modelling notebook's seed


def opssat_dir(*parts) -> str:
    return os.path.join(feature_dir("health", "opssat_ad"), *parts)


def load_opssat_ad() -> dict:
    """Features (official) and raw segments, keyed by segment id, official split untouched."""
    feats = pd.read_csv(opssat_dir("raw", "dataset.csv"), index_col="segment")
    segs = pd.read_csv(opssat_dir("raw", "segments.csv"), parse_dates=["timestamp"])
    return {"features": feats, "segments": segs, "feature_names": OPSSAT_FEATURES}


def opssat_tensors(segs: pd.DataFrame, length: int = 256) -> tuple:
    """Fixed-length inputs for the deep models: ``(X, mask, seg_ids)``.

    Each fragment is standardised by its OWN mean and std (no information crosses fragments,
    so nothing leaks between train and test) and resampled to ``length`` points by linear
    interpolation over its sample index.  Channel 1 is the standardised value, channel 2 the
    log time gap to the previous sample (fragments are irregularly sampled), channel 3 the
    raw value scaled by the fragment's own max |value| (keeps level information the z-score
    removes).  ``mask`` marks real (not padded) positions; with interpolation it is all ones,
    kept so the same models can run on padded inputs.
    """
    out, ids = [], []
    for sid, g in segs.groupby("segment", sort=True):
        v = g.value.to_numpy(np.float64)
        t = g.timestamp.astype("int64").to_numpy() / 1e9
        dt = np.r_[0.0, np.diff(t)]
        z = (v - v.mean()) / (v.std() + 1e-12)
        sc = v / (np.abs(v).max() + 1e-12)
        src = np.linspace(0.0, 1.0, v.size)
        dst = np.linspace(0.0, 1.0, length)
        out.append(np.stack([np.interp(dst, src, z), np.interp(dst, src, np.log1p(dt)),
                             np.interp(dst, src, sc)]))
        ids.append(sid)
    X = np.stack(out).astype(np.float32)
    return X, np.ones((X.shape[0], length), np.float32), np.array(ids)


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    for m in [int(a) for a in argv] or [1, 2]:
        print(f"== ESA-ADB mission {m} (official protocol)")
        prepare_esa_official(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
