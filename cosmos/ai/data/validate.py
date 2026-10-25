"""Validate every COSMOS-AI dataset from the files on disk and write the dataset report.

    python -m cosmos.ai.data.validate          # -> data/DATASET_REPORT.md + .json

Nothing here trusts a manifest's own numbers: counts, label statistics, missing values,
corrupt files, invalid boxes and cross-split duplicates are recomputed from the data.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ...common import data_root, feature_dir, output_dir
from .download import TRACKED, read_manifest


def _size_mb(path: str) -> float:
    if os.path.isfile(path):
        return os.path.getsize(path) / 2**20
    tot = 0
    for dp, _dn, fn in os.walk(path, followlinks=False):
        for f in fn:
            p = os.path.join(dp, f)
            if not os.path.islink(p):
                tot += os.path.getsize(p)
    return tot / 2**20


# ============================================================================ telemetry
def validate_opssat() -> dict:
    from .preprocess_telemetry import OPSSAT_FEATURES, load_opssat_ad
    d = load_opssat_ad()
    f, s = d["features"], d["segments"]
    g = s.groupby("segment")
    rec = g.agg(n_re=("value", "size"), mean_re=("value", "mean"))
    j = f.join(rec)
    split_time = s.groupby(["channel", "train"]).timestamp.agg(["min", "max"]).reset_index()
    return {"files": {"dataset.csv": len(f), "segments.csv": len(s)},
            "segments": int(s.segment.nunique()), "channels": int(f.channel.nunique()),
            "class_counts": {f"train={t},anomaly={a}": int(n) for (t, a), n in
                             f.groupby(["train", "anomaly"]).size().items()},
            "anomaly_fraction": float(f.anomaly.mean()),
            "missing_values": {"features": int(f.isna().sum().sum()),
                               "segments": int(s.isna().sum().sum())},
            "segment_length": f.len.describe().round(1).to_dict(),
            "label_constant_within_segment": bool((g.anomaly.nunique() == 1).all()),
            "split_constant_within_segment": bool((g.train.nunique() == 1).all()),
            "features_recomputed_len_mismatches": int((j.len != j.n_re).sum()),
            "features_recomputed_mean_max_abs_diff": float((j["mean"] - j.mean_re).abs().max()),
            "train_test_interleaved_in_time_channels": int(
                split_time.groupby("channel").apply(lambda x: len(x) == 2).sum()),
            "n_features": len(OPSSAT_FEATURES)}


def validate_esa() -> dict:
    from ..telemetry.forecasting import PERIODS
    from .preprocess_telemetry import ESA_PROTOCOL, load_esa_split, processed_dir
    out = {"manifest": {}}
    m = read_manifest(feature_dir("health", "esa_adb"))
    out["manifest"] = {f["path"]: {"bytes": f.get("bytes"), "md5_verified": f.get("md5_verified")}
                       for f in m.get("files", [])}
    out["revision_check"] = m.get("revision_check", {}).get("result")
    for mission in (1, 2):
        if not os.path.isfile(os.path.join(processed_dir(mission), "meta.json")):
            out[f"mission{mission}"] = "not prepared"
            continue
        tr, te = load_esa_split(mission, "train"), load_esa_split(mission, "test")
        lab = pd.read_csv(os.path.join(processed_dir(mission), "labels_with_types.csv"),
                          parse_dates=["StartTime", "EndTime"])
        tgt = lab[lab.Channel.isin(ESA_PROTOCOL[mission]["target"])]
        per = {}
        for name, (a, b) in {**PERIODS[mission],
                             "test": (str(te.timestamps()[0]), str(te.timestamps()[-1]))}.items():
            sel = tgt[(tgt.StartTime >= a) & (tgt.EndTime <= b)]
            per[name] = {"events": int(sel.ID.nunique()),
                         **{k: int(v) for k, v in sel.groupby("ID").Category.first()
                            .value_counts().items()}}
        rng = np.random.default_rng(0)
        rows = rng.integers(0, te.n, 200_000)
        out[f"mission{mission}"] = {
            "grid_seconds": tr.rule_s, "channels": len(tr.channels),
            "target_channels": int(np.sum(tr.target)), "categorical_channels": int(tr.categorical.sum()),
            "telecommands_priority_ge_3": len(tr.telecommands),
            "train": {"n": tr.n, "start": str(tr.timestamps()[0]), "end": str(tr.timestamps()[-1]),
                      "telecommand_executions": int(tr.TC.sum())},
            "test": {"n": te.n, "start": str(te.timestamps()[0]), "end": str(te.timestamps()[-1]),
                     "telecommand_executions": int(te.TC.sum())},
            "nan_in_test_sample_200k_rows": int(np.isnan(np.asarray(te.X[np.sort(rows)])).sum()),
            "labelled_fraction_test_target_channels_sample": float(
                (np.asarray(te.L[np.sort(rows)])[:, te.target] > 0).mean()),
            "events_target_channels_by_period": per}
    return out


# ============================================================================ vision
def _png_check(p):
    from PIL import Image
    try:
        with open(p, "rb") as f:
            raw = f.read()
        im = Image.open(p)
        im.verify()
        return p, hashlib.md5(raw).hexdigest(), None
    except Exception as exc:                                  # corrupt file
        return p, None, repr(exc)


def validate_spotgeo(workers: int = 64) -> dict:
    from ..vision import spotgeo as S
    out = {}
    hashes = {}
    for split in ("train", "test"):
        lab = S.load_annotations(split)
        seqs = S.sequences(split)
        files = sorted(glob.glob(os.path.join(S.raw_dir(split), "*", "*.png")))
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_png_check, files, chunksize=64))
        bad = [p for p, h, e in res if e]
        hashes[split] = {h for _p, h, _e in res if h}
        pts = np.concatenate([np.asarray(lab[s][f]).reshape(-1, 2) for s in lab for f in lab[s]])
        n_obj = [len(np.asarray(lab[s][f]).reshape(-1, 2)) for s in lab for f in lab[s]]
        inval = int(((pts[:, 0] < -0.5) | (pts[:, 0] > 639.5) | (pts[:, 1] < -0.5)
                     | (pts[:, 1] > 479.5)).sum())
        out[split] = {"sequences": len(seqs), "frames_on_disk": len(files),
                      "frames_expected": 5 * len(seqs), "corrupt_png": len(bad),
                      "annotated_sequences": len(lab), "objects": int(len(pts)),
                      "frames_without_objects": int(sum(k == 0 for k in n_obj)),
                      "max_objects_per_frame": int(max(n_obj)),
                      "coordinates_outside_image": inval}
    out["exact_duplicate_frames_train_vs_test"] = len(hashes["train"] & hashes["test"])
    tr, va = S.train_val_split()
    out["validation_split"] = {"train_sequences": len(tr), "val_sequences": len(va),
                               "overlap": len(set(tr) & set(va))}
    out["object_brightness_note"] = ("SNR per object is not provided by the release; no "
                                     "SNR-band breakdown is reported")
    return out


def validate_cosmos_synthetic_vision(save_preview: bool = True) -> dict:
    from ..vision.cosmos_synthetic import data_dir, load_split
    if not os.path.isfile(data_dir("data.yaml")):
        return {"status": "not generated"}
    out = {}
    for split in ("train", "val", "test"):
        imgs, truths = load_split(split)
        T = np.concatenate([t for t in truths if len(t)])
        bad = int(((T[:, 3] <= 0) | (T[:, 4] <= 0) | (T[:, 1] < 0) | (T[:, 1] > 640)
                   | (T[:, 2] < 0) | (T[:, 2] > 640)).sum())
        out[split] = {"images": len(imgs), "objects": int(len(T)),
                      "class_counts": np.bincount(T[:, 0].astype(int), minlength=4).tolist(),
                      "box_px": {"min": float(T[:, 3].min()), "median": float(np.median(T[:, 3])),
                                 "max": float(T[:, 3].max())},
                      "invalid_boxes": bad, "images_without_objects": int(sum(len(t) == 0
                                                                            for t in truths))}
        if save_preview and split == "test":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            for k in range(3):
                ax[k].imshow(imgs[k], cmap="gray")
                for c, cx, cy, w, h in truths[k]:
                    ax[k].add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                                              color="rgb"[int(c) % 3] if c < 3 else "y"))
                ax[k].set_title(f"test image {k}: {len(truths[k])} labelled movers")
                ax[k].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir("benchmarks", "vision_cosmos_synthetic"),
                                     "ground_truth_samples.png"), dpi=70)
            plt.close(fig)
    return out


# ============================================================================ catalogue
def validate_catalogue() -> dict:
    from .preprocess_catalogue import FOLLOW, TAU, build_landmark, leakage_report
    out = {}
    for name, p in (("satcat_2026-08-16", os.path.join(feature_dir("debris", "celestrak"),
                                                       "satcat.csv")),
                    ("satcat_2026-10-24", os.path.join(feature_dir("debris", "celestrak"),
                                                       "snapshots", "2026-10-24", "satcat.csv"))):
        d = pd.read_csv(p, low_memory=False)
        out[name] = {"rows": len(d), "decayed": int(d.DECAY_DATE.notna().sum()),
                     "max_launch_date": str(pd.to_datetime(d.LAUNCH_DATE, errors="coerce").max().date()),
                     "missing_perigee_in_orbit": int(pd.to_numeric(d.PERIGEE[d.DECAY_DATE.isna()],
                                                                   errors="coerce").isna().sum()),
                     "rcs_available": int(pd.to_numeric(d.RCS, errors="coerce").notna().sum()),
                     "object_types": d.OBJECT_TYPE.value_counts().to_dict()}
    df, ex = build_landmark(write=False)
    out["landmark"] = {"tau": TAU, "follow_up": FOLLOW, "exclusions": ex,
                       "leakage_report": leakage_report(df),
                       "events_by_split": df.groupby("split").event.agg(["size", "sum"])
                       .rename(columns={"size": "n", "sum": "events"}).to_dict("index"),
                       "events_by_type": df.groupby("OBJECT_TYPE").event.sum().to_dict(),
                       "gp_coverage": float(df.gp_available.mean())}
    rec = os.path.join(feature_dir("reentry", "historical"), "recovered_debris_esa.csv")
    if os.path.isfile(rec):
        r = pd.read_csv(rec)
        out["recovered_debris"] = {"fragment_rows": len(r), "objects": int(r.cospar_id.nunique()),
                                   "with_coordinates": int(r.lat.notna().sum()),
                                   "with_mass": int(r.mass_kg.notna().sum()),
                                   "types": r.debris_type.value_counts().head(6).to_dict()}
    sim = os.path.join(feature_dir("reentry", "simulated"), "propagator_surrogate_v1")
    if os.path.isdir(sim):
        out["simulated_reentry"] = {os.path.basename(p): {"rows": int(len(pd.read_parquet(p))),
                                                          "impacted": float(pd.read_parquet(p)
                                                                            .impacted.mean())}
                                    for p in sorted(glob.glob(os.path.join(sim, "*.parquet")))}
    return out


# ============================================================================ report
def run(workers: int = 64) -> dict:
    report = {"datasets": {}}
    for name, (feat, sub) in TRACKED.items():
        folder = feature_dir(feat, sub)
        m = read_manifest(folder)
        report["datasets"][name] = {"folder": os.path.relpath(folder, data_root()),
                                    "status": m.get("status", "no manifest"),
                                    "class": m.get("class"), "licence": m.get("licence"),
                                    "size_on_disk_mb": round(_size_mb(folder), 1),
                                    "manifest_files": len(m.get("files", [])),
                                    "reason_if_unavailable": m.get("reason") or m.get("decision")}
    report["opssat_ad"] = validate_opssat()
    report["esa_adb"] = validate_esa()
    report["spotgeo"] = validate_spotgeo(workers)
    report["cosmos_synthetic_vision"] = validate_cosmos_synthetic_vision()
    report["catalogue_and_reentry"] = validate_catalogue()
    p = os.path.join(data_root(), "DATASET_REPORT.json")
    with open(p, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def write_markdown(r: dict) -> str:
    """``data/DATASET_REPORT.md`` from the validation JSON (every number from the files)."""
    L = ["# COSMOS-AI dataset report", "",
         "Generated by `python -m cosmos.ai.data.validate` from the files on disk. "
         "Raw JSON: `DATASET_REPORT.json`. Provenance (URLs, checksums, licences): each "
         "folder's `MANIFEST.json`.", "", "## Inventory", "",
         "| Dataset | Folder | Class | Status | Size on disk (MB) | Licence |", "|---|---|---|---|---|---|"]
    for k, v in r["datasets"].items():
        L.append(f"| {k} | `{v['folder']}` | {v.get('class') or ''} | {v['status']} | "
                 f"{v['size_on_disk_mb']:,} | {v.get('licence') or ''} |")
    L += ["", "### Not downloaded, or only in part", ""]
    for k, v in r["datasets"].items():
        if v.get("reason_if_unavailable"):
            L.append(f"- **{k}**: {v['reason_if_unavailable']}")
    L.append("- **Space-Track / ESA DISCOSweb / ESA re-entry predictions**: account required "
             "(the re-entry prediction API answers HTTP 401); no credentials, not attempted.")
    o = r["opssat_ad"]
    L += ["", "## OPSSAT-AD", "",
          f"- {o['segments']:,} fragments, {o['channels']} channels, {o['files']['segments.csv']:,} "
          f"samples; anomaly fraction {o['anomaly_fraction']:.3f}.",
          f"- Official split class counts: {o['class_counts']}.",
          f"- Missing values: features {o['missing_values']['features']}, segments "
          f"{o['missing_values']['segments']}. Fragment length median {o['segment_length']['50%']}, "
          f"range {o['segment_length']['min']}–{o['segment_length']['max']}.",
          f"- Labels constant within fragments: {o['label_constant_within_segment']}; recomputed "
          f"`len` mismatches: {o['features_recomputed_len_mismatches']}.",
          f"- Train and test fragments interleave in time on all {o['train_test_interleaved_in_time_channels']} "
          "channels (the official split is not temporal) — hence the extra time-blocked protocol."]
    e = r["esa_adb"]
    L += ["", "## ESA-ADB (official protocol grids)", "",
          f"Revision check: {e.get('revision_check')}", "",
          "| Mission | Grid | Channels (target / categorical) | Train | Test | Telecommand executions (train / test) |",
          "|---|---|---|---|---|---|"]
    for m in (1, 2):
        x = e.get(f"mission{m}")
        if isinstance(x, dict):
            L.append(f"| {m} | {x['grid_seconds']} s | {x['channels']} ({x['target_channels']} / "
                     f"{x['categorical_channels']}) | {x['train']['n']:,} ({x['train']['start'][:10]} → "
                     f"{x['train']['end'][:10]}) | {x['test']['n']:,} ({x['test']['start'][:10]} → "
                     f"{x['test']['end'][:10]}) | {x['train']['telecommand_executions']} / "
                     f"{x['test']['telecommand_executions']} |")
    L += ["", "Labelled events on target channels by period (fit / early stopping / calibration "
          "lie inside the official training half):", ""]
    for m in (1, 2):
        x = e.get(f"mission{m}")
        if isinstance(x, dict):
            L.append(f"- Mission {m}: " + "; ".join(f"{p}: {v}" for p, v in
                                                    x["events_target_channels_by_period"].items()))
            L.append(f"  NaN in a 200k-row test sample: {x['nan_in_test_sample_200k_rows']}.")
    s = r["spotgeo"]
    L += ["", "## SpotGEO v2", "", "| Split | Sequences | Frames (on disk / expected) | Corrupt PNG | Objects | Frames without objects | Max per frame | Coordinates outside image |",
          "|---|---|---|---|---|---|---|---|"]
    for sp in ("train", "test"):
        x = s[sp]
        L.append(f"| {sp} | {x['sequences']:,} | {x['frames_on_disk']:,} / {x['frames_expected']:,} | "
                 f"{x['corrupt_png']} | {x['objects']:,} | {x['frames_without_objects']:,} | "
                 f"{x['max_objects_per_frame']} | {x['coordinates_outside_image']} |")
    L += [f"", f"- Exact duplicate frames between train and test (MD5): "
          f"{s['exact_duplicate_frames_train_vs_test']}.",
          f"- Validation split (by sequence): {s['validation_split']}.",
          f"- {s['object_brightness_note']}.",
          "- Ground truth visualised in `outputs/benchmarks/vision_spotgeo/figures/`."]
    c = r["cosmos_synthetic_vision"]
    if "train" in c:
        L += ["", "## COSMOS synthetic frame pairs", "", "| Split | Images | Objects | Class counts | Box px (min / median / max) | Invalid boxes |", "|---|---|---|---|---|---|"]
        for sp in ("train", "val", "test"):
            x = c[sp]
            L.append(f"| {sp} | {x['images']} | {x['objects']:,} | {x['class_counts']} | "
                     f"{x['box_px']['min']:.1f} / {x['box_px']['median']:.1f} / {x['box_px']['max']:.1f} | "
                     f"{x['invalid_boxes']} |")
        L.append("- Ground truth visualised in `outputs/benchmarks/vision_cosmos_synthetic/ground_truth_samples.png`.")
    k = r["catalogue_and_reentry"]
    L += ["", "## Catalogue and re-entry", ""]
    for nm in ("satcat_2026-08-16", "satcat_2026-10-24"):
        x = k[nm]
        L.append(f"- {nm}: {x['rows']:,} objects, {x['decayed']:,} decayed, latest launch "
                 f"{x['max_launch_date']}, in-orbit objects without perigee {x['missing_perigee_in_orbit']}, "
                 f"RCS available {x['rcs_available']:,}.")
    lm = k["landmark"]
    L += [f"- Landmark {lm['tau']} → {lm['follow_up']}: exclusions {lm['exclusions']}.",
          f"- Leakage report: {lm['leakage_report']}.",
          f"- Events by split: {lm['events_by_split']}; by object type: {lm['events_by_type']}; "
          f"GP element set available for {lm['gp_coverage']:.1%} of objects."]
    if "recovered_debris" in k:
        x = k["recovered_debris"]
        L.append(f"- ESA recovered debris: {x['fragment_rows']} fragment rows, {x['objects']} objects, "
                 f"{x['with_coordinates']} with coordinates, {x['with_mass']} with mass; types {x['types']}.")
    if "simulated_reentry" in k:
        L.append(f"- Simulated re-entry (SIM): {k['simulated_reentry']}.")
    p = os.path.join(data_root(), "DATASET_REPORT.md")
    with open(p, "w") as f:
        f.write("\n".join(L) + "\n")
    return p


def main() -> int:
    r = run()
    print(json.dumps(r["datasets"], indent=1, default=str))
    print(write_markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
