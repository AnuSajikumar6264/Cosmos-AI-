"""FEATURE 2 (e) — real space-object imagery (SPARK, DUST) for the learned detector.

The frame-pair simulator answers the question this project needs (unresolved movers in a
star field); public space-object image sets answer a neighbouring one — resolved spacecraft
and debris, one object per image, with a box and a class.  Training and scoring the same
detector on them tests the one claim simulated frames cannot: that it works on pixels from
a real camera model.

Access to SPARK and DUST is by request, so the archives are not downloaded by this project.
Place them, as received, under ``data/feature2_debris/external/<name>/`` and run
:func:`convert_to_yolo`.  Three label layouts are recognised:

``csv``    one row per object: an image column, a class column and a box, either one
           ``bbox`` string ``[a, b, c, d]`` or four numeric columns (the SPARK layout).
``coco``   a COCO ``instances`` JSON (``images`` / ``annotations`` / ``categories``).
``yolo``   ``images/*.png|jpg`` with ``labels/*.txt`` already in YOLO form.

SPARK writes boxes as ``[row_min, col_min, row_max, col_max]``; pass ``bbox_order`` if a
release differs, and use :func:`preview_boxes` to confirm visually before training.

    info = convert_to_yolo("spark")              # -> data/feature2_debris/vision/spark_yolo
    model = TinyYOLO(class_names=info["classes"]).fit(info["path"], epochs=30)
    model.evaluate(info["path"], match="iou")
"""
from __future__ import annotations

import ast
import glob
import json
import os

import numpy as np
import pandas as pd

from ..common import feature_dir

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_FILE_COLS = ("filename", "image", "file", "img", "image_name", "image_id")
_CLASS_COLS = ("class", "label", "category", "class_name")
_BOX_COLS = ("bbox", "box", "bounding_box")
_FOUR = [("x_min", "y_min", "x_max", "y_max"), ("xmin", "ymin", "xmax", "ymax"),
         ("c_min", "r_min", "c_max", "r_max")]


def external_dir(name: str) -> str:
    return feature_dir("debris", "external", name)


def _find_images(root: str) -> dict:
    """Map every image's base name (with and without extension) to its path."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(IMAGE_EXT):
            b = os.path.basename(p)
            out.setdefault(b, p)
            out.setdefault(os.path.splitext(b)[0], p)
    return out


def _col(df: pd.DataFrame, names) -> str | None:
    low = {c.lower(): c for c in df.columns}
    return next((low[n] for n in names if n in low), None)


def read_csv_labels(csv_path: str, bbox_order: str = "rc") -> pd.DataFrame:
    """SPARK-style CSV -> ``file, cls, x0, y0, x1, y1`` in pixels.

    ``bbox_order="rc"`` reads ``[row_min, col_min, row_max, col_max]`` (SPARK);
    ``"xy"`` reads ``[x_min, y_min, x_max, y_max]``.
    """
    df = pd.read_csv(csv_path)
    fcol, ccol, bcol = _col(df, _FILE_COLS), _col(df, _CLASS_COLS), _col(df, _BOX_COLS)
    if fcol is None or ccol is None:
        raise KeyError(f"{csv_path}: need an image column and a class column, got "
                       f"{list(df.columns)}")
    if bcol is not None:
        b = np.array([ast.literal_eval(str(v)) for v in df[bcol]], float)
        if bbox_order == "rc":
            x0, y0, x1, y1 = b[:, 1], b[:, 0], b[:, 3], b[:, 2]
        else:
            x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    else:
        # four named columns say which axis they are, so no order needs guessing
        four = next((f for f in _FOUR if all(_col(df, [k]) for k in f)), None)
        if four is None:
            raise KeyError(f"{csv_path}: no bbox column found in {list(df.columns)}")
        x0, y0, x1, y1 = (df[_col(df, [k])].to_numpy(float) for k in four)
    return pd.DataFrame({"file": df[fcol].astype(str), "cls": df[ccol].astype(str),
                         "x0": x0, "y0": y0, "x1": x1, "y1": y1})


def read_coco_labels(json_path: str) -> pd.DataFrame:
    """COCO ``instances`` JSON -> ``file, cls, x0, y0, x1, y1`` in pixels."""
    with open(json_path) as f:
        coco = json.load(f)
    imgs = {i["id"]: i["file_name"] for i in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    rows = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        rows.append({"file": imgs[a["image_id"]], "cls": cats[a["category_id"]],
                     "x0": x, "y0": y, "x1": x + w, "y1": y + h})
    return pd.DataFrame(rows)


def detect_layout(root: str) -> tuple:
    """Find the label files under ``root``: ``(layout, [paths])``."""
    js = [p for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
          if "annotation" in os.path.basename(p).lower() or "instances" in p.lower()]
    if js:
        return "coco", js
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if csvs:
        return "csv", csvs
    if glob.glob(os.path.join(root, "**", "labels", "*.txt"), recursive=True):
        return "yolo", []
    raise FileNotFoundError(f"no labels (COCO json, CSV or YOLO txt) found under {root}")


def _split_of(path: str) -> str:
    p = path.lower()
    return "val" if ("val" in p or "valid" in p) else ("test" if "test" in p else "train")


def convert_to_yolo(name: str, root: str | None = None, out: str | None = None,
                    size: int = 640, max_per_split: int | None = None, bbox_order: str = "rc",
                    val_frac: float = 0.2, seed: int = 0) -> dict:
    """Convert a real image set into the YOLO layout the detectors train on.

    Images are converted to grayscale (the detector is single-channel) and resized to
    ``size``; labels become normalised ``class cx cy w h``.  If the source has no validation
    split, ``val_frac`` of the training images is held out.
    """
    from PIL import Image
    root = root or external_dir(name)
    out = out or feature_dir("debris", "vision", f"{name}_yolo")
    layout, files = detect_layout(root)
    if layout == "yolo":
        return {"path": root, "layout": "yolo", "classes": None}
    frames = []
    for f in files:
        df = read_coco_labels(f) if layout == "coco" else read_csv_labels(f, bbox_order)
        frames.append(df.assign(split=_split_of(f)))
    lab = pd.concat(frames, ignore_index=True)
    if (lab.split == "test").all():
        lab["split"] = "train"          # a single unlabelled-split file is the training set
    else:
        lab = lab[lab.split != "test"]  # test labels are usually withheld by the challenge
    classes = sorted(lab.cls.unique())
    cid = {c: i for i, c in enumerate(classes)}
    if (lab.split == "val").sum() == 0:
        rng = np.random.default_rng(seed)
        files_u = lab.file.unique()
        held = set(rng.choice(files_u, int(val_frac * len(files_u)), replace=False))
        lab.loc[lab.file.isin(held), "split"] = "val"
    images = _find_images(root)
    counts = {}
    for split, g in lab.groupby("split"):
        os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)
        names = g.file.unique()[:max_per_split] if max_per_split else g.file.unique()
        n = 0
        for fn in names:
            src = images.get(fn) or images.get(os.path.splitext(fn)[0])
            if src is None:
                continue
            im = Image.open(src)
            W, H = im.size
            im.convert("L").resize((size, size)).save(
                os.path.join(out, split, "images", os.path.splitext(os.path.basename(fn))[0]
                             + ".png"))
            rows = g[g.file == fn]
            with open(os.path.join(out, split, "labels",
                                   os.path.splitext(os.path.basename(fn))[0] + ".txt"), "w") as f:
                for r in rows.itertuples():
                    cx, cy = (r.x0 + r.x1) / 2 / W, (r.y0 + r.y1) / 2 / H
                    w, h = (r.x1 - r.x0) / W, (r.y1 - r.y0) / H
                    f.write(f"{cid[r.cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n += 1
        counts[split] = n
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\ntrain: train/images\nval: val/images\n"
                f"nc: {len(classes)}\nnames: {classes}\n")
    return {"path": out, "layout": layout, "classes": classes, "images": counts,
            "objects": int(len(lab))}


def preview_boxes(name: str, n: int = 4, root: str | None = None, bbox_order: str = "rc",
                  out: str | None = None) -> str:
    """Draw the labelled boxes on a few images, to confirm the box convention by eye."""
    from PIL import Image, ImageDraw
    root = root or external_dir(name)
    layout, files = detect_layout(root)
    lab = read_coco_labels(files[0]) if layout == "coco" else read_csv_labels(files[0],
                                                                                 bbox_order)
    images = _find_images(root)
    tiles = []
    for fn in lab.file.unique()[:n]:
        src = images.get(fn) or images.get(os.path.splitext(fn)[0])
        if src is None:
            continue
        im = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(im)
        for r in lab[lab.file == fn].itertuples():
            d.rectangle([r.x0, r.y0, r.x1, r.y1], outline=(255, 40, 40), width=4)
            d.text((r.x0 + 4, r.y0 + 4), r.cls, fill=(255, 255, 0))
        tiles.append(im.resize((384, 384)))
    if not tiles:
        raise FileNotFoundError("none of the labelled images were found")
    sheet = Image.new("RGB", (384 * len(tiles), 384))
    for i, t in enumerate(tiles):
        sheet.paste(t, (384 * i, 0))
    out = out or os.path.join(feature_dir("debris", "vision"), f"{name}_box_preview.png")
    sheet.save(out)
    return out
