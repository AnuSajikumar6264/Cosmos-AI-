"""COSMOS synthetic frame-pair benchmark: existing TinyYOLO vs YOLOv8n vs heat-map detector.

    python -m cosmos.ai.vision.cosmos_synthetic generate       # once
    python -m cosmos.ai.vision.cosmos_synthetic tinyyolo --seed 0
    python -m cosmos.ai.vision.cosmos_synthetic yolov8n  --seed 0
    python -m cosmos.ai.vision.cosmos_synthetic heatmap  --seed 0

Data: scenes from the unchanged COSMOS generator (:func:`cosmos.debris.scene.make_scene`,
domain-randomised exactly as :func:`cosmos.debris.generate_dataset`), registered and
differenced by the unchanged COSMOS front end, exported in YOLO format.  Every scene is a new
random star field, and the three splits use disjoint seed ranges, so no scene (and no
augmented copy of one) crosses a split.  Train images are augmented as the original export
does; val and test are not.  The original 48-image export in ``vision/yolo_dataset`` is left
untouched.

All three models see the same 640 x 640 8-bit difference images and are scored by the same
evaluator (:mod:`.metrics`, COCO-style, all confidences), so the comparison is like for like.
These are SYNTHETIC results; they say nothing about real imagery (see SpotGEO).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from ...common import feature_dir, output_dir
from ...debris.scene import CLASSES, IMG_SIZE, _augment, make_scene, to_uint8
from ..experiment import Run, count_parameters, model_size_mb, pick_device, set_seed
from .metrics import coco_map, prf_at

SPLITS = {"train": (800, 100_000), "val": (200, 200_000), "test": (200, 300_000)}
BOX_PAD = 1.0                       # boxes as generated: max(5, 6 sigma) px


def data_dir(*parts) -> str:
    return os.path.join(feature_dir("debris", "cosmos_synthetic"), "yolo_v1", *parts)


def _scene_params(rng):
    return dict(n_stars=int(rng.integers(90, 220)), n_objects=int(rng.integers(5, 12)),
                jitter_px=float(1.0 * rng.uniform(0.4, 1.8)),
                psf_sigma=float(rng.uniform(1.3, 2.1)), dt_s=float(rng.uniform(2.5, 6.0)))


def _export_one(args):
    from PIL import Image
    from ...debris.detect import register_and_difference
    split, i, seed = args
    rng = np.random.default_rng(seed)
    params = _scene_params(rng)
    sc = make_scene(seed=int(rng.integers(1 << 30)), **params)
    diff, shift = register_and_difference(sc.frame_a, sc.frame_b)
    img, boxes = diff, sc.boxes()
    if split == "train":
        img, boxes = _augment(img, boxes, rng)
    u8 = to_uint8(img)
    name = f"{split}_{i:05d}"
    Image.fromarray(u8).save(data_dir(split, "images", f"{name}.png"))
    lines = [f"{int(c)} {cx / IMG_SIZE:.6f} {cy / IMG_SIZE:.6f} {w / IMG_SIZE:.6f} {h / IMG_SIZE:.6f}"
             for (cx, cy, w, h), c in zip(boxes, sc.truth.cls.values)]
    with open(data_dir(split, "labels", f"{name}.txt"), "w") as f:
        f.write("\n".join(lines))
    return {"split": split, "image": name, "seed": seed, "objects": len(lines),
            "classes": sc.truth.cls.tolist(), "box_px": [float(b[2]) for b in boxes],
            "registration_shift": [float(shift[0]), float(shift[1])], **params}


def generate(workers: int = 64) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from ..data.download import file_hashes, utc_now, write_manifest
    for sp in SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(data_dir(sp, sub), exist_ok=True)
    jobs = [(sp, i, base + i) for sp, (n, base) in SPLITS.items() for i in range(n)]
    with ProcessPoolExecutor(workers) as ex:
        recs = list(ex.map(_export_one, jobs, chunksize=4))
    with open(data_dir("data.yaml"), "w") as f:
        f.write(f"path: {data_dir()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    with open(data_dir("scenes.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    seeds = {sp: {r["seed"] for r in recs if r["split"] == sp} for sp in SPLITS}
    assert not (seeds["train"] & seeds["val"] or seeds["train"] & seeds["test"]
                or seeds["val"] & seeds["test"])
    summary = {sp: {"images": sum(r["split"] == sp for r in recs),
                    "objects": sum(r["objects"] for r in recs if r["split"] == sp),
                    "class_counts": np.bincount([c for r in recs if r["split"] == sp
                                                 for c in r["classes"]],
                                                minlength=len(CLASSES)).tolist(),
                    "box_px_median": float(np.median([b for r in recs if r["split"] == sp
                                                      for b in r["box_px"]]))}
               for sp in SPLITS}
    write_manifest(feature_dir("debris", "cosmos_synthetic"), {
        "dataset": "cosmos_synthetic_vision", "class": "SYNTHETIC", "status": "generated",
        "generator": "cosmos.debris.scene.make_scene + cosmos.debris.detect."
                     "register_and_difference (unchanged), via cosmos.ai.vision."
                     "cosmos_synthetic.generate", "generated_utc": utc_now(),
        "splits": {sp: {"n": n, "seed_range": [b, b + n - 1]} for sp, (n, b) in SPLITS.items()},
        "leakage_control": "one independent random scene per image; disjoint seed ranges per "
                           "split; augmentation only inside train",
        "classes": CLASSES, "summary": summary, "licence": "generated by this project"},
        [{"path": "yolo_v1/data.yaml", **file_hashes(data_dir("data.yaml"))},
         {"path": "yolo_v1/scenes.jsonl", **file_hashes(data_dir("scenes.jsonl"))}])
    return summary


# ============================================================================ data access
def load_split(split: str):
    """``(images uint8 (N, 640, 640), truths list of (n, 5) cls cx cy w h in px)``."""
    from PIL import Image
    paths = sorted(glob.glob(data_dir(split, "images", "*.png")))
    imgs, truths = [], []
    for p in paths:
        imgs.append(np.asarray(Image.open(p)))
        lp = p.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(".png", ".txt")
        rows = [list(map(float, l.split())) for l in open(lp) if l.strip()]
        t = np.array(rows, np.float64).reshape(-1, 5)
        t[:, 1:] *= IMG_SIZE
        truths.append(t)
    return np.stack(imgs), truths


def evaluate(dets: list, truths: list, conf: float) -> dict:
    m = coco_map(dets, truths, len(CLASSES))
    small = coco_map(dets, truths, len(CLASSES), area_max=10.0 ** 2)
    agn = [np.c_[d[:, :5], np.zeros(len(d))] if len(d) else d for d in dets]
    agt = [np.c_[np.zeros(len(t)), t[:, 1:]] if len(t) else t for t in truths]
    m_agn = coco_map(agn, agt, 1)
    return {"map50": m["map50"], "map50_95": m["map50_95"],
            "map50_class_agnostic": m_agn["map50"], "map50_95_class_agnostic": m_agn["map50_95"],
            "small_map50": small["map50"], "ap50_per_class": m["ap50_per_class"],
            **{f"{k}@conf": v for k, v in prf_at(dets, truths, conf).items()}}


def _save(model_name, seed, row):
    d = output_dir("benchmarks", "vision_cosmos_synthetic", "runs")
    with open(os.path.join(d, f"{model_name}_s{seed}.json"), "w") as f:
        json.dump(row, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}, indent=1,
                     default=float))


def _latency(fn, x, device):
    from ..evaluation import measure_inference
    return measure_inference(fn, x, device, n_items=len(x))


# ============================================================================ models
def run_tinyyolo(seed: int = 0, epochs: int = 50) -> dict:
    """The existing COSMOS TinyYOLO, trained with its own fit() (defaults unchanged)."""
    import torch
    from ...debris.yolo import CONF_DEFAULT, TinyYOLO, load_yolo_split
    device = pick_device()
    with Run("vision_synth_tinyyolo", {"epochs": epochs, "size": 320}, seed, device,
             group="vision_cosmos_synthetic") as run:
        t0 = time.time()
        m = TinyYOLO(device=device).fit(data_dir(), epochs=epochs, seed=seed)
        t_train = time.time() - t0
        x, _ = load_yolo_split(data_dir(), "test", m.size)
        _, truths = load_split("test")
        preds = m.predict(x, conf=0.01, min_separation_px=m.size / 64)
        sc = IMG_SIZE / m.size
        dets = [np.c_[p[["cx", "cy", "w", "h"]].to_numpy() * sc, p.score.to_numpy(),
                      p.cls.to_numpy()] if len(p) else np.zeros((0, 6)) for p in preds]
        row = {"model": "tinyyolo (existing COSMOS)", "seed": seed, **evaluate(dets, truths,
                                                                            CONF_DEFAULT),
               "legacy_centre_metric": {k: v for k, v in m.evaluate(data_dir(), "test").items()
                                        if k != "confusion_matrix"},
               "params": count_parameters(m.net), "size_mb": model_size_mb(m.net),
               "train_time_s": t_train,
               **_latency(lambda b: m.net(b), torch.as_tensor(x[:16], device=device), device)}
        run.save_checkpoint(m.net, "final")
        run.finish({k: v for k, v in row.items() if np.isscalar(v)})
    _save("tinyyolo", seed, row)
    return row


def run_yolov8n(seed: int = 0, epochs: int = 100, weights: str = "yolov8n.pt") -> dict:
    """Ultralytics YOLOv8n (COCO-pretrained), trained on the train split, val for selection."""
    import torch
    from ultralytics import YOLO
    device = pick_device()
    wdir = os.path.join(feature_dir("debris", "cosmos_synthetic"), "weights")
    os.makedirs(wdir, exist_ok=True)
    cfg = {"weights": weights, "epochs": epochs, "imgsz": 640, "batch": 16}
    with Run("vision_synth_yolov8n", cfg, seed, device, group="vision_cosmos_synthetic") as run:
        cwd = os.getcwd()
        os.chdir(wdir)                                       # ultralytics downloads weights here
        try:
            m = YOLO(weights)
            t0 = time.time()
            m.train(data=data_dir("data.yaml"), epochs=epochs, imgsz=640, batch=16, seed=seed,
                    device=device, workers=8, project=os.path.join(run.dir, "ultralytics"),
                    name="train", exist_ok=True, plots=False, verbose=False, deterministic=True)
            t_train = time.time() - t0
            best = YOLO(os.path.join(run.dir, "ultralytics", "train", "weights", "best.pt"))
            v = best.val(data=data_dir("data.yaml"), split="test", imgsz=640, device=device,
                         plots=False, verbose=False, conf=0.001)
        finally:
            os.chdir(cwd)
        imgs, truths = load_split("test")
        dets = []
        for im in imgs:
            r = best.predict(np.repeat(im[..., None], 3, -1), imgsz=640, conf=0.001,
                             device=device, verbose=False, max_det=300)[0]
            b = r.boxes
            dets.append(np.c_[b.xywh.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()]
                        if len(b) else np.zeros((0, 6)))
        x = torch.zeros((16, 3, 640, 640), device=device)
        net = best.model.to(device).eval()
        row = {"model": "yolov8n (ultralytics, COCO-pretrained)", "seed": seed,
               **evaluate(dets, truths, 0.25),
               "ultralytics_test_map50": float(v.box.map50),
               "ultralytics_test_map50_95": float(v.box.map),
               "params": count_parameters(net), "size_mb": model_size_mb(net),
               "train_time_s": t_train, **_latency(lambda b: net(b), x, device)}
        run.finish({k: v for k, v in row.items() if np.isscalar(v)})
    _save("yolov8n", seed, row)
    return row


def run_heatmap(seed: int = 0, epochs: int = 60, model_kind: str = "unet") -> dict:
    """Heat-map detector (4 class maps + offset + size), val-selected threshold and epoch."""
    import torch
    from .heatmap import build, decode, detection_loss, render
    device = pick_device()
    set_seed(seed)
    Xtr, Ttr = load_split("train")
    Xva, Tva = load_split("val")
    Xte, Tte = load_split("test")
    norm = lambda u8: (u8.astype(np.float32) - 128.0) / 64.0
    model = build(model_kind, 1, len(CLASSES), wh=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs, crop = 8, 320
    steps = max(1, len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)

    def targets(t, x0=0, y0=0, size=IMG_SIZE):
        pts = [t[t[:, 0] == c][:, 1:3] - [x0, y0] for c in range(len(CLASSES))]
        wh = [t[t[:, 0] == c][:, 3:5] for c in range(len(CLASSES))]
        return render(pts, (size, size), sigma=1.5, sizes=wh)

    def infer(X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8):
                o = model(torch.as_tensor(norm(X[i:i + 8])[:, None], device=device))
                for per in decode(o, 0.01, max_det=100):
                    out.append(np.concatenate([np.c_[d[:, 0], d[:, 1], d[:, 3], d[:, 4], d[:, 2],
                                                     np.full(len(d), c)]
                                               for c, d in enumerate(per)], 0))
        return out

    tag = f"vision_synth_heatmap_{model_kind}"
    best = {"map50_95": -1}
    with Run(tag, {"epochs": epochs, "model": model_kind, "crop": crop}, seed, device,
             group="vision_cosmos_synthetic") as run:
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            tot = 0.0
            for _ in range(steps):
                xs, hs, os_, whs, ms = [], [], [], [], []
                for i in rng.integers(0, len(Xtr), bs):
                    y0, x0 = rng.integers(0, IMG_SIZE - crop + 1, 2)
                    xs.append(norm(Xtr[i, y0:y0 + crop, x0:x0 + crop])[None])
                    h, o, wh, m = targets(Ttr[i], x0, y0, crop)
                    hs.append(h), os_.append(o), whs.append(wh), ms.append(m)
                out = model(torch.as_tensor(np.stack(xs), device=device))
                T = lambda a: torch.as_tensor(np.stack(a), device=device)
                loss, _ = detection_loss(out, T(hs), T(os_), T(ms), T(whs))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach())
            if ep % 5 == 0 or ep == epochs:
                v = coco_map(infer(Xva), Tva, len(CLASSES))
                run.log_epoch(epoch=ep, loss=tot / steps, val_map50=v["map50"],
                              val_map50_95=v["map50_95"])
                print(f"  ep {ep} loss {tot / steps:.4f} val mAP50 {v['map50']:.3f} "
                      f"mAP50-95 {v['map50_95']:.3f}", flush=True)
                if v["map50_95"] > best["map50_95"]:
                    best = {"map50_95": v["map50_95"], "epoch": ep}
                    run.save_checkpoint(model, "best")
        t_train = time.time() - t0
        model.load_state_dict(torch.load(os.path.join(run.dir, "checkpoints", "best.pt"),
                                         map_location=device)["state_dict"])
        dv = infer(Xva)
        conf = max((0.2, 0.3, 0.4, 0.5, 0.6), key=lambda c: prf_at(dv, Tva, c)["f1"])
        dets = infer(Xte)
        x = torch.as_tensor(norm(Xte[:16])[:, None], device=device)
        row = {"model": f"heatmap-{model_kind} (COSMOS-AI)", "seed": seed,
               **evaluate(dets, Tte, conf), "selected_conf": conf, "best_epoch": best["epoch"],
               "params": count_parameters(model), "size_mb": model_size_mb(model),
               "train_time_s": t_train, **_latency(lambda b: model(b), x, device)}
        run.finish({k: v for k, v in row.items() if np.isscalar(v)})
    _save(f"heatmap_{model_kind}", seed, row)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["generate", "tinyyolo", "yolov8n", "heatmap"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(argv)
    if a.what == "generate":
        print(json.dumps(generate(), indent=1))
    elif a.what == "tinyyolo":
        run_tinyyolo(a.seed, a.epochs or 50)
    elif a.what == "yolov8n":
        run_yolov8n(a.seed, a.epochs or 100)
    else:
        run_heatmap(a.seed, a.epochs or 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
