"""Train and evaluate heat-map detectors on SpotGEO (validation-selected, test scored once).

    python -m cosmos.ai.vision.train_spotgeo --model unet --inputs both --seed 0

``--inputs``  ``both`` = registered differences + high-passed frames (10 channels),
              ``diff`` = registered differences only (5), ``raw`` = high-passed frames only
              (5; no COSMOS registration front end) — the ablation of the front end.
``--train-frac``  fraction of the training sequences (label-efficiency curve).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ...common import output_dir
from ..experiment import Run, count_parameters, model_size_mb, pick_device, set_seed
from . import spotgeo as S
from .heatmap import build, decode, detection_loss, render

CHANNELS = {"both": [0, 1], "diff": [0], "raw": [1]}
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def to_input(X: np.ndarray, inputs: str) -> np.ndarray:
    """(5, 2, H, W) front-end array -> (5 * len(channels), H, W) network input."""
    ch = CHANNELS[inputs]
    return X[:, ch].transpose(1, 0, 2, 3).reshape(-1, X.shape[2], X.shape[3])


def predict(model, arrays, inputs: str, device, thresholds=THRESHOLDS, batch: int = 4):
    """``{thr: {seq: {frame: (k, 2)}}}`` for full frames."""
    import torch
    model.eval()
    out = {t: {} for t in thresholds}
    keys = list(arrays) if isinstance(arrays, dict) else None
    items = arrays.items() if isinstance(arrays, dict) else arrays
    buf = []

    def flush():
        x = torch.as_tensor(np.stack([b[1] for b in buf]), device=device)
        with torch.no_grad():
            o = model(x)
        for t in thresholds:
            dets = decode(o, t)
            for (seq, _), d in zip(buf, dets):
                out[t][seq] = {f + 1: d[f][:, :2] for f in range(S.N_FRAMES)}
        buf.clear()

    for seq, X in items:
        buf.append((seq, to_input(X, inputs)))
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    return out


def _test_iter(seqs):
    for s in seqs:
        yield s, S.load_cached("test", s)


def train(cfg: dict) -> dict:
    import torch
    device = cfg.get("device") or pick_device()
    set_seed(cfg["seed"])
    tr, va = S.train_val_split()
    if cfg.get("train_frac", 1.0) < 1.0:
        rng = np.random.default_rng(1000 + cfg["seed"])
        tr = sorted(rng.choice(tr, int(round(cfg["train_frac"] * len(tr))), replace=False))
    lab = S.load_annotations("train")
    t0 = time.time()
    cache = {s: np.load(S.cache_path("train", s)) for s in tr + va}       # float16 in RAM
    in_ch = S.N_FRAMES * len(CHANNELS[cfg["inputs"]])
    model = build(cfg["model"], in_ch, S.N_FRAMES).to(device)
    lr = cfg.get("lr", 1e-3 if cfg["model"] == "unet" else 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs, crop, bs = cfg.get("epochs", 40), cfg.get("crop", 320), cfg.get("batch", 12)
    steps_per_epoch = max(1, len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.1)
    rng = np.random.default_rng(cfg["seed"])
    tag = (f"spotgeo_{cfg['model']}_{cfg['inputs']}_f{int(100 * cfg.get('train_frac', 1))}")
    best = {"f1": -1}
    with Run(tag, cfg, cfg["seed"], device, group="vision_spotgeo") as run:
        for ep in range(1, epochs + 1):
            model.train()
            tot = 0.0
            for it in range(steps_per_epoch):
                xs, hs, os_, ms = [], [], [], []
                for s in rng.choice(tr, bs):
                    X = cache[s].astype(np.float32)
                    pts = [lab[s][f].reshape(-1, 2).copy() for f in range(1, 6)]
                    if rng.random() < 0.5:                   # time reversal
                        X, pts = X[::-1], pts[::-1]
                    if rng.random() < 0.5:                   # horizontal flip
                        X = X[..., ::-1]
                        pts = [np.c_[S.W - 1 - p[:, 0], p[:, 1]] for p in pts]
                    if rng.random() < 0.5:                   # vertical flip
                        X = X[..., ::-1, :]
                        pts = [np.c_[p[:, 0], S.H - 1 - p[:, 1]] for p in pts]
                    y0 = rng.integers(0, S.H - crop + 1)
                    x0 = rng.integers(0, S.W - crop + 1)
                    Xc = X[..., y0:y0 + crop, x0:x0 + crop] * rng.uniform(0.8, 1.25)
                    pc = [p - np.array([x0, y0]) for p in pts]
                    h, o, _wh, m = render(pc, (crop, crop))
                    xs.append(to_input(np.ascontiguousarray(Xc), cfg["inputs"]))
                    hs.append(h)
                    os_.append(o)
                    ms.append(m)
                x = torch.as_tensor(np.stack(xs), device=device)
                out = model(x)
                loss, parts = detection_loss(out, torch.as_tensor(np.stack(hs), device=device),
                                             torch.as_tensor(np.stack(os_), device=device),
                                             torch.as_tensor(np.stack(ms), device=device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                tot += float(loss.detach())
            if ep % cfg.get("eval_every", 2) == 0 or ep == epochs:
                pv = predict(model, {s: cache[s].astype(np.float32) for s in va},
                             cfg["inputs"], device)
                sc = {t: S.score(pv[t], "train", va, lab) for t in THRESHOLDS}
                t_best = max(THRESHOLDS, key=lambda t: sc[t]["f1"])
                run.log_epoch(epoch=ep, train_loss=tot / steps_per_epoch,
                              val_f1=sc[t_best]["f1"], val_thr=t_best,
                              elapsed_s=time.time() - t0)
                print(f"  ep {ep:3d} loss {tot / steps_per_epoch:.4f} val F1 "
                      f"{sc[t_best]['f1']:.4f} @ {t_best}", flush=True)
                if sc[t_best]["f1"] > best["f1"]:
                    best = {"f1": sc[t_best]["f1"], "thr": t_best, "epoch": ep,
                            "val": sc[t_best]}
                    run.save_checkpoint(model, "best", threshold=t_best)
        train_time = time.time() - t0
        ck = torch.load(os.path.join(run.dir, "checkpoints", "best.pt"), map_location=device)
        model.load_state_dict(ck["state_dict"])
        t1 = time.time()
        seqs = S.sequences("test")
        pt = predict(model, _test_iter(seqs), cfg["inputs"], device, (best["thr"],))[best["thr"]]
        t_inf = time.time() - t1
        test = S.score(pt, "test", seqs)
        from ..evaluation import measure_inference
        x1 = torch.as_tensor(to_input(S.load_cached("test", seqs[0]), cfg["inputs"])[None],
                             device=device)
        eff = measure_inference(lambda b: model(b), x1, device, n_items=1)
        metrics = {**{f"test_{k}": v for k, v in test.items()},
                   "val_f1": best["f1"], "selected_threshold": best["thr"],
                   "best_epoch": best["epoch"], "train_time_s": train_time,
                   "test_inference_s": t_inf, "params": count_parameters(model),
                   "size_mb": model_size_mb(model),
                   "latency_ms_per_sequence": eff["latency_ms_per_item"],
                   "peak_memory_mb": eff.get("peak_memory_mb"), "n_train_sequences": len(tr)}
        run.finish(metrics)
        with open(os.path.join(run.dir, "test_submission.json"), "w") as f:
            json.dump(S.to_submission(pt, seqs), f)
    row = {"model": cfg["model"], "inputs": cfg["inputs"], "seed": cfg["seed"],
           "train_frac": cfg.get("train_frac", 1.0), "run_id": run.run_id, **metrics}
    d = output_dir("benchmarks", "vision_spotgeo", "runs")
    with open(os.path.join(d, f"{tag}_s{cfg['seed']}.json"), "w") as f:
        json.dump(row, f, indent=2, default=float)
    print(json.dumps(row, indent=1, default=float))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet")
    ap.add_argument("--inputs", default="both", choices=list(CHANNELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    train({"model": a.model, "inputs": a.inputs, "seed": a.seed, "epochs": a.epochs,
           "train_frac": a.train_frac, "device": a.device})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
