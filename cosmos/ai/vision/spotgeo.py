"""SpotGEO (Chen et al., CVPR Workshops 2021): real, unresolved GEO targets in 5-frame sequences.

    python -m cosmos.ai.vision.spotgeo preprocess        # registered differences, once
    python -m cosmos.ai.vision.spotgeo classical         # the COSMOS front end as a baseline

Facts measured on the downloaded v2 release: 1,280 train and 5,120 test sequences of five
640 x 480 8-bit frames; 11,250 / 44,550 annotated objects; star streaks, clouds, and a
pointing drift of up to ~50 px per frame between frames.

Front end (shared by every model, so models differ only after it): for each frame, the other
four frames are registered onto it with the COSMOS phase-correlation registration
(:func:`cosmos.debris.detect.estimate_shift`), their median is subtracted, and the result is
divided by its robust sigma.  The star field cancels; a GEO target, which moves differently
from the stars, survives.  A high-passed copy of the raw frame (frame minus a 6-px Gaussian
background, robust-scaled) is kept as a second channel.

Scoring is the official ``validation.py`` from the spotGEO starter kit, vendored unmodified
(:mod:`.spotgeo_validation`, tau = 10 px, eps = 3 px).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from ...common import feature_dir
from . import spotgeo_validation as official

N_FRAMES, H, W = 5, 480, 640


def root(*parts) -> str:
    return os.path.join(feature_dir("debris", "spotgeo"), *parts)


def raw_dir(split: str) -> str:
    return root("raw", "extracted", "SpotGEOv2", split)


def load_annotations(split: str) -> dict:
    """``{sequence_id: {frame: (k, 2) array of (x, y)}}`` exactly as the official scorer."""
    with open(root("raw", "extracted", "SpotGEOv2", f"{split}_anno.json")) as f:
        return official.flat_to_hierarchical(json.load(f))


def sequences(split: str) -> list:
    return sorted(int(s) for s in os.listdir(raw_dir(split)) if s.isdigit())


def train_val_split(val_frac: float = 0.1, salt: str = "cosmos-ai-spotgeo") -> tuple:
    """Deterministic by-SEQUENCE split of the official train set (no frame of a validation
    sequence is ever in training).  Hash-based, so it does not depend on any RNG state."""
    tr, va = [], []
    for s in sequences("train"):
        h = int(hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (va if h < val_frac else tr).append(s)
    return tr, va


def read_frames(split: str, seq: int) -> np.ndarray:
    from PIL import Image
    return np.stack([np.asarray(Image.open(os.path.join(raw_dir(split), str(seq), f"{f}.png")),
                                np.float32) for f in range(1, N_FRAMES + 1)])


def _robust(x):
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    return (x - med) / max(s, 1e-6)


def front_end(F: np.ndarray) -> tuple:
    """``(X (5, 2, H, W) float32, shifts (5, 5, 2))``: registered difference + high-pass."""
    from scipy.ndimage import gaussian_filter
    from ...debris.detect import apply_shift, estimate_shift
    X = np.zeros((N_FRAMES, 2, H, W), np.float32)
    shifts = np.zeros((N_FRAMES, N_FRAMES, 2), np.float32)
    for i in range(N_FRAMES):
        others = []
        for j in range(N_FRAMES):
            if j == i:
                continue
            dy, dx = estimate_shift(F[i], F[j])
            shifts[i, j] = dy, dx
            others.append(apply_shift(F[j], dy, dx))
        X[i, 0] = _robust(F[i] - np.median(np.stack(others), 0))
        X[i, 1] = _robust(F[i] - gaussian_filter(F[i], 6.0))
    return np.clip(X, -50, 200), shifts


def cache_path(split: str, seq: int) -> str:
    return root("processed", "front_end_f16", split, f"{seq}.npy")


def _prep_one(args):
    split, seq = args
    p = cache_path(split, seq)
    if os.path.isfile(p):
        return 0
    X, shifts = front_end(read_frames(split, seq))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p + ".tmp.npy", X.astype(np.float16))
    os.replace(p + ".tmp.npy", p)
    np.save(p.replace(".npy", "_shifts.npy"), shifts)
    return 1


def preprocess(workers: int = 64) -> int:
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(sp, s) for sp in ("train", "test") for s in sequences(sp)]
    with ProcessPoolExecutor(workers) as ex:
        done = sum(ex.map(_prep_one, jobs, chunksize=8))
    return done


def load_cached(split: str, seq: int) -> np.ndarray:
    return np.load(cache_path(split, seq)).astype(np.float32)


# ============================================================================ scoring
def score(pred: dict, split: str = "test", seqs=None, labels: dict | None = None) -> dict:
    """Official precision / recall / F1 / MSE (and 1 - F1) on ``seqs`` of ``split``.

    ``pred`` is ``{seq: {frame: (k, 2) array}}``; sequences missing from ``pred`` count as
    "no detections".  MSE is reported as the official code computes it (sum of per-sequence
    MSEs) and, separately, as the page's pooled definition is not what the server ran.
    """
    labels = labels or load_annotations(split)
    seqs = list(seqs) if seqs is not None else sorted(labels)
    X = {s: {f: np.asarray(pred.get(s, {}).get(f, np.zeros((0, 2)))).reshape(-1, 2)
             for f in range(1, N_FRAMES + 1)} for s in seqs}
    Y = {s: {f: np.asarray(labels[s][f]).reshape(-1, 2) for f in range(1, N_FRAMES + 1)}
         for s in seqs}
    # official.score_sequences, with its divisions guarded: the official code raises
    # ZeroDivisionError when there are no predictions at all (precision then 0, as usual)
    sc = [official.score_sequence(X[k], Y[k]) for k in seqs]
    TP, FN, FP = (sum(x[i] for x in sc) for i in range(3))
    mse = sum(x[3] for x in sc)
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "one_minus_f1": float(1 - f1), "official_mse_sum": float(mse),
            "official_mse_mean_per_sequence": float(mse) / max(len(seqs), 1),
            "tp": int(TP), "fn": int(FN), "fp": int(FP), "n_sequences": len(seqs)}


def to_submission(pred: dict, seqs) -> list:
    """The official flat JSON format (validated by :func:`official.validate_json` for test)."""
    out = []
    for s in seqs:
        for f in range(1, N_FRAMES + 1):
            c = np.asarray(pred.get(s, {}).get(f, np.zeros((0, 2)))).reshape(-1, 2)[:30]
            c = np.clip(c, [-0.5, -0.5], [639.5, 479.5])
            out.append({"sequence_id": int(s), "frame": f, "num_objects": int(len(c)),
                        "object_coords": c.tolist()})
    return out


# ============================================================================ classical
def classical_detect(X: np.ndarray, nsigma: float, max_det: int = 30) -> dict:
    """COSMOS DoG detector on each frame's registered difference (dipoles rejected)."""
    from ...debris.detect import dog_detect
    out = {}
    for f in range(N_FRAMES):
        d = dog_detect(X[f, 0], nsigma=nsigma, sigmas=(1.0, 1.6, 2.4), min_separation=4.0,
                       max_detections=max_det)
        out[f + 1] = d.frame[["cx", "cy"]].to_numpy() if len(d.frame) else np.zeros((0, 2))
    return out


def _classical_one(args):
    split, seq, grid = args
    X = load_cached(split, seq)
    return seq, {g: classical_detect(X, g) for g in grid}


def run_classical(grid=(5.0, 6.0, 7.0, 8.0, 10.0, 12.0), workers: int = 64) -> dict:
    """Select nsigma on the validation sequences (official F1), then score test once."""
    from concurrent.futures import ProcessPoolExecutor
    _tr, va = train_val_split()
    with ProcessPoolExecutor(workers) as ex:
        val = dict(ex.map(_classical_one, [("train", s, grid) for s in va], chunksize=4))
    lab = load_annotations("train")
    sel = {g: score({s: val[s][g] for s in va}, "train", va, lab) for g in grid}
    best = max(grid, key=lambda g: sel[g]["f1"])
    with ProcessPoolExecutor(workers) as ex:
        te = dict(ex.map(_classical_one, [("test", s, (best,)) for s in sequences("test")],
                         chunksize=8))
    pred = {s: te[s][best] for s in te}
    return {"selected_nsigma": best, "validation": sel, "test": score(pred, "test"),
            "pred": pred}


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "preprocess"
    if cmd == "preprocess":
        print("front end written for", preprocess(), "new sequences")
    elif cmd == "classical":
        from ...common import output_dir
        r = run_classical()
        pred = r.pop("pred")
        d = output_dir("benchmarks", "vision_spotgeo")
        with open(os.path.join(d, "classical_cosmos_frontend.json"), "w") as f:
            json.dump({k: (v if k != "validation" else {str(g): s for g, s in v.items()})
                       for k, v in r.items()}, f, indent=2)
        with open(os.path.join(d, "classical_cosmos_frontend_test_submission.json"), "w") as f:
            json.dump(to_submission(pred, sequences("test")), f)
        print(json.dumps(r["test"], indent=2), "selected nsigma", r["selected_nsigma"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
