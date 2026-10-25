"""Supervised residual event model: deep per-channel event scores from telemetry and forecast
residuals, trained on the labels of the official TRAINING half.

    python -m cosmos.ai.telemetry.event_model --mission 1 --seed 0

Each target channel at each grid step gets seven causal features:

``asinh_z``       the robust-normalised value (``MissionData``), compressed
``log_s``         log1p of the out-of-fold AR(8) non-conformity score (:mod:`oof`)
``ewm_1h``, ``ewm_12h``, ``ewm_3d``  trailing exponential means of ``log_s`` (long events
                  last days on Mission 1, longer than the network's receptive field)
``asinh_dev_1d``  value minus its trailing 1-day exponential mean (slow drift removed)
``finite``        1 where a forecast exists

:class:`EventTCN` is a causal dilated TCN shared across channels.  It adds a learned channel
embedding and, every two levels, a permutation-equivariant cross-channel mixing step (the
mean over channels, projected and added back).  It outputs three logits per channel and step
(nominal / anomaly / rare nominal event) and is trained with class-weighted cross-entropy;
communication gaps are ignored.  Two scores are kept:

``event``    log P(anomaly or rare) - log P(nominal)  (the DMLR primary target)
``anomaly``  log P(anomaly) - log(1 - P(anomaly))

Out-of-fold protocol: the training half is split into the same K contiguous blocks as the
OOF forecaster scores.  For each block, a model is trained on the other blocks (minus a
purge margin) and predicts that block.  A final model is trained on all blocks and predicts
the test half.  Nothing here reads test labels.  Test predictions are written but not
scored; scoring is :mod:`oof_benchmark`'s job and is gated by a frozen pre-registration.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ...common import output_dir
from ..experiment import Run, count_parameters, pick_device, set_seed
from . import oof
from .forecasting import MissionData

FEATURES = ("asinh_z", "log_s", "ewm_1h", "ewm_12h", "ewm_3d", "asinh_dev_1d", "finite")
LABEL_IGNORE = 3                      # communication gaps (Mission 1 only)


@dataclass
class EventConfig:
    mission: int = 1
    channels: str = "target"
    source: str = "ar8"                   # OOF forecaster scores used as the residual input
    d_model: int = 64
    levels: int = 8                       # receptive field 1 + 2 * 2 * (2^8 - 1) = 1021
    kernel: int = 3
    dropout: float = 0.1
    mix_every: int = 2
    window: int = 2048                    # input length; loss on the last ``window - rf`` steps
    batch: int = 8                        # windows per step (each with every channel)
    steps: int = 1500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    event_fraction: float = 0.5           # share of windows centred on a labelled sample
    class_weights: tuple = (1.0, 5.0, 5.0)
    n_classes: int = 3
    inputs: str = "all"                   # all | no_residual | residual_only
    purge_days: float = 1.0
    seed: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


# ============================================================================ features
def _ewm(x: np.ndarray, rule_s: int, tau_s: float, init=None) -> np.ndarray:
    from scipy.signal import lfilter
    a = 1.0 - np.exp(-rule_s / tau_s)
    zi = None if init is None else (1 - a) * np.asarray(init, np.float64)[None, :]
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, axis=0,
                   zi=zi if zi is not None else np.zeros((1, x.shape[1])))
    return y


def features(Z: np.ndarray, S: np.ndarray, rule_s: int) -> np.ndarray:
    """``(T, C, 7)`` float32 causal features for one contiguous record."""
    T, C = Z.shape
    F = np.empty((T, C, len(FEATURES)), np.float32)
    fin = np.isfinite(S)
    ls = np.log1p(np.where(fin, np.minimum(S, 1e6), 0.0)).astype(np.float64)
    z = np.clip(np.asarray(Z, np.float64), -1e6, 1e6)
    F[..., 0] = np.arcsinh(z)
    F[..., 1] = ls
    F[..., 2] = _ewm(ls, rule_s, 3600.0, ls[0])
    F[..., 3] = _ewm(ls, rule_s, 12 * 3600.0, ls[0])
    F[..., 4] = _ewm(ls, rule_s, 3 * 86400.0, ls[0])
    F[..., 5] = np.arcsinh(z - _ewm(z, rule_s, 86400.0, z[0]))
    F[..., 6] = fin
    return F


def mission_features(D: MissionData, S_tr: np.ndarray, S_te: np.ndarray):
    """Features for the train and test halves, computed over the joined record so that the
    trailing means at the start of the test half carry the end of the training half."""
    Z = np.vstack([D.Ztr[:-1], D.Zte])                # last train grid point = first test one
    S = np.vstack([np.asarray(S_tr)[:-1], np.asarray(S_te)])
    F = features(Z, S, D.train.rule_s)
    n = D.Ztr.shape[0] - 1
    F_tr = np.concatenate([F[:n], F[n:n + 1]], 0)     # restore the shared grid point
    return F_tr, F[n:]


def select_inputs(F: np.ndarray, inputs: str) -> np.ndarray:
    if inputs == "all":
        return F
    if inputs == "no_residual":
        return F[..., [0, 5]]
    if inputs == "residual_only":
        return F[..., [1, 2, 3, 4, 6]]
    raise KeyError(inputs)


# ============================================================================ model
def build(cfg: EventConfig, n_channels: int, n_features: int):
    import torch
    from torch import nn
    import torch.nn.functional as Fn
    from .models import ChannelNorm, _CausalConv

    class _Block(nn.Module):
        def __init__(self, d, k, dil, p):
            super().__init__()
            self.c1, self.c2 = _CausalConv(d, d, k, dil), _CausalConv(d, d, k, dil)
            self.n1, self.n2 = ChannelNorm(d), ChannelNorm(d)
            self.drop = nn.Dropout(p)

        def forward(self, x):
            h = self.drop(Fn.silu(self.n1(self.c1(x))))
            return x + self.drop(Fn.silu(self.n2(self.c2(h))))

    class EventTCN(nn.Module):
        """``(B, C, F, T)`` -> logits ``(B, C, n_classes, T)``; causal along T."""

        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.inp = nn.Conv1d(n_features, d, 1)
            self.chan = nn.Embedding(n_channels, d)
            self.blocks = nn.ModuleList([_Block(d, cfg.kernel, 2 ** i, cfg.dropout)
                                         for i in range(cfg.levels)])
            self.mix = nn.ModuleList([nn.Conv1d(d, d, 1) for _ in
                                      range(cfg.levels // cfg.mix_every)])
            self.head = nn.Conv1d(d, cfg.n_classes, 1)
            self.receptive_field = 1 + 2 * (cfg.kernel - 1) * (2 ** cfg.levels - 1)

        def forward(self, x):
            B, C, Fdim, T = x.shape
            ids = torch.arange(C, device=x.device)
            h = self.inp(x.reshape(B * C, Fdim, T)) + self.chan(ids).repeat(B, 1)[:, :, None]
            j = 0
            for i, blk in enumerate(self.blocks):
                h = blk(h)
                if (i + 1) % cfg.mix_every == 0:
                    hv = h.view(B, C, -1, T)
                    h = (hv + self.mix[j](hv.mean(1)).unsqueeze(1)).reshape(B * C, -1, T)
                    j += 1
            return self.head(h).view(B, C, cfg.n_classes, T)

    return EventTCN()


def scores_from_logits(logits):
    """``(event, anomaly)`` log-odds from 3-class (or 2-class) logits along dim -2."""
    import torch
    lp = torch.log_softmax(logits.float(), dim=-2)
    if logits.shape[-2] == 2:
        ev = lp[..., 1, :] - lp[..., 0, :]
        return ev, ev
    ev = torch.logsumexp(lp[..., 1:, :], dim=-2) - lp[..., 0, :]
    an = lp[..., 1, :] - torch.log1p(-lp[..., 1, :].exp().clamp(max=1 - 1e-6))
    return ev, an


# ============================================================================ training
def _allowed(n: int, folds: list, held: int | None, purge: int) -> np.ndarray:
    ok = np.ones(n, bool)
    if held is not None:
        a, b = folds[held]
        ok[max(0, a - purge):min(n, b + purge)] = False
    return ok


def train_one(F: np.ndarray, L: np.ndarray, allowed: np.ndarray, cfg: EventConfig, seed: int,
              run=None, tag: str = ""):
    import torch
    set_seed(seed)
    dev = cfg.device
    T, C, Fdim = F.shape
    model = build(cfg, C, Fdim).to(dev)
    rf = model.receptive_field
    W = cfg.window
    assert W > rf + 64, "window must exceed the receptive field"
    out_len = W - rf
    rng = np.random.default_rng(seed)
    # window END positions e (exclusive): rows [e - W, e) must all be allowed
    bad = np.r_[0, np.cumsum(~allowed)]
    ends = np.arange(W, T + 1)
    ends = ends[(bad[ends] - bad[ends - W]) == 0]
    lab_rows = np.flatnonzero(((L == 1) | (L == 2)).any(1) & allowed)
    lab_rows = lab_rows[lab_rows >= W]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps,
                                                pct_start=0.05)
    cw = torch.tensor(cfg.class_weights[:cfg.n_classes], dtype=torch.float32, device=dev)
    t0, hist = time.time(), []

    def make_batch():
        e = np.empty(cfg.batch, np.int64)
        for i in range(cfg.batch):
            if lab_rows.size and rng.random() < cfg.event_fraction:
                r = lab_rows[rng.integers(lab_rows.size)]
                cand = r + 1 + rng.integers(out_len)
                e[i] = cand if (cand <= T and (bad[cand] - bad[cand - W]) == 0) \
                    else ends[rng.integers(ends.size)]
            else:
                e[i] = ends[rng.integers(ends.size)]
        xb = np.stack([F[k - W:k] for k in e])                       # (B, W, C, F)
        yb = np.stack([L[k - out_len:k] for k in e]).astype(np.int64)  # (B, out, C)
        if cfg.n_classes == 2:
            yb = np.where(yb == LABEL_IGNORE, -100, (yb > 0).astype(np.int64))
        else:
            yb = np.where(yb == LABEL_IGNORE, -100, yb)
        return xb, yb

    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(1)              # one worker: the RNG is not thread-safe
    pending = [pool.submit(make_batch) for _ in range(2)]
    for step in range(cfg.steps):
        xb, yb = pending.pop(0).result()
        pending.append(pool.submit(make_batch))
        x = torch.from_numpy(xb).to(dev).permute(0, 2, 3, 1).contiguous()  # (B, C, F, W)
        y = torch.from_numpy(yb).to(dev).permute(0, 2, 1)                    # (B, C, out)
        model.train()
        logits = model(x)[..., rf:]                                          # (B, C, K, out)
        loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 1, 3, 2).reshape(-1, cfg.n_classes), y.reshape(-1),
            weight=cw, ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(loss.item())
        if run is not None and (step + 1) % 250 == 0:
            run.log_epoch(step=step + 1, fold=tag, train_loss=float(np.mean(hist[-250:])),
                          seconds=time.time() - t0)
    pool.shutdown(wait=False, cancel_futures=True)
    return model, {"seconds": time.time() - t0, "final_loss": float(np.mean(hist[-250:])),
                   "params": count_parameters(model)}


def predict(model, F: np.ndarray, rows: tuple, device: str, chunk: int = 1 << 15):
    """Event and anomaly log-odds for rows [a, b) of ``F`` (context read from before ``a``)."""
    import torch
    a, b = rows
    rf = model.receptive_field
    T, C, _ = F.shape
    ev = np.zeros((b - a, C), np.float16)
    an = np.zeros((b - a, C), np.float16)
    model.eval()
    with torch.no_grad():
        for s in range(a, b, chunk):
            e = min(b, s + chunk)
            lo = max(0, s - rf)
            x = torch.from_numpy(np.ascontiguousarray(F[lo:e])).to(device)
            x = x.permute(1, 2, 0).unsqueeze(0)                              # (1, C, F, T)
            sc_e, sc_a = scores_from_logits(model(x)[0])                     # (C, T)
            ev[s - a:e - a] = sc_e[:, s - lo:].clamp(-60, 60).T.cpu().numpy()
            an[s - a:e - a] = sc_a[:, s - lo:].clamp(-60, 60).T.cpu().numpy()
    return ev, an


def model_dir(cfg: EventConfig) -> str:
    tag = (f"m{cfg.mission}_{cfg.channels}_event_{cfg.source}_{cfg.inputs}"
           f"_c{cfg.n_classes}_s{cfg.seed}")
    return output_dir("benchmarks", "telemetry_esa_adb", "event_model", tag)


def run(cfg: EventConfig) -> str:
    D = MissionData(cfg.mission, cfg.channels)
    S_tr, S_te, meta = oof.load(cfg.mission, cfg.channels, cfg.source)
    folds = [tuple(f) for f in meta["folds"]]
    purge = int(cfg.purge_days * 86400 / D.train.rule_s)
    F_tr, F_te = mission_features(D, S_tr, S_te)
    F_tr, F_te = select_inputs(F_tr, cfg.inputs), select_inputs(F_te, cfg.inputs)
    L = D.Ltr
    n = F_tr.shape[0]
    d = model_dir(cfg)
    tag = os.path.basename(d)
    ev_tr = np.zeros((n, F_tr.shape[1]), np.float16)
    an_tr = np.zeros_like(ev_tr)
    info = {"folds": []}
    with Run(f"esa_event_{tag}", dataclasses.asdict(cfg), cfg.seed, cfg.device,
             group="esa_event_model") as rn:
        for b in range(len(folds)):
            model, inf = train_one(F_tr, L, _allowed(n, folds, b, purge), cfg,
                                   cfg.seed * 100 + b, rn, f"fold{b}")
            e, a = predict(model, F_tr, folds[b], cfg.device)
            ev_tr[folds[b][0]:folds[b][1]], an_tr[folds[b][0]:folds[b][1]] = e, a
            info["folds"].append(inf)
            print(f"  {tag} fold {b}: loss {inf['final_loss']:.4f} in {inf['seconds']:.0f} s",
                  flush=True)
            del model
        model, inf = train_one(F_tr, L, _allowed(n, folds, None, purge), cfg,
                               cfg.seed * 100 + 99, rn, "final")
        rn.save_checkpoint(model, "final", features=list(FEATURES), inputs=cfg.inputs)
        ev_te, an_te = predict(model, F_te, (0, F_te.shape[0]), cfg.device)
        info["final"] = inf
        for name, arr in (("event_train_oof", ev_tr), ("anomaly_train_oof", an_tr),
                          ("event_test", ev_te), ("anomaly_test", an_te)):
            np.save(os.path.join(d, f"{name}.npy"), arr)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"config": dataclasses.asdict(cfg), "folds": folds, "purge": purge,
                       "run_id": rn.run_id, "receptive_field": model.receptive_field,
                       **info}, f, indent=1, default=str)
        rn.finish({"final_loss": inf["final_loss"]}, params=inf["params"])
    print(f"{tag}: written to {d}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, default=1)
    ap.add_argument("--channels", default="target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--inputs", default="all")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    cfg = EventConfig(mission=a.mission, channels=a.channels, seed=a.seed, steps=a.steps,
                      inputs=a.inputs, n_classes=a.classes, device=a.device or pick_device())
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
