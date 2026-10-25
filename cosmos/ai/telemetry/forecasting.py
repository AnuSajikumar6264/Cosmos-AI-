"""Channel-independent forecasting on ESA-ADB, from nominal training data to test-set scores.

The pipeline every forecaster goes through (so encoders differ in ONE component only):

1. **Periods inside the official training half** (nothing from the test half is used):
   ``fit`` (model training) | ``es`` (early stopping) | ``cal`` (conformal calibration and
   calibrator selection), see :data:`PERIODS`.  The official test half is scored once.
2. **Normalisation** — per channel median and IQR of the nominal ``fit`` samples.
3. **Training windows** — random (channel, time) pairs whose context and horizon contain no
   labelled sample of that channel (nominal-only, self-supervised forecasting).
4. **Forecast schedule** — forecast origins every ``H`` steps; each origin sees the previous
   ``L`` samples and forecasts quantiles of the next ``H``.  Every model, deep or not, uses
   the same schedule, so each target time is forecast at the same lead.
5. **Non-conformity score** per (time, channel):
   ``|y - q50| / max(q75 - q25, delta)`` for quantile models,
   ``|y - yhat| / max(sigma_c, delta)`` for point models (sigma_c = robust residual scale on
   ``fit``), with ``delta = 0.05`` robust units.  Scores then go to :mod:`.alarms`.

Encoders: ``persistence`` (yhat = last value), ``ar8`` (the COSMOS AR(8) forecaster),
``tcn``, ``patchtst`` (trained here), ``chronos-bolt-*`` / ``chronos-2`` (pretrained,
zero-shot).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.preprocess_telemetry import ESA_PROTOCOL, ESASplit, load_esa_split
from .models import QUANTILES, build_forecaster, pinball_loss

#: periods inside the official TRAIN half (fit | early stopping | calibration)
PERIODS = {1: {"fit": ("2000-01-01", "2006-01-01"), "es": ("2006-01-01", "2006-04-01"),
               "cal": ("2006-04-01", "2007-01-01")},
           2: {"fit": ("2000-01-01", "2001-04-01"), "es": ("2001-04-01", "2001-05-15"),
               "cal": ("2001-05-15", "2001-10-01")}}
DELTA = 0.05
Q_INDEX = {q: i for i, q in enumerate(QUANTILES)}


@dataclass
class ForecastConfig:
    mission: int = 1
    encoder: str = "tcn"                  # persistence | ar8 | tcn | patchtst | chronos-...
    channels: str = "target"              # target | lightweight
    context: int = 512
    horizon: int = 16
    telecommands: bool = False
    cov_dropout: float = 0.3
    d_model: int = 64
    levels: int = 7                       # TCN receptive field 1 + 4 (2^7 - 1) = 509 <= L
    n_layers: int = 3
    patch: int = 16
    steps: int = 20000
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 1000
    patience: int = 6                     # evaluations without improvement
    seed: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


# ============================================================================ data
class MissionData:
    """Train and test splits of one mission, normalised, with period boundaries."""

    def __init__(self, mission: int, channels: str = "target"):
        self.mission = mission
        self.train = load_esa_split(mission, "train")
        self.test = load_esa_split(mission, "test")
        proto = ESA_PROTOCOL[mission]
        self.channels = sorted(proto["lightweight" if channels == "lightweight" else "target"],
                               key=lambda c: int(c.split("_")[1]))
        self.jtr = self.train.channel_index(self.channels)
        self.jte = self.test.channel_index(self.channels)
        self.p = {k: (self.train.index_of(a), self.train.index_of(b))
                  for k, (a, b) in PERIODS[mission].items()}
        f0, f1 = self.p["fit"]
        X = np.asarray(self.train.X[f0:f1][:, self.jtr], np.float64)
        L = np.asarray(self.train.L[f0:f1][:, self.jtr])
        med, scale = np.zeros(len(self.channels)), np.ones(len(self.channels))
        for k in range(len(self.channels)):
            v = X[L[:, k] == 0, k]
            if v.size == 0:
                v = X[:, k]
            med[k] = np.median(v)
            q1, q3 = np.percentile(v, [25, 75])
            s = (q3 - q1) / 1.349
            if s <= 1e-12:                                   # discrete / ZOH channels
                q01, q99 = np.percentile(v, [1, 99])
                s = (q99 - q01) / 4.653
            if s <= 1e-12:
                s = max(np.std(v), 1e-6 * (abs(med[k]) + 1.0), 1e-9)
            scale[k] = s
        self.med, self.scale = med.astype(np.float32), scale.astype(np.float32)
        self.Ztr = ((np.asarray(self.train.X[:, self.jtr], np.float32) - self.med)
                    / self.scale).astype(np.float32)
        self.Zte = ((np.asarray(self.test.X[:, self.jte], np.float32) - self.med)
                    / self.scale).astype(np.float32)
        self.Ltr = np.asarray(self.train.L[:, self.jtr])
        self.Lte = np.asarray(self.test.L[:, self.jte])
        # labelled-sample prefix counts per channel, for O(1) "window is nominal" checks
        self.cum = np.vstack([np.zeros((1, len(self.channels)), np.int64),
                              np.cumsum(self.Ltr > 0, axis=0)])
        self.cov_tr = telecommand_features(self.train.TC, self.train.rule_s)
        # test covariates continue from the end of training (time since the last command)
        self.cov_te = telecommand_features(np.vstack([self.train.TC[-50001:-1], self.test.TC]),
                                           self.test.rule_s)[50000:]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def record(self, period: str):
        """``(Z, L, cov, warm)`` for a period, with ``warm`` samples of preceding context."""
        if period == "test":
            # the last training grid point IS the first test grid point; do not repeat it
            ctx = slice(-4097, -1)
            return (np.vstack([self.Ztr[ctx], self.Zte]), np.vstack([self.Ltr[ctx], self.Lte]),
                    np.vstack([self.cov_tr[ctx], self.cov_te]), 4096)
        a, b = self.p[period]
        w = min(4096, a)
        return self.Ztr[a - w:b], self.Ltr[a - w:b], self.cov_tr[a - w:b], w


def telecommand_features(TC: np.ndarray, rule_s: int, taus_h=(1.0, 12.0)) -> np.ndarray:
    """Causal ``exp(-time since last execution / tau)`` per telecommand and tau."""
    T, K = TC.shape
    out = np.zeros((T, K * len(taus_h)), np.float32)
    idx = np.arange(T)
    for k in range(K):
        hit = np.flatnonzero(TC[:, k] > 0)
        if hit.size == 0:
            continue
        last = np.full(T, -10 ** 9, np.int64)
        last[hit] = hit
        last = np.maximum.accumulate(last)
        dt_h = (idx - last) * rule_s / 3600.0
        for j, tau in enumerate(taus_h):
            out[:, k * len(taus_h) + j] = np.where(last >= 0, np.exp(-dt_h / tau), 0.0)
    return out


# ============================================================================ training
def sample_windows(D: MissionData, rng, n: int, span: int, period: str = "fit"):
    """``n`` (channel, start) pairs whose ``span`` samples are all nominal in that channel."""
    a, b = D.p[period]
    ch = rng.integers(0, D.n_channels, n * 4)
    st = rng.integers(a, b - span, n * 4)
    ok = (D.cum[st + span, ch] - D.cum[st, ch]) == 0
    ch, st = ch[ok][:n], st[ok][:n]
    if len(ch) < n:                                          # very contaminated channel set
        extra = sample_windows(D, rng, n - len(ch), span, period)
        ch, st = np.r_[ch, extra[0]], np.r_[st, extra[1]]
    return ch, st


def _batch(D, ch, st, L, H, cfg, device, torch):
    idx = st[:, None] + np.arange(L + H)[None, :]
    z = D.Ztr[idx, ch[:, None]]                              # (B, L + H)
    x = torch.as_tensor(z[:, :L], device=device)[:, None, :]
    y = torch.as_tensor(z[:, L:], device=device)
    cov = None
    if cfg.telecommands:
        cov = torch.as_tensor(D.cov_tr[idx[:, :L]], device=device).permute(0, 2, 1)
    return x, y, cov, torch.as_tensor(ch, device=device)


def build_model(cfg: ForecastConfig, n_channels: int, n_cov: int):
    if cfg.encoder == "tcn":
        return build_forecaster("tcn", n_channels=n_channels, horizon=cfg.horizon,
                                d_model=cfg.d_model, levels=cfg.levels,
                                n_cov=n_cov if cfg.telecommands else 0)
    if cfg.encoder == "patchtst":
        return build_forecaster("patchtst", n_channels=n_channels, context=cfg.context,
                                horizon=cfg.horizon, patch=cfg.patch,
                                d_model=max(cfg.d_model, 64), n_layers=cfg.n_layers,
                                n_cov=n_cov if cfg.telecommands else 0)
    raise KeyError(cfg.encoder)


def _forward_last(model, x, ch, cov, cfg, train: bool):
    """Quantiles (B, H, Q) for the step after the last context sample."""
    if cfg.encoder == "tcn":
        q, gate = model(x, ch, cov, cfg.cov_dropout if train else 0.0, last_only=True)
        return q[:, -1], gate
    return model(x, ch, cov, cfg.cov_dropout if train else 0.0)


def train_forecaster(D: MissionData, cfg: ForecastConfig, run=None):
    """Nominal-only quantile training with early stopping on the ``es`` period."""
    import torch
    from ..experiment import set_seed
    set_seed(cfg.seed)
    torch_dev = cfg.device
    model = build_model(cfg, D.n_channels, D.cov_tr.shape[1]).to(torch_dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps,
                                                pct_start=0.05)
    rng = np.random.default_rng(cfg.seed)
    L, H = cfg.context, cfg.horizon
    es_rng = np.random.default_rng(12345)                    # fixed early-stopping windows
    es_ch, es_st = sample_windows(D, es_rng, 8192, L + H, "es")
    best, best_state, bad, t0 = math.inf, None, 0, time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        ch, st = sample_windows(D, rng, cfg.batch, L + H)
        x, y, cov, cht = _batch(D, ch, st, L, H, cfg, torch_dev, torch)
        q, _ = _forward_last(model, x, cht, cov, cfg, True)
        loss = pinball_loss(q, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(es_ch), 1024):
                    x, y, cov, cht = _batch(D, es_ch[i:i + 1024], es_st[i:i + 1024], L, H,
                                            cfg, torch_dev, torch)
                    q, _ = _forward_last(model, x, cht, cov, cfg, False)
                    tot += float(pinball_loss(q, y)) * len(cht)
                    n += len(cht)
            val = tot / n
            if run is not None:
                run.log_epoch(step=step, train_loss=float(loss.detach()), es_loss=val,
                              lr=sched.get_last_lr()[0], elapsed_s=time.time() - t0)
            if val < best - 1e-5:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    model.load_state_dict(best_state)
    return model, {"best_es_loss": best, "steps_run": step, "train_time_s": time.time() - t0}


# ============================================================================ inference
def origins(n: int, warm: int, H: int) -> np.ndarray:
    """Forecast origins: the last context index before each block of H targets."""
    return np.arange(warm - 1, n - 1, H)


def forecast_record(model_or_pipe, D: MissionData, period: str, cfg: ForecastConfig,
                    batch: int = 4096):
    """Median, IQR and quantile spread for every target sample of a period.

    Returns ``(Y, q25, q50, q75, L)`` shaped (T, C) for the period's own samples.
    """
    Z, Lab, cov, warm = D.record(period)
    n, C = Z.shape
    H, Lc = cfg.horizon, cfg.context
    o = origins(n, warm, H)
    Q50, Q25, Q75 = (np.full((n, C), np.nan, np.float32) for _ in range(3))
    enc = cfg.encoder
    if enc == "persistence":
        tgt = np.minimum(o[:, None] + np.arange(1, H + 1)[None, :], n - 1).ravel()
        src = np.repeat(o, H)
        Q50[tgt] = Z[src]
    elif enc == "ar8":
        Q50 = _ar_forecast(D, Z, o, H)
    elif enc.startswith("chronos"):
        _chronos_forecast(model_or_pipe, Z, o, H, Lc, Q25, Q50, Q75, batch)
    else:
        _torch_forecast(model_or_pipe, Z, cov, o, H, Lc, Q25, Q50, Q75, cfg, batch)
    sl = slice(warm, n)
    return Z[sl], Q25[sl], Q50[sl], Q75[sl], Lab[sl]


def _ar_forecast(D: MissionData, Z: np.ndarray, o: np.ndarray, H: int, order: int = 8):
    """The COSMOS AR(8) ridge forecaster, fitted on ``fit`` nominal data, iterated H steps."""
    n, C = Z.shape
    out = np.full((n, C), np.nan, np.float32)
    a, b = D.p["fit"]
    rng = np.random.default_rng(0)
    for c in range(C):
        z = D.Ztr[a:b, c].astype(np.float64)
        nom = D.Ltr[a:b, c] == 0
        idx = np.flatnonzero(nom[order:] & nom[:-order])[:]
        idx = idx[rng.permutation(len(idx))[:500000]] + order
        A = np.stack([z[idx - k] for k in range(1, order + 1)] + [np.ones(len(idx))], 1)
        w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(order + 1), A.T @ z[idx])
        hist = np.stack([Z[np.maximum(o - k + 1, 0), c] for k in range(1, order + 1)], 1)
        hist = hist.astype(np.float64)                       # lag 1 .. order at the origin
        for h in range(1, H + 1):
            pred = hist @ w[:order] + w[order]
            t = o + h
            ok = t < n
            out[t[ok], c] = pred[ok]
            hist = np.concatenate([pred[:, None], hist[:, :-1]], 1)
    return out


def _torch_forecast(model, Z, cov, o, H, Lc, Q25, Q50, Q75, cfg, batch):
    import torch
    dev = cfg.device
    model.eval()
    C = Z.shape[1]
    i25, i50, i75 = Q_INDEX[0.25], Q_INDEX[0.5], Q_INDEX[0.75]
    if cfg.encoder == "tcn":
        # fully convolutional: one pass over the record per channel (chunked, with the
        # receptive field as overlap), then read the outputs at the origins
        rf = model.receptive_field
        chunk = 1 << 18
        n = Z.shape[0]
        with torch.no_grad():
            for c in range(C):
                for s in range(0, n, chunk):
                    a = max(0, s - rf)
                    e = min(n, s + chunk)
                    x = torch.as_tensor(Z[a:e, c], device=dev)[None, None, :]
                    cv = (torch.as_tensor(cov[a:e], device=dev).T[None] if cfg.telecommands
                          else None)
                    q, _ = model(x, torch.tensor([c], device=dev), cv)
                    q = q[0].float().cpu().numpy()            # (T_chunk, H, Q)
                    oo = o[(o >= s) & (o < e)]
                    for h in range(H):
                        t = oo + 1 + h
                        ok = t < n
                        r = oo[ok] - a
                        Q25[t[ok], c] = q[r, h, i25]
                        Q50[t[ok], c] = q[r, h, i50]
                        Q75[t[ok], c] = q[r, h, i75]
        return
    o = o[o >= Lc - 1]
    with torch.no_grad():
        for c in range(C):
            for i in range(0, len(o), batch):
                oo = o[i:i + batch]
                idx = oo[:, None] - Lc + 1 + np.arange(Lc)[None, :]
                x = torch.as_tensor(Z[idx, c], device=dev)[:, None, :]
                cv = (torch.as_tensor(cov[idx], device=dev).permute(0, 2, 1)
                      if cfg.telecommands else None)
                q, _ = model(x, torch.full((len(oo),), c, device=dev), cv)
                q = q.float().cpu().numpy()                   # (B, H, Q)
                _scatter(oo, q[:, :, i25], q[:, :, i50], q[:, :, i75], c, H, Q25, Q50, Q75)


def _scatter(oo, a25, a50, a75, c, H, Q25, Q50, Q75):
    n = Q50.shape[0]
    for h in range(H):
        t = oo + 1 + h
        ok = t < n
        Q25[t[ok], c] = a25[ok, h]
        Q50[t[ok], c] = a50[ok, h]
        Q75[t[ok], c] = a75[ok, h]


def load_chronos(name: str, device: str):
    import torch
    from chronos import BaseChronosPipeline
    repo = {"chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
            "chronos-bolt-small": "amazon/chronos-bolt-small",
            "chronos-bolt-base": "amazon/chronos-bolt-base",
            "chronos-2": "amazon/chronos-2"}[name]
    return BaseChronosPipeline.from_pretrained(repo, device_map=device,
                                               torch_dtype=torch.float32)


def _chronos_forecast(pipe, Z, o, H, Lc, Q25, Q50, Q75, batch):
    import torch
    o = o[o >= Lc - 1]
    C = Z.shape[1]
    is2 = type(pipe).__name__ == "Chronos2Pipeline"
    for c in range(C):
        for i in range(0, len(o), batch):
            oo = o[i:i + batch]
            idx = oo[:, None] - Lc + 1 + np.arange(Lc)[None, :]
            x = torch.as_tensor(Z[idx, c])
            if is2:
                qs, _ = pipe.predict_quantiles(list(x[:, None, :]), prediction_length=H,
                                               quantile_levels=[0.25, 0.5, 0.75])
                q = torch.stack([t[0] for t in qs]).float().numpy()    # (B, H, 3)
            else:
                q, _ = pipe.predict_quantiles(x, prediction_length=H,
                                              quantile_levels=[0.25, 0.5, 0.75])
                q = q.float().cpu().numpy()                             # (B, H, 3)
            _scatter(oo, q[:, :, 0], q[:, :, 1], q[:, :, 2], c, H, Q25, Q50, Q75)


# ============================================================================ scores
def residual_scale(D: MissionData, cfg: ForecastConfig, model=None) -> np.ndarray:
    """Robust residual scale per channel for point forecasters, from the ``es`` period."""
    Y, _q25, q50, _q75, Lab = forecast_record(model, D, "es", cfg)
    r = np.abs(Y - q50)
    out = np.ones(Y.shape[1], np.float32)
    for c in range(Y.shape[1]):
        v = r[(Lab[:, c] == 0) & np.isfinite(r[:, c]), c]
        if v.size:
            out[c] = max(1.4826 * np.median(v), np.percentile(v, 90) / 1.645, DELTA)
    return out


def nonconformity(Y, q25, q50, q75, sigma=None) -> np.ndarray:
    """Normalised absolute forecast error (see the module docstring); NaN where unforecast."""
    if sigma is not None:
        s = np.abs(Y - q50) / np.maximum(sigma[None, :], DELTA)
    else:
        s = np.abs(Y - q50) / np.maximum(q75 - q25, DELTA)
    return s.astype(np.float32)
