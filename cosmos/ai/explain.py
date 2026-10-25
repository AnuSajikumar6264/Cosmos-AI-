"""Explanations with their own checks, so that an attribution is never shown unverified.

``integrated_gradients``   Sundararajan et al. (ICML 2017): path integral of gradients from a
                           baseline to the input (Riemann sum, ``steps`` points).  Satisfies
                           completeness: attributions sum to f(x) - f(baseline), which
                           :func:`completeness_gap` reports.
``randomisation_check``    Adebayo et al. (NeurIPS 2018): re-initialise the model's weights
                           and recompute; a faithful method must change (low rank correlation).
``deletion_curve``         remove the most-attributed inputs first and track the output; a
                           faithful attribution makes it fall faster than random removal
                           (Turbé et al., Nat. Mach. Intell. 2023 use this style of test).
``channel_evidence``       for the telemetry alarms the explanation is the decision rule
                           itself: each channel's conformal p-value at the alarm.

Attention weights are deliberately not offered as explanations (Jain & Wallace 2019).
"""
from __future__ import annotations

import copy

import numpy as np


def integrated_gradients(f, x, baseline=None, steps: int = 64):
    """``f`` maps a batch (B, ...) tensor to (B,) outputs; returns attributions like ``x``."""
    import torch
    x = x.detach()
    b = torch.zeros_like(x) if baseline is None else baseline.detach()
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=x.device)[1:]
    total = torch.zeros_like(x)
    for a in alphas:
        xi = (b + a * (x - b)).requires_grad_(True)
        out = f(xi).sum()
        g, = torch.autograd.grad(out, xi)
        total += g
    return (x - b) * total / steps


def completeness_gap(f, x, attr, baseline=None) -> float:
    """|sum(attr) - (f(x) - f(baseline))| relative to |f(x) - f(baseline)|, per batch mean."""
    import torch
    with torch.no_grad():
        b = torch.zeros_like(x) if baseline is None else baseline
        d = f(x) - f(b)
    s = attr.reshape(attr.shape[0], -1).sum(1)
    return float(((s - d).abs() / d.abs().clamp(min=1e-9)).mean())


def randomisation_check(model, f_of_model, x, steps: int = 32, seed: int = 0) -> float:
    """Spearman rank correlation between IG on the trained and on a re-initialised model."""
    import torch
    from scipy.stats import spearmanr
    a1 = integrated_gradients(lambda z: f_of_model(model, z), x, steps=steps)
    m2 = copy.deepcopy(model)
    torch.manual_seed(seed)
    for mod in m2.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    a2 = integrated_gradients(lambda z: f_of_model(m2, z), x, steps=steps)
    r = [spearmanr(np.abs(p.ravel()), np.abs(q.ravel())).statistic
         for p, q in zip(a1.cpu().numpy(), a2.cpu().numpy())]
    return float(np.nanmean(r))


def deletion_curve(f, x, attr, fractions=(0.0, 0.05, 0.1, 0.2, 0.4), seed: int = 0) -> dict:
    """Output after zeroing the top-|attr| fraction of inputs, against random zeroing."""
    import torch
    rng = np.random.default_rng(seed)
    flat = attr.abs().reshape(attr.shape[0], -1)
    order = torch.argsort(flat, dim=1, descending=True)
    out_attr, out_rand = [], []
    n = flat.shape[1]
    with torch.no_grad():
        for fr in fractions:
            k = int(round(fr * n))
            xa = x.clone().reshape(x.shape[0], -1)
            xr = x.clone().reshape(x.shape[0], -1)
            if k:
                xa.scatter_(1, order[:, :k], 0.0)
                ridx = torch.as_tensor(np.stack([rng.permutation(n)[:k] for _ in range(len(x))]),
                                       device=x.device)
                xr.scatter_(1, ridx, 0.0)
            out_attr.append(float(f(xa.reshape(x.shape)).mean()))
            out_rand.append(float(f(xr.reshape(x.shape)).mean()))
    return {"fractions": list(fractions), "attributed_first": out_attr, "random": out_rand}


def channel_evidence(p_row: np.ndarray, channels: list, q: float) -> list:
    """Channels ordered by conformal p-value at one alarm step, with the BH decision."""
    from .uncertainty import bh_reject
    rej = bh_reject(np.asarray(p_row)[None, :], q)[0]
    order = np.argsort(p_row)
    return [{"channel": channels[i], "p_value": float(p_row[i]), "rejected": bool(rej[i])}
            for i in order]
