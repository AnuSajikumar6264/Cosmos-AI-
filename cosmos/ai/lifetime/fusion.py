"""Reliability-gated fusion of a physics expert and a learned expert, and conformal calibration.

Fusion (quantile averaging, "Vincentisation")::

    q_fused(tau | x) = g(r) q_phys(tau | x) + (1 - g(r)) q_ml(tau | x),   g(r) = sigmoid(w . r + b)

``r`` holds *reliability* descriptors of the physics expert that are known at prediction time:
the source of its area-to-mass input (design values, a class prior of design values, or a
class prior with the breakup model), the spread of its own Monte-Carlo predictive
distribution, orbit eccentricity and perigee band, and the object class.  (The object's own
radar cross-section is not used anywhere: its availability in the catalogue depends on the
outcome; see physics.py.)  Averaging two monotone quantile
functions gives a monotone quantile function, so the fused output is a valid distribution.
``w, b`` are fitted on a tuning split by minimising the mean pinball loss (L2-regularised).
Special cases used as baselines: ``g = 1`` (physics only), ``g = 0`` (learned only),
``g = 0.5`` (equal average) and a constant ``g`` (global stacking weight).

Calibration (split conformal quantile regression, Romano et al. 2019, applied per Mondrian
group, Vovk et al. 2005): on a calibration split disjoint from training and gating, the score
``s = max(q_lo - z, z - q_hi)`` is computed and its ``(1 - alpha)(1 + 1/n)`` empirical quantile
widens (or narrows) the interval.  Groups are object class x perigee band; a group with fewer
than ``MIN_GROUP`` calibration points falls back to its class, then to the pooled set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import QUANTILES, Z_CAP, Z_FLOOR, snap_bounds  # noqa: F401  (snap_bounds re-exported)

MIN_GROUP = 40
PERIGEE_BANDS = [0, 300, 450, 600, 800, 2000]


def reliability_features(df: pd.DataFrame, phys_feat: pd.DataFrame, am_source: np.ndarray) -> np.ndarray:
    d = df.reset_index(drop=True)
    src = pd.Series(am_source)
    hp = d["perigee"].to_numpy(float)
    cols = [src.eq("design").to_numpy(float), src.eq("class_prior_design").to_numpy(float),
            phys_feat["phys_spread"].to_numpy(float), np.log10(np.clip(d["ecc"].to_numpy(float), 1e-4, None)),
            (hp < 300).astype(float), ((hp >= 300) & (hp < 600)).astype(float),
            d["type1"].eq("D").to_numpy(float), d["type1"].eq("R").to_numpy(float),
            d["type1"].eq("P").to_numpy(float), phys_feat["phys_p_cap"].to_numpy(float)]
    R = np.column_stack(cols)
    return R


class Gate:
    def __init__(self, l2: float = 1e-3, mode: str = "gated"):
        self.l2, self.mode = l2, mode

    def _g(self, R, theta):
        if self.mode == "constant":
            return np.full(len(R), 1 / (1 + np.exp(-theta[0])))
        return 1 / (1 + np.exp(-(R @ theta[1:] + theta[0])))

    def fit(self, R, qp, qm, z, steps: int = 3000, lr: float = 0.05):
        """Minimise the (sub-differentiable) pinball loss of the fused quantiles with Adam;
        finite-difference quasi-Newton stalls on this piecewise-linear objective."""
        import torch
        sd = R.std(0)
        # a descriptor that is constant on the fitting set gets scale 1 (its weight stays ~0)
        # instead of a near-zero scale that would blow up on any other value at prediction time
        self.mu, self.sd = R.mean(0), np.where(sd > 1e-12, sd, 1.0)
        Rs = torch.tensor((R - self.mu) / self.sd, dtype=torch.float64)
        qp_t, qm_t = torch.tensor(qp), torch.tensor(qm)
        z_t = torch.tensor(z)[:, None]
        taus = torch.tensor(QUANTILES)
        k = 1 if self.mode == "constant" else 1 + R.shape[1]
        theta = torch.zeros(k, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            lin = theta[0] if self.mode == "constant" else Rs @ theta[1:] + theta[0]
            g = torch.sigmoid(lin).reshape(-1, 1) * torch.ones(len(z_t), 1, dtype=torch.float64)
            d = z_t - (g * qp_t + (1 - g) * qm_t)
            loss = torch.maximum(taus * d, (taus - 1) * d).mean() + self.l2 * (theta[1:] ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.theta = theta.detach().numpy()
        self.final_loss = float(loss.detach())
        return self

    def weight(self, R):
        return self._g((R - self.mu) / self.sd, self.theta)

    def fuse(self, R, qp, qm):
        g = self.weight(R)[:, None]
        return g * qp + (1 - g) * qm


def mondrian_groups(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    band = pd.cut(d["perigee"], PERIGEE_BANDS, labels=False, include_lowest=True).fillna(len(PERIGEE_BANDS) - 2)
    return pd.DataFrame({"fine": d["type1"].astype(str) + "_" + band.astype(int).astype(str),
                         "coarse": d["type1"].astype(str), "all": "all"})


class MondrianCQR:
    """Conformalised quantile regression with a Mondrian (group-conditional) correction."""

    def __init__(self, levels=(0.8, 0.9), mondrian: bool = True, bounds=(Z_FLOOR, Z_CAP)):
        self.levels, self.mondrian, self.bounds = levels, mondrian, bounds

    @staticmethod
    def _idx(level):
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return int(np.argmin(np.abs(QUANTILES - lo))), int(np.argmin(np.abs(QUANTILES - hi)))

    def fit(self, q, z, groups: pd.DataFrame):
        self.corr = {}
        q = snap_bounds(q, *self.bounds)
        for lev in self.levels:
            i, j = self._idx(lev)
            s = np.maximum(q[:, i] - z, z - q[:, j])
            alpha = 1 - lev
            table = {}
            for col in (["fine", "coarse", "all"] if self.mondrian else ["all"]):
                for key, idx in groups.groupby(col).indices.items():
                    n = len(idx)
                    if n >= MIN_GROUP or col == "all":
                        k = min(int(np.ceil((1 - alpha) * (n + 1))), n) - 1
                        table[(col, key)] = float(np.sort(s[idx])[k])
            self.corr[lev] = table
        return self

    def interval(self, q, groups: pd.DataFrame, level):
        i, j = self._idx(level)
        q = snap_bounds(q, *self.bounds)
        t = self.corr[level]
        c = np.empty(len(q))
        for r, (f, co) in enumerate(zip(groups["fine"], groups["coarse"])):
            if self.mondrian and ("fine", f) in t:
                c[r] = t[("fine", f)]
            elif self.mondrian and ("coarse", co) in t:
                c[r] = t[("coarse", co)]
            else:
                c[r] = t[("all", "all")]
        return snap_bounds(q[:, i] - c, *self.bounds), snap_bounds(q[:, j] + c, *self.bounds)


def horizon_prob(q: np.ndarray, horizon_days: float) -> np.ndarray:
    """P(T <= H) read off the quantile function by linear interpolation in tau (tails clipped)."""
    zh = np.log10(horizon_days)
    taus = np.concatenate([[0.0], QUANTILES, [1.0]])
    out = np.empty(len(q))
    for r in range(len(q)):
        qq = np.concatenate([[Z_FLOOR - 1e-6], q[r], [Z_CAP + 1e-6]])
        qq = np.maximum.accumulate(qq + np.arange(qq.size) * 1e-9)
        out[r] = np.interp(zh, qq, taus)
    return np.clip(out, 0.0, 1.0)
