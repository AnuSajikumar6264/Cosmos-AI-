"""Learned experts for capped time to re-entry, all predicting the same quantiles of
``z = log10(min(T, 5 yr) in days)`` so that they can be fused and calibrated identically.

Every learner gets the same tuning budget (``N_TRIALS`` random configurations, scored by the
mean pinball loss on the tuning split) and is refitted with its best configuration.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
N_TRIALS = 12
Z_FLOOR = np.log10(0.5)
Z_CAP = np.log10(1826.25)
SNAP_TOL = 1e-6

NUMERIC = {
    "orbit": ["perigee", "apogee", "inc", "ecc", "sma_km", "mean_alt_km"],
    "age": ["age_at_t0_d"],
    "design": ["log_mass", "length", "diameter", "span", "log_area_design", "log_am_design", "mass_uncertain"],
    "rcs": ["log_rcs"],              # 2026 snapshot; availability depends on the outcome (leak demonstration only)
    "space_weather": ["f107_last81", "f107_last365", "ap_last81", "f107_trend365"],
    "family": ["launch_family_n_known", "launch_family_n_decayed", "launch_family_frac_decayed",
               "launch_family_mean_log_life", "breakup_family_n_known", "breakup_family_n_decayed",
               "breakup_family_frac_decayed", "breakup_family_mean_log_life"],
    "physics": ["phys_q10", "phys_q50", "phys_q90", "phys_spread", "phys_p_cap"],
}
CATEGORICAL = {"class": ["type1", "parent_type", "shape_top"], "orbit": ["orbit_uncertain"]}


def pinball(z, q, taus=QUANTILES) -> float:
    d = z[:, None] - q
    return float(np.mean(np.maximum(taus * d, (taus - 1) * d)))


def physics_features(samples_days: np.ndarray) -> pd.DataFrame:
    z = np.log10(np.clip(samples_days, 0.5, None))
    q = np.quantile(z, [0.1, 0.5, 0.9], axis=1).T
    return pd.DataFrame({"phys_q10": q[:, 0], "phys_q50": q[:, 1], "phys_q90": q[:, 2],
                         "phys_spread": q[:, 2] - q[:, 0],
                         "phys_p_cap": (samples_days >= 1826.0).mean(1)})


def physics_quantiles(samples_days: np.ndarray) -> np.ndarray:
    z = np.log10(np.clip(samples_days, 0.5, None))
    return np.quantile(z, QUANTILES, axis=1).T


def design_matrix(df: pd.DataFrame, phys: pd.DataFrame, groups: tuple, shapes: list | None = None) -> tuple:
    """Feature matrix with the listed feature groups (NaN kept: all learners handle it)."""
    d = df.reset_index(drop=True).copy()
    d["log_mass"] = np.log10(d["mass"])
    d["log_area_design"] = np.log10(d["area_design_m2"])
    d["log_am_design"] = np.log10(d["am_design"])
    d["log_rcs"] = np.log10(d["rcs_m2"])
    d["mass_uncertain"] = d["mass_uncertain"].astype(float)
    d["orbit_uncertain"] = d["orbit_uncertain"].astype(float)
    shapes = shapes if shapes is not None else d["shape"].value_counts().index[:12].tolist()
    d["shape_top"] = np.where(d["shape"].isin(shapes), d["shape"], "other")
    d = pd.concat([d, phys.reset_index(drop=True)], axis=1)
    cols = []
    for g in groups:
        cols += NUMERIC.get(g, [])
    X = d[cols].astype(float)
    cats = []
    for g in groups:
        cats += CATEGORICAL.get(g, [])
    for c in cats:
        if c == "orbit_uncertain":
            X[c] = d[c]
            continue
        for v in sorted(d[c].astype(str).unique()):
            X[f"{c}={v}"] = (d[c].astype(str) == v).astype(float)
    return X, shapes


def snap_bounds(q, lo, hi, tol: float = SNAP_TOL):
    """Clip to [lo, hi] and set values within ``tol`` of a bound to the bound exactly.

    The targets have point masses at the bounds (the 5-yr cap, the -30 risk floor).  Learners
    that store labels in float32 (LightGBM, CatBoost) and the convex fusion leave such values a
    few ulps off the bound, and exact comparisons (coverage, P(T <= cap)) then fail."""
    q = np.clip(np.asarray(q, float), lo, hi)
    q = np.where(q > hi - tol, hi, q)
    return np.where(q < lo + tol, lo, q)


def _monotone(q: np.ndarray, lo: float = Z_FLOOR, hi: float = Z_CAP) -> np.ndarray:
    return np.sort(snap_bounds(q, lo, hi), axis=1)


class _Bounded:
    """Clipping bounds of the target (default: the capped log10 lifetime range)."""
    lo, hi = Z_FLOOR, Z_CAP

    def bounds(self, lo, hi):
        self.lo, self.hi = lo, hi
        return self


def align_columns(X: pd.DataFrame, cols: list) -> pd.DataFrame:
    return X.reindex(columns=cols, fill_value=0.0)


# ------------------------------------------------------------------------------ learners
class XGBQuantile(_Bounded):
    name = "xgboost"

    def __init__(self, seed=0, n_jobs=32):
        self.seed, self.n_jobs = seed, n_jobs

    def space(self, rng):
        return {"max_depth": int(rng.integers(3, 9)), "learning_rate": float(10 ** rng.uniform(-1.7, -0.7)),
                "subsample": float(rng.uniform(0.6, 1.0)), "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_weight": float(10 ** rng.uniform(0, 1.5)), "reg_lambda": float(10 ** rng.uniform(-1, 1))}

    def fit(self, X, z, Xv, zv, params):
        import xgboost as xgb
        self.cols = list(X.columns)
        self.m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES, tree_method="hist",
                                  n_estimators=2000, early_stopping_rounds=50, random_state=self.seed,
                                  n_jobs=self.n_jobs, **params)
        self.m.fit(X, z, eval_set=[(Xv, zv)], verbose=False)
        return self

    def predict(self, X):
        return _monotone(self.m.predict(align_columns(X, self.cols)), self.lo, self.hi)


class LGBMQuantile(_Bounded):
    name = "lightgbm"

    def __init__(self, seed=0, n_jobs=32):
        self.seed, self.n_jobs = seed, n_jobs

    def space(self, rng):
        return {"num_leaves": int(rng.integers(15, 128)), "learning_rate": float(10 ** rng.uniform(-1.7, -0.7)),
                "subsample": float(rng.uniform(0.6, 1.0)), "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_samples": int(rng.integers(10, 80)), "reg_lambda": float(10 ** rng.uniform(-1, 1))}

    def fit(self, X, z, Xv, zv, params):
        import lightgbm as lgb
        self.cols = list(X.columns)
        self.ms = []
        for a in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(a), n_estimators=2000, subsample_freq=1,
                                  random_state=self.seed, n_jobs=self.n_jobs, verbose=-1, **params)
            m.fit(X, z, eval_set=[(Xv, zv)], callbacks=[lgb.early_stopping(50, verbose=False)])
            self.ms.append(m)
        return self

    def predict(self, X):
        Xa = align_columns(X, self.cols)
        return _monotone(np.column_stack([m.predict(Xa) for m in self.ms]), self.lo, self.hi)


class CatQuantile(_Bounded):
    name = "catboost"

    def __init__(self, seed=0, n_jobs=32):
        self.seed, self.n_jobs = seed, n_jobs

    def space(self, rng):
        return {"depth": int(rng.integers(4, 9)), "learning_rate": float(10 ** rng.uniform(-1.7, -0.7)),
                "l2_leaf_reg": float(10 ** rng.uniform(0, 1.3)), "bagging_temperature": float(rng.uniform(0, 1))}

    def fit(self, X, z, Xv, zv, params):
        from catboost import CatBoostRegressor
        self.cols = list(X.columns)
        loss = "MultiQuantile:alpha=" + ",".join(f"{a:g}" for a in QUANTILES)
        self.m = CatBoostRegressor(loss_function=loss, iterations=3000, od_type="Iter", od_wait=60,
                                   random_seed=self.seed, thread_count=self.n_jobs, verbose=False, **params)
        self.m.fit(X, z, eval_set=(Xv, zv), use_best_model=True)
        return self

    def predict(self, X):
        return _monotone(np.asarray(self.m.predict(align_columns(X, self.cols))), self.lo, self.hi)


class MLPQuantile(_Bounded):
    """Feed-forward quantile network: non-crossing quantiles via cumulative softplus increments,
    pinball loss, standardised inputs with missing-value indicators."""
    name = "mlp"

    def __init__(self, seed=0, device="cpu"):
        self.seed, self.device = seed, device

    def space(self, rng):
        return {"width": int(rng.choice([128, 256, 512])), "depth": int(rng.integers(2, 5)),
                "dropout": float(rng.uniform(0.0, 0.3)), "lr": float(10 ** rng.uniform(-3.5, -2.3)),
                "weight_decay": float(10 ** rng.uniform(-6, -3))}

    def _prep(self, X, fit=False):
        A = X.to_numpy(float)
        miss = np.isnan(A)
        if fit:
            self.mu = np.nanmean(A, 0)
            self.sd = np.nanstd(A, 0) + 1e-6
            self.keep_miss = miss.any(0)
        A = np.where(miss, self.mu, A)
        A = (A - self.mu) / self.sd
        return np.hstack([A, miss[:, self.keep_miss].astype(float)]).astype(np.float32)

    def fit(self, X, z, Xv, zv, params):
        import os as _os
        import torch
        import torch.nn as nn
        torch.set_num_threads(int(_os.environ.get("COSMOS_TORCH_THREADS", "16")))   # avoid oversubscription
        torch.manual_seed(self.seed)
        self.cols = list(X.columns)
        A, Av = self._prep(X, fit=True), self._prep(align_columns(Xv, self.cols))
        nin, nq = A.shape[1], len(QUANTILES)
        layers, w = [], params["width"]
        for i in range(params["depth"]):
            layers += [nn.Linear(nin if i == 0 else w, w), nn.GELU(), nn.Dropout(params["dropout"])]
        self.net = nn.Sequential(*layers, nn.Linear(w, nq)).to(self.device)
        taus = torch.tensor(QUANTILES, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        Xt = torch.tensor(A, device=self.device)
        zt = torch.tensor(np.asarray(z, np.float32), device=self.device)
        Xvt = torch.tensor(Av, device=self.device)
        zvt = torch.tensor(np.asarray(zv, np.float32), device=self.device)
        best, best_state, bad = np.inf, None, 0
        n = len(A)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for epoch in range(300):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 512):
                b = perm[i:i + 512].to(self.device)
                q = self._quant(self.net(Xt[b]))
                d = zt[b, None] - q
                loss = torch.maximum(taus * d, (taus - 1) * d).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                d = zvt[:, None] - self._quant(self.net(Xvt))
                vl = float(torch.maximum(taus * d, (taus - 1) * d).mean())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        self.net.load_state_dict(best_state)
        self.epochs = epoch + 1
        return self

    @staticmethod
    def _quant(out):
        import torch
        import torch.nn.functional as F
        return out[:, :1] + torch.cat([torch.zeros_like(out[:, :1]), torch.cumsum(F.softplus(out[:, 1:]), 1)], 1)

    def predict(self, X):
        import torch
        A = self._prep(align_columns(X, self.cols))
        self.net.eval()
        with torch.no_grad():
            q = self._quant(self.net(torch.tensor(A, device=self.device))).cpu().numpy()
        return _monotone(q, self.lo, self.hi)


class TabMQuantile(MLPQuantile):
    """TabM (Gorishniy et al., ICLR 2025; tabm 0.0.3, Apache-2.0): a parameter-efficient
    ensemble of k MLPs sharing most weights (BatchEnsemble-style), each member trained with its
    own pinball loss and the k predictions averaged.  Plain TabM (no numerical embeddings), the
    paper's hyperparameter space, AdamW, early stopping on the validation pinball loss.  The
    quantile head is the same non-crossing cumulative-softplus head as the MLP.  Runs on an
    Intel XPU when one is free (full precision), else on CPU."""
    name = "tabm"

    def __init__(self, seed=0, device=None, k=32):
        self.seed, self.device, self.k = seed, device, k

    def space(self, rng):
        return {"n_blocks": int(rng.integers(1, 5)), "d_block": int(rng.choice([128, 256, 512, 1024])),
                "dropout": float(rng.choice([0.0, rng.uniform(0.0, 0.5)])), "lr": float(10 ** rng.uniform(-4, -2.7)),
                "weight_decay": float(rng.choice([0.0, 10 ** rng.uniform(-4, -1)]))}

    def fit(self, X, z, Xv, zv, params):
        import os as _os
        import torch
        import tabm
        torch.set_num_threads(int(_os.environ.get("COSMOS_TORCH_THREADS", "16")))
        if self.device is None:
            from ..experiment import pick_device
            self.device = pick_device()
        torch.manual_seed(self.seed)
        self.cols = list(X.columns)
        A, Av = self._prep(X, fit=True), self._prep(align_columns(Xv, self.cols))
        nq = len(QUANTILES)
        self.net = tabm.TabM.make(n_num_features=A.shape[1], d_out=nq, n_blocks=params["n_blocks"],
                                  d_block=params["d_block"], dropout=params["dropout"], k=self.k).to(self.device)
        taus = torch.tensor(QUANTILES, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        Xt, Xvt = torch.tensor(A, device=self.device), torch.tensor(Av, device=self.device)
        zt = torch.tensor(np.asarray(z, np.float32), device=self.device)
        zvt = torch.tensor(np.asarray(zv, np.float32), device=self.device)
        best, best_state, bad = np.inf, None, 0
        n = len(A)
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        def members(x):                                  # (batch, k, nq), non-crossing per member
            out = self.net(x)
            return out[..., :1] + torch.cat([torch.zeros_like(out[..., :1]),
                                             torch.cumsum(torch.nn.functional.softplus(out[..., 1:]), -1)], -1)
        for epoch in range(300):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 256):
                b = perm[i:i + 256].to(self.device)
                d = zt[b, None, None] - members(Xt[b])
                loss = torch.maximum(taus * d, (taus - 1) * d).mean()   # mean over members = per-member losses
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                d = zvt[:, None] - torch.cat([members(Xvt[i:i + 4096]).mean(1) for i in range(0, len(Av), 4096)])
                vl = float(torch.maximum(taus * d, (taus - 1) * d).mean())
            if not np.isfinite(vl):
                raise FloatingPointError("TabM validation loss is not finite")
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k_: v.detach().clone() for k_, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 16:
                    break
        self.net.load_state_dict(best_state)
        self._members = members
        self.epochs = epoch + 1
        return self

    def predict(self, X):
        import torch
        A = self._prep(align_columns(X, self.cols))
        self.net.eval()
        with torch.no_grad():
            At = torch.tensor(A, device=self.device)
            q = torch.cat([self._members(At[i:i + 4096]).mean(1) for i in range(0, len(A), 4096)]).cpu().numpy()
        if not np.isfinite(q).all():
            raise FloatingPointError("TabM returned non-finite predictions")
        return _monotone(q, self.lo, self.hi)


class TabICLQuantile(_Bounded):
    """TabICLv2 (Qu et al.; tabicl 2.2.0, BSD-3) as an in-context tabular foundation model.

    Used with its default inference settings (no hyperparameter search: the model is not
    trained on our data); ``space`` returns the single default configuration, so the tuning
    budget is one evaluation."""
    name = "tabicl"

    def __init__(self, seed=0, device="cpu", n_estimators=8):
        # CPU, full precision: on the Intel XPU, mixed-precision inference returned NaN for whole
        # prediction calls sporadically (origin 2011, seeds 0, 2, 3); on CPU every call was finite
        self.seed, self.device, self.n_estimators = seed, device, n_estimators

    def space(self, rng):
        return {"n_estimators": self.n_estimators}

    def fit(self, X, z, Xv, zv, params):
        from tabicl import TabICLRegressor
        if self.device is None:
            from ..experiment import pick_device
            self.device = pick_device()
        self.cols = list(X.columns)
        self.m = TabICLRegressor(device=self.device, random_state=self.seed, verbose=False, use_amp=False,
                                 n_estimators=params.get("n_estimators", self.n_estimators))
        self.m.fit(X.to_numpy(np.float32), np.asarray(z, float))
        return self

    def predict(self, X):
        q = np.asarray(self.m.predict(align_columns(X, self.cols).to_numpy(np.float32), output_type="quantiles",
                                      alphas=list(QUANTILES)))
        if not np.isfinite(q).all():
            raise FloatingPointError(f"TabICL returned {int((~np.isfinite(q)).any(1).sum())} non-finite rows")
        return _monotone(q, self.lo, self.hi)


def _sdm_query_chunked_attention():
    """On Intel XPUs, PyTorch's scaled-dot-product attention materialises the full score
    matrix; for a 20k-row context the context self-attention alone needs about 26 GB.  This
    shim (installed into structured-data-models' attention module only) splits the call along
    the query dimension when the scores would exceed 1 GiB.  Each query row's attention is
    independent of the others, so the result is unchanged."""
    import torch
    import torch.nn.functional as F
    import sdm.nn.attention as A
    if getattr(A, "_cosmos_query_chunked", False):
        return

    class _F:
        def __getattr__(self, name):
            return getattr(F, name)

        @staticmethod
        def scaled_dot_product_attention(query, key, value, attn_mask=None, **kw):
            per_row = query[..., :1, :].numel() // query.size(-1) * key.size(-2) * 4
            step = max(1, (1 << 30) // max(per_row, 1))
            if query.device.type != "xpu" or step >= query.size(-2):
                return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, **kw)
            out = []
            for s in range(0, query.size(-2), step):
                m = attn_mask if attn_mask is None or attn_mask.size(-2) == 1 else attn_mask[..., s:s + step, :]
                out.append(F.scaled_dot_product_attention(query[..., s:s + step, :], key, value, attn_mask=m, **kw))
            return torch.cat(out, dim=-2)
    A.F = _F()
    A._cosmos_query_chunked = True


class KumoQuantile(_Bounded):
    """Kumo-Tabular "large" (NVIDIA, 2026; structured-data-models 0.2.1, Apache-2.0 code,
    OpenMDW-1.1 weights) as an in-context tabular foundation model.  It predicts 999 quantiles
    (q001..q999); the seven used here are read off them.  Default inference settings, no
    hyperparameter search (``space`` returns the single default configuration).  All inputs are
    passed as numerical columns (one-hot columns included); missing values are left to the
    model.  XPU when one is free (full precision; 27x faster than CPU with identical results in
    a smoke test), else CPU.  The encoded context of a 25k-row training set takes most of a 24 GB
    card, so at most one instance holds it on the device at a time; another instance that
    predicts later re-encodes its own (stored) context first.  A 25k-row context still exceeds a
    24 GB card (the row embedding), so the context is capped at ``max_context`` rows (default
    16,000; environment variable COSMOS_KUMO_CONTEXT), a random subsample drawn per seed."""
    name = "kumo"
    _live = None                                     # the instance whose context is on the device

    def __init__(self, seed=0, device=None, n_estimators=8, chunk=4096, max_context=None):
        import os as _os
        self.seed, self.device, self.n_estimators, self.chunk = seed, device, n_estimators, chunk
        self.max_context = int(_os.environ.get("COSMOS_KUMO_CONTEXT", "16000")) if max_context is None else max_context
        self.m = None

    def space(self, rng):
        return {"n_estimators": self.n_estimators}

    def _table(self, X, z=None):
        import sdm
        df = align_columns(X, self.cols).astype(float).copy()
        df.columns = [f"f{i}" for i in range(df.shape[1])]
        if z is not None:
            df["target"] = np.asarray(z, float)
        return sdm.TableTensor.from_pandas(df=df, stypes={c: "numerical" for c in df.columns}, device=self.device)

    def fit(self, X, z, Xv, zv, params):
        if self.device is None:
            from ..experiment import pick_device
            self.device = pick_device()
        self.cols = list(X.columns)
        z = np.asarray(z, float)
        if len(X) > self.max_context:
            keep = np.sort(np.random.default_rng(10_000 + self.seed).choice(len(X), self.max_context, replace=False))
            X, z = X.iloc[keep], z[keep]
        self.context_rows = len(X)
        self._ctx = (X.copy(), z.copy(), int(params.get("n_estimators", self.n_estimators)))
        self._encode()
        return self

    def _encode(self):
        import sdm
        import torch
        if KumoQuantile._live is not None and KumoQuantile._live is not self:
            KumoQuantile._live._release()
        if self.m is not None:
            return
        _sdm_query_chunked_attention()
        X, z, n_est = self._ctx
        t = self._table(X, z)
        self.m = sdm.models.KumoTabular(task="regression", device=self.device)
        self.m.fit(x=t.drop_columns("target"), y=t[:, "target"], num_estimators=n_est,
                   generator=torch.Generator(device=self.device).manual_seed(self.seed))
        KumoQuantile._live = self

    def _release(self):
        import torch
        if self.m is not None:
            self.m.clear()
            self.m = None
            if str(self.device).startswith("xpu"):
                torch.xpu.empty_cache()
        if KumoQuantile._live is self:
            KumoQuantile._live = None

    def predict(self, X):
        self._encode()
        cols = [f"q{int(round(a * 1000)):03d}" for a in QUANTILES]
        out = []
        for i in range(0, len(X), self.chunk):
            out.append(self.m.predict(self._table(X.iloc[i:i + self.chunk])).to_pandas()[cols].to_numpy(float))
        q = np.concatenate(out)
        if not np.isfinite(q).all():
            raise FloatingPointError(f"Kumo-Tabular returned {int((~np.isfinite(q)).any(1).sum())} non-finite rows")
        return _monotone(q, self.lo, self.hi)


LEARNERS = {"xgboost": XGBQuantile, "lightgbm": LGBMQuantile, "catboost": CatQuantile, "mlp": MLPQuantile,
            "tabicl": TabICLQuantile, "tabm": TabMQuantile, "kumo": KumoQuantile}
# models published 2022-2026 (the learned-expert candidates); the others are classic reference baselines
RECENT = ("tabm", "tabicl", "kumo")


def tune_and_fit(name, X, z, Xv, zv, seed=0, n_trials=N_TRIALS, bounds=None, **kw):
    """Random search with a fixed budget; returns the best fitted model and the trial log."""
    rng = np.random.default_rng(1000 + seed)
    best, log = None, []
    if name in ("tabicl", "kumo"):                    # pretrained in-context models: no search
        n_trials = 1
    for t in range(n_trials):
        m = LEARNERS[name](seed=seed, **kw)
        if bounds is not None:
            m.bounds(*bounds)
        p = m.space(rng)
        t0 = time.time()
        m.fit(X, z, Xv, zv, p)
        s = pinball(np.asarray(zv), m.predict(Xv))
        log.append({"trial": t, "params": p, "val_pinball": s, "seconds": time.time() - t0})
        if best is None or s < best[0]:
            best = (s, m, p)
    return best[1], best[2], log
