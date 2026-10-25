"""A learned health monitor for the COSMOS telemetry simulator, interchangeable with
:class:`cosmos.health.HealthMonitor` (same ``fit`` / ``score`` -> ``MonitorResult``).

    python -m cosmos.ai.telemetry.learned_monitor generate   # nominal training corpus
    python -m cosmos.ai.telemetry.learned_monitor train
    python -m cosmos.ai.telemetry.learned_monitor benchmark  # vs the classical monitor

How it scores a run (the classical monitor's protocol is kept: the first 40 % of the run is
the healthy window, and nothing after it is used for fitting):

1. a channel-independent PatchTST forecaster, pre-trained OFFLINE on a corpus of nominal
   simulated runs (different seeds, no faults), forecasts every channel 8 steps ahead;
2. per channel, the normalised forecast errors are EWMA-smoothed (alpha 0.08, the classical
   monitor's smoothing) so that evidence integrates over time;
3. the conformal calibration set is OFFLINE: smoothed errors of 20 held-out nominal corpus
   runs (never used for training; exchangeable with a new fault-free run, which is what
   conformal validity needs).  Calibrating on the run's own healthy window was tried first
   and failed in both directions — with ~1,000 serially dependent samples the smallest
   attainable Simes p-value is 16 / 1001 = 0.016, so q = 1e-3 could never alarm, and q = 0.05
   alarmed on every fault-free run;
4. Benjamini-Hochberg across the 16 channels at every step; the Simes p-value ``p_S`` gives
   the aggregate anomaly score ``a = clip(-log10(p_S) / (2 * -log10(q)), 0, 1)``, so 0.5 is
   exactly the BH rejection level ``q`` (the classical monitor's alarm level is also 0.5);
5. subsystem degradation = the same mapping of each subsystem's Bonferroni min-p.

SYNTHETIC ONLY: the simulator designed the faults, so results here are a functional check of
the integration, not evidence about real spacecraft (that is the ESA-ADB study).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...common import feature_dir, output_dir
from ...health.monitor import MonitorResult, _ewma
from ...health.telemetry import FAULTS, SUBSYSTEMS, TelemetrySeries, simulate_telemetry
from ..uncertainty import conformal_pvalues, simes

CONTEXT, HORIZON = 128, 8
TRAIN_FRAC = 0.4


def corpus_dir(*parts) -> str:
    return os.path.join(feature_dir("health", "cosmos_synthetic"), *parts)


def _normalise(x: np.ndarray, n_fit: int):
    med = np.median(x[:n_fit], 0)
    mad = 1.4826 * np.median(np.abs(x[:n_fit] - med), 0)
    return (x - med) / np.maximum(mad, 1e-9)


def generate_corpus(n_runs: int = 200, seed0: int = 50_000) -> str:
    """Nominal (fault-free) simulated runs for offline pre-training; never used for testing."""
    from ..data.download import file_hashes, utc_now, write_manifest
    os.makedirs(corpus_dir(), exist_ok=True)
    runs = []
    for k in range(n_runs):
        ts = simulate_telemetry(hours=24, faults=None, seed=seed0 + k)
        runs.append(_normalise(ts.frame[ts.channels].to_numpy(np.float64),
                               int(TRAIN_FRAC * ts.n)).astype(np.float32))
    X = np.stack(runs)                                        # (runs, T, 16)
    p = corpus_dir("nominal_runs.npy")
    np.save(p, X)
    write_manifest(feature_dir("health", "cosmos_synthetic"), {
        "dataset": "cosmos_synthetic_telemetry", "class": "SYNTHETIC", "status": "generated",
        "generator": "cosmos.health.telemetry.simulate_telemetry (unchanged), 24 h, 30 s, "
                     "no faults", "seeds": [seed0, seed0 + n_runs - 1],
        "normalisation": "per run: median / MAD of the first 40 % (the monitor's healthy "
                         "window)", "generated_utc": utc_now(),
        "evaluation_seeds": "benchmark() uses seeds 0..4 x 49 records, disjoint from these",
        "note": "Synthetic; never reported as real satellite performance."},
        [{"path": "nominal_runs.npy", **file_hashes(p), "shape": list(X.shape)}])
    return p


def train_forecaster(epochs: int = 30, device: str | None = None, seed: int = 0):
    import torch
    from ..experiment import Run, count_parameters, pick_device, set_seed
    from .models import PatchTSTForecaster, pinball_loss
    device = device or pick_device()
    set_seed(seed)
    X = np.load(corpus_dir("nominal_runs.npy"))
    R, T, C = X.shape
    tr, va = np.arange(R - 20), np.arange(R - 20, R)          # last 20 runs = validation
    model = PatchTSTForecaster(C, CONTEXT, HORIZON, patch=16, d_model=64, n_heads=4,
                               n_layers=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def batch(runs, n):
        r = rng.choice(runs, n)
        c = rng.integers(0, C, n)
        s = rng.integers(0, T - CONTEXT - HORIZON, n)
        idx = s[:, None] + np.arange(CONTEXT + HORIZON)[None]
        z = X[r[:, None], idx, c[:, None]]
        return (torch.as_tensor(z[:, :CONTEXT, None].transpose(0, 2, 1), device=device),
                torch.as_tensor(z[:, CONTEXT:], device=device), torch.as_tensor(c, device=device))

    best, state = np.inf, None
    with Run("cosmos_learned_monitor_forecaster", {"epochs": epochs, "context": CONTEXT,
                                                   "horizon": HORIZON}, seed, device,
             group="cosmos_synthetic_telemetry") as run:
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            for _ in range(200):
                x, y, c = batch(tr, 256)
                q, _ = model(x, c)
                loss = pinball_loss(q, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                x, y, c = batch(va, 4096)
                v = float(pinball_loss(model(x, c)[0], y))
            run.log_epoch(epoch=ep, val_pinball=v)
            if v < best:
                best = v
                state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        model.load_state_dict(state)
        p = run.save_checkpoint(model, "best")
        run.finish({"val_pinball": best, "train_time_s": time.time() - t0,
                    "params": count_parameters(model)})
    import shutil
    shutil.copy(p, corpus_dir("learned_monitor_forecaster.pt"))
    return model


@dataclass
class LearnedHealthMonitor:
    """Drop-in replacement for :class:`cosmos.health.HealthMonitor` (see the module doc)."""
    q: float = 1e-3
    alpha: float = 0.08
    alarm_threshold: float = 0.5
    alarm_persistence: int = 5
    device: str = "cpu"
    method: str = "learned-patchtst-conformal-bh"
    train_n: int = 0
    _model: object = field(default=None, repr=False)
    _cal: list = field(default=None, repr=False)
    _cols: list = field(default_factory=list, repr=False)

    def _load(self):
        if self._model is None:
            import torch
            from .models import PatchTSTForecaster
            p = corpus_dir("learned_monitor_forecaster.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"{p} missing — run: python -m "
                                        "cosmos.ai.telemetry.learned_monitor train")
            m = PatchTSTForecaster(16, CONTEXT, HORIZON, patch=16, d_model=64, n_heads=4,
                                   n_layers=2)
            m.load_state_dict(torch.load(p, map_location="cpu")["state_dict"])
            self._model = m.to(self.device).eval()
        return self._model

    def _scores(self, Z: np.ndarray) -> np.ndarray:
        """|y - q50| / max(q75 - q25, 0.05) per (t, channel); NaN for the first CONTEXT."""
        import torch
        m = self._load()
        T, C = Z.shape
        S = np.full((T, C), np.nan, np.float32)
        o = np.arange(CONTEXT - 1, T - 1, HORIZON)
        idx = o[:, None] - CONTEXT + 1 + np.arange(CONTEXT)[None]
        with torch.no_grad():
            for c in range(C):
                x = torch.as_tensor(Z[idx, c][:, None, :].astype(np.float32), device=self.device)
                q, _ = m(x, torch.full((len(o),), c, device=self.device))
                q = q.cpu().numpy()
                for h in range(HORIZON):
                    t = o + 1 + h
                    ok = t < T
                    y = Z[t[ok], c]
                    S[t[ok], c] = np.abs(y - q[ok, h, 3]) / np.maximum(q[ok, h, 4] - q[ok, h, 2],
                                                                    0.05)
        return S

    def fit(self, ts: TelemetrySeries, train_frac: float = TRAIN_FRAC) -> "LearnedHealthMonitor":
        n = max(CONTEXT + 4 * HORIZON, int(round(train_frac * ts.n)))
        if ts.fault_onset_index > 0:
            n = min(n, ts.fault_onset_index)                  # never calibrate on faults
        self.train_n = n
        self._cols = ts.channels
        self._cal = self._offline_calibration()
        return self

    def _offline_calibration(self) -> list:
        p = corpus_dir("learned_monitor_calibration.npz")
        if os.path.isfile(p):
            z = np.load(p)
            return [z[f"c{c}"] for c in range(len(z.files))]
        X = np.load(corpus_dir("nominal_runs.npy"))[-20:]      # the 20 validation runs
        burn = CONTEXT + 2 * int(1 / self.alpha)
        pools = [[] for _ in range(X.shape[2])]
        for run in X:
            S = self._smooth(self._scores(run.astype(np.float64)))
            for c in range(S.shape[1]):
                pools[c].append(S[burn:, c])
        cal = [np.sort(np.concatenate(v)).astype(np.float32) for v in pools]
        np.savez(p, **{f"c{c}": v for c, v in enumerate(cal)})
        return cal

    def _smooth(self, S: np.ndarray) -> np.ndarray:
        S = np.nan_to_num(S, nan=0.0)
        return np.column_stack([_ewma(S[:, c], self.alpha) for c in range(S.shape[1])])

    def score(self, ts: TelemetrySeries) -> MonitorResult:
        if self._cal is None:
            self.fit(ts)
        Z = _normalise(ts.frame[self._cols].to_numpy(np.float64), self.train_n)
        S = self._smooth(self._scores(Z))
        P = np.ones(S.shape)
        for c in range(S.shape[1]):
            s = np.where(np.isfinite(S[:, c]), S[:, c], -np.inf)
            P[:, c] = conformal_pvalues(self._cal[c], s)
        lvl = -np.log10(self.q)
        to_score = lambda p: np.clip(-np.log10(np.maximum(p, 1e-12)) / (2 * lvl), 0.0, 1.0)
        anomaly = to_score(simes(P))                          # scores are already smoothed
        deg = {}
        for sub, cols in ts.channel_map.items():
            j = [self._cols.index(c) for c in cols]
            pmin = np.minimum(P[:, j].min(1) * len(j), 1.0)    # Bonferroni within subsystem
            deg[sub] = to_score(pmin)
        degdf = pd.DataFrame(deg)[SUBSYSTEMS]
        z = pd.DataFrame(-np.log10(P), columns=self._cols)
        hot = anomaly > self.alarm_threshold
        det, run = -1, 0
        for i, h in enumerate(hot):
            run = run + 1 if h else 0
            if run >= self.alarm_persistence:
                det = i - self.alarm_persistence + 1
                break
        return MonitorResult(ts.frame["t_s"].values, anomaly, degdf, z, self.method,
                             self.train_n, det, self.alarm_threshold)


# ============================================================================ benchmark
def benchmark(seed_sets=(0, 1, 2, 3, 4), device: str | None = None,
              tag: str = "development") -> pd.DataFrame:
    """Every subsystem x fault, plus fault-free runs, for both monitors (SYNTHETIC).

    Seed sets 0-4 were used while the calibration design was being fixed (they are a
    DEVELOPMENT set); seed sets 5-9 were scored once afterwards (``tag="heldout"``).
    """
    from ...health.monitor import HealthMonitor, detection_delay
    from ..experiment import pick_device
    device = device or pick_device()
    rows = []
    # "stuck" is excluded: simulate_telemetry raises ValueError for it (its noise multiplier
    # 0.6 gives a negative noise scale, sigma * (0.6 - 1)).  This is a pre-existing simulator
    # bug, left unfixed here because the simulator is not part of COSMOS-AI.
    faults = [f for f in FAULTS if f != "stuck"]
    for s in seed_sets:
        cases = [(sub, f) for sub in SUBSYSTEMS for f in faults] + [(None, None)] * 8
        for k, (sub, f) in enumerate(cases):
            seed = 1000 * s + k
            ts = simulate_telemetry(hours=24, faults={sub: f} if sub else None, seed=seed)
            for name, mon in (("classical_ewma", HealthMonitor()),
                              ("learned_conformal", LearnedHealthMonitor(device=device))):
                t0 = time.time()
                res = mon.fit(ts).score(ts)
                secs = time.time() - t0
                d = detection_delay(ts, res)
                after = slice(ts.fault_onset_index, None) if sub else slice(res.train_n, None)
                worst = res.degradation.iloc[after].mean().idxmax()
                rows.append({"monitor": name, "seed_set": s, "subsystem": sub or "none",
                             "fault": f or "none", "faulted": bool(sub),
                             "alarm": res.detected_index >= 0,
                             "alarm_before_onset": bool(sub) and d.get("before_onset", False),
                             "delay_samples": d.get("delay_samples", np.nan),
                             "localised": (worst == sub) if sub else np.nan,
                             "seconds": secs})
    df = pd.DataFrame(rows)
    d = output_dir("benchmarks", "telemetry_cosmos_synthetic", tag)
    df.to_csv(os.path.join(d, "per_record.csv"), index=False)
    summ = []
    for (mon, s), g in df.groupby(["monitor", "seed_set"]):
        f, n = g[g.faulted], g[~g.faulted]
        tp = int((f.alarm & ~f.alarm_before_onset).sum())
        fn = int(len(f) - tp)
        fp = int(n.alarm.sum() + f.alarm_before_onset.sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        summ.append({"monitor": mon, "seed_set": s, "precision": p, "recall": r,
                     "f1": 2 * p * r / max(p + r, 1e-12), "false_alarms_on_nominal": int(n.alarm.sum()),
                     "median_delay_samples": float(f.delay_samples[f.alarm].median()),
                     "localisation_accuracy": float(f.localised.mean()),
                     "seconds_per_record": float(g.seconds.mean())})
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(d, "summary_by_seed.csv"), index=False)
    per_fault = (df[df.faulted].groupby(["monitor", "fault"])
                 .agg(recall=("alarm", "mean"), localised=("localised", "mean")).reset_index())
    per_fault.to_csv(os.path.join(d, "per_fault.csv"), index=False)
    return S


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "benchmark"
    if cmd == "generate":
        print(generate_corpus())
    elif cmd == "train":
        train_forecaster()
    elif cmd == "heldout":
        S = benchmark((5, 6, 7, 8, 9), tag="heldout")
        print(S.groupby("monitor").agg(["mean", "std"]).round(3).T.to_string())
    else:
        S = benchmark()
        print(S.groupby("monitor").agg(["mean", "std"]).round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
