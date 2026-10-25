"""Experiment bookkeeping: one folder per run, everything needed to reproduce or audit it.

    with Run("opssat_cnn", config, seed=0) as run:
        ...train...
        run.log_epoch(epoch=1, train_loss=0.3, val_loss=0.4)
        run.save_checkpoint(model, "best")
        run.finish(test_metrics)

Each run writes ``outputs/experiments/<run_id>/``:

``config.yaml``   the resolved configuration (plus its SHA-256, recorded in the index)
``env.json``      Python / PyTorch / NumPy versions, device names, host, CPU count
``epochs.csv``    per-epoch training log
``metrics.json``  final metrics, training time, parameter count, model size, best epoch
``checkpoints/``  model weights (``best.pt`` = the validation-selected one)

and appends one row to ``outputs/experiments/index.jsonl``.  A run that raises is recorded with
``status = failed`` and the exception text — failed runs are never silently dropped.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
import traceback

import numpy as np

from ..common import output_dir


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and XPU/CUDA when present)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
    except ImportError:                                       # pragma: no cover
        pass


def free_xpus(min_free_gib: float = 8.0) -> list:
    """XPU indices with at least ``min_free_gib`` free (other users' jobs are left alone)."""
    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return []
        out = []
        for i in range(torch.xpu.device_count()):
            free, _tot = torch.xpu.mem_get_info(i)
            if free / 2**30 >= min_free_gib:
                out.append(i)
        return out
    except Exception:                                         # pragma: no cover
        return []


def pick_device(prefer: str | None = None, index: int | None = None) -> str:
    """``"xpu:<i>"`` on a free Intel GPU, else ``"cpu"``.  ``COSMOS_DEVICE`` overrides."""
    env = os.environ.get("COSMOS_DEVICE")
    if env:
        return env
    if prefer == "cpu":
        return "cpu"
    free = free_xpus()
    if not free:
        return "cpu"
    if index is not None:
        return f"xpu:{free[index % len(free)]}"
    return f"xpu:{free[0]}"


def environment(device: str | None = None) -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "host": socket.gethostname(), "cpu_count": os.cpu_count(),
           "numpy": np.__version__, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    try:
        import pandas as pd
        env["pandas"] = pd.__version__
    except ImportError:                                       # pragma: no cover
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        if device and device.startswith("xpu"):
            env["device_name"] = torch.xpu.get_device_name(int(device.split(":")[1]))
    except Exception:                                         # pragma: no cover
        pass
    env["device"] = device or "cpu"
    return env


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_mb(model) -> float:
    return float(sum(p.numel() * p.element_size() for p in model.parameters())
                 + sum(b.numel() * b.element_size() for b in model.buffers())) / 2**20


class Run:
    """Context manager that records one experiment run (see the module docstring)."""

    def __init__(self, name: str, config: dict, seed: int = 0, device: str | None = None,
                 group: str | None = None, root: str | None = None):
        self.name, self.config, self.seed, self.device = name, dict(config), seed, device
        self.group = group or name
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}_{name}_s{seed}"
        self.root = root or output_dir("experiments")
        self.dir = os.path.join(self.root, self.run_id)
        self.epochs = []
        self.metrics = {}
        self.extra = {}
        self._t0 = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "Run":
        os.makedirs(os.path.join(self.dir, "checkpoints"), exist_ok=True)
        set_seed(self.seed)
        import yaml
        with open(os.path.join(self.dir, "config.yaml"), "w") as f:
            yaml.safe_dump({"name": self.name, "group": self.group, "seed": self.seed,
                            **self.config}, f, sort_keys=False)
        with open(os.path.join(self.dir, "env.json"), "w") as f:
            json.dump(environment(self.device), f, indent=2)
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "ok" if exc_type is None else "failed"
        if exc_type is not None:
            self.extra["error"] = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        self._write(status)
        return False                                          # never swallow the error

    # ------------------------------------------------------------------ logging
    def log_epoch(self, **row) -> None:
        """Record one row and stream it to ``epochs.jsonl`` (long runs stay observable)."""
        self.epochs.append(row)
        with open(os.path.join(self.dir, "epochs.jsonl"), "a") as f:
            f.write(json.dumps(row, default=_jsonable) + "\n")

    def save_checkpoint(self, model, tag: str = "best", **extra) -> str:
        import torch
        p = os.path.join(self.dir, "checkpoints", f"{tag}.pt")
        torch.save({"state_dict": model.state_dict(), "config": self.config, **extra}, p)
        return p

    def save_array(self, name: str, **arrays) -> str:
        p = os.path.join(self.dir, f"{name}.npz")
        np.savez_compressed(p, **arrays)
        return p

    def finish(self, metrics: dict, **extra) -> None:
        self.metrics.update(metrics)
        self.extra.update(extra)

    def _write(self, status: str) -> None:
        import pandas as pd
        dur = time.time() - (self._t0 or time.time())
        if self.epochs:
            pd.DataFrame(self.epochs).to_csv(os.path.join(self.dir, "epochs.csv"), index=False)
        out = {"run_id": self.run_id, "name": self.name, "group": self.group,
               "seed": self.seed, "status": status, "wall_time_s": round(dur, 2),
               "config_sha256": config_hash(self.config), **self.extra,
               "metrics": self.metrics}
        with open(os.path.join(self.dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=2, default=_jsonable)
        row = {"run_id": self.run_id, "name": self.name, "group": self.group,
               "seed": self.seed, "status": status, "wall_time_s": round(dur, 2),
               "config_sha256": out["config_sha256"][:16],
               **{k: v for k, v in self.metrics.items() if np.isscalar(v)}}
        with open(os.path.join(self.root, "index.jsonl"), "a") as f:
            f.write(json.dumps(row, default=_jsonable) + "\n")


def load_index(root: str | None = None) -> "pd.DataFrame":
    """Every recorded run (one row each), newest last."""
    import pandas as pd
    p = os.path.join(root or output_dir("experiments"), "index.jsonl")
    if not os.path.isfile(p):
        return pd.DataFrame()
    with open(p) as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
