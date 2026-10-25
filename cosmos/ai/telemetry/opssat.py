"""OPSSAT-AD study: classical, deep and foundation-model fragment classifiers on one protocol.

    python -m cosmos.ai.telemetry.opssat --config configs/telemetry/opssat.yaml

Protocol (fixed before any test number was seen)
------------------------------------------------
* **Official split** — the ``train`` flag of ``dataset.csv`` (1,594 train / 529 test).
* **Validation** — a stratified 20 % of the official TRAIN set (seed 2137, the official seed)
  is held out for early stopping, epoch selection and probe regularisation.  The test set is
  scored once per trained model.
* **Decision rule** — supervised models predict ``P >= 0.5``; nothing is tuned on test.
  Unsupervised / semi-supervised baselines get the threshold stated next to them.
* **Seeds** — 5 per stochastic model; deterministic models run once.
* **Extra protocols** — label efficiency (10/25/50/100 % of the training fragments) and a
  time-blocked split (train on fragments that start before the 75th time percentile, test
  on the rest), because the official split interleaves train and test fragments in time.

Every number is written to ``outputs/benchmarks/telemetry_opssat/``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ...common import output_dir
from ..data.preprocess_telemetry import (OPSSAT_FEATURES, OPSSAT_SEED, load_opssat_ad,
                                         opssat_tensors)
from ..evaluation import classification_metrics, expected_calibration_error, brier_score
from ..experiment import Run, count_parameters, model_size_mb, pick_device, set_seed

DEEP = ("fcnn", "cnn", "transformer", "hybrid")


# ============================================================================ data
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class OpssatData:
    """Aligned features, raw tensors and labels, with the official and extra splits."""

    def __init__(self, length: int = 256):
        d = load_opssat_ad()
        f = d["features"].sort_index()
        X, mask, ids = opssat_tensors(d["segments"], length)
        assert np.array_equal(ids, f.index.to_numpy()), "segment order mismatch"
        self.feats = f
        self.F = signed_log(f[OPSSAT_FEATURES].to_numpy(np.float64))
        self.X, self.mask = X, mask
        self.y = f.anomaly.to_numpy(int)
        self.train_official = f.train.to_numpy(int) == 1
        self.segments = d["segments"]
        self.start = d["segments"].groupby("segment").timestamp.min().sort_index()

    def split(self, kind: str = "official", seed: int = OPSSAT_SEED, val_frac: float = 0.2):
        """Index arrays ``(train, val, test)``; validation is always carved from train."""
        from sklearn.model_selection import train_test_split
        n = len(self.y)
        if kind == "official":
            tr_all = np.flatnonzero(self.train_official)
            te = np.flatnonzero(~self.train_official)
        elif kind == "time_blocked":
            cut = self.start.quantile(0.75)
            tr_all = np.flatnonzero((self.start < cut).to_numpy())
            te = np.flatnonzero((self.start >= cut).to_numpy())
        else:
            raise KeyError(kind)
        tr, va = train_test_split(tr_all, test_size=val_frac, stratify=self.y[tr_all],
                                  random_state=seed)
        assert len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
        assert len(tr) + len(va) + len(te) == n
        return np.sort(tr), np.sort(va), np.sort(te)


def subsample(idx, y, frac: float, seed: int):
    if frac >= 1.0:
        return idx
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(idx, train_size=frac, stratify=y[idx], random_state=seed)
    return np.sort(keep)


# ============================================================================ classical
def classical_scores(kind: str, D: OpssatData, tr, va, te, seed: int) -> dict:
    """Fit a classical model; return validation/test scores and binary predictions."""
    from sklearn.preprocessing import StandardScaler
    nominal = tr[D.y[tr] == 0]
    out = {}
    if kind == "cosmos_robust_z":
        # the COSMOS monitor's statistic on fragments: robust z per feature (median / MAD of
        # the NOMINAL training fragments), score = max |z|.  Threshold: the 95th percentile of
        # the nominal training scores, i.e. a 5 % false-alarm budget set without anomalies.
        med = np.median(D.F[nominal], 0)
        mad = 1.4826 * np.median(np.abs(D.F[nominal] - med), 0)
        sig = np.maximum(mad, 1e-9)
        score = np.abs((D.F - med) / sig).max(1)
        thr = np.quantile(score[nominal], 0.95)
        out = {"score": score, "pred": (score > thr).astype(int), "threshold": float(thr)}
    elif kind == "iforest":
        # the official notebook's unsupervised example: scaler fitted on nominal training
        # fragments, IForest on all (unlabelled) training fragments, default contamination
        from sklearn.ensemble import IsolationForest
        sc = StandardScaler().fit(D.F[nominal])
        m = IsolationForest(random_state=seed, contamination=0.1).fit(sc.transform(D.F[tr]))
        Z = sc.transform(D.F)
        out = {"score": -m.score_samples(Z), "pred": (m.predict(Z) == -1).astype(int)}
    else:
        sc = StandardScaler().fit(D.F[nominal])
        Z = sc.transform(D.F)
        if kind == "adaboost":                               # the official supervised example
            from sklearn.ensemble import AdaBoostClassifier
            m = AdaBoostClassifier(random_state=seed).fit(Z[tr], D.y[tr])
        elif kind == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(500, random_state=seed, n_jobs=16).fit(Z[tr], D.y[tr])
        elif kind == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=seed,
                              n_jobs=16, eval_metric="aucpr", early_stopping_rounds=50)
            m.fit(Z[tr], D.y[tr], eval_set=[(Z[va], D.y[va])], verbose=False)
        elif kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=5000, C=1.0).fit(Z[tr], D.y[tr])
        else:
            raise KeyError(kind)
        p = m.predict_proba(Z)[:, 1]
        out = {"score": p, "pred": (p >= 0.5).astype(int), "prob": p}
    return out


# ============================================================================ deep
def _batches(idx, bs, rng=None):
    idx = np.asarray(idx)
    if rng is not None:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def train_deep(kind: str, D: OpssatData, tr, va, seed: int, device: str, cfg: dict,
               run: Run | None = None):
    """Train with early stopping on validation AUC-PR; return the best model and its scaler."""
    import torch
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    from .models import build_classifier
    set_seed(seed)
    sc = StandardScaler().fit(D.F[tr])
    Ft = torch.tensor(sc.transform(D.F), dtype=torch.float32, device=device)
    Xt = torch.tensor(D.X, device=device)
    Mt = torch.tensor(D.mask, device=device)
    yt = torch.tensor(D.y, dtype=torch.float32, device=device)
    kw = {"fcnn": dict(n_features=len(OPSSAT_FEATURES)),
          "cnn": dict(in_ch=D.X.shape[1], width=cfg.get("width", 64)),
          "transformer": dict(in_ch=D.X.shape[1], length=D.X.shape[2]),
          "hybrid": dict(in_ch=D.X.shape[1], n_features=len(OPSSAT_FEATURES),
                         width=cfg.get("width", 64))}[kind]
    model = build_classifier(kind, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, cfg.get("max_epochs", 300) + 1):
        model.train()
        tot = 0.0
        for b in _batches(tr, cfg.get("batch", 64), rng):
            bi = torch.as_tensor(b, device=device)
            x = Xt[bi]
            if cfg.get("augment", True) and kind != "fcnn":
                # physically harmless augmentation: amplitude jitter of the z-scored value
                x = x.clone()
                x[:, 0] = x[:, 0] * (1 + 0.05 * torch.randn(len(b), 1, device=device))
            logit = model(x, Ft[bi], Mt[bi])
            loss = lossf(logit, yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        pv = predict_deep(model, D, va, Xt, Ft, Mt)
        ap = average_precision_score(D.y[va], pv)
        if run is not None:
            run.log_epoch(epoch=ep, train_loss=tot / len(tr), val_auc_pr=ap)
        if ap > best + 1e-6:
            best, best_ep, bad = ap, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get("patience", 30):
                break
    model.load_state_dict(best_state)
    return model, (Xt, Ft, Mt), {"best_epoch": best_ep, "val_auc_pr": best,
                                 "epochs_run": ep, "train_time_s": time.time() - t0}


def predict_deep(model, D, idx, Xt, Ft, Mt, bs: int = 512):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(idx, bs):
            bi = torch.as_tensor(b, device=Xt.device)
            out.append(torch.sigmoid(model(Xt[bi], Ft[bi], Mt[bi])).cpu().numpy())
    return np.concatenate(out)


# ============================================================================ foundation
def chronos_embeddings(D: OpssatData, repo: str, device: str, batch: int = 256) -> np.ndarray:
    """Frozen Chronos-Bolt encoder embedding per fragment (mean over patch tokens).

    Fragments are passed at native length (no interpolation); Chronos left-pads internally.
    Embeddings are cached under ``data/feature1_health/opssat_ad/processed/``.
    """
    import torch
    from chronos import BaseChronosPipeline
    from ..data.preprocess_telemetry import opssat_dir
    cache = opssat_dir("processed", f"emb_{repo.replace('/', '__')}.npy")
    if os.path.isfile(cache):
        return np.load(cache)
    pipe = BaseChronosPipeline.from_pretrained(repo, device_map=device,
                                               torch_dtype=torch.float32)
    series = [torch.tensor(g.value.to_numpy(np.float32))
              for _, g in D.segments.groupby("segment", sort=True)]
    out = []
    for i in range(0, len(series), batch):
        emb, _ = pipe.embed(series[i:i + batch])
        e = emb.float()
        # the last token is the [REG] summary token; average the patch tokens with it
        out.append(e.nanmean(1).cpu().numpy())
    E = np.concatenate(out).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, E)
    return E


def probe_scores(E: np.ndarray, D: OpssatData, tr, va, seed: int) -> dict:
    """Logistic-regression probe on frozen embeddings; C chosen on validation AUC-PR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(E[tr])
    Z = sc.transform(E)
    best, best_c = -1, None
    for c in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = LogisticRegression(C=c, max_iter=5000, random_state=seed).fit(Z[tr], D.y[tr])
        ap = average_precision_score(D.y[va], m.predict_proba(Z[va])[:, 1])
        if ap > best:
            best, best_c = ap, c
    m = LogisticRegression(C=best_c, max_iter=5000, random_state=seed).fit(Z[tr], D.y[tr])
    p = m.predict_proba(Z)[:, 1]
    return {"score": p, "pred": (p >= 0.5).astype(int), "prob": p, "C": best_c,
            "val_auc_pr": best}


# ============================================================================ study
def _row(model, protocol, frac, seed, D, te, res, extra=None):
    m = classification_metrics(D.y[te], res["pred"][te], res["score"][te])
    if "prob" in res:
        m["brier"] = brier_score(D.y[te], res["prob"][te])
        m["ece"] = expected_calibration_error(D.y[te], res["prob"][te])
    return {"model": model, "protocol": protocol, "train_frac": frac, "seed": seed,
            "n_train": None, "n_test": int(len(te)), **m, **(extra or {})}


def run_study(cfg: dict) -> pd.DataFrame:
    import torch
    out_dir = output_dir("benchmarks", "telemetry_opssat")
    D = OpssatData(cfg.get("length", 256))
    device = pick_device(index=cfg.get("device_index", 0))
    seeds = cfg.get("seeds", [0, 1, 2, 3, 4])
    rows, preds = [], {}
    tag_all = cfg.get("tag", "all")
    protocols = cfg.get("protocols", ["official"])
    fracs = cfg.get("train_fracs", [1.0])
    E = None
    if cfg.get("foundation"):
        E = {repo: chronos_embeddings(D, repo, device) for repo in cfg["foundation"]}
    for protocol in protocols:
        tr0, va, te = D.split(protocol)
        for frac in fracs:
            if protocol != "official" and frac < 1.0:
                continue
            for model in cfg["models"]:
                if frac < 1.0 and model not in cfg.get("label_efficiency_models", []):
                    continue
                det = model in ("cosmos_robust_z", "logreg")
                for seed in ([0] if det else seeds):
                    tr = subsample(tr0, D.y, frac, seed + 101)
                    tag = f"opssat_{model}_{protocol}_f{int(100 * frac)}"
                    if model in DEEP:
                        with Run(tag, {**cfg.get("deep", {}), "model": model,
                                       "protocol": protocol, "train_frac": frac,
                                       "n_train": len(tr)}, seed, device,
                                 group="opssat") as run:
                            net, tensors, info = train_deep(model, D, tr, va, seed, device,
                                                            cfg.get("deep", {}), run)
                            p = predict_deep(net, D, np.arange(len(D.y)), *tensors)
                            res = {"score": p, "pred": (p >= 0.5).astype(int), "prob": p}
                            from ..evaluation import measure_inference
                            Xt, Ft, Mt = tensors
                            bi = torch.arange(min(256, len(te)), device=device)
                            eff = measure_inference(lambda b: net(Xt[b], Ft[b], Mt[b]), bi,
                                                    device)
                            extra = {**info, "params": count_parameters(net),
                                     "size_mb": model_size_mb(net),
                                     "latency_ms_per_fragment": eff["latency_ms_per_item"],
                                     "throughput_fragments_s": eff["throughput_items_s"],
                                     "peak_memory_mb": eff.get("peak_memory_mb")}
                            r = _row(model, protocol, frac, seed, D, te, res, extra)
                            run.save_checkpoint(net, "best", scaler_note="StandardScaler on "
                                                "signed-log features of the train fold")
                            run.finish({k: v for k, v in r.items() if k not in
                                        ("model", "protocol")}, run_kind="opssat")
                            r["run_id"] = run.run_id
                    elif model.startswith("chronos:"):
                        repo = model.split(":", 1)[1]
                        t0 = time.time()
                        res = probe_scores(E[repo], D, tr, va, seed)
                        r = _row(model, protocol, frac, seed, D, te, res,
                                 {"probe_C": res["C"], "val_auc_pr": res["val_auc_pr"],
                                  "train_time_s": time.time() - t0})
                    else:
                        t0 = time.time()
                        res = classical_scores(model, D, tr, va, te, seed)
                        r = _row(model, protocol, frac, seed, D, te, res,
                                 {"train_time_s": time.time() - t0})
                    r["n_train"] = int(len(tr))
                    rows.append(r)
                    preds[f"{protocol}|{frac}|{model}|{seed}"] = res["score"][te]
                    preds[f"{protocol}|test_index"] = te
                    preds[f"{protocol}|y"] = D.y[te]
                    print(f"  {protocol:12s} f={frac:4.2f} {model:28s} seed {seed}: "
                          f"F1 {r['f1']:.3f}  AUC-PR {r['auc_pr']:.3f}  MCC {r['mcc']:.3f}",
                          flush=True)
                    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"per_run_{tag_all}.csv"),
                                              index=False)
    df = pd.DataFrame(rows)
    np.savez_compressed(os.path.join(out_dir, f"test_scores_{tag_all}.npz"), **preds)
    summarise(df, out_dir, tag_all)
    return df


def summarise(df: pd.DataFrame, out_dir: str, tag: str = "all") -> pd.DataFrame:
    cols = ["accuracy", "precision", "recall", "f1", "mcc", "auc_pr", "auc_roc",
            "false_alarm_rate", "brier", "ece"]
    cols = [c for c in cols if c in df]
    g = df.groupby(["protocol", "train_frac", "model"])
    s = g[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_seeds"] = g.size()
    for c in ("params", "latency_ms_per_fragment", "train_time_s"):
        if c in df:
            s[c] = g[c].mean()
    s = s.reset_index()
    s.to_csv(os.path.join(out_dir, f"summary_{tag}.csv"), index=False)
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(s.to_dict("records"), f, indent=2, default=float)
    return s


def main(argv=None) -> int:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/telemetry/opssat.yaml")
    ap.add_argument("--protocol", default=None, help="run only this protocol")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.protocol:
        cfg["protocols"], cfg["tag"] = [a.protocol], a.protocol
    run_study(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
