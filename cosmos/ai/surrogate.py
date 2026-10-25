"""A neural surrogate for the COSMOS re-entry propagator (SIMULATED data only).

    python -m cosmos.ai.surrogate generate        # simulator-labelled data, once
    python -m cosmos.ai.surrogate train           # 5-member MLP ensemble
    python -m cosmos.ai.surrogate evaluate        # vs the reference simulator

Why: the Monte Carlo footprint (:func:`cosmos.reentry.monte_carlo_footprint`) spends almost
all its time in :func:`cosmos.reentry.propagator.propagate_ensemble` (about 220 fragment
trajectories per second measured here).  A surrogate that maps the sampled fragment state to
its impact point turns a minutes-long footprint into milliseconds — IF it is accurate where
it is used, which is what this module measures.

Inputs (the quantities the footprint code samples per fragment): break-up altitude, entry
speed, flight-path angle, azimuth (sin, cos), entry latitude (sin, cos), log10 ballistic
coefficient, log10 fragment mass, log10 density multiplier, break-up delta-v in local ENU.
Outputs: downrange and crossrange of the impact relative to the entry point and heading
[km], log fall time, log impact speed, and whether the fragment reaches the ground.

Everything here is SIM: the reference is COSMOS's own propagator, not real re-entries.  The
"out-of-range" split (break-up above the training range) checks whether the ensemble's
disagreement flags extrapolation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from ..common import feature_dir, output_dir

RANGES = {"alt_km": (50.0, 100.0), "speed_ms": (7000.0, 7900.0), "gamma_deg": (-3.0, -0.5),
          "azimuth_deg": (0.0, 360.0), "lat_deg": (-60.0, 60.0), "log_bc": (-0.6, 4.1),
          "log_mass": (-3.0, 2.2), "log_density": (-0.3, 0.3), "dv_ms": (0.0, 60.0)}
OOD_ALT = (100.0, 120.0)
SPLITS = {"train": (400_000, 11), "val": (40_000, 22), "test": (40_000, 33),
          "ood_test": (20_000, 44)}
CHUNK = 5000
INPUTS = ["alt_km", "speed_ms", "gamma_deg", "sin_az", "cos_az", "sin_lat", "cos_lat",
          "log_bc", "log_mass", "log_density", "dv_e", "dv_n", "dv_u"]
TARGETS = ["downrange_km", "crossrange_km", "log_fall_time", "log_speed"]


def sim_dir(*parts) -> str:
    return os.path.join(feature_dir("reentry", "simulated"), "propagator_surrogate_v1", *parts)


def sample_inputs(n: int, rng, alt_range=None) -> pd.DataFrame:
    u = lambda k: rng.uniform(*RANGES[k], n)
    alt = rng.uniform(*(alt_range or RANGES["alt_km"]), n)
    d = pd.DataFrame({"alt_km": alt, "speed_ms": u("speed_ms"), "gamma_deg": u("gamma_deg"),
                      "azimuth_deg": u("azimuth_deg"), "lat_deg": u("lat_deg"),
                      "log_bc": u("log_bc"), "log_mass": u("log_mass"),
                      "log_density": u("log_density")})
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    dv = u("dv_ms")
    d["dv_e"], d["dv_n"], d["dv_u"] = (v * dv[:, None]).T
    return d


def features(d: pd.DataFrame) -> np.ndarray:
    d = d.copy()
    d["sin_az"], d["cos_az"] = np.sin(np.radians(d.azimuth_deg)), np.cos(np.radians(d.azimuth_deg))
    d["sin_lat"], d["cos_lat"] = np.sin(np.radians(d.lat_deg)), np.cos(np.radians(d.lat_deg))
    return d[INPUTS].to_numpy(np.float32)


def _along_cross(lat0, lon0, az_deg, lat, lon):
    from ..common import great_circle_km
    d = great_circle_km(lat0, lon0, lat, lon)
    p1, p2 = np.radians(lat0), np.radians(lat)
    dl = np.radians(lon - lon0)
    brg = np.arctan2(np.sin(dl) * np.cos(p2),
                     np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))
    rel = brg - np.radians(az_deg)
    return d * np.cos(rel), d * np.sin(rel)


def simulate(d: pd.DataFrame) -> pd.DataFrame:
    """Run the reference propagator on every row (one entry state per row)."""
    from ..common import enu_basis
    from ..reentry.drag import SHAPES
    from ..reentry.propagator import entry_state, propagate_ensemble
    n = len(d)
    R, V = np.zeros((n, 3)), np.zeros((n, 3))
    for i, r in enumerate(d.itertuples()):
        st = entry_state(r.alt_km * 1e3, r.speed_ms, r.gamma_deg, r.azimuth_deg, r.lat_deg, 0.0)
        e, nn_, up = enu_basis(r.lat_deg, 0.0)
        R[i], V[i] = st.r, st.v + r.dv_e * e + r.dv_n * nn_ + r.dv_u * up
    bc, mass = 10.0 ** d.log_bc.to_numpy(), 10.0 ** d.log_mass.to_numpy()
    cdh = np.full(n, SHAPES["tumbling"]["cd_hyper"])
    cds = np.full(n, SHAPES["tumbling"]["cd_sub"])
    area = mass / (cdh * bc)
    res = propagate_ensemble(R, V, mass, area, cdh, cds, np.sqrt(area),
                             density_scale=10.0 ** d.log_density.to_numpy(), dt=0.5)
    out = d.copy()
    out["impacted"] = res.impacted
    al, cr = _along_cross(d.lat_deg.to_numpy(), np.zeros(n), d.azimuth_deg.to_numpy(),
                          res.lat, res.lon)
    out["downrange_km"], out["crossrange_km"] = al, cr
    out["log_fall_time"] = np.log10(np.maximum(res.fall_time, 1e-3))
    out["log_speed"] = np.log10(np.maximum(res.speed, 1e-3))
    return out


def _gen_chunk(args):
    split, k, seed = args
    rng = np.random.default_rng([seed, k])
    alt = OOD_ALT if split == "ood_test" else None
    d = sample_inputs(CHUNK, rng, alt)
    return simulate(d)


def generate(workers: int = 96) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from .data.download import file_hashes, utc_now, write_manifest
    os.makedirs(sim_dir(), exist_ok=True)
    files, summary = [], {}
    for split, (n, seed) in SPLITS.items():
        t0 = time.time()
        jobs = [(split, k, seed) for k in range(n // CHUNK)]
        with ProcessPoolExecutor(workers) as ex:
            df = pd.concat(list(ex.map(_gen_chunk, jobs)), ignore_index=True)
        p = sim_dir(f"{split}.parquet")
        df.to_parquet(p, index=False)
        secs = time.time() - t0
        summary[split] = {"n": len(df), "impacted": float(df.impacted.mean()),
                          "generation_s": secs, "seed": seed}
        files.append({"path": f"propagator_surrogate_v1/{split}.parquet", **file_hashes(p),
                      "rows": len(df)})
        print(f"  {split}: {len(df):,} trajectories in {secs:.0f} s", flush=True)
    write_manifest(feature_dir("reentry", "simulated"), {
        "dataset": "reentry_simulated", "class": "SIM", "status": "generated",
        "generator": "cosmos.reentry.propagator.propagate_ensemble (unchanged; dt 0.5 s, "
                     "US76, tumbling drag law) via cosmos.ai.surrogate.simulate",
        "generated_utc": utc_now(), "input_ranges": RANGES, "ood_altitude_km": OOD_ALT,
        "splits": summary,
        "note": "Simulator output only. Never mixed with historical observations."}, files)
    return summary


# ============================================================================ model
def _load(split):
    d = pd.read_parquet(sim_dir(f"{split}.parquet"))
    return d, features(d)


def train(n_members: int = 5, epochs: int = 200, device: str | None = None) -> dict:
    import torch
    from torch import nn
    from .experiment import Run, count_parameters, pick_device, set_seed
    device = device or pick_device()
    dtr, Xtr = _load("train")
    dva, Xva = _load("val")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m_tr, m_va = dtr.impacted.to_numpy(), dva.impacted.to_numpy()
    Ytr = dtr[TARGETS].to_numpy(np.float32)
    ymu, ysd = Ytr[m_tr].mean(0), Ytr[m_tr].std(0) + 1e-6
    norm_x = lambda X: torch.as_tensor((X - mu) / sd, device=device)
    norm_y = lambda Y: torch.as_tensor((Y - ymu) / ysd, device=device)
    Xt, Xv = norm_x(Xtr), norm_x(Xva)
    Yt, Yv = norm_y(np.nan_to_num(Ytr)), norm_y(np.nan_to_num(dva[TARGETS].to_numpy(np.float32)))
    It = torch.as_tensor(m_tr.astype(np.float32), device=device)
    Iv = torch.as_tensor(m_va.astype(np.float32), device=device)
    ckdir = sim_dir("checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    infos = []
    with Run("reentry_surrogate_ensemble", {"members": n_members, "epochs": epochs,
                                             "inputs": INPUTS, "targets": TARGETS}, 0, device,
             group="reentry_surrogate") as run:
        t0 = time.time()
        for k in range(n_members):
            set_seed(k)
            net = nn.Sequential(nn.Linear(len(INPUTS), 256), nn.SiLU(), nn.Linear(256, 256),
                                nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256),
                                nn.SiLU(), nn.Linear(256, len(TARGETS) + 1)).to(device)
            opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            best, state, bad = np.inf, None, 0
            g = torch.Generator(device="cpu").manual_seed(k)
            for ep in range(epochs):
                net.train()
                perm = torch.randperm(len(Xt), generator=g).to(device)
                for i in range(0, len(perm), 2048):
                    b = perm[i:i + 2048]
                    o = net(Xt[b])
                    reg = (((o[:, :-1] - Yt[b]) ** 2).mean(1) * It[b]).sum() / It[b].sum().clamp(min=1)
                    cls = nn.functional.binary_cross_entropy_with_logits(o[:, -1], It[b])
                    loss = reg + 0.1 * cls
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    o = net(Xv)
                    v = float(((((o[:, :-1] - Yv) ** 2).mean(1) * Iv).sum() / Iv.sum()))
                if v < best - 1e-6:
                    best, bad = v, 0
                    state = {a: b.detach().clone() for a, b in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 20:
                        break
                if ep % 20 == 0:
                    run.log_epoch(member=k, epoch=ep, val_mse_norm=v)
            net.load_state_dict(state)
            torch.save({"state_dict": state, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd},
                       os.path.join(ckdir, f"member_{k}.pt"))
            infos.append({"member": k, "val_mse_norm": best, "epochs": ep + 1})
            print(f"  member {k}: val MSE (normalised) {best:.5f} after {ep + 1} epochs",
                  flush=True)
        run.finish({"train_time_s": time.time() - t0, "params_per_member": count_parameters(net)},
                   members=infos)
    return {"members": infos}


class Surrogate:
    """Load the ensemble; ``predict(df)`` -> mean and std of the targets, P(impact)."""

    def __init__(self, device: str = "cpu"):
        import glob
        import torch
        from torch import nn
        self.device = device
        self.nets, self.norm = [], None
        for p in sorted(glob.glob(sim_dir("checkpoints", "member_*.pt"))):
            ck = torch.load(p, map_location=device, weights_only=False)
            net = nn.Sequential(nn.Linear(len(INPUTS), 256), nn.SiLU(), nn.Linear(256, 256),
                                nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256),
                                nn.SiLU(), nn.Linear(256, len(TARGETS) + 1)).to(device)
            net.load_state_dict(ck["state_dict"])
            net.eval()
            self.nets.append(net)
            self.norm = (ck["mu"], ck["sd"], ck["ymu"], ck["ysd"])

    def predict_features(self, X: np.ndarray):
        import torch
        mu, sd, ymu, ysd = self.norm
        x = torch.as_tensor((X - mu) / sd, device=self.device)
        with torch.no_grad():
            outs = torch.stack([n(x) for n in self.nets]).cpu().numpy()
        y = outs[:, :, :-1] * ysd + ymu
        p = 1 / (1 + np.exp(-outs[:, :, -1]))
        return y.mean(0), y.std(0), p.mean(0)

    def predict(self, d: pd.DataFrame):
        return self.predict_features(features(d))

    def spread_threshold(self, quantile: float = 0.95) -> float:
        """Ensemble-spread level that flags inputs as unusual: the ``quantile`` of the spread
        on the in-distribution VALIDATION split (a 5 % false-flag rate by construction).
        Cached in ``spread_threshold.json``."""
        p = sim_dir("spread_threshold.json")
        if os.path.isfile(p):
            with open(p) as f:
                d = json.load(f)
            if d.get("quantile") == quantile:
                return float(d["km"])
        _, Xv = _load("val")
        _m, std, _p = self.predict_features(Xv)
        km = float(np.quantile(np.hypot(std[:, 0], std[:, 1]), quantile))
        with open(p, "w") as f:
            json.dump({"quantile": quantile, "km": km, "source": "val split ensemble spread"}, f)
        return km


def evaluate(device: str | None = None) -> dict:
    from .experiment import pick_device
    device = device or pick_device()
    S = Surrogate(device)
    out = {}
    for split in ("test", "ood_test"):
        d, X = _load(split)
        t0 = time.time()
        mean, std, p = S.predict_features(X)
        t_pred = time.time() - t0
        m = d.impacted.to_numpy()
        err = np.hypot(mean[m, 0] - d.downrange_km.to_numpy()[m],
                       mean[m, 1] - d.crossrange_km.to_numpy()[m])
        rel_dr = np.abs(mean[m, 0] - d.downrange_km.to_numpy()[m]) / np.maximum(
            np.abs(d.downrange_km.to_numpy()[m]), 1.0)
        unc = np.hypot(std[m, 0], std[m, 1])
        out[split] = {"n": int(len(d)), "impacted_fraction": float(m.mean()),
                      "impact_classification_accuracy": float(((p > 0.5) == m).mean()),
                      "position_error_km_median": float(np.median(err)),
                      "position_error_km_p95": float(np.percentile(err, 95)),
                      "downrange_relative_error_median": float(np.median(rel_dr)),
                      "fall_time_mae_s": float(np.mean(np.abs(10 ** mean[m, 2]
                                                              - 10 ** d.log_fall_time.to_numpy()[m]))),
                      "ensemble_std_km_median": float(np.median(unc)),
                      "error_vs_std_spearman": float(pd.Series(err).corr(pd.Series(unc),
                                                                         method="spearman")),
                      "surrogate_samples_per_s": float(len(d) / max(t_pred, 1e-9))}
    # OOD detection: does ensemble disagreement separate out-of-range inputs?
    from sklearn.metrics import roc_auc_score
    _, Xi = _load("test")
    _, Xo = _load("ood_test")
    _, si, _ = S.predict_features(Xi)
    _, so, _ = S.predict_features(Xo)
    ui, uo = np.hypot(si[:, 0], si[:, 1]), np.hypot(so[:, 0], so[:, 1])
    out["ood_auroc_ensemble_std"] = float(roc_auc_score(np.r_[np.zeros(len(ui)), np.ones(len(uo))],
                                                        np.r_[ui, uo]))
    out["footprints"] = footprint_check(S)
    d = output_dir("benchmarks", "reentry_surrogate")
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def footprint_check(S: "Surrogate", n_scen: int = 12, n_samples: int = 3000) -> list:
    """Whole Monte Carlo footprints: reference propagator vs surrogate, same sampled inputs.

    Scenarios mirror :func:`cosmos.reentry.monte_carlo_footprint`: a break-up cloud from the
    NASA SBM, the hazard's ballistic-coefficient spread, density and flight-path noise.
    Compared: median downrange, CEP50 around the footprint centre, and whether a 95 %
    split-conformal radius fitted on SURROGATE impacts covers the REFERENCE impacts.
    """
    from ..reentry.breakup import sample_cloud
    rows = []
    for k in range(n_scen):
        rng = np.random.default_rng(1000 + k)
        cloud = sample_cloud(parent_mass=float(rng.uniform(500, 3000)),
                             kind=["explosion", "aerodynamic"][k % 2], rng=k)
        frag = cloud.frame.sample(n=n_samples, replace=True, random_state=k)
        bc = frag.bc.to_numpy() * 10 ** rng.normal(0, rng.uniform(0.16, 0.29), n_samples)
        d = pd.DataFrame({"alt_km": np.full(n_samples, rng.uniform(55, 92)),
                          "speed_ms": np.full(n_samples, 7600.0),
                          "gamma_deg": -1.4 + rng.normal(0, 0.15, n_samples),
                          "azimuth_deg": np.full(n_samples, rng.uniform(0, 360)),
                          "lat_deg": np.full(n_samples, rng.uniform(-50, 50)),
                          "log_bc": np.log10(np.clip(bc, 10 ** RANGES["log_bc"][0],
                                                     10 ** RANGES["log_bc"][1])),
                          "log_mass": np.log10(np.clip(frag.mass_kg.to_numpy(), 1e-3, 150.0)),
                          "log_density": rng.normal(0, 0.08, n_samples)})
        v = rng.normal(size=(n_samples, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        dv = np.clip(frag.dv_ms.to_numpy(), 0, 60)
        d["dv_e"], d["dv_n"], d["dv_u"] = (v * dv[:, None]).T
        t0 = time.time()
        ref = simulate(d)
        t_ref = time.time() - t0
        t0 = time.time()
        mean, _std, _p = S.predict(d)
        t_sur = time.time() - t0
        m = ref.impacted.to_numpy()
        R = ref[["downrange_km", "crossrange_km"]].to_numpy()[m]
        P = mean[m, :2]
        cen_r, cen_p = np.median(R, 0), np.median(P, 0)
        cep_r = np.median(np.hypot(*(R - cen_r).T))
        cep_p = np.median(np.hypot(*(P - cen_p).T))
        half = len(P) // 2
        from .uncertainty import split_conformal_quantile
        rad = split_conformal_quantile(np.hypot(*(P[:half] - cen_p).T), 0.05)
        cover = float((np.hypot(*(R[half:] - cen_p).T) <= rad).mean())
        rows.append({"scenario": k, "alt_km": float(d.alt_km.iloc[0]),
                     "median_downrange_ref_km": float(cen_r[0]),
                     "median_downrange_sur_km": float(cen_p[0]),
                     "cep50_ref_km": float(cep_r), "cep50_sur_km": float(cep_p),
                     "conformal95_radius_from_surrogate_km": float(rad),
                     "reference_coverage_of_surrogate_zone": cover,
                     "reference_s": t_ref, "surrogate_s": t_sur,
                     "speedup": t_ref / max(t_sur, 1e-9)})
        print(f"  scenario {k}: CEP50 ref {cep_r:.1f} km sur {cep_p:.1f} km, coverage {cover:.3f}, "
              f"speed-up {t_ref / max(t_sur, 1e-9):.0f}x", flush=True)
    return rows


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "evaluate"
    if cmd == "generate":
        print(json.dumps(generate(), indent=1))
    elif cmd == "train":
        print(json.dumps(train(), indent=1))
    else:
        r = evaluate()
        print(json.dumps({k: v for k, v in r.items() if k != "footprints"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
