"""NASA SMAP/MSL telemetry benchmark (Hundman et al., KDD 2018) with honest metrics.

    python -m cosmos.ai.telemetry.smap_msl [--out DIR] [--channels A-1,P-2,...]

Label rows are merged by channel (P-2 has two overlapping rows in labeled_anomalies.csv; its
sequences are united), so each of the 81 distinct channels is evaluated once.  Per channel, a
detector is fitted on the (anomaly-free) training series and scores every test step.  Alarms
come from conformal p-values against the channel's own training scores, with the level and
persistence fixed in advance (alpha = 1e-3, 3 consecutive steps; no test labels are used for
any choice).  An alarm is raised at the 3rd consecutive hot step and never backdated.

Calibration scores are out of fold: the training windows are cut into 5 contiguous blocks and
each block is scored by the detector fitted on the other windows, minus W windows on each side
of the block, so no fitted window shares a time step with a scored one.  Test scores come from
the detector refitted on the whole training series.  This is a cross-conformal approximation
(Vovk 2015), not exact split conformal: the calibration scores come from five fold models, none
of which is the test model.  Scores are floored at EPS = 1e-6 (data units), so the rounding
residue (~1e-16) of a fit to a constant series cannot raise an alarm; EPS lies far below the
alarm threshold of every channel whose calibration scores are not degenerate.  Channels whose
calibration scores have (near-)zero spread (sd < EPS: for ar_ridge a constant training target,
channel 0; for pca and knn a constant training series) alarm on any test deviation above EPS;
they are listed in config.json and counted in summary.csv.  The smallest p-value is
1 / (n_cal + 1), so each channel uses alpha_eff = max(alpha, 1 / (n_cal + 1)) (in
per_channel.csv); a channel with n_cal < 999 alarms only when its score exceeds every
calibration score.  Some channels change level or slope between their training and test series
(A-1 is 0.999 throughout training and 1.0 in test), so a detector can stay in alarm; a channel
in alarm on more than 90% of its test steps is flagged saturated (label-free; at most 51% of
any channel's test steps are labelled) and listed in config.json.  Detectors:

* ``ar_ridge``     ridge autoregression of the telemetry value (channel 0) on the previous 30
                   steps of all inputs; score = |residual|
* ``pca``          reconstruction error (Euclidean) of 30-step windows, 10 principal components
                   (exact SVD)
* ``knn``          distance of each 30-step window to its nearest training window (subsampled)
* ``chronos2``     Chronos-2 (Ansari et al. 2025; amazon/chronos-2, 120M, Apache-2.0), a pretrained
                   time-series foundation model used zero-shot: rolling 8-step forecasts of the
                   telemetry value (channel 0) from up to 512 past steps (training tail included
                   for the first test steps); score = |value - forecast median|.  It is never
                   fitted to these series, so its training-series scores are out of sample and
                   serve directly as the calibration set
* ``telemanom``    the smoothed prediction errors published with the original telemanom run
                   (archive folder 2018-05-19_15.00.10), i.e. the published LSTM detector; it has
                   no training scores, so it enters the point-wise metrics only
* ``random``       uniform scores, seeded per channel: chance level of the point-wise metrics
                   (at alpha_eff with 3 consecutive steps it essentially never alarms)
* ``always_on``    reference, not a detector: in alarm on every test step (event metrics only)

Event metrics on all test steps, pooled over the channels of a spacecraft, over all channels and
over three channel sets common to every detector: the channels where alpha itself is attainable
for every detector (suffix ``_attainable``), the channels no detector saturates
(``_unsaturated``) and, of those, the channels where no detector has near-zero calibration
spread (``_nondegenerate``).  A sequence is detected if an alarm falls inside it; alarm runs are
split at the sequence boundaries and every maximal alarm segment outside all sequences is one
false positive; precision, recall, F1 and median detection delay.  Also the ESA-ADB corrected
event-wise precision of esa_metrics.ESAScores, computed on steps (TP / (TP + alarm runs touching
no sequence), times the point-wise true-negative rate on nominal steps), with its F0.5 and F1,
and the fraction of nominal time in alarm.

Neither score shows skill by itself (always_on keeps a high event F1, one long alarm block a
high corrected F0.5), and the labels are not placed at random: most labelled steps lie in the
second half of their channel's test series (share in config.json), so late alarms (e.g. from
drift between training and test) score well without detecting any particular event.  Each
detector is compared with label-free surrogates built from its own alarms, 200 draws each, by
the one-sided permutation p-value (1 + #{surrogate >= detector}) / 201.  ``_shift`` circularly
shifts its alarms by a uniform offset (same runs) and ``_block`` is one block of its alarm time
at a uniform start; both are position-uniform, so beating them can come from late alarms alone.
``_swap`` reassigns the detector's alarms among the channels of the spacecraft by a uniform
random permutation, each stretched to its new channel's length: it keeps the detector's alarm
positions and breaks only their link to the channel's own sequences, so it is the null for
channel-specific detection.  ``_late`` (one draw; a reference, not a null) is one block of the
detector's alarm time at the end of every test series.

Point-wise metrics, for every detector on the steps telemanom scores (250 .. n_test - 10):
``pointwise_auc_pr`` is the mean over channels of the per-channel AUC-PR (no point adjustment;
scale-free); ``pointwise_auc_pr_pooled`` is one AUC-PR over the pooled steps after each
channel's scores are mapped to conformal p-values against its out-of-fold training scores (a
bounded common scale; steps above every calibration score tie at p = 1 / (n_cal + 1), so a
saturated or near-zero-spread channel fills the top of the pooled ranking, ordered by n_cal).
Telemanom has no training scores: ``pointwise_auc_pr_pooled_raw`` pools its raw errors (the
published convention).  Point-adjusted F1, on the same steps, is for contrast only.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ...common import feature_dir, output_dir
from ..uncertainty import conformal_pvalues
from .alarms import runs

W = 30
ALPHA, K = 1e-3, 3
FOLDS = 5
EPS = 1e-6
SAT = 0.9                    # a channel in alarm on more than this fraction of its test steps is saturated
DRAWS = 200                  # label-free surrogate alarms per channel and detector
SCORED = ("ar_ridge", "pca", "knn", "chronos2", "random")
CHRONOS = {"model": "amazon/chronos-2", "context": 512, "min_context": 64, "horizon": 8, "batch": 256}
_PIPE = {}
NAMES = {"telemanom": "telemanom (published scores)", "always_on": "always_on (reference)"}


def root():
    return feature_dir("health", "smap_msl", "raw", "extracted")


def merge(seqs):
    """Union of closed intervals [a, b], merged where they overlap."""
    out = []
    for a, b in sorted(seqs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def channels():
    """One row per distinct channel; its label rows are merged."""
    lab = pd.read_csv(os.path.join(root(), "labeled_anomalies.csv"))
    lab["seq"] = lab["anomaly_sequences"].map(ast.literal_eval)
    return pd.DataFrame([{"chan_id": c, "spacecraft": g.spacecraft.iloc[0], "label_rows": len(g),
                          "seq": merge(sum(g.seq, []))} for c, g in lab.groupby("chan_id", sort=False)])


def load(chan):
    d = os.path.join(root(), "data", "data")
    return np.load(os.path.join(d, "train", f"{chan}.npy")), np.load(os.path.join(d, "test", f"{chan}.npy"))


def windows(x, w=W):
    n = len(x) - w + 1
    idx = np.arange(w)[None, :] + np.arange(n)[:, None]
    return x[idx].reshape(n, -1)


def cross_fit(fit, A, B, folds=FOLDS, gap=W):
    """Out-of-fold scores of the training rows ``A`` and full-fit scores of the test rows ``B``.

    ``fit(rows)`` returns a scoring function.  Block [a, b) of ``A`` is scored by the model
    fitted on the rows outside [a - gap, b + gap)."""
    n, s = len(A), np.empty(len(A))
    e = np.linspace(0, n, folds + 1).round().astype(int)
    for a, b in zip(e[:-1], e[1:]):
        s[a:b] = fit(np.concatenate([A[:max(a - gap, 0)], A[b + gap:]]))(A[a:b])
    return s, fit(A)(B)


def score_ar(tr, te):
    from sklearn.linear_model import Ridge
    c = W * tr.shape[1]                          # a row: 30 steps of all inputs, then the target step

    def fit(Z):
        m = Ridge(alpha=1.0).fit(Z[:, :c], Z[:, c])
        return lambda Z: np.abs(Z[:, c] - m.predict(Z[:, :c]))
    return cross_fit(fit, windows(tr, W + 1), windows(np.vstack([tr[-W:], te]), W + 1))


def score_pca(tr, te, k=10):
    from sklearn.decomposition import PCA

    def fit(A):
        with np.errstate(invalid="ignore", divide="ignore"):         # constant training windows
            p = PCA(n_components=min(k, A.shape[1], A.shape[0] - 1), svd_solver="full").fit(A)
        return lambda M: np.sqrt(np.sum((M - p.inverse_transform(p.transform(M))) ** 2, 1))
    return cross_fit(fit, windows(tr), windows(np.vstack([tr[-(W - 1):], te])))


def score_knn(tr, te, n_ref=4000, seed=0):
    from sklearn.neighbors import NearestNeighbors

    def fit(A):
        ref = A[np.random.default_rng(seed).choice(len(A), size=min(n_ref, len(A)), replace=False)]
        nn = NearestNeighbors(n_neighbors=1).fit(ref)
        return lambda M: nn.kneighbors(M, 1)[0][:, 0]
    return cross_fit(fit, windows(tr), windows(np.vstack([tr[-(W - 1):], te])))


def score_chronos2(tr, te):
    """Zero-shot Chronos-2 forecast residuals of channel 0 on the training and test series."""
    import torch
    if "p" not in _PIPE:
        from chronos import Chronos2Pipeline
        from ..experiment import pick_device
        _PIPE["p"] = Chronos2Pipeline.from_pretrained(CHRONOS["model"], device_map=pick_device(), dtype=torch.float32)
    x = np.concatenate([tr[:, 0], te[:, 0]]).astype(np.float32)
    C, H, n0 = CHRONOS["context"], CHRONOS["horizon"], CHRONOS["min_context"]
    starts = np.arange(n0, len(x), H)
    s = np.full(len(x), np.nan)
    for i in range(0, len(starts), CHRONOS["batch"]):
        b = starts[i:i + CHRONOS["batch"]]
        q, _ = _PIPE["p"].predict_quantiles([x[max(0, t - C):t][None, :] for t in b], prediction_length=H, quantile_levels=[0.5])
        for t, qq in zip(b, q):
            m = qq[0, :, 0].float().cpu().numpy()[:len(x) - t]
            s[t:t + len(m)] = np.abs(x[t:t + len(m)] - m)
    return s[:len(tr)], s[len(tr):]


def score_telemanom(chan, n_test):
    p = os.path.join(root(), "data", "data", "2018-05-19_15.00.10", "smoothed_errors", f"{chan}.npy")
    if not os.path.isfile(p):
        return None
    e = np.load(p).ravel()
    out = np.full(n_test, np.nan)
    l_s = 250                                    # telemanom params.log: l_s = 250, n_predictions = 10;
    out[l_s:l_s + len(e)] = e[:n_test - l_s]     # error i belongs to test step i + l_s
    return out


def conformal_alarms(s_tr, s_te, alpha=ALPHA, k=K):
    """``(alarm, alpha_eff, p)``: the alarm is on from the k-th consecutive step with p <= alpha_eff."""
    cal = np.sort(s_tr[np.isfinite(s_tr)])
    a = max(alpha, 1.0 / (len(cal) + 1.0))
    p = conformal_pvalues(cal, np.where(np.isnan(s_te), -np.inf, s_te)) if len(cal) else np.ones(len(s_te))
    hot = (p <= a) & (len(cal) > 0)
    return np.convolve(hot, np.ones(k, int))[:len(hot)] >= k, a, p


def labels(seqs, n):
    lab = np.zeros(n, bool)
    for a, b in seqs:
        lab[a:b + 1] = True
    return lab


def event_counts(A, lab, seqs):
    """``(tp, fp, fp_runs, nominal alarm steps)`` of every row of the alarm matrix ``A``.

    fp: maximal alarm segments outside every sequence (alarm runs split at the sequence
    boundaries); fp_runs: alarm runs touching no sequence (the ESA-ADB false positive)."""
    R, n = A.shape
    out = A & ~lab
    Ap, Op = np.pad(A, ((0, 0), (0, 1))).ravel(), np.pad(out, ((0, 0), (0, 1))).ravel()   # rows kept apart
    s, e = runs(Op)
    row, whole = s // (n + 1), ~Ap[s - 1] & ~Ap[e]          # whole: the segment is an entire alarm run
    tp = np.sum([A[:, a:b + 1].any(1) for a, b in seqs], 0) if seqs else np.zeros(R, int)
    return tp, np.bincount(row, minlength=R), np.bincount(row[whole], minlength=R), out.sum(1)


def event_metrics(alarm, seqs, n, ok=None):
    """Event counts of one channel's alarm on all steps; point-adjusted counts on the steps ``ok``."""
    lab, ok = labels(seqs, n), np.ones(n, bool) if ok is None else ok
    tp, fp, fpr, na = (int(v[0]) for v in event_counts(alarm[None], lab, seqs))
    pa = alarm & ok                              # point-adjusted (for contrast only)
    for a, b in seqs:
        if pa[a:b + 1].any():
            pa[a:b + 1] = ok[a:b + 1]
    return {"tp": tp, "fn": len(seqs) - tp, "fp": fp, "fp_runs": fpr, "nominal_steps": int((~lab).sum()),
            "nominal_alarm_steps": na, "nominal_alarm_fraction": na / max(int((~lab).sum()), 1),
            "alarm_fraction": float(alarm.mean()),
            "delays": [int(np.argmax(alarm[a:b + 1])) for a, b in seqs if alarm[a:b + 1].any()],
            "pa_tp": int((pa & lab).sum()), "pa_fp": int((pa & ~lab).sum()), "pa_fn": int((~pa & lab & ok).sum())}


def surrogates(alarm, seqs, rng, draws=DRAWS):
    """Event counts (draws x 4) of label-free alarms with the alarm time of ``alarm``: ``shift``
    circularly shifts it by a uniform offset (same runs), ``block`` is one block at a uniform start;
    ``late`` (1 x 4) is one block at the end of the series."""
    n, m = len(alarm), int(alarm.sum())
    j, u = np.arange(n), rng.integers(0, n - m + 1, draws)[:, None]
    A = {"shift": alarm[(j - rng.integers(0, n, draws)[:, None]) % n], "block": (j >= u) & (j < u + m),
         "late": (j >= n - m)[None]}
    lab = labels(seqs, n)
    return {k: np.stack(event_counts(a, lab, seqs), 1) for k, a in A.items()}


def stretch(alarm, n):
    """``alarm`` mapped to ``n`` steps by relative position (a step is on if a step it covers is on)."""
    return np.logical_or.reduceat(alarm, np.arange(n) * len(alarm) // n)


def swaps(chans, rng, draws=DRAWS):
    """Pooled event counts (draws x 4) of the cross-channel surrogate of the ``(alarm, seqs)`` list
    ``chans``: a uniform random permutation reassigns the alarms to the channels, each stretched to
    its new channel's length (None for fewer than two channels)."""
    if len(chans) < 2:
        return None
    P, tot = np.argsort(rng.random((draws, len(chans))), 1), 0
    for i, (al, seq) in enumerate(chans):
        D = np.stack([stretch(a, len(al)) for a, _ in chans])
        tot = tot + np.stack(event_counts(D[P[:, i]], labels(seq, len(al)), seq), 1)
    return tot


def fbeta(p, r, b=1.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p + r > 0, (1 + b * b) * p * r / (b * b * p + r), 0.0)


def rates(tp, fn, fp, fpr, na):
    """Event precision, recall, F1, ESA-ADB corrected precision, its F0.5 and F1 (also on arrays)."""
    tp, fn, fp, fpr = (np.asarray(v, float) for v in (tp, fn, fp, fpr))
    with np.errstate(invalid="ignore", divide="ignore"):
        p, rc = np.where(tp + fp > 0, tp / (tp + fp), 0.0), np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        pe = np.where(tp > 0, tp / (tp + fpr) * (1 - na), 0.0)
    return p, rc, fbeta(p, rc), pe, fbeta(pe, rc, 0.5), fbeta(pe, rc)


def event_summary(g):
    """Event metrics of the channel rows ``g``, counts pooled as if the channels were one record
    (rates NaN if ``g`` is empty)."""
    tp, fp, fn, fpr, na, ns = (int(g[c].sum()) for c in ("tp", "fp", "fn", "fp_runs", "nominal_alarm_steps", "nominal_steps"))
    p, rc, f1, pe, f05, fe1 = (float(v) if len(g) else np.nan for v in rates(tp, fn, fp, fpr, na / max(ns, 1)))
    return {"channels": len(g), "events": tp + fn, "event_tp": tp, "event_fp": fp, "event_precision": p,
            "event_recall": rc, "event_f1": f1, "event_fp_runs": fpr, "esa_ew_precision": pe, "esa_ew_f0.5": f05,
            "esa_ew_f1": fe1, "nominal_alarm_fraction": na / ns if ns else np.nan,
            "median_delay_steps": float(g.median_delay.median())}


def auc_pr(y, s):
    return float(average_precision_score(y, s)) if y.any() else np.nan


def run(out=None, chans=None):
    lab = channels()
    if chans is not None:
        bad = sorted(set(chans) - set(lab.chan_id))
        if bad:
            raise ValueError(f"unknown chan_id: {', '.join(bad)}")
        lab = lab[lab.chan_id.isin(chans)].reset_index(drop=True)
    rows, pw, nulls, als, half = [], {}, {}, {}, [0, 0]
    for _, r in lab.iterrows():
        c, sc, seq = r["chan_id"], r["spacecraft"], r["seq"]
        tr, te = load(c)
        n = len(te)
        y = labels(seq, n)
        half = [half[0] + int(y[n // 2:].sum()), half[1] + int(y.sum())]     # labelled steps in the second half
        tm = score_telemanom(c, n)
        ok = np.isfinite(tm) if tm is not None else np.zeros(n, bool)      # point-wise steps, all detectors
        rng = np.random.default_rng(zlib.crc32(c.encode()))
        scores = {name: fn(tr, te) for name, fn in (("ar_ridge", score_ar), ("pca", score_pca), ("knn", score_knn),
                                                       ("chronos2", score_chronos2))}
        scores["random"] = (rng.random(len(tr)), rng.random(n))
        scores["always_on"] = (None, None)
        if tm is not None:
            scores["telemanom"] = (None, tm)
        for name, (s_tr, s_te) in scores.items():
            row = {"chan": c, "spacecraft": sc, "model": name}
            al = np.ones(n, bool) if name == "always_on" else None
            if s_tr is not None:
                s_tr, s_te = np.maximum(s_tr, EPS), np.maximum(s_te, EPS)
                al, a_eff, p = conformal_alarms(s_tr, s_te)
                row.update({"n_cal": int(np.isfinite(s_tr).sum()), "alpha_eff": a_eff, "near_zero_spread": bool(np.nanstd(s_tr) < EPS)})
                acc = nulls.setdefault((sc, name), {})
                for k, v in surrogates(al, seq, np.random.default_rng(zlib.crc32(f"{c}/{name}".encode()))).items():
                    acc[k] = acc.get(k, 0) + v
                als.setdefault((sc, name), []).append((al, seq))
            if al is not None:
                m = event_metrics(al, seq, n, ok)
                d = m.pop("delays")
                row.update({**m, "saturated": m["alarm_fraction"] > SAT, "median_delay": float(np.median(d)) if d else np.nan})
            if s_te is not None:
                row["auc_pr"] = auc_pr(y[ok], s_te[ok])
                if ok.any():                         # pooled: conformal p-values (telemanom: raw errors)
                    pw.setdefault((sc, name), []).append((y[ok], (-p if s_tr is not None else s_te)[ok]))
            rows.append(row)
        print(f"{c:>5} {sc}", flush=True)
    for (sc, name), v in als.items():                # cross-channel surrogate, within each spacecraft
        C = swaps(v, np.random.default_rng(zlib.crc32(f"{sc}/{name}/swap".encode())))
        if C is not None:
            nulls[(sc, name)]["swap"] = C
    df = pd.DataFrame(rows)
    by = df[df.model.isin(SCORED)].groupby("chan")             # channel sets common to every detector
    df["attainable"] = df.chan.map(by.alpha_eff.max() <= ALPHA)
    sat, nzs = (by[c].any().astype(bool) for c in ("saturated", "near_zero_spread"))
    df["unsaturated"], df["nondegenerate"] = df.chan.map(~sat), df.chan.map(~sat & ~nzs)
    df = df.astype({c: "Int64" for c in ("n_cal", "tp", "fn", "fp", "fp_runs", "nominal_steps", "nominal_alarm_steps",
                                         "pa_tp", "pa_fp", "pa_fn")}
                   | {c: "boolean" for c in ("near_zero_spread", "saturated", "attainable", "unsaturated", "nondegenerate")})
    summ, subsets = [], ("attainable", "unsaturated", "nondegenerate")
    for (sc, name), g in sorted(df.groupby(["spacecraft", "model"]), key=lambda t: (t[0][0], t[0][1] == "always_on", t[0][1])):
        row = {"spacecraft": sc, "model": NAMES.get(name, name)}
        if name != "telemanom":
            row.update({**event_summary(g), "channels_saturated": int(g.saturated.sum())})
            if name in SCORED:
                row.update({"channels_alpha_raised": int((g.alpha_eff > ALPHA).sum()),
                            "channels_near_zero_spread": int(g.near_zero_spread.sum())})
                ns = int(g.nominal_steps.sum())
                for kind, C in nulls[(sc, name)].items():                 # pooled surrogate counts (draws x 4)
                    v = rates(C[:, 0], row["events"] - C[:, 0], C[:, 1], C[:, 2], C[:, 3] / max(ns, 1))
                    for col, x in (("event_f1", v[2]), ("esa_ew_f0.5", v[4])):
                        row[f"{col}_{kind}"] = float(x.mean())
                        if len(x) > 1:
                            row[f"{col}_{kind}_p"] = float((1 + (x >= row[col] - 1e-12).sum()) / (len(x) + 1))
            ptp, pfp, pfn = g.pa_tp.sum(), g.pa_fp.sum(), g.pa_fn.sum()
            row["point_adjusted_f1"] = float(fbeta(ptp / max(ptp + pfp, 1), ptp / max(ptp + pfn, 1)))
        if (sc, name) in pw:
            ys, ss = (np.concatenate(v) for v in zip(*pw[(sc, name)]))
            row.update({"pointwise_auc_pr": float(g.auc_pr.mean()),
                        "pointwise_auc_pr_pooled" + ("_raw" if name == "telemanom" else ""): auc_pr(ys, ss),
                        "anomaly_share": float(ys.mean()), "channels_with_scores": len(pw[(sc, name)])})
        if name != "telemanom":
            for s in subsets:
                row.update({f"{k}_{s}": v for k, v in event_summary(g[g[s]]).items()})
        summ.append(row)
    S = pd.DataFrame(summ)
    counts = [f"{c}{s}" for s in ("", *(f"_{s}" for s in subsets)) for c in ("channels", "events", "event_tp", "event_fp", "event_fp_runs")]
    S = S.astype({c: "Int64" for c in (*counts, "channels_saturated", "channels_alpha_raised", "channels_near_zero_spread",
                                       "channels_with_scores") if c in S})
    ev = df[df.model.isin(SCORED)]
    raised = {m: sorted(g.chan[g.alpha_eff > ALPHA]) for m, g in ev.groupby("model")}
    od = out or output_dir("benchmarks", "telemetry_smap_msl")
    os.makedirs(od, exist_ok=True)
    df.to_csv(os.path.join(od, "per_channel.csv"), index=False)
    S.to_csv(os.path.join(od, "summary.csv"), index=False)
    with open(os.path.join(od, "config.json"), "w") as f:
        json.dump({"window": W, "alpha": ALPHA, "persistence": K, "alarm": "causal, at the K-th consecutive step with p <= alpha_eff",
                   "calibration": f"out-of-fold scores, {FOLDS} contiguous blocks, {W}-window purge on each side; test scores "
                                  "from the full-training fit (cross-conformal approximation, Vovk 2015); chronos2 is "
                                  "zero-shot, so its training-series scores are used directly (split conformal)",
                   "chronos2": CHRONOS, "score_floor": EPS, "channels": int(len(lab)),
                   "channels_by_spacecraft": {k: int(v) for k, v in lab.spacecraft.value_counts().sort_index().items()},
                   "label_rows": int(lab.label_rows.sum()), "merged_label_rows": lab.chan_id[lab.label_rows > 1].tolist(),
                   "alpha_eff": "max(alpha, 1 / (n_cal + 1)) per channel",
                   "n_channels_alpha_raised": {m: len(v) for m, v in raised.items()}, "channels_alpha_raised": raised,
                   "near_zero_spread": f"sd of the out-of-fold calibration scores < {EPS} (ar_ridge: constant training "
                                       "target; pca, knn: constant training series)",
                   "near_zero_spread_channels": {m: sorted(g.chan[g.near_zero_spread.astype(bool)]) for m, g in ev.groupby("model")},
                   "saturation": f"alarm on more than {SAT} of the test steps (label-free)",
                   "saturated_channels": {m: sorted(g.chan[g.saturated.astype(bool)]) for m, g in ev.groupby("model")},
                   "subsets": {"attainable": "channels where alpha_eff == alpha for every detector",
                               "unsaturated": "channels no detector saturates",
                               "nondegenerate": "unsaturated channels where no detector has near-zero spread",
                               "note": "channel sets common to every detector (per_channel.csv flags; summary suffixes)"},
                   "subset_channels_excluded": {s: sorted(df.chan[~df[s].astype(bool)].unique()) for s in subsets},
                   "event_steps": "all test steps",
                   "event_fp": "maximal alarm segments outside every labelled sequence",
                   "esa_ew_precision": "TP / (TP + alarm runs touching no sequence) * (1 - nominal_alarm_fraction)",
                   "reference": "always_on: in alarm on every test step",
                   "labelled_steps_in_second_half": half[0] / max(half[1], 1),
                   "surrogates": f"{DRAWS} label-free alarms per channel and detector built from its alarms; shift: circularly "
                                 "shifted by a uniform offset; block: one block of its alarm time at a uniform start (both "
                                 "position-uniform); swap: the alarms reassigned among the channels of the spacecraft by a "
                                 "uniform random permutation, each stretched to its new channel's length by relative position "
                                 "(keeps the alarm positions; the null for channel-specific detection); late: one block of its alarm time at the end of the series "
                                 "(one draw, a reference; no p-value); *_<kind>: mean pooled metric; *_<kind>_p: (1 + "
                                 "#{surrogate >= detector}) / (draws + 1)",
                   "pointwise_steps": "test steps where telemanom has scores (250 .. n_test - 10), for every detector",
                   "pointwise_auc_pr": "mean over channels of the per-channel AUC-PR",
                   "pointwise_auc_pr_pooled": "pooled steps; each channel's scores mapped to conformal p-values against its "
                                              "out-of-fold training scores; steps above every calibration score tie at "
                                              "1 / (n_cal + 1), so saturated or near-zero-spread channels fill the top of the "
                                              "ranking (ordered by n_cal); not for telemanom",
                   "pointwise_auc_pr_pooled_raw": "telemanom only (no training scores): pooled raw errors, the published convention",
                   "point_adjusted_f1": "on the pointwise steps; for contrast only",
                   "note": "thresholds fixed a priori; no test labels used for any choice"}, f, indent=1)
    print(S.to_string(index=False))
    return S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory (default outputs/benchmarks/telemetry_smap_msl)")
    ap.add_argument("--channels", default=None, help="comma-separated chan_ids (default: all)")
    a = ap.parse_args(argv)
    run(a.out, a.channels.split(",") if a.channels else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
