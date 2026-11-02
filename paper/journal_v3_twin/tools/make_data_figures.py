"""Dataset and model-output figures of the twin manuscript (vector PDF + PNG), drawn from the data
and result files.  Every example (object, event, channel) is chosen by a seed or a rule written
below, not by how it looks.

    PYTHONPATH=. COSMOS_DEVICE=xpu:0 python paper/journal_v3_twin/tools/make_data_figures.py [key ...]

Recomputed illustrations (detector scores of two SMAP/MSL channels, area-to-mass posteriors of three
objects) use the study's own functions and are cached in outputs/figure_data/; the script checks
that they reproduce the stored results before drawing them.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_figures import (AMBER, AXIS, B, BLUE, GREEN, GRID, INK, INK2, LABEL, MUTED, ORANGE, ROOT, fig_name,  # noqa: E402
                          plt, save, style)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "outputs", "figure_data")
CLASSES = [("D", "fragments", BLUE), ("R", "rocket stages", ORANGE), ("C", "components", GREEN), ("P", "payloads", MUTED)]
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CAP = np.log10(1826.25)
DAY_TICKS = [(np.log10(1.0), "1 d"), (np.log10(7.0), "1 wk"), (np.log10(30.44), "1 mo"), (np.log10(365.25), "1 yr"),
             (CAP, "5 yr"), (np.log10(3652.5), "10 yr"), (np.log10(36525.0), "100 yr")]
def _smap_rule():
    """Rule for the example channels: per spacecraft, the median channel by Chronos-2 per-channel AUC-PR, taking the lower
    of the two middle channels when their number is even (so the rule always names a real channel; "closest to the median"
    would be an exact tie between those two).  Ties in AUC-PR are broken by the channel identifier."""
    d = pd.read_csv(os.path.join(B, "telemetry_smap_msl", "per_channel.csv"))
    d = d[d.model.eq("chronos2") & d.auc_pr.notna()]
    out = {}
    for sc, g in d.groupby("spacecraft"):
        g = g.sort_values(["auc_pr", "chan"]).reset_index(drop=True)
        out[sc] = str(g.chan.iloc[(len(g) - 1) // 2])
    return out


SMAP_EXAMPLES = _smap_rule()
AS_OF = "2026-10-27"


def ylab(ax, text):
    ax.set_ylabel(text, fontsize=7.5, color=INK2)


def title(ax, text, pad=6):
    ax.set_title(text, fontsize=8, color=INK, loc="left", pad=pad)


def day_ticks(ax, axis="y", lo=-0.5, hi=CAP + 0.05):
    t = [(v, s) for v, s in DAY_TICKS if lo <= v <= hi]
    (ax.set_yticks if axis == "y" else ax.set_xticks)([v for v, _ in t], [s for _, s in t])
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)


def yearfrac(t):
    t = pd.to_datetime(t)
    return t.dt.year + (t.dt.dayofyear - 0.5) / 365.25 if isinstance(t, pd.Series) else t.year + (t.dayofyear - 0.5) / 365.25


# ------------------------------------------------------------------ dataset views

def fig_data_reentry(name=None):
    from cosmos.ai.lifetime import run as LR
    from cosmos.ai.twin import engine as E
    name = name or fig_name("data_reentry")
    c = LR.load_cohort()
    fig = plt.figure(figsize=(8.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.05], hspace=0.38, wspace=0.28)
    gb = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.28], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    counts = c["type1"].value_counts()
    for t, lab, col in sorted(CLASSES, key=lambda x: -counts.get(x[0], 0)):   # largest class first: rarer ones stay visible
        s = c["type1"].eq(t)
        ax.scatter(c.loc[s, "perigee"], c.loc[s, "apogee"], s=1.2, color=col, alpha=0.35, linewidth=0, rasterized=True,
                   label=f"{lab} ({s.sum():,})")
    ax.set_yscale("log")
    ax.set_xlim(80, 2000)
    ax.set_ylim(100, 6e4)
    style(ax, "perigee of the canonical orbit (km)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "apogee (km, log scale)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right", markerscale=6)
    title(ax, f"(a) {len(c):,} catalogued objects, one GCAT canonical orbit each")

    ax = fig.add_subplot(gb[0])
    yr = yearfrac(c["t0"]).astype(float)
    ax.hist(yr, bins=np.arange(1957, 2027.01, 1.0), color=BLUE, rwidth=0.85)
    ax.set_xlim(1957, 2027)
    style(ax, None, xgrid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "objects per year")
    title(ax, "(b) objects by canonical-orbit epoch, and the two forecast origins")
    st = fig.add_subplot(gb[1], sharex=ax)
    seg = (("train", BLUE), ("tune-A / tune-B", GREEN), ("test", ORANGE))
    for k, o in enumerate(("2016", "2011")):
        tr_end, va_end, start, te_end = (yearfrac(pd.Timestamp(x)) for x in LR.ORIGINS[o])
        for (lab, col), (a, b) in zip(seg, ((1957.0, tr_end), (tr_end, va_end), (start, te_end))):
            st.barh(k, b - a, left=a, height=0.62, color=col, label=lab if k == 0 else None)
    st.set_yticks([0, 1], ["origin 2016", "origin 2011"], fontsize=6.6)
    st.set_ylim(-0.6, 1.6)
    style(st, "year of the canonical-orbit epoch", xgrid=False)
    h, lab = st.get_legend_handles_labels()
    ax.legend(h, lab, frameon=False, fontsize=6.5, loc="upper left", title="cohorts of each origin", title_fontsize=6.5)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = fig.add_subplot(gs[1, 0])
    lt = np.log10(np.maximum(c["time_days"].astype(float), 0.1))
    bins = np.linspace(-1, 5, 61)
    dec = c["event"].eq(1)
    ax.hist(lt[dec], bins=bins, histtype="step", color=BLUE, linewidth=1.4, label=f"decayed naturally ({dec.sum():,})")
    ax.hist(lt[~dec], bins=bins, histtype="step", color=MUTED, linewidth=1.4,
            label=f"censored: in orbit, or phase ended without natural decay ({(~dec).sum():,})")
    ax.axvline(CAP, color=INK2, linestyle="--", linewidth=0.8)
    ax.text(CAP, 1.0, " 5-yr cap of the target", fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform(), va="top")
    t = [(v, s_) for v, s_ in DAY_TICKS if s_ != "10 yr"]
    ax.set_xticks([v for v, _ in t], [s_ for _, s_ in t])
    ax.set_xlim(-1, 5)
    n0 = int((c["time_days"].astype(float) == 0).sum())
    ax.annotate(f"{n0} objects decayed on their\ncanonical-orbit epoch (0 days),\nplotted at the left edge", (-1, 850),
                xytext=(0.1, 0.62), textcoords="axes fraction", fontsize=6.0, color=INK2, va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5})
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    style(ax, "time from canonical-orbit epoch to decay or censoring")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "objects per bin")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    title(ax, "(c) observed lifetimes and censoring")

    ax = fig.add_subplot(gs[1, 1])
    sw = E.load_space_weather()
    x = yearfrac(pd.Series(sw.index)).to_numpy()
    for k, o in enumerate(("2011", "2016")):
        a, b = (yearfrac(pd.Timestamp(x)) for x in LR.ORIGINS[o][2:])
        ax.plot([a, b], [318 - 14 * k] * 2, color=INK2, linewidth=2.0, solid_capstyle="butt")
        ax.text(a - 0.5, 318 - 14 * k, f"test of origin {o}", fontsize=6, color=INK2, va="center", ha="right")
    ax.plot(x, sw["f107_daily"].to_numpy(), color=AXIS, linewidth=0.3, rasterized=True, label="daily")
    ax.plot(x, sw["f107_last81"].to_numpy(), color=INK, linewidth=0.9, label="81-day mean (model input)")
    ax.set_xlim(1957, 2027)
    ax.set_ylim(50, 330)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    style(ax, "year")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "F10.7 solar flux (sfu)")
    title(ax, "(d) space weather (CelesTrak SW-All)")
    save(fig, name)


def fig_data_debris(name=None):
    from cosmos.ai.lifetime import run as LR
    from cosmos.ai.conjunction import data as CD
    name = name or fig_name("data_debris")
    c = LR.load_cohort()
    fam = c[c["breakup_family"].notna()]
    top = fam["breakup_family"].value_counts().index[:4]           # rule: the four largest breakup families
    names = c.set_index("JCAT")["Name"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    ax = axes[0, 0]
    grid = pd.date_range("1996-01-01", AS_OF, freq="MS")
    for f, col in zip(top, (BLUE, ORANGE, GREEN, AMBER)):
        g = fam[fam["breakup_family"].eq(f)]
        end = pd.to_datetime(g["t_end"]).where(~g["end_kind"].eq("in_orbit"), pd.Timestamp("2100-01-01"))
        t0 = pd.to_datetime(g["t0"]).to_numpy()
        n = [int(((t0 <= t) & (end.to_numpy() > t)).sum()) for t in grid.to_numpy()]
        xs = yearfrac(pd.Series(grid)).to_numpy()
        ax.plot(xs, n, color=col, linewidth=1.4, label=f"{names.get(f, f)} ({len(g):,} catalogued)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    style(ax, "year")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "catalogued fragments in orbit")
    title(ax, "(a) the four largest breakup clouds in the cohort")

    ax = axes[0, 1]
    g = fam[fam["breakup_family"].eq(top[0])]
    T = 2 * np.pi * np.sqrt(g["sma_km"].astype(float) ** 3 / 398600.4418) / 60.0
    ax.scatter(T, g["apogee"], s=3, color=BLUE, linewidth=0, alpha=0.7, label="apogee", rasterized=True)
    ax.scatter(T, g["perigee"], s=3, color=ORANGE, linewidth=0, alpha=0.7, label="perigee", rasterized=True)
    style(ax, "orbital period of the canonical orbit (min)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "altitude (km)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", markerscale=3)
    title(ax, f"(b) Gabbard diagram of the {names.get(top[0], top[0])} cloud")

    raw = CD.load_raw()["train"]
    fin = raw.sort_values("time_to_tca").groupby("event_id").first()
    early = raw[raw["time_to_tca"] >= 2].groupby("event_id").size()
    kept = fin.index[(fin["time_to_tca"] < 1)].intersection(early.index)
    hi = fin.loc[kept].index[fin.loc[kept, "risk"].ge(CD.HIGH_RISK)]
    lo = kept.difference(hi)
    rng = np.random.default_rng(0)
    ax = axes[1, 0]
    for ev_set, col, n, lw, lab in ((lo, MUTED, 60, 0.6, "final risk < −6 (60 of {:,} events)"),
                                    (hi, ORANGE, 15, 0.9, "final risk ≥ −6 (15 of {:,} events)")):
        pick = rng.choice(np.asarray(ev_set), size=min(n, len(ev_set)), replace=False)
        for k, e in enumerate(pick):
            s = raw[raw["event_id"].eq(e)].sort_values("time_to_tca", ascending=False)
            ax.plot(s["time_to_tca"], s["risk"], color=col, linewidth=lw, alpha=0.8, marker="o", markersize=1.6,
                    label=lab.format(len(ev_set)) if k == 0 else None)
    ax.axvline(2.0, color=INK2, linestyle="--", linewidth=0.8)
    ax.axhline(CD.HIGH_RISK, color=INK2, linestyle=":", linewidth=0.8)
    ax.text(2.0, 1.0, " inputs end 2 days before TCA", fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlim(7.2, 0)
    ax.set_ylim(-34.2, 0.5)                                        # headroom below the −30 floor for the legend
    style(ax, "time to closest approach (days)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "risk, log10 collision probability")
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    title(ax, "(c) ESA Kelvins conjunction data messages, training events")

    ax = axes[1, 1]
    y = fin.loc[kept, "risk"].to_numpy()
    bins = np.arange(-30.0, 0.1, 1.0)                              # integer edges: the colour boundary is exactly -6
    cnt, _ = np.histogram(y, bins)
    left = bins[:-1]
    hi_bin = left >= CD.HIGH_RISK
    assert int(cnt[hi_bin].sum()) == int((y >= CD.HIGH_RISK).sum())
    ax.bar(left, cnt, width=0.85, align="edge", color=[ORANGE if h else BLUE for h in hi_bin])
    ax.set_yscale("log")
    ax.axvline(CD.HIGH_RISK, color=INK2, linestyle=":", linewidth=0.8)
    ax.text(-29.6, cnt[0] * 1.1, f"floor −30: {np.mean(y <= -30 + 1e-9):.0%}", fontsize=6.5, color=INK2, va="bottom")
    ax.text(0.98, 0.97, f"high risk (≥ −6, right of the dotted line): {int((y >= CD.HIGH_RISK).sum())}", fontsize=6.5,
            color=INK2, va="top", ha="right", transform=ax.transAxes)
    style(ax, "final risk of the event (log10 collision probability)", xgrid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, f"events (log scale; {len(y):,} training events)")
    title(ax, "(d) the forecast target: final risk")
    fig.tight_layout()
    save(fig, name)


def fig_data_telemetry(name=None):
    from cosmos.ai.telemetry import smap_msl as SM
    name = name or fig_name("data_telemetry")
    lab = SM.channels()
    rng = np.random.default_rng(2026)
    picks = []
    for sc in ("SMAP", "MSL"):                                     # rule: two random channels per spacecraft
        pool = lab[lab.spacecraft.eq(sc) & ~lab.chan_id.eq(SMAP_EXAMPLES[sc])]
        picks += [(sc, r) for _, r in pool.iloc[rng.choice(len(pool), 2, replace=False)].iterrows()]
    fig = plt.figure(figsize=(8.6, 6.6))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.18)
    for k, (sc, r) in enumerate(picks):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        tr, te = SM.load(r["chan_id"])
        x = np.concatenate([tr[:, 0], te[:, 0]])
        ax.plot(np.arange(len(tr)), tr[:, 0], color=MUTED, linewidth=0.5, rasterized=True)
        ax.plot(np.arange(len(tr), len(x)), te[:, 0], color=INK, linewidth=0.5, rasterized=True)
        for a, b in r["seq"]:
            ax.axvspan(len(tr) + a, len(tr) + b, color=ORANGE, alpha=0.3, linewidth=0)
        ax.axvline(len(tr), color=INK2, linewidth=0.8, linestyle="--")
        style(ax, "time step" if k >= 2 else None)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        if k % 2 == 0:
            ylab(ax, "scaled value")
        title(ax, f"({'abcd'[k]}) {sc} channel {r['chan_id']}: {len(r['seq'])} labelled anomal{'y' if len(r['seq']) == 1 else 'ies'}")
        if k == 0:
            ax.text(len(tr), 1.0, " test series", fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform(), va="top")
            ax.text(len(tr), 1.0, "training series ", fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform(), va="top", ha="right")

    d = os.path.join(ROOT, "data", "feature1_health", "esa_adb", "processed", "mission1")
    meta = json.load(open(os.path.join(d, "meta.json")))
    L = pd.read_csv(os.path.join(d, "labels_with_types.csv"), parse_dates=["StartTime", "EndTime"])
    target = {c for c, t in zip(meta["channels"], meta["target"]) if t}
    t0 = pd.Timestamp(meta["splits"]["test"]["t0"])
    ev = L[(L.StartTime >= t0) & L.Category.eq("Anomaly") & L.Channel.isin(target)].sort_values("StartTime")
    e = ev.iloc[0]                                                 # rule: first labelled anomaly of the test half
    chans = [ch for ch in L[L.ID.eq(e.ID)].Channel.unique() if ch in target][:3]
    X = np.load(os.path.join(d, "test_X.npy"), mmap_mode="r")
    Lab = np.load(os.path.join(d, "test_L.npy"), mmap_mode="r")
    step = pd.Timedelta(seconds=meta["rule_s"])
    a = int((e.StartTime - t0 - pd.Timedelta(days=3)) / step)
    b = int((L[L.ID.eq(e.ID)].EndTime.max() - t0 + pd.Timedelta(days=3)) / step)
    a, b = max(a, 0), min(b, X.shape[0])
    ax = fig.add_subplot(gs[2, :])
    tt = t0 + np.arange(a, b) * step
    for k, ch in enumerate(chans):
        j = meta["channels"].index(ch)
        v = np.asarray(X[a:b, j], float)
        v = (v - np.nanmin(v)) / max(np.nanmax(v) - np.nanmin(v), 1e-12)
        ax.plot(tt, v + 1.15 * k, color=(BLUE, GREEN, AMBER)[k], linewidth=0.6, label=ch.replace("_", " "))
        lab_on = np.asarray(Lab[a:b, j]) > 0
        if lab_on.any():
            edges = np.flatnonzero(np.diff(np.r_[0, lab_on.astype(int), 0]))
            for s0, s1 in zip(edges[::2], edges[1::2]):
                ax.axvspan(tt[s0], tt[min(s1, len(tt) - 1)], ymin=0, ymax=1, color=ORANGE, alpha=0.12, linewidth=0)
    style(ax, "time (UTC, anonymised mission timeline)")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3)
    title(ax, f"(e) ESA-ADB Mission 1, first labelled anomaly of the test half ({e.ID}; values min–max scaled per channel)")
    save(fig, name)


# ------------------------------------------------------------------ model outputs

def fig_reentry_outputs(name=None):
    from cosmos.ai.lifetime import run as LR
    name = name or fig_name("reentry_outputs")
    d = os.path.join(B, "reentry_lifetime", "origin2016_uncontrolled")
    P = np.load(os.path.join(d, "test_predictions.npz"))
    t = LR.split(LR.load_cohort(), "2016", "uncontrolled")["test"].reset_index(drop=True)
    z, m = P["z"], P["known5"].astype(bool)
    assert len(t) == len(z)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(m), 40, replace=False)        # rule: 40 random test objects with known outcome
    idx = idx[np.argsort(z[idx])]
    fig = plt.figure(figsize=(8.6, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.5, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    xs = np.arange(len(idx))
    for off, mod, col in ((-0.17, "physics_only", MUTED), (0.17, "cosmos_gated", ORANGE)):
        q = P[f"{mod}__q"][idx]
        ax.vlines(xs + off, P[f"{mod}__lo90"][idx], P[f"{mod}__hi90"][idx], color=col, linewidth=1.3,
                  label=f"{LABEL[mod]}: 90 % conformal interval and median")
        ax.scatter(xs + off, q[:, 3], s=10, color=col, zorder=3)
    cap = z[idx] >= CAP - 1e-6
    ax.scatter(xs[~cap], z[idx][~cap], marker="D", s=14, color=INK, zorder=4, label="observed time to decay")
    ax.scatter(xs[cap], z[idx][cap], marker="^", s=18, color=INK, zorder=4, label="still in orbit after 5 years")
    ax.set_xticks(xs, t["type1"].to_numpy()[idx], fontsize=6.3)
    day_ticks(ax, "y", -0.35, CAP + 0.12)
    style(ax, "test objects (40 at random), sorted by outcome; letter: D fragment, R rocket stage, C component", xgrid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "time to re-entry from the canonical orbit")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    title(ax, "(a) re-entry forecasts for individual test objects, origin 2016 (test 2016–2021)")
    for k, mod in enumerate(("physics_only", "tabicl", "cosmos_gated")):
        ax = fig.add_subplot(gs[1, k])
        med = P[f"{mod}__q"][m][:, 3]
        hb = ax.hexbin(z[m], med, gridsize=32, bins="log", cmap=SEQ, mincnt=1, extent=(-0.35, CAP + 0.05, -0.35, CAP + 0.05),
                       linewidths=0)
        ax.plot([-0.35, CAP], [-0.35, CAP], color=INK2, linewidth=0.8, linestyle=":")
        day_ticks(ax, "x", -0.35, CAP + 0.05)
        day_ticks(ax, "y", -0.35, CAP + 0.05)
        style(ax, "observed (capped at 5 yr)", xgrid=False)
        ax.tick_params(labelsize=6.3)
        if k == 0:
            ylab(ax, "predicted median")
        mae = float(np.mean(np.abs(med - z[m])))
        title(ax, f"({'bcd'[k]}) {LABEL[mod]}\nMAE {mae:.3f} log10 days, n = {int(m.sum()):,}")
    cb = fig.colorbar(hb, ax=fig.axes[-3:], fraction=0.025, pad=0.02)
    cb.set_label("objects per cell (log scale)", fontsize=6.5, color=INK2)
    cb.ax.tick_params(labelsize=6)
    save(fig, name)


def fig_cac_outputs(name=None):
    from cosmos.ai.conjunction import data as CD
    name = name or fig_name("cac_outputs")
    P = np.load(os.path.join(B, "conjunction_kelvins_cac", "test_predictions.npz"))
    Bd = CD.build()
    te = Bd["test"]
    assert np.allclose(te["y_final"].to_numpy(), P["y"])
    y, lat, q = P["y"], P["latest"], P["kumo__q"]
    hi = y >= CD.HIGH_RISK
    fig = plt.figure(figsize=(8.6, 5.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.9)
    for k, (x, lab) in enumerate(((lat, "latest risk at least 2 days before TCA (persistence)"),
                                  (q[:, 3], "Kumo-Tabular median forecast of the final risk"))):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        ax.scatter(x[~hi], y[~hi], s=6, color=MUTED, alpha=0.5, linewidth=0, label=f"final risk < −6 ({(~hi).sum():,})")
        ax.scatter(x[hi], y[hi], s=12, color=ORANGE, alpha=0.9, linewidth=0, label=f"final risk ≥ −6 ({hi.sum()})")
        ax.plot([-31, 0], [-31, 0], color=INK2, linewidth=0.8, linestyle=":")
        ax.axhline(CD.HIGH_RISK, color=INK2, linewidth=0.6, linestyle="--")
        ax.axvline(CD.HIGH_RISK, color=INK2, linewidth=0.6, linestyle="--")
        ax.set_xlim(-31, 0.5)
        ax.set_ylim(-31, 0.5)
        style(ax, lab)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ylab(ax, "final risk (log10 Pc)")
        ax.legend(frameon=False, fontsize=6.4, loc="lower right", bbox_to_anchor=(1.0, 0.08))
        title(ax, f"({'ab'[k]}) official test set, {len(y):,} events")
    raw = CD.load_raw()["test"]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.flatnonzero(hi), 3, replace=False))  # rule: three random high-risk test events
    for k, i in enumerate(pick):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        ev = te.index[i]
        s = raw[raw["event_id"].eq(ev)].sort_values("time_to_tca", ascending=False)
        ax.plot(s["time_to_tca"], s["risk"], color=INK2, linewidth=0.9, marker="o", markersize=2.5, label="input CDMs")
        ax.axvline(0.0, color=AXIS, linewidth=0.8)                 # right of this line: not on the time axis
        ax.vlines(-0.35, q[i, 0], q[i, 6], color=GREEN, linewidth=1.2, label="Kumo-Tabular 5–95 %")
        ax.vlines(-0.35, q[i, 2], q[i, 4], color=GREEN, linewidth=4.0, label="25–75 %")
        ax.scatter([-0.35], [q[i, 3]], s=16, color="white", edgecolor=GREEN, zorder=4, label="median")
        ax.scatter([-0.8], [y[i]], marker="*", s=60, color=INK, zorder=5, label="final risk")
        ax.axhline(CD.HIGH_RISK, color=INK2, linewidth=0.6, linestyle="--")
        ax.set_xlim(max(7.0, float(s["time_to_tca"].max()) + 0.2), -1.1)
        ax.set_xticks([t_ for t_ in (7, 6, 5, 4, 3, 2, 1, 0) if t_ <= max(7.0, float(s["time_to_tca"].max()))])
        lo = min(float(s["risk"].min()), float(q[i, 0]), float(y[i])) - 1.0
        ax.set_ylim(max(lo, -31), 0)
        style(ax, "time to closest approach (days)")
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        if k == 0:
            ylab(ax, "risk (log10 Pc)")
            ax.legend(frameon=False, fontsize=5.8, loc="lower left")
        title(ax, f"({'cde'[k]}) test event {ev}")
    save(fig, name)


def _smap_examples():
    from cosmos.ai.telemetry import smap_msl as SM
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "smap_examples.npz")
    if os.path.isfile(p):
        return dict(np.load(p, allow_pickle=True))
    lab = SM.channels().set_index("chan_id")
    out = {}
    pc = pd.read_csv(os.path.join(B, "telemetry_smap_msl", "per_channel.csv"))
    for sc, ch in SMAP_EXAMPLES.items():
        tr, te = SM.load(ch)
        seq = lab.loc[ch, "seq"]
        tm = SM.score_telemanom(ch, len(te))
        ok = np.isfinite(tm) if tm is not None else np.zeros(len(te), bool)
        for det, fn in (("pca", SM.score_pca), ("chronos2", SM.score_chronos2)):
            s_tr, s_te = fn(tr, te)
            s_tr, s_te = np.maximum(s_tr, SM.EPS), np.maximum(s_te, SM.EPS)
            al, a_eff, pv = SM.conformal_alarms(s_tr, s_te)
            m = SM.event_metrics(al, seq, len(te), ok)
            ref = pc[(pc.chan == ch) & (pc.model == det)].iloc[0]
            print(f"{ch} {det}: tp {m['tp']} fp {m['fp']} (stored {ref.tp} {ref.fp})", flush=True)
            assert (m["tp"], m["fp"]) == (int(ref.tp), int(ref.fp)), "recomputed alarms do not reproduce the stored run"
            out[f"{ch}|{det}|p"], out[f"{ch}|{det}|alarm"], out[f"{ch}|{det}|alpha"] = pv, al, a_eff
        out[f"{ch}|train"], out[f"{ch}|test"], out[f"{ch}|seq"] = tr[:, 0], te[:, 0], np.asarray(seq)
    np.savez_compressed(p, **out)
    return out


def fig_smap_outputs(name=None):
    name = name or fig_name("smap_outputs")
    D = _smap_examples()
    fig = plt.figure(figsize=(8.6, 5.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 0.55], hspace=0.35, wspace=0.18)
    for k, (sc, ch) in enumerate(SMAP_EXAMPLES.items()):
        te, seq = D[f"{ch}|test"], D[f"{ch}|seq"]
        n = len(te)
        ax0 = fig.add_subplot(gs[0, k])
        ax0.plot(te, color=INK, linewidth=0.5, rasterized=True)
        for a, b in seq:
            ax0.axvspan(a, b, color=ORANGE, alpha=0.3, linewidth=0)
        style(ax0, None)
        ax0.yaxis.grid(True, color=GRID, linewidth=0.7)
        if k == 0:
            ylab(ax0, "telemetry value")
        title(ax0, f"({'ab'[k]}) {sc} channel {ch}, test series; shaded: labelled anomalies")
        ax1 = fig.add_subplot(gs[1, k], sharex=ax0)
        for det, col, lab, lw in (("pca", BLUE, "PCA reconstruction", 0.6), ("chronos2", GREEN, "Chronos-2, zero-shot", 1.0)):
            ax1.plot(-np.log10(np.maximum(D[f"{ch}|{det}|p"], 1e-12)), color=col, linewidth=lw, label=lab, rasterized=True, zorder=3)
        a_eff = float(D[f"{ch}|chronos2|alpha"])
        ax1.axhline(-np.log10(a_eff), color=INK2, linestyle="--", linewidth=0.8, zorder=1)
        if a_eff > 0.001 + 1e-12:
            ax1.set_ylim(top=-np.log10(a_eff) * 1.55)
            ax1.text(n * 0.98, -np.log10(a_eff) * 1.5, "constant training series: every calibration score sits at the score\n"
                     "floor, so any non-zero residual alarms (Chronos-2 from step 210)", fontsize=6.0, color=INK2,
                     ha="right", va="top")
        ax1.text(0, -np.log10(a_eff), f" alarm level α = {a_eff:.3g}", fontsize=6.2, color=INK2, va="bottom")
        style(ax1, None)
        ax1.yaxis.grid(True, color=GRID, linewidth=0.7)
        if k == 0:
            ylab(ax1, "−log10 conformal p-value")
            ax1.legend(frameon=False, fontsize=6.4, loc="upper right")
        ax2 = fig.add_subplot(gs[2, k], sharex=ax0)
        rows = (("labels", None, ORANGE), ("PCA alarms", "pca", BLUE), ("Chronos-2 alarms", "chronos2", GREEN))
        for j, (lab, det, col) in enumerate(rows):
            on = np.zeros(n, bool)
            if det is None:
                for a, b in seq:
                    on[a:b + 1] = True
            else:
                on = D[f"{ch}|{det}|alarm"].astype(bool)
            e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            ax2.broken_barh([(s0, max(s1 - s0, 3)) for s0, s1 in zip(e[::2], e[1::2])], (2 - j - 0.35, 0.7), color=col)
        ax2.set_yticks([2, 1, 0], [r[0] for r in rows], fontsize=6.5)
        ax2.set_ylim(-0.6, 2.6)
        ax2.set_xlim(0, n)
        style(ax2, "time step")
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
    save(fig, name)


def fig_twin_forecast(name=None):
    name = name or fig_name("twin_forecast")
    T = pd.read_csv(os.path.join(ROOT, "outputs", "twin", f"as_of_{AS_OF}", "forecast_table.csv.gz"))
    fig = plt.figure(figsize=(8.6, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    reg = (("natural_decay", "natural decay", BLUE), ("operator_controlled", "operator-controlled payloads", ORANGE),
           ("unknown", "payloads without reported status", MUTED))
    counts = T["regime"].value_counts()
    for r, lab, col in sorted(reg, key=lambda x: -counts.get(x[0], 0)):   # largest regime first: the rest stay visible
        s = T["regime"].eq(r)
        ax.scatter(T.loc[s, "perigee"], T.loc[s, "apogee"], s=1.2, color=col, alpha=0.4, linewidth=0, rasterized=True,
                   label=f"{lab} ({s.sum():,})")
    ax.set_yscale("log")
    ax.set_xlim(80, 2000)
    ax.set_ylim(100, 6e4)
    style(ax, "current perigee (km)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "current apogee (km, log scale)")
    ax.legend(frameon=False, fontsize=6.4, loc="lower right", markerscale=6)
    title(ax, f"(a) the twin on {AS_OF}: {len(T):,} objects by regime")
    ax = fig.add_subplot(gs[0, 1])
    nat = T[T["regime"].eq("natural_decay")]
    ax.scatter(nat["perigee"], nat["p_30d"], s=4, color=BLUE, alpha=0.5, linewidth=0, rasterized=True)
    ax.set_xlim(80, 650)
    ax.set_ylim(-0.03, 1.03)
    style(ax, "current perigee (km)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ylab(ax, "P(re-entry within 30 days)")
    title(ax, f"(b) natural-decay regime: expected re-entries in 30 days {nat['p_30d'].sum():.1f}")
    ax = fig.add_subplot(gs[1, :])
    top = nat.sort_values(["p_30d", "median_days"], ascending=[False, True]).head(15)
    y = np.arange(len(top))
    ax.hlines(y, top["lo90_days"], top["hi90_days"], color=BLUE, linewidth=2.0, label="90 % Monte-Carlo interval")
    ax.scatter(top["median_days"], y, s=18, color="white", edgecolor=BLUE, zorder=3, label="median")
    for yy, (_, r) in zip(y, top.iterrows()):
        ax.text(r["hi90_days"] + 0.8, yy, f"P30 = {r['p_30d']:.2f}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(y, [f"{r.Name} ({r.norad}, {r.perigee:.0f} km)" for r in top.itertuples()], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, float(top["hi90_days"].max()) + 9)
    style(ax, f"days after {AS_OF}")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    title(ax, "(c) the 15 natural-decay objects with the highest P(re-entry within 30 days): forecast re-entry time")
    save(fig, name)


def _twin_state():
    from cosmos.ai.lifetime import physics as PH
    from cosmos.ai.lifetime import run as LR
    from cosmos.ai.twin import engine as E
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "twin_state_examples.json")
    if os.path.isfile(p):
        return json.load(open(p))
    T = pd.read_csv(os.path.join(ROOT, "outputs", "twin", f"as_of_{AS_OF}", "forecast_table.csv.gz"))
    # rule: one random object per class among natural-decay objects whose history-conditioned median forecast is
    # within five years (so that the forecast is not at the cap)
    pool = T[T["regime"].eq("natural_decay") & T["history_kind"].eq("history") & T["median_days_physics_history"].lt(1826.0)]
    rng = np.random.default_rng(0)
    chosen = [int(pool[pool.type1.eq(c)].norad.iloc[rng.integers(pool.type1.eq(c).sum())]) for c in ("D", "R", "C")]
    prior = E.physics_prior_as_of(AS_OF)
    d = E.current_state(AS_OF)
    d = d[d["type1"].isin(LR.POPULATIONS["uncontrolled"])].reset_index(drop=True)
    sub = d.set_index("norad").loc[chosen].reset_index()
    am0, _ = PH.sample_log_am(sub, 128, 0, prior)
    post, info = PH.history_posterior(sub, prior, AS_OF, E.load_space_weather(), n_out=32)
    life0 = PH.predict(sub, prior, log_am=am0[:, :32])["samples_days"]   # same draws the panel shows
    life1 = PH.predict(sub, prior, log_am=post)["samples_days"]
    ref = T.set_index("norad").loc[chosen]
    for k in range(len(chosen)):                                  # the recomputation must reproduce the stored forecast
        assert abs(min(np.median(life1[k]), 1826.25) - ref["median_days_physics_history"].iloc[k]) < 1e-6
        assert abs(np.mean(life1[k] <= 30) - ref["p_physics_history_30d"].iloc[k]) < 1e-9
    out = {"objects": [{"norad": int(n), "name": str(sub.loc[k, "Name"]), "type1": str(sub.loc[k, "type1"]),
                        "perigee0": float(sub.loc[k, "perigee0"]), "apogee0": float(sub.loc[k, "apogee0"]),
                        "perigee": float(sub.loc[k, "perigee"]), "apogee": float(sub.loc[k, "apogee"]),
                        "t0": str(pd.Timestamp(sub.loc[k, "t0_canonical"]).date()),
                        "log_am_prior": am0[k, :32].tolist(), "log_am_post": np.asarray(post[k]).tolist(),
                        "life_prior": np.asarray(life0[k]).tolist(), "life_post": np.asarray(life1[k]).tolist()}
                       for k, n in enumerate(chosen)]}
    json.dump(out, open(p, "w"))
    return out


def fig_twin_state(name=None):
    name = name or fig_name("twin_state")
    S = _twin_state()["objects"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    rng = np.random.default_rng(1)
    for k, o in enumerate(S):
        for ax, a, b in ((axes[0], o["log_am_prior"], o["log_am_post"]),
                         (axes[1], np.log10(np.minimum(o["life_prior"], 1826.25)), np.log10(np.minimum(o["life_post"], 1826.25)))):
            ax.scatter(np.asarray(a), k - 0.15 + rng.uniform(-0.08, 0.08, len(a)), s=6, color=MUTED, alpha=0.6, linewidth=0,
                       label="prior (design value or class prior)" if k == 0 else None)
            ax.scatter(np.asarray(b), k + 0.15 + rng.uniform(-0.08, 0.08, len(b)), s=8, color=ORANGE, alpha=0.8, linewidth=0,
                       label="conditioned on the catalogue history (twin)" if k == 0 else None)
    labels = [f"{o['name']} ({o['type1']})\n{o['perigee0']:.0f}×{o['apogee0']:.0f} km in {o['t0'][:4]}\n"
              f"→ {o['perigee']:.0f}×{o['apogee']:.0f} km now" for o in S]
    for ax in axes:
        ax.set_yticks(range(len(S)), labels, fontsize=6.3)
        ax.invert_yaxis()
    style(axes[0], "log10 area-to-mass ratio (m²/kg)")
    title(axes[0], "(a) area-to-mass samples")
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    day_ticks(axes[1], "x", 0.0, CAP + 0.1)
    style(axes[1], f"forecast time to re-entry after {AS_OF} (capped at 5 yr)")
    title(axes[1], "(b) resulting re-entry forecasts")
    plt.setp(axes[1].get_yticklabels(), visible=False)
    fig.tight_layout()
    save(fig, name)


FIGS = {"data_reentry": fig_data_reentry, "data_debris": fig_data_debris, "data_telemetry": fig_data_telemetry,
        "reentry_outputs": fig_reentry_outputs, "cac_outputs": fig_cac_outputs, "smap_outputs": fig_smap_outputs,
        "twin_forecast": fig_twin_forecast, "twin_state": fig_twin_state}

if __name__ == "__main__":
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
